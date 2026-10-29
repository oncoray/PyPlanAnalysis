"""
PyPlanAnalysis.plan
==================
Main user-facing API: PatientPlan (loads DICOM data and runs the
analysis) and AnalysisResults (metrics, plots, NTCP, and exports).
See the project README for usage examples.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import re
import pydicom
from typing import Union
import matplotlib.pyplot as plt
from .io       import (load_ct_series,load_dose_grid, get_grid_geometry, get_fractional_mask_on_grid,get_structure_mask_on_grid,
                        get_all_structure_names, find_dicom_files,resample_dose_on_ct,_np_to_sitk,resample_dose_to_new_grid,get_roi_center_of_mass)
from .rbe      import RBEConfig, rbe_fixed, compute_rbe_dose
from .metrics  import (RadiobiologyConfig, MetricConfig,
                        compute_dvh_metrics, compute_let_metrics,
                        compute_cumulative_histogram, compute_2d_histogram)
from .plots    import (plot_cumulative_histogram, plot_dvh_comparison,
                        plot_dlvh_2d, plot_dlvh_3d, save_figures_to_pdf, save_figure)

from .NTCP import (NTCPConfig,NTCPModelBase)
import copy
import SimpleITK as sitk
from dataclasses import dataclass
# ============================================================
#  Shared types and constants
# ============================================================
 
ArrayInput = Union[np.ndarray, sitk.Image]
 
#: Label of the physical dose in dvh_data / metrics_df. Cannot be used for a vRBE array.
PHYS_LABEL = "Phys"
 
# Plausibility thresholds used by PatientPlan.validate_inputs(). They only trigger
# warnings (never errors): they exist to catch swapped, mis-scaled or mis-ordered arrays.
_VOXEL_FLOOR_FRACTION   = 0.05          # ratios evaluated where dose > 5 % of its maximum
_VRBE_RATIO_MEDIAN      = (0.95, 1.60)  # median vRBE / physical
_VRBE_RATIO_TAILS       = (0.80, 3.00)  # 0.1 % and 99.9 % percentiles of vRBE / physical
_EQD2_RATIO_MEDIAN      = (0.40, 3.00)  # median EQD2 / dose of the same label
_MAX_RATIO_SAMPLES      = 2_000_000     # subsample very large grids for the percentile checks
 
LABELS = {"mcnamara": "MCN",
          "wedenberg": "WED",
          "carabe": "CAR",
          "linear": "LIN" 
          }
 
@dataclass
class _GridData:
    """
    Arrays and geometry used for ONE analysis pass, after the optional
    resampling. Everything in here lives on the same grid.
    """
    dose        : np.ndarray | None     # total physical dose [Gy]
    let         : np.ndarray | None     # total dose-averaged LET [keV/um]
    vrbe        : dict                  # {label: total variable-RBE dose [Gy(RBE)]}
    eqd2        : dict                  # {label: total EQD2 [Gy / Gy(RBE)]}
    origin      : tuple
    spacing     : tuple                 # x, y, z
    shape       : tuple                 # z, y, x
    z_positions : np.ndarray
 
 
def _ratio_percentiles(num: np.ndarray, den: np.ndarray, floor_fraction: float):
    """
    (0.1 %, 50 %, 99.9 %) percentiles of num/den where den > floor_fraction*max(den).
    Returns None when no voxel qualifies.
    """
    dmax = float(den.max())
    if dmax <= 0.0:
        return None
    sel = den > floor_fraction * dmax
    if not sel.any():
        return None
    ratio = num[sel] / den[sel]
    if ratio.size > _MAX_RATIO_SAMPLES:
        ratio = ratio[:: int(np.ceil(ratio.size / _MAX_RATIO_SAMPLES))]
    return tuple(np.percentile(ratio, [0.1, 50.0, 99.9]))
 
# ============================================================
#  PatientPlan
# ============================================================

class PatientPlan:
    """
    Represents a single patient's proton therapy plan.

    Parameters
    ----------
    patient_id  : str
        Identifier used in output filenames and CSV column.
    dose_file   : str or Path
        DICOM RT Dose file containing physical dose [Gy].
    let_file    : str or Path
        DICOM RT Dose file containing LETd [keV/µm] stored as dose grid.
    rtstruct    : str or Path
        DICOM RT Struct file.
    CT          : str or Path
        folder of DICOM CT files

    Class method
    ------------
    PatientPlan.from_folder(folder, patient_id=None)
        Auto-discover DICOM files from a folder.
        
    PatientPlan.from_array(dose_array, dose_reference, rtstruct, ...)
        Build a plan from arrays that were accumulated outside (several
        treatment sessions, adaptations). The total PHYSICAL dose is the
        primary dose; total variable-RBE doses and total EQD2 doses computed
        outside can be attached to it (see from_array / add_vrbe / add_eqd2).
    """

    def __init__(self,
                 patient_id     : str | None = None,
                 plan_file      : Union[str, Path, None] = None,
                 dose_file      : Union[str, Path, None] = None,
                 let_file       : Union[str, Path, None] = None,
                 rtstruct       : Union[str, Path, None] = None,
                 CT_folder_path : Union[str, Path, None] = None,
                 n_fractions    : int  | None = None):
        
        
        self.plan_file   = Path(plan_file)      if plan_file      else None
        self.dose_file   = Path(dose_file)      if dose_file      else None
        self.let_file    = Path(let_file)       if let_file       else None
        self.rtstruct    = Path(rtstruct)       if rtstruct       else None
        self.CT_folder   = Path(CT_folder_path) if CT_folder_path else None
        self.n_fractions = n_fractions
        self._is_scaling_performed_from_RBEw_dose  = False
        
        if patient_id is not None:
            self.patient_id  = patient_id
            
        elif dose_file is not None:
            ds = pydicom.dcmread(str(dose_file), stop_before_pixels=True, force=True)
            self.patient_id = ds.PatientID
            
        elif rtstruct is not None:
            ds = pydicom.dcmread(str(rtstruct), stop_before_pixels=True, force=True)
            self.patient_id = ds.PatientID
        else:
            print("can´t resolve patient name, setting it to UNKNOWN")
            self.patient_id = "UNKNOWN"
        
        self._plan_ds       = None
        self._dose_arr     = None
        self._dose_ds       = None
        self._let_arr       = None
        self._let_ds         = None
        self._rtstruct_ds    = None
        self._CT_sitk        = None
        self._CT_geom        = None
        self._loaded         = False
        self._CT_loaded      = False
        
        # Externally computed (multi-session / adaptive) results, all on the reference grid.
        self._imported_vrbe       : dict = {}     # {label: total vRBE dose}
        self._imported_eqd2       : dict = {}     # {label: total EQD2}
        self._ref_img        = None          # cached SimpleITK template of the reference grid
        
        # Sparse ROI-mask cache: flat voxel indices plus optional fractional weights.
        self._roi_mask_cache = {}
    # ----------------------------------------------------------
    @classmethod
    def from_array(cls,
                   dose_array     : ArrayInput,
                   dose_reference : Union[str, Path],
                   rtstruct       : Union[str, Path],
                   *,
                   let_array      : ArrayInput | None = None,
                   vrbe_arrays    : dict[str, ArrayInput] | None = None,
                   eqd2_arrays    : dict[str, ArrayInput] | None = None,
                   n_fractions    : int | None = None,
                   CT_folder_path : Union[str, Path, None] = None,
                   patient_id     : str | None = None) -> "PatientPlan":
        """
        Build a PatientPlan from arrays that were accumulated outside
        (several treatment sessions, adaptations, ...).
 
        The total PHYSICAL dose is always the primary dose. Variable-RBE
        doses and EQD2 doses are optional add-ons stored next to it, so ONE
        plan serves any number of RBE models: structure masks are computed
        once and every label ends up in the same metrics row / DVH dict.
 
        Conventions (the caller is responsible for them)
        ------------------------------------------------
        - Every array is a TOTAL over all sessions, in Gy (physical) or
          Gy(RBE) (vRBE), on the grid of `dose_reference`, ordered
          (frames, rows, cols) = (z, y, x), the same order as RTDOSE pixel data.
        - EQD2 arrays must be accumulated session by session outside, e.g.
          sum_i EQD2(d_i, n_i): this is why the plan does not assume a uniform
          fractionation.
        - let_array is the total dose-averaged LET:
          LETd_tot = sum_i(D_i * LETd_i) / sum_i(D_i)  (NOT a plain sum/mean).
 
        Parameters
        ----------
        dose_array     : total physical dose [Gy].
        dose_reference : any RTDOSE on the target grid. Used ONLY for grid
                         geometry and PatientID, never for dose values.
        rtstruct       : RTSTRUCT file.
        let_array      : total LETd [keV/um], optional.
        vrbe_arrays    : {label: total vRBE dose}, e.g. {"mcnamara": arr, "wedenberg": arr2}.
                         Labels become column suffixes and DVH keys and are matched by
                         NTCP models, so use the same names as RBEConfig.models.
                         "Phys" is reserved.
        eqd2_arrays    : {label: total EQD2}. Valid labels: "Phys", the fixed-RBE label
                         (e.g. "RBE1.1") and every key of vrbe_arrays.
        n_fractions    : OPTIONAL when EQD2 arrays are supplied. Needed only (a) to
                         compute internal RBE models from the summed dose, which assumes
                         a uniform fractionation, and (b) as uniform-fractionation
                         fallback for labels without an EQD2 array.
        CT_folder_path : optional CT series.
        patient_id     : defaults to the PatientID of dose_reference.
 
        Grid handling
        -------------
        - numpy arrays carry no geometry: their shape must equal the reference grid,
          otherwise a ValueError is raised.
        - SimpleITK images carry geometry: if it matches the reference grid (size,
          spacing, origin, direction) they are used as is; otherwise they are
          resampled (linear) onto the reference grid with a warning. Voxels outside
          the source field of view become 0.
        """
        ref_ds = pydicom.dcmread(str(dose_reference), stop_before_pixels=True, force=True)
 
        plan = cls(patient_id=patient_id or getattr(ref_ds, "PatientID", "UNKNOWN"),
                   rtstruct=rtstruct, CT_folder_path=CT_folder_path,
                   n_fractions=n_fractions)
 
        plan._dose_ds  = ref_ds
        plan._dose_arr = plan._conform_to_reference(dose_array, "dose_array", dtype=np.float64)
        if not plan._dose_arr.any():
            raise ValueError("dose_array is identically zero.")
 
        if let_array is not None:
            plan._let_arr = plan._conform_to_reference(let_array, "let_array", dtype=np.float64)
            plan._let_ds  = ref_ds
 
        for label, arr in (vrbe_arrays or {}).items():
            plan.add_vrbe(label, arr)
        for label, arr in (eqd2_arrays or {}).items():
            plan.add_eqd2(label, arr)
 
        if n_fractions is None and not (plan._imported_vrbe or plan._imported_eqd2):
            raise ValueError(
                "n_fractions is required when neither vrbe_arrays nor eqd2_arrays are "
                "given (EQD2 would otherwise be undefined).")
 
        plan.load()   # reads RTSTRUCT + CT; the seeded arrays are left untouched
        return plan
 
    # ----------------------------------------------------------
    #  External (pre-accumulated) data
    # ----------------------------------------------------------
    
    def add_vrbe(self, label: str, array: ArrayInput) -> None:
        """
        Attach a TOTAL variable-RBE dose computed outside (e.g. summed over
        sessions). Replaces an existing array with the same label.
        """
        if self._dose_arr is None:
            raise RuntimeError("Set the total physical dose before adding vRBE doses.")
        if label == PHYS_LABEL:
            raise ValueError(f"'{PHYS_LABEL}' is reserved for the physical dose.")
        self._imported_vrbe[label] = self._conform_to_reference(array, f"vrbe_arrays['{label}']")
    
    def add_eqd2(self, label: str, array: ArrayInput) -> None:
        """
        Attach a TOTAL EQD2 dose computed outside, accumulated session by
        session. `label` must match a dose label: "Phys", the fixed-RBE label
        (e.g. "RBE1.1") or a label given to add_vrbe().
        """
        if self._dose_arr is None:
            raise RuntimeError("Set the total physical dose before adding EQD2 doses.")
        self._imported_eqd2[label] = self._conform_to_reference(array, f"eqd2_arrays['{label}']")
 
    # ----------------------------------------------------------
    #  Grid conformity
    # ----------------------------------------------------------
 
    def _reference_image(self) -> sitk.Image:
        """SimpleITK template carrying the reference-grid geometry (built once)."""
        if self._ref_img is None:
            shape = (int(self._dose_ds.NumberOfFrames),
                     int(self._dose_ds.Rows),
                     int(self._dose_ds.Columns))
            self._ref_img = _np_to_sitk(np.zeros(shape, dtype=np.float32), self._dose_ds)
        return self._ref_img
 
    @staticmethod
    def _same_geometry(a: sitk.Image, b: sitk.Image,
                       tol_mm: float = 1e-3, tol_dir: float = 1e-4) -> bool:
        return (a.GetSize() == b.GetSize()
                and np.allclose(a.GetSpacing(),   b.GetSpacing(),   atol=tol_mm)
                and np.allclose(a.GetOrigin(),    b.GetOrigin(),    atol=tol_mm)
                and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol_dir))
 
    def _conform_to_reference(self, array: ArrayInput, name: str,
                              dtype=np.float32) -> np.ndarray:
        """
        Return `array` as an independent ndarray on the reference grid.
 
        ndarray      -> shape check only (no geometry available), error on mismatch.
        sitk.Image   -> geometry check; resampled onto the reference grid if different.
        In both cases the values must be finite, non-negative and not all zero.
        """
        ref_shape = (int(self._dose_ds.NumberOfFrames),
                     int(self._dose_ds.Rows),
                     int(self._dose_ds.Columns))
 
        if isinstance(array, sitk.Image):
            ref_img = self._reference_image()
            if self._same_geometry(array, ref_img):
                arr = sitk.GetArrayFromImage(array)
            else:
                warnings.warn(
                    f"{name}: grid differs from the reference grid "
                    f"(size {array.GetSize()} vs {ref_img.GetSize()}, spacing "
                    f"{tuple(round(v, 3) for v in array.GetSpacing())} vs "
                    f"{tuple(round(v, 3) for v in ref_img.GetSpacing())}). "
                    "Resampling (linear) onto the reference grid; voxels outside "
                    "the source field of view are set to 0.")
                res = sitk.Resample(array, ref_img, sitk.Transform(),
                                    sitk.sitkLinear, 0.0, sitk.sitkFloat64)
                arr = sitk.GetArrayFromImage(res)
        else:
            arr = np.asarray(array)
            if arr.shape != ref_shape:
                hint = ""
                if arr.shape == ref_shape[::-1]:
                    hint = (" The shape looks like (x, y, z); the expected order is "
                            "(frames, rows, cols) = (z, y, x).")
                raise ValueError(
                    f"{name}: shape {arr.shape} does not match the reference grid "
                    f"{ref_shape}.{hint} A numpy array has no geometry and cannot be "
                    "resampled: put it on the reference grid, or pass a SimpleITK image "
                    "to enable automatic resampling.")
 
        arr = np.array(arr, dtype=dtype, copy=True)    # never alias the caller's memory
        if not np.isfinite(arr).all():
            raise ValueError(f"{name}: contains NaN or inf values.")
        if arr.min() < -1e-6:
            raise ValueError(f"{name}: contains negative values (min={arr.min():.4g}).")
        if not arr.any():
            raise ValueError(f"{name}: is identically zero.")
        return arr
 
    def validate_inputs(self) -> None:
        """
        Plausibility checks of the external arrays against the physical dose, on
        the native grid. Only warns: it catches swapped, mis-scaled or mis-ordered
        inputs, it cannot prove correctness. Called automatically by analyse().
        """
        phys = self._dose_arr
        if phys is None:
            return
 
        for label, vrbe in self._imported_vrbe.items():
            stats = _ratio_percentiles(vrbe, phys, _VOXEL_FLOOR_FRACTION)
            if stats is None:
                continue
            lo, med, hi = stats
            if np.isclose(lo, hi, atol=1e-6):
                warnings.warn(f"vRBE '{label}' is a constant multiple ({med:.3f}) of the "
                              "physical dose: did you pass the 1.1-weighted dose?")
            elif (lo < _VRBE_RATIO_TAILS[0] or hi > _VRBE_RATIO_TAILS[1]
                  or not (_VRBE_RATIO_MEDIAN[0] < med < _VRBE_RATIO_MEDIAN[1])):
                warnings.warn(
                    f"vRBE '{label}' / physical ratio looks implausible "
                    f"(0.1%={lo:.2f}, median={med:.2f}, 99.9%={hi:.2f}). "
                    "Check units, scaling and array order.")
 
        for label, eqd2 in self._imported_eqd2.items():
            base = phys if label == PHYS_LABEL else self._imported_vrbe.get(label)
            if base is None:         # e.g. fixed-RBE label: its dose is built in analyse()
                continue
            stats = _ratio_percentiles(eqd2, base, _VOXEL_FLOOR_FRACTION)
            if stats is None:
                continue
            med = stats[1]
            if not (_EQD2_RATIO_MEDIAN[0] < med < _EQD2_RATIO_MEDIAN[1]):
                warnings.warn(
                    f"EQD2 '{label}' / total dose median ratio is {med:.2f}; "
                    "check that it is a total EQD2 (not per fraction) of the same label.")
    
    

    @classmethod
    def from_folder(cls,
                    folder: Union[str, Path],
                    n_fractions: int | None = None,
                    rad_type: str = "PROTON") -> "PatientPlan":
        """
        Create a PatientPlan by auto-discovering DICOM files in folder.

        Parameters
        ----------
        folder     : path containing .dcm files
        patient_id : if None, uses the folder name
        """
        folder = Path(folder)
        files  = find_dicom_files(folder, rad_type)
                
        missing = [k for k, v in files.items() if v is None]
        if missing:
            if 'rtplan' in missing:
                print(
                    f"⚠ Could not find {missing} in {folder}."
                )
            else:
                print(
                    f"⚠ Could not find {missing} in {folder}. "
                    "Set file paths manually via PatientPlan(...)."
                )
        found = [k for k, v in files.items() if v is not None]
        if found:
            pat = files["Patient_ID"]
            print(f"\nSearched for patient {pat}")
            for f in found:
                if f not in ["linked","link_warnings","Patient_ID"]:
                    print(f"✓ Ready to load {f}: {files[f]}")
        return cls(
            patient_id = files["Patient_ID"],
            plan_file  = files["rtplan"],
            dose_file  = files["dose"],
            let_file   = files["let"],
            rtstruct   = files["rtstruct"],
            CT_folder_path=  files["CT"],
            n_fractions =  n_fractions
        )

    # ----------------------------------------------------------
    

    def load(self):
        """Load and cache all DICOM data. Called automatically by analyse()."""
        if self._loaded:
            return
    
        print("\n")
        print(f"Loading DICOM files for {self.patient_id}...")
        if self.plan_file is not None:
            self._plan_ds  = pydicom.dcmread(str(self.plan_file), force=True)
            n_fractions_from_rtplan = self._plan_ds.FractionGroupSequence[0].NumberOfFractionsPlanned
            print(f"Selected plan: {self._plan_ds.RTPlanName}")
            try: 
                radtype= self._plan_ds.RadiationType
                print(f"Radiation Type: {radtype}")
            except:
                print("Radiation Type: unknown")
                
            if self.n_fractions is not None:
                if n_fractions_from_rtplan != self.n_fractions:
                    print(f'⚠ Number of fractions specified does not match RTPLAN, overwriting with RTPLAN value: {n_fractions_from_rtplan} fractions')
                    self.n_fractions = n_fractions_from_rtplan
            else:
                self.n_fractions = n_fractions_from_rtplan
                
        if (self.plan_file is None and self.n_fractions is None
                and not (self._imported_vrbe or self._imported_eqd2)):
            raise AttributeError(
                "No fraction number is given, please define it or select RTPLAN"
            )
                
        if self.dose_file is not None:
            #if available dose is effective,  scale down by 10%
            
            self._dose_arr, self._dose_ds = load_dose_grid(self.dose_file)
            dose_type = getattr(self._dose_ds, "DoseType", "").upper()
            
            if dose_type == "EFFECTIVE":
                self._is_scaling_performed_from_RBEw_dose  = True
                self._dose_arr = self._dose_arr/1.1  
            
        if self.let_file is not None:
            self._let_arr, self._let_ds = load_dose_grid(self.let_file)
    
        if self.rtstruct is not None:
            self._rtstruct_ds = pydicom.dcmread(str(self.rtstruct), force=True)
    
        if self.CT_folder is not None:
            try:
                self._CT_sitk, self._CT_geom = load_ct_series(str(self.CT_folder))
                self._CT_loaded = True
            except Exception as e:
                warnings.warn(f"Could not load CT series: {e}")
                self._CT_loaded = False
        else:
            self._CT_loaded = False
    
        self._loaded = True
        
    

    @property
    def structure_names(self) -> list:
        """List of all structure names in the RT Struct."""
        if not self._loaded:
            self.load()
        return get_all_structure_names(self._rtstruct_ds)

    
    #Not used
    @property
    def voxel_volume_cc(self) -> float:
        """Volume of one dose voxel in cc."""
        if not self._loaded:
            self.load()
        if self._CT_loaded:
            spacing = self._CT_geom['spacing']
        elif self._dose_ds is not None:
            _, spacing = get_grid_geometry(self._dose_ds)
        elif self._let_ds is not None:
            _, spacing = get_grid_geometry(self._let_ds)
        else:
            raise ValueError(
                "No grid geometry available — need at least one of "
                "dose, LET, or CT to determine voxel volume."
            )
        return float(np.prod(spacing)) / 1000.0
    
   

    # ----------------------------------------------------------
    
    
    
    def _sanity_check_and_resample_dose(self,
                                structures              : list = None,
                                metric_cfg              : MetricConfig   = None,
                                resample_on_CT          : bool = False,
                                resample_on_custom_grid : bool = False) -> "_GridData | None":
        """
        Validate the loaded data and bring dose, LET and every external vRBE/EQD2
        array onto the analysis grid (reference grid, CT grid or custom grid).
        Works on local copies: self's arrays are never modified, so repeated
        calls with any flag combination start from the same pristine data.
        Returns None (after a warning) when no analysis is possible.
        """
        self.load()
        # ---------------------------------------------------------
        # Validate loaded data
        # ---------------------------------------------------------
        if self._dose_ds is not None and self._dose_arr is not None:
            expected = (int(self._dose_ds.NumberOfFrames),
                        int(self._dose_ds.Rows),
                        int(self._dose_ds.Columns))
            if tuple(self._dose_arr.shape) != expected:
                raise ValueError(f"Dose array shape {tuple(self._dose_arr.shape)} does not "
                                 f"match its DICOM header {expected}.")
 
        if self._dose_arr is None and self._let_arr is None:
            warnings.warn("Neither dose nor LET could be loaded — nothing to analyse.")
            return None
        if self._rtstruct_ds is None:
            warnings.warn("No RT Struct loaded — cannot extract any structures.")
            return None
 
        if resample_on_CT and not self._CT_loaded:
            raise ValueError("No CT has been loaded for resampling")
        if resample_on_custom_grid and resample_on_CT:
            raise ValueError("Two resampling criteria have been selected. Please select only one.")
 
        # ---------------------------------------------------------
        # Local copies: never modify self's pristine data
        # ---------------------------------------------------------
        dose_arr = self._dose_arr.copy() if self._dose_arr is not None else None
        dose_ds  = copy.deepcopy(self._dose_ds) if self._dose_ds is not None else None
        let_arr  = self._let_arr.copy() if self._let_arr is not None else None
        let_ds   = copy.deepcopy(self._let_ds) if self._let_ds is not None else None
        vrbe     = dict(self._imported_vrbe)    # arrays are only read below, never modified
        eqd2     = dict(self._imported_eqd2)
 
        sitk_dose = _np_to_sitk(dose_arr, dose_ds) if dose_arr is not None else None
        sitk_let  = _np_to_sitk(let_arr,  let_ds)  if let_arr  is not None else None
 
        grid_origin = grid_spacing = grid_shape = grid_z_positions = None
 
        def _resample_extra(arr, resampler):
            """Resample an external array exactly like the physical dose."""
            return resampler(_np_to_sitk(np.asarray(arr, dtype=np.float64),
                                         copy.deepcopy(self._dose_ds)))
 
        if self._CT_loaded and resample_on_CT:
            print("  Resampling dose, LET, vRBE and EQD2 onto CT grid...")
 
            def _on_ct(img):
                return sitk.GetArrayFromImage(resample_dose_on_ct(img, self._CT_sitk))
 
            if sitk_dose is not None:
                dose_arr = _on_ct(sitk_dose)
            if sitk_let is not None:
                let_arr = _on_ct(sitk_let)
            vrbe = {k: _resample_extra(v, _on_ct) for k, v in vrbe.items()}
            eqd2 = {k: _resample_extra(v, _on_ct) for k, v in eqd2.items()}
 
            grid_origin      = self._CT_geom['origin']
            grid_spacing     = self._CT_geom['spacing']      # x,y,z
            grid_shape       = self._CT_geom['shape']
            grid_z_positions = self._CT_geom['z_positions']
 
        elif resample_on_custom_grid:
            grid_new = metric_cfg.New_grid
            print(f"  Resampling dose, LET, vRBE and EQD2 onto custom {grid_new} grid...")
 
            def _on_grid(img):
                _, arr_new, _, _ = resample_dose_to_new_grid(img, copy.deepcopy(self._dose_ds),
                                                             grid_new)
                return arr_new
 
            if sitk_dose is not None:
                _, dose_arr, dose_geom, _ = resample_dose_to_new_grid(sitk_dose, dose_ds, grid_new)
            if sitk_let is not None:
                _, let_arr, let_geom, let_ds = resample_dose_to_new_grid(sitk_let, let_ds, grid_new)
                if sitk_dose is None:
                    dose_geom = let_geom
            vrbe = {k: _resample_extra(v, _on_grid) for k, v in vrbe.items()}
            eqd2 = {k: _resample_extra(v, _on_grid) for k, v in eqd2.items()}
 
            grid_origin      = dose_geom['origin']
            grid_spacing     = dose_geom['spacing']          # x,y,z
            grid_shape       = dose_geom['shape']            # z,y,x
            grid_z_positions = dose_geom['z_positions']
 
        else:
            print("No Resampling for dose and LET")
 
            # no CT, no resample requested: use whichever grid (dose preferred, else LET) is available
            ref_ds = dose_ds if dose_ds is not None else let_ds
            if ref_ds is not None:
                grid_origin, grid_spacing = get_grid_geometry(ref_ds)
                grid_shape       = (dose_arr.shape if dose_arr is not None
                                    else let_arr.shape)
                z_offsets        = [float(v) for v in ref_ds.GridFrameOffsetVector]
                grid_z_positions = np.array([grid_origin[2] + o for o in z_offsets])
 
        if grid_origin is None:
            warnings.warn("Could not determine a voxel grid — aborting analysis.")
            return None
 
        return _GridData(dose=dose_arr, let=let_arr, vrbe=vrbe, eqd2=eqd2,
                         origin=grid_origin, spacing=grid_spacing,
                         shape=grid_shape, z_positions=grid_z_positions)
    
    def _roi_mask_cache_key(self, struct_name, rtstruct_ds, grid_origin, grid_spacing,
                            grid_shape, grid_z_positions, use_fractional,
                            supersample, max_supersample):
        return (
            id(rtstruct_ds), struct_name.strip().lower(),
            tuple(float(v) for v in grid_origin),
            tuple(float(v) for v in grid_spacing), tuple(int(v) for v in grid_shape),
            hash(np.asarray(grid_z_positions, dtype=np.float64).tobytes()),
            bool(use_fractional), int(supersample), int(max_supersample),
        )

    def _get_roi_mask_indices(self, struct_name, rtstruct_ds, grid_origin, grid_spacing,
                              grid_shape, grid_z_positions, use_fractional=False,
                              supersample=4, max_supersample=12):
        """Return cached sparse flat indices and optional fractional weights."""
        key = self._roi_mask_cache_key(
            struct_name, rtstruct_ds, grid_origin, grid_spacing, grid_shape,
            grid_z_positions, use_fractional, supersample, max_supersample)
        cached = self._roi_mask_cache.get(key)
        if cached is None:
            if use_fractional:
                frac = get_fractional_mask_on_grid(
                    struct_name, rtstruct_ds, grid_origin, grid_spacing,
                    grid_shape, grid_z_positions, supersample=supersample,
                    max_supersample=max_supersample)
                indices = np.flatnonzero(frac)
                weights = np.asarray(frac.ravel()[indices], dtype=np.float32)
            else:
                mask = get_structure_mask_on_grid(
                    struct_name, rtstruct_ds, grid_origin, grid_spacing,
                    grid_shape, grid_z_positions)
                indices = np.flatnonzero(mask)
                weights = None
            indices = indices.astype(np.intp, copy=False)
            self._roi_mask_cache[key] = (indices, weights)
        else:
            indices, weights = cached
        return indices, weights
    
    def get_roi_mask(self, struct_name, rtstruct_ds, grid_origin, grid_spacing,
                     grid_shape, grid_z_positions, use_fractional=False,
                     supersample=4, max_supersample=12):
        indices, weights = self._get_roi_mask_indices(
            struct_name, rtstruct_ds, grid_origin, grid_spacing, grid_shape,
            grid_z_positions, use_fractional, supersample, max_supersample)
        mask = np.zeros(int(np.prod(grid_shape)), dtype=bool)
        mask[indices] = True
        mask = mask.reshape(grid_shape)
        return (weights, mask) if use_fractional else (None, mask)
    
    
    def _get_priority(self, struct_name: str, priority_map: dict) -> int:
        """Substring match on structure name, same as alpha_beta lookup."""
        name = struct_name.lower()
        for key, val in priority_map.items():
            if key.lower() in name:
                return val 
        return 999 # lowest priority when not listed
        
    def build_alpha_beta_map(self, rtstruct_ds, structures, grid_origin, grid_spacing,
                         grid_shape, grid_z_positions, radiobio_cfg,
                         use_fractional, supersample, max_supersample):
        ab_map = np.full(grid_shape, radiobio_cfg.alpha_beta_default, dtype=np.float32)
        priority_map = np.full(grid_shape, 1000, dtype=np.int32)

        # Preserve the original overlap semantics: EVERY structure claims its
        # priority region, including structures whose alpha/beta equals the
        # default. This is important because a higher-priority default-alpha/beta
        # ROI must prevent a lower-priority non-default ROI from overwriting it.
        # Unconfigured structures still participate with priority 999.
        
        selected = []
        for struct_name in structures:
            priority = self._get_priority(struct_name, radiobio_cfg.structure_priority)
            try:
                ab = radiobio_cfg.get_alpha_beta(struct_name)
            except Exception:
                continue
            selected.append((struct_name, priority, ab))

        # Keep rasterisation serial (no paralle lprocessing). Sparse indices avoid dense
        # mask operations and reuse cached rasterisation across calls.
        flat_ab = ab_map.ravel()
        flat_priority = priority_map.ravel()
        for struct_name, priority, ab in selected:
            try:
                indices, _ = self._get_roi_mask_indices(
                    struct_name, rtstruct_ds, grid_origin, grid_spacing,
                    grid_shape, grid_z_positions, use_fractional,
                    supersample, max_supersample)
            except Exception:
                continue
            if indices.size == 0:
                continue
            update = indices[flat_priority[indices] > priority]
            if update.size:
                flat_ab[update] = ab
                flat_priority[update] = priority

        return ab_map, priority_map
    
    def get_alpha_beta_map(self, radiobio_cfg=None, structures=None,
                           use_fractional=False, supersample=4, max_supersample=12):
        radiobio_cfg = radiobio_cfg or RadiobiologyConfig()
        self.load()
        available = self.structure_names
        structures = available if structures is None else [s for s in structures if s in available]
        
        grid = self._sanity_check_and_resample_dose(structures, MetricConfig(), False, False)
        if grid is None:
            raise ValueError("Could not determine the voxel grid.")
        
        ab_map, _ = self.build_alpha_beta_map(
            self._rtstruct_ds, structures,
            grid.origin, grid.spacing, grid.shape, grid.z_positions,
            radiobio_cfg, use_fractional, supersample, max_supersample)
       
        return ab_map.astype(np.float64)

    def compute_vRBE(self,
                    model : str,
                    structures              : list = None,
                    metric_cfg    : MetricConfig         = None,
                    radiobio_cfg  : RadiobiologyConfig   = None,
                    resample_on_CT          : bool = False,
                    resample_on_custom_grid : bool = False,
                    use_fractional: bool                 = False,
                    supersample   : int                  = 4,
                    max_supersample: int                 = 12):
        """
        Compute a voxel-wise variable-RBE dose volume for the whole patient grid.
        It Works independently of analyse() — no DVH metrics are computed, only the 3D RBE-weighted dose map and the α/β map used to produce it.
        It works independetly of RBEConfig
       
        Parameters
        ----------
        model           : str — RBE model to apply: "linear", "mcnamara", "wedenberg", or "carabe"
        structures      : list of structure name strings used to build the α/β map, or None (= all).
                          Voxels not inside any listed structure receive the default α/β.
        metric_cfg      : MetricConfig — only New_grid is used, when resample_on_custom_grid=True
        radiobio_cfg    : RadiobiologyConfig — supplies per-structure α/β values and overlap
                          priority for building the voxel-wise α/β map
        resample_on_CT  : if True, resample dose and LET onto the CT grid before computing RBE.
                          Requires CT to have been loaded (ct_dir passed to PatientPlan)
        resample_on_custom_grid : if True, resample dose and LET onto the isotropic grid defined
                          by metric_cfg.New_grid [mm]. Mutually exclusive with resample_on_CT.
        use_fractional  : if True, use fractional voxel membership for the α/β map boundary voxels.
                          More accurate for small structures. Default: False.
        supersample     : baseline in-plane supersampling factor (N×N sub-points per voxel).
                          Only used when use_fractional=True. Scaled up automatically for small
                          structures — see io._adaptive_supersample. Default: 4.
        max_supersample : hard cap on the adaptive in-plane supersampling factor, to bound
                          compute cost for very small ROIs. Default: 12.
       
        Returns
        -------
        vRBEResults
        A container holding the 3D vRBE dose array, the α/β map, grid geometry,
        and a to_nii() method to save both volumes as NIfTI files.
        """
        if self._dose_arr is None:
            raise ValueError("compute_vRBE needs a physical dose.")
        if self.n_fractions is None:
            raise ValueError("compute_vRBE needs n_fractions: the RBE models depend on the "
                             "dose per fraction and assume a uniform fractionation.")
        if self._imported_vrbe:
            warnings.warn("This plan carries externally computed vRBE doses (multi-session). "
                          "compute_vRBE applies the model to the SUMMED physical dose assuming "
                          "uniform fractionation, which is not valid for heterogeneous sessions.")
        if model.lower() != "linear" and self._let_arr is None:
            raise ValueError(f"RBE model '{model}' needs a LET map.")
 
        metric_cfg   = metric_cfg   or MetricConfig()
        radiobio_cfg = radiobio_cfg or RadiobiologyConfig()
 
        if structures is None:
            structures = self.structure_names
        else:
            available = self.structure_names
            missing   = [s for s in structures if s not in available]
            if missing:
                warnings.warn(f"Structures not in RT Struct: {missing}")
            structures = [s for s in structures if s in available]
 
        mode = f"fractional (supersample={supersample})" if use_fractional else "binary"
        print(f"Analysing {len(structures)} structures [{mode} mask]...")
 
        #--------------------
        # Perform resampling and sanity checks
        #--------------------
        grid = self._sanity_check_and_resample_dose(structures, metric_cfg,
                                                    resample_on_CT, resample_on_custom_grid)
        if grid is None:
            raise RuntimeError("compute_vRBE aborted: no usable dose grid (see warnings).")
 
        ab_map,_ = self.build_alpha_beta_map(self._rtstruct_ds, structures, grid.origin, grid.spacing,
                                             grid.shape, grid.z_positions, radiobio_cfg,
                                             use_fractional, supersample, max_supersample)
 
        vRBE = compute_rbe_dose(grid.dose, grid.let, self.n_fractions, ab_map, model)
 
        return vRBEResults(patient_id  = self.patient_id,
                           model = model.lower(),
                           dose_arr = grid.dose,
                           let_arr = grid.let,
                           vRBE = vRBE,
                           ab_map = ab_map,
                           geom = {"origin" : grid.origin,"spacing" : grid.spacing} )
    
    
    def analyse(self,
                structures    : list = None,
                rbe_cfg       : RBEConfig            = None,
                metric_cfg    : MetricConfig         = None,
                radiobio_cfg  : RadiobiologyConfig   = None,
                resample_on_CT      : bool                 = False,
                resample_on_custom_grid      : bool                 = False,
                use_fractional: bool                 = False,
                supersample   : int                  = 4,
                max_supersample: int                 = 12,
                compute_internal_models : bool | None = None) -> "AnalysisResults":
        """
        Run the full analysis pipeline for this patient.

        Parameters
        ----------
        structures    : list of structure name strings, or None (= all)
        rbe_cfg       : RBEConfig  (default: all three models, RBE=1.1)
        metric_cfg    : MetricConfig  (default values)
        radiobio_cfg  : RadiobiologyConfig  (default per-tissue α/β and gEUD-a)
        resample: if True, interpolate dose values to the new spacing defined in MetricConfig
        use_fractional: if True, use fractional voxel membership mask instead
                        of binary mask. More accurate for small structures.
                        Each voxel is weighted by the fraction of its volume
                        inside the contour. Default: False.
        supersample   : baseline in-plane supersampling factor (N x N sub-points
                        per voxel). Only used when use_fractional=True.
                        Scaled up automatically for small structures — see
                        io._adaptive_supersample. Default: 4.
        max_supersample: hard cap on the adaptive in-plane supersampling
                        factor, to bound compute cost for very small ROIs.

        compute_internal_models : whether to apply rbe_cfg.models to the physical dose.
                        These models assume a uniform fractionation (n_fractions), so the
                        default is True only if NO imported vRBE arrays are attached and
                        n_fractions is known. Imported arrays (vrbe_arrays) always win over
                        an internal model with the same label.
 
        Labels in metrics_df / dvh_data
        -------------------------------
        "Phys", "RBE<fixed_rbe>", internal models, then imported vRBE labels.
        EQD2-based metrics use the imported EQD2 array of the label when present;
        otherwise they fall back to the uniform-fractionation conversion with
        n_fractions (warning), or are left undefined when n_fractions is None.
        DLVH and LET metrics are computed on the physical / fixed-RBE dose, so they do
        not depend on the RBE model.
 
        Returns
        -------
        AnalysisResults
        """
        rbe_cfg      = rbe_cfg      or RBEConfig()
        metric_cfg   = metric_cfg   or MetricConfig()
        radiobio_cfg = radiobio_cfg or RadiobiologyConfig()

        if structures is None:
            structures = self.structure_names
        else:
            available = self.structure_names
            missing   = [s for s in structures if s not in available]
            if missing:
                warnings.warn(f"Structures not in RT Struct: {missing}")
            structures = [s for s in structures if s in available]

        mode = f"fractional (supersample={supersample})" if use_fractional else "binary"
        print(f"Analysing {len(structures)} structures [{mode} mask]...")
        
        #--------------------
        # Perform resampling and sanity checks
        #-----------------------------
        self.validate_inputs()
        grid = self._sanity_check_and_resample_dose(structures, metric_cfg,
                                                    resample_on_CT, resample_on_custom_grid)
        if grid is None:
            raise RuntimeError("Analysis aborted: no usable dose grid (see warnings).")
 
        dose_arr, let_arr = grid.dose, grid.let
        grid_origin, grid_spacing = grid.origin, grid.spacing
        grid_shape, grid_z_positions = grid.shape, grid.z_positions  
        
        # ---- label bookkeeping and consistency checks (once, before the structure loop) ----
        fixed_lbl = f"RBE{rbe_cfg.fixed_rbe}"
        if fixed_lbl in grid.vrbe:
            raise ValueError(f"vRBE label '{fixed_lbl}' collides with the fixed-RBE label.")
 
        if compute_internal_models is None:
            compute_internal_models = (not grid.vrbe) and self.n_fractions is not None
        if compute_internal_models and self.n_fractions is None:
            raise ValueError("Internal RBE models need n_fractions (uniform fractionation).")
 
        ext_lower      = {k.lower() for k in grid.vrbe}
        internal_models = [m for m in rbe_cfg.models if m.lower() not in ext_lower]
        shadowed        = [m for m in rbe_cfg.models if m.lower() in ext_lower]
        if shadowed:
            print(f"  Imported vRBE overrides internal model(s): {shadowed}")
        if grid.vrbe and compute_internal_models and internal_models:
            warnings.warn(f"Internal models {internal_models} are applied to the summed physical "
                          "dose assuming uniform fractionation; import vrbe_arrays for multi-session or adapted plans.")
 
        valid_eqd2 = {PHYS_LABEL, fixed_lbl, *grid.vrbe}
        unused     = sorted(set(grid.eqd2) - valid_eqd2)
        if unused:
            warnings.warn(f"EQD2 arrays ignored (no matching dose label): {unused}. "
                          f"Valid labels: {sorted(valid_eqd2)}.")
        grid.eqd2 = {k: v for k, v in grid.eqd2.items() if k in valid_eqd2}
 
        labels_needing_eqd2 = [PHYS_LABEL, fixed_lbl, *grid.vrbe]
        no_eqd2 = [l for l in labels_needing_eqd2 if l not in grid.eqd2]
        if no_eqd2 and (grid.vrbe or self._imported_eqd2):
            if self.n_fractions is None:
                print(f"No EQD2 array and no n_fractions for {no_eqd2}: "
                              "EQD2-based metrics will be undefined for these labels.")
            else:
                print(f"No EQD2 array for {no_eqd2}: falling back to EQD2 from "
                              f"{self.n_fractions} uniform fractions for these.")
                
        all_rows    = []
        dvh_data    = {}              # {struct: {dose_type: (edges, cum)}}
        lvh_data    = {}              # {struct: (edges, cum)}
        dlvh_data_diff   = {}         # {struct: (H, d_edges, l_edges)}
        dlvh_data_cum   = {}          # {struct: (H, d_edges, l_edges)}

        # Voxel volume MUST come from the grid actually used to build the
        # mask (grid_spacing, set above per the branch actually taken), not
        # from self.voxel_volume_cc.
        
        vol_cc = float(np.prod(grid_spacing)) / 1000.0
        
        def _metrics_and_dvh(label, dose_v, eqd2_v, weights, ab, geud_a, store_dvh=True):
            """DVH metrics (+ cumulative histogram) of one label on one structure."""
            kwargs = dict(weights=weights, n_fractions=self.n_fractions)
            if eqd2_v is not None:
                kwargs["eqd2_dose"] = eqd2_v     # total EQD2 imported
            row.update(compute_dvh_metrics(dose_v, vol_cc, label, metric_cfg, ab, geud_a, **kwargs))
            if store_dvh:
                dvh_data[struct_name][label] = compute_cumulative_histogram(
                    dose_v, metric_cfg.dvh_bins, weights=weights)
                
        for struct_name in structures:
            print(f"    → {struct_name}")

            try:
                weights,mask = self.get_roi_mask(struct_name,
                                self._rtstruct_ds,
                                grid_origin,
                                grid_spacing,
                                grid_shape,
                                grid_z_positions,
                                use_fractional,
                                supersample = supersample,
                                max_supersample = max_supersample)
                
            except Exception as e:
                warnings.warn(f"Skipping '{struct_name}': {e}")
                continue
            
            if mask.sum() == 0:
                warnings.warn(f"Empty mask for '{struct_name}' – skipping.")
                continue
            ab     = radiobio_cfg.get_alpha_beta(struct_name)
            geud_a = radiobio_cfg.get_geud_a(struct_name)
            
            dose_vox = dose_arr[mask] if dose_arr is not None else None
            let_vox  = let_arr[mask]  if let_arr  is not None else None 
            vrbe_vox = {k: v[mask].astype(np.float64, copy=False) for k, v in grid.vrbe.items()}
            eqd2_vox = {k: v[mask].astype(np.float64, copy=False) for k, v in grid.eqd2.items()}
            
            # effective volume: sum of fractional weights × voxel volume
            if weights is not None:
                eff_vol_cc = float(weights.sum() * vol_cc)
            else:
                eff_vol_cc = float(mask.sum() * vol_cc)

            
            try:
                com = get_roi_center_of_mass(struct_name, self._rtstruct_ds)
            except Exception as e:
                warnings.warn(f"Could not compute center of mass for '{struct_name}': {e}")
                com = None

            row = {
                "patient_id"  : self.patient_id,
                "ROI_Name"   : struct_name,
                "volume_cc"   : eff_vol_cc,
                "alpha_beta"  : ab,
                "geud_a"      : geud_a,
                "COM_x"       : com[0] if com is not None else np.nan,
                "COM_y"       : com[1] if com is not None else np.nan,
                "COM_z"       : com[2] if com is not None else np.nan,
            }

            dvh_data[struct_name] = {}
           
            # ---- Physical dose ----
            if dose_arr is not None:
                # ---- Physical dose ----
                _metrics_and_dvh(PHYS_LABEL, dose_vox, eqd2_vox.get(PHYS_LABEL), weights, ab, geud_a)
                
                if rbe_cfg.fixed_rbe:
                    # ---- Fixed RBE ----
                    dose_fixed = rbe_fixed(dose_vox, rbe_cfg.fixed_rbe)
                    _metrics_and_dvh(fixed_lbl, dose_fixed, eqd2_vox.get(fixed_lbl), weights, ab, geud_a)
                    dose_ref      = dose_fixed
                    highlet_label = "highLET_fixed_rbe"
 
                # ---- Variable RBE, internal models (uniform fractionation) ----
                if let_arr is not None and compute_internal_models:
                    for model in internal_models:
                        rbe_dose_vox = compute_rbe_dose(dose_vox, let_vox, self.n_fractions, ab, model)
                        _metrics_and_dvh(LABELS[model], rbe_dose_vox, None, weights, ab, geud_a)
 
                # ---- Variable RBE, external (multi-session) arrays ----
                for model, v in vrbe_vox.items():
                    _metrics_and_dvh(LABELS[model], v, eqd2_vox.get(model), weights, ab, geud_a)
 
                if let_arr is not None:
                    # ---- LET-thresholded DVH metrics (high-LET subvolume) ----
                    high_let_idx = let_vox >= metric_cfg.LET_thr
 
                    if high_let_idx.sum() > 0:
                        w_high = weights[high_let_idx] if weights is not None else None
                        eq_ref = eqd2_vox.get(fixed_lbl)
                        _metrics_and_dvh(highlet_label, dose_ref[high_let_idx],
                                         eq_ref[high_let_idx] if eq_ref is not None else None,
                                         w_high, ab, geud_a, store_dvh=False)
 
                    # ---- 2-D DLVH (model-independent: fixed-RBE dose vs LET) ----
                    H_diff, H_cum, d_edges, l_edges_2d = compute_2d_histogram(
                        dose_ref, let_vox,
                        metric_cfg.dlvh_dose_bins,
                        metric_cfg.dlvh_let_bins,
                        weights=weights)
                    dlvh_data_diff[struct_name] = (H_diff, d_edges, l_edges_2d)
                    dlvh_data_cum[struct_name]  = (H_cum, d_edges, l_edges_2d)
 
            if let_arr is not None:
                # ---- LET metrics & LVH ----
                row.update(compute_let_metrics(let_vox, dose_vox, metric_cfg.lx,
                                               weights=weights))
                l_edges, l_cum = compute_cumulative_histogram(let_vox,
                                                               metric_cfg.lvh_bins,
                                                               weights=weights)
                lvh_data[struct_name] = (l_edges, l_cum)
 
            all_rows.append(row)
 
 
        metrics_df = pd.DataFrame(all_rows)
 
        return AnalysisResults(
            patient_id  = self.patient_id,
            metrics_df  = metrics_df,
            dvh_data    = dvh_data,
            lvh_data    = lvh_data,
            dlvh_data_diff   = dlvh_data_diff,
            dlvh_data_cum   = dlvh_data_cum,
        )


# ============================================================
#  AnalysisResults
# ============================================================

class AnalysisResults:
    """
    Container for all outputs from PatientPlan.analyse().

    Attributes
    ----------
    patient_id  : str
    metrics_df  : pd.DataFrame  – all DVH/LVH metrics, one row per structure
    dvh_data    : dict  {struct: {dose_type: (edges, cum)}} e.g. # {"CTV": {"Physical": (edges, cum), "RBE×1.1": (edges, cum), ...}, "Brainstem": {"Physical": (edges, cum), ...},
    lvh_data    : dict  {struct: (edges, cum)}
    dlvh_data   : dict  {struct: (H, dose_edges, let_edges)}
    """

    def __init__(self,
                 patient_id : str,
                 metrics_df : pd.DataFrame,
                 dvh_data   : dict,
                 lvh_data   : dict,
                 dlvh_data_diff  : dict,
                 dlvh_data_cum  : dict):

        self.patient_id = patient_id
        self.metrics_df = metrics_df
        self.dvh_data   = dvh_data
        self.lvh_data   = lvh_data
        self.dlvh_data_diff  = dlvh_data_diff
        self.dlvh_data_cum  = dlvh_data_cum
    
    # ----------------------------------
    # NTCP calculation
    # ----------------------------------

                
    def CalcNTCP (self, NTCPConfig:  NTCPConfig, path_models_xls: str = None):
        # load list with model names and corresponding parameter names
        list_RBE_calculated = list(self.dvh_data[list(self.dvh_data.keys())[0]].keys())
        NTCP_results =  {}
        for NTCP_model in NTCPConfig.models:
            roi_override = NTCPConfig.roi_overrides.get(NTCP_model) if NTCPConfig.roi_overrides else None
            NTCP_base  = NTCPModelBase(NTCP_model, path_models_xls,
                                        roi_name = roi_override,
                                        ctv_name = NTCPConfig.ctv_name)
            NTCP_dict = {}
            NTCP_dict["ROI"] = ""
            for RBE in list_RBE_calculated:
                NTCP_RBE, roi_used  = NTCP_base.compute_NTCP(RBE, self.metrics_df)
                NTCP_dict[RBE] = NTCP_RBE
            
            NTCP_dict["ROI"] = roi_used
            
            NTCP_results[NTCP_model] = NTCP_dict
        return NTCP_results       
 
    
    # ----------------------------------------------------------
    # Export methods
    # ----------------------------------------------------------
    
        
    def to_csv(self, path: Union[str, Path] = None) -> Path:
        """
        Save metrics to CSV.

        Parameters
        ----------
        path : output path. Default: "<patient_id>/dvh_metrics.csv"
        """
        path = Path(path or f"{self.patient_id}/dvh_metrics.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_df.to_csv(path, index=False)
        print(f"  Saved CSV: {path}")
        return path

    def to_excel(self, path: Union[str, Path] = None) -> Path:
        """
        Save metrics to Excel (.xlsx).

        Parameters
        ----------
        path : output path. Default: "<patient_id>/dvh_metrics.xlsx"
        """
        path = Path(path or f"{self.patient_id}/dvh_metrics.xlsx")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics_df.to_excel(path, index=False, sheet_name=re.sub(r'[/*?\[\]<>|:]', '',self.patient_id))
        print(f"  Saved Excel: {path}")
        return path

    # ----------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------

    def plot_dvh(self,
                 save    : Union[str, Path] = None,
                 dpi     : int = 350) -> list:
        """
        Generate DVH curve figures.

        Creates one figure per dose type (Physical, RBE×1.1, each model)
        showing all structures, PLUS one per-structure comparison figure.

        Parameters
        ----------
        save : path for output PDF. Default: "<patient_id>/dvh_curves.pdf"
        dpi  : figure DPI

        Returns
        -------
        list of matplotlib Figures
        """
        save = Path(save or f"{self.patient_id}/dvh_curves.pdf")
        save.parent.mkdir(parents=True, exist_ok=True)

        figs = []

        # One figure per dose label (all structures overlaid)
        if self.dvh_data:
            all_labels = list(next(iter(self.dvh_data.values())).keys()) 
            for dose_type in all_labels:
                curves = {struct: data[dose_type]
                          for struct, data in self.dvh_data.items()
                          if dose_type in data}
                fig = plot_cumulative_histogram(
                    curves,
                    title  = f"{self.patient_id} – DVH ({dose_type})",
                    xlabel = "Dose [Gy / Gy(RBE)]",
                    dpi    = dpi)
                figs.append(fig)

            # Per-structure comparison (all dose types on one plot)
            for struct, label_curves in self.dvh_data.items():
                fig = plot_dvh_comparison(label_curves, struct, dpi=dpi)
                figs.append(fig)

        save_figures_to_pdf(figs, save, dpi=dpi)
        print(f"  Saved DVH PDF: {save}")
        return figs

    def plot_lvh(self,
                 save : Union[str, Path] = None,
                 dpi  : int = 150) -> plt.Figure:
        """
        Generate LVH curve figure (all structures on one plot).

        Parameters
        ----------
        save : path for output PDF. Default: "<patient_id>/lvh_curves.pdf"

        Returns
        -------
        matplotlib Figure
        """
        save = Path(save or f"{self.patient_id}/lvh_curves.pdf")
        save.parent.mkdir(parents=True, exist_ok=True)

        fig = plot_cumulative_histogram(
            self.lvh_data,
            title  = f"{self.patient_id} – LVH",
            xlabel = "LETd [keV/μm]",
            ylabel = "Volume fraction",
            dpi    = dpi)

        save_figures_to_pdf([fig], save, dpi=dpi)
        print(f"  Saved LVH PDF: {save}")
        return fig

    def plot_dlvh(self,
                  save : Union[str, Path] = None,
                  dpi  : int = 150) -> list:
        """
        Generate 2-D DLVH figures – one PNG per structure.

        Parameters
        ----------
        save : output directory. Default: "<patient_id>/dlvh_2d/"

        Returns
        -------
        list of (structure_name, Path) tuples
        """
        save_dir = Path(save or f"{self.patient_id}/dlvh_2d")
        save_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for struct, (H, d_edges, l_edges) in self.dlvh_data_diff.items():
            fig  = plot_dlvh_2d(H, d_edges, l_edges, struct, dpi=dpi)
            name = struct.replace(" ", "_").replace("/", "-")
            out  = save_dir / f"{name}_Diff.png"
            save_figure(fig, out, dpi=dpi)
            saved.append((struct, out))
        saved = []
        
        for struct, (H, d_edges, l_edges) in self.dlvh_data_cum.items():
            fig  = plot_dlvh_2d(H, d_edges, l_edges, struct, dpi=dpi)
            name = struct.replace(" ", "_").replace("/", "-")
            out  = save_dir / f"{name}_Cumulative.png"
            save_figure(fig, out, dpi=dpi)
            saved.append((struct, out))
            
        for struct, (H, d_edges, l_edges) in self.dlvh_data_cum.items():
            fig  = plot_dlvh_3d(H, d_edges, l_edges, struct, dpi=dpi)
            name = struct.replace(" ", "_").replace("/", "-")
            out  = save_dir / f"{name}_Cumulative3D.png"
            save_figure(fig, out, dpi=dpi)
            saved.append((struct, out))

        #print(f"  Saved {len(saved)} DLVH plots → {save_dir}/")
        return saved

    def save_all(self,
                 output_dir : Union[str, Path] = None,
                 csv_name   : str = "dvh_metrics.csv",
                 excel_name : str = "dvh_metrics.xlsx",
                 dpi        : int = 150):
        """
        Convenience method: save CSV, Excel, DVH PDF, LVH PDF, DLVH PNGs.

        Parameters
        ----------
        output_dir : base output directory. Default: patient_id/
        csv_name   : filename for the CSV output
        excel_name : filename for the Excel output
        dpi        : figure DPI for all plots
        """
        base = Path(output_dir or self.patient_id)
        base.mkdir(parents=True, exist_ok=True)

        self.to_csv(   base / csv_name)
        self.to_excel( base / excel_name)
        self.plot_dvh( base / "dvh_curves.pdf",  dpi=dpi)
        self.plot_lvh( base / "lvh_curves.pdf",  dpi=dpi)
        self.plot_dlvh(base / "dlvh_2d",         dpi=dpi)
        

# ============================================================
#  vRBEResults
# ============================================================

class vRBEResults:
    """
    Container for all outputs from PatientPlan.compute_vRBE().
    
    Attributes
    ----------
    patient_id  : str
    model       : str
    dose_arr    : np.array(), of 1.1-weighted dose
    let_arr     : np.array(), of Let in kev/um
    vRBE        : np.array (float) of variablRBE-weighted dose
    ab_map      : np.array (int)?
    geom        : dict (órigin,spacing) in (x,y,z). Reminder:  SimpleITK uses x-y-z. Numpy uses z-y-x.
    
    """

    def __init__(self,
                 patient_id : str,
                 model       : str,
                 dose_arr    : np.ndarray, 
                 let_arr     : np.ndarray,
                 vRBE        : np.ndarray, 
                 ab_map      : np.ndarray,
                 geom        : dict ):

        self.patient_id = patient_id
        self.model      = model
        self.dose_arr   = dose_arr 
        self.let_arr    = let_arr 
        self.vRBE       = vRBE 
        self.ab_map     = ab_map if not model.lower() == "linear" else None
        self.geom       = geom
    # ----------------------------------------------------------
    # Export methods
    # ----------------------------------------------------------
    
        
    def to_nii(self,
           path      : Union[str, Path],
           name_vRBE : str = "vRBE.nii.gz",
           name_ab   : str = "ab_map.nii.gz",
           name_cRBE   : str = "cRBE.nii.gz",
           name_let   : str = "LETd.nii.gz") -> None:
        """
        Save vRBE dose and α/β map as NIfTI files.
    
        Parameters
        ----------
        path      : output directory (created if it does not exist)
        name_vrbe : filename for the vRBE dose volume
        name_ab   : filename for the α/β map volume
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
    
        def _arr_to_sitk(arr, origin, spacing):
            img = sitk.GetImageFromArray(arr.astype(np.float32))
            img.SetOrigin([float(v) for v in origin])
            img.SetSpacing([float(v) for v in spacing])
            return img
    
        origin  = self.geom["origin"]   # adjust if geom is a dataclass: self.geom.origin
        spacing = self.geom["spacing"]
    
        sitk.WriteImage(_arr_to_sitk(self.vRBE,  origin, spacing),
                        str(out_dir / name_vRBE))
        
        sitk.WriteImage(_arr_to_sitk(self.dose_arr,  origin, spacing),
                        str(out_dir / name_cRBE))
        
        sitk.WriteImage(_arr_to_sitk(self.let_arr,  origin, spacing),
                        str(out_dir / name_let))
        
        if self.ab_map is not None:
            sitk.WriteImage(_arr_to_sitk(self.ab_map, origin, spacing),
                        str(out_dir / name_ab))
    
        print(f"  ✓ Saved {name_cRBE},{name_vRBE},{name_let} and {name_ab} → {out_dir}")
        
        