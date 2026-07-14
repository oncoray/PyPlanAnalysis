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

import SimpleITK as sitk

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
            ds = pydicom.dcmread(str(dose_file), stop_before_pixels=True)
            self.patient_id = ds.PatientID
            
        elif rtstruct is not None:
            ds = pydicom.dcmread(str(rtstruct), stop_before_pixels=True)
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
        
    # ----------------------------------------------------------
    @classmethod
    def from_folder(cls,
                    folder: Union[str, Path],
                    n_fractions: int | None = None) -> "PatientPlan":
        """
        Create a PatientPlan by auto-discovering DICOM files in folder.

        Parameters
        ----------
        folder     : path containing .dcm files
        patient_id : if None, uses the folder name
        """
        folder = Path(folder)
        files  = find_dicom_files(folder)
                
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
                if f not in ["linked","link_warnings"]:
                    print(f"✓ Ready to load {f}")
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
            self._plan_ds  = pydicom.dcmread(str(self.plan_file))
            n_fractions_from_rtplan = self._plan_ds.FractionGroupSequence[0].NumberOfFractionsPlanned
            if self.n_fractions is not None:
                if n_fractions_from_rtplan != self.n_fractions:
                    print(f'⚠ Number of fractions specified does not match RTPLAN, overwriting with RTPLAN value: {n_fractions_from_rtplan} fractions')
                    self.n_fractions = n_fractions_from_rtplan
            else:
                self.n_fractions = n_fractions_from_rtplan
                
        if self.plan_file is None and self.n_fractions is None:
            raise AttributeError(
                "No fraction number is given, please define it or select RTPLAN"
            )
                
        if self.dose_file is not None:
            #if available dose is effective,  scale down by 10%
            
            self._dose_arr, self._dose_ds = load_dose_grid(self.dose_file)
            dose_type = getattr(self._dose_ds, "DoseType", "").upper()
            
            if dose_type in ("EFFECTIVE"):
                self._is_scaling_performed_from_RBEw_dose  = True
                self._dose_arr = self._dose_arr/1.1
                
        if self.let_file is not None:
            self._let_arr, self._let_ds = load_dose_grid(self.let_file)
    
        if self.rtstruct is not None:
            self._rtstruct_ds = pydicom.dcmread(str(self.rtstruct))
    
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
    def analyse(self,
                structures    : list = None,
                rbe_cfg       : RBEConfig            = None,
                metric_cfg    : MetricConfig         = None,
                radiobio_cfg  : RadiobiologyConfig   = None,
                resample_on_CT      : bool                 = False,
                resample_on_custom_grid      : bool                 = False,
                use_fractional: bool                 = False,
                supersample   : int                  = 4) -> "AnalysisResults":
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
        supersample   : supersampling factor for fractional mask (N×N sub-points
                        per voxel). Only used when use_fractional=True.
                        Default: 4 (16 sub-points per voxel, ~6% accuracy).

        Returns
        -------
        AnalysisResults
        """
        self.load()
        if self._dose_arr is None and self._let_arr is None:
            warnings.warn("Neither dose nor LET could be loaded — nothing to analyse.")
            return None
        if self._rtstruct_ds is None:
            warnings.warn("No RT Struct loaded — cannot extract any structures.")
            return None   
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
        print(f"  Analysing {len(structures)} structures [{mode} mask]...")

        
        # --- resample dose/LET only if the array exists ---
        sitk_dose = _np_to_sitk(self._dose_arr, self._dose_ds) if self._dose_arr is not None else None
        sitk_let  = _np_to_sitk(self._let_arr,  self._let_ds)  if self._let_arr  is not None else None

        grid_origin = grid_spacing = grid_shape = grid_z_positions = None
        
        if (not self._CT_loaded) & (resample_on_CT):
            raise ValueError("No CT has been loaded for resampling")
            
            
        if (resample_on_custom_grid) & (resample_on_CT):
            raise ValueError("Two resampling criteria have been selected. Please select only one.")
            
        
        if (self._CT_loaded) & (resample_on_CT):
            
            print("  Resampling dose and LET onto CT grid...")
           
            if sitk_dose is not None:
                res_dose = resample_dose_on_ct(sitk_dose,self._CT_sitk)  
                self._dose_arr = sitk.GetArrayFromImage(res_dose)     #sitk_dose	
                
            if sitk_let is not None:    
                res_let = resample_dose_on_ct(sitk_let, self._CT_sitk)
                self._let_arr = sitk.GetArrayFromImage(res_let)       #sitk_let	
            
            #Set grid for resampling and creating struct masks
            grid_origin = self._CT_geom['origin']
            grid_spacing = self._CT_geom['spacing'] #x,y,z
            grid_shape = self._CT_geom['shape']
            grid_z_positions = self._CT_geom['z_positions']
            
        elif resample_on_custom_grid:
                grid_new = metric_cfg.New_grid
                
                print(f"  Resampling dose and LET onto custom  {grid_new} grid...")
                
                if sitk_dose is not None:
                    res_dose, self._dose_arr, dose_geom, self._dose_ds = resample_dose_to_new_grid(sitk_dose,
                                                                                            self._dose_ds,
                                                                                            grid_new)
                if sitk_let is not None:  
                    res_let, self._let_arr, _, self._let_ds = resample_dose_to_new_grid (sitk_let,
                                                                             self._let_ds,
                                                                             grid_new)
                
                #Set grid for resampling and creating struct masks
                grid_origin = dose_geom['origin']
                grid_spacing = dose_geom['spacing'] #x,y,z
                grid_shape = dose_geom['shape'] #z,y,x
                grid_z_positions = dose_geom['z_positions']
                
        else:
            
            print("No Resampling for dose and LET")
            
            # no CT, no resample requested — use whichever grid (dose preferred, else LET) is available
            ref_ds = self._dose_ds if self._dose_ds is not None else self._let_ds
            if ref_ds is not None:
                grid_origin, grid_spacing = get_grid_geometry(ref_ds)
                grid_shape       = (self._dose_arr.shape if self._dose_arr is not None
                                    else self._let_arr.shape)
                z_offsets        = [float(v) for v in ref_ds.GridFrameOffsetVector]
                grid_z_positions = np.array([grid_origin[2] + o for o in z_offsets])
        
        if grid_origin is None:
            warnings.warn("Could not determine a voxel grid — aborting analysis.")
            return None
        
            
        all_rows    = []
        dvh_data    = {}              # {struct: {dose_type: (edges, cum)}}
        lvh_data    = {}              # {struct: (edges, cum)}
        dlvh_data_diff   = {}         # {struct: (H, d_edges, l_edges)}
        dlvh_data_cum   = {}          # {struct: (H, d_edges, l_edges)}

        vol_cc = self.voxel_volume_cc
        for struct_name in structures:
            print(f"    → {struct_name}")

            try:
                if use_fractional and (grid_origin):
                    
                    frac  = get_fractional_mask_on_grid(struct_name,
                                                    self._rtstruct_ds,
                                                    grid_origin,
                                                    grid_spacing,
                                                    grid_shape,
                                                    grid_z_positions,
                                                    supersample = supersample)
                    mask    = frac > 0
                    weights = frac[mask]
                    
                elif not use_fractional and (grid_origin):
                        
                    mask = get_structure_mask_on_grid(struct_name,
                                                   self._rtstruct_ds,
                                                  grid_origin,
                                                  grid_spacing,
                                                  grid_shape,
                                                  grid_z_positions)
                    
                    weights = None   # binary → unweighted
                else:
                    mask = np.zeros(1)
                    weights = None

                    
            except Exception as e:
                warnings.warn(f"Skipping '{struct_name}': {e}")
                continue
            
            if mask.sum() == 0:
                warnings.warn(f"Empty mask for '{struct_name}' – skipping.")
                continue
            ab     = radiobio_cfg.get_alpha_beta(struct_name)
            geud_a = radiobio_cfg.get_geud_a(struct_name)
            
            dose_vox = self._dose_arr[mask > 0] if self._dose_arr is not None else None
            let_vox  = self._let_arr[mask > 0]  if self._let_arr  is not None else None 
            
            if self._dose_arr is not None:
                dose_vox = self._dose_arr[mask>0]
            if self._let_arr is not None: 
                let_vox  = self._let_arr[mask>0]
            
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
            
            if self._dose_arr is not None:
                row.update(compute_dvh_metrics(dose_vox, vol_cc, "Phys",
                                               metric_cfg, ab, geud_a,
                                               weights=weights,n_fractions = self.n_fractions))
                
                edges, cum = compute_cumulative_histogram(dose_vox,
                                                          metric_cfg.dvh_bins,
                                                          weights=weights)
                dvh_data[struct_name]["Phys"] = (edges, cum)
                # ---- Fixed RBE ----
                
                dose_fixed = rbe_fixed(dose_vox, rbe_cfg.fixed_rbe)
                lbl_fixed  = f"RBE{rbe_cfg.fixed_rbe}"
                row.update(compute_dvh_metrics(dose_fixed, vol_cc,  lbl_fixed ,
                                               metric_cfg, ab, geud_a,
                                               weights=weights,n_fractions = self.n_fractions))
                edges_f, cum_f = compute_cumulative_histogram(dose_fixed,
                                                               metric_cfg.dvh_bins,
                                                               weights=weights)
                dvh_data[struct_name][lbl_fixed] = (edges_f, cum_f)
                
            
                if self._let_arr is not None:
                    # ---- Variable RBE (per model) ----
                    for model in rbe_cfg.models:
                        rbe_dose_vox = compute_rbe_dose(dose_vox, let_vox, self.n_fractions, ab, model)
                        row.update(compute_dvh_metrics(rbe_dose_vox, vol_cc, model,
                                                       metric_cfg, ab, geud_a,
                                                       weights=weights,n_fractions = self.n_fractions))
                        e_m, c_m = compute_cumulative_histogram(rbe_dose_vox,
                                                                 metric_cfg.dvh_bins,
                                                                 weights=weights)
        
                        dvh_data[struct_name][model] = (e_m, c_m)
                    # ---- LET-thresholded DVH metrics (high-LET subvolume on fixed RBE - dirty dose) ----
                    
                    high_let_idx = let_vox >= metric_cfg.LET_thr
                    
                    if high_let_idx.sum() > 0:
                        w_high = weights[high_let_idx] if weights is not None else None
                        row.update(compute_dvh_metrics(
                            dose_vox[high_let_idx], vol_cc,
                            "highLET_fixed_rbe", metric_cfg, ab, geud_a,
                            weights=w_high,n_fractions = self.n_fractions))

                    # ---- 2-D DLVH ----
                    H_diff, H_cum, d_edges, l_edges_2d = compute_2d_histogram(
                        dose_fixed, let_vox,
                        metric_cfg.dlvh_dose_bins,
                        metric_cfg.dlvh_let_bins)
                    dlvh_data_diff[struct_name] = (H_diff, d_edges, l_edges_2d)
                    dlvh_data_cum[struct_name] = (H_cum, d_edges, l_edges_2d)
                    
                    
            if self._let_arr is not None:       
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
            for RBE in list_RBE_calculated:
                NTCP_dict[RBE]  = NTCP_base.compute_NTCP(RBE, self.metrics_df)
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
