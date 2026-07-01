"""
PyPlanAnalysis.io
=============
DICOM discovery and loading, CT/dose grid resampling, and RT Struct
mask extraction (binary and fractional).
"""

import warnings

import numpy as np
from pathlib import Path
from typing import Union
from matplotlib.path import Path as MplPath

import pydicom
from scipy.interpolate import splprep, splev
from collections import defaultdict

from skimage.draw import polygon as sk_polygon

# We read RT Struct contours directly via pydicom for reliability.

"""
DICOM auto-discovery for an RT (proton/photon) dataset folder.

Given a folder that may contain a mix of CT slices, one or more RTSTRUCT,
RTPLAN and RTDOSE (physical dose + LET) files — possibly several
candidates of each, possibly nested in sub-folders, possibly with some
files that don't actually belong together — this finds the one
self-consistent set by walking the standard DICOM cross-reference chain:

    RTPLAN
        ReferencedStructureSetSequence[0]
            (0008,1150) ReferencedSOPClassUID
            (0008,1155) ReferencedSOPInstanceUID   -> RTSTRUCT
                                                        (0008,0016) SOPClassUID
                                                        (0008,0018) SOPInstanceUID
    RTSTRUCT
        ReferencedFrameOfReferenceSequence
            -> RTReferencedStudySequence
            -> RTReferencedSeriesSequence
            -> (0020,000E) SeriesInstanceUID        -> CT series
    RTDOSE
        ReferencedRTPlanSequence[0]
            (0008,1155) ReferencedSOPInstanceUID    -> RTPLAN
                                                        (0008,0018) SOPInstanceUID

If the first candidate of each type doesn't line up, instead of failing
the function tries other candidates present in the folder until it finds
a combination where every link checks out. If no fully-verified
combination exists, it falls back to the best partial match it can build
and explains exactly what couldn't be confirmed via `link_warnings`.
"""
try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError(
        "SimpleITK is required for CT-grid resampling. "
        "Install with: pip install SimpleITK"
    )


from dataclasses import dataclass, field
from typing import Optional


def _collect_candidates(folder: Path):
    
    
    @dataclass
    class CTSeries:
        series_uid: str
        directory: Path
        patient_id: Optional[str] = None
        frame_of_ref_uid: Optional[str] = None
    
    @dataclass
    class RTStructCandidate:
        path: Path
        sop_class: Optional[str]
        sop_uid: Optional[str]
        patient_id: Optional[str]
        ref_series_uids: set = field(default_factory=set)
        frame_of_ref_uid: Optional[str] = None
    
    @dataclass
    class RTPlanCandidate:
        path: Path
        sop_class: Optional[str]
        sop_uid: Optional[str]
        patient_id: Optional[str]
        ref_struct: Optional[tuple] = None  # (ReferencedSOPClassUID, ReferencedSOPInstanceUID)
    
    
    @dataclass
    class RTDoseCandidate:
        path: Path
        sop_uid: Optional[str]
        patient_id: Optional[str]
        dose_kind: str  # "PHYSICAL", "LET", or "EFFECTIVE"
        dose_SumType: str #"PLAN" or "BEAM"
        ref_plan_sop: Optional[str] = None
        frame_of_ref_uid: Optional[str] = None 

    """One recursive pass over every *.dcm file in `folder` (including
    sub-folders), sorting each into a lightweight candidate record keyed
    by modality. Searching recursively means files don't need to live
    directly in `folder` for this to find them."""
    ct_series: dict = {}
    rtstructs: list = []
    rtplans: list = []
    rtdoses: list = []

    for f in folder.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except Exception:
            continue
        modality = getattr(ds, "Modality", "")
        patient_id = getattr(ds, "PatientID", None)

        if modality == "CT":
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if series_uid and series_uid not in ct_series:
                ct_series[series_uid] = CTSeries(series_uid, f.parent, patient_id,
                                  frame_of_ref_uid=getattr(ds, "FrameOfReferenceUID", None))

                
        elif modality == "RTSTRUCT":
            ref_series_uids = set()
            for frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
                struct_frame_uid = getattr(frame, "FrameOfReferenceUID", None)
                for study in getattr(frame, "RTReferencedStudySequence", []):
                    for series in getattr(study, "RTReferencedSeriesSequence", []):
                        uid = getattr(series, "SeriesInstanceUID", None)
                        if uid:
                            ref_series_uids.add(uid)
            rtstructs.append(RTStructCandidate(
                path=f,
                sop_class=getattr(ds, "SOPClassUID", None),
                sop_uid=getattr(ds, "SOPInstanceUID", None),
                patient_id=patient_id,
                ref_series_uids=ref_series_uids,
                frame_of_ref_uid=struct_frame_uid,
            ))

        elif modality == "RTPLAN":
            ref_struct = None
            ref_seq = getattr(ds, "ReferencedStructureSetSequence", None)
            if ref_seq:
                ref = ref_seq[0]
                ref_struct = (
                    getattr(ref, "ReferencedSOPClassUID", None),
                    getattr(ref, "ReferencedSOPInstanceUID", None),
                )
            rtplans.append(RTPlanCandidate(
                path=f,
                sop_class=getattr(ds, "SOPClassUID", None),
                sop_uid=getattr(ds, "SOPInstanceUID", None),
                patient_id=patient_id,
                ref_struct=ref_struct,
            ))

        elif modality == "RTDOSE":
            dose_type = getattr(ds, "DoseType", "").upper()
            label = getattr(ds, "DoseComment", "").upper()
            dose_SumType = getattr(ds, "DoseSummationType", "").upper()
            fname = f.name.upper()
            is_let = ("LET" in label or "LET" in fname or dose_type == "LET")
            if is_let:
                dose_kind = "LET"
            elif dose_type in ("PHYSICAL"):
                dose_kind = "PHYSICAL"
            elif dose_type in ("EFFECTIVE"):
                dose_kind = "EFFECTIVE"
            else:
                dose_kind = "UNKNOWN"

            ref_plan_sop = None
            ref_plan_seq = getattr(ds, "ReferencedRTPlanSequence", None)
            if ref_plan_seq:
                ref_plan_sop = getattr(ref_plan_seq[0], "ReferencedSOPInstanceUID", None)

            rtdoses.append(RTDoseCandidate(
                path=f,
                sop_uid=getattr(ds, "SOPInstanceUID", None),
                patient_id=patient_id,
                dose_kind=dose_kind,
                dose_SumType=dose_SumType,
                ref_plan_sop=ref_plan_sop,
                frame_of_ref_uid=getattr(ds, "FrameOfReferenceUID", None)
            ))

    return ct_series, rtstructs, rtplans, rtdoses

def _best_dose(doses_of_kind, plan, fallback_pool, kind_label,
               frame_of_ref_uid, link_warnings):
    """
    Pick the best RTDOSE candidate of one kind ("physical" or "LET"),
    trying progressively weaker (but still verifiable) links instead of
    ever guessing. If nothing can be verified, return a sentinel object
    that will cause a clear crash downstream rather than silently
    proceeding with a wrong file.
    """

    # if no candidates of this kind AND no fallback candidates, nothing to pick
    if not doses_of_kind and not fallback_pool:
        return None

    # --- opt 1: RTPLAN is available and a candidate references it ---
    # this is the strongest possible link: RTDOSE -> RTPLAN -> RTSTRUCT
    if plan is not None and doses_of_kind:
        match = next((d for d in doses_of_kind if d.ref_plan_sop == plan.sop_uid), None)
        if match is not None:
            return match

    # --- opt 2: no RTPLAN match (or no RTPLAN at all), but a candidate
    # shares the FrameOfReferenceUID with the matched CT/RTSTRUCT ---
    # this works even when RTPLAN is completely missing from the folder
    if frame_of_ref_uid and doses_of_kind:
        match = next((d for d in doses_of_kind
                      if d.frame_of_ref_uid == frame_of_ref_uid), None)
        if match is not None:
            if plan is not None:
                link_warnings.append(
                    f"No {kind_label} RTDOSE references the matched RTPLAN, "
                    "but one shares the same FrameOfReferenceUID as the "
                    "matched CT/RTSTRUCT; using that instead."
                )
            else:
                link_warnings.append(
                    f"No RTPLAN available, but one {kind_label} RTDOSE shares the same FrameOfReferenceUID as the "
                    "matched CT/RTSTRUCT; using that instead."
                )
            return match


    # --- no candidates of the requested kind at all: repeat the same
    # three tiers against the fallback pool (e.g. EFFECTIVE in case of physical dose) ---

    # RTPLAN reference
    if plan is not None and fallback_pool:
        match = next((d for d in fallback_pool if d.ref_plan_sop == plan.sop_uid), None)
        if match is not None:
            link_warnings.append(
                f"No RTDOSE was tagged or named as {kind_label}; using an "
                "EFFECTIVE RTDOSE because it references the matched RTPLAN."
            )
            
            return match

    #  FrameOfReferenceUID match
    if frame_of_ref_uid and fallback_pool:
        match = next((d for d in fallback_pool
                      if d.frame_of_ref_uid == frame_of_ref_uid), None)
        if match is not None:
            if plan is not None:
                link_warnings.append(
                    f"No fallback RTDOSE references the matched RTPLAN, "
                    "but one shares the same FrameOfReferenceUID as the "
                    "matched CT/RTSTRUCT; using that instead."
                )
            else:
                link_warnings.append(
                    f"No RTPLAN available, but one fallback RTDOSE shares the same FrameOfReferenceUID as the "
                    "matched CT/RTSTRUCT; using that instead."
                )
            return match

    # If no matches are found, and exactly one unlinked candidate of the selected kind exists, pick it ---
    # weak evidence, but in single-plan-per-folder layouts this is
    # usually correct; flag it so the caller can decide whether to trust it
    if len(doses_of_kind) == 1:
        link_warnings.append(
            f"Only one {kind_label} RTDOSE file found; USING IT WITHOUT MATCH to the RTPLAN or CT/RTSTRUCT."
        )
        return doses_of_kind[0]

    # Multiple candidates of the selected kind, none verifiable ---
    # force a hard failure 
    if len(doses_of_kind) > 1:
        link_warnings.append(
            f"Multiple {kind_label} RTDOSE files found and none verifiably "
            "link to the matched RTPLAN or CT/RTSTRUCT — refusing to guess."
        )
        return _AMBIGUOUS_DOSE
    
    
    # If no matches are found, and exactly one unlinked candidate of the fallback dose type exists, pick it
    if len(fallback_pool) == 1:
        link_warnings.append(
            f"No RTDOSE was tagged or named as {kind_label}; using the only "
            "EFFECTIVE RTDOSE found, without a verified link."
        )
        return fallback_pool[0]

    #  multiple unlinked fallback candidates — force a hard failure
    if len(fallback_pool) > 1:
        link_warnings.append(
            f"No RTDOSE was tagged or named as {kind_label}, and multiple "
            "EFFECTIVE RTDOSE files exist with no verifiable link — "
            "refusing to guess."
        )
        return _AMBIGUOUS_DOSE

    return None


def _strict_chain(ct_series, rtstructs, rtplans):
    """Try every RTPLAN/RTSTRUCT pairing (or RTSTRUCT alone, if no RTPLAN
    exists) until one is found whose reference tags fully check out
    against a CT series actually present in the folder. Returns
    (plan, struct, series_uid, fully_verified) or (None, None, None, False)
    if nothing checks out."""
    plan_candidates = rtplans if rtplans else [None]

    for plan in plan_candidates:
        if plan is not None:
            struct = next(
                (s for s in rtstructs if plan.ref_struct == (s.sop_class, s.sop_uid)),
                None,
            )
        else:
            struct = next((s for s in rtstructs if s.ref_series_uids & set(ct_series)), None)

        if struct is None:
            continue

        series_uid = next((uid for uid in struct.ref_series_uids if uid in ct_series), None)
        if series_uid is None:
            continue

        return plan, struct, series_uid, (plan is not None)

    return None, None, None, False


def _fallback_chain(ct_series, rtstructs, rtplans, link_warnings):
    """No fully cross-referenced combination exists. Build the best
    available guess one piece at a time, logging exactly what had to be
    assumed instead of confirmed."""
    struct = next((s for s in rtstructs if s.ref_series_uids & set(ct_series)), None)
    if struct is not None:
        series_uid = next(uid for uid in struct.ref_series_uids if uid in ct_series)
    else:
        series_uid = next(iter(ct_series), None)
        if rtstructs:
            link_warnings.append(
                "No RTSTRUCT references any of the discovered CT series; using "
                "the first RTSTRUCT found without a verified CT link."
            )
            struct = rtstructs[0]
            if series_uid is not None:
                link_warnings.append(
                    f"Assuming CT series {series_uid} since it could not be "
                    "confirmed via RTSTRUCT reference tags."
                )

    plan = None
    if struct is not None:
        plan = next(
            (p for p in rtplans if p.ref_struct == (struct.sop_class, struct.sop_uid)),
            None,
        )
    if plan is None and rtplans:
        link_warnings.append(
            "No RTPLAN references the selected RTSTRUCT; using the first "
            "RTPLAN found without a verified link."
        )
        plan = rtplans[0]

    return plan, struct, series_uid

class _AmbiguousDose:
    """
    Sentinel returned when multiple RTDOSE candidates exist and none can
    be verifiably linked to the matched plan/structure. Any attempt to use
    this as a real candidate (accessing .path, .sop_uid, etc.) raises
    AttributeError immediately, so the ambiguity surfaces as a hard crash
    instead of a silently wrong file being picked.
    """
    def __getattr__(self, name):
        raise AttributeError(
            f"Ambiguous RTDOSE match: cannot access '.{name}' — multiple "
            "unlinked candidates were found and none could be picked safely. "
            "Resolve manually (check link_warnings) before proceeding."
        )

    def __bool__(self):
        # so `if dose:` style checks still behave like "something is there"
        # forcing any downstream .path access to be the point of failure
        return True


_AMBIGUOUS_DOSE = _AmbiguousDose()

def find_dicom_files(folder: Path) -> dict:
    """
    Auto-discover RT Dose (physical dose, LET), RT Struct, RT Plan and CT
    files belonging to the same plan, by inspecting DICOM modality tags
    and cross-reference (Referenced UID) tags. The folder (and its
    sub-folders) may contain extra or unrelated files of any of these
    types; this function searches through all of them for the one set that is
    actually linked together.

    Returns
    -------
    dict with keys:
        "dose", "let", "rtstruct", "rtplan" : Path or None
        "CT"          : Path to the directory holding the matched CT
                        series, or None
        "Patient_ID"  : str or None
        "linked"      : True if RTPLAN -> RTSTRUCT -> CT was fully
                        confirmed via DICOM reference tags; False if a
                        fallback/best-guess match had to be used; None
                        if there wasn't enough data to even attempt the
                        check (e.g. no RTSTRUCT and no CT found at all).
        "link_warnings": list[str], one entry per fallback
                        that was needed, explaining what couldn't be
                        confirmed and what was used instead.
    """
    folder = Path(folder)
    link_warnings: list = []

    ct_series, rtstructs, rtplans, rtdoses = _collect_candidates(folder)

    plan, struct, series_uid, verified = _strict_chain(ct_series, rtstructs, rtplans)

    if struct is None and (rtstructs or ct_series):
        # The strict search found nothing usable at all; fall back.
        msg = "Could not find an RTPLAN/RTSTRUCT/CT combination that fully "
        "cross-references; falling back to best-effort matching."
        link_warnings.append(msg)
        print(msg)
        
        plan, struct, series_uid = _fallback_chain(ct_series, rtstructs, rtplans, link_warnings)
        verified = False
    elif plan is None and rtplans:
        
        # A struct/CT-only chain was found (e.g. no RTPLAN references it),
        # but there are RTPLAN files sitting in the folder we never matched.
        msg = f"{len(rtplans)} RTPLAN file(s) found but none reference the "
        "matched RTSTRUCT; proceeding without a confirmed RTPLAN."
        link_warnings.append(msg)

    physical_doses = [d for d in rtdoses if (d.dose_kind == "PHYSICAL" and d.dose_SumType == "PLAN")]
    effective_doses = [d for d in rtdoses if (d.dose_kind == "EFFECTIVE" and d.dose_SumType == "PLAN")]
    let_doses = [d for d in rtdoses if d.dose_kind == "LET"]
    
    frame_of_ref_uid = struct.frame_of_ref_uid if struct is not None else None
    
    #look for best matching physical dose, at worst, look for effective dose and scale by 10%
    dose = _best_dose(physical_doses, plan, effective_doses, "physical",
                      frame_of_ref_uid, link_warnings)
    let = _best_dose(let_doses, plan, [], "LET",
                     frame_of_ref_uid, link_warnings)
    
    ct_dir = ct_series[series_uid].directory if series_uid in ct_series else None

    patient_id = None
    for obj in (struct, plan, dose, let):
        if obj is not None and obj.patient_id:
            patient_id = obj.patient_id
            break
    if patient_id is None and series_uid in ct_series:
        patient_id = ct_series[series_uid].patient_id

    linked = None
    if struct is not None or ct_series:
        linked = verified and len(link_warnings) == 0

    found = {
        "dose": dose.path if dose else None,
        "let": let.path if let else None,
        "rtstruct": struct.path if struct else None,
        "rtplan": plan.path if plan else None,
        "CT": ct_dir,
        "Patient_ID": patient_id,
        "linked": linked,
        "link_warnings": link_warnings
    }
    
    for w in link_warnings:
        print("⚠", w)
    
    return found


#%%

def load_ct_series(ct_folder: Union[str, Path]) -> tuple:
    """
    Load a multi-slice CT DICOM series from a folder.
 
    Slices are sorted by ImagePositionPatient z-coordinate.
    Spacing is taken from PixelSpacing of the first slice and the
    z-step between consecutive slice positions.
 
    Returns
    -------
    sitk_image : SimpleITK.Image  (x, y, z ordering internally)
    ct_geometry : dict with keys:
        "origin"      : [x0, y0, z0]  mm  (corner of first voxel)
        "spacing"     : [dx, dy, dz]  mm
        "shape"       : (nz, ny, nx)  — numpy (z,y,x) convention
        "z_positions" : np.ndarray of slice z-coordinates  length nz
    """
    ct_folder = Path(ct_folder)
    slices = []
    for f in ct_folder.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "CT":
                slices.append((float(ds.ImagePositionPatient[2]), str(f), ds))
        except Exception:
            continue
 
    if not slices:
        raise FileNotFoundError(f"No CT DICOM files found in {ct_folder}")
 
    slices.sort(key=lambda t: t[0])          # sort by z-position
    z_positions = np.array([s[0] for s in slices])
    first_ds    = slices[0][2]
 
    pix_sp = [float(v) for v in first_ds.PixelSpacing]   # [row_sp=dy, col_sp=dx]
    dx, dy = pix_sp[1], pix_sp[0]
    dz     = float(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else float(first_ds.SliceThickness)
    origin = [float(v) for v in first_ds.ImagePositionPatient]  # [x0, y0, z0]
 
    # Use SimpleITK series reader for correct pixel data ordering
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(ct_folder))
    if not dicom_names:
        # fallback: use our sorted file list
        dicom_names = [s[1] for s in slices]
    reader.SetFileNames(dicom_names)
    sitk_image = reader.Execute()
 
    nz = len(slices)
    ny = int(first_ds.Rows)
    nx = int(first_ds.Columns)
 
    ct_geometry = {
        "origin"     : origin,          # itk [x0, y0, z0]
        "spacing"    : [dx, dy, dz],    # itk [dx, dy, dz]
        "shape"      : (nz, ny, nx),    # numpy (x, y, z)
        "z_positions": z_positions,
    }
 
    print(f"CT loaded: {nz} slices  spacing=({dx:.2f},{dy:.2f},{dz:.2f}) mm  "
          f"shape={ct_geometry['shape']}")
    return sitk_image, ct_geometry

def _np_to_sitk(arr, ds):
    """Wrap a numpy (z,y,x) dose array as a properly georeferenced sitk image."""
    img = sitk.GetImageFromArray(arr)
    origin, spacing = get_grid_geometry(ds) #[dx, dy, dz]
    
    # SimpleITK spacing order: (x, y, z)
    img.SetSpacing(spacing)
    img.SetOrigin(tuple(origin))
    return img

def resample_dose_to_new_grid(
                                dose_sitk,
                                dose_ds,
                                new_spacing,
                                interpolator = sitk.sitkLinear):
    """
    Resample RTDOSE onto a new isotropic/anisotropic grid
    Parameters
    ----------
    dose_sitk : sitk.Image Original dose image
    dose_ds : pydicom Dataset RTDOSE dataset
    new_spacing : tuple/list (sx, sy, sz) in mm

    Returns
    -------
    resampled_dose : sitk.Image
    updated_info : dict
    dose_ds : updated dataset
    """

    old_spacing = np.array(dose_sitk.GetSpacing())
    old_size = np.array(dose_sitk.GetSize())
    old_origin = dose_sitk.GetOrigin()
    old_direction = dose_sitk.GetDirection()
    # physical extent
    physical_size = old_spacing * old_size
    # -----------------------------
    new_spacing = np.array(new_spacing)

    new_size = np.ceil( physical_size / new_spacing ).astype(int)

    # reference image
    ref = sitk.Image( [int(v) for v in new_size], dose_sitk.GetPixelID() )

    ref.SetSpacing(tuple(new_spacing))
    ref.SetOrigin(old_origin)
    ref.SetDirection(old_direction)

    # -----------------------------
    # RESAMPLE
    resampled_dose = sitk.Resample( dose_sitk,
                                    ref,
                                    sitk.Transform(),
                                    interpolator,
                                    0.0)

    dose_arr_resampled = sitk.GetArrayFromImage(  resampled_dose   )
    # numpy shape = z,y,x
    shape_np = dose_arr_resampled.shape
    z_spacing = new_spacing[2]
    z_positions = (old_origin[2]+ np.arange(shape_np[0]) * z_spacing )
    z_offsets = (z_positions - z_positions[0])

    # -----------------------------
    # UPDATE DICOM RTDOSE
    dose_ds.PixelSpacing = [ float(new_spacing[1]), float(new_spacing[0])]
    dose_ds.SliceThickness = float(z_spacing)
    dose_ds.GridFrameOffsetVector = [float(v) for v in z_offsets]
    dose_ds.Rows = shape_np[1]
    dose_ds.Columns = shape_np[2]
    dose_ds.NumberOfFrames = shape_np[0]

    updated_info = {
        "spacing": tuple(new_spacing),
        "origin": old_origin,
        "shape": shape_np,
        "z_positions": z_positions,
        "z_offsets": z_offsets,
    }

    return (resampled_dose,
        dose_arr_resampled,
        updated_info,
        dose_ds)


 
def resample_dose_on_ct(sitk_dose: sitk.Image,
                        sitk_ct:   sitk.Image) -> sitk.Image:
    """
    Resample a dose (or LET) SimpleITK image onto the CT grid.
 
    The CT image defines the output origin, spacing, direction, and size.
    This ensures that the resampled dose array is perfectly aligned with
    the CT grid on which contours will be rasterised.
 
    Parameters
    ----------
    sitk_dose : SimpleITK.Image  (dose or LET, in dose-grid coordinates)
    sitk_ct   : SimpleITK.Image  (full CT series)
 
    Returns
    -------
    SimpleITK.Image  same grid as sitk_ct
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(sitk_ct.GetSpacing())
    resampler.SetSize(sitk_ct.GetSize())
    resampler.SetOutputDirection(sitk_ct.GetDirection())
    resampler.SetOutputOrigin(sitk_ct.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(sitk_dose)
# **************************




#=======================================

def load_dose_grid(path: Union[str, Path]) -> tuple:
    """
    Load a DICOM RT Dose file (dose or LET stored as dose grid).

    Returns
    -------
    array : np.ndarray  shape (z, y, x)
    ds    : pydicom Dataset
    """
    ds    = pydicom.dcmread(str(path))
    scale = float(ds.DoseGridScaling)
    array = ds.pixel_array.astype(np.float64) * scale
    return array, ds


def get_grid_geometry(ds) -> tuple:
    """
    Extract (origin, spacing) from an RT Dose dataset.

    Returns
    -------
    origin  : [x0, y0, z0]  mm
    spacing : [dx, dy, dz]  mm
    """
    origin = [float(v) for v in ds.ImagePositionPatient]
    pix_sp = [float(v) for v in ds.PixelSpacing]  # [row_spacing=dy, col_spacing=dx]
    dz = float(ds.SliceThickness)
  
    # PixelSpacing = [row_spacing, col_spacing] = [dy, dx]
    return origin, [pix_sp[1], pix_sp[0], dz]   # [dx, dy, dz]



# ============================================================
#  Structure mask extraction  

def _build_roi_maps(rtstruct_ds) -> tuple:
    """
    Parse RT Struct and return two lookup dicts.

    Returns
    -------
    name_to_roi  : {roi_name_lower: roi_number}
    roi_to_contours : {roi_number: [ np.ndarray shape(N,3) ]}
        Each array is one contour polygon with columns [x, y, z] in mm.
    """
    # ROI names from StructureSetROISequence
    name_to_roi = {}
    for item in rtstruct_ds.StructureSetROISequence:
        name_to_roi[item.ROIName.strip().lower()] = int(item.ROINumber)

    # Contour coordinates from ROIContourSequence
    roi_to_contours = {}
    for roi_contour in rtstruct_ds.ROIContourSequence:
        roi_num = int(roi_contour.ReferencedROINumber)
        contours = []
        if not hasattr(roi_contour, "ContourSequence"):
            roi_to_contours[roi_num] = contours
            continue
        for contour in roi_contour.ContourSequence:
            raw = [float(v) for v in contour.ContourData]
            pts = np.array(raw).reshape(-1, 3)   # (N, 3)  x,y,z
            contours.append(pts)
        roi_to_contours[roi_num] = contours

    return name_to_roi, roi_to_contours


def get_all_structure_names(rtstruct_ds) -> list:
    """Return list of all structure names in an RT Struct dataset."""
    return [item.ROIName.strip()
            for item in rtstruct_ds.StructureSetROISequence]




def get_structure_mask_on_grid(struct_name: str,
                               rtstruct_ds,
                               origin:      list,
                               spacing:     list,
                               shape:       tuple,
                               z_positions: np.ndarray) -> np.ndarray:
    """
    Rasterise RT Struct contours for `struct_name` onto an arbitrary grid.
 
    This is the core function used for both dose-grid and CT-grid masking.
    Contour z-values are matched to the nearest z in z_positions.
 
    Parameters
    ----------
    struct_name  : str
    rtstruct_ds  : pydicom Dataset
    origin       : [x0, y0, z0]  mm — physical coordinate of voxel (0,0,0) corner
    spacing      : [dx, dy, dz]  mm
    shape        : (nz, ny, nx) — numpy array shape
    z_positions  : 1-D array of z-coordinates for each slice (length nz)
 
    Returns
    -------
    mask : np.ndarray bool, shape (nz, ny, nx)
    """
    name_to_roi, roi_to_contours = _build_roi_maps(rtstruct_ds)
 
    key = struct_name.strip().lower()
    if key not in name_to_roi:
        raise ValueError(
            f"Structure '{struct_name}' not found in RT Struct. "
            f"Available: {[item.ROIName for item in rtstruct_ds.StructureSetROISequence]}"
        )
    roi_number = name_to_roi[key]
    contours   = roi_to_contours.get(roi_number, [])
 
    x0, y0, z0 = origin
    dx, dy, dz  = spacing
    nz, ny, nx  = shape
 
    mask = np.zeros(shape, dtype=bool)
 
    if not contours:
        warnings.warn(f"No contour data for '{struct_name}'.")
        return mask
 
    # Build grid of voxel-centre x,y coordinates (voxel centres = origin + (i+0.5)*spacing)
    # Note: DICOM ImagePositionPatient is the centre of the first voxel, so:
    #   voxel centre i  →  x0 + i*dx
    xi = np.arange(nx)
    yi = np.arange(ny)
    XX, YY  = np.meshgrid(x0 + xi * dx, y0 + yi * dy)
    grid_xy = np.column_stack([XX.ravel(), YY.ravel()])
 
    for pts in contours:
        z_val = float(pts[0, 2])
        z_idx = int(np.argmin(np.abs(z_positions - z_val)))
 
        poly_xy = pts[:, :2]
        if len(poly_xy) < 3:
            continue
 
        poly   = MplPath(poly_xy)
        inside = poly.contains_points(grid_xy).reshape(ny, nx)
        mask[z_idx] |= inside
 
    return mask
 
def contour_area_signed(xy):
    """Shoelace formula — positive = CCW, negative = CW"""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def smooth_contour(poly_xy, n_pts=300):
    try:
        x, y = poly_xy[:, 0], poly_xy[:, 1]
        tck, _ = splprep([x, y], s=0, per=True)
        x_s, y_s = splev(np.linspace(0, 1, n_pts), tck)
        return np.column_stack([x_s, y_s]) 
    except Exception as e:
        warnings.warn(f"Spline failed: {e}")
        return poly_xy

def rasterize_supersampled(smooth_xy, x0, y0, dx, dy, ny, nx, N):
    xi = (smooth_xy[:, 0] - x0) / dx * N
    yi = (smooth_xy[:, 1] - y0) / dy * N

    rr, cc = sk_polygon(yi, xi, shape=(ny * N, nx * N))
    super_mask = np.zeros((ny * N, nx * N), dtype=np.float32)
    super_mask[rr, cc] = 1.0
    return super_mask.reshape(ny, N, nx, N).mean(axis=(1, 3))


def compute_roi_volume_comparison(frac_mask, dx, dy, dz):
    
    # Your fractional volume
    vol_frac = frac_mask.sum() * dx * dy * dz / 1000.0


    # Per-slice breakdown
    frac_per_slice   = frac_mask.sum(axis=(1,2)) * dx * dy * dz / 1000.0

    print(f"Fractional volume : {vol_frac:.4f} cc")
    print("\nPer-slice :")
    for z, f in enumerate(frac_per_slice):
        if f > 0 :
            print(f"  slice {z:3d}: {f:.4f}")

    return vol_frac

def prismatoid_volume(frac_mask, dx, dy, dz):
    areas = frac_mask.sum(axis=(1,2)) * dx * dy
    nz = len(areas)
    if nz < 2:
        return areas.sum() * dz / 1000.0
    vol = 0.0
    for i in range(nz - 1):
        A0, A1 = areas[i], areas[i+1]
        Am = (A0 + A1) / 2.0
        vol += (dz / 6.0) * (A0 + 4*Am + A1)
    return vol / 1000.0

         
def get_fractional_mask_on_grid(struct_name: str,
                                rtstruct_ds,
                                origin:      list,
                                spacing:     list,
                                shape:       tuple,
                                z_positions: np.ndarray,
                                supersample: int = 4) -> np.ndarray:
    """
    Compute a fractional voxel membership mask on an arbitrary grid.
 
    Each voxel receives a value in [0,1] — the fraction of its physical
    area (in xy) that lies inside the RT Struct contour, estimated by
    supersampling (supersample² sub-points per voxel).
 
    Parameters
    ----------
    struct_name  : str
    rtstruct_ds  : pydicom Dataset
    origin       : [x0, y0, z0]  mm
    spacing      : [dx, dy, dz]  mm
    shape        : (nz, ny, nx)
    z_positions  : 1-D array length nz
    supersample  : N subdivisions per side (default 4 → 16 sub-points/voxel)
 
    Returns
    -------
    frac_mask : np.ndarray float32, shape (nz, ny, nx), values in [0, 1]
    """
    name_to_roi, roi_to_contours = _build_roi_maps(rtstruct_ds)
 
    key = struct_name.strip().lower()
    if key not in name_to_roi:
        raise ValueError(
            f"Structure '{struct_name}' not found in RT Struct. "
            f"Available: {[item.ROIName for item in rtstruct_ds.StructureSetROISequence]}"
        )
    roi_number = name_to_roi[key]
    contours   = roi_to_contours.get(roi_number, [])
 
    x0, y0, z0 = origin
    dx, dy, dz  = spacing
    nz, ny, nx  = shape
 
    frac_mask = np.zeros(shape, dtype=np.float32)
 
    if not contours:
        warnings.warn(f"No contour data for '{struct_name}'.")
        return frac_mask
 
    N  = supersample

    # Group contours by slice
    slice_contours = defaultdict(list)
    n = 0
    for pts in contours:
        n+=1
        z_val = float(pts[0, 2])
        diffs = np.abs(z_positions - z_val)
        z_idx = int(np.argmin(diffs))
        if diffs[z_idx] > dz * 0.5:
            warnings.warn(f"Contour z={z_val:.2f}mm is {diffs[z_idx]:.2f}mm from nearest slice")
        slice_contours[z_idx].append(pts[:, :2])

    
    for z_idx, slice_polys in slice_contours.items():
        if not slice_polys:
            continue
    
        areas = [abs(contour_area_signed(p)) for p in slice_polys]
        slice_polys = [slice_polys[i] for i in np.argsort(areas)[::-1]]
    
        slice_fraction = np.zeros((ny, nx), dtype=np.float32)
    
        for i, poly_xy in enumerate(slice_polys):
            if len(poly_xy) < 3:
                continue
    
            smooth_xy = poly_xy #smooth_contour(poly_xy) #Avoid soothing, no gain in accuracy compared to RayStation.
            
            # Hole detections
            is_hole = False
            if i > 0:# assume the biggest area as non-hole
                for j in range(i): # handling holes if holes do not have any island inside. 
                    outer_path = MplPath(slice_polys[j], closed=True) 
                    test_pts   = poly_xy[::max(1, len(poly_xy) // 5)]
                    if outer_path.contains_points(test_pts).mean() > 0.5:
                        is_hole = True
                        break
           
            fraction = rasterize_supersampled(smooth_xy, x0, y0, dx, dy, ny, nx, N)
    
            if is_hole:
                slice_fraction -= fraction
            else:
                slice_fraction += fraction
    
        frac_mask[z_idx] = np.clip(slice_fraction, 0.0, 1.0) #clipping shouldn#t change anything, it should be already in [0-1]
        
        
    return frac_mask


