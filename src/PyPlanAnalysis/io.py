"""
PyPlanAnalysis.io
==================

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
from scipy.ndimage import distance_transform_edt
from collections import defaultdict

from skimage.draw import polygon as sk_polygon
import SimpleITK as sitk

from dataclasses import dataclass, field
from typing import Optional

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



def _collect_candidates(folder: Path, rad_type: str):
    
    
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
        radiationType: Optional[str] #"PROTON"
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
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
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
            
            try:
                radiationType= getattr(ds, "RadiationType", "").upper()
            except:
                print("RadiationTag not defined in {f}, filter disabled")
                radiationType = None
                
            rtplans.append(RTPlanCandidate(
                path=f,
                sop_class=getattr(ds, "SOPClassUID", None),
                sop_uid=getattr(ds, "SOPInstanceUID", None),                
                radiationType=radiationType,
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

def _strict_chain(ct_series, rtstructs, rtplans, rad_type):
    """Resolve a valid RTPLAN/RTSTRUCT/CT-series chain for a given folder.

    Candidate RTPLANs are tried in the following priority order:
      1. Plans whose RadiationType matches the requested `rad_type`.
      2. Any other available plans (regardless of RadiationType), used
         as a fallback if no matching plan yields a fully valid chain.
      3. If no plan (matching or otherwise) resolves to a valid chain,
         or if no RTPLAN exists at all, fall back to an RTSTRUCT that is
         directly associated with one of the given CT series.

    For each candidate plan, the function attempts to resolve:
      RTPLAN -> RTSTRUCT (via plan.ref_struct) -> CT series (via
      struct.ref_series_uids), verifying that the referenced CT series
      is actually present in `ct_series`. The first candidate that
      resolves a complete, valid chain is returned.

    Args:
        ct_series: Iterable of CT series UIDs present in the folder, if any.
        rtstructs: List of RTSTRUCT objects, each exposing sop_class,
            sop_uid, and ref_series_uids.
        rtplans: List of RTPLAN objects, each exposing RadiationType,
            and ref_struct (a (sop_class, sop_uid) tuple).
        rad_type: The radiation type requested by the user (e.g.
            "PROTON"), used to prioritize matching plans.

    Returns:
        A tuple (plan, struct, series_uid, fully_verified):
            - plan: The resolved RTPLAN, or None if resolved via the
              RTSTRUCT-only fallback.
            - struct: The resolved RTSTRUCT.
            - series_uid: The CT series UID referenced by the struct.
            - fully_verified: True if a plan was found and its chain
              verified (regardless of whether its RadiationType matches
              `rad_type`); False if resolved through the struct-only
              fallback.
        If no valid chain can be resolved at all, returns
        (None, None, None, False).
    """
    matching = [p for p in rtplans if p.radiationType == rad_type]
    others = [p for p in rtplans if p.radiationType != rad_type]
    plan_candidates = matching + others + [None]

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

def find_dicom_files(folder: Path, rad_type: str) -> dict:
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

    ct_series, rtstructs, rtplans, rtdoses = _collect_candidates(folder, rad_type)

    plan, struct, series_uid, verified = _strict_chain(ct_series, rtstructs, rtplans, rad_type)

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

    physical_doses = [d for d in rtdoses if (d.dose_kind == "PHYSICAL" and d.dose_SumType == "PLAN" )]
    effective_doses = [d for d in rtdoses if (d.dose_kind == "EFFECTIVE" and d.dose_SumType == "PLAN" )]
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
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
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
    series_uid = first_ds.SeriesInstanceUID
    
    # Use SimpleITK series reader for correct pixel data ordering
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(ct_folder),series_uid)
    if not len(dicom_names) == len(slices):
        warnings.warn("\nCheck CT reading, not all slices are correctly read\n")
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

def _get_reference_ct_volume(sitk_ct):
    """Extract a single 3D CT volume from a 4D (dual-energy / multi-channel)
    stack, to be used as a resampling reference.

    If sitk_ct is already 3D, it is returned unchanged. If it is 4D
    (e.g. dual-energy CT with two stacked energy volumes), the first
    volume along the 4th dimension is extracted and returned as a
    proper 3D image (spacing/direction/origin trimmed accordingly).
    """
    if sitk_ct.GetDimension() == 3:
        return sitk_ct

    size = list(sitk_ct.GetSize())
    # Extract index 0 along the 4th dimension -> collapse it to size 0
    extract_size = size[:3] + [0]
    extract_index = [0, 0, 0, 0]

    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(extract_size)
    extractor.SetIndex(extract_index)
    return extractor.Execute(sitk_ct)
 
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
    sitk_ct = _get_reference_ct_volume(sitk_ct)
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
    ds    = pydicom.dcmread(str(path), force=True)
    scale = float(ds.DoseGridScaling)
    if not ds.DoseUnits == "GY":
        warnings.warn(f"Dose Units are not correct for: '{path}'.")
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
    dz = float(ds.SliceThickness) if ds.SliceThickness is not None else pix_sp[0]
  
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
    Rasterise RT Struct contours as BINARY MASKS for `struct_name` onto an arbitrary grid.
 
    This is the core function used for both dose-grid and CT-grid masking.
    Contour z-values are matched to the nearest z in z_positions.
 
    Parameters
    ----------
    struct_name  : str
    rtstruct_ds  : pydicom Dataset
    origin       : [x0, y0, z0]  mm — physical coordinate of voxel (0,0,0) corner, then converted to voxel center
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

def get_roi_center_of_mass(struct_name: str, rtstruct_ds) -> np.ndarray:
    """
    Compute the 3-D center of mass (mm, in the RT Struct's patient
    coordinate system) of a structure directly from its contour
    polygons — no dose/CT grid required.
 
    Each contour (one per slice) contributes its own 2-D polygon
    centroid, weighted by that contour's area, so slices with more
    cross-sectional area count more towards the overall COM volume-weighted approximation without needing a full
    3-D mask.
 
    Parameters
    ----------
    struct_name : str
        Structure name as it appears in the RT Struct.
    rtstruct_ds : pydicom Dataset
 
    Returns
    -------
    np.ndarray, shape (3,), or None
        ``[x, y, z]`` center of mass in mm. Returns ``None`` if no
        usable contour data is available for this structure.
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
 
    if not contours:
        warnings.warn(f"No contour data for '{struct_name}' — cannot compute center of mass.")
        return None
 
    centroids = []
    weights   = []
    for pts in contours:
        xy = pts[:, :2]
        if len(xy) < 3:
            continue
        area = contour_area_signed(xy)
        if area == 0:
            continue
        x, y   = xy[:, 0], xy[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross  = x * y1 - x1 * y
        cx = np.sum((x + x1) * cross) / (6 * area)
        cy = np.sum((y + y1) * cross) / (6 * area)
        z  = float(pts[0, 2])
        centroids.append([cx, cy, z])
        weights.append(abs(area))
 
    if not centroids:
        warnings.warn(f"Could not compute a valid centroid for '{struct_name}'.")
        return None
 
    centroids = np.array(centroids)
    weights   = np.array(weights)
    return np.average(centroids, axis=0, weights=weights)

def contour_area_signed(xy):
    """
    Signed polygon area via the shoelace formula.

    Parameters
    ----------
    xy : np.ndarray, shape (N, 2)
        Polygon vertices [x, y] in mm.

    Returns
    -------
    float
        Signed area; positive for counter-clockwise vertex order,
        negative for clockwise. Used to detect contour holes.
    """
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def smooth_contour(poly_xy, n_pts=300):
    """
    Resample a contour polygon onto ``n_pts`` evenly-spaced points using
    a periodic B-spline fit. Not used by default in the fractional-mask
    pipeline (see inline note in ``get_fractional_mask_on_grid``); kept
    available for callers who want smoothed contours.

    Parameters
    ----------
    poly_xy : np.ndarray, shape (N, 2)
        Polygon vertices [x, y] in mm.
    n_pts : int, default 300
        Number of points in the resampled output.

    Returns
    -------
    np.ndarray, shape (n_pts, 2)
        Smoothed polygon vertices. Falls back to the original polygon,
        unchanged, if the spline fit fails.
    """
    try:
        x, y = poly_xy[:, 0], poly_xy[:, 1]
        tck, _ = splprep([x, y], s=0, per=True)
        x_s, y_s = splev(np.linspace(0, 1, n_pts), tck)
        return np.column_stack([x_s, y_s]) 
    except Exception as e:
        warnings.warn(f"Spline failed: {e}")
        return poly_xy

def rasterize_supersampled(smooth_xy, x0, y0, dx, dy, ny, nx, N):
    """
    Rasterise one contour polygon onto a grid at N times finer resolution,
    then average back down — giving each output voxel a fractional
    [0, 1] membership value instead of a hard binary in/out.

    Parameters
    ----------
    smooth_xy : np.ndarray, shape (N, 2)
        Polygon vertices [x, y] in mm.
    x0, y0 : float
        Grid origin (mm) in x and y.
    dx, dy : float
        Grid voxel spacing (mm) in x and y.
    ny, nx : int
        Output grid shape.
    N : int
        Supersampling factor per side (N² sub-samples per voxel).

    Returns
    -------
    np.ndarray, shape (ny, nx), float32
        Fractional coverage of each voxel by the polygon, in [0, 1].
    """
    xi = (smooth_xy[:, 0] - x0 + dx/2) / dx * N # account for voxel center! dx/2 shift
    yi = (smooth_xy[:, 1] - y0 + dy/2) / dy * N

    rr, cc = sk_polygon(yi, xi, shape=(ny * N, nx * N))
    super_mask = np.zeros((ny * N, nx * N), dtype=np.float32)
    super_mask[rr, cc] = 1.0
    return super_mask.reshape(ny, N, nx, N).mean(axis=(1, 3))


def compute_roi_volume_comparison(frac_mask, dx, dy, dz):
    """
    Print and return the total volume of a fractional mask, plus a
    per-slice breakdown — useful for sanity-checking
    ``get_fractional_mask_on_grid`` output against a TPS-reported volume.

    Parameters
    ----------
    frac_mask : np.ndarray, shape (nz, ny, nx)
        Fractional voxel membership mask, values in [0, 1].
    dx, dy, dz : float
        Voxel spacing (mm) in x, y, z.

    Returns
    -------
    float
        Total volume in cc (sum of fractional weights x voxel volume).
    """
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
    """
    Volume of a fractional mask via the prismatoid (Simpson's-rule-like)
    formula between consecutive slices, instead of a flat sum-of-slices
    approximation. Slightly more accurate for structures with rapidly
    changing cross-sectional area between slices.

    Parameters
    ----------
    frac_mask : np.ndarray, shape (nz, ny, nx)
        Fractional voxel membership mask, values in [0, 1].
    dx, dy, dz : float
        Voxel spacing (mm) in x, y, z.

    Returns
    -------
    float
        Total volume in cc.
    """
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

def _classify_holes_by_nesting(slice_polys):
    """
    Determine hole/island status for a set of coplanar polygon loops
    using point-in-polygon NESTING DEPTH (even-odd rule), instead of only
    testing each polygon against larger-area ones in sequence.


    Parameters
    ----------
    slice_polys : list of np.ndarray, shape (M, 2)
        All polygon loops belonging to one contour slice (one true z).

    Returns
    -------
    list of bool, same length as slice_polys
        True where the polygon at that index is a hole (odd nesting depth).
    """
    n = len(slice_polys)
    depth = [0] * n
    for i in range(n):
        poly_i    = slice_polys[i]
        test_pts  = poly_i[::max(1, len(poly_i) // 5)]
        for j in range(n):
            if i == j:
                continue
            path_j = MplPath(slice_polys[j], closed=True)
            if path_j.contains_points(test_pts).mean() > 0.5:
                depth[i] += 1
    return [d % 2 == 1 for d in depth]


def _slice_net_area_mm2(slice_polys) -> float:
    """
    Net in-plane area (mm²) of one contour slice, subtracting hole loops
    (nested contours representing a cavity) from their parent polygon's
    area via ``_classify_holes_by_nesting`` + ``contour_area_signed``.

    Parameters
    ----------
    slice_polys : list of np.ndarray, shape (M, 2)
        All polygon loops belonging to one contour slice (one true z).

    Returns
    -------
    float — net area, mm², always >= 0 for a well-formed contour set.
    """
    is_hole = _classify_holes_by_nesting(slice_polys)
    net = 0.0
    for poly, hole in zip(slice_polys, is_hole):
        a = abs(contour_area_signed(poly))
        net += -a if hole else a
    return net


def compute_roi_volume_from_contours(struct_name: str, rtstruct_ds) -> float:
    """
    Compute the ROI's total volume (cc) directly from its contour polygon
    geometry, at the NATIVE contour z-spacing — independent of any
    dose/LET/analysis grid resolution.

    TECHNICAL JUSTIFICATION: ``get_fractional_mask_on_grid`` (and hence
    ``compute_dvh_metrics``/``compute_let_metrics`` in metrics.py, which
    normalise Vx%/Dx%/Lx% by the sum of that mask's fractional weights)
    reports volume on whatever analysis grid the mask was rasterised on
    (e.g. a resampled dose/LET grid, ``New_grid`` — commonly coarser than
    the CT the structure was actually contoured on). For structures that
    only span a handful of voxels across that grid (small serial OARs:
    optic nerves, chiasm, lenses, lacrimal glands), that grid-resolution
    dependence is the dominant source of volume error, which then
    propagates into every weighted DVH/LVH percentile metric for that
    structure (they all share the same total-weight normalisation).

    This function instead reconstructs volume the way most TPS systems
    report "ROI volume" — directly from the contour polygons themselves:
    net in-plane area per true contour slice (``_slice_net_area_mm2``,
    correctly excluding holes), trapezoidal integration between
    consecutive TRUE contour z-planes at their native (possibly
    irregular) spacing, plus a half-native-spacing end-cap extension at
    each pole — the same physical convention ``get_fractional_mask_on_grid``
    uses for its own z-extent padding, so the two stay consistent with
    each other even though this number is otherwise grid-independent.

    Use this as the authoritative reported ``volume_cc`` for a structure.
    Note that it is NOT automatically a safe drop-in replacement for the
    weighted-percentile normalisation used inside ``compute_dvh_metrics``/
    ``compute_let_metrics`` (Vx%, Dx%, Lx%, ...): those numerators are
    still computed from the analysis-grid mask, so swapping only the
    denominator to a different, higher-resolution source is only
    self-consistent once the grid-captured voxel population is itself a
    low-bias subsample of the true structure (see the exact z-integration
    in ``get_fractional_mask_on_grid``, which is what makes that
    assumption reasonable in the first place).

    Parameters
    ----------
    struct_name : str
    rtstruct_ds : pydicom Dataset

    Returns
    -------
    float — ROI volume in cc.
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

    if not contours:
        warnings.warn(f"No contour data for '{struct_name}'.")
        return 0.0

    slice_polys_by_z = defaultdict(list)
    for pts in contours:
        z_val = round(float(pts[0, 2]), 3)
        slice_polys_by_z[z_val].append(pts[:, :2])

    sorted_z = np.array(sorted(slice_polys_by_z.keys()))
    areas = np.array([_slice_net_area_mm2(slice_polys_by_z[z]) for z in sorted_z])

    if len(sorted_z) == 1:
        # A single contour slice carries no native z-spacing to infer a
        # slice thickness from; that has to come from the CT/RTSTRUCT
        # geometry, which this function deliberately doesn't depend on.
        warnings.warn(
            f"'{struct_name}': only one contour slice — cannot infer a "
            "native slice spacing from contour z-positions alone; "
            "returning 0.0. Use get_fractional_mask_on_grid() with a "
            "known dz instead for single-slice ROIs."
        )
        return 0.0

    # trapezoidal integration between consecutive TRUE contour planes, at
    # their TRUE (possibly irregular) native spacing — not any grid's dz.
    dz_native = np.diff(sorted_z)
    vol_mm3 = float(np.sum(dz_native * (areas[:-1] + areas[1:]) / 2.0))

    # half-native-spacing end-cap extension at each pole, held at that end
    # slice's own area — matches get_fractional_mask_on_grid's convention
    pad_lo = dz_native[0] / 2.0
    pad_hi = dz_native[-1] / 2.0
    vol_mm3 += pad_lo * areas[0] + pad_hi * areas[-1]

    return vol_mm3 / 1000.0

def rasterize_slice_coverage(slice_polys, x0, y0, dx, dy, ny, nx, N):
    """
    Rasterise ALL polygon loops belonging to ONE true contour z-position
    (i.e. one DICOM ContourData slice) into a single in-plane fractional
    coverage map, islands added and holes subtracted per
    ``_classify_holes_by_nesting``.

    Factored out of ``get_fractional_mask_on_grid`` so that in-plane
    coverage can be computed once per ORIGINAL contour z (see that
    function's docstring for why this must be decoupled from the output
    grid's z-slices).

    Parameters
    ----------
    slice_polys : list of np.ndarray, shape (M, 2)
        Polygon loops (islands + holes) for one contour slice, in mm.
    x0, y0, dx, dy, ny, nx, N : see ``rasterize_supersampled``.

    Returns
    -------
    np.ndarray, shape (ny, nx), float32
        Fractional in-plane coverage, clipped to [0, 1].
    """
    coverage = np.zeros((ny, nx), dtype=np.float32)
    valid_polys = [p for p in slice_polys if len(p) >= 3]
    if not valid_polys:
        return coverage

    is_hole = _classify_holes_by_nesting(valid_polys)

    for poly_xy, hole in zip(valid_polys, is_hole):
        fraction = rasterize_supersampled(poly_xy, x0, y0, dx, dy, ny, nx, N)
        coverage = coverage - fraction if hole else coverage + fraction

    return np.clip(coverage, 0.0, 1.0)


def _directional_pixel_size(sdf: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Per-pixel local antialiasing width for a possibly anisotropic
    (dx != dy) in-plane grid, used in place of a single scalar
    ``(dx + dy) / 2`` estimate.

    TECHNICAL JUSTIFICATION: an axis-aligned rectangular pixel of size
    dx (x) by dy (y), sliced by a straight boundary crossing it at angle
    theta to the x-axis, spans a length dx*|cos theta| + dy*|sin theta|
    along the boundary-normal direction — the standard "screen-space
    derivative" / fwidth footprint used for antialiasing distance-field
    edges (e.g. Green, "Improved Alpha-Tested Magnenta Field Textures",
    SIGGRAPH 2007). A flat (dx + dy) / 2 average is only correct for the
    two axis-aligned cases (theta = 0 or 90 deg) and, for anisotropic
    grids, silently uses the WRONG antialiasing width everywhere the
    boundary isn't axis-aligned — e.g. with dx=0.5 mm, dy=3 mm, a
    boundary running mostly along y (normal mostly along x, theta~0) has
    a true footprint of ~0.5 mm, not the averaged 1.75 mm, so the old
    code was over-smoothing/over-blurring the reconstructed edge in the
    fine (x) direction and under-resolving it relative to the coarse (y)
    direction.

    Because a proper Euclidean SDF has unit-magnitude gradient almost
    everywhere (|grad sdf| = 1), the boundary-normal direction at each
    pixel can be read directly off the SDF's own gradient, without any
    extra geometry — this is exactly what makes an SDF representation
    convenient here, on top of its use for shape-based z-interpolation.

    Parameters
    ----------
    sdf      : np.ndarray, shape (ny, nx) — signed distance field, mm.
        Only its local gradient DIRECTION is used, not its magnitude.
    dx, dy   : float — in-plane voxel spacing, mm.

    Returns
    -------
    np.ndarray, float32, shape (ny, nx)
        Local antialiasing width in mm, one value per pixel. Falls back
        to the isotropic ``(dx + dy) / 2`` estimate wherever the
        gradient is degenerate (e.g. deep in a uniform interior/exterior
        region, or a perfectly flat plateau) — irrelevant there anyway,
        since the antialiasing ramp saturates to 0 or 1 far from the
        boundary regardless of its width.
    """
    gy, gx = np.gradient(sdf, dy, dx)
    norm = np.sqrt(gx * gx + gy * gy)
    fallback = (dx + dy) / 2.0

    safe_norm = np.where(norm > 1e-6, norm, 1.0)
    nx = np.abs(gx) / safe_norm
    ny = np.abs(gy) / safe_norm
    directional = dx * nx + dy * ny

    return np.where(norm > 1e-6, directional, fallback).astype(np.float32)


def _coverage_to_sdf(coverage: np.ndarray, dx: float, dy: float,
                      subpixel_refine: bool = True) -> np.ndarray:
    """
    Convert an in-plane fractional coverage raster into a 2-D signed
    distance field (SDF), in mm, positive INSIDE the structure and
    negative OUTSIDE, magnitude = distance to the nearest boundary.

    This is the per-slice building block for shape-based z-interpolation
    (see ``get_fractional_mask_on_grid``): rather than linearly blending
    opacity/coverage values between two contour planes — which is known
    to erode or "melt" the structure wherever its cross-section changes
    shape, size, or position between planes (a classic artifact of alpha
    cross-dissolving two masks) — we interpolate the *geometry* of the
    boundary itself, which is what an SDF encodes.

    Parameters
    ----------
    coverage : np.ndarray, shape (ny, nx)
        Fractional in-plane coverage map in [0, 1], as produced by
        ``rasterize_slice_coverage``.
    dx, dy   : float — in-plane voxel spacing, mm. Passed as the EDT
        ``sampling`` (already correctly anisotropic-aware there) and,
        when ``subpixel_refine`` is set, used via
        ``_directional_pixel_size`` to scale the boundary refinement per
        pixel according to local boundary orientation rather than a
        single isotropic average — see that function's docstring.
    subpixel_refine : bool
        If True, overwrite the boundary-adjacent band of the distance
        transform (|sdf| <= local pixel footprint) with a direct
        estimate derived from the antialiased coverage fraction itself,
        (coverage - 0.5) * local_pixel_size. The plain Euclidean
        distance transform only "sees" the binarised (coverage >= 0.5)
        raster and is therefore blind to sub-pixel boundary position;
        this refinement folds that information back in near the
        boundary where it matters most.

    Returns
    -------
    np.ndarray, float32, shape (ny, nx)
        Signed distance field in mm. Fully-outside or fully-inside
        rasters (no boundary present) return a uniform large-magnitude
        constant field so they behave correctly under interpolation
        with a neighbouring slice that does have a boundary.
    """
    inside = coverage >= 0.5

    if not inside.any():
        return np.full(coverage.shape, -1.0e3, dtype=np.float32)
    if inside.all():
        return np.full(coverage.shape, 1.0e3, dtype=np.float32)

    dist_in  = distance_transform_edt(inside,  sampling=(dy, dx)) #compute the eclidean distance to the outside (border)
    dist_out = distance_transform_edt(~inside, sampling=(dy, dx))
    sdf = (dist_in - dist_out).astype(np.float32)

    if subpixel_refine:
        pixel_size_map = _directional_pixel_size(sdf, dx, dy)
        boundary_band = np.abs(sdf) <= pixel_size_map
        refined = (coverage.astype(np.float32) - 0.5) * pixel_size_map
        sdf = np.where(boundary_band, refined, sdf)

    return sdf


def _sdf_to_coverage(sdf: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Reconstruct an antialiased fractional coverage raster from a signed
    distance field, the inverse operation of ``_coverage_to_sdf``.

    A locally boundary-orientation-aware ramp (see
    ``_directional_pixel_size``) is used to convert distance-to-boundary
    into partial coverage — sdf = 0 at the boundary -> 0.5; sdf >=
    +half the local pixel footprint, fully inside -> 1.0; sdf <= -half
    the local pixel footprint, fully outside -> 0.0 — the standard
    antialiasing reconstruction used for SDF-represented shapes,
    generalised from a single isotropic pixel width to a per-pixel
    directional one so anisotropic grids (dx != dy) get the correct
    ramp width regardless of local boundary orientation.

    Parameters
    ----------
    sdf      : np.ndarray, shape (ny, nx) — signed distance field, mm,
        positive inside (see ``_coverage_to_sdf``).
    dx, dy   : float — in-plane voxel spacing, mm.

    Returns
    -------
    np.ndarray, float32, shape (ny, nx), values in [0, 1]
    """
    pixel_size_map = _directional_pixel_size(sdf, dx, dy)
    coverage = 0.5 + sdf / pixel_size_map
    return np.clip(coverage, 0.0, 1.0).astype(np.float32)



def _sdf_at_z(z: float, sorted_z: np.ndarray, sdf_by_z: list) -> np.ndarray:
    """
    Evaluate the piecewise-linear-in-z SDF field at an arbitrary depth z
    (mm), by linearly interpolating between the two bracketing true
    contour planes — the same rule ``get_fractional_mask_on_grid`` uses
    per z-subsample, factored out so it can be called at exact segment
    breakpoints instead of a fixed sub-sampling grid (see
    ``_slab_avg_sdf_exact``).

    Beyond the first/last contour plane the field is held constant at
    that end slice's SDF (matching the half-slice end-cap convention
    documented in ``get_fractional_mask_on_grid``) — trivially a
    zero-slope linear segment, so it fits the same piecewise-linear model.

    Parameters
    ----------
    z         : float — depth to evaluate, mm.
    sorted_z  : np.ndarray — true contour z-positions, ascending, mm.
    sdf_by_z  : list of np.ndarray, shape (ny, nx) — per-slice SDFs,
        same order as ``sorted_z``.

    Returns
    -------
    np.ndarray, shape (ny, nx)
    """
    if len(sorted_z) == 1:
        return sdf_by_z[0]
    z_min, z_max = sorted_z[0], sorted_z[-1]
    if z <= z_min:
        return sdf_by_z[0]
    if z >= z_max:
        return sdf_by_z[-1]
    j = int(np.searchsorted(sorted_z, z, side="right") - 1)
    j = int(np.clip(j, 0, len(sorted_z) - 2))
    z0_, z1_ = sorted_z[j], sorted_z[j + 1]
    t = 0.0 if z1_ == z0_ else (z - z0_) / (z1_ - z0_)
    return (1.0 - t) * sdf_by_z[j] + t * sdf_by_z[j + 1]


def _slab_avg_sdf_exact(z_lo: float, z_hi: float, sorted_z: np.ndarray,
                         sdf_by_z: list, z_extent_lo: float, z_extent_hi: float,
                         ny: int, nx: int):
    """
    Exact (zero quadrature error) average of the piecewise-linear-in-z
    SDF field over an output voxel's z-slab ``[z_lo, z_hi]``, replacing
    finite-``Nz`` sub-sampling.

    TECHNICAL JUSTIFICATION: by construction the SDF field is piecewise
    LINEAR in z — linear between each pair of adjacent true contour
    planes, constant beyond the first/last (see ``_sdf_at_z``). The exact
    integral average of ANY linear segment over an interval is available
    in closed form from just its two endpoint values (trapezoid rule,
    exact for linear functions — no discretisation error regardless of
    how coarse the output grid's dz is relative to the native contour
    spacing, and regardless of how many true contour planes fall inside
    one output voxel's z-extent). This replaces sampling ``Nz`` points on
    a uniform sub-grid — which is only exact when the whole slab happens
    to fall within a single linear segment, and otherwise carries
    residual quadrature error from the slope discontinuity at each
    interior contour plane — with a handful of *exact* evaluations at
    the slab's clipped bounds plus every interior contour z.

    The slab is also clipped to the structure's true (padded) z-extent
    ``[z_extent_lo, z_extent_hi]``: outside that extent there is no
    structure at all (coverage is exactly 0, not merely "unsampled"), so
    a voxel whose z-range only partially overlaps the structure — e.g.
    the ROI's very first/last output slice — has its covered-region
    coverage scaled down by the overlap fraction, rather than reporting
    the covered portion's coverage as if it applied to the whole voxel.

    Parameters
    ----------
    z_lo, z_hi : float — output voxel's full z-extent, mm.
    sorted_z   : np.ndarray — true contour z-positions, ascending, mm.
    sdf_by_z   : list of np.ndarray, shape (ny, nx) — per-slice SDFs.
    z_extent_lo, z_extent_hi : float — structure's padded true z-extent, mm.
    ny, nx     : int — in-plane grid shape.

    Returns
    -------
    (avg_sdf_covered, overlap_frac) : (np.ndarray shape (ny, nx), float)
        or None if ``[z_lo, z_hi]`` does not overlap the structure's
        extent at all.
    """
    lo = max(z_lo, z_extent_lo)
    hi = min(z_hi, z_extent_hi)
    total_len = z_hi - z_lo

    if hi <= lo or total_len <= 0:
        return None

    interior = sorted_z[(sorted_z > lo) & (sorted_z < hi)]
    breakpoints = np.concatenate(([lo], interior, [hi]))

    acc = np.zeros((ny, nx), dtype=np.float32)
    for i in range(len(breakpoints) - 1):
        a, b = float(breakpoints[i]), float(breakpoints[i + 1])
        seg_len = b - a
        if seg_len <= 0:
            continue
        sdf_a = _sdf_at_z(a, sorted_z, sdf_by_z)
        sdf_b = _sdf_at_z(b, sorted_z, sdf_by_z)
        acc += seg_len * 0.5 * (sdf_a + sdf_b)   # exact trapezoid: average of a linear segment

    covered_len = hi - lo
    avg_sdf_covered = acc / covered_len
    overlap_frac = covered_len / total_len
    return avg_sdf_covered, overlap_frac

def _adaptive_supersample(contours, dx, dy, base_N, max_N=12, target_subsamples=16):
    """
    Scale the in-plane supersampling factor up for small structures.

    TECHNICAL JUSTIFICATION: a fixed N gives a fixed number of sub-points
    (N²) per OUTPUT VOXEL, regardless of how many voxels the structure
    itself spans. A 40-voxel-wide PTV and a 2-voxel-wide lens both get
    the same N=4 (~6% accuracy, per the original docstring) even though
    the lens's boundary curvature is resolved by only ~8 voxels total —
    the actual accuracy-limiting factor is sub-samples PER STRUCTURE
    WIDTH, not sub-samples per voxel. This scales N so the number of
    sub-samples spanning the structure's narrowest in-plane bounding-box
    dimension stays roughly constant (~target_subsamples) whether the
    structure is a PTV or a 3-voxel-wide serial OAR, bounded by max_N to
    cap the compute cost.

    Parameters
    ----------
    contours : list of np.ndarray, shape (M, 3)
        All raw contour point arrays for the ROI (x, y, z in mm).
    dx, dy   : float — in-plane voxel spacing, mm.
    base_N   : int — minimum/default supersampling factor (large structures).
    max_N    : int — hard cap on the supersampling factor.
    target_subsamples : int — desired sub-sample count across the
        structure's narrowest bounding-box dimension.

    Returns
    -------
    int
        Effective supersampling factor N to use for this structure.
    """
    all_xy  = np.concatenate([c[:, :2] for c in contours], axis=0)
    bbox_w  = all_xy[:, 0].max() - all_xy[:, 0].min()
    bbox_h  = all_xy[:, 1].max() - all_xy[:, 1].min()
    bbox_vox = min(bbox_w / dx, bbox_h / dy)
    if bbox_vox <= 0:
        return base_N
    n_req = int(np.ceil(target_subsamples / bbox_vox))
    return int(np.clip(n_req, base_N, max_N))



def get_fractional_mask_on_grid(struct_name: str,
                                rtstruct_ds,
                                origin:      list,
                                spacing:     list,
                                shape:       tuple,
                                z_positions: np.ndarray,
                                supersample: int = 4,
                                supersample_z: int = None,
                                max_supersample: int = 12) -> np.ndarray:
    """
    Compute a fractional voxel membership mask on an arbitrary grid.

    Each voxel receives a value in [0,1] — the fraction of its physical
    VOLUME (in-plane AND through-slice) that lies inside the RT Struct
    contour. In-plane coverage is estimated by supersampling
    (adaptive N² sub-points per voxel, see ``_adaptive_supersample``).
    Through-slice (z) coverage is estimated by treating the true contour
    z-positions as control planes and interpolating the STRUCTURE'S
    BOUNDARY GEOMETRY — not its raw in-plane coverage/opacity — between
    the bracketing contour slices across each output voxel's z-extent.
    This is done via shape-based interpolation: each contour slice's
    fractional coverage raster is first converted to a 2-D signed
    distance field (SDF, see ``_coverage_to_sdf``); the per-voxel z-slab
    is then integrated EXACTLY in SDF space (``_slab_avg_sdf_exact`` —
    see that function's docstring, and NOTE below), not by sub-sampling;
    and the resulting averaged SDF is converted back to a fractional
    coverage raster only once, at the end (``_sdf_to_coverage``).

    NOTE on the z-integration being exact rather than sub-sampled: the
    SDF field is piecewise LINEAR in z by construction (linear between
    each pair of adjacent true contour planes, constant beyond the
    first/last). The integral average of a piecewise-linear function
    over any interval is available in closed form from its endpoint
    values at each linear segment (trapezoid rule, exact for linear
    functions) — so no ``Nz`` sub-sampling parameter is needed, and there
    is zero z-quadrature error regardless of how coarse the output
    grid's dz is relative to the native contour spacing, or how many true
    contour planes fall inside one output voxel's z-slab. (A generic
    fixed-``Nz`` sub-sampling grid — the previous approach — is only
    exact when a slab happens to fall entirely within one linear segment;
    it carries residual error whenever a slab straddles an interior
    contour plane, which is exactly the case where the output grid is
    coarser than the contour spacing.) The exact integration also
    correctly handles voxels whose z-extent only PARTIALLY overlaps the
    structure's true (padded) extent — e.g. the ROI's first/last output
    slice — by scaling the covered region's coverage down by the actual
    overlap fraction, rather than reporting the covered portion's
    coverage as if it applied to the voxel's full z-extent.

    TECHNICAL JUSTIFICATION for the z-interpolation (this is the main
    change from the previous version): contours were previously snapped
    to the single nearest output z-index, so an output voxel's z-extent
    was always either "fully in" or "fully out" of the structure — no
    different, in the z direction, from a hard binary mask. This
    reproduces the same staircase/quantization error along the
    cranial-caudal axis that fractional in-plane weighting was written to
    avoid in-plane, and it gets worse whenever the analysis grid's z
    spacing does not match the original contour spacing (e.g. resampling
    dose/LET/mask onto a coarser custom grid).

    A direct linear blend of the raw coverage rasters (alpha
    cross-dissolving) was tried first, but that interpolates OPACITY, not
    GEOMETRY: wherever the cross-section shifts, rotates, or changes size
    between two contour planes (tapering structures, branching anatomy,
    slightly mis-registered slices), an opacity blend produces a faint,
    eroded double-exposure of both shapes rather than a smoothly moving
    boundary, understating volume in exactly the transition regions where
    accuracy matters most. Interpolating signed distance fields instead
    — the classical "shape-based interpolation" approach (Raya & Udupa,
    1990) — averages DISTANCE TO THE BOUNDARY rather than opacity, so the
    reconstructed boundary at any intermediate depth is the correct
    (piecewise-linear-in-distance) locus of points equidistant between
    the two true contours; this is the discretised equivalent of lofting
    a triangulated surface between them, without an opacity blend's
    tendency to thin the structure out.

    Outside the structure's own z-extent, the boundary geometry (i.e. the
    SDF, hence coverage) is held constant at the nearest edge slice's
    value (rather than tapering to zero), because RTSTRUCT contours
    conventionally represent the slice-thickness worth of structure
    centred on each contour, so the ROI is assumed to occupy a
    half-slice-thickness beyond the first/last contour plane, not to end
    exactly on it.

    Parameters
    ----------
    struct_name     : str
    rtstruct_ds     : pydicom Dataset
    origin          : [x0, y0, z0]  mm
    spacing         : [dx, dy, dz]  mm
    shape           : (nz, ny, nx)
    z_positions     : 1-D array length nz
    supersample     : baseline in-plane N (default 4). Scaled up
                      per-structure by ``_adaptive_supersample``.
    supersample_z   : DEPRECATED / ignored. Z-integration is now exact
                      (see docstring NOTE above) and needs no sub-sampling
                      count. Kept only so existing call sites that pass
                      this argument don't break; a warning is issued if
                      it's explicitly set to a non-None value.
    max_supersample : hard cap for the adaptive in-plane N.

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

    if supersample_z is not None:
        warnings.warn(
            "'supersample_z' is deprecated and ignored: z-integration is "
            "now exact (closed-form, piecewise-linear-in-z SDF averaging) "
            "and no longer needs a sub-sampling count.",
            DeprecationWarning,
        )
    N  = _adaptive_supersample(contours, dx, dy, base_N=supersample, max_N=max_supersample)

    # ---- group contours by their TRUE z (mm), NOT by nearest grid index ----
    # (rounded to 3 decimals only to merge floating-point duplicates of the
    # same physical slice; this is not a grid-snapping step)
    slice_polys_by_z = defaultdict(list)
    for pts in contours:
        z_val = round(float(pts[0, 2]), 3)
        slice_polys_by_z[z_val].append(pts[:, :2])

    sorted_z = np.array(sorted(slice_polys_by_z.keys()))

    if len(sorted_z) > 1:
        gaps = np.diff(sorted_z)
        if gaps.max() > 2.0 * np.median(gaps):
            warnings.warn(
                f"'{struct_name}': contour spacing is irregular (max gap "
                f"{gaps.max():.2f}mm vs median {np.median(gaps):.2f}mm) — "
                "possible missing slice(s); linear z-interpolation across a "
                "large gap is a weaker approximation than across evenly "
                "spaced contours."
            )

    # rasterise in-plane coverage ONCE per true contour z (not per output slab)
    coverage_by_z = [
        rasterize_slice_coverage(slice_polys_by_z[z], x0, y0, dx, dy, ny, nx, N)
        for z in sorted_z
    ]
    # ...and convert each slice's coverage raster to a signed distance field
    # ONCE as well — z-interpolation below blends SDFs (geometry), not the
    # raw coverage rasters (opacity); see docstring for why.
    sdf_by_z = [_coverage_to_sdf(cov, dx, dy) for cov in coverage_by_z]
    z_min, z_max = float(sorted_z[0]), float(sorted_z[-1])

    # Physically bounded end-cap extension: each contour conventionally
    # represents the half-slice-thickness of structure centred on it, using
    # the LOCAL native contour spacing at that end — NOT the output grid's
    # dz. Using the output dz here (or clipping every sample unconditionally
    # into [z_min, z_max]) over-extends the ROI whenever the analysis grid
    # is coarser than the original contour spacing, e.g. resampling a plan
    # contoured on a 1-2mm CT onto a 3mm dose/LET grid: an end-cap output
    # voxel would incorrectly get 100% coverage across its FULL (coarse)
    # thickness instead of just the true ~0.5-1mm the structure actually
    # extends past the last contour, inflating volume at both structure poles.
    pad_lo = (sorted_z[1] - sorted_z[0]) / 2.0 if len(sorted_z) > 1 else dz / 2.0
    pad_hi = (sorted_z[-1] - sorted_z[-2]) / 2.0 if len(sorted_z) > 1 else dz / 2.0
    z_extent_lo, z_extent_hi = z_min - pad_lo, z_max + pad_hi

    # ---- resample through z with linear interpolation between contour planes ----
    for z_idx in range(nz):
        z_center = z_positions[z_idx]
        z_lo = z_center - dz / 2.0
        z_hi = z_center + dz / 2.0

        # skip output slabs entirely outside the (padded) structure extent
        if z_hi < z_extent_lo or z_lo > z_extent_hi:
            continue

        # Exact (closed-form) z-integration of the piecewise-linear-in-z
        # SDF field over this voxel's slab — see ``_slab_avg_sdf_exact``
        # and the docstring NOTE above for why this replaces sub-sampling.
        # Also handles voxels whose z-extent only partially overlaps the
        # structure's true (padded) extent, via ``overlap_frac``.
        result = _slab_avg_sdf_exact(z_lo, z_hi, sorted_z, sdf_by_z,
                                      z_extent_lo, z_extent_hi, ny, nx)
        if result is None:
            continue  # no overlap with the structure's extent after all
        avg_sdf_covered, overlap_frac = result

        coverage_covered = _sdf_to_coverage(avg_sdf_covered, dx, dy)
        frac_mask[z_idx] = coverage_covered * overlap_frac

    return frac_mask