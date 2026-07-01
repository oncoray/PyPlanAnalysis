"""
test_mask_comparison.py
=======================
Compare get_structure_mask() against an independent rasterisation
built on pymedphys contour coordinates.

pymedphys.pull_structure() reads ContourData independently from our
code and returns (x, y, z) arrays per slice. We rasterise those
with the same point-in-polygon approach so the comparison isolates
differences in contour parsing, not rasterisation algorithm.

Requirements
------------
    pip install pymedphys pydicom numpy matplotlib
    (pymedphys and pydicom already in your DVHenv)

Usage
-----
    python test_mask_comparison.py
"""

import numpy as np
import pydicom
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MplPath
from pathlib import Path

import os
# your module — adjust path if needed
from proton_dvh.io import (get_structure_mask, get_all_structure_names,
                            get_grid_geometry)
from pymedphys.dcm import pull_structure,list_structures


# ============================================================
#  EDIT THESE
# ============================================================
test_path = r"\\sv-onc-fs1\Home$/parrellgi/Documents/MBRO/DVH_code_local"
os.chdir(test_path)
RTSTRUCT_PATH = Path("test/Test_case/DATA/RTSTRUCT.dcm")
DOSE_PATH     = Path("test/Test_case/DATA/Physicaldose.dcm")
CT_PATH     = Path("test/Test_case/DATA/CT/CT1.2.826.0.1.3680043.9.7275.1.10060890079958039572264163566079191.dcm")
STRUCT_NAME   = ["CTV_TBV_4800" ,'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasm']         # must match name in RT Struct exactly
N_SLICES      = 6              # slices to show in the visual comparison
# ============================================================


# ============================================================
#  Method A — your get_structure_mask()
# ============================================================

def mask_method_A(struct_name, rtstruct_ds, dose_ds):
    return get_structure_mask(struct_name, rtstruct_ds, dose_ds)


# ============================================================
#  Method B — pymedphys contour coordinates, same rasterisation
# ============================================================

def mask_method_B(struct_name, rtstruct_ds, dose_ds):
    """
    Pull contour coordinates via pymedphys.pull_structure(), then
    rasterise onto the dose grid using the same point-in-polygon
    approach as Method A.

    pymedphys returns:
        x : list of np.ndarray  — x coords per contour slice
        y : list of np.ndarray  — y coords per contour slice
        z : list of np.ndarray  — z coords per contour slice
    Each list element is one contour polygon.
    """
    # pull_structure raises KeyError if name not found
    x_list, y_list, z_list = pull_structure(struct_name, rtstruct_ds)

    origin, spacing = get_grid_geometry(dose_ds)
    x0, y0, z0      = origin
    dx, dy, dz      = spacing
    shape            = dose_ds.pixel_array.shape        # (z, y, x)
    z_offsets        = [float(v) for v in dose_ds.GridFrameOffsetVector]
    z_positions      = np.array([z0 + o for o in z_offsets])

    mask = np.zeros(shape, dtype=bool)

    # pre-build grid point array once
    xi      = np.arange(shape[2])
    yi      = np.arange(shape[1])
    XX, YY  = np.meshgrid(x0 + xi * dx, y0 + yi * dy)
    grid_xy = np.column_stack([XX.ravel(), YY.ravel()])

    for x_pts, y_pts, z_pts in zip(x_list, y_list, z_list):
        if len(x_pts) < 3:
            continue

        z_val = float(z_pts[0])
        z_idx = int(np.argmin(np.abs(z_positions - z_val)))

        poly_xy = np.column_stack([x_pts, y_pts])
        inside  = MplPath(poly_xy).contains_points(grid_xy)
        mask[z_idx] |= inside.reshape(shape[1], shape[2])

    return mask


# ============================================================
#  Comparison metrics
# ============================================================

def dice_coefficient(mask_a, mask_b):
    intersection = (mask_a & mask_b).sum()
    denom        = mask_a.sum() + mask_b.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def hausdorff_distance_2d(mask_a, mask_b):
    """
    Mean per-slice Hausdorff distance on non-empty slices [voxels].
    Full 3D Hausdorff needs scipy.ndimage — this is a lightweight proxy.
    """
    from scipy.spatial.distance import directed_hausdorff
    distances = []
    for z in range(mask_a.shape[0]):
        a = mask_a[z]
        b = mask_b[z]
        if not a.any() and not b.any():
            continue
        if not a.any() or not b.any():
            distances.append(np.nan)
            continue
        pts_a = np.argwhere(a)
        pts_b = np.argwhere(b)
        d = max(directed_hausdorff(pts_a, pts_b)[0],
                directed_hausdorff(pts_b, pts_a)[0])
        distances.append(d)
    valid = [d for d in distances if not np.isnan(d)]
    return float(np.mean(valid)) if valid else float("nan")


def compare_masks(mask_a, mask_b,
                  label_a="Method A",
                  label_b="Method B",
                  voxel_vol_cc=None):

    dice     = dice_coefficient(mask_a, mask_b)
    vol_a    = int(mask_a.sum())
    vol_b    = int(mask_b.sum())
    vol_diff = abs(vol_a - vol_b) / max(vol_a, 1) * 100.0
    hd       = hausdorff_distance_2d(mask_a, mask_b)

    print(f"\n{'='*55}")
    print(f"  Mask comparison: {label_a}  vs  {label_b}")
    print(f"{'='*55}")
    print(f"  {label_a:<30} voxels : {vol_a}")
    print(f"  {label_b:<30} voxels : {vol_b}")
    if voxel_vol_cc is not None:
        print(f"  Volume A : {vol_a * voxel_vol_cc:.2f} cc")
        print(f"  Volume B : {vol_b * voxel_vol_cc:.2f} cc")
    print(f"  Volume difference          : {vol_diff:.2f} %")
    print(f"  Dice coefficient           : {dice:.4f}  (1.0 = perfect)")
    print(f"  Mean Hausdorff (voxels)    : {hd:.2f}")
    print(f"{'='*55}")

    if dice > 0.98:
        print("  ✓ Excellent agreement")
    elif dice > 0.95:
        print("  ~ Good agreement — small boundary differences")
    else:
        print("  ✗ Significant differences — investigate slice plot")

    return {"dice": dice,
            "vol_a": vol_a, "vol_b": vol_b,
            "vol_diff_pct": vol_diff,
            "hausdorff_mean_vox": hd}


# ============================================================
#  Visual comparison
# ============================================================

def plot_mask_comparison(mask_a, mask_b, dose_ds, struct_name,
                         label_a="get_structure_mask",
                         label_b="pymedphys",
                         n_slices=6):
    """
    For n_slices evenly spaced active slices show:
      Left   — mask A overlay (green)
      Middle — mask B overlay (blue)
      Right  — pixel-level diff:
                 white  = both agree (inside)
                 green  = A only
                 blue   = B only
                 black  = both agree (outside)
    """
    active = np.where(mask_a.any(axis=(1, 2)) |
                      mask_b.any(axis=(1, 2)))[0]
    if len(active) == 0:
        print("Both masks are completely empty — nothing to plot.")
        return

    indices = active[np.linspace(0, len(active) - 1,
                                  min(n_slices, len(active)),
                                  dtype=int)]

    scale   = float(dose_ds.DoseGridScaling)
    fig     = plt.figure(figsize=(13, 3.2 * len(indices)))
    gs      = gridspec.GridSpec(len(indices), 3,
                                hspace=0.45, wspace=0.25)

    for row, z in enumerate(indices):
        dose_bg = dose_ds.pixel_array[z].astype(float) * scale
        a       = mask_a[z]
        b       = mask_b[z]

        # RGB overlap image
        rgb           = np.zeros((*a.shape, 3))
        rgb[a & b]    = [1.0, 1.0, 1.0]   # both    → white
        rgb[a & ~b]   = [0.2, 0.9, 0.2]   # A only  → green
        rgb[~a & b]   = [0.2, 0.4, 1.0]   # B only  → blue
        # outside both stays black [0,0,0]

        panels = [
            (a.astype(float), "Greens",  f"{label_a}  z={z}"),
            (b.astype(float), "Blues",   f"{label_b}  z={z}"),
            (rgb,             None,      f"Diff  z={z}\n"
                                         "white=both  green=A  blue=B"),
        ]

        for col, (data, cmap, title) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(dose_bg, cmap="gray", origin="lower",
                      interpolation="none",
                      vmin=0, vmax=np.percentile(dose_bg[dose_bg > 0], 99)
                                   if dose_bg.any() else 1)
            if cmap is not None:
                ax.imshow(data, cmap=cmap, alpha=0.45,
                          origin="lower", vmin=0, vmax=1,
                          interpolation="none")
            else:
                ax.imshow(data, origin="lower",
                          interpolation="none", alpha=0.7)
            ax.set_title(title, fontsize=7)
            ax.axis("off")

    fig.suptitle(f"Mask comparison — {struct_name}",
                 fontsize=11, fontweight="bold", y=1.01)

    out = f"mask_comparison_{struct_name}.png"
    plt.savefig(out, dpi=850, bbox_inches="tight")
    #plt.show()
    print(f"\nSaved: {out}")


# ============================================================
#  Per-slice volume profile
# ============================================================

def plot_volume_profile(mask_a, mask_b, voxel_vol_cc,
                        label_a="get_structure_mask",
                        label_b="pymedphys",
                        struct_name=""):
    """
    Plot per-slice volume (cc) for both masks.
    Makes slice-level disagreements immediately visible.
    """
    vol_a = mask_a.sum(axis=(1, 2)) * voxel_vol_cc
    vol_b = mask_b.sum(axis=(1, 2)) * voxel_vol_cc
    slices = np.arange(len(vol_a))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(slices, vol_a, label=label_a, color="#2ca02c", lw=1.5)
    ax1.plot(slices, vol_b, label=label_b, color="#1f77b4", lw=1.5,
             linestyle="--")
    ax1.set_ylabel("Volume per slice (cc)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Per-slice volume profile — {struct_name}")

    diff = vol_a - vol_b
    ax2.bar(slices, diff,
            color=["#d62728" if d < 0 else "#2ca02c" for d in diff],
            width=1.0)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Slice index (z)")
    ax2.set_ylabel("Difference A − B (cc)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = f"volume_profile_{struct_name}.png"
    plt.savefig(out, dpi=850, bbox_inches="tight")
    #plt.show()
    print(f"Saved: {out}")


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":

    print(f"Loading DICOM files...")
    rtstruct_ds = pydicom.dcmread(str(RTSTRUCT_PATH))
    dose_ds     = pydicom.dcmread(str(DOSE_PATH))
    CT_ds  = pydicom.dcmread(str(CT_PATH))
    print("Structures available:")
    print(" ", get_all_structure_names(rtstruct_ds))
    print("\nStructures via pymedphys:")
    print(" ", list_structures(rtstruct_ds))

    # voxel volume
    _, spacing   = get_grid_geometry(dose_ds)
    voxel_vol_cc = float(np.prod(spacing)) / 1000.0
    for STRUCT in STRUCT_NAME:
        print(f"\nComputing Method A (get_structure_mask)...")
        mask_a = mask_method_A(STRUCT, rtstruct_ds, dose_ds)
    
        print(f"Computing Method B (pymedphys contours)...")
        mask_b = mask_method_B(STRUCT, rtstruct_ds, dose_ds)
    
        # quantitative metrics
        stats = compare_masks(mask_a, mask_b,
                              label_a="get_structure_mask",
                              label_b="pymedphys",
                              voxel_vol_cc=voxel_vol_cc)
    
        # visual slice comparison
        # plot_mask_comparison(mask_a, mask_b, CT_ds, STRUCT,
        #                      n_slices=N_SLICES)
    
        # per-slice volume profile
        plot_volume_profile(mask_a, mask_b, voxel_vol_cc,
                            struct_name=STRUCT)