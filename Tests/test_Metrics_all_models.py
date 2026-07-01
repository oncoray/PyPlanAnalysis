# -*- coding: utf-8 -*-
"""
test_metrics_all_models.py
=====================================
Compare DVH and LVH metrics from proton_dvh output (xlsx files)
against RayStation reference curves.

"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Utils import Paths

# ============================================================
#  CONFIG — edit these
# ============================================================

Test_folder_name = "DD-0ZKEKUPU"

Metrics_file     = Paths.TEST_OUTPUT / Test_folder_name / Paths.TEST_PA_OUTPUT / "dvh_metrics.xlsx"


CASES = [
            {
                "case_dir"    : Paths.TEST_OUTPUT / Test_folder_name,
                "ref_dvh_dir" : Paths.TEST_DATA / Test_folder_name / Paths.TEST_DATA_REFERENCE,
                "metrics_xls" : Metrics_file ,   # single file, all labels
                "ref_lvh_csv" : "NOMINAL_LVH.csv",            # shared LVH reference
                "let_label"   : "LET",                        # column prefix for LET metrics
                "labels": [
                    {"name": "mcnamara",  "ref_dvh_csv": "NOMINAL_DVH_mcnamara.csv"},
                    {"name": "wedenberg", "ref_dvh_csv": "NOMINAL_DVH_wedenberg.csv"},
                    {"name": "carabe",    "ref_dvh_csv": "NOMINAL_DVH_carabe.csv"},
                    {"name": "RBE1.1",    "ref_dvh_csv": "NOMINAL_DVH.csv"},
                ],
            }
            # Add more cases 
        ]

# Output root
OUTPUT_ROOT =  Paths.TEST_OUTPUT / Test_folder_name / Paths.TEST_RESULTS

# Tolerances
REL_TOL_GY   = 3      # %   relative tolerance for Dx metrics
ABS_TOL_GY   = 0.5    # Gy  absolute tolerance for Dx metrics
REL_TOL_PCT  = 1.0    # %   relative tolerance for Vx_% metrics
ABS_TOL_PCT  = 5      # cc  absolute tolerance for Vx_cc metrics
DMEAN_TOL_GY = 0.3    # Gy  tolerance for Dmean

SAVE_PLOTS = True

# ============================================================
#  1. Parse RayStation reference CSV
# ============================================================

def parse_raystation_csv(csv_path: Path) -> dict:
    """
    Parse a RayStation DVH/LVH export CSV.

    Returns dict with keys:
        structures, volumes_cc, dmean,
        curves: {struct: (vol_pct_array, dose_array)}
    """
    with open(csv_path, "r") as f:
        lines = f.read().splitlines()

    header_cells = lines[0].split(";")
    structures   = [c.strip() for c in header_cells[1:] if c.strip()]

    vol_cells  = lines[1].split(";")
    volumes_cc = {}
    for i, s in enumerate(structures):
        try:
            volumes_cc[s] = float(vol_cells[i + 1])
        except (ValueError, IndexError):
            volumes_cc[s] = float("nan")

    dmean_cells = lines[-1].split(";")
    dmean = {}
    if dmean_cells[0].strip().lower() == "dmean":
        for i, s in enumerate(structures):
            try:
                dmean[s] = float(dmean_cells[i + 1])
            except (ValueError, IndexError):
                dmean[s] = float("nan")

    vol_pct_list = {s: [] for s in structures}
    dose_list    = {s: [] for s in structures}

    for line in lines[2:]:
        cells = line.split(";")
        first = cells[0].strip()
        if first.lower() == "dmean" or first == "":
            continue
        try:
            vp = float(first)
        except ValueError:
            continue
        for i, s in enumerate(structures):
            try:
                d = float(cells[i + 1])
            except (ValueError, IndexError):
                d = float("nan")
            vol_pct_list[s].append(vp)
            dose_list[s].append(d)

    curves = {}
    for s in structures:
        vp  = np.array(vol_pct_list[s])
        dos = np.array(dose_list[s])
        order     = np.argsort(vp)
        curves[s] = (vp[order], dos[order])

    return {"structures": structures,
            "volumes_cc": volumes_cc,
            "dmean"     : dmean,
            "curves"    : curves}


# ============================================================
#  2. Interpolation helpers
# ============================================================

def dx_from_curve(vol_pct, dose, x):
    valid = ~(np.isnan(vol_pct) | np.isnan(dose))
    vp, d = vol_pct[valid], dose[valid]
    if len(vp) < 2:
        return float("nan")
    f = interp1d(vp, d, kind="linear", bounds_error=False,
                 fill_value=(d[np.argmin(vp)], d[np.argmax(vp)]))
    return float(f(x))


def vx_from_curve(vol_pct, dose, x_gy):
    valid = ~(np.isnan(vol_pct) | np.isnan(dose))
    vp, d = vol_pct[valid], dose[valid]
    if len(vp) < 2:
        return float("nan")
    if x_gy > np.nanmax(d):
        return 0.0
    if x_gy <= np.nanmin(d):
        return 100.0
    order = np.argsort(d)
    d_s, vp_s = d[order], vp[order]
    _, unique_idx = np.unique(d_s, return_index=True)
    d_s, vp_s = d_s[unique_idx], vp_s[unique_idx]
    if len(d_s) < 2:
        return float("nan")
    f = interp1d(d_s, vp_s, kind="linear", bounds_error=False,
                 fill_value=(vp_s[0], vp_s[-1]))
    return float(f(x_gy))


# ============================================================
#  3. Extract reference metric from column name
# ============================================================

def extract_ref_metric(col_name, struct, ref, dose_label):
    if struct not in ref["curves"]:
        return float("nan")
    vp, dos = ref["curves"][struct]
    suffix  = col_name.replace(f"{dose_label}_", "", 1)

    if suffix in ("Dmean", "mean"):
        return ref["dmean"].get(struct, float("nan"))
    if suffix in ("Dmax", "max"):
        return dx_from_curve(vp, dos, 0.0)
    if suffix in ("Dmin", "min"):
        return dx_from_curve(vp, dos, 100.0)

    if suffix.startswith("D") and suffix.endswith("%"):
        try:
            return dx_from_curve(vp, dos, float(suffix[1:-1]))
        except ValueError:
            return float("nan")

    if suffix.startswith("L") and suffix.endswith("%"):
        try:
            return dx_from_curve(vp, dos, float(suffix[1:-1]))
        except ValueError:
            return float("nan")

    if suffix.startswith("V") and suffix.endswith("Gy_%"):
        try:
            return vx_from_curve(vp, dos, float(suffix[1:-4]))
        except ValueError:
            return float("nan")

    if suffix.startswith("V") and suffix.endswith("Gy_cc"):
        try:
            x      = float(suffix[1:-5])
            vx_pct = vx_from_curve(vp, dos, x)
            vol_cc = ref["volumes_cc"].get(struct, float("nan"))
            return vx_pct / 100.0 * vol_cc
        except ValueError:
            return float("nan")

    return float("nan")


# ============================================================
#  4. Comparison
# ============================================================

def run_comparison(xlsx_path, ref, dose_label, metric_type="DVH"):
    if not xlsx_path.exists():
        print(f"  [SKIP] File not found: {xlsx_path}")
        return pd.DataFrame()

    df   = pd.read_excel(xlsx_path)
    rows = []

    for _, patient_row in df.iterrows():
        struct = str(patient_row.get("ROI_Name", "")).strip()
        if not struct:
            continue

        target_cols = [c for c in df.columns
                       if str(c).startswith(f"{dose_label}_")]

        for col in target_cols:
            our_val = patient_row[col]
            if pd.isna(our_val):
                continue

            ref_val = extract_ref_metric(str(col), struct, ref, dose_label)
            if np.isnan(ref_val):
                continue

            diff     = float(our_val) - ref_val
            rel_diff = diff / ref_val * 100.0 if abs(ref_val) > 1e-6 else float("nan")

            suffix = str(col).replace(f"{dose_label}_", "", 1)
            if "Dmean" in suffix:
                passed = abs(diff) <= DMEAN_TOL_GY
            elif suffix.startswith("V") and "%" in suffix:
                passed = abs(diff) <= REL_TOL_PCT
            elif suffix.startswith("V") and "cc" in suffix:
                passed = abs(diff) <= ABS_TOL_PCT
            else:
                passed = (abs(diff) <= ABS_TOL_GY) or (abs(rel_diff) <= REL_TOL_GY)

            rows.append({
                "metric_type" : metric_type,
                "structure"   : struct,
                "metric"      : str(col),
                "proton_dvh"  : float(our_val),
                "reference"   : ref_val,
                "difference"  : diff,
                "rel_diff_%"  : rel_diff,
                "pass"        : passed,
            })

    return pd.DataFrame(rows)


# ============================================================
#  5. Reporting
# ============================================================

def print_summary(results, label):
    if results.empty:
        print(f"\n  No results for {label}.")
        return
    n_total = len(results)
    n_pass  = results["pass"].sum()
    print(f"\n{'='*60}")
    print(f"  {label} — {n_pass}/{n_total} metrics passed")
    print(f"{'='*60}")
    failed = results[~results["pass"]]
    if not failed.empty:
        print("  FAILED metrics:")
        for _, r in failed.sort_values("difference", key=abs, ascending=False).iterrows():
            print(f"    {r['structure']:<30} {r['metric']:<30} "
                  f"ours={r['proton_dvh']:7.3f}  ref={r['reference']:7.3f}  "
                  f"diff={r['difference']:+7.3f}")
    else:
        print("  All metrics passed ✓")


def plot_comparison(results, label, save_dir):
    if results.empty:
        return
    structures = results["structure"].unique()
    n     = len(structures)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for i, struct in enumerate(structures):
        ax  = axes[i // ncols][i % ncols]
        sub = results[results["structure"] == struct]

        passed = sub[sub["pass"]]
        failed = sub[~sub["pass"]]

        all_vals = pd.concat([sub["reference"], sub["proton_dvh"]])
        lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]

        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="identity")
        if not passed.empty:
            ax.scatter(passed["reference"], passed["proton_dvh"],
                       c="#2ca02c", s=30, alpha=0.8, label="pass")
        if not failed.empty:
            ax.scatter(failed["reference"], failed["proton_dvh"],
                       c="#d62728", s=40, marker="x", linewidths=1.5, label="fail")

        ax.set_xlabel("Reference", fontsize=8)
        ax.set_ylabel("proton_dvh", fontsize=8)
        ax.set_title(struct, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(lims); ax.set_ylim(lims)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{label} — reference vs proton_dvh", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{label.replace(' ', '_')}_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dvh_overlay(ref, xlsx_path, dose_label, label, save_dir):
    if not xlsx_path.exists():
        return
    df         = pd.read_excel(xlsx_path)
    structures = [s for s in df["ROI_Name"].unique() if s in ref["curves"]]
    if not structures:
        return

    n     = len(structures)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)

    dx_points = [2, 5, 50, 95, 98]

    for i, struct in enumerate(structures):
        ax  = axes[i // ncols][i % ncols]
        row = df[df["ROI_Name"] == struct].iloc[0]
        vp_ref, dos_ref = ref["curves"][struct]

        ax.plot(dos_ref, vp_ref, color="#1f77b4", lw=1.5,
                label="RayStation", zorder=2)

        our_dx, ref_dx, vols = [], [], []
        for x in dx_points:
            col = f"{dose_label}_D{x}%"
            if col in row.index and not pd.isna(row[col]):
                our_dx.append(float(row[col]))
                ref_dx.append(dx_from_curve(vp_ref, dos_ref, x))
                vols.append(x)

        if our_dx:
            ax.scatter(our_dx, vols, color="#d62728", s=50, zorder=5,
                       label=f"{dose_label} Dx%", marker="o")
            ax.scatter(ref_dx, vols, color="#1f77b4", s=50, zorder=5,
                       label="ref Dx%", marker="^")
            for o, r, v in zip(our_dx, ref_dx, vols):
                ax.plot([o, r], [v, v], color="gray", lw=0.7, alpha=0.6)

        ax.set_xlabel("Dose [Gy]", fontsize=8)
        ax.set_ylabel("Volume [%]", fontsize=8)
        ax.set_title(struct, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 105); ax.set_xlim(left=0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{label} — DVH overlay (ref curve + Dx% points)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{label.replace(' ', '_')}_dvh_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
#  6. Per-case summary plot  (all labels side by side)
# ============================================================

def plot_pass_rate_summary(all_label_results: dict, case_name: str, save_dir: Path):
    """
    Bar chart comparing pass-rate per structure for every dose label.

    Parameters
    ----------
    all_label_results : {dose_label: DataFrame}  — DVH results per label
    case_name         : string identifier for the case (used in title)
    save_dir          : output folder for the figure
    """
    # collect structures present in any label
    structures = sorted({s for df in all_label_results.values()
                         if not df.empty
                         for s in df["structure"].unique()})
    if not structures:
        return

    labels  = [lbl for lbl, df in all_label_results.items() if not df.empty]
    x       = np.arange(len(structures))
    width   = 0.8 / max(len(labels), 1)
    colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(structures)), 5))

    for k, lbl in enumerate(labels):
        df = all_label_results[lbl]
        rates = []
        for s in structures:
            sub = df[df["structure"] == s]
            rates.append(sub["pass"].mean() * 100 if not sub.empty else float("nan"))

        offset = (k - len(labels) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, rates, width, label=lbl,
                        color=colors[k % len(colors)], alpha=0.85)
        for bar, rate in zip(bars, rates):
            if not np.isnan(rate):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f"{rate:.0f}%", ha="center", va="bottom",
                        fontsize=7)

    ax.axhline(100, color="green", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(structures, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pass rate [%]")
    ax.set_ylim(0, 115)
    ax.set_title(f"DVH pass rate by structure — {case_name}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"pass_rate_summary_{case_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
#  7. Main — iterate cases × dose labels
# ============================================================

if __name__ == "__main__":

    global_rows = []   # accumulates all results across cases and labels

    for case_cfg in CASES:
        case_dir      = Path(case_cfg["case_dir"])
        ref_dvh_dir   = Path(case_cfg["ref_dvh_dir"])
        xlsx_path     = case_dir / case_cfg.get("metrics_xls", Test_folder_name / Paths.TEST_PA_OUTPUT / "dvh_metrics.xlsx")
        ref_lvh_path  = ref_dvh_dir / case_cfg.get("ref_lvh_csv", "NOMINAL_LVH.csv")
        let_label     = case_cfg.get("let_label", "LET")
        label_cfgs    = case_cfg["labels"]          # list of {name, ref_dvh_csv}
        case_name     = case_dir.name
        case_out_root = OUTPUT_ROOT 

        print(f"\n{'#'*70}")
        print(f"  CASE: {case_name}")
        print(f"{'#'*70}")

        if not xlsx_path.exists():
            print(f"  [SKIP] xlsx not found: {xlsx_path}")
            continue

        # ── Load LVH reference once per case (shared across labels) ──────────
        ref_lvh = None
        if ref_lvh_path.exists():
            ref_lvh = parse_raystation_csv(ref_lvh_path)
            print(f"  Reference LVH structures: {ref_lvh['structures']}")
        else:
            print(f"  [INFO] LVH reference not found, skipping LVH: {ref_lvh_path}")

        # LVH is the same regardless of dose label — run it once and save to
        # the case root so it is not duplicated in every label subfolder.
        lvh_results_case = pd.DataFrame()
        if ref_lvh is not None:
            lvh_results_case = run_comparison(xlsx_path, ref_lvh, let_label, "LVH")
            print_summary(lvh_results_case, f"LVH [{let_label}]  case={case_name}")
            if SAVE_PLOTS and not lvh_results_case.empty:
                plot_comparison(lvh_results_case,
                                f"LVH_{let_label}_{case_name}",
                                case_out_root / let_label)
        # ── LVH CSV at case level ────────────────────────────────────────────
        if not lvh_results_case.empty:
            lvh_results_case["case"]       = case_name
            lvh_results_case["dose_label"] = let_label
            case_out_root.mkdir(parents=True, exist_ok=True)
            lvh_csv = case_out_root /  let_label /  "LVH_metric_comparison.csv"
            lvh_results_case.to_csv(lvh_csv, index=False)
            print(f"  LVH CSV saved: {lvh_csv}")
            global_rows.append(lvh_results_case)

        # ── Iterate over dose labels ─────────────────────────────────────────
        all_label_dvh = {}   # {label_name: dvh_results}  for summary plot

        for lbl_cfg in label_cfgs:
            dose_label    = lbl_cfg["name"]
            ref_dvh_path  = ref_dvh_dir  / lbl_cfg["ref_dvh_csv"]
            label_out_dir = case_out_root / dose_label

            print(f"\n  --- Label: {dose_label} ---")

            # load the DVH reference specific to this label
            if not ref_dvh_path.exists():
                print(f"  [SKIP] Reference DVH not found: {ref_dvh_path}")
                all_label_dvh[dose_label] = pd.DataFrame()
                continue

            ref_dvh = parse_raystation_csv(ref_dvh_path)
            print(f"  Reference DVH structures: {ref_dvh['structures']}")

            # ---- DVH comparison ----
            dvh_results = run_comparison(xlsx_path, ref_dvh, dose_label, "DVH")
            print_summary(dvh_results, f"DVH [{dose_label}]  case={case_name}")
            all_label_dvh[dose_label] = dvh_results

            if SAVE_PLOTS and not dvh_results.empty:
                plot_comparison(dvh_results,
                                f"DVH_{dose_label}_{case_name}",
                                label_out_dir)
                plot_dvh_overlay(ref_dvh, xlsx_path, dose_label,
                                 f"DVH_{dose_label}_{case_name}",
                                 label_out_dir)

            # ---- Save per-label CSV (DVH only; LVH saved at case level) ----
            if not dvh_results.empty:
                dvh_results["case"]       = case_name
                dvh_results["dose_label"] = dose_label
                label_out_dir.mkdir(parents=True, exist_ok=True)
                out_csv = label_out_dir / "metric_comparison.csv"
                dvh_results.to_csv(out_csv, index=False)
                print(f"  CSV saved: {out_csv}")
                global_rows.append(dvh_results)

        
        # ── Per-case summary plot (all DVH labels side-by-side) ──────────────
        if SAVE_PLOTS:
            plot_pass_rate_summary(all_label_dvh, case_name, case_out_root)

    # ============================================================
    #  Global summary across all cases and labels
    # ============================================================
    if global_rows:
        global_df = pd.concat(global_rows, ignore_index=True)

        global_csv = OUTPUT_ROOT / "global_metric_comparison.csv"
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        global_df.to_csv(global_csv, index=False)
        print(f"\n{'='*60}")
        print(f"  Global CSV saved: {global_csv}")

        print("\nGlobal summary by case / label / structure:")
        summary = (global_df
                   .groupby(["case", "dose_label", "metric_type", "structure"])
                   .agg(n_metrics   =("pass", "count"),
                        n_pass      =("pass", "sum"),
                        max_abs_diff=("difference", lambda x: x.abs().max()))
                   .reset_index())
        summary["pass_rate_%"] = summary["n_pass"] / summary["n_metrics"] * 100
        print(summary.to_string(index=False))
    else:
        print("\nNo results to aggregate.")