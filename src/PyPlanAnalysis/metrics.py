"""
PyPlanAnalysis.metrics
==================
DVH/LVH metric computation (Dx%, Vx, gEUD, EQD2, D_Lx%, L_Dx%) and
per-structure radiobiological parameter configuration.

"""

import numpy as np
from dataclasses import dataclass, field


# ============================================================
#  Configuration dataclasses
# ============================================================

@dataclass
class RadiobiologyConfig:
    """
    Per-structure radiobiological parameters.

    Parameters
    ----------
    alpha_beta_map : dict
        Keys are substrings of structure names (case-insensitive).
        Values are α/β [Gy].
    alpha_beta_default : float
        Fallback α/β when no key matches.
    geud_a_map : dict
        Keys are substrings of structure names.
        Values are gEUD parameter a.
    geud_a_default : float
        Fallback gEUD a.
    """
    alpha_beta_map     : dict  = field(default_factory=lambda: {
        "CTV"       : 10.0,
        "GTV"       : 10.0,
        "PTV"       : 10.0,
        "brainstem" : 2.0,
        "parotid"   : 3.0,
        "lens"      : 1.2,
    })
    alpha_beta_default : float = 2.0

    geud_a_map         : dict  = field(default_factory=lambda: {
        "CTV"      : -10.0,
        "GTV"      : -10.0,
        "PTV"      : -10.0,
        "brainstem": 4.0,
        "parotid"  : 1.0,
    })
    geud_a_default     : float = 1.0

    def get_alpha_beta(self, structure_name: str) -> float:
        name = structure_name.lower()
        for key, val in self.alpha_beta_map.items():
            if key.lower() in name:
                return val
        return self.alpha_beta_default

    def get_geud_a(self, structure_name: str) -> float:
        name = structure_name.lower()
        for key, val in self.geud_a_map.items():
            if key.lower() in name:
                return val
        return self.geud_a_default


@dataclass
class MetricConfig:
    """
    Configure which DVH metrics to compute.

    Parameters
    ----------
    dx : list of int/float
        Dx% points (% volume). e.g. [2, 5, 50, 95, 98]
    lx : list of int/float
        Lx% points (% volume). e.g. [2, 50, 98]
    vx : list of int/float
        Vx dose thresholds [Gy(RBE)]. e.g. [20, 30, 40, 50]
    LET_thr : float
        LETd threshold used to consider dirty dose (D where L>LET_thr) [#keV/um]
    compute_eqd2 : bool
        Whether to compute EQD2 (requires alpha_beta).
    compute_geud : bool
        Whether to compute gEUD.
    dvh_bins : int
        Number of bins for DVH histograms.
    lvh_bins : int
        Number of bins for LVH histograms.
    dlvh_dose_bins : int
        Dose axis bins for 2-D DLVH.
    dlvh_let_bins : int
        LET axis bins for 2-D DLVH.
    """
    dx             : list = field(default_factory=lambda: [2, 5, 50, 95, 98])
    lx             : list = field(default_factory=lambda:[2,50,98])
    vx             : list = field(default_factory=lambda: [20, 30, 40, 50])
    LET_thr        : float = 2.5 #keV/um
    compute_eqd2   : bool = True
    compute_geud   : bool = True
    dvh_bins       : int  = 1000
    lvh_bins       : int  = 1000
    dlvh_dose_bins : int  = 1000
    dlvh_let_bins  : int  = 1000
    New_grid       : list = field(default_factory=lambda: [1.5,1.5,1.5])


# ============================================================
#  Metric computation
# ============================================================

def _dx_percent(sorted_ascending: np.ndarray, x: float) -> float:
    """
    Return Dx% or Lx% – the minimum dose or LETd delivered to the hottest x% of the volume.

    Definition: sort voxels ascending, the hottest x% are the top x% of the
    array. Dx% is the lowest value among those, i.e. the boundary dose.

    Example – D2% for n=100 voxels:
        top 2% = voxels at indices [98, 99]  (0-based, ascending sort)
        D2% = dose_sort[98]   ← lowest dose in that hottest 2%

    Derivation of index:
        number of voxels in top x% = ceil(x/100 * n)
        first index of that group   = n - ceil(x/100 * n)
        clipped to [0, n-1] for edge cases (x=0 → Dmax, x=100 → Dmin)
    """
    n         = len(sorted_ascending)
    n_hot     = int(np.ceil(x / 100.0 * n))          # how many voxels in top x%
    first_idx = int(np.clip(n - n_hot, 0, n - 1))    # index of the lowest of those
    return float(sorted_ascending[first_idx])


def _dx_percent_weighted(values: np.ndarray,
                         weights: np.ndarray,
                         x: float) -> float:
    """
    Weighted Dx% — minimum value in the top x% of cumulative weight.

    Sort values descending, accumulate weights, find where cumulative
    weight first exceeds x% of total weight. That value is Dx%.

    This is the correct generalisation of Dx% when voxels have
    fractional volumes (weights = fractional mask values).
    """
    total_w = weights.sum()
    if total_w == 0:
        return float("nan")

    # sort descending by value
    order    = np.argsort(values)[::-1]
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w    = np.cumsum(w_sorted) / total_w * 100.0   # cumulative % of total weight
    
    idx = np.searchsorted(cum_w, x)
    idx = int(np.clip(idx, 0, len(v_sorted) - 1))
    
    # print(cum_w[idx])
    return float(v_sorted[idx])

def _top_x_percent_weighted_idx(values: np.ndarray,
                                  weights: np.ndarray,
                                  x: float):
    """
    Return the indices of voxels in the top x% by weighted cumulative volume.
    These are the voxels that define Dx% — the ones whose cumulative
    weight from the top reaches x% of total weight.
    """
    total_w = weights.sum()
    if total_w == 0:
        return np.array([], dtype=int)

    order   = np.argsort(values)[::-1]          # descending
    cum_w   = np.cumsum(weights[order]) / total_w * 100.0
    n_keep  = int(np.searchsorted(cum_w, x)) + 1 # +1: include boundary voxel
    n_keep  = int(np.clip(n_keep, 1, len(values)))
    return order[:n_keep]


def _L_Dx_metric_weighted(dose_voxels: np.ndarray,
                           let_voxels: np.ndarray,
                           weights: np.ndarray,
                           x: float) -> float:
    """
    Weighted L_Dx% — among the top x% dose voxels (by weighted volume),
    find Lx% of their LET distribution (also weighted).

    Strictly equivalent to unweighted version when weights == ones.
    """
    hot_idx    = _top_x_percent_weighted_idx(dose_voxels, weights, x)
    if len(hot_idx) == 0:
        return float("nan")
    return _dx_percent_weighted(let_voxels[hot_idx],
                                 weights[hot_idx],
                                 x)


def _D_Lx_metric_weighted(dose_voxels: np.ndarray,
                           let_voxels: np.ndarray,
                           weights: np.ndarray,
                           x: float) -> float:
    """
    Weighted D_Lx% — among the top x% LET voxels (by weighted volume),
    find Dx% of their dose distribution (also weighted).

    Strictly equivalent to unweighted version when weights == ones.
    """
    hot_idx    = _top_x_percent_weighted_idx(let_voxels, weights, x)
    if len(hot_idx) == 0:
        return float("nan")
    return _dx_percent_weighted(dose_voxels[hot_idx],
                                 weights[hot_idx],
                                 x)


def compute_dvh_metrics(dose_voxels: np.ndarray,
                        voxel_vol_cc: float,
                        label: str,
                        metric_cfg: MetricConfig,
                        alpha_beta: float = 3.0,
                        geud_a: float = 1.0,
                        weights: np.ndarray = None,
                        n_fractions: int = 30) -> dict:
    """
    Compute all requested DVH metrics for a 1-D voxel dose array.

    Parameters
    ----------
    dose_voxels  : 1-D array of dose values [Gy or Gy(RBE)], already masked
    voxel_vol_cc : volume of one voxel in cc
    label        : prefix string for metric names (e.g. "phys", "RBE1.1")
    metric_cfg   : MetricConfig instance
    alpha_beta   : α/β ratio [Gy] for EQD2
    geud_a       : gEUD parameter a
    weights      : optional 1-D array of fractional volumes [0,1] per voxel.
                   If None, binary mask assumed (all weights = 1).
                   When provided, all metrics are volume-weighted.

    Returns
    -------
    dict of {metric_name: value}
    """
    m = {}
    n = len(dose_voxels)
    if n == 0:
        return m

    use_weights = weights is not None
    w = weights if use_weights else np.ones(n, dtype=np.float32)
    total_w = w.sum()

    # Basic statistics — weighted
    m[f"{label}_Dmin"]  = float(dose_voxels[w > 0].min()) if (w > 0).any() else 0.0
    m[f"{label}_Dmean"] = float(np.average(dose_voxels, weights=w))
    m[f"{label}_Dmax"]  = float(dose_voxels[w > 0].max()) if (w > 0).any() else 0.0

    # Dx% — weighted version when fractional mask provided
    for x in metric_cfg.dx:
        if use_weights:
            m[f"{label}_D{x}%"] = _dx_percent_weighted(dose_voxels, w, x)
        else:
            dose_sort = np.sort(dose_voxels)
            m[f"{label}_D{x}%"] = _dx_percent(dose_sort, x)

    # Vx — weighted volume receiving >= x Gy(RBE)
    for x in metric_cfg.vx:
        above      = dose_voxels >= x
        w_above    = w[above].sum()
        vol_cc     = float(w_above * voxel_vol_cc)
        vol_pct    = float(w_above / total_w * 100.0) if total_w > 0 else 0.0
        m[f"{label}_V{x}Gy_cc"] = vol_cc
        m[f"{label}_V{x}Gy_%"]  = vol_pct

    # gEUD — weighted power mean
    if metric_cfg.compute_geud:
        d_pos = dose_voxels[dose_voxels > 0]
        w_pos = w[dose_voxels > 0]
        if len(d_pos) > 0 and w_pos.sum() > 0:
            if geud_a != 0:
                geud = float(np.average(d_pos**geud_a, weights=w_pos) ** (1.0 / geud_a))
            else:
                geud = float(np.exp(np.average(np.log(d_pos), weights=w_pos)))
        else:
            geud = 0.0
        m[f"{label}_gEUD"] = geud

    # EQD2 — weighted mean and max
    if metric_cfg.compute_eqd2:
        dose_per_fx = dose_voxels / n_fractions
        eqd2 = dose_voxels * (dose_per_fx + alpha_beta) / (2.0 + alpha_beta)
        m[f"{label}_EQD2_D40%"]  = _dx_percent_weighted(eqd2, w, 40)
        m[f"{label}_EQD2_mean"] = float(np.average(eqd2, weights=w))
        m[f"{label}_EQD2_max"]  = float(eqd2[w > 0].max()) if (w > 0).any() else 0.0

    return m


def compute_let_metrics(let_voxels: np.ndarray,
                        dose_voxels: np.ndarray,
                        lx_points: list,
                        label: str = "LET",
                        weights: np.ndarray = None) -> dict:
    """
    Compute LET summary metrics for a structure.

    Lx% is defined exactly like Dx% but on the LET distribution:
    the minimum LETd in the highest-LET x% of the structure volume.

    Parameters
    ----------
    let_voxels  : 1-D array of LETd values [keV/µm]
    dose_voxels : 1-D array of physical dose values [Gy], same voxels
    lx_points   : list of x values for Lx%
    label       : prefix for metric names
    weights     : optional fractional volumes [0,1] per voxel

    Returns
    -------
    dict of {metric_name: value}
    """
    m = {}
    n = len(let_voxels)
    if n == 0:
        return m

    w       = weights if weights is not None else np.ones(n, dtype=np.float32)
    

    m[f"{label}_mean"] = float(np.average(let_voxels, weights=w))
    m[f"{label}_max"]  = float(let_voxels[w > 0].max()) if (w > 0).any() else 0.0
    m[f"{label}_min"]  = float(let_voxels[w > 0].min()) if (w > 0).any() else 0.0

    # Lx% — weighted
    for x in lx_points:
        m[f"{label}_L{x}%"] = _dx_percent_weighted(let_voxels, w, x)

    # Dose-weighted mean LETd — also accounts for fractional voxel weight
    d_sum = (dose_voxels * w).sum()
    m[f"{label}_doseWeighted_mean"] = (
        float((dose_voxels * let_voxels * w).sum() / d_sum) if d_sum > 0 else 0.0
    )

    return m


# ============================================================
#  Histogram helpers
# ============================================================

def compute_cumulative_histogram(values: np.ndarray,
                                 n_bins: int,
                                 weights: np.ndarray = None) -> tuple:
    """
    Compute a weighted cumulative histogram (DVH or LVH style).

    When weights are provided (fractional mask), each voxel contributes
    its fractional volume rather than 1 full voxel.

    Returns
    -------
    edges : bin edges (length n_bins + 1)
    cum   : cumulative volume fraction from 1 → 0 (length n_bins + 1)
    """
    v_max = values.max() * 1.05 if values.max() > 0 else 1.0
    bins  = np.linspace(0, v_max, n_bins + 1)

    if weights is not None:
        hist, edges = np.histogram(values, bins=bins, weights=weights)
        total = weights.sum()
    else:
        hist, edges = np.histogram(values, bins=bins)
        total = hist.sum()

    if total == 0:
        return edges, np.ones(n_bins + 1)

    cum = 1.0 - np.cumsum(hist) / total
    cum = np.concatenate([[1.0], cum])
    return edges, cum


def compute_2d_histogram(dose_voxels: np.ndarray,
                         let_voxels: np.ndarray,
                         dose_bins: int,
                         let_bins: int) -> tuple:
    """
    Compute 2-D dose-LET volume histogram (DLVH).

    Returns
    -------
    H          : 2-D array (dose_bins × let_bins), volume fractions
    dose_edges : dose bin edges
    let_edges  : LET bin edges
    """
    d_edges = np.linspace(0, dose_voxels.max() * 1.05, dose_bins + 1)
    l_edges = np.linspace(0, let_voxels.max()  * 1.05, let_bins  + 1)

    H_diff, _, _ = np.histogram2d(dose_voxels, let_voxels,
                                  bins=[d_edges, l_edges])

    # Convert to cumulative: H[i,j] = volume where dose >= d_edges[i] AND let >= l_edges[j]
    # Flip both axes, cumsum, flip back
    H_cum = np.flip(np.flip(H_diff, axis=0).cumsum(axis=0), axis=0)
    H_cum = np.flip(np.flip(H_cum,  axis=1).cumsum(axis=1), axis=1)
    
    H_cum = H_cum *100 / H_diff.sum() if H_cum.sum() > 0 else H_cum
    H_diff = H_diff *dose_bins*let_bins / H_diff.sum() if H_diff.sum() > 0 else H_diff
    
    return H_diff, H_cum, d_edges, l_edges
