"""
PyPlanAnalysis.plots
===================
Matplotlib plotting helpers for DVH/LVH curves and 2-D/3-D DLVH maps.
"""

import numpy as np
import matplotlib
from typing import Union
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# ============================================================
#  Shared style helpers
# ============================================================

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _cycle_color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


# ============================================================
#  DVH / LVH curve plots
# ============================================================

def plot_cumulative_histogram(curves: dict,
                              title: str,
                              xlabel: str = "Dose [Gy(RBE)]",
                              ylabel: str = "Volume fraction",
                              dpi: int = 150) -> plt.Figure:
    """
    Plot cumulative DVH or LVH curves for multiple structures.

    Parameters
    ----------
    curves : dict  {structure_name: (edges, cum_volume_fraction)}
    title  : plot title string
    xlabel : x-axis label
    ylabel : y-axis label
    dpi    : figure DPI

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    for i, (name, (edges, cum)) in enumerate(curves.items()):
        ax.plot(edges, cum,
                label=name,
                color=_cycle_color(i),
                linewidth=1.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right",
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_xlim(left=0)
    if "LET" in xlabel:
        ax.set_xlim (0,25)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_dvh_comparison(curves_per_model: dict,
                        structure_name: str,
                        dpi: int = 150) -> plt.Figure:
    """
    Overlay DVH curves for multiple dose types on one structure.

    Parameters
    ----------
    curves_per_model : dict  {model_label: (edges, cum)}
                       e.g. {"Physical": (...), "RBE1.1": (...),
                              "McNamara": (...)}
    structure_name   : used in the plot title

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)

    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (label, (edges, cum)) in enumerate(curves_per_model.items()):
        ax.plot(edges, cum,
                label=label,
                color=_cycle_color(i),
                linestyle=styles[i % len(styles)],
                linewidth=2.0)

    ax.set_xlabel("Dose [Gy / Gy(RBE)]", fontsize=12)
    ax.set_ylabel("Volume fraction", fontsize=12)
    ax.set_title(f"DVH comparison – {structure_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================
#  2-D DLVH
# ============================================================

def plot_dlvh_2d(H: np.ndarray,
                 dose_edges: np.ndarray,
                 let_edges: np.ndarray,
                 structure_name: str,
                 dpi: int = 150) -> plt.Figure:
    """
    2-D dose-LET volume histogram (DLVH) as a colourmap.

    Parameters
    ----------
    H             : 2-D array (dose_bins × let_bins), volume fractions
    dose_edges    : dose bin edges [Gy(RBE)]
    let_edges     : LET bin edges [keV/µm]
    structure_name: used in title
    dpi           : figure DPI

    Returns
    -------
    matplotlib Figure
    """
    h_pos = H[H > 0]
    vmin  = h_pos.min() if len(h_pos) > 0 else 1e-6
    vmax  = H.max()     if H.max()  > 0    else 1.0


    #colormap
    n_levels   = 11                              
    boundaries = np.linspace(0, 100, n_levels)  
    base_cmap  = plt.colormaps["jet"].resampled(n_levels - 1) #plt.cm.get_cmap("jet", n_levels - 1)
    cmap       = mcolors.BoundaryNorm(boundaries, base_cmap.N)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    pcm = ax.pcolormesh(dose_edges, let_edges, H.T,
                        norm=cmap ,#mcolors.LogNorm(vmin=vmin, vmax=vmax),
                        cmap=base_cmap,
                        shading="auto")
    

    cb = fig.colorbar(pcm, ax=ax, pad=0.02,
                     ticks=boundaries,
                     boundaries=boundaries)
   
    #cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Volume [%]", fontsize=11)
    ax.set_xlabel("Dose [Gy(RBE)] - fixed RBE", fontsize=12)
    ax.set_ylabel("LETd [keV/μm]",  fontsize=12)
    ax.set_title(f"DLVH – {structure_name}",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_dlvh_3d(H_pct, d_edges, l_edges, structure_name, dpi=150):
    """
    3D surface plot of the cumulative DLVH.
    Z axis = volume [%], X = dose, Y = LET.
    """
    from mpl_toolkits.mplot3d import Axes3D

    d_centres = (d_edges[:-1] + d_edges[1:]) / 2
    l_centres = (l_edges[:-1] + l_edges[1:]) / 2
    D, L = np.meshgrid(d_centres, l_centres)   # both shape (let_bins, dose_bins)

    fig = plt.figure(figsize=(10, 7), dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(D, L, H_pct.T,         # .T because meshgrid is (LET, dose)
                           cmap="jet",
                           vmin=0, vmax=100,
                           linewidth=0,
                           antialiased=True,
                           alpha=0.9)

    cb = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.5)
    cb.set_label("Volume [%]", fontsize=10)

    ax.set_xlabel("Dose [Gy(RBE)]", fontsize=10, labelpad=8)
    ax.set_ylabel("LETd [keV/μm]",  fontsize=10, labelpad=8)
    ax.set_zlabel("Volume [%]",     fontsize=10, labelpad=8)
    ax.set_zlim(0, 100)
    ax.set_title(f"DLVH 3D – {structure_name}", fontsize=12, fontweight="bold")

    # good default viewing angle
    ax.view_init(elev=30, azim=25)

    fig.tight_layout()
    return fig
# ============================================================
#  Multi-page PDF helpers
# ============================================================

def save_figures_to_pdf(figures: list, path: Union[str, Path], dpi: int = 150):
    """
    Save a list of matplotlib Figures to a multi-page PDF.

    Parameters
    ----------
    figures : list of (title_str, Figure) or just [Figure, ...]
    path    : output PDF path
    """
    with PdfPages(str(path)) as pdf:
        for item in figures:
            fig = item[1] if isinstance(item, tuple) else item
            pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
            plt.close(fig)


def save_figure(fig: plt.Figure,
                path: Union[str, Path],
                dpi: int = 150):
    """Save a single figure and close it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
