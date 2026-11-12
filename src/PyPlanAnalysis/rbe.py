"""
PyPlanAnalysis.rbe
==============
Variable and fixed RBE models for proton therapy.

All functions accept numpy arrays and return RBE-weighted dose arrays
of the same shape.

References
----------
McNamara   : McNamara et al., Med Phys 42(2):678-89, 2015
Wedenberg  : Wedenberg et al., Acta Oncol 52(3):580-8, 2013
Carabe     : Carabe et al., Br J Radiol 85(1011):304-14, 2012
"""

import numpy as np
from dataclasses import dataclass, field
import re

# ============================================================
# HELPERS FOR LINEAR MODEL
# ============================================================

_LINEAR_NUMERIC      = re.compile(r"^linear_(\d*\.?\d+)$")
_DEFAULT_SLOPE       = 0.1

def canonical_model_name(model: str) -> str:
    """Lower-case; normalise numeric slopes so 'linear_0.050' and 'linear_0.05' share one label."""
    name = model.lower().strip()
    m = _LINEAR_NUMERIC.match(name)
    return f"linear_{float(m.group(1)):g}" if m else name

def linear_slope(model: str):
    """Slope encoded in a linear model name, or None if `model` is not a linear model.
    "linear" -> 0.1 | "linear_<number>" (e.g. "linear_0.07")."""
    name = model.lower().strip()
    if name == "linear":
        return _DEFAULT_SLOPE
    if not name.startswith("linear_"):
        return None
    m = _LINEAR_NUMERIC.match(name)
    if m:
        return float(m.group(1))
    raise ValueError(f"Unknown linear RBE model '{model}'. Use 'linear_<slope>' such as 'linear_0.07'.")
    
def is_linear_model(model: str) -> bool:
    return linear_slope(model) is not None

# ============================================================
#  Configuration dataclass
# ============================================================

@dataclass
class RBEConfig:
    """
    Configure which RBE models to run.

    Parameters
    ----------
    models : list of str
        Any subset of {"mcnamara", "wedenberg", "carabe"}.
        Default: all 4.
    fixed_rbe : float
        Constant RBE value used for the RBE×fixed calculation.
        Default: 1.1
    """
    models    : list = field(default_factory=lambda: ["mcnamara",
                                                       "wedenberg",
                                                       "carabe",
                                                       "linear_0.1",
                                                       "linear_0.05"])
    fixed_rbe : float = 1.1
    
    slope_linear : list  =  field(default_factory=lambda: [0.1,0.05])

    def __post_init__(self):
        # normalise names and drop duplicates, keeping order
        self.models = list(dict.fromkeys(canonical_model_name(m) for m in self.models))
        bad = []
        for m in self.models:
            try:
                ok = m in VARIABLE_RBE_MODELS or linear_slope(m) is not None
            except ValueError:
                ok = False
            if not ok:
                bad.append(m)
        if bad:
            raise ValueError(
                f"Unknown RBE model(s): {bad}. Valid: "
                f"{[k for k in VARIABLE_RBE_MODELS if k != 'linear']}, "
                f"or 'linear_<slope>' (e.g. 'linear_0.07').")


# ============================================================
#  Fixed RBE
# ============================================================

def rbe_fixed(dose_phys: np.ndarray, rbe: float = 1.1) -> np.ndarray:
    """Return dose_phys * rbe (elementwise)."""
    return dose_phys * rbe


# ============================================================
#  Variable RBE models
# ============================================================

def _safe_dose(dose: np.ndarray) -> np.ndarray:
    """Avoid division by zero in dose arrays."""
    d = dose.copy()
    d[d <= 0] = 1e-9
    return d




def rbe_linear(dose_phys: np.ndarray,
                 let_d: np.ndarray,
                 n_fractions: int,
                 alpha_beta: float,
                 coeff: float = 0.1):
    """
    Linear variable-RBE model: RBE = 1 + coeff * LETd.

    Parameters
    ----------
    dose_phys   : physical dose array [Gy]
    let_d       : LETd array [keV/µm], same shape as dose_phys
    n_fractions : number of fractions (unused by this model, kept for a
                  consistent signature with the other RBE models)
    alpha_beta  : α/β ratio [Gy] (unused by this model)

    Returns
    -------
    np.ndarray  [Gy(RBE)]
    """
    rbe = 1 + coeff * let_d   
    return dose_phys * rbe

def rbe_mcnamara(dose_phys: np.ndarray,
                 let_d: np.ndarray,
                 n_fractions: int,
                 alpha_beta: float) -> np.ndarray:
    """
    McNamara et al. (2015) variable RBE model.
    Parameters taken from Table I of the original paper.
    """
    
    fraction_dose = dose_phys/n_fractions
    
    
    p1, p2, p3, p4 = 0.99064, 0.35605, 1.1012, -0.0038703
    ab = alpha_beta
    d = _safe_dose(fraction_dose)
    RBEmax = p1 + p2 * let_d / ab
    RBEmin = p3 + p4 * np.sqrt(ab) * let_d
    
    discriminant = ab**2 + 4.0 * ab * RBEmax * d + 4.0 * d**2 * RBEmin**2
    
    rbe_factor = (-ab + np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * d)
    
    
    return dose_phys * rbe_factor
    


def rbe_wedenberg(dose_phys: np.ndarray,
                  let_d: np.ndarray,
                  n_fractions: int,
                  alpha_beta: float) -> np.ndarray:
    """
    Wedenberg et al. (2013) variable RBE model.

    alpha/alpha_x = 1 + q * LETd / (alpha/beta)_x   with q = 0.434
    beta = beta_x  (constant)
    """
    fraction_dose = dose_phys/n_fractions
    q = 0.434
    ab = alpha_beta
    d = _safe_dose(fraction_dose)
    RBE_max = 1.0 + q * let_d / ab
    discriminant = ab**2 + 4.0 * ab * RBE_max * d + 4.0 * d**2
    rbe = (-ab + np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * d)
        
    return dose_phys * rbe


def rbe_carabe(dose_phys: np.ndarray,
               let_d: np.ndarray,
               n_fractions: int,
               alpha_beta: float) -> np.ndarray:
    """
    Carabe et al. (2012) variable RBE model.

    p=0.843, r=0.154, s=1.09, t=0.006
    """
    fraction_dose = dose_phys/n_fractions
    p1, p2, p3, p4 = 0.843, 0.154, 1.09, 0.006
    AB = 2.686
    ab = alpha_beta
    d = _safe_dose(fraction_dose)
    RBE_max = p1 + p2 * AB * let_d / ab
    RBE_min = p3 + p4 * AB * let_d / ab
    discriminant = ab**2 + 4.0 * ab * RBE_max * d + 4.0 * d**2 * RBE_min**2
    rbe =  (-ab + np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * d)

    return dose_phys * rbe


# ============================================================
#  Registry  –  makes it easy to add new models later
# ============================================================

VARIABLE_RBE_MODELS = {
    "mcnamara" : rbe_mcnamara,
    "wedenberg": rbe_wedenberg,
    "carabe"   : rbe_carabe,
    "linear"   : rbe_linear
}


def compute_rbe_dose(dose_phys: np.ndarray,
                     let_d: np.ndarray,
                     n_fractions: int,
                     alpha_beta: float,
                     model: str) -> np.ndarray:
    """
    Compute variable RBE-weighted dose for a single model.

    Parameters
    ----------
    dose_phys  : physical dose array [Gy]
    let_d      : LETd array [keV/µm], same shape as dose_phys
    n_fractions: int (not needed for linear)
    alpha_beta : α/β ratio [Gy] for this structure(not needed for linear), both int and array (numpy)
    model      : one of "linear","mcnamara", "wedenberg", "carabe"

    Returns
    -------
    rbe_dose : np.ndarray  [Gy(RBE)]
    """
    if np.ndim(alpha_beta) > 0 and np.shape(alpha_beta) != np.shape(dose_phys):
        raise ValueError("alpha_beta array must match dose shape")
    
    name  = canonical_model_name(model)
    slope = linear_slope(name)
    if slope is not None:
        return rbe_linear(dose_phys, let_d, n_fractions, alpha_beta, coeff=slope)
    
    if name not in VARIABLE_RBE_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(VARIABLE_RBE_MODELS)} "
                         "or a linear model ('linear_<preset>' / 'linear_<slope>').")
    return VARIABLE_RBE_MODELS[name](dose_phys, let_d, n_fractions, alpha_beta)





