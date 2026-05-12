"""
model.py — 3-Parameter Non-Linear Bucket Rainfall-Runoff Model.

Converts a daily precipitation time-series into simulated discharge using
three calibratable parameters:

    alpha (α)  — runoff coefficient               [0.01 – 0.50]
    beta  (β)  — non-linearity exponent            [1.00 – 3.00]
    k          — recession constant                [0.01 – 0.99]

Model equations (per daily timestep t):
    EffectiveRainfall(t) = α · P(t)^β
    Discharge(t)         = k · Discharge(t-1)  +  (1 - k) · EffectiveRainfall(t)

The recession formula produces a smooth hydrograph where:
    - k close to 1  → slow recession, sustained baseflow
    - k close to 0  → flashy response, almost no memory
"""

import numpy as np

from config import PARAM_NAMES, MODEL_PARAM_BOUNDS


# =========================================================================
# Core model
# =========================================================================

def simulate_runoff(precipitation: np.ndarray, params: dict) -> np.ndarray:
    """
    Simple bucket-style rainfall-runoff model.

    Parameters
    ----------
    precipitation : np.ndarray, shape (n,)
        Daily precipitation in mm (non-negative).
    params : dict
        Model parameters with keys:
            alpha (float) : runoff coefficient          [0.01, 0.5]
            beta  (float) : nonlinearity exponent       [1.0,  3.0]
            k     (float) : recession constant          [0.01, 0.99]

    Returns
    -------
    simulated_discharge : np.ndarray, shape (n,)
        Simulated daily discharge (same length as precipitation).

    Notes
    -----
    • Effective rainfall is computed vectorised via ``np.power`` and
      ``np.clip`` (no Python loop for that step).
    • The recession filter requires a sequential scan — the loop is
      kept minimal (one multiply-add per timestep).
    • Negative values are clipped to 0 at every stage.
    """
    alpha = params["alpha"]
    beta  = params["beta"]
    k     = params["k"]

    n = len(precipitation)

    # ── Step 1: Effective rainfall (vectorised) ───────────────────────
    #   Clip precip to ≥ 0 first, then raise to power β, scale by α
    precip_safe = np.clip(precipitation, 0.0, None)            # no negatives
    effective_rainfall = alpha * np.power(precip_safe, beta)    # α · P^β
    effective_rainfall = np.clip(effective_rainfall, 0.0, None) # safety clip

    # ── Step 2: Recession filter (sequential — depends on Q(t-1)) ────
    #   Q(t) = k · Q(t-1) + (1 - k) · EffRain(t)
    discharge = np.zeros(n, dtype=np.float64)
    one_minus_k = 1.0 - k

    for t in range(n):
        if t == 0:
            discharge[t] = one_minus_k * effective_rainfall[t]
        else:
            discharge[t] = k * discharge[t - 1] + one_minus_k * effective_rainfall[t]

    # ── Step 3: Final safety clip — discharge cannot be negative ──────
    discharge = np.clip(discharge, 0.0, None)

    return discharge


# =========================================================================
# Convenience helpers  (used by calibrate.py / predict.py)
# =========================================================================

def params_dict_to_array(params_dict: dict) -> np.ndarray:
    """Convert ``{name: value}`` to an ordered numpy array (same order as
    :data:`config.PARAM_NAMES`)."""
    return np.array([params_dict[name] for name in PARAM_NAMES])


def params_array_to_dict(params_array: np.ndarray) -> dict:
    """Convert an ordered numpy array back to ``{name: value}`` dict."""
    return {name: float(val) for name, val in zip(PARAM_NAMES, params_array)}


def get_bounds_arrays():
    """Return ``(lower, upper)`` numpy arrays in :data:`PARAM_NAMES` order."""
    lower = np.array([MODEL_PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    upper = np.array([MODEL_PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
    return lower, upper
