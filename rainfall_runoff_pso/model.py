"""
model.py — 3-Parameter Non-Linear Bucket Rainfall-Runoff Model.

The model transforms a precipitation time-series into a simulated discharge
series using three calibratable parameters:

    alpha (α)  — surface-runoff coefficient           [0.01 – 0.50]
    beta  (β)  — non-linear precipitation exponent    [1.00 – 3.00]
    k          — baseflow recession coefficient       [0.01 – 0.99]

Model equations (per daily timestep t):
    Q_surface(t) = α · P(t)^β          (non-linear surface runoff)
    Q_base(t)    = k · S(t-1)          (linear baseflow from storage)
    Q(t)         = Q_surface(t) + Q_base(t)
    S(t)         = max(0,  S(t-1) + P(t) - Q(t))
"""

import numpy as np

from config import PARAM_NAMES


def run_model(
    params: np.ndarray,
    precipitation: np.ndarray,
    initial_storage: float = 50.0,
) -> tuple[np.ndarray, float]:
    """
    Execute the bucket model over a full precipitation timeseries.

    Parameters
    ----------
    params : np.ndarray, shape (3,)
        Model parameters in order [alpha, beta, k].
    precipitation : np.ndarray, shape (n,)
        Daily precipitation in mm.
    initial_storage : float
        Soil-moisture storage at the start (mm).  Default 50 mm.

    Returns
    -------
    discharge : np.ndarray, shape (n,)
        Simulated daily discharge (combined surface + baseflow).
    final_storage : float
        Storage at the last timestep — pass this as initial_storage
        when running the model on the validation period to maintain
        hydrological continuity.
    """
    alpha, beta, k = params
    n = len(precipitation)

    discharge = np.zeros(n)
    S = initial_storage

    for t in range(n):
        P = precipitation[t]

        # Surface runoff: non-linear transformation of rainfall
        Q_surface = alpha * (P ** beta) if P > 0.0 else 0.0

        # Baseflow: fraction of current storage released per day
        Q_base = k * S

        # Total discharge for the day
        discharge[t] = Q_surface + Q_base

        # Update soil-moisture storage (mass balance)
        S = S + P - discharge[t]
        S = max(S, 0.0)

    return discharge, S


def params_dict_to_array(params_dict: dict) -> np.ndarray:
    """Convert a {name: value} dict to an ordered numpy array."""
    return np.array([params_dict[name] for name in PARAM_NAMES])


def params_array_to_dict(params_array: np.ndarray) -> dict:
    """Convert an ordered numpy array back to a {name: value} dict."""
    return {name: float(val) for name, val in zip(PARAM_NAMES, params_array)}
