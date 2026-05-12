"""
predict.py — Validation / prediction using calibrated parameters.

Provides ``run_prediction(precip_validation, best_params)`` which runs the
bucket model on unseen data and returns simulated discharge.
"""

import json

import numpy as np

from config import CALIBRATED_PARAMS_FILE
from model import simulate_runoff


def run_prediction(
    precip_validation: np.ndarray,
    best_params: dict | None = None,
) -> np.ndarray:
    """
    Run the rainfall-runoff model on the validation period.

    Parameters
    ----------
    precip_validation : np.ndarray
        Precipitation for the validation period (mm/day).
    best_params : dict or None
        ``{"alpha": float, "beta": float, "k": float}``.
        If None, loads from ``results/calibrated_params.json``.

    Returns
    -------
    simulated_discharge : np.ndarray
        Model-predicted discharge for the validation period.
    """
    # ── Load calibrated params from disk if not provided ──────────────
    if best_params is None:
        with open(CALIBRATED_PARAMS_FILE) as f:
            best_params = json.load(f)
        print(f"  [predict] Loaded params from {CALIBRATED_PARAMS_FILE}")

    # ── Run model ─────────────────────────────────────────────────────
    simulated_discharge = simulate_runoff(precip_validation, best_params)

    print(f"  [predict] Simulated {len(simulated_discharge)} days "
          f"with params {best_params}")

    return simulated_discharge
