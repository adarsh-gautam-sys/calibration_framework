"""
calibrate.py — PSO-driven model calibration.

Provides ``run_calibration(precip_train, obs_discharge_train)`` which:
1. Defines the objective function (RMSE between simulated and observed).
2. Wraps it for ParticleSwarmOptimizer (dict-based interface).
3. Runs PSO with hyperparameters from config.py.
4. Saves best parameters to results/calibrated_params.json.
5. Returns (best_params dict, convergence history list).
"""

import json
import os
import time

import numpy as np
import pandas as pd

from config import (
    CALIBRATION_RATIO,
    MODEL_PARAM_BOUNDS,
    PARAM_NAMES,
    PSO_CONFIG,
    RAW_DATA_FILE,
    RESULTS_DIR,
    CALIBRATED_PARAMS_FILE,
)
from model import simulate_runoff
from pso import ParticleSwarmOptimizer
from evaluate import rmse


# =========================================================================
# Data loading  (shared with main.py)
# =========================================================================

def load_and_split(
    filepath: str = RAW_DATA_FILE,
    cal_ratio: float = CALIBRATION_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the CSV and perform a temporal train/test split.

    Returns
    -------
    df_cal, df_val : pd.DataFrame
        Calibration (first ``cal_ratio`` %) and validation subsets.
        No shuffling — split is purely temporal.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.dropna().reset_index(drop=True)
    split_idx = int(len(df) * cal_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# =========================================================================
# Core calibration routine
# =========================================================================

def run_calibration(
    precip_train: np.ndarray,
    obs_discharge_train: np.ndarray,
) -> tuple[dict, list[float]]:
    """
    Calibrate the rainfall-runoff model via PSO.

    Parameters
    ----------
    precip_train : np.ndarray
        Precipitation for the calibration period (mm/day).
    obs_discharge_train : np.ndarray
        Observed discharge for the calibration period (m³/s).

    Returns
    -------
    best_params : dict
        ``{"alpha": float, "beta": float, "k": float}``
    history : list[float]
        Best RMSE cost at each PSO iteration.
    """

    # ── Objective: RMSE between simulated and observed ────────────────
    def objective(params: dict) -> float:
        """PSO minimises this.  Lower RMSE = better fit."""
        sim = simulate_runoff(precip_train, params)
        return rmse(obs_discharge_train, sim)

    # ── Instantiate PSO ───────────────────────────────────────────────
    pso = ParticleSwarmOptimizer(
        objective_fn=objective,
        bounds=MODEL_PARAM_BOUNDS,
        **PSO_CONFIG,
    )

    # ── Run optimisation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" CALIBRATION  ({len(precip_train)} days)")
    print(f"{'='*60}")
    print(f" PSO config : {PSO_CONFIG}")
    print(f" Bounds     : {MODEL_PARAM_BOUNDS}")
    print()

    best_params, best_cost, history = pso.optimize()

    print(f"\n Best RMSE  : {best_cost:.6f}")
    print(f" Best params: {best_params}")

    # ── Save to JSON ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CALIBRATED_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f" Saved to   : {CALIBRATED_PARAMS_FILE}")

    return best_params, history
