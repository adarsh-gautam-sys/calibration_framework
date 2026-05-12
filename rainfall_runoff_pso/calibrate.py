"""
calibrate.py — PSO-driven model calibration.

Reads the raw data, splits it 70/30 (calibration / validation),
builds an objective function that wraps the bucket model + NSE,
runs PSO, and returns the calibrated parameters.
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
)
from model import simulate_runoff
from pso import ParticleSwarmOptimizer
from evaluate import nse


def load_and_split(
    filepath: str = RAW_DATA_FILE,
    cal_ratio: float = CALIBRATION_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the CSV and perform a temporal train/test split.

    Returns
    -------
    df_cal, df_val : pd.DataFrame
        Calibration and validation subsets (contiguous, no shuffle).
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.dropna().reset_index(drop=True)
    split_idx = int(len(df) * cal_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_objective(precip_cal: np.ndarray, obs_cal: np.ndarray):
    """
    Return a closure that PSO can call:  objective(params_dict) → float.

    The objective function receives a **dict** ``{alpha, beta, k}``
    (matching the ParticleSwarmOptimizer interface) and returns a
    scalar cost = 1 − NSE.

        - When NSE = 1 (perfect)  →  cost = 0
        - When NSE = 0            →  cost = 1
    """
    def objective(params: dict) -> float:
        sim = simulate_runoff(precip_cal, params)
        score = nse(obs_cal, sim)
        # Guard against NaN (e.g. all-zero observed)
        if np.isnan(score):
            return 1.0
        return 1.0 - score
    return objective


def calibrate(verbose: bool = True) -> dict:
    """
    Run the full calibration pipeline.

    Returns
    -------
    dict with keys:
        best_params       — dict  {alpha, beta, k}
        best_params_array — list  [alpha, beta, k]  (PARAM_NAMES order)
        objective_value   — float (1 - NSE on calibration set)
        nse_cal           — float (NSE on calibration set)
        history           — list  (best cost per PSO iteration)
        elapsed_seconds   — float
    """
    # ── Load & split ──────────────────────────────────────────────────
    df_cal, _ = load_and_split()
    precip_cal = df_cal["Precipitation_mm"].values
    obs_cal    = df_cal["Observed_Discharge_m3s"].values

    if verbose:
        print(f"\n{'='*60}")
        print(f" CALIBRATION  ({len(df_cal)} days)")
        print(f"{'='*60}")
        print(f" PSO config: {PSO_CONFIG}")
        print(f" Bounds:     {MODEL_PARAM_BOUNDS}")

    # ── Build objective & run PSO ─────────────────────────────────────
    objective = build_objective(precip_cal, obs_cal)

    pso = ParticleSwarmOptimizer(
        objective_fn=objective,
        bounds=MODEL_PARAM_BOUNDS,
        **PSO_CONFIG,
    )

    t0 = time.perf_counter()
    best_params, best_cost, history = pso.optimize()
    elapsed = time.perf_counter() - t0

    # ── Re-run model with best params for metrics ─────────────────────
    sim_cal = simulate_runoff(precip_cal, best_params)
    nse_cal = nse(obs_cal, sim_cal)

    result = {
        "best_params":       best_params,
        "best_params_array": [best_params[p] for p in PARAM_NAMES],
        "objective_value":   best_cost,
        "nse_cal":           nse_cal,
        "history":           history,
        "elapsed_seconds":   round(elapsed, 2),
    }

    # ── Persist to JSON ───────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "calibration_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"\n Calibrated params: {best_params}")
        print(f" NSE (cal):  {nse_cal:.4f}")
        print(f" Time:       {elapsed:.1f}s")
        print(f" Saved to:   {json_path}")

    return result
