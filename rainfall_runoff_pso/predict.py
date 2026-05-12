"""
predict.py — Validation / prediction using calibrated parameters.

Takes the best parameters from calibration, runs the bucket model on the
held-out validation period, and computes evaluation metrics.
"""

import json
import os

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from model import simulate_runoff
from evaluate import evaluate_all
from calibrate import load_and_split


def predict(
    cal_result: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the bucket model on the validation period with calibrated params.

    Parameters
    ----------
    cal_result : dict or None
        Output of ``calibrate.calibrate()``.  If None, loads from
        ``results/calibration_results.json``.

    Returns
    -------
    dict with keys:
        sim_val     — list   (simulated discharge on val period)
        obs_val     — list   (observed discharge on val period)
        dates_val   — list   (ISO date strings)
        metrics_val — dict   {RMSE, NSE, PBIAS}
        best_params — dict   {alpha, beta, k}
    """
    # ── Load calibration results if not passed directly ───────────────
    if cal_result is None:
        json_path = os.path.join(RESULTS_DIR, "calibration_results.json")
        with open(json_path) as f:
            cal_result = json.load(f)

    best_params = cal_result["best_params"]       # dict {alpha, beta, k}

    # ── Load validation data ──────────────────────────────────────────
    _, df_val = load_and_split()
    precip_val = df_val["Precipitation_mm"].values
    obs_val    = df_val["Observed_Discharge_m3s"].values
    dates_val  = df_val["Date"].dt.strftime("%Y-%m-%d").tolist()

    # ── Run model with calibrated parameters ──────────────────────────
    sim_val = simulate_runoff(precip_val, best_params)

    # ── Evaluate ──────────────────────────────────────────────────────
    metrics = evaluate_all(obs_val, sim_val)

    result = {
        "sim_val":     sim_val.tolist(),
        "obs_val":     obs_val.tolist(),
        "dates_val":   dates_val,
        "metrics_val": metrics,
        "best_params": best_params,
    }

    # ── Persist ───────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "validation_results.json")
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in result.items()
                   if k not in ("sim_val", "obs_val")}, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f" VALIDATION  ({len(df_val)} days)")
        print(f"{'='*60}")
        print(f" Params: {best_params}")
        for k, v in metrics.items():
            print(f" {k:>6s}: {v:.4f}")
        print(f" Saved to: {out_path}")

    return result
