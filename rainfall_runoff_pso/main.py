"""
main.py — Entry point for the PSO Rainfall-Runoff Calibration Framework.

Orchestrates the full pipeline in order:
    1. Load and preprocess data
    2. Split into calibration / validation
    3. Run calibration → get best_params + convergence history
    4. Run prediction on validation set
    5. Evaluate and print metrics
    6. Plot all results (3-subplot figure)
    7. Print final summary table

Each step is timed individually.

Usage
-----
    cd rainfall_runoff_pso
    pip install -r requirements.txt
    python main.py
"""

import os
import time

import numpy as np
import pandas as pd

from config import (
    RAW_DATA_FILE,
    RESULTS_DIR,
    TRUE_PARAMS,
    PARAM_NAMES,
    CALIBRATION_RATIO,
)
from data.download import download_data
from calibrate import run_calibration, load_and_split
from predict import run_prediction
from model import simulate_runoff
from evaluate import rmse, nash_sutcliffe_efficiency, print_metrics
from visualize import plot_all_results


def _timer(label: str):
    """Context manager that prints elapsed time for a pipeline step."""
    class _T:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self._t0 = time.time()
            return self
        def __exit__(self, *_):
            self.elapsed = time.time() - self._t0
            print(f"  ⏱  {label} completed in {self.elapsed:.2f}s")
    return _T()


def main() -> None:
    """Run the complete calibration → prediction → evaluation pipeline."""

    timings: dict[str, float] = {}

    # ==================================================================
    # Step 1: Load and preprocess data
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 1 — Load & Preprocess Data")
    print("=" * 60)

    with _timer("Data loading") as t:
        if not os.path.exists(RAW_DATA_FILE):
            download_data(force_synthetic=True)
        else:
            print(f"  Data already exists at {RAW_DATA_FILE}")

        df_full = pd.read_csv(RAW_DATA_FILE, parse_dates=["Date"])
        df_full = df_full.dropna().reset_index(drop=True)
        print(f"  Loaded {len(df_full)} rows, columns: {list(df_full.columns)}")
    timings["1. Data loading"] = t.elapsed

    # ==================================================================
    # Step 2: Split into calibration / validation
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 2 — Train / Validation Split")
    print("=" * 60)

    with _timer("Data splitting") as t:
        df_cal, df_val = load_and_split()
        split_idx = len(df_cal)

        precip_cal = df_cal["Precipitation_mm"].values
        obs_cal    = df_cal["Observed_Discharge_m3s"].values
        precip_val = df_val["Precipitation_mm"].values
        obs_val    = df_val["Observed_Discharge_m3s"].values

        print(f"  Calibration : {len(df_cal)} days  "
              f"({df_cal['Date'].iloc[0].date()} → {df_cal['Date'].iloc[-1].date()})")
        print(f"  Validation  : {len(df_val)} days  "
              f"({df_val['Date'].iloc[0].date()} → {df_val['Date'].iloc[-1].date()})")
    timings["2. Data splitting"] = t.elapsed

    # ==================================================================
    # Step 3: Run calibration (PSO)
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 3 — PSO Calibration")
    print("=" * 60)

    with _timer("PSO calibration") as t:
        best_params, pso_history = run_calibration(precip_cal, obs_cal)
    timings["3. PSO calibration"] = t.elapsed

    # ==================================================================
    # Step 4: Run prediction on validation set
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 4 — Validation Prediction")
    print("=" * 60)

    with _timer("Prediction") as t:
        sim_val = run_prediction(precip_val, best_params)
    timings["4. Prediction"] = t.elapsed

    # ==================================================================
    # Step 5: Evaluate and print metrics
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 5 — Evaluation")
    print("=" * 60)

    with _timer("Evaluation") as t:
        # ── Calibration metrics ───────────────────────────────────────
        sim_cal = simulate_runoff(precip_cal, best_params)
        print("\n  — Calibration Period —")
        metrics_cal = print_metrics(obs_cal, sim_cal)

        # ── Validation metrics ────────────────────────────────────────
        print("\n  — Validation Period —")
        metrics_val = print_metrics(obs_val, sim_val)
    timings["5. Evaluation"] = t.elapsed

    # ==================================================================
    # Step 6: Plot all results
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 6 — Visualisation")
    print("=" * 60)

    with _timer("Plotting") as t:
        all_dates  = pd.to_datetime(
            pd.concat([df_cal["Date"], df_val["Date"]], ignore_index=True)
        )
        all_obs    = np.concatenate([obs_cal, obs_val])
        all_sim    = np.concatenate([sim_cal, sim_val])

        plot_all_results(
            dates=all_dates,
            obs=all_obs,
            sim=all_sim,
            pso_history=pso_history,
            split_idx=split_idx,
        )
    timings["6. Plotting"] = t.elapsed

    # ==================================================================
    # Step 7: Final summary table
    # ==================================================================
    print("\n" + "=" * 60)
    print(" FINAL SUMMARY")
    print("=" * 60)

    # Parameters comparison
    print(f"\n  True parameters       : {TRUE_PARAMS}")
    print(f"  Calibrated parameters : {best_params}")

    # Parameter recovery accuracy
    print(f"\n  {'Parameter':<10s}  {'True':>8s}  {'Calibrated':>10s}  {'Error %':>8s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}")
    for name in PARAM_NAMES:
        true_val = TRUE_PARAMS[name]
        cal_val  = best_params[name]
        err_pct  = abs(cal_val - true_val) / true_val * 100
        print(f"  {name:<10s}  {true_val:>8.4f}  {cal_val:>10.4f}  {err_pct:>7.1f}%")

    # Metrics comparison
    print(f"\n  {'Metric':<8s}  {'Calibration':>12s}  {'Validation':>12s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}")
    for key in ["RMSE", "NSE"]:
        c = metrics_cal.get(key, float("nan"))
        v = metrics_val.get(key, float("nan"))
        print(f"  {key:<8s}  {c:>12.4f}  {v:>12.4f}")

    # Timing summary
    total = sum(timings.values())
    print(f"\n  {'Step':<25s}  {'Time (s)':>10s}")
    print(f"  {'─'*25}  {'─'*10}")
    for step, sec in timings.items():
        print(f"  {step:<25s}  {sec:>10.2f}")
    print(f"  {'─'*25}  {'─'*10}")
    print(f"  {'TOTAL':<25s}  {total:>10.2f}")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Files generated:")
    for root, _, files in os.walk(RESULTS_DIR):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), RESULTS_DIR)
            print(f"    • {rel}")
    print()


if __name__ == "__main__":
    main()
