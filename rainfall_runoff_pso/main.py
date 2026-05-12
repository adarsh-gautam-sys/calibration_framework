"""
main.py — Entry point for the PSO Rainfall-Runoff Calibration Framework.

Pipeline
--------
1. Download / generate data           (data/download.py)
2. Calibrate model via PSO             (calibrate.py)
3. Predict on validation period        (predict.py)
4. Generate all evaluation plots       (visualize.py)
5. Print summary report

Usage
-----
    cd rainfall_runoff_pso
    pip install -r requirements.txt
    python main.py
"""

import os
import sys
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
from calibrate import calibrate, load_and_split
from predict import predict
from model import run_model
from evaluate import evaluate_all
from visualize import (
    plot_hydrograph,
    plot_scatter,
    plot_convergence,
    plot_rainfall_runoff,
)


def main() -> None:
    """Run the complete calibration → prediction → visualisation pipeline."""

    # ==================================================================
    # Step 1: Acquire data
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 1 — Data Acquisition")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_FILE):
        download_data(force_synthetic=True)
    else:
        print(f"  Data already exists at {RAW_DATA_FILE}, skipping download.")

    # ==================================================================
    # Step 2: Calibration (PSO)
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 2 — PSO Calibration")
    print("=" * 60)

    cal_result = calibrate(verbose=True)

    # ==================================================================
    # Step 3: Prediction (Validation)
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 3 — Validation Prediction")
    print("=" * 60)

    val_result = predict(cal_result=cal_result, verbose=True)

    # ==================================================================
    # Step 4: Plots
    # ==================================================================
    print("\n" + "=" * 60)
    print(" STEP 4 — Generating Plots")
    print("=" * 60)

    # Load full data for plotting
    df_cal, df_val = load_and_split()

    precip_cal = df_cal["Precipitation_mm"].values
    obs_cal    = df_cal["Observed_Discharge_m3s"].values
    dates_cal  = pd.to_datetime(df_cal["Date"])

    precip_val = df_val["Precipitation_mm"].values
    obs_val    = np.array(val_result["obs_val"])
    sim_val    = np.array(val_result["sim_val"])
    dates_val  = pd.to_datetime(val_result["dates_val"])

    # Re-run model on cal period for cal plots
    best_params = np.array(cal_result["best_params_array"])
    sim_cal, _ = run_model(best_params, precip_cal)
    metrics_cal = evaluate_all(obs_cal, sim_cal)
    metrics_val = val_result["metrics_val"]

    # 4a. Hydrograph — Calibration
    plot_hydrograph(
        dates_cal, obs_cal, sim_cal,
        title="Hydrograph — Calibration Period",
        metrics=metrics_cal,
        filename="hydrograph_calibration.png",
    )

    # 4b. Hydrograph — Validation
    plot_hydrograph(
        dates_val, obs_val, sim_val,
        title="Hydrograph — Validation Period",
        metrics=metrics_val,
        filename="hydrograph_validation.png",
    )

    # 4c. Scatter plot (validation)
    plot_scatter(
        obs_val, sim_val,
        title="Scatter Plot — Validation Period",
        filename="scatter_plot.png",
    )

    # 4d. Convergence curve
    plot_convergence(
        cal_result["history"],
        filename="convergence_curve.png",
    )

    # 4e. Rainfall-runoff overview (full dataset)
    all_dates  = pd.to_datetime(pd.concat([df_cal["Date"], df_val["Date"]]))
    all_precip = np.concatenate([precip_cal, precip_val])
    all_obs    = np.concatenate([obs_cal, obs_val])
    all_sim    = np.concatenate([sim_cal, sim_val])
    plot_rainfall_runoff(
        all_dates, all_precip, all_obs, all_sim,
        filename="rainfall_runoff_overview.png",
    )

    # ==================================================================
    # Step 5: Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  True parameters:       {TRUE_PARAMS}")
    print(f"  Calibrated parameters: {cal_result['best_params']}")
    print(f"\n  {'Metric':<8s}  {'Calibration':>12s}  {'Validation':>12s}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}")
    for key in ["RMSE", "NSE", "PBIAS"]:
        c = metrics_cal.get(key, float("nan"))
        v = metrics_val.get(key, float("nan"))
        print(f"  {key:<8s}  {c:>12.4f}  {v:>12.4f}")
    print(f"\n  PSO runtime: {cal_result['elapsed_seconds']:.1f}s")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"\n  Plots generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"    • {f}")
    print()


if __name__ == "__main__":
    main()
