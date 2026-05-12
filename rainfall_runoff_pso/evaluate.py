"""
evaluate.py — Hydrological evaluation metrics.

Provides the two standard metrics for rainfall-runoff model assessment:
    RMSE  – Root Mean Square Error
    NSE   – Nash-Sutcliffe Efficiency

Plus a convenience printer.
"""

import numpy as np


def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Root Mean Square Error.

        RMSE = √[ (1/n) · Σ (Qobs_i − Qsim_i)² ]

    Parameters
    ----------
    observed  : np.ndarray — measured discharge
    simulated : np.ndarray — model-predicted discharge

    Returns
    -------
    float — RMSE value (same units as input).  Lower is better; 0 is perfect.
    """
    return float(np.sqrt(np.mean((observed - simulated) ** 2)))


def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE).

        NSE = 1 − [ Σ (Qobs_i − Qsim_i)² ] / [ Σ (Qobs_i − Q̄obs)² ]

    Parameters
    ----------
    observed  : np.ndarray — measured discharge
    simulated : np.ndarray — model-predicted discharge

    Returns
    -------
    float — NSE ∈ (−∞, 1].
        1.0  = perfect model
        0.0  = model equals the mean of observations
       <0.0  = model is worse than the mean

    Interpretation (Moriasi et al., 2007):
        NSE > 0.75  →  "Very good"
        NSE > 0.65  →  "Good"
        NSE > 0.50  →  "Satisfactory"
    """
    numerator   = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def print_metrics(observed: np.ndarray, simulated: np.ndarray) -> dict:
    """
    Compute and print RMSE and NSE in a formatted table.

    Parameters
    ----------
    observed  : np.ndarray — measured discharge
    simulated : np.ndarray — model-predicted discharge

    Returns
    -------
    dict — ``{"RMSE": float, "NSE": float}``
    """
    r = rmse(observed, simulated)
    n = nash_sutcliffe_efficiency(observed, simulated)

    metrics = {"RMSE": r, "NSE": n}

    print(f"\n  {'Metric':<8s}  {'Value':>12s}")
    print(f"  {'-' * 8}  {'-' * 12}")
    print(f"  {'RMSE':<8s}  {r:>12.4f}")
    print(f"  {'NSE':<8s}  {n:>12.4f}")

    # Quality label
    if n >= 0.75:
        label = "Very good"
    elif n >= 0.65:
        label = "Good"
    elif n >= 0.50:
        label = "Satisfactory"
    else:
        label = "Unsatisfactory"
    print(f"\n  Model rating: {label}  (NSE = {n:.4f})")

    return metrics
