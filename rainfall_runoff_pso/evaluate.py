"""
evaluate.py — Hydrological evaluation metrics.

Provides three standard metrics recommended by Moriasi et al. (2007):
    RMSE   – Root Mean Square Error
    NSE    – Nash-Sutcliffe Efficiency
    PBIAS  – Percent Bias
"""

import numpy as np


def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Root Mean Square Error.

        RMSE = √[ (1/n) · Σ (Qobs_i − Qsim_i)² ]

    Returns
    -------
    float — RMSE in the same units as input (e.g. m³/s).
    Lower is better; 0 is perfect.
    """
    return float(np.sqrt(np.mean((observed - simulated) ** 2)))


def nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.

        NSE = 1 − [ Σ (Qobs_i − Qsim_i)² ] / [ Σ (Qobs_i − Q̄obs)² ]

    Returns
    -------
    float — NSE ∈ (−∞, 1].
        1   = perfect model
        0   = model is as good as the mean of observations
       <0   = model is worse than the mean
    """
    numerator   = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Percent Bias.

        PBIAS = 100 · [ Σ (Qsim_i − Qobs_i) ] / [ Σ Qobs_i ]

    Returns
    -------
    float — PBIAS in percent.
        0%   = no bias
       >0%   = model over-predicts (wet bias)
       <0%   = model under-predicts (dry bias)
    """
    obs_sum = np.sum(observed)
    if obs_sum == 0.0:
        return float("nan")
    return float(100.0 * np.sum(simulated - observed) / obs_sum)


def evaluate_all(
    observed: np.ndarray, simulated: np.ndarray
) -> dict[str, float]:
    """Compute all three metrics and return as a dict."""
    return {
        "RMSE":  rmse(observed, simulated),
        "NSE":   nse(observed, simulated),
        "PBIAS": pbias(observed, simulated),
    }
