"""
visualize.py — Publication-quality hydrological plots.

Generates five plots:
    1. Hydrograph (calibration period)
    2. Hydrograph (validation period)
    3. Scatter plot with 1:1 line
    4. PSO convergence curve
    5. Rainfall-runoff overview (dual-axis)
"""

import os

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR, PLOT_DPI, FIG_SIZE


# ─── Colour palette ──────────────────────────────────────────────────────
C_OBS  = "#1a73e8"      # Google Blue
C_SIM  = "#ea4335"      # Google Red
C_RAIN = "#5f6368"      # Neutral grey
C_FILL = "#e8f0fe"      # Light blue fill


def _save(fig, filename: str) -> str:
    path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [plot] Saved {path}")
    return path


# =====================================================================
# 1 & 2.  Hydrograph
# =====================================================================

def plot_hydrograph(
    dates,
    observed: np.ndarray,
    simulated: np.ndarray,
    title: str,
    metrics: dict,
    filename: str,
) -> str:
    """Overlay observed vs simulated discharge with metrics annotation."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(dates, observed,  color=C_OBS, linewidth=1.2, label="Observed",  alpha=0.85)
    ax.plot(dates, simulated, color=C_SIM, linewidth=1.0, label="Simulated", alpha=0.85)
    ax.fill_between(dates, observed, simulated, color=C_FILL, alpha=0.35, label="Residual")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Discharge (m³/s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    # Metrics box
    txt = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    ax.text(
        0.02, 0.95, txt, transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    fig.autofmt_xdate()
    return _save(fig, filename)


# =====================================================================
# 3.  Scatter plot
# =====================================================================

def plot_scatter(
    observed: np.ndarray,
    simulated: np.ndarray,
    title: str,
    filename: str = "scatter_plot.png",
) -> str:
    """Observed (x) vs Simulated (y) with 1:1 line and R² annotation."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(observed, simulated, s=12, alpha=0.5, color=C_OBS, edgecolors="none")

    lim_max = max(observed.max(), simulated.max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "--", color="grey", linewidth=1, label="1:1 line")

    # R²
    ss_res = np.sum((observed - simulated) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax.text(
        0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Observed Discharge (m³/s)", fontsize=11)
    ax.set_ylabel("Simulated Discharge (m³/s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)

    return _save(fig, filename)


# =====================================================================
# 4.  Convergence curve
# =====================================================================

def plot_convergence(
    history: list[float],
    filename: str = "convergence_curve.png",
) -> str:
    """PSO best-objective value versus iteration."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(history)), history, color=C_SIM, linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Objective  (1 − NSE)", fontsize=11)
    ax.set_title("PSO Convergence", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    return _save(fig, filename)


# =====================================================================
# 5.  Rainfall-runoff overview (dual-axis)
# =====================================================================

def plot_rainfall_runoff(
    dates,
    precipitation: np.ndarray,
    observed: np.ndarray,
    simulated: np.ndarray,
    filename: str = "rainfall_runoff_overview.png",
) -> str:
    """Inverted precipitation bars on top axis + runoff lines below."""
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)

    # Discharge (primary axis)
    ax1.plot(dates, observed,  color=C_OBS, linewidth=1.1, label="Observed Q")
    ax1.plot(dates, simulated, color=C_SIM, linewidth=1.0, label="Simulated Q", alpha=0.8)
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Discharge (m³/s)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)

    # Precipitation (secondary axis, inverted)
    ax2 = ax1.twinx()
    ax2.bar(dates, precipitation, color=C_RAIN, alpha=0.45, width=1.0, label="Precipitation")
    ax2.set_ylabel("Precipitation (mm)", fontsize=11)
    ax2.invert_yaxis()
    ax2.set_ylim(precipitation.max() * 3, 0)
    ax2.legend(loc="upper right", fontsize=9)

    ax1.set_title("Rainfall–Runoff Overview", fontsize=13, fontweight="bold")
    fig.autofmt_xdate()

    return _save(fig, filename)
