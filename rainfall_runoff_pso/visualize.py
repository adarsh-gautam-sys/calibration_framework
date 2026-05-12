"""
visualize.py — Publication-quality results visualisation.

Provides ``plot_all_results(dates, obs, sim, pso_history, split_idx)``
which creates a single figure with three subplots:

    1. Observed vs Simulated Discharge (line plot, with train/val split line)
    2. PSO Convergence Curve (iteration vs best RMSE cost)
    3. Scatter plot: Observed vs Simulated with 1:1 line and R² annotation

Saved to  results/figures/results.png  at 300 DPI.
"""

import os

import matplotlib
matplotlib.use("Agg")                           # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, PLOT_DPI


# ─── Colour palette ──────────────────────────────────────────────────────
C_OBS     = "#1a73e8"      # Google Blue
C_SIM     = "#ea4335"      # Google Red
C_SPLIT   = "#34a853"      # Google Green
C_SCATTER = "#4285f4"      # Material Blue
C_LINE    = "#757575"      # Neutral grey


def plot_all_results(
    dates: np.ndarray,
    obs: np.ndarray,
    sim: np.ndarray,
    pso_history: list[float],
    split_idx: int,
    save_path: str | None = None,
) -> str:
    """
    Generate a 3-subplot figure summarising all results.

    Parameters
    ----------
    dates : array-like
        Full date axis (calibration + validation).
    obs : np.ndarray
        Full observed discharge (cal + val concatenated).
    sim : np.ndarray
        Full simulated discharge (cal + val concatenated).
    pso_history : list[float]
        Best RMSE cost per PSO iteration.
    split_idx : int
        Index separating calibration from validation in the arrays.
    save_path : str or None
        Override default save location.

    Returns
    -------
    str — path to the saved figure.
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "results.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(
        "PSO-Calibrated Rainfall-Runoff Model — Results Summary",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ==================================================================
    # Subplot 1 — Hydrograph: Observed vs Simulated
    # ==================================================================
    ax1 = axes[0]
    ax1.plot(dates, obs, color=C_OBS, linewidth=1.0, label="Observed", alpha=0.85)
    ax1.plot(dates, sim, color=C_SIM, linewidth=0.9, label="Simulated", alpha=0.85)

    # Train / val split line
    split_date = dates[split_idx]
    ax1.axvline(
        x=split_date, color=C_SPLIT, linestyle="--", linewidth=1.5,
        label="Cal / Val split",
    )
    ax1.fill_between(dates, obs, sim, color="#e8f0fe", alpha=0.3)

    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Discharge (m³/s)", fontsize=10)
    ax1.set_title("Observed vs Simulated Discharge", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # Rotate date labels
    for label in ax1.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
        label.set_fontsize(8)

    # ==================================================================
    # Subplot 2 — PSO Convergence
    # ==================================================================
    ax2 = axes[1]
    iterations = range(len(pso_history))
    ax2.plot(iterations, pso_history, color=C_SIM, linewidth=1.5)
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.set_ylabel("Best RMSE Cost", fontsize=10)
    ax2.set_title("PSO Convergence Curve", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)

    # Annotate final cost
    final_cost = pso_history[-1]
    ax2.annotate(
        f"Final: {final_cost:.4f}",
        xy=(len(pso_history) - 1, final_cost),
        xytext=(-60, 20), textcoords="offset points",
        fontsize=9, color=C_SIM,
        arrowprops=dict(arrowstyle="->", color=C_SIM, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # ==================================================================
    # Subplot 3 — Scatter plot with 1:1 line and R²
    # ==================================================================
    ax3 = axes[2]
    ax3.scatter(obs, sim, s=10, alpha=0.45, color=C_SCATTER, edgecolors="none")

    # 1:1 reference line
    lim_max = max(obs.max(), sim.max()) * 1.05
    ax3.plot([0, lim_max], [0, lim_max], "--", color=C_LINE, linewidth=1.0, label="1:1 line")

    # R²  (coefficient of determination)
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ax3.text(
        0.05, 0.92, f"R² = {r2:.4f}",
        transform=ax3.transAxes, fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax3.set_xlabel("Observed Discharge (m³/s)", fontsize=10)
    ax3.set_ylabel("Simulated Discharge (m³/s)", fontsize=10)
    ax3.set_title("Scatter: Observed vs Simulated", fontsize=12, fontweight="bold")
    ax3.set_xlim(0, lim_max)
    ax3.set_ylim(0, lim_max)
    ax3.set_aspect("equal")
    ax3.legend(fontsize=9)

    # ==================================================================
    # Save
    # ==================================================================
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [plot] Saved results figure to {save_path}")

    return save_path


# =========================================================================
# Workflow diagram
# =========================================================================

def generate_workflow_diagram(save_path: str | None = None) -> str:
    """
    Create a clean flowchart-style workflow diagram using matplotlib patches.

    Nodes
    -----
    Raw Environmental Data  ->  Preprocessing & Train/Val Split
      -> Rainfall-Runoff Model (3 params)  <->  PSO Calibration Engine
      -> Simulated Discharge               ->  Convergence History
      -> Evaluation: RMSE + NSE
      -> Prediction on Validation Set
      -> Results & Visualisation

    Returns
    -------
    str - path to saved diagram.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "workflow_diagram.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Style constants ───────────────────────────────────────────────
    BOX_COLOR    = "#ffffff"
    BORDER_COLOR = "#1a73e8"     # Google Blue
    ARROW_COLOR  = "#424242"     # Dark grey
    TITLE_COLOR  = "#1a237e"     # Deep indigo
    TEXT_SIZE    = 10
    BOX_W        = 3.8           # box width
    BOX_H        = 0.7           # box height

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(
        6, 13.5,
        "PSO Calibration Framework",
        fontsize=18, fontweight="bold", color=TITLE_COLOR,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8eaf6",
                  edgecolor=TITLE_COLOR, linewidth=1.5),
    )
    ax.text(
        6, 12.9,
        "Automated Rainfall-Runoff Prediction Pipeline",
        fontsize=10, color="#616161", ha="center", va="center",
        style="italic",
    )

    # ── Node definitions: (x_center, y_center, label) ─────────────────
    #   Left column (x=3.8) = main pipeline
    #   Right column (x=8.5) = PSO branch
    nodes = {
        "data":    (3.8, 12.0, "Raw Environmental Data"),
        "preproc": (3.8, 10.8, "Preprocessing &\nTrain / Val Split"),
        "model":   (3.8,  9.4, "Rainfall-Runoff Model\n(3 params: a, b, k)"),
        "pso":     (8.5,  9.4, "PSO Calibration\nEngine"),
        "sim":     (3.8,  7.8, "Simulated Discharge"),
        "conv":    (8.5,  7.8, "Convergence History"),
        "eval":    (3.8,  6.2, "Evaluation:\nRMSE + NSE"),
        "pred":    (3.8,  4.6, "Prediction on\nValidation Set"),
        "result":  (3.8,  3.0, "Results &\nVisualisation"),
    }

    # ── Draw boxes ────────────────────────────────────────────────────
    box_rects = {}
    for key, (cx, cy, label) in nodes.items():
        x0 = cx - BOX_W / 2
        y0 = cy - BOX_H / 2
        rect = FancyBboxPatch(
            (x0, y0), BOX_W, BOX_H,
            boxstyle="round,pad=0.15",
            facecolor=BOX_COLOR,
            edgecolor=BORDER_COLOR,
            linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, label, fontsize=TEXT_SIZE, ha="center", va="center",
                color="#212121", fontweight="medium")
        box_rects[key] = (cx, cy)

    # ── Arrow helper ──────────────────────────────────────────────────
    def add_arrow(src_key, dst_key, style="-|>", color=ARROW_COLOR,
                  connectionstyle="arc3,rad=0", bidirectional=False):
        sx, sy = box_rects[src_key]
        dx, dy = box_rects[dst_key]

        # Adjust start/end to box edges
        if sy > dy:        # going down
            sy -= BOX_H / 2
            dy += BOX_H / 2
        elif sy < dy:      # going up
            sy += BOX_H / 2
            dy -= BOX_H / 2
        if sx < dx:        # going right
            sx += BOX_W / 2
            dx -= BOX_W / 2
        elif sx > dx:      # going left
            sx -= BOX_W / 2
            dx += BOX_W / 2

        arrow = FancyArrowPatch(
            (sx, sy), (dx, dy),
            arrowstyle=style,
            mutation_scale=15,
            color=color,
            linewidth=1.5,
            connectionstyle=connectionstyle,
        )
        ax.add_patch(arrow)

        if bidirectional:
            arrow2 = FancyArrowPatch(
                (dx, dy), (sx, sy),
                arrowstyle=style,
                mutation_scale=15,
                color=color,
                linewidth=1.5,
                connectionstyle=connectionstyle,
            )
            ax.add_patch(arrow2)

    # ── Vertical main pipeline arrows ─────────────────────────────────
    add_arrow("data",    "preproc")
    add_arrow("preproc", "model")
    add_arrow("model",   "sim")
    add_arrow("sim",     "eval")
    add_arrow("eval",    "pred")
    add_arrow("pred",    "result")

    # ── Horizontal / branch arrows ────────────────────────────────────
    # Model <-> PSO (bidirectional)
    add_arrow("model", "pso", bidirectional=True)

    # PSO -> Convergence History (down)
    add_arrow("pso", "conv")

    # ── Legend annotation ─────────────────────────────────────────────
    ax.text(
        6, 1.8,
        "Pipeline: data/download.py -> calibrate.py -> predict.py -> evaluate.py -> visualize.py",
        fontsize=8, color="#9e9e9e", ha="center", va="center",
        style="italic",
    )
    ax.text(
        6, 1.3,
        "Orchestrated by main.py  |  All parameters defined in config.py",
        fontsize=8, color="#9e9e9e", ha="center", va="center",
        style="italic",
    )

    # ── Save ──────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [plot] Saved workflow diagram to {save_path}")

    return save_path

