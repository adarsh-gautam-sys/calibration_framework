"""
config.py — Global constants for the PSO Rainfall-Runoff Calibration Framework.

All hyperparameters, file paths, model bounds, and reproducibility settings
are defined here. Every other module imports from this file.
"""

import os

# ---------------------------------------------------------------------------
# Project root (resolves relative to this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Random seed — ensures reproducibility across data generation, PSO, plots
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# PSO hyperparameters
# ---------------------------------------------------------------------------
PSO_CONFIG = {
    "n_particles": 30,       # Number of candidate solutions in the swarm
    "n_iterations": 100,     # Maximum optimisation iterations
    "w": 0.7,                # Inertia weight  (exploration ↔ exploitation)
    "c1": 1.5,               # Cognitive coefficient  (personal best pull)
    "c2": 1.5,               # Social coefficient     (global best pull)
}

# ---------------------------------------------------------------------------
# Rainfall-runoff model parameter bounds
#   alpha — surface‐runoff coefficient       [0.01, 0.50]
#   beta  — non‐linear storage exponent      [1.00, 3.00]
#   k     — baseflow recession coefficient   [0.01, 0.99]
# ---------------------------------------------------------------------------
MODEL_PARAM_BOUNDS = {
    "alpha": [0.01, 0.50],
    "beta":  [1.00, 3.00],
    "k":     [0.01, 0.99],
}

# Ordered list so every module iterates parameters in the same order
PARAM_NAMES = list(MODEL_PARAM_BOUNDS.keys())       # ["alpha", "beta", "k"]
PARAM_LOWER = [MODEL_PARAM_BOUNDS[p][0] for p in PARAM_NAMES]
PARAM_UPPER = [MODEL_PARAM_BOUNDS[p][1] for p in PARAM_NAMES]

# True (known) parameters used to generate the synthetic dataset
# PSO should recover values close to these
TRUE_PARAMS = {
    "alpha": 0.25,
    "beta":  2.0,
    "k":     0.50,
}

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
RAW_DATA_FILE   = os.path.join(RAW_DATA_DIR, "data.csv")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR     = os.path.join(RESULTS_DIR, "figures")
CALIBRATED_PARAMS_FILE = os.path.join(RESULTS_DIR, "calibrated_params.json")

# ---------------------------------------------------------------------------
# Train / test (calibration / validation) split
# ---------------------------------------------------------------------------
CALIBRATION_RATIO = 0.70      # First 70 % → calibration
VALIDATION_RATIO  = 0.30      # Last  30 % → validation

# ---------------------------------------------------------------------------
# Synthetic data generation defaults
# ---------------------------------------------------------------------------
SYNTHETIC_N_DAYS   = 1095     # 3 years of daily data
SYNTHETIC_START    = "2018-01-01"

# ---------------------------------------------------------------------------
# Visualisation settings
# ---------------------------------------------------------------------------
PLOT_DPI    = 300
PLOT_STYLE  = "seaborn-v0_8-whitegrid"   # matplotlib style
FIG_SIZE    = (14, 5)

# ---------------------------------------------------------------------------
# Ensure output directories exist on import
# ---------------------------------------------------------------------------
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)
