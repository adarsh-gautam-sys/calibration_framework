"""
download.py — Data acquisition for the Rainfall-Runoff PSO framework.

Strategy
--------
1. Attempt to download real daily discharge data from USGS NWIS
   (Leaf River at Hattiesburg, MS — site 02472000).
2. If the download fails for any reason (network, API change, etc.),
   fall back to generating a realistic synthetic rainfall-runoff dataset
   using the bucket model with known (TRUE) parameters + noise.

Either way the output is saved to  data/raw/data.csv  with columns:
    Date  |  Precipitation_mm  |  Observed_Discharge_m3s

Usage
-----
    python -m data.download          # from the rainfall_runoff_pso/ dir
    python data/download.py          # direct invocation
"""

import os
import sys

import numpy as np
import pandas as pd

# ── Resolve imports when run as a script or as a module ──────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    RAW_DATA_FILE,
    RAW_DATA_DIR,
    RANDOM_SEED,
    SYNTHETIC_N_DAYS,
    SYNTHETIC_START,
    TRUE_PARAMS,
)


# =========================================================================
# 1.  Attempt real-data download  (USGS NWIS)
# =========================================================================

USGS_URL = (
    "https://waterservices.usgs.gov/nwis/dv/"
    "?format=rdb"
    "&sites=02472000"              # Leaf River at Hattiesburg, MS
    "&startDT=2018-01-01"
    "&endDT=2020-12-31"
    "&parameterCd=00060"           # discharge in cfs
    "&statCd=00003"                # daily mean
)


def _try_download_usgs() -> pd.DataFrame | None:
    """Return a cleaned DataFrame or None on failure."""
    try:
        import requests

        print("[download] Attempting USGS NWIS download …")
        resp = requests.get(USGS_URL, timeout=30)
        resp.raise_for_status()

        # USGS RDB format has comment lines starting with '#' and a header
        lines = [
            ln for ln in resp.text.splitlines()
            if not ln.startswith("#") and ln.strip()
        ]
        if len(lines) < 3:
            raise ValueError("USGS response too short")

        # Write filtered lines to a temp buffer and read with pandas
        from io import StringIO
        buf = StringIO("\n".join(lines))
        df = pd.read_csv(buf, sep="\t")

        # Drop the format-description row (2nd row in RDB files)
        df = df.iloc[1:].reset_index(drop=True)

        # Locate the discharge column (name varies; usually ends with _00060_00003)
        q_col = [c for c in df.columns if "00060" in c and "cd" not in c.lower()]
        if not q_col:
            raise ValueError("Discharge column not found in USGS response")

        df = df.rename(columns={q_col[0]: "discharge_cfs", "datetime": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
        df = df.dropna(subset=["discharge_cfs"])

        # Convert cfs → m³/s  (1 cfs ≈ 0.0283168 m³/s)
        df["Observed_Discharge_m3s"] = df["discharge_cfs"] * 0.0283168

        # Generate synthetic precipitation (USGS doesn't serve precip)
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(df)
        wet = rng.random(n) < 0.35
        amounts = rng.exponential(scale=10.0, size=n)
        df["Precipitation_mm"] = np.where(wet, amounts, 0.0).round(2)

        df = df[["Date", "Precipitation_mm", "Observed_Discharge_m3s"]].copy()
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"[download] USGS data retrieved: {df.shape[0]} rows")
        return df

    except Exception as exc:
        print(f"[download] USGS download failed ({exc}). Falling back to synthetic data.")
        return None


# =========================================================================
# 2.  Synthetic data generator  (always-works fallback)
# =========================================================================

def _generate_synthetic() -> pd.DataFrame:
    """
    Produce a realistic synthetic rainfall-runoff dataset.

    Method
    ------
    - Precipitation: Markov-chain wet/dry state with exponential rain depth
    - Runoff: 3-parameter bucket model driven by TRUE_PARAMS from config
    - Noise: additive Gaussian (σ = 0.02 × mean discharge) for realism
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = SYNTHETIC_N_DAYS

    # ── Dates ─────────────────────────────────────────────────────────
    dates = pd.date_range(start=SYNTHETIC_START, periods=n, freq="D")

    # ── Precipitation (Markov chain) ──────────────────────────────────
    P_wet_given_dry = 0.30
    P_wet_given_wet = 0.60
    mean_rain_depth = 10.0          # mm on a wet day

    precip = np.zeros(n)
    wet = False
    for t in range(n):
        p_wet = P_wet_given_wet if wet else P_wet_given_dry
        wet = rng.random() < p_wet
        if wet:
            precip[t] = rng.exponential(mean_rain_depth)
    precip = np.round(precip, 2)

    # ── Run bucket model with TRUE parameters to get "observed" Q ─────
    #   Uses the same simulate_runoff() that PSO will calibrate against,
    #   ensuring the synthetic data is perfectly consistent with the model.
    from model import simulate_runoff
    Q = simulate_runoff(precip, TRUE_PARAMS)

    # ── Add realistic measurement noise ───────────────────────────────
    noise_sigma = 0.02 * np.mean(Q)
    Q = Q + rng.normal(0, noise_sigma, size=n)
    Q = np.maximum(Q, 0.0)         # discharge cannot be negative
    Q = np.round(Q, 4)

    df = pd.DataFrame({
        "Date": dates,
        "Precipitation_mm": precip,
        "Observed_Discharge_m3s": Q,
    })
    print(f"[download] Synthetic data generated: {df.shape[0]} rows, "
          f"true params α={alpha}, β={beta}, k={k}")
    return df


# =========================================================================
# 3.  Main entry point
# =========================================================================

def download_data(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Acquire data (real or synthetic) and save to RAW_DATA_FILE.

    Parameters
    ----------
    force_synthetic : bool
        If True, skip the USGS download and go straight to synthetic.

    Returns
    -------
    pd.DataFrame  with columns [Date, Precipitation_mm, Observed_Discharge_m3s]
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    df = None
    if not force_synthetic:
        df = _try_download_usgs()

    if df is None:
        df = _generate_synthetic()

    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"[download] Saved to {RAW_DATA_FILE}")
    print(f"[download] Shape: {df.shape}")
    print(f"[download] Head:\n{df.head().to_string(index=False)}")
    return df


if __name__ == "__main__":
    download_data()
