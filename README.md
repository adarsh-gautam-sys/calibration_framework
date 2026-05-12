# PSO-Calibrated Rainfall-Runoff Prediction Framework

> Automated parameter calibration for hydrological models using Particle Swarm Optimisation (PSO), implemented entirely in Python with NumPy.

![Workflow Diagram](results/figures/workflow_diagram.png)

---

## Overview

This project implements a complete **rainfall-runoff calibration and prediction pipeline**:

1. **Data acquisition** - Downloads real USGS streamflow data or generates realistic synthetic rainfall-runoff datasets using a Markov-chain precipitation model.
2. **Model** - A 3-parameter non-linear bucket model converts daily precipitation into simulated discharge.
3. **Calibration** - A from-scratch Particle Swarm Optimisation (PSO) engine automatically tunes model parameters to minimise RMSE against observed discharge.
4. **Prediction** - The calibrated model is applied to a held-out validation period.
5. **Evaluation** - Standard hydrological metrics (RMSE, NSE) quantify model performance.
6. **Visualisation** - Publication-ready plots summarise results at 300 DPI.

---

## Mathematical Foundation

### Rainfall-Runoff Model (Bucket Model)

The model uses three calibratable parameters to transform daily precipitation `P(t)` into discharge `Q(t)`:

```
Effective Rainfall:   ER(t) = alpha * P(t)^beta
Discharge:            Q(t)  = k * Q(t-1) + (1 - k) * ER(t)
```

| Parameter | Symbol | Range         | Physical Meaning                    |
|-----------|--------|---------------|--------------------------------------|
| alpha     | a      | [0.01, 0.50]  | Runoff coefficient                   |
| beta      | b      | [1.00, 3.00]  | Non-linearity exponent               |
| k         | k      | [0.01, 0.99]  | Recession constant (baseflow memory) |

### Particle Swarm Optimisation (PSO)

PSO is a population-based metaheuristic. Each "particle" represents a candidate parameter set `{alpha, beta, k}`:

```
Velocity update:
    v_i(t+1) = w * v_i(t)
             + c1 * r1 * (pbest_i - x_i(t))
             + c2 * r2 * (gbest   - x_i(t))

Position update:
    x_i(t+1) = x_i(t) + v_i(t+1)
    x_i(t+1) = clip(x_i(t+1), lower_bound, upper_bound)
```

| Hyperparameter | Value | Role                          |
|----------------|-------|-------------------------------|
| n_particles    | 30    | Swarm size                    |
| n_iterations   | 100   | Maximum optimisation steps    |
| w              | 0.7   | Inertia weight                |
| c1             | 1.5   | Cognitive (personal-best) pull|
| c2             | 1.5   | Social (global-best) pull     |

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE   | `sqrt(mean((Qobs - Qsim)^2))` | Lower is better; 0 = perfect |
| NSE    | `1 - sum((Qobs-Qsim)^2) / sum((Qobs-mean(Qobs))^2)` | 1 = perfect; >0.75 = very good |

---

## Project Structure

```
rainfall_runoff_pso/
|-- config.py            # All hyperparameters, paths, bounds, seed
|-- model.py             # simulate_runoff(precipitation, params_dict)
|-- pso.py               # ParticleSwarmOptimizer class (from scratch)
|-- calibrate.py          # run_calibration(precip_train, obs_train)
|-- predict.py            # run_prediction(precip_val, best_params)
|-- evaluate.py           # rmse(), nash_sutcliffe_efficiency(), print_metrics()
|-- visualize.py          # plot_all_results(), generate_workflow_diagram()
|-- main.py               # Full pipeline orchestrator with timing
|-- requirements.txt      # numpy, pandas, matplotlib, requests
|-- data/
|   |-- download.py       # USGS NWIS download + synthetic fallback
|   |-- raw/
|       |-- data.csv      # Generated/downloaded dataset
|-- results/
    |-- calibrated_params.json
    |-- figures/
        |-- results.png           # 3-subplot results figure
        |-- workflow_diagram.png  # Pipeline flowchart
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation & Run

```bash
cd rainfall_runoff_pso
pip install -r requirements.txt
python main.py
```

The pipeline will:
1. Generate synthetic data (or download from USGS if available)
2. Split 70% calibration / 30% validation
3. Run PSO calibration (~1 second)
4. Predict on validation period
5. Print metrics and summary tables
6. Save plots to `results/figures/`

### Expected Output

```
  Parameter       True  Calibrated   Error %
  ----------  --------  ----------  --------
  alpha         0.2500      0.2492      0.3%
  beta          2.0000      2.0007      0.0%
  k             0.5000      0.4995      0.1%

  Metric     Calibration    Validation
  --------  ------------  ------------
  RMSE            0.3978        0.4468
  NSE             0.9999        0.9999
```

---

## Module API Reference

### `model.simulate_runoff(precipitation, params)`
```python
simulate_runoff(
    precipitation: np.ndarray,  # daily precip (mm)
    params: dict                # {"alpha": float, "beta": float, "k": float}
) -> np.ndarray                 # simulated discharge
```

### `pso.ParticleSwarmOptimizer`
```python
pso = ParticleSwarmOptimizer(
    objective_fn=fn,            # callable(dict) -> float
    bounds={"alpha": [0.01, 0.5], "beta": [1.0, 3.0], "k": [0.01, 0.99]},
    n_particles=30, n_iterations=100, w=0.7, c1=1.5, c2=1.5
)
best_params, best_cost, history = pso.optimize()
```

### `calibrate.run_calibration(precip_train, obs_discharge_train)`
Returns `(best_params: dict, history: list[float])`

### `predict.run_prediction(precip_validation, best_params=None)`
Returns `simulated_discharge: np.ndarray`

### `evaluate.rmse(observed, simulated)` / `evaluate.nash_sutcliffe_efficiency(observed, simulated)`
Returns `float`

---

## Results

![Results Summary](results/figures/results.png)

The 3-subplot figure shows:
1. **Hydrograph** - Observed vs simulated discharge with train/val split line
2. **Convergence** - PSO cost (RMSE) decreasing over iterations (log scale)
3. **Scatter** - 1:1 line with R-squared annotation

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Dict-based parameter interface | Readable, self-documenting, avoids index errors |
| RMSE as PSO objective | Standard, interpretable, directly comparable across datasets |
| Vectorised effective rainfall | NumPy `np.power` + `np.clip` avoids slow Python loops |
| Minimal recession loop | Only the sequential `Q(t) = k*Q(t-1) + ...` requires iteration |
| No external PSO libraries | Full control, educational value, no `pyswarms` dependency |
| Reproducible via seed | `RANDOM_SEED = 42` in config, passed to all RNG instances |

---

## Dependencies

| Package    | Version  | Purpose                        |
|------------|----------|--------------------------------|
| numpy      | >= 1.24  | Array computation, vectorisation |
| pandas     | >= 2.0   | Time-series I/O, date handling |
| matplotlib | >= 3.7   | Plotting and diagram generation |
| requests   | >= 2.31  | Optional USGS data download    |

---

## License

This project is developed for academic purposes as part of the IITB Mini Project curriculum.

## Author

Adarsh Gautam — [GitHub](https://github.com/adarsh-gautam-sys)
