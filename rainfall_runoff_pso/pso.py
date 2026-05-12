"""
pso.py — Particle Swarm Optimization (PSO) from scratch.

Implements the canonical PSO algorithm with inertia weight.  No external
optimisation libraries are used — everything is built on top of NumPy.

Velocity update  (per particle *i*, per dimension):
    v_i(t+1) = w  · v_i(t)
             + c1 · r1 · (pbest_i − x_i(t))
             + c2 · r2 · (gbest   − x_i(t))

Position update:
    x_i(t+1) = x_i(t) + v_i(t+1)
    x_i(t+1) = clip(x_i(t+1), lower, upper)

The optimiser **minimises** the supplied objective function.
"""

from __future__ import annotations

import numpy as np

from config import RANDOM_SEED, PARAM_NAMES


class ParticleSwarmOptimizer:
    """Full-featured PSO that minimises an arbitrary objective function.

    Parameters
    ----------
    objective_fn : callable(params: dict) -> float
        Function to **minimise**.  Receives a dict ``{name: value}``
        and returns a scalar cost.  Lower is better.
    bounds : dict
        ``{param_name: [lower, upper]}`` — search-space boundaries.
        Example: ``{"alpha": [0.01, 0.5], "beta": [1.0, 3.0], "k": [0.01, 0.99]}``
    n_particles  : int   — swarm size  (default 30).
    n_iterations : int   — maximum iterations  (default 100).
    w            : float — inertia weight  (default 0.7).
    c1           : float — cognitive (personal-best) coefficient  (default 1.5).
    c2           : float — social (global-best) coefficient  (default 1.5).
    """

    def __init__(
        self,
        objective_fn,
        bounds: dict,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Ordered param names — consistent with config.PARAM_NAMES
        self.param_names: list[str] = list(bounds.keys())
        self.n_dims = len(self.param_names)

        # Numeric bound arrays (shape: (n_dims,))
        self.lower = np.array([bounds[p][0] for p in self.param_names])
        self.upper = np.array([bounds[p][1] for p in self.param_names])

        # Reproducible RNG
        self.rng = np.random.default_rng(RANDOM_SEED)

        # ── Swarm state (initialised in optimize()) ──────────────────
        self.positions:      np.ndarray | None = None   # (n_particles, n_dims)
        self.velocities:     np.ndarray | None = None   # (n_particles, n_dims)
        self.pbest_positions: np.ndarray | None = None
        self.pbest_costs:    np.ndarray | None = None
        self.gbest_position: np.ndarray | None = None
        self.gbest_cost:     float = np.inf
        self.history:        list[float] = []           # best cost per iteration

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _array_to_dict(self, arr: np.ndarray) -> dict:
        """Convert a position vector to a ``{name: value}`` dict."""
        return {name: float(arr[i]) for i, name in enumerate(self.param_names)}

    def _evaluate_particle(self, position: np.ndarray) -> float:
        """Evaluate objective for a single particle's position."""
        return self.objective_fn(self._array_to_dict(position))

    def _initialize(self) -> None:
        """Seed particles uniformly within bounds; velocities start at 0."""
        # Positions: uniform random in [lower, upper]
        self.positions = self.rng.uniform(
            self.lower, self.upper, size=(self.n_particles, self.n_dims)
        )
        # Velocities: start at zero
        self.velocities = np.zeros((self.n_particles, self.n_dims))

        # Evaluate every particle
        costs = np.array([
            self._evaluate_particle(self.positions[i])
            for i in range(self.n_particles)
        ])

        # Personal bests = initial positions
        self.pbest_positions = self.positions.copy()
        self.pbest_costs = costs.copy()

        # Global best = best of the initial swarm
        best_idx = int(np.argmin(costs))
        self.gbest_position = self.positions[best_idx].copy()
        self.gbest_cost = float(costs[best_idx])

    def _step(self) -> None:
        """Execute one full PSO iteration: velocity → position → evaluate."""
        # Random matrices for stochastic component
        r1 = self.rng.random((self.n_particles, self.n_dims))
        r2 = self.rng.random((self.n_particles, self.n_dims))

        # ── Velocity update (vectorised across all particles) ─────────
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social    = self.c2 * r2 * (self.gbest_position  - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social

        # ── Position update + clamping to bounds ──────────────────────
        self.positions = self.positions + self.velocities
        self.positions = np.clip(self.positions, self.lower, self.upper)

        # ── Evaluate all particles ────────────────────────────────────
        costs = np.array([
            self._evaluate_particle(self.positions[i])
            for i in range(self.n_particles)
        ])

        # ── Update personal bests ─────────────────────────────────────
        improved = costs < self.pbest_costs
        self.pbest_positions[improved] = self.positions[improved]
        self.pbest_costs[improved] = costs[improved]

        # ── Update global best ────────────────────────────────────────
        best_idx = int(np.argmin(self.pbest_costs))
        if self.pbest_costs[best_idx] < self.gbest_cost:
            self.gbest_position = self.pbest_positions[best_idx].copy()
            self.gbest_cost = float(self.pbest_costs[best_idx])

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def optimize(self) -> tuple[dict, float, list[float]]:
        """
        Run the PSO optimisation loop.

        Returns
        -------
        best_params : dict
            ``{param_name: best_value}`` for each parameter.
        best_cost : float
            Lowest objective value found.
        history : list[float]
            Best objective value recorded at each iteration (length =
            n_iterations + 1, including initial evaluation).
        """
        self._initialize()
        self.history = [self.gbest_cost]

        for it in range(1, self.n_iterations + 1):
            self._step()
            self.history.append(self.gbest_cost)

            # ── Progress logging every 10 iterations ──────────────────
            if it % 10 == 0:
                print(
                    f"Iteration {it}/{self.n_iterations} "
                    f"| Best Cost: {self.gbest_cost:.6f}"
                )

            # ── Early stopping: cost ≈ 0 means NSE ≈ 1 ───────────────
            if self.gbest_cost < 1e-6:
                print(
                    f"Iteration {it}/{self.n_iterations} "
                    f"| Best Cost: {self.gbest_cost:.6f}  <-- early stop"
                )
                break

        best_params = self._array_to_dict(self.gbest_position)
        return best_params, self.gbest_cost, self.history
