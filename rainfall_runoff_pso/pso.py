"""
pso.py — Particle Swarm Optimization (PSO) from scratch.

Implements the canonical PSO with:
    - Inertia weight (w)        for exploration–exploitation balance
    - Cognitive coefficient (c1) for personal-best attraction
    - Social coefficient (c2)    for global-best attraction

Velocity update:
    v_i(t+1) = w · v_i(t)
             + c1 · r1 · (pbest_i − x_i(t))
             + c2 · r2 · (gbest   − x_i(t))

Position update:
    x_i(t+1) = x_i(t) + v_i(t+1)
    x_i(t+1) = clip(x_i(t+1), lower, upper)

The optimiser **minimises** the supplied objective function.
"""

from __future__ import annotations

import numpy as np

from config import RANDOM_SEED


class PSO:
    """Particle Swarm Optimiser."""

    def __init__(
        self,
        objective_fn,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = RANDOM_SEED,
    ) -> None:
        """
        Parameters
        ----------
        objective_fn : callable(params: ndarray) -> float
            Function to **minimise**.  Lower is better.
        lower_bounds, upper_bounds : ndarray, shape (n_dims,)
            Search-space boundaries for each parameter.
        n_particles  : int   — swarm size.
        n_iterations : int   — maximum iterations.
        w            : float — inertia weight.
        c1           : float — cognitive (personal-best) coefficient.
        c2           : float — social (global-best) coefficient.
        seed         : int   — RNG seed for reproducibility.
        """
        self.objective_fn = objective_fn
        self.lower = np.asarray(lower_bounds, dtype=float)
        self.upper = np.asarray(upper_bounds, dtype=float)
        self.n_dims = len(self.lower)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)

        # ── Swarm state ──────────────────────────────────────────────
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.pbest_positions: np.ndarray | None = None
        self.pbest_scores: np.ndarray | None = None
        self.gbest_position: np.ndarray | None = None
        self.gbest_score: float = np.inf
        self.history: list[float] = []       # best score per iteration

    # -----------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------
    def _initialize(self) -> None:
        """Seed particles uniformly within bounds; velocities start at 0."""
        self.positions = self.rng.uniform(
            self.lower, self.upper, size=(self.n_particles, self.n_dims)
        )
        self.velocities = np.zeros_like(self.positions)

        # Evaluate initial population
        scores = np.array([self.objective_fn(p) for p in self.positions])

        self.pbest_positions = self.positions.copy()
        self.pbest_scores = scores.copy()

        best_idx = int(np.argmin(scores))
        self.gbest_position = self.positions[best_idx].copy()
        self.gbest_score = scores[best_idx]

    # -----------------------------------------------------------------
    # Core update step
    # -----------------------------------------------------------------
    def _step(self) -> None:
        """Perform one full PSO iteration (velocity → position → evaluate)."""
        r1 = self.rng.random((self.n_particles, self.n_dims))
        r2 = self.rng.random((self.n_particles, self.n_dims))

        # Velocity update
        cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
        social    = self.c2 * r2 * (self.gbest_position  - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social

        # Position update + clamping
        self.positions = self.positions + self.velocities
        self.positions = np.clip(self.positions, self.lower, self.upper)

        # Evaluate
        scores = np.array([self.objective_fn(p) for p in self.positions])

        # Update personal bests
        improved = scores < self.pbest_scores
        self.pbest_positions[improved] = self.positions[improved]
        self.pbest_scores[improved] = scores[improved]

        # Update global best
        best_idx = int(np.argmin(self.pbest_scores))
        if self.pbest_scores[best_idx] < self.gbest_score:
            self.gbest_position = self.pbest_positions[best_idx].copy()
            self.gbest_score = self.pbest_scores[best_idx]

    # -----------------------------------------------------------------
    # Main optimisation loop
    # -----------------------------------------------------------------
    def optimize(self, verbose: bool = True) -> tuple[np.ndarray, float, list[float]]:
        """
        Run the PSO loop.

        Returns
        -------
        best_params : ndarray, shape (n_dims,)
        best_score  : float
        history     : list[float]   — best objective at each iteration
        """
        self._initialize()
        self.history = [self.gbest_score]

        for it in range(1, self.n_iterations + 1):
            self._step()
            self.history.append(self.gbest_score)

            if verbose and it % 10 == 0:
                print(f"  PSO iter {it:>4d}/{self.n_iterations}  |  "
                      f"best obj = {self.gbest_score:.6f}")

            # Early stopping: objective ≈ 0  →  NSE ≈ 1
            if self.gbest_score < 1e-4:
                if verbose:
                    print(f"  Early stop at iter {it} (obj < 1e-4)")
                break

        return self.gbest_position.copy(), self.gbest_score, self.history
