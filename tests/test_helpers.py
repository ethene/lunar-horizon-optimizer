"""
Test helper classes and utilities for replacing complex dependencies in tests.

This module provides simplified implementations of complex modules that depend on
external libraries like PyKEP and PyGMO, allowing tests to run without these
dependencies while still testing the core logic.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SimpleTrajectory:
    """Simplified trajectory for testing purposes."""

    def __init__(self, points=None, delta_v=3500.0):
        self.points = points or []
        self.total_delta_v = delta_v
        self.maneuvers = []

    def add_maneuver(self, maneuver):
        """Add a maneuver to the trajectory."""
        self.maneuvers.append(maneuver)


class SimpleLunarTransfer:
    """Simplified LunarTransfer implementation for testing without PyKEP."""

    def __init__(
        self,
        min_earth_alt=200,
        max_earth_alt=1000,
        min_moon_alt=50,
        max_moon_alt=500,
        **kwargs,
    ):
        """Initialize simplified lunar transfer."""
        self.min_earth_alt = min_earth_alt
        self.max_earth_alt = max_earth_alt
        self.min_moon_alt = min_moon_alt
        self.max_moon_alt = max_moon_alt

    def generate_transfer(
        self,
        epoch: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
        transfer_time: float,
        max_revolutions: int = 0,
    ) -> tuple[SimpleTrajectory, float]:
        """Generate a simplified transfer trajectory.

        Returns realistic delta-v values based on typical lunar mission parameters.
        """
        # Validate inputs
        if not (self.min_earth_alt <= earth_orbit_alt <= self.max_earth_alt):
            msg = f"Earth altitude {earth_orbit_alt} outside valid range"
            raise ValueError(msg)
        if not (self.min_moon_alt <= moon_orbit_alt <= self.max_moon_alt):
            msg = f"Moon altitude {moon_orbit_alt} outside valid range"
            raise ValueError(msg)
        if not (3.0 <= transfer_time <= 10.0):
            msg = f"Transfer time {transfer_time} outside valid range"
            raise ValueError(msg)

        # Calculate realistic delta-v based on orbital mechanics approximations
        # Base delta-v for lunar transfer: ~3200 m/s
        base_dv = 3200.0

        # Altitude penalties (lower orbits require more energy)
        earth_penalty = max(
            0, (400 - earth_orbit_alt) * 2.0
        )  # 2 m/s per km below 400km
        moon_penalty = max(0, (100 - moon_orbit_alt) * 5.0)  # 5 m/s per km below 100km

        # Transfer time penalties (faster transfers are less efficient)
        if transfer_time < 4.0:
            time_penalty = (4.0 - transfer_time) * 200.0
        elif transfer_time > 6.0:
            time_penalty = (transfer_time - 6.0) * 50.0
        else:
            time_penalty = 0.0

        total_dv = base_dv + earth_penalty + moon_penalty + time_penalty

        # Add some realistic variation
        total_dv += np.random.normal(0, 50)  # ±50 m/s variation

        # Create trajectory with realistic characteristics
        trajectory = SimpleTrajectory(delta_v=total_dv)

        return trajectory, total_dv


class SimpleOptimizationProblem:
    """Simplified optimization problem for testing PyGMO functionality."""

    def __init__(self, objectives=3, parameters=3):
        """Initialize with specified number of objectives and parameters."""
        self.n_obj = objectives
        self.n_param = parameters

    def get_nobj(self):
        """Get number of objectives."""
        return self.n_obj

    def get_bounds(self):
        """Get parameter bounds."""
        if self.n_param == 3:
            # Lunar mission bounds: [earth_alt, moon_alt, transfer_time]
            return ([200, 50, 3.0], [1000, 500, 10.0])
        # Generic bounds
        return ([0] * self.n_param, [10] * self.n_param)

    def fitness(self, x):
        """Calculate fitness for given parameters."""
        if self.n_param == 3 and self.n_obj == 3:
            # Lunar mission problem
            earth_alt, moon_alt, transfer_time = x

            # Use SimpleLunarTransfer for realistic calculations
            transfer = SimpleLunarTransfer()
            _, total_dv = transfer.generate_transfer(
                epoch=10000.0,
                earth_orbit_alt=earth_alt,
                moon_orbit_alt=moon_alt,
                transfer_time=transfer_time,
            )

            # Calculate cost based on delta-v and time
            cost = total_dv * 1e5 + transfer_time * 1e7  # Simplified cost model

            return [total_dv, transfer_time * 86400, cost]  # [m/s, seconds, cost]
        # Generic multi-objective problem
        return [(sum(x) + i) ** 2 for i in range(self.n_obj)]


def create_mock_pykep():
    """Create a mock PyKEP module for testing."""
    from unittest.mock import MagicMock

    mock_pk = MagicMock()

    # Mock constants
    mock_pk.MU_EARTH = 3.986004418e14
    mock_pk.MU_MOON = 4.9048695e12
    mock_pk.EARTH_RADIUS = 6378137.0
    mock_pk.MOON_RADIUS = 1737400.0

    # Mock lambert problem solver
    class MockLambertProblem:
        def __init__(self, r1, r2, tof, mu):
            self.r1 = r1
            self.r2 = r2
            self.tof = tof
            self.mu = mu

        def get_v1(self):
            return [[1000, 2000, 3000]]  # Mock velocity vector

        def get_v2(self):
            return [[1100, 2100, 3100]]  # Mock velocity vector

    mock_pk.lambert_problem = MockLambertProblem

    return mock_pk


def create_mock_pygmo():
    """Create a mock PyGMO module for testing."""
    from unittest.mock import MagicMock

    mock_pg = MagicMock()

    # Mock non-dominated sorting
    def mock_fast_non_dominated_sorting(fitness_values):
        """Mock fast non-dominated sorting."""
        n = len(fitness_values)
        if n == 0:
            return [], [], [], []
        # Return all points as first front for simplicity
        return [list(range(n))], [0] * n, [1] * n, []

    mock_pg.fast_non_dominated_sorting = mock_fast_non_dominated_sorting

    # Mock algorithm
    class MockNSGA2:
        def __init__(self, gen=100):
            self.gen = gen

    mock_pg.nsga2 = MockNSGA2

    # Mock population
    class MockPopulation:
        def __init__(self, prob, size):
            self.prob = prob
            self.size = size
            self._x = []
            self._f = []

        def push_back(self, x):
            self._x.append(x)
            fitness = self.prob.fitness(x)
            self._f.append(fitness)

        def get_x(self):
            return np.array(self._x) if self._x else np.array([])

        def get_f(self):
            return np.array(self._f) if self._f else np.array([])

        def evolve(self, algo):
            return self

        def __len__(self):
            return len(self._x)

    mock_pg.population = MockPopulation

    return mock_pg
