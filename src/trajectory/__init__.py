"""Trajectory generation and analysis module.

This module provides functionality for creating, analyzing, and optimizing
trajectories between Earth and Moon using PyKEP.
"""

from src.trajectory.models import OrbitState, Maneuver, Trajectory
from src.trajectory.elements import orbital_period, velocity_at_point

__all__ = [
    "Maneuver",
    "OrbitState",
    "Trajectory",
    "orbital_period",
    "velocity_at_point",
]
