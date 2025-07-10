"""Trajectory generation and analysis module.

This module provides functionality for creating, analyzing, and optimizing
trajectories between Earth and Moon using PyKEP.
"""

from src.trajectory.elements import orbital_period, velocity_at_point
from src.trajectory.models import Maneuver, OrbitState, Trajectory

__all__ = [
    "Maneuver",
    "OrbitState",
    "Trajectory",
    "orbital_period",
    "velocity_at_point",
]
