"""Trajectory generation and analysis module.

This module provides functionality for creating, analyzing, and optimizing
trajectories between Earth and Moon using PyKEP.
"""

from trajectory.models import OrbitState, Maneuver, Trajectory
from trajectory.elements import orbital_period, velocity_at_point

__all__ = [
    'OrbitState',
    'Maneuver',
    'Trajectory',
    'orbital_period',
    'velocity_at_point',
] 