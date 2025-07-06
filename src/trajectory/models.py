"""Legacy imports for backward compatibility.

This module now serves as a compatibility layer, re-exporting classes that have been
moved to dedicated modules. New code should import directly from the specific modules.
"""

from .orbit_state import OrbitState
from .maneuver import Maneuver
from .trajectory_base import Trajectory
from .lunar_transfer import LunarTrajectory

__all__ = ['OrbitState', 'Maneuver', 'Trajectory', 'LunarTrajectory'] 