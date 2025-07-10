"""Legacy imports for backward compatibility.

This module now serves as a compatibility layer, re-exporting classes that have been
moved to dedicated modules. New code should import directly from the specific modules.
"""

from .lunar_transfer import LunarTransfer as LunarTrajectory
from .maneuver import Maneuver
from .orbit_state import OrbitState
from .trajectory_base import Trajectory

__all__ = ["LunarTrajectory", "Maneuver", "OrbitState", "Trajectory"]
