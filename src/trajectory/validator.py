"""Backward compatibility module for trajectory validation.

DEPRECATED: This module is maintained for backward compatibility.
Please import directly from src.trajectory.validation instead.

This module will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "validator.py is deprecated. Import from src.trajectory.validation instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the new validation module
from .validation import TrajectoryValidator

__all__ = ['TrajectoryValidator']