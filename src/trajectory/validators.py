"""Backward compatibility module for trajectory validation functions.

DEPRECATED: This module is maintained for backward compatibility.
Please import directly from src.trajectory.validation instead.

This module will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "validators.py is deprecated. Import from src.trajectory.validation instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the new validation module
from .validation import (
    validate_epoch,
    validate_orbit_altitude, 
    validate_transfer_parameters,
    validate_initial_orbit,
    validate_final_orbit
)

__all__ = [
    'validate_epoch',
    'validate_orbit_altitude', 
    'validate_transfer_parameters',
    'validate_initial_orbit',
    'validate_final_orbit'
]