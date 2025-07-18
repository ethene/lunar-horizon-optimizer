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
    stacklevel=2,
)

# Re-export from the trajectory_validator module
from .trajectory_validator import (
    validate_epoch,
    validate_final_orbit,
    validate_initial_orbit,
    validate_orbit_altitude,
    validate_transfer_parameters,
)

# Alias for backward compatibility
validate_inputs = validate_transfer_parameters

__all__ = [
    "validate_epoch",
    "validate_final_orbit",
    "validate_initial_orbit",
    "validate_inputs",  # Backward compatibility alias
    "validate_orbit_altitude",
    "validate_transfer_parameters",
]
