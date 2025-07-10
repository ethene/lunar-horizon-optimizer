"""Legacy trajectory physics module - DEPRECATED.

This module is maintained for backward compatibility only.
Please import from the new validation package modules:
- src.trajectory.validation.physics_validation
- src.trajectory.validation.constraint_validation
- src.trajectory.validation.vector_validation

This module will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "trajectory_physics.py is deprecated. Import from src.trajectory.validation package instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new validation modules for backward compatibility
from .validation.constraint_validation import validate_trajectory_constraints
from .validation.physics_validation import (
    calculate_circular_velocity,
    validate_basic_orbital_mechanics,
    validate_solution_physics,
    validate_transfer_time,
)
from .validation.vector_validation import (
    propagate_orbit,
    validate_delta_v,
    validate_state_vector,
    validate_vector_units,
)

__all__ = [
    "calculate_circular_velocity",
    "propagate_orbit",
    # Physics validation
    "validate_basic_orbital_mechanics",
    "validate_delta_v",
    "validate_solution_physics",
    "validate_state_vector",
    # Constraint validation
    "validate_trajectory_constraints",
    "validate_transfer_time",
    # Vector validation
    "validate_vector_units",
]
