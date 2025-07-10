"""Trajectory validation package.

This package provides comprehensive validation functionality for trajectory calculations,
split into focused modules for maintainability:

- physics_validation: Basic orbital mechanics validation
- constraint_validation: Trajectory constraint validation
- vector_validation: Vector unit and magnitude validation

For backward compatibility, the TrajectoryValidator class is re-exported from the original validation.py module.
"""

from .constraint_validation import validate_trajectory_constraints
from .physics_validation import (
    calculate_circular_velocity,
    validate_basic_orbital_mechanics,
    validate_solution_physics,
    validate_transfer_time,
)
from .vector_validation import (
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
