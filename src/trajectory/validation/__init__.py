"""Trajectory validation package.

This package provides comprehensive validation functionality for trajectory calculations,
split into focused modules for maintainability:

- physics_validation: Basic orbital mechanics validation
- constraint_validation: Trajectory constraint validation  
- vector_validation: Vector unit and magnitude validation

For backward compatibility, the TrajectoryValidator class is re-exported from the original validation.py module.
"""

from .physics_validation import (
    validate_basic_orbital_mechanics,
    validate_transfer_time,
    validate_solution_physics,
    calculate_circular_velocity
)
from .constraint_validation import (
    validate_trajectory_constraints
)
from .vector_validation import (
    validate_vector_units,
    validate_delta_v,
    validate_state_vector,
    propagate_orbit
)

__all__ = [
    # Physics validation
    "validate_basic_orbital_mechanics",
    "validate_transfer_time",
    "validate_solution_physics",
    "calculate_circular_velocity",
    # Constraint validation
    "validate_trajectory_constraints",
    # Vector validation
    "validate_vector_units",
    "validate_delta_v",
    "validate_state_vector",
    "propagate_orbit"
]
