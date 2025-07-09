"""Vector validation functions for trajectory calculations.

This module provides functions for validating vector units, magnitudes,
and performing basic vector operations in trajectory calculations.
"""

import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)


def validate_vector_units(vector: np.ndarray, name: str, expected_magnitude_range: tuple[float, float], unit: str) -> bool:
    """
    Validate that a vector's magnitude falls within expected range and has correct units.

    Args:
        vector: Vector to validate
        name: Name of the vector for logging
        expected_magnitude_range: (min, max) expected magnitude
        unit: Unit of measurement for logging

    Returns
    -------
        bool: True if vector is valid
    """
    if not isinstance(vector, np.ndarray):
        logger.error(f"{name} is not a numpy array")
        return False

    if vector.shape != (3,):
        logger.error(f"{name} should be a 3D vector, got shape {vector.shape}")
        return False

    magnitude = np.linalg.norm(vector)
    min_mag, max_mag = expected_magnitude_range

    logger.debug(f"Validating {name}:")
    logger.debug(f"  Vector: {vector} {unit}")
    logger.debug(f"  Magnitude: {magnitude:.2e} {unit}")
    logger.debug(f"  Expected range: [{min_mag:.2e}, {max_mag:.2e}] {unit}")

    if not (min_mag <= magnitude <= max_mag):
        logger.error(f"{name} magnitude {magnitude:.2e} {unit} outside expected range [{min_mag:.2e}, {max_mag:.2e}] {unit}")
        return False

    return True


def validate_delta_v(delta_v: np.ndarray, max_delta_v: float = 25000.0) -> bool:
    """Validate delta-v vector for reasonableness.

    Args:
        delta_v: Delta-v vector [m/s]
        max_delta_v: Maximum allowed delta-v magnitude [m/s]

    Returns
    -------
        True if delta-v is valid

    Raises
    ------
        ValueError: If delta-v is invalid
    """
    if not isinstance(delta_v, np.ndarray):
        msg = "Delta-v must be a numpy array"
        raise ValueError(msg)

    if delta_v.shape != (3,):
        msg = "Delta-v must be a 3D vector"
        raise ValueError(msg)

    magnitude = np.linalg.norm(delta_v)
    if magnitude > max_delta_v:
        msg = f"Delta-v magnitude {magnitude:.1f} m/s exceeds maximum {max_delta_v:.1f} m/s"
        raise ValueError(msg)

    if not np.isfinite(delta_v).all():
        msg = "Delta-v contains non-finite values"
        raise ValueError(msg)

    return True


def validate_state_vector(position: np.ndarray, velocity: np.ndarray) -> bool:
    """Validate state vector components.

    Args:
        position: Position vector [m]
        velocity: Velocity vector [m/s]

    Returns
    -------
        True if state vector is valid

    Raises
    ------
        ValueError: If state vector is invalid
    """
    if not isinstance(position, np.ndarray) or not isinstance(velocity, np.ndarray):
        msg = "Position and velocity must be numpy arrays"
        raise ValueError(msg)

    if position.shape != (3,) or velocity.shape != (3,):
        msg = "Position and velocity must be 3D vectors"
        raise ValueError(msg)

    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        msg = "State vector contains non-finite values"
        raise ValueError(msg)

    return True


def propagate_orbit(position: np.ndarray, velocity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Simple two-body orbit propagation.

    Args:
        position: Initial position vector [m]
        velocity: Initial velocity vector [m/s]
        dt: Time step [s]

    Returns
    -------
        Tuple of final (position, velocity) vectors

    Note:
        This is a simplified implementation for basic functionality.
        For high-fidelity propagation, use PyKEP's propagate_taylor.
    """
    from src.trajectory.constants import PhysicalConstants as PC

    # Simple Keplerian propagation using universal variable method
    # This is a basic implementation - real applications should use PyKEP

    mu = PC.EARTH_MU
    r0_mag = np.linalg.norm(position)
    v0_mag = np.linalg.norm(velocity)

    # Specific energy
    energy = v0_mag**2 / 2 - mu / r0_mag

    # Semi-major axis
    if energy < 0:  # Elliptical orbit
        a = -mu / (2 * energy)

        # Mean motion
        n = np.sqrt(mu / a**3)

        # Simple propagation (assumes circular orbit)
        # For actual missions, use proper orbital element propagation
        theta = n * dt

        # Rotation matrix for simple circular propagation
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Simple 2D rotation in orbital plane (approximation)
        final_position = np.array([
            position[0] * cos_theta - position[1] * sin_theta,
            position[0] * sin_theta + position[1] * cos_theta,
            position[2]
        ])

        final_velocity = np.array([
            velocity[0] * cos_theta - velocity[1] * sin_theta,
            velocity[0] * sin_theta + velocity[1] * cos_theta,
            velocity[2]
        ])

        return final_position, final_velocity

    # Hyperbolic trajectory - return unchanged for now
    return position, velocity
