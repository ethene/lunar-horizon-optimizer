"""Target state calculations for lunar transfer trajectories.

This module handles the calculation of target states for lunar orbit insertion,
including position and velocity calculations relative to the Moon.
"""

import logging

import numpy as np

from .constants import PhysicalConstants as PC

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_target_state(
    moon_pos: np.ndarray,
    moon_vel: np.ndarray,
    orbit_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate target state for lunar orbit insertion.

    Args:
        moon_pos: Moon position vector [m]
        moon_vel: Moon velocity vector [m/s]
        orbit_radius: Target orbit radius around Moon [m]

    Returns
    -------
        Tuple containing:
            - Target position vector [m]
            - Target velocity vector [m/s]

    Raises
    ------
        ValueError: If inputs are invalid
    """
    # Log input values with more detail
    logger.debug(f"Input moon_pos [m]: {moon_pos}, magnitude: {np.linalg.norm(moon_pos):.2f}")
    logger.debug(f"Input moon_vel [m/s]: {moon_vel}, magnitude: {np.linalg.norm(moon_vel):.2f}")
    logger.debug(f"Input orbit_radius [m]: {orbit_radius}")

    # Convert inputs to numpy arrays and validate
    moon_pos = np.array(moon_pos, dtype=float)
    moon_vel = np.array(moon_vel, dtype=float)

    # Validate vector shapes
    if moon_pos.shape != (3,) or moon_vel.shape != (3,):
        msg = "Position and velocity vectors must be 3D"
        raise ValueError(msg)

    # Calculate unit vectors and magnitudes
    moon_pos_mag = np.linalg.norm(moon_pos)
    moon_vel_mag = np.linalg.norm(moon_vel)
    moon_pos_unit = moon_pos / moon_pos_mag
    moon_vel_unit = moon_vel / moon_vel_mag

    logger.debug(f"Moon position magnitude [m]: {moon_pos_mag}")
    logger.debug(f"Moon velocity magnitude [m/s]: {moon_vel_mag}")

    # Calculate Moon's orbit normal (defines the orbit plane)
    orbit_normal = np.cross(moon_pos_unit, moon_vel_unit)
    orbit_normal = orbit_normal / np.linalg.norm(orbit_normal)
    logger.debug(f"Orbit normal vector: {orbit_normal}")

    # Calculate target position (5 degrees ahead in orbit)
    lead_angle = np.radians(5)
    rotation_matrix = _rotation_matrix(orbit_normal, lead_angle)
    target_radial = rotation_matrix @ moon_pos_unit

    # Position relative to Moon's center
    target_pos_rel_moon = target_radial * orbit_radius
    # Convert to Earth-centered position
    target_pos = moon_pos + target_pos_rel_moon

    # Calculate circular orbit velocity around Moon
    v_circ = np.sqrt(PC.MOON_MU / orbit_radius)

    # Calculate velocity direction for circular orbit around Moon
    target_vel_dir = np.cross(orbit_normal, target_radial)
    target_vel_dir = target_vel_dir / np.linalg.norm(target_vel_dir)

    # Calculate velocity relative to Moon
    target_vel_rel_moon = target_vel_dir * v_circ

    # Add small radial component for capture (2 m/s inward)
    target_vel_rel_moon -= target_radial * 2.0

    # Convert to Earth-centered velocity by adding Moon's velocity
    target_vel = moon_vel + target_vel_rel_moon

    # Log final states
    logger.debug(f"Target position relative to Moon [m]: {target_pos_rel_moon}")
    logger.debug(f"Target position Earth-centered [m]: {target_pos}")
    logger.debug(f"Target velocity relative to Moon [m/s]: {target_vel_rel_moon}")
    logger.debug(f"Target velocity Earth-centered [m/s]: {target_vel}")

    # Verify target position is at correct distance from Moon
    target_moon_dist = np.linalg.norm(target_pos - moon_pos)
    logger.debug(f"Target-Moon distance [m]: {target_moon_dist}")

    if not np.isclose(target_moon_dist, orbit_radius, rtol=1e-6):
        logger.error(f"Target position is not at requested orbit radius: {target_moon_dist/1000:.1f} km vs {orbit_radius/1000:.1f} km")

    # Log velocity difference components
    vel_diff = target_vel - moon_vel
    vel_diff_mag = np.linalg.norm(vel_diff)
    vel_diff_radial = np.dot(vel_diff, target_radial)
    vel_diff_normal = np.dot(vel_diff, orbit_normal)
    vel_diff_tangential = np.dot(vel_diff, np.cross(orbit_normal, target_radial))

    logger.debug(f"Velocity difference components [m/s]: radial={vel_diff_radial:.2f}, normal={vel_diff_normal:.2f}, tangential={vel_diff_tangential:.2f}")
    logger.debug(f"Total velocity difference [m/s]: {vel_diff_mag:.2f}")

    if vel_diff_mag > 100:
        logger.warning(f"Target velocity differs from Moon velocity by {vel_diff_mag:.1f} m/s (> 100 m/s)")

    return target_pos, target_vel

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix for rotating around axis by angle.

    Uses Rodriguez rotation formula.

    Args:
        axis: Unit vector of rotation axis
        angle: Rotation angle in radians

    Returns
    -------
        3x3 rotation matrix
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    # Rodriguez rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
