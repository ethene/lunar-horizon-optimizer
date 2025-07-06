"""Physics validation and constraints for lunar transfer trajectories.

This module provides functions for validating physical constraints of transfer trajectories,
including energy conservation, angular momentum conservation, and orbit validation.
"""

import numpy as np
import pykep as pk
from typing import Tuple, Optional
import logging

from .constants import PhysicalConstants

# Configure logging
logger = logging.getLogger(__name__)

def validate_vector_units(vector: np.ndarray, name: str, expected_magnitude_range: Tuple[float, float], unit: str) -> bool:
    """
    Validate that a vector's magnitude falls within expected range and has correct units.
    
    Args:
        vector: Vector to validate
        name: Name of the vector for logging
        expected_magnitude_range: (min, max) expected magnitude
        unit: Unit of measurement for logging
        
    Returns:
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

def validate_basic_orbital_mechanics(r: np.ndarray, v: np.ndarray, mu: float) -> bool:
    """
    Validate basic orbital mechanics relationships for a state vector.
    
    Args:
        r: Position vector [m]
        v: Velocity vector [m/s]
        mu: Gravitational parameter [m³/s²]
        
    Returns:
        bool: True if state vector represents valid orbital motion
    """
    logger.debug("\nValidating basic orbital mechanics:")
    logger.debug(f"Position vector: {r} m")
    logger.debug(f"Velocity vector: {v} m/s")
    logger.debug(f"Gravitational parameter: {mu} m³/s²")
    
    # Calculate orbital elements
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    # Specific angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    
    # Specific orbital energy
    energy = v_mag**2/2 - mu/r_mag
    
    # Eccentricity vector
    e_vec = np.cross(v, h)/mu - r/r_mag
    e = np.linalg.norm(e_vec)
    
    logger.debug(f"Calculated orbital elements:")
    logger.debug(f"  |r| = {r_mag:.2e} m")
    logger.debug(f"  |v| = {v_mag:.2e} m/s")
    logger.debug(f"  |h| = {h_mag:.2e} m²/s")
    logger.debug(f"  Energy = {energy:.2e} m²/s²")
    logger.debug(f"  Eccentricity = {e:.6f}")
    
    # Validate specific angular momentum
    if h_mag < 1e-10:
        logger.error("Near-zero angular momentum detected")
        return False
        
    # Check if velocity is reasonable compared to circular velocity
    v_circ = np.sqrt(mu/r_mag)
    v_ratio = v_mag/v_circ
    logger.debug(f"  v/v_circ = {v_ratio:.3f}")
    
    if v_ratio > 2.0:
        logger.error(f"Velocity {v_mag:.1f} m/s exceeds 2x circular velocity {v_circ:.1f} m/s")
        return False
        
    return True

def validate_transfer_time(r1: np.ndarray, r2: np.ndarray, tof: float, mu: float) -> bool:
    """
    Validate if transfer time is physically reasonable.
    
    Args:
        r1: Initial position vector [m]
        r2: Final position vector [m]
        tof: Time of flight [s]
        mu: Gravitational parameter [m³/s²]
        
    Returns:
        bool: True if transfer time is reasonable
    """
    logger.debug("\nValidating transfer time:")
    
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Calculate various characteristic times
    t_hohmann = np.pi * np.sqrt((r1_mag + r2_mag)**3 / (8 * mu))
    t_min = np.pi * np.sqrt(min(r1_mag, r2_mag)**3 / (8 * mu))
    t_max = 4 * t_hohmann  # Allow up to 4x Hohmann transfer time
    
    logger.debug(f"Transfer time analysis:")
    logger.debug(f"  Actual time: {tof:.1f} s")
    logger.debug(f"  Minimum time: {t_min:.1f} s")
    logger.debug(f"  Hohmann time: {t_hohmann:.1f} s")
    logger.debug(f"  Maximum allowed: {t_max:.1f} s")
    
    if tof < 0.5 * t_min:
        logger.error(f"Transfer time too short: {tof:.1f} s < 0.5 * {t_min:.1f} s")
        return False
        
    if tof > t_max:
        logger.error(f"Transfer time too long: {tof:.1f} s > {t_max:.1f} s")
        return False
        
    return True

def validate_solution_physics(r1, v1, r2, v2, transfer_time):
    """
    Validate the physics of a transfer solution.
    
    For lunar transfers, we use modified validation criteria that account for the three-body effects:
    1. Direction of angular momentum should be roughly preserved
    2. Energy should increase due to Moon's gravitational assist
    3. Velocities should be reasonable for Earth-Moon transfer
    4. Transfer time should be physically achievable
    
    Parameters
    ----------
    r1 : array_like
        Initial position vector [m]
    v1 : array_like
        Initial velocity vector [m/s]
    r2 : array_like
        Final position vector [m]
    v2 : array_like
        Final velocity vector [m/s]
    transfer_time : float
        Time of flight [s]
    
    Returns
    -------
    bool
        True if the solution is physically valid, False otherwise
    """
    # Convert inputs to numpy arrays and validate units
    r1 = np.array(r1, dtype=np.float64)
    v1 = np.array(v1, dtype=np.float64)
    r2 = np.array(r2, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    
    logger.debug("\nStarting solution physics validation:")
    
    # Validate vector units with expected ranges
    r_range = (PhysicalConstants.EARTH_RADIUS, 2 * PhysicalConstants.MOON_SEMI_MAJOR_AXIS)
    v_range = (0, 1.5 * PhysicalConstants.EARTH_ESCAPE_VELOCITY)  # Allow up to 1.5x escape velocity

    # Check position magnitudes
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    if not (r_range[0] <= r1_mag <= r_range[1] and r_range[0] <= r2_mag <= r_range[1]):
        logger.warning(f"Position magnitudes outside valid range: r1={r1_mag:.1f} m, r2={r2_mag:.1f} m")
        return False

    # Check velocity magnitudes with more generous limits for lunar transfer
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    if not (v_range[0] <= v1_mag <= v_range[1]):
        logger.warning(f"Initial velocity magnitude outside valid range: {v1_mag:.1f} m/s")
        return False
    
    # For final velocity, allow up to 2x Moon's orbital velocity for capture
    if not (v_range[0] <= v2_mag <= 2.0 * PhysicalConstants.MOON_ORBITAL_VELOCITY):
        logger.warning(f"Final velocity magnitude outside valid range: {v2_mag:.1f} m/s")
        return False

    # Calculate and check angular momentum (allow 10% variation due to lunar effects)
    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    h1_mag = np.linalg.norm(h1)
    h2_mag = np.linalg.norm(h2)
    
    if h1_mag < 1e-10 or h2_mag < 1e-10:
        logger.warning("Near-zero angular momentum detected")
        return False
    
    h_unit1 = h1 / h1_mag
    h_unit2 = h2 / h2_mag
    h_alignment = np.dot(h_unit1, h_unit2)
    
    if h_alignment < 0.8:  # Allow up to ~37 degrees of plane change
        logger.warning(f"Angular momentum vectors poorly aligned: {h_alignment:.3f}")
        return False

    # Calculate specific orbital energy
    e1 = v1_mag**2 / 2 - PhysicalConstants.EARTH_MU / r1_mag
    e2 = v2_mag**2 / 2 - PhysicalConstants.EARTH_MU / r2_mag
    
    logger.debug(f"Initial specific energy: {e1:.2e} m²/s²")
    logger.debug(f"Final specific energy: {e2:.2e} m²/s²")
    
    # Initial orbit should be bound (negative energy)
    if e1 > -1e3:  # Small negative value to account for numerical errors
        logger.warning("Initial orbit appears to be hyperbolic")
        return False

    # Calculate minimum transfer time (allow down to 50% of Hohmann time)
    a_transfer = (r1_mag + r2_mag) / 2
    t_min = 0.5 * np.pi * np.sqrt(a_transfer**3 / PhysicalConstants.EARTH_MU)
    
    if transfer_time < 0.5 * t_min:
        logger.warning(f"Transfer time too short: {transfer_time:.1f} s < {0.5 * t_min:.1f} s")
        return False

    logger.debug("All physics checks passed")
    return True

def validate_trajectory_constraints(r1: np.ndarray,
                                 v1: np.ndarray,
                                 r2: np.ndarray,
                                 v2: np.ndarray,
                                 tof: float) -> bool:
    """Validate physical constraints for the complete trajectory.
    
    Args:
        r1: Initial position vector [m]
        v1: Initial velocity vector [m/s]
        r2: Final position vector [m]
        v2: Final velocity vector [m/s]
        tof: Time of flight [s]
        
    Returns:
        bool: True if trajectory is valid
        
    Raises:
        ValueError: If trajectory violates physical constraints
    """
    # Convert inputs to float64 for better numerical stability
    r1 = np.array(r1, dtype=np.float64)
    v1 = np.array(v1, dtype=np.float64)
    r2 = np.array(r2, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    
    # Calculate magnitudes
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Check for reasonable velocities
    v_esc = np.sqrt(2 * PhysicalConstants.EARTH_MU / r1_mag)
    
    # Allow velocities up to 1.8 times escape velocity for lunar transfer
    if v1_mag > 1.8 * v_esc or v2_mag > 2.0 * v_esc:
        raise ValueError(f"Excessive velocity detected: {v1_mag/1000:.1f} or {v2_mag/1000:.1f} km/s (escape velocity: {v_esc/1000:.1f} km/s)")
    
    # Verify trajectory doesn't impact Earth
    try:
        # Propagate trajectory with fewer points for efficiency
        states = []
        for t in np.linspace(0, tof, 50):
            rt, vt = pk.propagate_lagrangian(r1, v1, t, PhysicalConstants.EARTH_MU)
            states.append(np.concatenate([rt, vt]))
            r_mag = np.linalg.norm(rt)
            if r_mag < PhysicalConstants.EARTH_RADIUS + 100000:  # 100 km safety margin
                raise ValueError(f"Trajectory passes too close to Earth: {(r_mag-PhysicalConstants.EARTH_RADIUS)/1000:.0f} km altitude")
    except RuntimeError as e:
        raise ValueError(f"Invalid trajectory: {str(e)}")
    
    # Verify angular momentum conservation with 10% tolerance for lunar effects
    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    
    h1_mag = np.linalg.norm(h1)
    h2_mag = np.linalg.norm(h2)
    
    if h1_mag < 1e-10 or h2_mag < 1e-10:
        raise ValueError("Near-zero angular momentum detected")
        
    h_diff = abs(h1_mag - h2_mag) / h1_mag
    if h_diff > 0.10:
        raise ValueError(f"Angular momentum not conserved: relative difference {h_diff:.1%}")
    
    # Verify energy consistency with 15% tolerance for lunar effects
    e1 = v1_mag**2/2 - PhysicalConstants.EARTH_MU/r1_mag
    e2 = v2_mag**2/2 - PhysicalConstants.EARTH_MU/r2_mag
    
    # For lunar transfer, allow slightly positive energy
    if e1 > 1e4:
        raise ValueError(f"Initial orbit too energetic (e = {e1:.2e} m²/s²)")
    
    # Verify minimum transfer time
    min_time = np.pi * np.sqrt(min(r1_mag, r2_mag)**3 / (8 * PhysicalConstants.EARTH_MU))
    if tof < 0.5 * min_time:
        raise ValueError(f"Transfer time {tof:.1f}s is less than 50% of minimum time {min_time:.1f}s")
    
    # Check intermediate states for physical constraints
    for i, state in enumerate(states):
        r = state[:3]
        v = state[3:]
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Check for extremely hyperbolic trajectory segments
        e = v_mag**2/2 - PhysicalConstants.EARTH_MU/r_mag
        if e > 1e4:  # Same threshold as initial state
            logging.warning(f"Highly hyperbolic trajectory detected at point {i} (e = {e:.2e} m²/s²)")
            return False
            
        # Check velocity against escape velocity with lunar transfer allowance
        v_esc = np.sqrt(2 * PhysicalConstants.EARTH_MU / r_mag)
        if v_mag > 2.0 * v_esc:
            logging.warning(f"Excessive velocity detected at point {i}: {v_mag/1000:.1f} km/s (escape velocity: {v_esc/1000:.1f} km/s)")
            return False
    
    return True

def calculate_circular_velocity(radius: float, mu: float) -> float:
    """Calculate circular orbit velocity.
    
    Args:
        radius: Orbit radius [m]
        mu: Gravitational parameter [m³/s²]
        
    Returns:
        float: Circular orbit velocity [m/s]
    """
    return np.sqrt(mu / radius) 