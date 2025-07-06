"""Phase angle optimization for lunar transfer trajectories.

This module handles the optimization of departure phase angles for lunar transfers,
including initial position calculation and transfer solution evaluation.
"""

import numpy as np
from typing import Tuple, Optional
import pykep as pk
import logging

from .trajectory_physics import validate_solution_physics
from .constants import PhysicalConstants as PC
from .target_state import calculate_target_state
from .lambert_solver import solve_lambert
from ..utils.unit_conversions import m_to_km, km_to_m, m3s2_to_km3s2, km3s2_to_m3s2

# Configure logging
logger = logging.getLogger(__name__)

def calculate_initial_position(r_park: float,
                             phase: float,
                             moon_h_unit: np.ndarray) -> np.ndarray:
    """Calculate initial position vector for given phase angle.
    
    Args:
        r_park: Parking orbit radius [m]
        phase: Phase angle [rad]
        moon_h_unit: Moon's orbital angular momentum unit vector
        
    Returns:
        np.ndarray: Initial position vector [m]
    """
    # Ensure input is numpy array with correct shape
    moon_h_unit = np.asarray(moon_h_unit).reshape(3)
    
    # Calculate reference direction in orbital plane
    ref_dir = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref_dir, moon_h_unit)) > 0.9:
        ref_dir = np.array([0.0, 1.0, 0.0])
    
    # Get radial vector in orbital plane
    try:
        radial = np.cross(moon_h_unit, ref_dir)
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 1e-10:
            raise ValueError("Failed to calculate radial vector")
        radial = radial / radial_norm
    except Exception as e:
        raise ValueError(f"Failed to calculate radial vector: {str(e)}")
    
    # Calculate initial position
    pos = r_park * (np.cos(phase) * ref_dir + np.sin(phase) * radial)
    return pos

def evaluate_transfer_solution(
    r1: np.ndarray,
    moon_pos: np.ndarray,
    moon_vel: np.ndarray,
    transfer_time: float,
    orbit_radius: float,
    max_revs: int = 0
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate a transfer solution for given initial conditions.
    
    Args:
        r1: Initial position vector [m]
        moon_pos: Moon position vector [m]
        moon_vel: Moon velocity vector [m/s]
        transfer_time: Transfer time [s]
        orbit_radius: Target orbit radius around Moon [m]
        max_revs: Maximum number of revolutions (default: 0)
        
    Returns:
        Tuple containing:
        - Total delta-v [m/s]
        - Initial velocity vector [m/s]
        - Final velocity vector [m/s]
        
    If no valid solution is found, returns (inf, None, None)
    """
    logger.debug("Evaluating transfer solution:")
    logger.debug(f"Initial position [m]: {r1}")
    logger.debug(f"Moon position [m]: {moon_pos}")
    logger.debug(f"Moon velocity [m/s]: {moon_vel}")
    logger.debug(f"Transfer time [s]: {transfer_time}")
    logger.debug(f"Target orbit radius [m]: {orbit_radius}")
    logger.debug(f"Max revolutions: {max_revs}")

    # Validate inputs
    r1 = np.array(r1, dtype=float)
    moon_pos = np.array(moon_pos, dtype=float)
    moon_vel = np.array(moon_vel, dtype=float)
    
    if r1.shape != (3,) or moon_pos.shape != (3,):
        raise ValueError("Position vectors must be 3D")

    # Calculate magnitudes for validation
    r1_mag = np.linalg.norm(r1)
    moon_pos_mag = np.linalg.norm(moon_pos)
    logger.debug(f"Initial position magnitude [m]: {r1_mag}")
    logger.debug(f"Moon position magnitude [m]: {moon_pos_mag}")

    # Calculate target state for arrival
    target_state = calculate_target_state(moon_pos, moon_vel, orbit_radius)
    r2, v2_target = target_state
    logger.debug(f"Target position [m]: {r2}")
    logger.debug(f"Target velocity [m/s]: {v2_target}")

    # Calculate parking orbit velocity
    mu_earth = 3.986004418e14  # Earth's gravitational parameter [m³/s²]
    v_circ = np.sqrt(mu_earth / r1_mag)  # Circular orbit velocity
    
    # Calculate parking orbit plane normal (assuming it's in Earth's equatorial plane)
    r1_unit = r1 / r1_mag
    v_park_dir = np.array([-r1_unit[1], r1_unit[0], 0.0])  # Perpendicular to r1 in xy-plane
    v_park_dir = v_park_dir / np.linalg.norm(v_park_dir)
    v_park = v_park_dir * v_circ
    
    logger.debug(f"Parking orbit velocity [m/s]: {v_park}")
    logger.debug(f"Parking orbit velocity magnitude [m/s]: {v_circ:.2f}")

    try:
        # Convert to km for Lambert solver
        r1_km = r1 / 1000.0
        r2_km = r2 / 1000.0
        mu_km = mu_earth / (1000.0 * 1000.0 * 1000.0)  # Convert to km³/s²
        
        logger.debug("Lambert solver inputs:")
        logger.debug(f"r1 [km]: {r1_km}")
        logger.debug(f"r2 [km]: {r2_km}")
        logger.debug(f"mu [km³/s²]: {mu_km}")
        
        # Get Lambert solutions
        lambert = pk.lambert_problem(
            r1_km, r2_km,
            transfer_time,
            mu_km,
            False,  # retrograde flag
            max_revs
        )
        
        # Get all v1 and v2 vectors
        v1_vectors = lambert.get_v1()
        v2_vectors = lambert.get_v2()
        num_solutions = len(v1_vectors)
        logger.debug(f"Number of Lambert solutions: {num_solutions}")
        
        # Initialize best solution tracking
        min_dv = float('inf')
        best_v1 = None
        best_v2 = None
        
        # Evaluate each solution
        for i in range(num_solutions):
            v1_km = np.array(v1_vectors[i])  # Convert to numpy array
            v2_km = np.array(v2_vectors[i])  # Convert to numpy array
            logger.debug(f"\nEvaluating solution {i+1}:")
            
            # Convert velocities back to m/s
            v1 = v1_km * 1000.0
            v2 = v2_km * 1000.0
            
            logger.debug(f"v1 [m/s]: {v1}")
            logger.debug(f"v2 [m/s]: {v2}")
            
            # Calculate delta-v components
            dv1 = np.linalg.norm(v1 - v_park)  # Departure delta-v
            dv2 = np.linalg.norm(v2 - v2_target)  # Arrival delta-v
            total_dv = dv1 + dv2
            
            logger.debug(f"Delta-v components [m/s]: {dv1:.1f} (departure), {dv2:.1f} (arrival)")
            logger.debug(f"Total delta-v [m/s]: {total_dv:.1f}")
            
            # Check if this is the best solution so far
            if total_dv < min_dv:
                min_dv = total_dv
                best_v1 = v1
                best_v2 = v2
            
            # Skip if delta-v is clearly too high
            if total_dv > 5000:  # Reduced from previous 8000 m/s limit
                logger.debug("Skipping solution - delta-v exceeds limit")
                continue
                
            # Validate solution physics
            r1_v1_angle = np.arccos(np.dot(r1/r1_mag, v1/np.linalg.norm(v1)))
            if not (0.1 < r1_v1_angle < np.pi - 0.1):
                logger.debug(f"Invalid departure angle: {np.degrees(r1_v1_angle):.1f} degrees")
                continue
            
            return total_dv, v1, v2
            
        if min_dv == float('inf'):
            logger.warning("No valid solutions found")
            return float('inf'), None, None
            
    except (ValueError, RuntimeError) as e:
        logger.error(f"Lambert solver failed: {str(e)}")
        return float('inf'), None, None
        
    return min_dv, best_v1, best_v2

def find_optimal_phase(
    r_park: float,
    moon_pos: np.ndarray,
    moon_vel: np.ndarray,
    transfer_time: float,
    orbit_radius: float,
    max_revs: int = 0,
    num_samples: int = 360
) -> Tuple[float, np.ndarray]:
    """Find the optimal phase angle for lunar transfer departure.

    Args:
        r_park: Parking orbit radius [m]
        moon_pos: Moon position vector [m]
        moon_vel: Moon velocity vector [m/s]
        transfer_time: Transfer time [s]
        orbit_radius: Target orbit radius around Moon [m]
        max_revs: Maximum number of revolutions (default: 0)
        num_samples: Number of phase angles to sample (default: 360)

    Returns:
        Tuple containing:
            - Optimal phase angle [rad]
            - Initial position vector [m]

    Raises:
        ValueError: If no valid transfer trajectory is found
    """
    # Convert inputs to numpy arrays
    moon_pos = np.array(moon_pos)
    moon_vel = np.array(moon_vel)

    # Calculate moon's orbital angular momentum unit vector
    h_unit = np.cross(moon_pos, moon_vel)
    h_unit = h_unit / np.linalg.norm(h_unit)

    # Sample phase angles uniformly
    phase_angles = np.linspace(0, 2*np.pi, num_samples)
    best_dv = float('inf')
    best_phase = None
    best_r1 = None

    for phase in phase_angles:
        # Calculate initial position for this phase angle
        r1 = calculate_initial_position(r_park, phase, h_unit)
        
        # Evaluate transfer solution
        dv, v1, v2 = evaluate_transfer_solution(
            r1, moon_pos, moon_vel, transfer_time, orbit_radius, max_revs
        )

        if dv < best_dv:
            best_dv = dv
            best_phase = phase
            best_r1 = r1

    if best_phase is None:
        raise ValueError("No valid transfer trajectory found")

    return best_phase, best_r1

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix for rotating around axis by angle.

    Uses Rodriguez rotation formula.

    Args:
        axis: Unit vector of rotation axis
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    # Rodriguez rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R
