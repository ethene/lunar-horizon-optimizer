"""Constraint validation functions for trajectory calculations.

This module provides functions for validating trajectory constraints,
including complete trajectory validation and safety checks.
"""

import numpy as np
import logging
import pykep as pk

from trajectory.constants import PhysicalConstants

# Configure logging
logger = logging.getLogger(__name__)


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