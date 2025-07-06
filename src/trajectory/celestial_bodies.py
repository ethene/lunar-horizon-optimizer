"""Celestial body definitions and state calculations.

This module provides functionality for obtaining state vectors of celestial bodies
using SPICE. All calculations are performed in the J2000 frame with PyKEP's native units:
- Distances: meters (m)
- Velocities: meters per second (m/s)
- Times: days since J2000 epoch

Note: While SPICE internally uses kilometers and seconds, all values are automatically
converted to PyKEP's native units (meters, m/s) before being returned.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List
from pathlib import Path
import logging
import os
import spiceypy as spice
import pykep as pk
from .constants import PhysicalConstants as PC
from ..utils.unit_conversions import datetime_to_j2000

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define the path to the SPICE kernel relative to the project root
KERNEL_PATH = Path(__file__).parent.parent.parent / "data" / "spice" / "de430.bsp"

# Constants for unit conversion
KM_TO_M = 1000.0  # Convert SPICE kilometers to meters
DAYS_TO_SECONDS = 86400.0  # Convert J2000 days to seconds for SPICE

# Load SPICE kernel at module initialization
try:
    if not KERNEL_PATH.exists():
        raise FileNotFoundError(f"SPICE kernel not found at {KERNEL_PATH}")
    
    logger.info(f"Loading SPICE kernel from {KERNEL_PATH}")
    spice.furnsh(str(KERNEL_PATH))
    logger.info("SPICE kernel loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load SPICE kernel: {e}")
    raise

class CelestialBody:
    """
    Provides methods to calculate state vectors of celestial bodies.
    
    All calculations are performed in the J2000 heliocentric ecliptic frame.
    All methods return values in PyKEP's native units:
        - Positions: meters
        - Velocities: meters/second
        - Times: days since J2000 epoch
        
    Note: SPICE returns positions in km and velocities in km/s, which are
    automatically converted to PyKEP's native units (meters and m/s) before return.
    """
    
    def __init__(self):
        """Initialize a CelestialBody instance."""
        pass

    @staticmethod
    def get_earth_state(epoch: float) -> Tuple[list, list]:
        """Get Earth's heliocentric state vector at the specified epoch.
        
        Args:
            epoch: Time in days since J2000 epoch
            
        Returns:
            Tuple containing:
                - position vector [x, y, z] in meters
                - velocity vector [vx, vy, vz] in meters/second
        """
        if not isinstance(epoch, (int, float)):
            raise TypeError(f"Epoch must be a number, got {type(epoch)}")
            
        try:
            # Convert epoch from days to seconds since J2000 for SPICE
            epoch_seconds = epoch * DAYS_TO_SECONDS
            state = spice.spkezr('EARTH', epoch_seconds, 'J2000', 'NONE', 'SUN')[0]
            # Convert from SPICE units (km, km/s) to PyKEP units (m, m/s)
            pos = [x * KM_TO_M for x in state[:3]]
            vel = [v * KM_TO_M for v in state[3:]]
            return pos, vel
        except Exception as e:
            logger.error(f"Failed to get Earth state at epoch {epoch}: {e}")
            raise

    @staticmethod
    def get_moon_state(epoch: float) -> Tuple[list, list]:
        """Get Moon's heliocentric state vector at the specified epoch.
        
        Args:
            epoch: Time in days since J2000 epoch
            
        Returns:
            Tuple containing:
                - position vector [x, y, z] in meters
                - velocity vector [vx, vy, vz] in meters/second
        """
        if not isinstance(epoch, (int, float)):
            raise TypeError(f"Epoch must be a number, got {type(epoch)}")
            
        try:
            # Convert epoch from days to seconds since J2000 for SPICE
            epoch_seconds = epoch * DAYS_TO_SECONDS
            state = spice.spkezr('MOON', epoch_seconds, 'J2000', 'NONE', 'SUN')[0]
            # Convert from SPICE units (km, km/s) to PyKEP units (m, m/s)
            pos = [x * KM_TO_M for x in state[:3]]
            vel = [v * KM_TO_M for v in state[3:]]
            return pos, vel
        except Exception as e:
            logger.error(f"Failed to get Moon state at epoch {epoch}: {e}")
            raise

    @staticmethod
    def get_moon_state_earth_centered(epoch: float) -> Tuple[list, list]:
        """Get Moon's state vector relative to Earth at the specified epoch.
        
        Args:
            epoch: Time in days since J2000 epoch
            
        Returns:
            Tuple containing:
                - position vector [x, y, z] in meters
                - velocity vector [vx, vy, vz] in meters/second
        """
        if not isinstance(epoch, (int, float)):
            raise TypeError(f"Epoch must be a number, got {type(epoch)}")
            
        try:
            # Convert epoch from days to seconds since J2000 for SPICE
            epoch_seconds = epoch * DAYS_TO_SECONDS
            state = spice.spkezr('MOON', epoch_seconds, 'J2000', 'NONE', 'EARTH')[0]
            # Convert from SPICE units (km, km/s) to PyKEP units (m, m/s)
            pos = [x * KM_TO_M for x in state[:3]]
            vel = [v * KM_TO_M for v in state[3:]]
            return pos, vel
        except Exception as e:
            logger.error(f"Failed to get Moon state relative to Earth at epoch {epoch}: {e}")
            raise

    @staticmethod
    def create_local_frame(r: np.ndarray, v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a local orbital reference frame.
        
        Args:
            r: Position vector defining primary direction (in meters)
            v: Optional velocity vector for secondary direction (in m/s)
            
        Returns:
            Tuple of unit vectors (x_hat, y_hat, z_hat) defining the frame
            where:
            - x_hat is along the position vector
            - z_hat is along the angular momentum (if v provided)
            - y_hat completes the right-handed system
        """
        r_unit = r / np.linalg.norm(r)
        
        if v is not None:
            h = np.cross(r, v)
            z_unit = h / np.linalg.norm(h)
            y_unit = np.cross(z_unit, r_unit)
        else:
            # If no velocity provided, use z-axis as reference
            z_unit = np.array([0, 0, 1])
            y_unit = np.cross(z_unit, r_unit)
            if np.allclose(y_unit, 0):
                # If r is along z-axis, use x-axis as reference
                z_unit = np.array([1, 0, 0])
                y_unit = np.cross(z_unit, r_unit)
            
        y_unit = y_unit / np.linalg.norm(y_unit)
        z_unit = np.cross(r_unit, y_unit)
        
        return r_unit, y_unit, z_unit

# Initialize class instances for common bodies
EARTH = CelestialBody()
MOON = CelestialBody() 