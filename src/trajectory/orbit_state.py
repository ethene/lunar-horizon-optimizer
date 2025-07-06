"""Orbital state representation and calculations.

This module provides the OrbitState class for representing and manipulating orbital states,
integrating with PyKEP for calculations and using utility functions from elements.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import pykep as pk
from datetime import datetime
import logging

from .elements import orbital_period, velocity_at_point, mean_to_true_anomaly, true_to_mean_anomaly
from .trajectory_physics import validate_vector_units, validate_basic_orbital_mechanics
from src.utils.unit_conversions import km_to_m, datetime_to_mjd2000, mps_to_kmps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class OrbitState:
    """Represents the state of an orbiting body.
    
    Attributes:
        semi_major_axis: Semi-major axis in kilometers
        eccentricity: Orbital eccentricity (>= 0)
        inclination: Orbital inclination in degrees
        raan: Right ascension of ascending node in degrees
        arg_periapsis: Argument of periapsis in degrees
        true_anomaly: True anomaly in degrees
        epoch: Epoch time (datetime with timezone or MJD2000 float)
    """
    semi_major_axis: float
    eccentricity: float
    inclination: float
    raan: float
    arg_periapsis: float
    true_anomaly: float
    epoch: Optional[Union[datetime, float]] = None
    
    def __post_init__(self):
        """Validate orbital elements after initialization."""
        if self.semi_major_axis <= 0:
            raise ValueError("Semi-major axis must be positive")
        if self.eccentricity < 0:
            raise ValueError("Eccentricity must be non-negative")
        if not 0 <= self.inclination <= 180:
            raise ValueError("Inclination must be in [0,180]")
        if not 0 <= self.raan < 360:
            raise ValueError("RAAN must be between 0 and 360 degrees")
        if not 0 <= self.arg_periapsis < 360:
            raise ValueError("Argument of periapsis must be between 0 and 360 degrees")
        if not 0 <= self.true_anomaly < 360:
            raise ValueError("True anomaly must be between 0 and 360 degrees")
        if isinstance(self.epoch, datetime) and self.epoch.tzinfo is None:
            raise ValueError("Datetime epoch must be timezone-aware")
            
    @property
    def position(self) -> np.ndarray:
        """Calculate the position vector in the inertial frame.

        Returns:
            np.ndarray: Position vector [x, y, z] in kilometers
        """
        # Calculate the semi-latus rectum in km
        p = self.semi_major_axis * (1 - self.eccentricity**2)
        
        # Calculate radius in km
        true_anomaly_rad = np.radians(self.true_anomaly)
        radius = p / (1 + self.eccentricity * np.cos(true_anomaly_rad))
        
        # Position in orbital plane (perifocal frame) in km
        x = radius * np.cos(true_anomaly_rad)
        y = radius * np.sin(true_anomaly_rad)
        r_pqw = np.array([x, y, 0])
        
        # Transform to inertial frame
        return self._get_rotation_matrix() @ r_pqw

    def velocity(self, mu: float) -> np.ndarray:
        """Calculate the velocity vector in the inertial frame.

        Args:
            mu (float): Gravitational parameter in m³/s²

        Returns:
            np.ndarray: Velocity vector [vx, vy, vz] in kilometers per second
        """
        # Use utility function from elements.py
        v_r, v_t = velocity_at_point(
            self.semi_major_axis,
            self.eccentricity,
            self.true_anomaly,
            mu
        )
        
        # Convert to vector in orbital plane
        true_anomaly_rad = np.radians(self.true_anomaly)
        v_pqw = np.array([
            v_r * np.cos(true_anomaly_rad) - v_t * np.sin(true_anomaly_rad),
            v_r * np.sin(true_anomaly_rad) + v_t * np.cos(true_anomaly_rad),
            0
        ])
        
        # Transform to inertial frame
        return self._get_rotation_matrix() @ v_pqw

    def _get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix from perifocal to inertial frame."""
        # Convert orbital elements to radians
        inc = np.radians(self.inclination)
        raan = np.radians(self.raan)
        argp = np.radians(self.arg_periapsis)
        
        # Rotation matrices
        R_argp = np.array([
            [np.cos(argp), -np.sin(argp), 0],
            [np.sin(argp), np.cos(argp), 0],
            [0, 0, 1]
        ])
        
        R_inc = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc), np.cos(inc)]
        ])
        
        R_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        
        # Combine rotation matrices
        return R_raan @ R_inc @ R_argp

    @classmethod
    def from_state_vectors(cls, position: Tuple[float, float, float], 
                         velocity: Tuple[float, float, float], 
                         epoch: Optional[datetime] = None, 
                         mu: float = pk.MU_EARTH) -> 'OrbitState':
        """Create an OrbitState from position and velocity vectors.
        
        Args:
            position: Position vector in kilometers [x, y, z]
            velocity: Velocity vector in kilometers per second [vx, vy, vz]
            epoch: Optional epoch time
            mu: Gravitational parameter (default: Earth's mu in m³/s²)
            
        Returns:
            OrbitState object
            
        Note:
            Input vectors must be in a consistent reference frame (e.g., Earth-centered inertial)
        """
        # Convert inputs to numpy arrays and proper units
        r = np.array(position) * 1000.0  # km to m
        v = np.array(velocity) * 1000.0  # km/s to m/s
        
        # Validate vectors using trajectory_physics
        if not validate_vector_units(r, "position", (1e6, 1e9), "m"):
            raise ValueError("Invalid position vector")
        if not validate_vector_units(v, "velocity", (0, 20000), "m/s"):
            raise ValueError("Invalid velocity vector")
        if not validate_basic_orbital_mechanics(r, v, mu):
            raise ValueError("Invalid orbital state")
        
        # Calculate angular momentum vector
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Unit vectors
        k = np.array([0, 0, 1])  # Reference direction
        n = np.cross(k, h)  # Node vector
        n_mag = np.linalg.norm(n)
        
        # Calculate eccentricity vector
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        e = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        ecc = np.linalg.norm(e)
        
        # Calculate specific orbital energy
        energy = v_mag**2/2 - mu/r_mag
        
        # Semi-major axis
        a = -mu / (2 * energy)
        
        # For hyperbolic orbits, modify to near-parabolic
        if ecc >= 1.0:
            logger.warning(f"Converting hyperbolic orbit (e={ecc:.3f}) to near-parabolic")
            scale_factor = 0.98 / ecc
            v = v * scale_factor
            ecc = 0.98
        
        # Calculate orbital elements
        inc = np.arccos(h[2] / h_mag)
        raan = np.arccos(n[0] / n_mag) if n_mag > 0 else 0.0
        if n[1] < 0:
            raan = 2*np.pi - raan
            
        argp = np.arccos(np.dot(n, e) / (n_mag * ecc)) if n_mag > 0 and ecc > 1e-10 else 0.0
        if e[2] < 0:
            argp = 2*np.pi - argp
            
        ta = np.arccos(np.dot(e, r) / (ecc * r_mag)) if ecc > 1e-10 else np.arccos(np.dot(n, r) / (n_mag * r_mag))
        if np.dot(r, v) < 0:
            ta = 2*np.pi - ta
            
        return cls(
            semi_major_axis=abs(a)/1000.0,
            eccentricity=ecc,
            inclination=np.degrees(inc),
            raan=np.degrees(raan),
            arg_periapsis=np.degrees(argp),
            true_anomaly=np.degrees(ta),
            epoch=epoch
        )
            
    def to_pykep(self, mu: float) -> pk.planet:
        """Convert to PyKEP planet object.
        
        Args:
            mu: Gravitational parameter in m³/s²
            
        Returns:
            PyKEP planet object
        """
        epoch = 0.0  # Default epoch if none provided
        if isinstance(self.epoch, datetime):
            epoch = pk.epoch_from_datetime(self.epoch)
        elif isinstance(self.epoch, float):
            epoch = pk.epoch(self.epoch)
            
        # Convert orbital elements to SI units
        elements = [
            km_to_m(self.semi_major_axis),
            self.eccentricity,
            np.radians(self.inclination),
            np.radians(self.raan),
            np.radians(self.arg_periapsis),
            np.radians(self.true_anomaly)
        ]
        
        return pk.planet.keplerian(
            epoch,
            elements,
            mu,
            0.0,  # Self gravitational parameter
            1.0,  # Radius (not used)
            1.0,  # Safe radius (not used)
            "temp"
        )

    def get_state_vectors(self, mu: float = pk.MU_EARTH) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and velocity vectors.
        
        Args:
            mu: Gravitational parameter in m³/s² (default: Earth's mu)
            
        Returns:
            Tuple of position (m) and velocity (m/s) vectors
        """
        return km_to_m(self.position), km_to_m(self.velocity(mu))

    def get_state_vectors_km(self, mu: float = pk.MU_EARTH) -> Tuple[np.ndarray, np.ndarray]:
        """Get position and velocity vectors in km and km/s.
        
        Args:
            mu: Gravitational parameter in m³/s² (default: Earth's mu)
            
        Returns:
            Tuple of position (km) and velocity (km/s) vectors
        """
        return self.position, self.velocity(mu) 