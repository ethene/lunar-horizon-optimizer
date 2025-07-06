"""Input validation functions for trajectory calculations.

This module provides validation functions for all input parameters used in
trajectory generation. Each function raises ValueError with a descriptive
message if validation fails.
"""

from datetime import datetime
from typing import Union, Optional
import pykep as pk
from .constants import TransferDefaults as TD, EphemerisLimits
from .models import OrbitState

def validate_epoch(dt: datetime, allow_none: bool = False) -> None:
    """Validate epoch is within supported range.
    
    Args:
        dt: Datetime to validate
        allow_none: Whether None is acceptable
        
    Raises:
        ValueError: If datetime is outside supported range
        TypeError: If input is not a datetime object
    """
    if dt is None:
        if allow_none:
            return
        raise TypeError("Epoch cannot be None")
        
    if not isinstance(dt, datetime):
        raise TypeError("Epoch must be a datetime object")
        
    year = dt.year
    if year < EphemerisLimits.MIN_YEAR or year > EphemerisLimits.MAX_YEAR:
        raise ValueError(
            f"Epoch must be between years {EphemerisLimits.MIN_YEAR} "
            f"and {EphemerisLimits.MAX_YEAR}"
        )

def validate_orbit_altitude(
    altitude: float,
    min_alt: float = 200.0,  # Minimum safe perigee altitude
    max_alt: float = 400000.0  # Maximum practical apogee altitude
) -> None:
    """Validate orbit altitude is within reasonable range.
    
    Args:
        altitude: Orbit altitude in kilometers
        min_alt: Minimum allowed altitude in kilometers
        max_alt: Maximum allowed altitude in kilometers
        
    Raises:
        ValueError: If altitude is outside allowed range
        TypeError: If altitude is not a number
    """
    if not isinstance(altitude, (int, float)):
        raise TypeError("Altitude must be a number")
        
    if altitude < min_alt or altitude > max_alt:
        raise ValueError(
            f"Orbit altitude must be between {min_alt} and {max_alt} km"
        )

def validate_transfer_parameters(
    tof_days: float,
    max_revs: int,
    min_dv: float = TD.MIN_TLI_DV,
    max_dv: float = TD.MAX_TLI_DV
) -> None:
    """Validate transfer trajectory parameters.
    
    Args:
        tof_days: Time of flight in days
        max_revs: Maximum number of revolutions
        min_dv: Minimum allowed delta-v in km/s
        max_dv: Maximum allowed delta-v in km/s
        
    Raises:
        ValueError: If any parameter is outside valid range
        TypeError: If parameters are not of correct type
    """
    if not isinstance(tof_days, (int, float)):
        raise TypeError("Time of flight must be a number")
    if not isinstance(max_revs, int):
        raise TypeError("Maximum revolutions must be an integer")
        
    if tof_days <= TD.MIN_TOF or tof_days > TD.MAX_TOF:
        raise ValueError(
            f"Time of flight must be between {TD.MIN_TOF} "
            f"and {TD.MAX_TOF} days"
        )
        
    if max_revs < 0 or max_revs > TD.MAX_REVOLUTIONS:
        raise ValueError(
            f"Maximum revolutions must be between 0 and {TD.MAX_REVOLUTIONS}"
        )
        
    if min_dv < 0 or max_dv < min_dv:
        raise ValueError(
            f"Delta-v range must be positive and max ({max_dv:.1f} km/s) must be "
            f"greater than min ({min_dv:.1f} km/s)"
        )

def validate_initial_orbit(orbit: Union[float, OrbitState]) -> None:
    """Validate initial orbit specification.
    
    Args:
        orbit: Either altitude in km or OrbitState object
        
    Raises:
        ValueError: If orbit parameters are invalid
        TypeError: If orbit is not float or OrbitState
    """
    if isinstance(orbit, (int, float)):
        validate_orbit_altitude(orbit)
    elif isinstance(orbit, OrbitState):
        earth_radius = pk.EARTH_RADIUS / 1000.0  # Convert to km
        if orbit.radius <= earth_radius:
            raise ValueError(f"Orbit radius must be greater than Earth radius ({earth_radius:.1f} km)")
        if orbit.velocity <= 0:
            raise ValueError("Orbit velocity must be positive")
    else:
        raise TypeError(
            "Initial orbit must be a float (altitude in km) or OrbitState object"
        )

def validate_final_orbit(
    final_radius: float,
    initial_radius: float
) -> None:
    """Validate final orbit radius.
    
    Args:
        final_radius: Final orbit radius in kilometers
        initial_radius: Initial orbit radius in kilometers
        
    Raises:
        ValueError: If final radius is invalid
        TypeError: If inputs are not numbers
    """
    if not isinstance(final_radius, (int, float)):
        raise TypeError("Final orbit radius must be a number")
        
    if final_radius <= initial_radius:
        raise ValueError(
            f"Final orbit radius ({final_radius:.1f} km) must be greater than "
            f"initial orbit radius ({initial_radius:.1f} km)"
        ) 