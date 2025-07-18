"""Default values and limits for trajectory calculations.

This module defines default values and limits used in trajectory generation
and optimization. All values are in SI units (meters, m/s) unless otherwise noted.
"""

from src.utils.unit_conversions import m_to_km

from .constants import PhysicalConstants as PC


class TransferDefaults:
    """Default values and limits for transfer trajectory generation.

    All distances are in kilometers and velocities in km/s for user interface.
    Internal calculations convert to meters and m/s.
    """

    # Earth parameters (km)
    EARTH_RADIUS = m_to_km(PC.EARTH_RADIUS)  # Convert m to km

    # Default orbit altitudes (km)
    DEFAULT_EARTH_ORBIT = 300.0  # km
    DEFAULT_MOON_ORBIT = 100.0  # km

    # Orbit altitude limits (km)
    MIN_EARTH_ORBIT = 200.0  # km
    MAX_EARTH_ORBIT = 1000.0  # km
    MIN_MOON_ORBIT = 50.0  # km
    MAX_MOON_ORBIT = 500.0  # km

    # Transfer time limits (days)
    MIN_TOF = 2.0  # days minimum time of flight
    MAX_TOF = 7.0  # days maximum time of flight
    MIN_TRANSFER_TIME = 2.0  # days (legacy)
    MAX_TRANSFER_TIME = 7.0  # days (legacy)
    DEFAULT_TRANSFER_TIME = 4.0  # days

    # Delta-v limits (m/s)
    MIN_TLI_DV = 2000.0  # m/s minimum for Trans-Lunar Injection
    MAX_TLI_DV = 3500.0  # m/s maximum for Trans-Lunar Injection
    MAX_TLI_DELTA_V = 3500.0  # m/s for Trans-Lunar Injection (legacy)
    MAX_LOI_DELTA_V = 1200.0  # m/s for Lunar Orbit Insertion

    # Moon parameters (km)
    MOON_SOI_RADIUS = m_to_km(PC.MOON_SOI)  # Convert m to km
    MOON_MEAN_RADIUS = m_to_km(PC.MOON_RADIUS)  # Convert m to km

    # Maximum number of revolutions for Lambert solver
    MAX_REVOLUTIONS = 1

    # Default inclination for parking orbits (degrees)
    DEFAULT_INCLINATION = 28.5  # degrees (approximately Kennedy Space Center latitude)

    @classmethod
    def validate_earth_orbit(cls, altitude_km: float) -> None:
        """Validate Earth orbit altitude.

        Args:
            altitude_km: Orbit altitude in kilometers

        Raises
        ------
            ValueError: If altitude is outside valid range
        """
        if not cls.MIN_EARTH_ORBIT <= altitude_km <= cls.MAX_EARTH_ORBIT:
            msg = (
                f"Earth orbit altitude must be between {cls.MIN_EARTH_ORBIT} "
                f"and {cls.MAX_EARTH_ORBIT} km"
            )
            raise ValueError(
                msg,
            )

    @classmethod
    def validate_moon_orbit(cls, altitude_km: float) -> None:
        """Validate lunar orbit altitude.

        Args:
            altitude_km: Orbit altitude in kilometers

        Raises
        ------
            ValueError: If altitude is outside valid range
        """
        if not cls.MIN_MOON_ORBIT <= altitude_km <= cls.MAX_MOON_ORBIT:
            msg = (
                f"Lunar orbit altitude must be between {cls.MIN_MOON_ORBIT} "
                f"and {cls.MAX_MOON_ORBIT} km"
            )
            raise ValueError(
                msg,
            )

    @classmethod
    def validate_transfer_time(cls, days: float) -> None:
        """Validate transfer time.

        Args:
            days: Transfer time in days

        Raises
        ------
            ValueError: If time is outside valid range
        """
        if not cls.MIN_TRANSFER_TIME <= days <= cls.MAX_TRANSFER_TIME:
            msg = (
                f"Transfer time must be between {cls.MIN_TRANSFER_TIME} "
                f"and {cls.MAX_TRANSFER_TIME} days"
            )
            raise ValueError(
                msg,
            )
