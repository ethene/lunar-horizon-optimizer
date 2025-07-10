"""Physical constants and unit definitions for trajectory calculations.

This module provides a centralized source of constants used throughout the trajectory
calculations. Values are sourced from PyKEP and Astropy libraries to ensure consistency
with established standards.

Recommended Usage:
- For physical constants, prefer Astropy constants (astropy.constants):
  - Earth constants: GM_earth, R_earth, M_earth
  - General constants: G (gravitational constant), au (astronomical unit)
  Note: Moon constants are not available in Astropy and are sourced from JPL DE440

- For astrodynamics calculations, use PyKEP constants (pykep):
  - Angular conversions: pk.DEG2RAD (0.017453), pk.RAD2DEG (57.29578)
  - Time conversions: pk.DAY2SEC (86400), pk.SEC2DAY, pk.DAY2YEAR
  - Earth-specific: pk.EARTH_RADIUS (6378137 m), pk.EARTH_J2, pk.MU_EARTH
  - Solar System: pk.AU (149597870691 m), pk.MU_SUN
  - Other: pk.G0 (9.80665 m/s²)

Unit Conventions:
- Distances: meters (m)
- Velocities: meters per second (m/s)
- Times: seconds (s)
- Angles: radians (rad)

Reference Frames:
- Positions and velocities are in Earth-centered inertial (ECI) frame
- Time is referenced to J2000 epoch (2000-01-01 12:00:00 UTC)
"""

import numpy as np
import pykep as pk


class Units:
    """Unit conversion constants from PyKEP.

    Note: Prefer using PyKEP's built-in conversions where available:
    - pk.DEG2RAD (0.017453) for degrees to radians
    - pk.RAD2DEG (57.29578) for radians to degrees
    - pk.DAY2SEC (86400) for days to seconds
    - pk.SEC2DAY (1.1574e-5) for seconds to days
    - pk.DAY2YEAR (0.002738) for days to years
    """

    # Angular conversions
    DEG2RAD = pk.DEG2RAD  # degrees to radians
    RAD2DEG = pk.RAD2DEG  # radians to degrees

    # Distance conversions (SI to PyKEP)
    M2KM = 1e-3  # meters to kilometers
    KM2M = 1000.0  # kilometers to meters

    # Velocity conversions
    MS2KMS = 1e-3  # m/s to km/s
    KMS2MS = KM2M  # km/s to m/s

    # Time conversions
    DAYS2SEC = pk.DAY2SEC  # days to seconds
    SEC2DAYS = pk.SEC2DAY  # seconds to days


class PhysicalConstants:
    """Physical constants in PyKEP native units.

    Available constants from PyKEP:
    - pk.MU_EARTH (3.986004418e14 m³/s²): Earth gravitational parameter
    - pk.EARTH_RADIUS (6378137 m): Earth equatorial radius
    - pk.EARTH_J2 (0.00108263): Earth's J2 gravitational harmonic
    - pk.MU_SUN (1.32712440018e20 m³/s²): Solar gravitational parameter
    - pk.AU (149597870691 m): Astronomical Unit
    - pk.G0 (9.80665 m/s²): Standard gravity

    Moon constants are sourced from JPL DE440 ephemeris as they're not in PyKEP.
    """

    # Common trajectory limits
    MIN_PERIGEE = 200e3  # m (minimum perigee altitude)
    MAX_APOGEE = 1000e3  # m (maximum apogee altitude)

    # Earth constants (using PyKEP values for consistency with the library)
    MU_EARTH = pk.MU_EARTH  # m^3/s^2
    EARTH_MU = MU_EARTH  # Alias for backward compatibility
    EARTH_RADIUS = pk.EARTH_RADIUS  # m
    EARTH_ESCAPE_VELOCITY = np.sqrt(2 * MU_EARTH / EARTH_RADIUS)  # m/s
    EARTH_ORBITAL_PERIOD = 365.25 * 24 * 3600  # s

    # Moon constants (from JPL DE440 ephemeris)
    MU_MOON = 4902.800118e9  # m^3/s^2 (4902.800118 km^3/s^2)
    MOON_MU = MU_MOON  # Alias for backward compatibility
    MOON_RADIUS = 1737.4e3  # m (mean radius from NASA fact sheet)
    MOON_ORBIT_RADIUS = 384400e3  # m (mean distance)
    MOON_SEMI_MAJOR_AXIS = MOON_ORBIT_RADIUS  # Alias for backward compatibility
    MOON_ORBITAL_PERIOD = 27.32 * 24 * 3600  # s
    MOON_ESCAPE_VELOCITY = np.sqrt(2 * MU_MOON / MOON_RADIUS)  # m/s
    MOON_ORBITAL_VELOCITY = np.sqrt(MU_EARTH / MOON_ORBIT_RADIUS)  # m/s

    # Moon's sphere of influence (SOI = a * (m/M)^(2/5))
    # where a is semi-major axis, m is Moon mass, M is Earth mass
    MOON_SOI = MOON_ORBIT_RADIUS * (MU_MOON / MU_EARTH) ** (0.4)  # m (~66,000 km)

    # Sun constants (using PyKEP values)
    MU_SUN = pk.MU_SUN  # m^3/s^2 (1.32712440018e20)
    SUN_MU = MU_SUN  # Alias for backward compatibility
    SUN_RADIUS = 696000e3  # m (solar radius)
    AU = pk.AU  # m (astronomical unit)


class EphemerisLimits:
    """Time limits for ephemeris calculations.

    These limits define the valid time range for trajectory calculations.
    The reference epoch is J2000 (2000-01-01 00:00:00 UTC).

    Note: For ephemeris calculations, prefer using:
    - PyKEP epoch objects for time handling
    - JPL ephemerides via PyKEP for accurate positions
    """

    MIN_YEAR = 2020
    MAX_YEAR = 2050
    EPOCH_REF = "2000-01-01 00:00:00"
