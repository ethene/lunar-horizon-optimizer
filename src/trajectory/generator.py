"""Trajectory generation functions for Earth-Moon transfers.

This module provides functions to generate and optimize transfer trajectories
between Earth parking orbits and lunar orbits. All calculations use PyKEP's
native units throughout.

Unit Conventions (PyKEP Native):
    - Distances: meters (m)
    - Velocities: meters per second (m/s)
    - Times: days for epochs, seconds for durations
    - Angles: radians
    - Gravitational Parameters: m³/s²

Time Conventions:
    - Input: timezone-aware datetime objects
    - Internal: MJD2000 (days since 2000-01-01 00:00:00 UTC)
    - PyKEP: epoch(0) = MJD2000, epoch(0.5) = J2000

Reference Frames:
    - All calculations are performed in the Earth-centered inertial (J2000) frame
    - Moon states are obtained in Earth-centered frame
    - Orbit elements are defined relative to Earth's equator and J2000 epoch

The trajectory generation process:
1. Calculate Earth and Moon states at departure and arrival epochs
2. Generate initial Earth parking orbit and final lunar orbit
3. Solve Lambert's problem for the transfer trajectory
4. Optimize departure and arrival times for minimum delta-v
5. Return the complete trajectory with maneuvers
"""

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pykep as pk

from src.trajectory.defaults import TransferDefaults as TD
from src.utils.unit_conversions import datetime_to_mjd2000, km_to_m

from .lunar_transfer import LunarTransfer
from .models import Trajectory

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants (converted to meters)
EARTH_RADIUS = pk.EARTH_RADIUS  # m
J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)  # J2000 epoch


def generate_lunar_transfer(
    departure_time: datetime,
    time_of_flight: float,
    initial_orbit_alt: float = TD.DEFAULT_EARTH_ORBIT,
    final_orbit_alt: float = TD.DEFAULT_MOON_ORBIT,
    max_tli_dv: float = TD.MAX_TLI_DELTA_V,
    min_tli_dv: float | None = None,
    max_revs: int = TD.MAX_REVOLUTIONS,
) -> tuple[Trajectory, float]:
    """Generate a lunar transfer trajectory.

    Args:
        departure_time: Departure time (must be timezone-aware)
        time_of_flight: Transfer time in days
        initial_orbit_alt: Initial parking orbit altitude in km (default: 300 km)
        final_orbit_alt: Final lunar orbit altitude in km (default: 100 km)
        max_tli_dv: Maximum allowed TLI delta-v in m/s (default: 3500 m/s)
        min_tli_dv: Minimum required TLI delta-v in m/s (default: None)
        max_revs: Maximum number of revolutions (default: 1)

    Returns
    -------
        Tuple containing:
            - Trajectory object with complete transfer
            - Total delta-v cost in m/s

    Raises
    ------
        ValueError: If any parameters are invalid
    """
    # Validate input parameters
    if departure_time.tzinfo is None:
        msg = "Departure time must be timezone-aware"
        raise ValueError(msg)

    TD.validate_earth_orbit(initial_orbit_alt)
    TD.validate_moon_orbit(final_orbit_alt)
    TD.validate_transfer_time(time_of_flight)

    if max_tli_dv <= 0:
        msg = "Maximum delta-v must be positive"
        raise ValueError(msg)
    if min_tli_dv is not None and min_tli_dv >= max_tli_dv:
        msg = "Maximum delta-v must be greater than minimum"
        raise ValueError(msg)
    if max_revs < 0:
        msg = "Maximum revolutions must be non-negative"
        raise ValueError(msg)
    if max_revs > TD.MAX_REVOLUTIONS:
        msg = f"Maximum revolutions must be less than {TD.MAX_REVOLUTIONS}"
        raise ValueError(msg)

    # Convert altitudes to meters
    km_to_m(TD.EARTH_RADIUS / 1000.0 + initial_orbit_alt)
    km_to_m(TD.MOON_MEAN_RADIUS / 1000.0 + final_orbit_alt)

    # Initialize lunar transfer solver
    transfer = LunarTransfer()

    # Generate transfer trajectory
    return transfer.generate_transfer(
        epoch=datetime_to_mjd2000(departure_time),
        earth_orbit_alt=initial_orbit_alt,
        moon_orbit_alt=final_orbit_alt,
        transfer_time=time_of_flight,
        max_revolutions=max_revs,
    )


def optimize_departure_time(
    reference_epoch: datetime,
    earth_orbit_altitude: float = TD.DEFAULT_EARTH_ORBIT,
    moon_orbit_altitude: float = TD.DEFAULT_MOON_ORBIT,
    transfer_time: float = TD.DEFAULT_TRANSFER_TIME,
    search_window: float = 1.0,  # days
) -> datetime:
    """Find optimal departure time for minimum delta-v transfer.

    Searches for the best departure time within a window around the
    reference epoch by generating trajectories at regular intervals
    and selecting the one with minimum total delta-v.

    Args:
        reference_epoch: Center of search window (timezone-aware)
        earth_orbit_altitude: Initial parking orbit altitude [km]
        moon_orbit_altitude: Final lunar orbit altitude [km]
        transfer_time: Nominal transfer duration [days]
        search_window: Time window to search around reference [days]

    Returns
    -------
        Optimal departure time
    """
    # Generate sample times
    num_samples = 12
    sample_spacing = search_window * 2 / num_samples

    min_dv = float("inf")
    best_departure = None

    for i in range(num_samples):
        departure = reference_epoch + timedelta(
            days=(-search_window + i * sample_spacing)
        )
        departure + timedelta(days=transfer_time)

        try:
            trajectory, total_dv = generate_lunar_transfer(
                departure_time=departure,
                time_of_flight=transfer_time,
            )

            if total_dv < min_dv:
                min_dv = total_dv
                best_departure = departure

        except ValueError as e:
            logger.debug(
                f"Skipping invalid trajectory at {departure.isoformat()}: {e!s}"
            )
            continue

    if best_departure is None:
        msg = "No valid trajectories found in search window"
        raise ValueError(msg)

    logger.info(f"Optimal departure time found: {best_departure.isoformat()}")
    logger.info(f"Minimum delta-v: {min_dv/1000:.2f} km/s")

    return best_departure


def estimate_hohmann_transfer_dv(r1: float, r2: float) -> tuple[float, float, float]:
    """Calculate delta-v and time of flight for a Hohmann transfer orbit.

    Args:
        r1: Initial orbit radius (km)
        r2: Final orbit radius (km)

    Returns
    -------
        Tuple[float, float, float]: (delta-v for first burn (km/s),
                                    delta-v for second burn (km/s),
                                    time of flight (days))

    Raises
    ------
        ValueError: If orbit radii are not positive
    """
    # Input validation
    if r1 <= 0 or r2 <= 0:
        msg = "Orbit radii must be positive"
        raise ValueError(msg)

    # Convert radii to meters for PyKEP
    r1 = r1 * 1000.0  # m
    r2 = r2 * 1000.0  # m

    # Calculate circular orbit velocities
    v1_circ = np.sqrt(pk.MU_EARTH / r1)  # m/s
    v2_circ = np.sqrt(pk.MU_EARTH / r2)  # m/s

    # Calculate transfer orbit parameters
    a = (r1 + r2) / 2.0  # m
    v1_trans = np.sqrt(pk.MU_EARTH * (2.0 / r1 - 1.0 / a))  # m/s
    v2_trans = np.sqrt(pk.MU_EARTH * (2.0 / r2 - 1.0 / a))  # m/s

    # Calculate delta-v magnitudes (convert to km/s)
    dv1 = abs(v1_trans - v1_circ) / 1000.0  # km/s
    dv2 = abs(v2_trans - v2_circ) / 1000.0  # km/s

    # Calculate time of flight (half orbit period)
    tof = np.pi * np.sqrt(a**3 / pk.MU_EARTH) / 86400.0  # days

    logger.debug(f"Initial circular velocity: {v1_circ/1000.0:.3f} km/s")
    logger.debug(f"Final circular velocity: {v2_circ/1000.0:.3f} km/s")
    logger.debug(
        f"Transfer orbit velocities: {v1_trans/1000.0:.3f} km/s, {v2_trans/1000.0:.3f} km/s"
    )
    logger.debug(f"Delta-v values: {dv1:.3f} km/s, {dv2:.3f} km/s")
    logger.debug(f"Time of flight: {tof:.3f} days")

    return dv1, dv2, tof
