"""Unit conversion utilities for trajectory calculations.

This module provides conversion functions between different units and coordinate systems
used throughout the trajectory calculations. All functions preserve input types (scalar,
list, tuple, or numpy array) and handle both single values and arrays.

Standard Units (PyKEP Native):
    - Distances: meters (m)
    - Velocities: meters per second (m/s)
    - Angles: radians (internal calculations)
    - Gravitational Parameters: m³/s²

Time Conventions:
    - MJD2000: Days since 2000-01-01 00:00:00 UTC (PyKEP's internal reference)
    - J2000: Days since 2000-01-01 12:00:00 UTC (standard astronomical epoch)
    Note: J2000 = MJD2000 + 0.5

Unit Conversion Guidelines:
    1. Use PyKEP native units (meters, m/s) for all internal calculations
    2. Convert user input/output as needed:
       - Input angles: degrees -> radians
       - Output angles: radians -> degrees
    3. Keep all ephemeris calculations in meters/m/s
    4. Document units explicitly in function docstrings

Example:
    >>> from datetime import datetime, timezone
    >>> dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    >>> epoch = datetime_to_mjd2000(dt)  # Get PyKEP epoch
    >>> r = [1000.0, 0.0, 0.0]  # Position in meters
    >>> v = [0.0, 7800.0, 0.0]  # Velocity in m/s
"""

from datetime import UTC, datetime, timedelta
from typing import Union

import numpy as np

# Type alias for numeric types
NumericType = Union[float, list[float], tuple[float, ...], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]

def ensure_array(x: NumericType) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Convert input to numpy array if it isn't already.

    Args:
        x: Input value or array

    Returns
    -------
        numpy.ndarray: Array version of input
    """
    return np.array(x) if not isinstance(x, np.ndarray) else x

def restore_type(x: NumericType, arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> NumericType:
    """Restore original type of input after array operations.

    Args:
        x: Original input
        arr: Numpy array to convert back

    Returns
    -------
        Same type as input x
    """
    if isinstance(x, list | tuple):
        return type(x)(arr.tolist())
    return arr if isinstance(x, np.ndarray) else float(arr)

def datetime_to_mjd2000(dt: datetime) -> float:
    """Convert datetime to Modified Julian Date 2000 (MJD2000).

    MJD2000 is days since 2000-01-01 00:00:00 UTC, which is PyKEP's reference epoch.
    A value of 0.0 corresponds to 2000-01-01 00:00:00 UTC.
    A value of 0.5 corresponds to 2000-01-01 12:00:00 UTC (J2000 epoch).

    Args:
        dt: Timezone-aware datetime object to convert

    Returns
    -------
        float: Days since MJD2000 epoch

    Raises
    ------
        ValueError: If datetime is naive (no timezone)
    """
    if dt.tzinfo is None:
        msg = "Datetime must be timezone-aware"
        raise ValueError(msg)

    # Convert to UTC if needed
    dt_utc = dt.astimezone(UTC)

    # Reference epoch (MJD2000)
    mjd2000_epoch = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)

    # Calculate days since MJD2000
    delta = dt_utc - mjd2000_epoch
    return delta.total_seconds() / (24.0 * 3600.0)

def datetime_to_j2000(dt: datetime) -> float:
    """Convert datetime to days since J2000 epoch.

    J2000 is days since 2000-01-01 12:00:00 UTC.
    A value of 0.0 corresponds to 2000-01-01 12:00:00 UTC.
    A value of -0.5 corresponds to 2000-01-01 00:00:00 UTC (MJD2000 epoch).

    Args:
        dt: Timezone-aware datetime object to convert

    Returns
    -------
        float: Days since J2000 epoch

    Raises
    ------
        ValueError: If datetime is naive (no timezone)
    """
    # Convert to MJD2000 and adjust for J2000 offset
    return datetime_to_mjd2000(dt) - 0.5

def datetime_to_pykep_epoch(dt: datetime) -> float:
    """Convert datetime to PyKEP epoch (MJD2000).

    This is an alias for datetime_to_mjd2000() to make PyKEP interfacing explicit.
    PyKEP uses MJD2000 internally, where:
        pk.epoch(0) = 2000-01-01 00:00:00 UTC
        pk.epoch(0.5) = 2000-01-01 12:00:00 UTC (J2000)

    Args:
        dt: Timezone-aware datetime object to convert

    Returns
    -------
        float: PyKEP epoch value (days since MJD2000)
    """
    return datetime_to_mjd2000(dt)

def km_to_m(km: NumericType) -> NumericType:
    """Convert kilometers to meters.

    Args:
        km: Distance in kilometers

    Returns
    -------
        Distance in meters, preserving input type
    """
    if isinstance(km, list | tuple):
        return type(km)([x * 1000.0 for x in km])
    if isinstance(km, np.ndarray):
        return km * 1000.0
    return km * 1000.0

def m_to_km(m: NumericType) -> NumericType:
    """Convert meters to kilometers.

    Args:
        m: Distance in meters

    Returns
    -------
        Distance in kilometers, preserving input type
    """
    if isinstance(m, list | tuple):
        return type(m)([x / 1000.0 for x in m])
    if isinstance(m, np.ndarray):
        return m / 1000.0
    return m / 1000.0

def kmps_to_mps(kmps: NumericType) -> NumericType:
    """Convert kilometers per second to meters per second.

    Args:
        kmps: Velocity in kilometers per second

    Returns
    -------
        Velocity in meters per second, preserving input type
    """
    if isinstance(kmps, list | tuple):
        return type(kmps)([x * 1000.0 for x in kmps])
    if isinstance(kmps, np.ndarray):
        return kmps * 1000.0
    return kmps * 1000.0

def mps_to_kmps(mps: NumericType) -> NumericType:
    """Convert meters per second to kilometers per second.

    Args:
        mps: Velocity in meters per second

    Returns
    -------
        Velocity in kilometers per second, preserving input type
    """
    arr = ensure_array(mps)
    return restore_type(mps, arr / 1000.0)

def deg_to_rad(deg: NumericType) -> NumericType:
    """Convert degrees to radians.

    Args:
        deg: Angle in degrees

    Returns
    -------
        Angle in radians, preserving input type
    """
    arr = ensure_array(deg)
    return restore_type(deg, np.radians(arr))

def rad_to_deg(rad: NumericType) -> NumericType:
    """Convert radians to degrees.

    Args:
        rad: Angle in radians

    Returns
    -------
        Angle in degrees, preserving input type
    """
    arr = ensure_array(rad)
    return restore_type(rad, np.degrees(arr))

def km3s2_to_m3s2(mu: float) -> float:
    """Convert gravitational parameter from km³/s² to m³/s².

    Args:
        mu: Gravitational parameter in km³/s²

    Returns
    -------
        float: Gravitational parameter in m³/s²
    """
    return mu * 1e9

def m3s2_to_km3s2(mu: float) -> float:
    """Convert gravitational parameter from m³/s² to km³/s².

    Args:
        mu: Gravitational parameter in m³/s²

    Returns
    -------
        float: Gravitational parameter in km³/s²
    """
    return mu * 1e-9

def days_to_seconds(days: float) -> float:
    """Convert days to seconds.

    Args:
        days: Time duration in days

    Returns
    -------
        float: Time duration in seconds
    """
    return days * 86400.0

def seconds_to_days(seconds: float) -> float:
    """Convert seconds to days.

    Args:
        seconds: Time duration in seconds

    Returns
    -------
        float: Time duration in days
    """
    return seconds / 86400.0

def pykep_epoch_to_datetime(epoch: float) -> datetime:
    """Convert PyKEP epoch (MJD2000) to datetime.

    Args:
        epoch: PyKEP epoch value (days since MJD2000)

    Returns
    -------
        datetime: Timezone-aware datetime object in UTC
    """
    # Reference epoch (MJD2000)
    mjd2000_epoch = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)

    # Add days to reference epoch
    delta = timedelta(days=epoch)
    return mjd2000_epoch + delta
