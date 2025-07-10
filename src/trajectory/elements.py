"""Utility functions for orbital calculations.

This module provides functions for basic orbital mechanics calculations
using PyKEP, including orbital period, velocity, and coordinate transformations.
"""

import numpy as np
import pykep as pk


def orbital_period(semi_major_axis: float, mu: float = pk.MU_EARTH) -> float:
    """Calculate orbital period using Kepler's Third Law.

    Args:
        semi_major_axis: Semi-major axis in kilometers
        mu: Gravitational parameter in m^3/s^2 (defaults to Earth's mu)

    Returns
    -------
        Orbital period in seconds

    Example:
        >>> period = orbital_period(6778.0)  # LEO orbit
        >>> print(f"Period: {period/60:.1f} minutes")
    """
    a = semi_major_axis * 1000  # Convert to meters
    return float(2 * np.pi * np.sqrt(a**3 / mu))

def velocity_at_point(
    semi_major_axis: float,
    eccentricity: float,
    true_anomaly: float,
    mu: float = pk.MU_EARTH,
) -> tuple[float, float]:
    """Calculate radial and tangential velocity components at a point in orbit.

    Args:
        semi_major_axis: Semi-major axis in kilometers
        eccentricity: Orbital eccentricity (0-1)
        true_anomaly: True anomaly in degrees
        mu: Gravitational parameter in m^3/s^2 (defaults to Earth's mu)

    Returns
    -------
        Tuple of (radial_velocity, tangential_velocity) in km/s

    Example:
        >>> v_r, v_t = velocity_at_point(6778.0, 0.001, 45.0)
        >>> print(f"Velocity: radial={v_r:.2f}, tangential={v_t:.2f} km/s")
    """
    # Convert inputs to proper units
    a = semi_major_axis * 1000  # km to m
    nu = np.radians(true_anomaly)

    # Calculate radius at true anomaly
    p = a * (1 - eccentricity**2)

    # Calculate velocity components
    v_r = np.sqrt(mu/p) * eccentricity * np.sin(nu)
    v_t = np.sqrt(mu/p) * (1 + eccentricity * np.cos(nu))

    # Convert back to km/s
    return v_r/1000, v_t/1000

def mean_to_true_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    """Convert mean anomaly to true anomaly using iterative solver.

    Args:
        mean_anomaly: Mean anomaly in degrees
        eccentricity: Orbital eccentricity (0-1)

    Returns
    -------
        True anomaly in degrees

    Example:
        >>> nu = mean_to_true_anomaly(45.0, 0.1)
        >>> print(f"True anomaly: {nu:.2f} degrees")
    """
    M = np.radians(mean_anomaly)

    # Initial guess for eccentric anomaly
    E = M if eccentricity < 0.8 else np.pi

    # Newton-Raphson iteration to solve Kepler's equation
    for _ in range(10):  # Usually converges in <10 iterations
        delta = (E - eccentricity * np.sin(E) - M) / (1 - eccentricity * np.cos(E))
        E = E - delta
        if abs(delta) < 1e-8:
            break

    # Convert eccentric anomaly to true anomaly
    nu = 2 * np.arctan(np.sqrt((1 + eccentricity)/(1 - eccentricity)) * np.tan(E/2))
    return float(np.degrees(nu) % 360)

def true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly to mean anomaly.

    Args:
        true_anomaly: True anomaly in degrees
        eccentricity: Orbital eccentricity (0-1)

    Returns
    -------
        Mean anomaly in degrees

    Example:
        >>> M = true_to_mean_anomaly(60.0, 0.1)
        >>> print(f"Mean anomaly: {M:.2f} degrees")
    """
    nu = np.radians(true_anomaly)

    # Calculate eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1 - eccentricity)/(1 + eccentricity)) * np.tan(nu/2))

    # Calculate mean anomaly
    M = E - eccentricity * np.sin(E)
    return float(np.degrees(M) % 360)
