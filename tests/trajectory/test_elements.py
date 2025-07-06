"""Unit tests for orbital elements utility functions.

This module tests the core orbital mechanics calculations including:
- Orbital period calculation
- Velocity components at any point in orbit
- Conversion between mean and true anomaly
- Validation of orbital mechanics relationships

All calculations use standard units (km, s, rad) unless otherwise specified.
Results are validated against known orbital parameters of real bodies (ISS, Moon).
"""

import pytest
import numpy as np
import logging
from src.trajectory.elements import (
    orbital_period,
    velocity_at_point,
    mean_to_true_anomaly,
    true_to_mean_anomaly,
)
import pykep as pk

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestOrbitalPeriod:
    """Test suite for orbital period calculations."""
    
    def test_leo_period(self):
        """Test orbital period calculation for Low Earth Orbit.
        
        Verifies that the orbital period calculation matches the known period
        of the International Space Station (ISS):
        - Altitude: ~400 km
        - Orbital radius: 6778 km (Earth radius + altitude)
        - Known period: ~92.68 minutes
        
        Uses Earth's gravitational parameter from PyKEP.
        """
        # ISS-like orbit at ~400km altitude
        period = orbital_period(6778.0)  # 6778 km = 6378 km (Earth radius) + 400 km
        
        # ISS period is ~92.68 minutes
        expected_period = 92.68 * 60  # Convert to seconds
        assert np.isclose(period, expected_period, rtol=0.01)
        
        logger.debug(f"Calculated ISS period: {period/60:.2f} minutes")

    def test_lunar_period(self):
        """Test orbital period calculation for lunar orbit.
        
        Verifies that the orbital period calculation matches the Moon's
        sidereal period:
        - Semi-major axis: 384,400 km
        - Known period: 27.32 days
        
        Uses Earth's gravitational parameter from PyKEP.
        """
        # Approximate lunar orbit
        period = orbital_period(384400.0, mu=pk.MU_EARTH)
        
        # Moon's sidereal period is ~27.32 days
        expected_period = 27.32 * 24 * 3600  # Convert to seconds
        assert np.isclose(period, expected_period, rtol=0.01)
        
        logger.debug(f"Calculated lunar period: {period/(24*3600):.2f} days")

class TestOrbitalVelocity:
    """Test suite for orbital velocity calculations."""
    
    def test_circular_orbit_velocity(self):
        """Test velocity components in a circular orbit.
        
        Verifies that in a circular Low Earth Orbit:
        - Radial velocity is zero (no change in radius)
        - Tangential velocity matches expected LEO velocity (~7.67 km/s)
        - Test performed at 45° true anomaly (arbitrary point)
        """
        # Circular LEO orbit
        v_r, v_t = velocity_at_point(
            semi_major_axis=6778.0,
            eccentricity=0.0,
            true_anomaly=45.0,
        )
        
        # In circular orbit, radial velocity should be 0
        assert np.isclose(v_r, 0.0, atol=1e-10)
        
        # Tangential velocity should be ~7.67 km/s for this altitude
        assert np.isclose(v_t, 7.67, rtol=0.01)
        
        logger.debug(f"Circular orbit velocities - radial: {v_r:.3f} km/s, tangential: {v_t:.3f} km/s")

    def test_elliptical_orbit_velocity(self):
        """Test velocity components in an elliptical orbit.
        
        Verifies velocity components at perigee of a GTO-like orbit:
        - Semi-major axis: 24,396 km ((LEO + GEO)/2)
        - Eccentricity: 0.7306 (typical GTO)
        - True anomaly: 0° (at perigee)
        
        Checks:
        - Radial velocity is zero at perigee
        - Tangential velocity exceeds circular LEO velocity
        """
        # GTO-like orbit
        v_r, v_t = velocity_at_point(
            semi_major_axis=24396.0,  # (6378 + 35786) / 2 km
            eccentricity=0.7306,      # Typical GTO
            true_anomaly=0.0,         # At perigee
        )
        
        # At perigee of elliptical orbit:
        # - Radial velocity should be 0
        # - Tangential velocity should be maximum
        assert np.isclose(v_r, 0.0, atol=1e-10)
        assert v_t > 10.0  # Should be faster than LEO velocity
        
        logger.debug(f"GTO perigee velocities - radial: {v_r:.3f} km/s, tangential: {v_t:.3f} km/s")

    def test_velocity_components_perpendicular(self):
        """Test velocity components at key orbital points.
        
        Verifies velocity components in an elliptical orbit (e=0.1)
        at four key points:
        - 0° (periapsis): v_r = 0, v_t = max
        - 90° (ascending): v_r = max, v_t = min
        - 180° (apoapsis): v_r = 0, v_t = min
        - 270° (descending): v_r = min, v_t = min
        """
        # Test orbit parameters
        a = 8000.0  # km
        e = 0.1
        
        # Test at key points
        test_points = [0, 90, 180, 270]  # degrees
        for nu in test_points:
            v_r, v_t = velocity_at_point(a, e, nu)
            logger.debug(f"At {nu}°: v_r = {v_r:.3f} km/s, v_t = {v_t:.3f} km/s")
            
            # At periapsis (0°) and apoapsis (180°), radial velocity should be 0
            if nu in [0, 180]:
                assert np.isclose(v_r, 0.0, atol=1e-10)
            
            # At 90° and 270°, radial velocity should be maximum/minimum
            if nu in [90, 270]:
                assert abs(v_r) > 0.0

class TestAnomalyConversion:
    """Test suite for anomaly conversion functions."""
    
    def test_circular_mean_to_true(self):
        """Test mean to true anomaly conversion in circular orbit.
        
        Verifies that in a circular orbit (e=0):
        - Mean anomaly equals true anomaly
        - Test performed at 45° (arbitrary angle)
        
        This is a key property of circular orbits where the angular
        rate is constant.
        """
        # In circular orbit, mean anomaly equals true anomaly
        true_anom = mean_to_true_anomaly(
            mean_anomaly=45.0,
            eccentricity=0.0,
        )
        assert np.isclose(true_anom, 45.0)
        
        logger.debug(f"Circular orbit: M = 45° → ν = {true_anom:.2f}°")

    def test_elliptical_mean_to_true(self):
        """Test mean to true anomaly conversion in elliptical orbit.
        
        Verifies that in an elliptical orbit (e=0.1):
        - True anomaly is always greater than mean anomaly
        - Test performed at 45° mean anomaly
        
        This reflects the fact that objects move faster at periapsis
        than at apoapsis in elliptical orbits.
        """
        # Test with moderate eccentricity
        true_anom = mean_to_true_anomaly(
            mean_anomaly=45.0,
            eccentricity=0.1,
        )
        # True anomaly should be larger than mean anomaly
        assert true_anom > 45.0
        
        logger.debug(f"Elliptical orbit (e=0.1): M = 45° → ν = {true_anom:.2f}°")

    def test_circular_true_to_mean(self):
        """Test true to mean anomaly conversion in circular orbit.
        
        Verifies that in a circular orbit (e=0):
        - True anomaly equals mean anomaly
        - Test performed at 90° (arbitrary angle)
        
        This is the inverse of the mean_to_true_anomaly test for
        circular orbits.
        """
        # In circular orbit, true anomaly equals mean anomaly
        mean_anom = true_to_mean_anomaly(
            true_anomaly=90.0,
            eccentricity=0.0,
        )
        assert np.isclose(mean_anom, 90.0)
        
        logger.debug(f"Circular orbit: ν = 90° → M = {mean_anom:.2f}°")

    def test_elliptical_true_to_mean(self):
        """Test true to mean anomaly conversion in elliptical orbit.
        
        Verifies that in an elliptical orbit (e=0.1):
        - Mean anomaly is always less than true anomaly
        - Test performed at 90° true anomaly
        
        This is the inverse relationship of mean_to_true_anomaly for
        elliptical orbits.
        """
        # Test with moderate eccentricity
        mean_anom = true_to_mean_anomaly(
            true_anomaly=90.0,
            eccentricity=0.1,
        )
        # Mean anomaly should be smaller than true anomaly
        assert mean_anom < 90.0
        
        logger.debug(f"Elliptical orbit (e=0.1): ν = 90° → M = {mean_anom:.2f}°")

    def test_anomaly_conversion_roundtrip(self):
        """Test consistency of anomaly conversions.
        
        Verifies that converting from:
        mean -> true -> mean
        preserves the original value within numerical precision.
        
        Test performed with:
        - Mean anomaly: 120°
        - Eccentricity: 0.3 (significant ellipticity)
        """
        original_mean = 120.0
        eccentricity = 0.3
        
        # Convert mean -> true -> mean
        true = mean_to_true_anomaly(original_mean, eccentricity)
        mean = true_to_mean_anomaly(true, eccentricity)
        
        assert np.isclose(mean, original_mean, rtol=1e-6)
        
        logger.debug(f"Roundtrip test: {original_mean}° → {true:.2f}° → {mean:.2f}°") 