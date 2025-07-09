"""Unit conversion test suite.

This module contains comprehensive tests for unit conversions, covering both basic
conversions and their integration with trajectory calculations.

Test Coverage:
1. Basic Conversions:
   - Distance: meters (m) ↔ kilometers (km)
   - Velocity: m/s ↔ km/s
   - Time: seconds ↔ days, datetime ↔ epochs
   - Gravitational Parameters: m³/s² ↔ km³/s²

2. Trajectory Integration:
   - Orbit state conversions
   - Maneuver calculations
   - Lunar transfer generation

Unit Conventions:
- PyKEP native units: meters (m), meters/second (m/s), seconds (s), m³/s² for mu
- User-facing units: kilometers (km), kilometers/second (km/s), days
- Time formats: datetime (UTC), Modified Julian Date 2000 (MJD2000), J2000
"""

import pytest
import numpy as np
import pykep as pk
from datetime import datetime, timedelta, UTC
import logging
from src.trajectory.models import OrbitState, Maneuver
from src.utils.unit_conversions import (
    datetime_to_mjd2000,
    datetime_to_j2000,
    datetime_to_pykep_epoch,
    km_to_m,
    m_to_km,
    kmps_to_mps,
    mps_to_kmps,
    km3s2_to_m3s2,
    m3s2_to_km3s2,
    days_to_seconds,
    seconds_to_days
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestBasicUnitConversions:
    """Test suite for basic unit conversions."""

    def test_distance_conversions(self):
        """Test distance unit conversions.
        
        Verifies:
        - Earth radius (~6378 km) converts correctly
        - Lunar distance (~384,400 km) converts correctly
        - Round-trip conversions preserve values
        - Vector conversions work correctly
        """
        # Scalar conversions
        distance_km = 384400.0  # Lunar distance
        distance_m = km_to_m(distance_km)
        distance_km_back = m_to_km(distance_m)

        assert abs(distance_km - distance_km_back) < 1e-10
        assert abs(distance_m - 384400000.0) < 1e-10

        # Earth radius
        earth_radius_km = 6378.137
        earth_radius_m = km_to_m(earth_radius_km)
        assert abs(earth_radius_m - pk.EARTH_RADIUS) < 1000.0

        # Vector conversions
        r_km = np.array([6678.0, 0.0, 0.0])
        r_m = km_to_m(r_km)
        r_km_back = m_to_km(r_m)
        np.testing.assert_array_almost_equal(r_km, r_km_back)

    def test_velocity_conversions(self):
        """Test velocity unit conversions.
        
        Verifies:
        - Earth escape velocity (~11.2 km/s) converts correctly
        - LEO orbital velocity (~7.8 km/s) converts correctly
        - Round-trip conversions preserve values
        - Vector conversions work correctly
        """
        # Scalar conversions
        velocity_kms = 11.2  # Earth escape velocity
        velocity_ms = kmps_to_mps(velocity_kms)
        velocity_kms_back = mps_to_kmps(velocity_ms)

        assert abs(velocity_kms - velocity_kms_back) < 1e-10
        assert abs(velocity_ms - 11200.0) < 1e-10

        # LEO velocity
        leo_velocity_kms = 7.8
        leo_velocity_ms = kmps_to_mps(leo_velocity_kms)
        assert abs(leo_velocity_ms - 7800.0) < 1.0

        # Vector conversions
        v_kms = np.array([7.8, 0.0, 0.0])
        v_ms = kmps_to_mps(v_kms)
        v_kms_back = mps_to_kmps(v_ms)
        np.testing.assert_array_almost_equal(v_kms, v_kms_back)

    def test_gravitational_parameters(self):
        """Test gravitational parameter conversions.
        
        Verifies:
        - Earth's mu (~398600.4418 km³/s²) converts correctly
        - Moon's mu (~4904.8695 km³/s²) converts correctly
        - Round-trip conversions preserve values
        """
        # Earth's mu
        mu_earth_m3s2 = pk.MU_EARTH
        mu_earth_km3s2 = m3s2_to_km3s2(mu_earth_m3s2)
        assert abs(mu_earth_km3s2 - 398600.4418) < 0.1
        assert abs(km3s2_to_m3s2(mu_earth_km3s2) - mu_earth_m3s2) < 1e3

        # Moon's mu
        mu_moon_m3s2 = 4.9048695e12
        mu_moon_km3s2 = m3s2_to_km3s2(mu_moon_m3s2)
        assert abs(mu_moon_km3s2 - 4904.8695) < 0.1
        assert abs(km3s2_to_m3s2(mu_moon_km3s2) - mu_moon_m3s2) < 1e3

    def test_time_duration_conversions(self):
        """Test time duration conversions.
        
        Verifies:
        - Days to seconds conversion is correct
        - Seconds to days conversion is correct
        - Round-trip conversions preserve values
        - Partial day conversions work correctly
        """
        # One day
        days = 1.0
        seconds = days_to_seconds(days)
        assert abs(seconds - 86400.0) < 1e-10

        # Round trip
        days_back = seconds_to_days(seconds)
        assert abs(days - days_back) < 1e-10

        # Partial day
        hours_6 = 0.25
        seconds_6 = days_to_seconds(hours_6)
        assert abs(seconds_6 - 21600.0) < 1e-10

    def test_edge_cases(self):
        """Test edge cases and potential error conditions.
        
        Verifies:
        - Zero values convert correctly
        - Negative values convert correctly
        - Very large values maintain precision
        """
        # Zero values
        assert km_to_m(0.0) == 0.0
        assert m_to_km(0.0) == 0.0
        assert kmps_to_mps(0.0) == 0.0
        assert mps_to_kmps(0.0) == 0.0

        # Negative values
        assert km_to_m(-1.0) == -1000.0
        assert m_to_km(-1000.0) == -1.0
        assert kmps_to_mps(-11.2) == -11200.0
        assert mps_to_kmps(-11200.0) == -11.2

        # Large values (1 light year)
        ly_km = 9.461e12
        ly_m = km_to_m(ly_km)
        assert abs(m_to_km(ly_m) - ly_km) / ly_km < 1e-10

class TestEpochConversions:
    """Test suite for epoch and time format conversions."""

    def test_mjd2000_conversion(self):
        """Verify datetime to MJD2000 conversion."""
        epoch = datetime(2000, 1, 1, tzinfo=UTC)
        assert abs(datetime_to_mjd2000(epoch)) < 1e-10

        one_day_after = epoch + timedelta(days=1)
        assert abs(datetime_to_mjd2000(one_day_after) - 1.0) < 1e-10

        one_day_before = epoch - timedelta(days=1)
        assert abs(datetime_to_mjd2000(one_day_before) + 1.0) < 1e-10

    def test_j2000_conversion(self):
        """Verify datetime to J2000 conversion."""
        epoch = datetime(2000, 1, 1, 12, tzinfo=UTC)
        assert abs(datetime_to_j2000(epoch)) < 1e-10

        one_day_after = epoch + timedelta(days=1)
        assert abs(datetime_to_j2000(one_day_after) - 1.0) < 1e-10

        one_day_before = epoch - timedelta(days=1)
        assert abs(datetime_to_j2000(one_day_before) + 1.0) < 1e-10

    def test_pykep_epoch_conversion(self):
        """Verify datetime to PyKEP epoch conversion."""
        test_date = datetime(2024, 1, 1, tzinfo=UTC)
        pykep_epoch = datetime_to_pykep_epoch(test_date)

        mjd2000_days = datetime_to_mjd2000(test_date)
        assert abs(pykep_epoch - mjd2000_days) < 1e-10, "PyKEP epoch should match MJD2000 days"

        # Verify relationship with J2000
        j2000_days = datetime_to_j2000(test_date)
        assert abs(pykep_epoch - (j2000_days + 0.5)) < 1e-10, "PyKEP epoch (MJD2000) should be J2000 + 0.5 days"

class TestTrajectoryUnitConversions:
    """Test unit conversions in trajectory generation components."""

    def test_orbit_state_units(self):
        """Test unit conversions in OrbitState class."""
        # Create a circular LEO orbit
        state = OrbitState(
            semi_major_axis=6778.0,  # km (400 km altitude)
            eccentricity=0.0,
            inclination=28.5,  # degrees
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
            epoch=datetime.now(UTC)
        )

        # Get state vectors with Earth's gravitational parameter
        r_m, v_ms = state.get_state_vectors(mu=pk.MU_EARTH)

        # Convert to km and km/s for validation
        r_km = m_to_km(r_m)
        v_kms = mps_to_kmps(v_ms)

        # Validate position and velocity magnitudes
        r_mag = np.linalg.norm(r_km)
        v_mag = np.linalg.norm(v_kms)

        assert abs(r_mag - 6778.0) < 1.0, f"Position magnitude {r_mag:.1f} km doesn't match semi-major axis"
        assert 7.5 < v_mag < 8.0, f"Velocity magnitude {v_mag:.2f} km/s outside expected range for LEO"

    def test_maneuver_units(self):
        """Test unit conversions in Maneuver class."""
        dv_kms = np.array([3.1, 0.0, 0.0])  # 3.1 km/s delta-v in x direction
        epoch = datetime.now(UTC)

        maneuver = Maneuver(
            delta_v=dv_kms,
            epoch=epoch
        )

        # Get delta-v in m/s and verify magnitude
        dv_ms = maneuver.get_delta_v_ms()
        assert abs(np.linalg.norm(dv_ms) - 3100.0) < 1.0, f"Delta-v magnitude {np.linalg.norm(dv_ms):.1f} m/s incorrect"

    def test_transfer_trajectory_units(self):
        """Test unit conversions in transfer trajectory generation."""
        # Initial LEO orbit
        initial_orbit = OrbitState(
            semi_major_axis=6778.0,  # km (400 km altitude)
            eccentricity=0.0,
            inclination=28.5,  # degrees
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
            epoch=datetime.now(UTC)
        )

        # Target lunar orbit
        target_orbit = OrbitState(
            semi_major_axis=384400.0,  # km (approximate Moon distance)
            eccentricity=0.0,
            inclination=5.145,  # degrees (Moon's inclination)
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
            epoch=datetime.now(UTC)
        )

        # Get state vectors with Earth's gravitational parameter
        r1_m, v1_ms = initial_orbit.get_state_vectors(mu=pk.MU_EARTH)
        r2_m, v2_ms = target_orbit.get_state_vectors(mu=pk.MU_EARTH)

        # Convert to km and km/s for validation
        r1_km = m_to_km(r1_m)
        r2_km = m_to_km(r2_m)
        v1_kms = mps_to_kmps(v1_ms)
        v2_kms = mps_to_kmps(v2_ms)

        # Validate position and velocity magnitudes
        r1_mag = np.linalg.norm(r1_km)
        r2_mag = np.linalg.norm(r2_km)
        v1_mag = np.linalg.norm(v1_kms)
        v2_mag = np.linalg.norm(v2_kms)

        assert abs(r1_mag - 6778.0) < 1.0, f"Initial position magnitude {r1_mag:.1f} km incorrect"
        assert abs(r2_mag - 384400.0) < 100.0, f"Target position magnitude {r2_mag:.1f} km incorrect"
        assert 7.5 < v1_mag < 8.0, f"Initial velocity {v1_mag:.2f} km/s outside expected range"
        assert 0.8 < v2_mag < 1.2, f"Target velocity {v2_mag:.2f} km/s outside expected range"

    def test_hohmann_estimate_units(self):
        """Test unit handling in Hohmann transfer estimation."""
        # Initial LEO orbit
        r1_km = 6778.0  # 400 km altitude
        # Target lunar orbit
        r2_km = 384400.0  # Approximate Moon distance

        # Convert to meters for calculations
        r1_m = km_to_m(r1_km)
        r2_m = km_to_m(r2_km)

        # Calculate Hohmann transfer delta-v
        def estimate_hohmann_transfer_dv(r1_m, r2_m, mu_m3s2):
            """Estimate delta-v for a Hohmann transfer between circular orbits."""
            if r1_m <= 0 or r2_m <= 0:
                raise ValueError("Orbit radii must be positive")
            if mu_m3s2 <= 0:
                raise ValueError("Gravitational parameter must be positive")

            # Calculate velocities in initial and final circular orbits
            v1_ms = np.sqrt(mu_m3s2 / r1_m)
            v2_ms = np.sqrt(mu_m3s2 / r2_m)

            # Calculate velocities at periapsis and apoapsis of transfer orbit
            a_m = (r1_m + r2_m) / 2  # Semi-major axis of transfer orbit
            vp_ms = np.sqrt(mu_m3s2 * (2/r1_m - 1/a_m))  # Velocity at periapsis
            va_ms = np.sqrt(mu_m3s2 * (2/r2_m - 1/a_m))  # Velocity at apoapsis

            # Calculate total delta-v
            dv1_ms = abs(vp_ms - v1_ms)  # First burn
            dv2_ms = abs(v2_ms - va_ms)  # Second burn
            total_dv_ms = dv1_ms + dv2_ms

            return total_dv_ms

        # Test with invalid inputs
        with pytest.raises(ValueError, match="Orbit radii must be positive"):
            estimate_hohmann_transfer_dv(-1.0, r2_m, pk.MU_EARTH)

        with pytest.raises(ValueError, match="Gravitational parameter must be positive"):
            estimate_hohmann_transfer_dv(r1_m, r2_m, -pk.MU_EARTH)

        # Calculate delta-v for valid inputs
        dv_ms = estimate_hohmann_transfer_dv(r1_m, r2_m, pk.MU_EARTH)
        dv_kms = mps_to_kmps(dv_ms)

        # Validate delta-v (should be around 3.9 km/s for Earth-Moon Hohmann)
        assert 3.8 < dv_kms < 4.0, f"Hohmann transfer delta-v {dv_kms:.2f} km/s outside expected range"
