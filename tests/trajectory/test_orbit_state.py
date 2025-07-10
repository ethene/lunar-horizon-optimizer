"""Tests for orbit state conversions and units.

This module contains tests for:
1. Orbit state conversions between internal models and PyKEP
2. Validation of orbital parameters and state vectors
3. Unit conversions and consistency checks
4. Celestial body state calculations

Unit Conventions (PyKEP Native):
- Positions: meters (m)
- Velocities: meters per second (m/s)
- Times: days since J2000 epoch
- Gravitational parameters: m³/s²
- Angles: degrees (converted to radians internally)

Note: While our OrbitState class accepts inputs in kilometers for human readability,
all interactions with PyKEP use native units (meters, m/s).
"""

import numpy as np
import pykep as pk
from src.trajectory.models import OrbitState
from src.trajectory.celestial_bodies import CelestialBody
from src.utils.unit_conversions import km_to_m, m_to_km, mps_to_kmps


class TestOrbitStateConversion:
    """Test suite for orbit state conversions."""

    def test_circular_orbit(self):
        """Test conversion of circular OrbitState to PyKEP planet.

        Verifies:
        - Position magnitude matches semi-major axis
        - Velocity magnitude matches circular orbit velocity
        - Units are correctly converted between user input (km) and PyKEP native (m)

        Test case: 300 km circular orbit at 28.5° inclination
        """
        # Create orbit with user-friendly units (km)
        orbit = OrbitState(
            semi_major_axis=6678.0,  # km (Earth radius + 300 km)
            eccentricity=0.0,  # circular orbit
            inclination=28.5,  # degrees (KSC latitude)
            raan=0.0,  # degrees
            arg_periapsis=0.0,  # degrees
            true_anomaly=0.0,  # degrees
        )

        # Convert to PyKEP planet (works in meters)
        planet = orbit.to_pykep()

        # Get state at epoch (returns m, m/s)
        r, v = planet.eph(0.0)

        # Position should match semi-major axis
        r_mag = np.linalg.norm(r)  # in meters
        expected_r = km_to_m(orbit.semi_major_axis)  # convert km to m
        assert (
            abs(r_mag - expected_r) < 1000.0
        ), f"Position magnitude {m_to_km(r_mag):.1f} km doesn't match semi-major axis {orbit.semi_major_axis:.1f} km"

        # Velocity should match circular orbit velocity
        v_mag = np.linalg.norm(v)  # in m/s
        mu = pk.MU_EARTH  # m³/s²
        v_expected = np.sqrt(mu / r_mag)  # m/s
        assert (
            abs(v_mag - v_expected) < 100.0
        ), f"Velocity magnitude {mps_to_kmps(v_mag):.3f} km/s doesn't match expected {mps_to_kmps(v_expected):.3f} km/s"

    def test_elliptical_orbit(self):
        """Test conversion of elliptical OrbitState to PyKEP planet.

        Verifies:
        - Position at apoapsis matches expected radius
        - Velocity at apoapsis matches vis-viva equation
        - Units are correctly converted between user input (km) and PyKEP native (m)

        Test case: Highly elliptical orbit (e=0.7) at apoapsis
        """
        # Create orbit with user-friendly units (km)
        orbit = OrbitState(
            semi_major_axis=24521.0,  # km
            eccentricity=0.7,  # highly elliptical
            inclination=28.5,  # degrees
            raan=45.0,  # degrees
            arg_periapsis=90.0,  # degrees
            true_anomaly=180.0,  # degrees (at apoapsis)
        )

        # Convert to PyKEP planet (works in meters)
        planet = orbit.to_pykep()

        # Get state at epoch (returns m, m/s)
        r, v = planet.eph(0.0)

        # Check apoapsis distance
        r_mag = np.linalg.norm(r)  # in meters
        r_a = km_to_m(
            orbit.semi_major_axis * (1 + orbit.eccentricity)
        )  # convert km to m
        assert (
            abs(r_mag - r_a) < 1000.0
        ), f"Position magnitude at apoapsis {m_to_km(r_mag):.1f} km doesn't match expected {m_to_km(r_a):.1f} km"

        # Check velocity at apoapsis using vis-viva equation
        v_mag = np.linalg.norm(v)  # in m/s
        mu = pk.MU_EARTH  # m³/s²
        v_expected = np.sqrt(
            mu * (2 / r_mag - 1 / km_to_m(orbit.semi_major_axis))
        )  # m/s
        assert (
            abs(v_mag - v_expected) < 100.0
        ), f"Velocity magnitude {mps_to_kmps(v_mag):.3f} km/s doesn't match expected {mps_to_kmps(v_expected):.3f} km/s"


class TestCelestialBodyStates:
    """Test celestial body state calculations."""

    def test_moon_state(self):
        """Test Moon state vector calculations.

        Verifies:
        - Position and velocity units are correct (returns m, m/s)
        - Distance from Earth is in expected range (~384,400 km)
        - Earth-centered orbital velocity is reasonable (~1.022 km/s)
        - Heliocentric velocity is ~30 km/s (Earth's velocity)
        """
        # Get Moon state at J2000 (returns m, m/s)
        moon_r, moon_v = CelestialBody.get_moon_state_earth_centered(0.0)

        # Verify Earth-Moon distance (384,400 km ± 10%)
        r_mag = np.linalg.norm(moon_r)  # in meters
        expected_distance = 384400e3  # meters
        assert (
            0.9 * expected_distance <= r_mag <= 1.1 * expected_distance
        ), f"Earth-Moon distance {m_to_km(r_mag):.1f} km outside expected range"

        # Verify Earth-centered velocity (~1.022 km/s ± 10%)
        v_mag = np.linalg.norm(moon_v)  # in m/s
        expected_velocity = 1022.0  # m/s
        assert (
            0.9 * expected_velocity <= v_mag <= 1.1 * expected_velocity
        ), f"Earth-centered velocity {mps_to_kmps(v_mag):.3f} km/s outside expected range"

        # Get heliocentric state for velocity check
        moon_r_helio, moon_v_helio = CelestialBody.get_moon_state(0.0)

        # Verify heliocentric velocity (~29.8 km/s ± 5%)
        v_helio_mag = np.linalg.norm(moon_v_helio)  # in m/s
        expected_helio_velocity = 29800.0  # m/s
        assert (
            0.95 * expected_helio_velocity
            <= v_helio_mag
            <= 1.05 * expected_helio_velocity
        ), f"Heliocentric velocity {mps_to_kmps(v_helio_mag):.3f} km/s outside expected range"

    def test_earth_state(self):
        """Test Earth state vector calculations.

        Verifies:
        - Position is at expected distance from Sun (~1 AU = 149.6e6 km)
        - Heliocentric orbital velocity matches expected value (~29.8 km/s)
        - Units are correctly handled in PyKEP's native format (m, m/s)
        """
        # Get Earth state at J2000 (returns m, m/s)
        r, v = CelestialBody.get_earth_state(0.0)

        # Earth should be at ~1 AU from Sun (149.6e9 m ± 5%)
        r_mag = np.linalg.norm(r)  # in meters
        expected_distance = 149.6e9  # meters (1 AU)
        assert (
            0.95 * expected_distance <= r_mag <= 1.05 * expected_distance
        ), f"Earth distance from Sun {m_to_km(r_mag)/1e6:.2f} million km not near 1 AU"

        # Earth's heliocentric orbital velocity should be ~29.8 km/s (± 5%)
        v_mag = np.linalg.norm(v)  # in m/s
        expected_velocity = 29800.0  # m/s
        assert (
            0.95 * expected_velocity <= v_mag <= 1.05 * expected_velocity
        ), f"Earth heliocentric velocity {mps_to_kmps(v_mag):.3f} km/s outside expected range"
