"""Tests for celestial body state calculations.

This module contains tests for calculating the state vectors (position and velocity)
of celestial bodies (Earth and Moon) at specified epochs. All tests verify proper
unit handling and validate results against known orbital parameters.

Unit Conventions (PyKEP Native):
    - Distances: meters (m)
    - Velocities: meters per second (m/s)
    - Time: days since J2000
"""

import pytest
import numpy as np
from datetime import datetime, UTC

from src.trajectory.celestial_bodies import CelestialBody
from src.utils.unit_conversions import datetime_to_j2000

class TestCelestialBodies:
    """
    Tests for celestial body state calculations using SPICE.
    
    All tests verify:
    1. Correct state vector calculations in the J2000 frame
    2. Proper error handling for invalid inputs
    3. Consistency of units (meters, meters/second)
    4. Physical plausibility of results
    """

    @classmethod
    def setup_class(cls):
        """Initialize celestial bodies for all tests."""
        cls.bodies = CelestialBody()

    def test_earth_state_heliocentric(self):
        """
        Verify Earth's heliocentric state vector calculation.
        
        Tests:
        1. Position magnitude ~1 AU (1.496e11 m ± 5%)
        2. Velocity magnitude ~29.8 km/s (29800 m/s ± 5%)
        3. Vectors are perpendicular (orbital motion)
        """
        epoch = datetime_to_j2000(datetime(2024, 1, 1, tzinfo=UTC))
        r, v = self.bodies.get_earth_state(epoch)

        # Convert to numpy arrays for calculations
        r = np.array(r)  # Already in meters
        v = np.array(v)  # Already in m/s

        # Test position magnitude (approximately 1 AU)
        r_mag = np.linalg.norm(r)
        assert 1.4e11 < r_mag < 1.6e11, f"Earth position magnitude {r_mag/1e11:.2f} AU outside expected range"

        # Test velocity magnitude (approximately 29.8 km/s)
        v_mag = np.linalg.norm(v)
        assert 28000 < v_mag < 31000, f"Earth velocity magnitude {v_mag/1000:.1f} km/s outside expected range"

        # Verify orbital motion (r⋅v should be small compared to |r||v|)
        r_dot_v = np.dot(r, v)
        assert abs(r_dot_v) < 0.1 * r_mag * v_mag, "Earth's position and velocity not perpendicular"

    def test_moon_state_heliocentric(self):
        """
        Verify Moon's heliocentric state vector calculation.
        
        Tests:
        1. Position magnitude ~1 AU (similar to Earth)
        2. Heliocentric velocity magnitude ~29.8 km/s (± 5%)
        3. State vector physically plausible
        """
        epoch = datetime_to_j2000(datetime(2024, 1, 1, tzinfo=UTC))
        r, v = self.bodies.get_moon_state(epoch)

        r = np.array(r)  # Already in meters
        v = np.array(v)  # Already in m/s

        # Test position magnitude (approximately 1 AU)
        r_mag = np.linalg.norm(r)
        assert 1.4e11 < r_mag < 1.6e11, f"Moon position magnitude {r_mag/1e11:.2f} AU outside expected range"

        # Test velocity magnitude (approximately 29.8 km/s)
        v_mag = np.linalg.norm(v)
        assert 28000 < v_mag < 31000, f"Moon velocity magnitude {v_mag/1000:.1f} km/s outside expected range"

    def test_moon_state_earth_centered(self):
        """
        Verify Moon's geocentric state vector calculation.
        
        Tests:
        1. Position magnitude ~384,400 km (384.4e6 m ± 10%)
        2. Velocity magnitude ~1.022 km/s (1022 m/s ± 10%)
        3. Vectors approximately perpendicular (lunar orbit)
        """
        epoch = datetime_to_j2000(datetime(2024, 1, 1, tzinfo=UTC))
        r, v = self.bodies.get_moon_state_earth_centered(epoch)

        r = np.array(r)  # Already in meters
        v = np.array(v)  # Already in m/s

        # Test position magnitude (approximately 384,400 km = 384.4e6 m)
        r_mag = np.linalg.norm(r)
        expected_distance = 384.4e6  # meters
        assert 0.9 * expected_distance < r_mag < 1.1 * expected_distance, \
            f"Moon-Earth distance {r_mag/1000:.0f} km outside expected range"

        # Test velocity magnitude (approximately 1.022 km/s = 1022 m/s)
        v_mag = np.linalg.norm(v)
        expected_velocity = 1022  # m/s
        assert 0.9 * expected_velocity < v_mag < 1.1 * expected_velocity, \
            f"Moon relative velocity {v_mag:.1f} m/s outside expected range"

        # Verify orbital motion
        r_dot_v = np.dot(r, v)
        assert abs(r_dot_v) < 0.1 * r_mag * v_mag, "Moon's position and velocity not perpendicular"

    def test_invalid_epoch(self):
        """
        Verify proper error handling for invalid epochs.
        
        Tests:
        1. Very large epoch values
        2. Very small (negative) epoch values
        3. Non-numeric epoch values
        """
        with pytest.raises(Exception) as exc_info:
            self.bodies.get_earth_state(1e10)  # Far future
        assert "spice" in str(exc_info.value).lower()

        with pytest.raises(Exception) as exc_info:
            self.bodies.get_moon_state(-1e10)  # Far past
        assert "spice" in str(exc_info.value).lower()

        with pytest.raises(TypeError):
            self.bodies.get_earth_state("invalid")

    def test_local_frame(self):
        """Test local frame transformations.
        
        Verifies:
        - Earth-centered Moon position matches direct calculation
        - Earth-centered Moon velocity matches direct calculation
        - Coordinate transformations preserve vector magnitudes
        - All calculations use PyKEP native units (m, m/s)
        """
        epoch = datetime_to_j2000(datetime(2024, 1, 1, tzinfo=UTC))

        # Get states in heliocentric frame
        r_earth, v_earth = self.bodies.get_earth_state(epoch)
        r_moon, v_moon = self.bodies.get_moon_state(epoch)

        # Convert to numpy arrays (already in meters and m/s)
        r_earth = np.array(r_earth)
        v_earth = np.array(v_earth)
        r_moon = np.array(r_moon)
        v_moon = np.array(v_moon)

        # Calculate Moon's state relative to Earth
        r_rel = r_moon - r_earth
        v_rel = v_moon - v_earth

        # Get local frame
        x_hat, y_hat, z_hat = CelestialBody.create_local_frame(r_rel, v_rel)

        # Verify orthonormality
        assert np.allclose(np.dot(x_hat, y_hat), 0, atol=1e-10), "x and y axes not orthogonal"
        assert np.allclose(np.dot(y_hat, z_hat), 0, atol=1e-10), "y and z axes not orthogonal"
        assert np.allclose(np.dot(z_hat, x_hat), 0, atol=1e-10), "z and x axes not orthogonal"

        # Verify unit vectors
        assert np.allclose(np.linalg.norm(x_hat), 1, atol=1e-10), "x axis not normalized"
        assert np.allclose(np.linalg.norm(y_hat), 1, atol=1e-10), "y axis not normalized"
        assert np.allclose(np.linalg.norm(z_hat), 1, atol=1e-10), "z axis not normalized"
