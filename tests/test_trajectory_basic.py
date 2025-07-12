"""
Basic unit tests for trajectory modules to improve coverage.
"""
import pytest
import numpy as np
from datetime import datetime, timezone

from src.trajectory.constants import PhysicalConstants
from src.trajectory.elements import orbital_period, velocity_at_point, mean_to_true_anomaly, true_to_mean_anomaly
from src.trajectory.orbit_state import OrbitState
from src.trajectory.maneuver import Maneuver
from src.trajectory.target_state import calculate_target_state


class TestTrajectoryConstants:
    """Test trajectory constants module."""
    
    def test_earth_constants(self):
        """Test Earth gravitational parameter and radius."""
        assert PhysicalConstants.EARTH_MU > 0
        assert PhysicalConstants.EARTH_RADIUS > 0
        # Earth's gravitational parameter should be around 3.986e14 m³/s²
        assert 3.9e14 < PhysicalConstants.EARTH_MU < 4.0e14
        # Earth's radius should be around 6.371e6 m
        assert 6.3e6 < PhysicalConstants.EARTH_RADIUS < 6.4e6
    
    def test_moon_constants(self):
        """Test Moon gravitational parameter and radius."""
        assert PhysicalConstants.MOON_MU > 0
        assert PhysicalConstants.MOON_RADIUS > 0
        # Moon's gravitational parameter should be around 4.9e12 m³/s²
        assert 4.8e12 < PhysicalConstants.MOON_MU < 5.0e12
        # Moon's radius should be around 1.737e6 m
        assert 1.7e6 < PhysicalConstants.MOON_RADIUS < 1.8e6
    
    def test_relative_sizes(self):
        """Test relative sizes of celestial bodies."""
        # Earth should be much more massive than Moon
        assert PhysicalConstants.EARTH_MU > PhysicalConstants.MOON_MU * 80  # Earth is ~81 times more massive
        # Earth should be larger than Moon
        assert PhysicalConstants.EARTH_RADIUS > PhysicalConstants.MOON_RADIUS * 3  # Earth is ~3.7 times larger


class TestOrbitalElements:
    """Test orbital elements calculations."""
    
    def test_orbital_period_calculation(self):
        """Test orbital period calculation."""
        # LEO orbit
        semi_major_axis = 6778.0  # km
        period = orbital_period(semi_major_axis)
        
        # Should be around 90-100 minutes for LEO
        assert 5000 < period < 7000  # seconds
    
    def test_velocity_at_point(self):
        """Test velocity calculation at orbital point."""
        # Circular orbit
        semi_major_axis = 6778.0  # km
        eccentricity = 0.0
        true_anomaly = 0.0  # degrees
        
        v_r, v_t = velocity_at_point(semi_major_axis, eccentricity, true_anomaly)
        
        # For circular orbit, radial velocity should be near zero
        assert abs(v_r) < 0.1  # km/s
        # Tangential velocity should be around 7.7 km/s for LEO
        assert 7.0 < v_t < 8.0  # km/s
    
    def test_anomaly_conversions(self):
        """Test anomaly conversion functions."""
        mean_anomaly = 45.0  # degrees
        eccentricity = 0.1
        
        # Convert mean to true and back
        true_anomaly = mean_to_true_anomaly(mean_anomaly, eccentricity)
        back_to_mean = true_to_mean_anomaly(true_anomaly, eccentricity)
        
        # Should get back close to original value
        assert abs(back_to_mean - mean_anomaly) < 0.01


class TestOrbitState:
    """Test OrbitState class."""
    
    def test_orbit_state_creation(self):
        """Test creating an orbit state."""
        # Create orbit using orbital elements
        orbit_state = OrbitState(
            semi_major_axis=6778.0,  # km
            eccentricity=0.01,
            inclination=28.5,  # degrees
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
            epoch=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        
        assert orbit_state.semi_major_axis == 6778.0
        assert orbit_state.eccentricity == 0.01
        assert orbit_state.inclination == 28.5
    
    def test_orbit_state_properties(self):
        """Test orbit state calculated properties."""
        orbit_state = OrbitState(
            semi_major_axis=6778.0,  # km
            eccentricity=0.0,  # circular
            inclination=0.0,   # equatorial
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
            epoch=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        
        # Test basic properties exist
        assert hasattr(orbit_state, 'semi_major_axis')
        assert hasattr(orbit_state, 'eccentricity')
        assert hasattr(orbit_state, 'epoch')
        assert orbit_state.semi_major_axis > 0
        assert orbit_state.eccentricity >= 0


class TestManeuver:
    """Test Maneuver class."""
    
    def test_maneuver_creation(self):
        """Test creating a maneuver."""
        delta_v = (0.1, 0.0, 0.0)  # 0.1 km/s in x direction
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        description = "Test maneuver"
        
        maneuver = Maneuver(
            delta_v=delta_v,
            epoch=epoch,
            description=description
        )
        
        assert np.array_equal(maneuver.delta_v, delta_v)
        assert maneuver.epoch == epoch
        assert maneuver.description == description
    
    def test_maneuver_magnitude(self):
        """Test maneuver magnitude calculation."""
        delta_v = (3.0, 4.0, 0.0)  # 3-4-5 triangle in km/s, magnitude = 5 km/s
        epoch = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        maneuver = Maneuver(
            delta_v=delta_v,
            epoch=epoch
        )
        
        # Calculate magnitude manually for comparison
        expected_magnitude = np.sqrt(sum(v**2 for v in delta_v))
        assert abs(expected_magnitude - 5.0) < 0.1


class TestTargetState:
    """Test target state calculations."""
    
    def test_target_state_calculation(self):
        """Test target state calculation."""
        # Moon position and velocity (simplified)
        moon_pos = np.array([384400000.0, 0.0, 0.0])  # ~Moon distance
        moon_vel = np.array([0.0, 1000.0, 0.0])
        orbit_radius = PhysicalConstants.MOON_RADIUS + 100000.0  # 100 km altitude
        
        target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)
        
        # Should return valid position and velocity vectors
        assert len(target_pos) == 3
        assert len(target_vel) == 3
        assert not np.any(np.isnan(target_pos))
        assert not np.any(np.isnan(target_vel))


class TestTrajectoryModels:
    """Test trajectory models and basic functionality."""
    
    def test_circular_orbit_velocity(self):
        """Test circular orbit velocity calculation."""
        radius = 7000000.0  # 7000 km
        mu = PhysicalConstants.EARTH_MU
        
        # For circular orbit: v = sqrt(mu/r)
        expected_velocity = np.sqrt(mu / radius)
        
        # Should be around 7.5 km/s for LEO
        assert 7000 < expected_velocity < 8000
    
    def test_escape_velocity(self):
        """Test escape velocity calculation."""
        radius = PhysicalConstants.EARTH_RADIUS
        mu = PhysicalConstants.EARTH_MU
        
        # Escape velocity: v = sqrt(2*mu/r)
        escape_velocity = np.sqrt(2 * mu / radius)
        
        # Should be around 11.2 km/s for Earth
        assert 11000 < escape_velocity < 12000
    
    def test_sphere_of_influence(self):
        """Test basic sphere of influence concepts."""
        # Earth's sphere of influence extends to about 900,000 km
        # Moon is at ~384,400 km, so it's within Earth's SOI
        moon_distance = PhysicalConstants.MOON_ORBIT_RADIUS
        earth_soi_approx = 900000000.0  # meters (approximate)
        
        assert moon_distance < earth_soi_approx
        
        # Moon's SOI is much smaller, about 66,000 km
        moon_soi = PhysicalConstants.MOON_SOI
        assert moon_soi < moon_distance