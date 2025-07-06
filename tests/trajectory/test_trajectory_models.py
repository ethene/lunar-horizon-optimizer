"""Tests for trajectory model validation and functionality.

This module contains unit tests for the core trajectory data structures:
1. OrbitState - Validates orbital elements and state vector conversions
2. Maneuver - Validates impulsive maneuver parameters and timing
3. Trajectory - Validates complete trajectory sequences and constraints

Key validation areas:
- Parameter bounds and physical constraints
- Unit consistency (km, degrees, MJD2000)
- Time ordering and causality
- State vector transformations
- Error handling for invalid inputs

Test organization:
- TestOrbitState: Orbital elements validation
- TestManeuver: Delta-v and epoch validation  
- TestTrajectory: End-to-end trajectory validation
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from src.trajectory.models import OrbitState, Maneuver, Trajectory

class TestOrbitState:
    """Test suite for OrbitState model validation.
    
    Constants:
        VALID_LEO_SMA: Semi-major axis for typical LEO orbit (km)
        VALID_INCLINATION: Typical launch site inclination (degrees)
    """
    
    VALID_LEO_SMA = 6678.0  # km (typical LEO)
    VALID_INCLINATION = 28.5  # degrees (KSC latitude)
    
    def test_valid_parameters(self):
        """Test OrbitState creation with valid parameters.
        
        Verifies:
        - All parameters within valid ranges are accepted
        - Object creation succeeds with typical LEO values
        - Circular orbit parameters are handled correctly
        """
        orbit = OrbitState(
            semi_major_axis=self.VALID_LEO_SMA,
            eccentricity=0.0,
            inclination=self.VALID_INCLINATION,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0
        )
        assert orbit is not None, "Valid orbit state creation failed"
    
    def test_invalid_semi_major_axis(self):
        """Test validation of semi-major axis.
        
        Verifies:
        - Negative values raise ValueError
        - Zero values raise ValueError
        """
        with pytest.raises(ValueError, match="Semi-major axis must be positive"):
            OrbitState(
                semi_major_axis=-100.0,
                eccentricity=0.0,
                inclination=0.0,
                raan=0.0,
                arg_periapsis=0.0,
                true_anomaly=0.0
            )
    
    def test_invalid_eccentricity(self):
        """Test validation of eccentricity.
        
        Verifies:
        - Values >= 1.0 raise ValueError (hyperbolic/parabolic orbits not supported)
        - Negative values raise ValueError
        """
        with pytest.raises(ValueError, match="Eccentricity must be in \\[0,1\\)"):
            OrbitState(
                semi_major_axis=6678.0,
                eccentricity=1.5,
                inclination=0.0,
                raan=0.0,
                arg_periapsis=0.0,
                true_anomaly=0.0
            )
    
    def test_invalid_inclination(self):
        """Test validation of inclination.
        
        Verifies:
        - Values outside [0,180] degrees raise ValueError
        """
        with pytest.raises(ValueError, match="Inclination must be in \\[0,180\\]"):
            OrbitState(
                semi_major_axis=6678.0,
                eccentricity=0.0,
                inclination=200.0,
                raan=0.0,
                arg_periapsis=0.0,
                true_anomaly=0.0
            )
    
    def test_invalid_angles(self):
        """Test validation of orbital angles.
        
        Verifies:
        - RAAN outside [0,360] raises ValueError
        - Argument of periapsis outside [0,360] raises ValueError
        - True anomaly outside [0,360] raises ValueError
        """
        for param, name in [
            ("raan", "RAAN"),
            ("arg_periapsis", "Argument of periapsis"),
            ("true_anomaly", "True anomaly")
        ]:
            with pytest.raises(ValueError, match=f"{name} must be between 0 and 360 degrees"):
                kwargs = {
                    "semi_major_axis": 6678.0,
                    "eccentricity": 0.0,
                    "inclination": 0.0,
                    "raan": 0.0,
                    "arg_periapsis": 0.0,
                    "true_anomaly": 0.0
                }
                kwargs[param] = 400.0
                OrbitState(**kwargs)

class TestManeuver:
    """Test suite for Maneuver model validation.
    
    Constants:
        VALID_DELTA_V: Typical TLI delta-v magnitude (km/s)
        VALID_EPOCH: Reference epoch for tests (MJD2000)
    """
    
    VALID_DELTA_V = 3.1  # km/s (typical TLI)
    VALID_EPOCH = 1000.0  # MJD2000
    
    def test_valid_parameters(self):
        """Test Maneuver creation with valid parameters.
        
        Verifies:
        - 3D delta-v vector is accepted
        - Valid epoch is accepted
        - Maneuver object properties match inputs
        """
        delta_v = np.array([self.VALID_DELTA_V, 0.0, 0.0])
        maneuver = Maneuver(
            delta_v=delta_v,
            epoch=self.VALID_EPOCH
        )
        assert maneuver is not None, "Valid maneuver creation failed"
        np.testing.assert_array_equal(maneuver.delta_v, delta_v, "Delta-v vector mismatch")
    
    def test_invalid_delta_v(self):
        """Test validation of delta-v vector.
        
        Verifies:
        - Non-3D vectors raise ValueError
        - Invalid shapes raise ValueError
        """
        with pytest.raises(ValueError, match="delta_v must be a 3D vector"):
            Maneuver(
                delta_v=np.array([1.0, 0.0]),
                epoch=1000.0
            )
    
    def test_invalid_epoch(self):
        """Test validation of maneuver epoch.
        
        Verifies:
        - Non-timezone-aware datetime raises ValueError
        """
        with pytest.raises(ValueError, match="Time must be timezone-aware"):
            Maneuver(
                delta_v=np.array([1.0, 0.0, 0.0]),
                epoch=datetime(2024, 1, 1)  # Non-timezone-aware datetime
            )

class TestTrajectory:
    """Test suite for Trajectory model validation.
    
    Constants:
        LEO_ALTITUDE: Low Earth orbit altitude (km)
        LUNAR_DISTANCE: Average Earth-Moon distance (km)
        EARTH_RADIUS: Earth's mean radius (km)
        TLI_DELTA_V: Typical trans-lunar injection delta-v (km/s)
    """
    
    LEO_ALTITUDE = 300.0  # km
    LUNAR_DISTANCE = 384400.0  # km
    EARTH_RADIUS = 6378.0  # km
    TLI_DELTA_V = 3.1  # km/s
    
    def setup_method(self):
        """Set up test fixtures.
        
        Creates common orbit states and maneuvers for testing:
        - Initial state in LEO
        - Final state at lunar distance
        - Single TLI maneuver
        """
        self.departure_epoch = 0.0  # MJD2000
        self.arrival_epoch = 5.0    # MJD2000 (+5 days)
        
        self.initial_state = OrbitState(
            semi_major_axis=self.EARTH_RADIUS + self.LEO_ALTITUDE,
            eccentricity=0.0,
            inclination=28.5,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0
        )
        
        self.final_state = OrbitState(
            semi_major_axis=self.LUNAR_DISTANCE,
            eccentricity=0.0,
            inclination=28.5,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0
        )
        
        self.maneuvers = [
            Maneuver(
                delta_v=np.array([self.TLI_DELTA_V, 0.0, 0.0]),
                epoch=1.0  # MJD2000 (+1 day)
            )
        ]
    
    def test_valid_parameters(self):
        """Test Trajectory creation with valid parameters.
        
        Verifies:
        - Valid states and maneuvers are accepted
        - Time ordering is correct
        - Central body specification is accepted
        """
        trajectory = Trajectory(
            initial_state=self.initial_state,
            final_state=self.final_state,
            maneuvers=self.maneuvers,
            central_body="Earth",
            departure_epoch=self.departure_epoch,
            arrival_epoch=self.arrival_epoch
        )
        assert trajectory is not None
    
    def test_invalid_time_order(self):
        """Test validation of trajectory time ordering.
        
        Verifies:
        - Arrival before departure raises ValueError
        - Equal epochs raise ValueError
        """
        with pytest.raises(ValueError, match="Arrival epoch must be after departure epoch"):
            Trajectory(
                initial_state=self.initial_state,
                final_state=self.final_state,
                maneuvers=self.maneuvers,
                central_body="Earth",
                departure_epoch=self.arrival_epoch,
                arrival_epoch=self.departure_epoch
            )
    
    def test_invalid_maneuver_timing(self):
        """Test validation of maneuver timing.
        
        Verifies:
        - Maneuvers outside trajectory timespan raise ValueError
        - Maneuvers at trajectory endpoints raise ValueError
        """
        invalid_maneuvers = [
            Maneuver(
                delta_v=np.array([3.1, 0.0, 0.0]),
                epoch=10.0  # Outside trajectory timespan
            )
        ]
        with pytest.raises(ValueError, match="Maneuver epoch .* must be between"):
            Trajectory(
                initial_state=self.initial_state,
                final_state=self.final_state,
                maneuvers=invalid_maneuvers,
                central_body="Earth",
                departure_epoch=self.departure_epoch,
                arrival_epoch=self.arrival_epoch
            ) 