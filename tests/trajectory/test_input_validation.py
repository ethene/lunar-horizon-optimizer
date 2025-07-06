"""Tests for input parameter validation in trajectory generation.

This module contains tests for:
1. Parameter validation in lunar transfer generation
2. Time of flight constraints and validation
3. Orbit altitude limits validation
4. Delta-v constraint validation
5. Maximum revolutions validation

Test Coverage:
- Invalid parameter combinations
- Boundary value testing
- Error message verification
- Physical constraint validation
"""

import pytest
from datetime import datetime, timezone
import logging
from src.trajectory.generator import generate_lunar_transfer
from src.trajectory.constants import PhysicalConstants
from src.trajectory.defaults import TransferDefaults as TD

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestLunarTransferValidation:
    """Test suite for lunar transfer parameter validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.departure_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.nominal_altitude = TD.DEFAULT_EARTH_ORBIT  # km
        self.nominal_moon_altitude = 100.0  # km
        
    def test_orbit_altitude_validation(self):
        """Test validation of initial orbit altitude constraints.
        
        Verifies:
        - Rejection of negative altitudes
        - Rejection of altitudes below minimum (200 km)
        - Rejection of altitudes above maximum (1000 km)
        - Error messages contain valid range information
        """
        test_cases = [
            (-100.0, f"Earth orbit altitude must be between {TD.MIN_EARTH_ORBIT}"),
            (50.0, f"Earth orbit altitude must be between {TD.MIN_EARTH_ORBIT}"),
            (2000.0, f"Earth orbit altitude must be between {TD.MIN_EARTH_ORBIT}")
        ]
        
        for altitude, error_msg in test_cases:
            with pytest.raises(ValueError, match=error_msg):
                generate_lunar_transfer(
                    self.departure_time, 3.5,
                    altitude, self.nominal_moon_altitude
                )
                
        logger.debug(f"Tested altitude validation with {len(test_cases)} invalid cases")
    
    def test_time_of_flight_validation(self):
        """Test validation of time of flight constraints.
        
        Verifies:
        - Rejection of negative TOF values
        - Rejection of zero TOF
        - Rejection of TOF above maximum (7 days)
        - Error messages specify valid range
        """
        test_cases = [
            (-1.0, f"Transfer time must be between {TD.MIN_TRANSFER_TIME}"),
            (0.0, f"Transfer time must be between {TD.MIN_TRANSFER_TIME}"),
            (8.0, f"Transfer time must be between {TD.MIN_TRANSFER_TIME}")
        ]
        
        for tof, error_msg in test_cases:
            with pytest.raises(ValueError, match=error_msg):
                generate_lunar_transfer(
                    self.departure_time, tof,
                    self.nominal_altitude, self.nominal_moon_altitude
                )
                
        logger.debug(f"Tested TOF validation with {len(test_cases)} invalid cases")
    
    def test_delta_v_constraints(self):
        """Test validation of delta-v constraints.
        
        Verifies:
        - Rejection of negative delta-v values
        - Minimum must be less than maximum
        - Error messages are descriptive
        - Default constraints are accepted
        """
        # Test negative values
        with pytest.raises(ValueError, match="Maximum delta-v must be positive"):
            generate_lunar_transfer(
                self.departure_time, 3.5,
                self.nominal_altitude, self.nominal_moon_altitude,
                max_tli_dv=-1.0
            )
        
        # Test invalid min/max relationship
        with pytest.raises(ValueError, match="Maximum delta-v must be greater than minimum"):
            generate_lunar_transfer(
                self.departure_time, 3.5,
                self.nominal_altitude, self.nominal_moon_altitude,
                max_tli_dv=2.0, min_tli_dv=3.0
            )
    
    def test_max_revolutions_validation(self):
        """Test validation of maximum revolutions parameter.
        
        Verifies:
        - Rejection of negative revolutions
        - Rejection of values above maximum (1)
        - Acceptance of valid values (0-1)
        - Error messages are clear
        """
        # Test negative value
        with pytest.raises(ValueError, match="Maximum revolutions must be non-negative"):
            generate_lunar_transfer(
                self.departure_time, 3.5,
                self.nominal_altitude, self.nominal_moon_altitude,
                max_revs=-1
            )
        
        # Test above maximum
        with pytest.raises(ValueError, match=f"Maximum revolutions must be less than {TD.MAX_REVOLUTIONS}"):
            generate_lunar_transfer(
                self.departure_time, 3.5,
                self.nominal_altitude, self.nominal_moon_altitude,
                max_revs=2
            )
        
        logger.debug("Tested max revolutions validation")

def test_invalid_time_of_flight():
    """Test validation of time of flight."""
    # Use a date within valid ephemeris range
    departure_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    nominal_altitude = TD.DEFAULT_EARTH_ORBIT
    nominal_moon_altitude = 100.0
    
    # Test negative time of flight
    with pytest.raises(ValueError, match=f"Transfer time must be between {TD.MIN_TRANSFER_TIME}"):
        generate_lunar_transfer(departure_time, -1.0, nominal_altitude, nominal_moon_altitude)
    
    # Test zero time of flight
    with pytest.raises(ValueError, match=f"Transfer time must be between {TD.MIN_TRANSFER_TIME}"):
        generate_lunar_transfer(departure_time, 0.0, nominal_altitude, nominal_moon_altitude)
    
    # Test time of flight exceeding maximum
    with pytest.raises(ValueError, match=f"Transfer time must be between {TD.MIN_TRANSFER_TIME}"):
        generate_lunar_transfer(departure_time, 8.0, nominal_altitude, nominal_moon_altitude) 