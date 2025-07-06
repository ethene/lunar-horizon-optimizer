"""Tests for epoch conversion utilities.

This module verifies the conversion functions between different time formats:
- datetime objects
- Modified Julian Date 2000 (MJD2000)
- PyKEP epochs
"""

import pytest
from datetime import datetime, timezone
import pykep as pk
from src.utils.unit_conversions import (
    datetime_to_mjd2000,
    datetime_to_pykep_epoch,
    pykep_epoch_to_datetime
)

def test_datetime_to_mjd2000():
    """Test conversion from datetime to MJD2000."""
    # Test J2000 epoch
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mjd = datetime_to_mjd2000(dt)
    assert abs(mjd - 0.5) < 1e-10, "J2000 epoch should be MJD2000 = 0.5"
    
    # Test arbitrary date
    dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    mjd = datetime_to_mjd2000(dt)
    assert 8766.0 <= mjd <= 8767.0, "2024-01-01 should be around MJD2000 = 8766.5"

def test_datetime_to_pykep():
    """Test conversion from datetime to PyKEP epoch."""
    # Test J2000 epoch
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    epoch = datetime_to_pykep_epoch(dt)
    assert abs(epoch.mjd2000 - 0.5) < 1e-10, "J2000 epoch should be MJD2000 = 0.5"
    
    # Test arbitrary date
    dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    epoch = datetime_to_pykep_epoch(dt)
    assert 8766.0 <= epoch.mjd2000 <= 8767.0, "2024-01-01 should be around MJD2000 = 8766.5"

def test_pykep_epoch_roundtrip():
    """Test roundtrip conversion between datetime and PyKEP epoch."""
    # Test current time
    dt = datetime.now(timezone.utc)
    epoch = datetime_to_pykep_epoch(dt)
    dt2 = pykep_epoch_to_datetime(epoch)
    
    # Compare timestamps (within 1 second due to floating point precision)
    assert abs((dt - dt2).total_seconds()) < 1.0, "Roundtrip conversion should preserve time"

class TestPhysicalConstants:
    """Test suite for physical constants validation.
    
    Ensures that our use of PyKEP physical constants matches expected values
    and that unit conversions are handled correctly.
    """
    
    def test_earth_gravitational_parameter(self):
        """Verify Earth's gravitational parameter (mu) conversion.
        
        Verifies:
            - PyKEP's Earth mu converts correctly from m³/s² to km³/s²
            - Converted value matches expected value within 0.1 km³/s² tolerance
        """
        mu_earth_expected = 398600.4418  # km³/s²
        mu_earth_pykep = pk.MU_EARTH / (1000.0**3)  # Convert from m³/s² to km³/s²
        assert abs(mu_earth_pykep - mu_earth_expected) < 0.1, \
            f"PyKEP Earth mu {mu_earth_pykep} km³/s² doesn't match expected {mu_earth_expected} km³/s²"
    
    def test_earth_radius(self):
        """Verify Earth radius conversion from PyKEP.
        
        Verifies:
            - PyKEP's Earth radius converts correctly from meters to kilometers
            - Converted value falls within expected range (6378.1 km to 6378.2 km)
        """
        earth_radius_pykep = pk.EARTH_RADIUS / 1000.0  # Convert from m to km
        assert 6378.1 <= earth_radius_pykep <= 6378.2, \
            f"PyKEP Earth radius {earth_radius_pykep} km outside expected range" 