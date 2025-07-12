"""
Simplified unit tests for utils modules to achieve 80%+ coverage.
"""
import pytest
import numpy as np
from datetime import datetime, timezone

from src.utils.unit_conversions import (
    km_to_m, m_to_km, deg_to_rad, rad_to_deg,
    days_to_seconds, seconds_to_days,
    mps_to_kmps, kmps_to_mps,
    datetime_to_mjd2000, datetime_to_j2000,
    datetime_to_pykep_epoch, pykep_epoch_to_datetime
)


class TestDistanceConversions:
    """Test distance unit conversions."""
    
    def test_km_to_m(self):
        """Test kilometers to meters conversion."""
        assert km_to_m(1.0) == 1000.0
        assert km_to_m(0.0) == 0.0
        assert km_to_m(10.5) == 10500.0
        
        # Test with list
        result = km_to_m([1.0, 2.0, 3.0])
        assert result == [1000.0, 2000.0, 3000.0]
        
        # Test with numpy array
        arr = np.array([1.0, 2.0])
        result = km_to_m(arr)
        np.testing.assert_array_equal(result, np.array([1000.0, 2000.0]))
    
    def test_m_to_km(self):
        """Test meters to kilometers conversion."""
        assert m_to_km(1000.0) == 1.0
        assert m_to_km(0.0) == 0.0
        assert m_to_km(5500.0) == 5.5
        
        # Test with list
        result = m_to_km([1000.0, 2000.0])
        assert result == [1.0, 2.0]
    
    def test_distance_conversion_roundtrip(self):
        """Test roundtrip distance conversions."""
        original = 42.7
        converted = m_to_km(km_to_m(original))
        assert abs(converted - original) < 1e-10


class TestAngleConversions:
    """Test angle unit conversions."""
    
    def test_deg_to_rad(self):
        """Test degrees to radians conversion."""
        assert abs(deg_to_rad(180.0) - np.pi) < 1e-10
        assert abs(deg_to_rad(90.0) - np.pi/2) < 1e-10
        assert deg_to_rad(0.0) == 0.0
        
        # Test with arrays
        angles = [0.0, 90.0, 180.0]
        result = deg_to_rad(angles)
        expected = [0.0, np.pi/2, np.pi]
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10
    
    def test_rad_to_deg(self):
        """Test radians to degrees conversion."""
        assert abs(rad_to_deg(np.pi) - 180.0) < 1e-10
        assert abs(rad_to_deg(np.pi/2) - 90.0) < 1e-10
        assert rad_to_deg(0.0) == 0.0
    
    def test_angle_conversion_roundtrip(self):
        """Test roundtrip angle conversions."""
        original = 45.0
        converted = rad_to_deg(deg_to_rad(original))
        assert abs(converted - original) < 1e-10


class TestTimeConversions:
    """Test time unit conversions."""
    
    def test_days_to_seconds(self):
        """Test days to seconds conversion."""
        assert days_to_seconds(1.0) == 86400.0
        assert days_to_seconds(0.0) == 0.0
        assert days_to_seconds(0.5) == 43200.0
    
    def test_seconds_to_days(self):
        """Test seconds to days conversion."""
        assert seconds_to_days(86400.0) == 1.0
        assert seconds_to_days(0.0) == 0.0
        assert seconds_to_days(43200.0) == 0.5
    
    def test_time_conversion_roundtrip(self):
        """Test roundtrip time conversions."""
        original = 24.0  # days
        converted = seconds_to_days(days_to_seconds(original))
        assert abs(converted - original) < 1e-10


class TestVelocityConversions:
    """Test velocity unit conversions."""
    
    def test_mps_to_kmps(self):
        """Test m/s to km/s conversion."""
        assert abs(mps_to_kmps(1000.0) - 1.0) < 1e-10
        assert mps_to_kmps(0.0) == 0.0
        assert abs(mps_to_kmps(7800.0) - 7.8) < 1e-10
    
    def test_kmps_to_mps(self):
        """Test km/s to m/s conversion.""" 
        assert abs(kmps_to_mps(1.0) - 1000.0) < 1e-10
        assert kmps_to_mps(0.0) == 0.0
        assert abs(kmps_to_mps(7.8) - 7800.0) < 1e-10
    
    def test_velocity_conversion_roundtrip(self):
        """Test roundtrip velocity conversions."""
        original = 7800.0  # m/s
        converted = kmps_to_mps(mps_to_kmps(original))
        assert abs(converted - original) < 1e-10


class TestDateTimeConversions:
    """Test datetime conversion functions."""
    
    def test_datetime_to_mjd2000(self):
        """Test datetime to MJD2000 conversion."""
        # Test reference epoch
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert abs(datetime_to_mjd2000(dt) - 0.0) < 1e-10
        
        # Test J2000 epoch
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert abs(datetime_to_mjd2000(dt) - 0.5) < 1e-10
        
        # Test one day later
        dt = datetime(2000, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        assert abs(datetime_to_mjd2000(dt) - 1.0) < 1e-10
    
    def test_datetime_to_j2000(self):
        """Test datetime to J2000 conversion."""
        # Test J2000 epoch
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert abs(datetime_to_j2000(dt) - 0.0) < 1e-10
        
        # Test MJD2000 epoch
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert abs(datetime_to_j2000(dt) - (-0.5)) < 1e-10
    
    def test_datetime_to_pykep_epoch(self):
        """Test datetime to PyKEP epoch conversion."""
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        mjd2000 = datetime_to_mjd2000(dt)
        pykep_epoch = datetime_to_pykep_epoch(dt)
        
        # Should be the same as MJD2000
        assert abs(mjd2000 - pykep_epoch) < 1e-10
    
    def test_pykep_epoch_to_datetime(self):
        """Test PyKEP epoch to datetime conversion."""
        epoch = 0.0  # MJD2000 reference
        dt = pykep_epoch_to_datetime(epoch)
        
        expected = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert dt == expected
    
    def test_datetime_conversion_roundtrip(self):
        """Test roundtrip datetime conversions."""
        original = datetime(2024, 7, 12, 15, 30, 45, tzinfo=timezone.utc)
        
        # Convert to epoch and back
        epoch = datetime_to_mjd2000(original)
        converted = pykep_epoch_to_datetime(epoch)
        
        # Should be equal (within microsecond precision)
        assert abs((converted - original).total_seconds()) < 1e-6
    
    def test_naive_datetime_error(self):
        """Test that naive datetime raises error."""
        naive_dt = datetime(2000, 1, 1, 0, 0, 0)  # No timezone
        
        with pytest.raises(ValueError):
            datetime_to_mjd2000(naive_dt)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_negative_values(self):
        """Test conversions with negative values."""
        assert km_to_m(-5.0) == -5000.0
        assert deg_to_rad(-90.0) == -np.pi/2
        assert days_to_seconds(-2.0) == -172800.0
    
    def test_zero_values(self):
        """Test conversions with zero values."""
        assert km_to_m(0.0) == 0.0
        assert deg_to_rad(0.0) == 0.0
        assert days_to_seconds(0.0) == 0.0
        assert mps_to_kmps(0.0) == 0.0
    
    def test_large_values(self):
        """Test conversions with large values."""
        large_value = 1e6
        assert km_to_m(large_value) == large_value * 1000
        assert days_to_seconds(large_value) == large_value * 86400
    
    def test_precision_limits(self):
        """Test precision limits for conversions."""
        tiny_value = 1e-15
        converted = m_to_km(km_to_m(tiny_value))
        assert abs(converted - tiny_value) < 1e-20
    
    def test_array_type_preservation(self):
        """Test that array types are preserved."""
        # Test list preservation
        km_list = [1.0, 2.0, 3.0]
        m_result = km_to_m(km_list)
        assert isinstance(m_result, list)
        
        # Test tuple preservation 
        km_tuple = (1.0, 2.0, 3.0)
        m_result = km_to_m(km_tuple)
        assert isinstance(m_result, tuple)
        
        # Test numpy array preservation
        km_array = np.array([1.0, 2.0, 3.0])
        m_result = km_to_m(km_array)
        assert isinstance(m_result, np.ndarray)