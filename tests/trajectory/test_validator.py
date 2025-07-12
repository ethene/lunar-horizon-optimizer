"""Test module for trajectory validation.

This module contains tests for the TrajectoryValidator class, which handles
validation of trajectory parameters and constraints. Tests cover:
- Input parameter validation
- Delta-v constraint validation
- Edge cases and error handling
- Unit conversion verification
"""

import pytest
from src.trajectory.validation import TrajectoryValidator
from src.utils.unit_conversions import km_to_m, m_to_km
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestTrajectoryValidator:
    """Test suite for TrajectoryValidator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.validator = TrajectoryValidator(
            min_earth_alt=200,
            max_earth_alt=1000,
            min_moon_alt=50,
            max_moon_alt=500,
            min_transfer_time=2.0,
            max_transfer_time=7.0,
        )

    def test_valid_inputs(self):
        """Test that valid inputs pass validation."""
        # Should not raise any exceptions
        self.validator.validate_inputs(
            earth_orbit_alt=300, moon_orbit_alt=100, transfer_time=3.5
        )

    def test_invalid_earth_altitude(self):
        """Test that invalid Earth orbit altitudes raise exceptions."""
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_inputs(
                earth_orbit_alt=100,  # Below minimum
                moon_orbit_alt=100,
                transfer_time=3.5,
            )
        assert "Earth orbit altitude must be between" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_inputs(
                earth_orbit_alt=1200,  # Above maximum
                moon_orbit_alt=100,
                transfer_time=3.5,
            )
        assert "Earth orbit altitude must be between" in str(exc_info.value)

    def test_invalid_moon_altitude(self):
        """Test that invalid lunar orbit altitudes raise exceptions."""
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_inputs(
                earth_orbit_alt=300,
                moon_orbit_alt=25,  # Below minimum
                transfer_time=3.5,
            )
        assert "Moon orbit altitude must be between" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_inputs(
                earth_orbit_alt=300,
                moon_orbit_alt=600,  # Above maximum
                transfer_time=3.5,
            )
        assert "Moon orbit altitude must be between" in str(exc_info.value)

    def test_invalid_transfer_time(self):
        """Test that invalid transfer times raise exceptions."""
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_inputs(
                earth_orbit_alt=300,
                moon_orbit_alt=100,
                transfer_time=1.5,  # Below minimum
            )
        assert "Transfer time must be between" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_inputs(
                earth_orbit_alt=300,
                moon_orbit_alt=100,
                transfer_time=8.0,  # Above maximum
            )
        assert "Transfer time must be between" in str(exc_info.value)

    def test_delta_v_validation(self):
        """Test validation of delta-v values."""
        # Valid delta-v values should not raise exceptions
        self.validator.validate_delta_v(tli_dv=3200, loi_dv=850)  # m/s

        # Test TLI delta-v limit (actual limit is 15000 m/s)
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_delta_v(
                tli_dv=16000, loi_dv=850  # Exceeds 15000 m/s limit
            )
        assert "TLI delta-v" in str(exc_info.value)

        # Test LOI delta-v limit (actual limit is 20000 m/s)
        with pytest.raises(ValueError) as exc_info:
            self.validator.validate_delta_v(
                tli_dv=3200, loi_dv=21000  # Exceeds 20000 m/s limit
            )
        assert "LOI delta-v" in str(exc_info.value)

    def test_edge_cases(self):
        """Test edge cases with minimum and maximum valid values."""
        # Test minimum valid values
        self.validator.validate_inputs(
            earth_orbit_alt=200,  # Minimum Earth altitude
            moon_orbit_alt=50,  # Minimum Moon altitude
            transfer_time=2.0,  # Minimum transfer time
        )

        # Test maximum valid values
        self.validator.validate_inputs(
            earth_orbit_alt=1000,  # Maximum Earth altitude
            moon_orbit_alt=500,  # Maximum Moon altitude
            transfer_time=7.0,  # Maximum transfer time
        )

    def test_unit_conversions(self):
        """Test that unit conversions are handled correctly."""
        # Create validator with values in km
        validator = TrajectoryValidator(
            min_earth_alt=200,  # km
            max_earth_alt=1000,  # km
            min_moon_alt=50,  # km
            max_moon_alt=500,  # km
            min_transfer_time=2.0,  # days
            max_transfer_time=7.0,  # days
        )

        # Internal values should be in meters
        assert validator.min_earth_alt == 200_000  # m
        assert validator.max_earth_alt == 1_000_000  # m
        assert validator.min_moon_alt == 50_000  # m
        assert validator.max_moon_alt == 500_000  # m

        # Test with values in meters
        earth_alt_m = km_to_m(300)  # 300 km to m
        moon_alt_m = km_to_m(100)  # 100 km to m

        # Should handle both km and m inputs correctly
        validator.validate_inputs(
            earth_orbit_alt=300,  # km
            moon_orbit_alt=m_to_km(moon_alt_m),  # Convert back to km
            transfer_time=3.5,
        )

        validator.validate_inputs(
            earth_orbit_alt=m_to_km(earth_alt_m),  # Convert back to km
            moon_orbit_alt=100,  # km
            transfer_time=3.5,
        )
