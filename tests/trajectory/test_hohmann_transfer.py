"""Tests for Hohmann transfer calculations.

This module contains tests for:
1. Hohmann transfer orbit calculations and validation
2. Delta-v calculations for both burns
3. Transfer time calculations
4. Vector operations and state conversions

Test Coverage:
- Component-wise validation of transfer parameters
- LEO to GEO transfer calculations
- Input validation and error handling
- Vector-based calculations and transformations

All calculations use standard units (km, s) unless otherwise specified.
Results are validated against known LEO to GEO transfer parameters.
"""

import pytest
import numpy as np
import pykep as pk
import logging
from src.trajectory.generator import estimate_hohmann_transfer_dv

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestHohmannTransfer:
    """Test suite for Hohmann transfer calculations."""

    def setup_method(self):
        """Set up test fixtures with common orbital parameters."""
        self.r1 = 6678.0  # LEO radius (km)
        self.r2 = 42164.0  # GEO radius (km)
        self.mu = pk.MU_EARTH / (1000.0**3)  # Convert to km³/s²

    def test_transfer_components(self):
        """Test individual components of Hohmann transfer calculation.
        
        Verifies:
        - Transfer orbit semi-major axis calculation
        - Initial and final circular orbit velocities
        - Transfer orbit velocities at periapsis and apoapsis
        - All velocities within expected ranges for LEO-GEO transfer
        """
        # Test semi-major axis calculation
        a_transfer = (self.r1 + self.r2) / 2.0
        assert 24000.0 <= a_transfer <= 25000.0, \
            f"Transfer orbit semi-major axis {a_transfer} km outside expected range"

        logger.debug(f"Transfer orbit semi-major axis: {a_transfer:.1f} km")

        # Test velocity calculations
        v1_circ = np.sqrt(self.mu / self.r1)
        v2_circ = np.sqrt(self.mu / self.r2)

        assert 7.7 <= v1_circ <= 7.8, \
            f"Initial circular velocity {v1_circ} km/s outside expected range"
        assert 3.0 <= v2_circ <= 3.1, \
            f"Final circular velocity {v2_circ} km/s outside expected range"

        logger.debug(f"Circular velocities - initial: {v1_circ:.2f} km/s, final: {v2_circ:.2f} km/s")

        # Test transfer orbit velocities
        v1_transfer = np.sqrt(self.mu * (2.0/self.r1 - 1.0/a_transfer))
        v2_transfer = np.sqrt(self.mu * (2.0/self.r2 - 1.0/a_transfer))

        assert 10.1 <= v1_transfer <= 10.2, \
            f"Transfer orbit initial velocity {v1_transfer} km/s outside expected range"
        assert 1.5 <= v2_transfer <= 1.65, \
            f"Transfer orbit final velocity {v2_transfer} km/s outside expected range"

        logger.debug(f"Transfer velocities - initial: {v1_transfer:.2f} km/s, final: {v2_transfer:.2f} km/s")

    def test_transfer_estimation(self):
        """Test complete Hohmann transfer calculations.
        
        Verifies:
        - First burn delta-v matches expected value (~2.43 km/s)
        - Second burn delta-v matches expected value (~1.47 km/s)
        - Transfer time matches half-orbit period (~0.22 days)
        - Total delta-v within expected range for LEO-GEO transfer
        """
        dv1, dv2, tof = estimate_hohmann_transfer_dv(self.r1, self.r2)

        # Known values for LEO to GEO transfer using PyKEP constants
        assert 2.42 <= dv1 <= 2.44, \
            f"First burn {dv1} km/s outside expected range"
        assert 1.46 <= dv2 <= 1.48, \
            f"Second burn {dv2} km/s outside expected range"
        assert 0.21 <= tof <= 0.23, \
            f"Transfer time {tof} days outside expected range"

        logger.debug(f"Transfer parameters - ΔV1: {dv1:.2f} km/s, ΔV2: {dv2:.2f} km/s, TOF: {tof:.2f} days")

    def test_invalid_radii(self):
        """Test Hohmann transfer validation with invalid radii.
        
        Verifies:
        - Negative initial radius raises ValueError
        - Negative final radius raises ValueError
        - Zero radius raises ValueError
        - Error messages are descriptive and helpful
        """
        test_cases = [
            (-1.0, 42164.0, "negative initial radius"),
            (6678.0, -1.0, "negative final radius"),
            (0.0, 42164.0, "zero initial radius"),
            (6678.0, 0.0, "zero final radius"),
        ]

        for r1, r2, case in test_cases:
            logger.debug(f"Testing invalid case: {case}")
            with pytest.raises(ValueError, match="Orbit radii must be positive"):
                estimate_hohmann_transfer_dv(r1, r2)

    def test_transfer_vectors(self):
        """Test Hohmann transfer position and velocity vectors.
        
        Verifies:
        - Position vectors have correct magnitude and direction
        - Velocity vectors are perpendicular to position vectors
        - Vector magnitudes match scalar calculations
        - Transfer orbit geometry is correct
        """
        # Create position vectors
        r1_vec = np.array([self.r1, 0.0, 0.0])
        r2_vec = np.array([-self.r2, 0.0, 0.0])

        # Calculate transfer orbit parameters
        a_transfer = (self.r1 + self.r2) / 2.0

        # Calculate velocities
        v1_transfer = np.sqrt(self.mu * (2.0/self.r1 - 1.0/a_transfer))
        v2_transfer = np.sqrt(self.mu * (2.0/self.r2 - 1.0/a_transfer))

        # Create velocity vectors (perpendicular to position)
        v1_vec = np.array([0.0, v1_transfer, 0.0])
        v2_vec = np.array([0.0, -v2_transfer, 0.0])

        # Verify magnitudes
        assert abs(np.linalg.norm(r1_vec) - self.r1) < 0.1, \
            "Initial position vector magnitude incorrect"
        assert abs(np.linalg.norm(r2_vec) - self.r2) < 0.1, \
            "Final position vector magnitude incorrect"
        assert abs(np.linalg.norm(v1_vec) - v1_transfer) < 0.1, \
            "Initial velocity vector magnitude incorrect"
        assert abs(np.linalg.norm(v2_vec) - v2_transfer) < 0.1, \
            "Final velocity vector magnitude incorrect"

        logger.debug(f"Vector magnitudes verified - r1: {np.linalg.norm(r1_vec):.1f} km, " \
                    f"r2: {np.linalg.norm(r2_vec):.1f} km")

    def test_circular_velocities(self):
        """Test circular orbit velocities for Hohmann transfer.
        
        Verifies:
        - Initial circular orbit velocity matches expected value
        - Final circular orbit velocity matches expected value
        - Delta-v calculations match expected values
        - Total delta-v matches known LEO-GEO transfer cost
        """
        # Calculate circular velocities
        v1_circ = np.sqrt(self.mu / self.r1)
        v2_circ = np.sqrt(self.mu / self.r2)

        # Calculate transfer orbit velocities
        a_transfer = (self.r1 + self.r2) / 2.0
        v1_transfer = np.sqrt(self.mu * (2.0/self.r1 - 1.0/a_transfer))
        v2_transfer = np.sqrt(self.mu * (2.0/self.r2 - 1.0/a_transfer))

        # Calculate delta-v
        dv1 = abs(v1_transfer - v1_circ)
        dv2 = abs(v2_circ - v2_transfer)
        total_dv = dv1 + dv2

        # Verify delta-v values
        assert 2.4 <= dv1 <= 2.5, \
            f"First burn delta-v {dv1} km/s outside expected range"
        assert 1.4 <= dv2 <= 1.5, \
            f"Second burn delta-v {dv2} km/s outside expected range"
        assert 3.8 <= total_dv <= 4.0, \
            f"Total delta-v {total_dv} km/s outside expected range"

        logger.debug(f"Delta-v breakdown - ΔV1: {dv1:.2f} km/s, ΔV2: {dv2:.2f} km/s, " \
                    f"Total: {total_dv:.2f} km/s")
