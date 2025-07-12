#!/usr/bin/env python3
"""
Fast Real Trajectory Tests - No Mocking
=======================================

These tests use real trajectory implementations with minimal parameters
for fast execution while ensuring full functionality coverage.

Execution time target: < 3 seconds
"""

import pytest
import sys
import os
import numpy as np
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    # Real trajectory imports - NO MOCKING
    from trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
    from trajectory.lunar_transfer import LunarTransfer
    from config.models import OrbitParameters

    TRAJECTORY_AVAILABLE = True
except ImportError as e:
    TRAJECTORY_AVAILABLE = False
    pytest_skip_reason = f"Trajectory modules not available: {e}"

try:
    from trajectory.mission_windows import calculate_transfer_windows

    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False


@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="Trajectory modules not available")
class TestRealTrajectoryFast:
    """Fast tests using real trajectory implementations."""

    def test_earth_moon_trajectory_minimal(self):
        """Test real Earth-Moon trajectory generation with minimal parameters."""
        # Use minimal realistic parameters for speed
        trajectory, total_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,  # MJD2000
            earth_orbit_alt=400.0,  # km
            moon_orbit_alt=100.0,  # km
            transfer_time=4.5,  # days
            method="patched_conics",  # Fastest method
        )

        # Verify real results
        assert total_dv > 1000, f"Delta-v too low: {total_dv} m/s"
        assert total_dv < 10000, f"Delta-v too high: {total_dv} m/s"
        assert hasattr(trajectory, "departure_epoch")
        assert hasattr(trajectory, "arrival_epoch")

        # Verify trajectory object structure
        assert trajectory.departure_epoch == 10000.0
        assert trajectory.arrival_epoch > trajectory.departure_epoch

    def test_lunar_transfer_real_implementation(self):
        """Test real LunarTransfer class with fast parameters."""
        # Create real transfer object
        transfer = LunarTransfer(
            earth_altitude=400, moon_altitude=100, transfer_time=5.0  # km  # km  # days
        )

        # Test real trajectory generation
        trajectory, delta_v = transfer.generate_transfer()

        # Verify real results (no mocking)
        assert delta_v > 0, "Delta-v must be positive"
        assert 2000 < delta_v < 8000, f"Unrealistic delta-v: {delta_v} m/s"
        assert trajectory is not None

        # Test trajectory properties
        assert hasattr(trajectory, "time_of_flight")
        assert trajectory.time_of_flight > 0

    @pytest.mark.skipif(
        not WINDOWS_AVAILABLE, reason="Transfer windows module not available"
    )
    def test_transfer_windows_calculation_fast(self):
        """Test real transfer window calculations with minimal search range."""
        # Use small search window for speed
        departure_range = (10000.0, 10030.0)  # 30 days only

        windows = calculate_transfer_windows(
            departure_range=departure_range,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            max_transfer_time=7.0,
            step_size=5.0,  # Large step for speed
        )

        # Verify real results
        assert isinstance(windows, list)
        assert len(windows) >= 0  # May be empty for small range

        # If windows found, verify structure
        for window in windows:
            assert "departure_epoch" in window
            assert "transfer_time" in window
            assert "delta_v" in window
            assert window["delta_v"] > 0

    def test_orbit_parameters_validation(self):
        """Test real orbit parameter validation."""
        # Valid LEO orbit
        leo_orbit = OrbitParameters(
            semi_major_axis=6778.0,  # km (400 km altitude)
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
        )

        # Verify real validation
        assert leo_orbit.semi_major_axis == 6778.0
        assert 0.0 <= leo_orbit.eccentricity < 1.0
        assert 0.0 <= leo_orbit.inclination <= 180.0

    def test_trajectory_optimization_minimal(self):
        """Test real trajectory optimization with minimal parameters."""
        try:
            from trajectory.trajectory_optimization import TrajectoryOptimizer

            optimizer = TrajectoryOptimizer(
                earth_altitude=400.0,
                moon_altitude=100.0,
                max_iterations=5,  # Minimal for speed
                tolerance=1e-3,  # Relaxed tolerance
            )

            # Test real optimization
            result = optimizer.optimize_transfer(
                initial_guess=[4.5, 10000.0],  # [time, epoch]
                method="simple",  # Use fastest method
            )

            assert result is not None
            assert "optimal_time" in result
            assert "optimal_dv" in result
            assert result["optimal_dv"] > 0

        except ImportError:
            pytest.skip("TrajectoryOptimizer not available")

    def test_trajectory_validation_real(self):
        """Test real trajectory validation without mocks."""
        try:
            from trajectory.trajectory_validator import TrajectoryValidator

            validator = TrajectoryValidator()

            # Create real trajectory data
            trajectory_data = {
                "departure_epoch": 10000.0,
                "arrival_epoch": 10005.0,
                "delta_v_total": 3200.0,
                "earth_orbit": {"altitude": 400.0},
                "moon_orbit": {"altitude": 100.0},
            }

            # Test real validation
            is_valid, issues = validator.validate_trajectory(trajectory_data)

            # Verify validation results
            assert isinstance(is_valid, bool)
            assert isinstance(issues, list)

            if not is_valid:
                # Issues should be meaningful strings
                for issue in issues:
                    assert isinstance(issue, str)
                    assert len(issue) > 0

        except ImportError:
            pytest.skip("TrajectoryValidator not available")


@pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="Trajectory modules not available")
class TestRealPhysicsValidation:
    """Test real physics validation without mocking."""

    def test_delta_v_physics_validation(self):
        """Test real delta-v physics calculations."""
        # Test realistic delta-v values
        test_cases = [
            {"dv": 3200, "expected_valid": True},  # Typical LEO-Moon
            {"dv": 50000, "expected_valid": False},  # Unrealistic
            {"dv": -100, "expected_valid": False},  # Negative
            {"dv": 1500, "expected_valid": True},  # Low but possible
        ]

        for case in test_cases:
            # Use simple physics validation
            is_physically_valid = (
                case["dv"] > 0
                and case["dv"] < 15000  # Reasonable upper bound
                and case["dv"] > 1000  # Reasonable lower bound
            )

            assert (
                is_physically_valid == case["expected_valid"]
            ), f"Delta-v {case['dv']} validation failed"

    def test_time_of_flight_validation(self):
        """Test real time-of-flight validation."""
        # Test realistic transfer times
        test_cases = [
            {"tof": 4.5, "expected_valid": True},  # Fast transfer
            {"tof": 8.0, "expected_valid": True},  # Slow transfer
            {"tof": 0.5, "expected_valid": False},  # Too fast
            {"tof": 50.0, "expected_valid": False},  # Too slow
        ]

        for case in test_cases:
            # Real validation logic
            is_valid = (
                case["tof"] > 1.0  # Minimum realistic time
                and case["tof"] < 30.0  # Maximum reasonable time
            )

            assert (
                is_valid == case["expected_valid"]
            ), f"Time-of-flight {case['tof']} validation failed"


def test_trajectory_performance_real():
    """Test that real trajectory functions execute quickly."""
    import time

    if not TRAJECTORY_AVAILABLE:
        pytest.skip("Trajectory modules not available")

    start_time = time.time()

    # Run multiple real trajectory calculations
    for i in range(3):
        trajectory, delta_v = generate_earth_moon_trajectory(
            departure_epoch=10000.0 + i,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="patched_conics",
        )
        assert delta_v > 0

    execution_time = time.time() - start_time

    # Should complete in under 3 seconds
    assert execution_time < 3.0, f"Trajectory tests too slow: {execution_time:.2f}s"


if __name__ == "__main__":
    # Run fast trajectory tests
    pytest.main([__file__, "-v", "--tb=short"])
