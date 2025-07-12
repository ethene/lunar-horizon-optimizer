#!/usr/bin/env python3
"""
Comprehensive Fast Real Tests - No Mocking
==========================================

Single file with all fast, real implementation tests for core functionality.
No mocks - all tests use actual implementations with minimal parameters.

Execution time target: < 15 seconds total
"""

import pytest
import sys
import os
import time
import tempfile

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Test all core modules with real implementations
try:
    # Trajectory modules
    from trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
    from trajectory.lunar_transfer import LunarTransfer
    from config.models import OrbitParameters

    TRAJECTORY_AVAILABLE = True
except ImportError:
    TRAJECTORY_AVAILABLE = False

try:
    # Optimization modules
    from optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
    from optimization.pareto_analysis import ParetoAnalyzer
    from config.costs import CostFactors

    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

try:
    # Integration modules
    from lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig
    from config.spacecraft import SpacecraftConfig

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    # Economics modules
    from economics.financial_models import (
        NPVAnalyzer,
        ROICalculator,
        FinancialParameters,
    )
    from economics.isru_benefits import ISRUBenefitAnalyzer

    ECONOMICS_AVAILABLE = True
except ImportError:
    ECONOMICS_AVAILABLE = False


class TestRealTrajectoryCore:
    """Fast real trajectory tests without mocks."""

    @pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="Trajectory not available")
    def test_earth_moon_trajectory_real(self):
        """Test real Earth-Moon trajectory generation."""
        trajectory, delta_v = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="patched_conics",
        )

        assert 1000 < delta_v < 10000, f"Delta-v out of range: {delta_v}"
        assert hasattr(trajectory, "departure_epoch")
        assert trajectory.departure_epoch == 10000.0

    @pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="Trajectory not available")
    def test_lunar_transfer_real(self):
        """Test real LunarTransfer implementation."""
        transfer = LunarTransfer(
            earth_altitude=400, moon_altitude=100, transfer_time=5.0
        )

        trajectory, delta_v = transfer.generate_transfer()
        assert delta_v > 0
        assert 2000 < delta_v < 8000, f"Delta-v unrealistic: {delta_v}"

    @pytest.mark.skipif(not TRAJECTORY_AVAILABLE, reason="Trajectory not available")
    def test_orbit_parameters_real(self):
        """Test real orbit parameter validation."""
        orbit = OrbitParameters(
            semi_major_axis=6778.0,
            eccentricity=0.0,
            inclination=0.0,
            raan=0.0,
            arg_periapsis=0.0,
            true_anomaly=0.0,
        )

        assert orbit.semi_major_axis == 6778.0
        assert 0.0 <= orbit.eccentricity < 1.0


class TestRealOptimizationCore:
    """Fast real optimization tests without mocks."""

    def setup_method(self):
        """Setup cost factors for optimization tests."""
        if OPTIMIZATION_AVAILABLE:
            self.cost_factors = CostFactors(
                launch_cost_per_kg=10000.0,
                operations_cost_per_day=100000.0,
                development_cost=500000000.0,
                contingency_percentage=15.0,
            )

    @pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization not available")
    def test_lunar_mission_problem_real(self):
        """Test real LunarMissionProblem without mocks."""
        problem = LunarMissionProblem(cost_factors=self.cost_factors)

        # Test real problem properties
        assert problem.get_nobj() == 3  # Three objectives: delta_v, time, cost
        bounds = problem.get_bounds()
        assert len(bounds[0]) == 3  # Three variables
        assert len(bounds[1]) == 3
        assert all(bounds[1][i] > bounds[0][i] for i in range(3))

    @pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization not available")
    def test_real_fitness_evaluation(self):
        """Test real fitness evaluation."""
        problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=400,
            max_earth_alt=450,
            min_moon_alt=100,
            max_moon_alt=120,
            min_transfer_time=4.5,
            max_transfer_time=5.5,
        )

        # Test with realistic values
        decision_vector = [425.0, 110.0, 5.0]
        fitness = problem.fitness(decision_vector)

        assert len(fitness) == 3
        delta_v, time_sec, cost = fitness
        assert 1000 < delta_v < 10000, f"Delta-v unrealistic: {delta_v}"
        assert 300000 < time_sec < 600000, f"Time unrealistic: {time_sec}"
        assert cost > 0, f"Cost should be positive: {cost}"

    @pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization not available")
    def test_global_optimizer_real(self):
        """Test real GlobalOptimizer with minimal parameters."""
        problem = LunarMissionProblem(cost_factors=self.cost_factors)

        optimizer = GlobalOptimizer(
            population_size=8,  # Minimum for NSGA-II
            generations=2,  # Very few for speed
            seed=42,
        )

        results = optimizer.optimize(problem)

        assert hasattr(results, "pareto_solutions")
        assert len(results.pareto_solutions) > 0

    @pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Optimization not available")
    def test_pareto_analyzer_real(self):
        """Test real ParetoAnalyzer without mocks."""
        analyzer = ParetoAnalyzer()

        # Real solution data
        solutions = [
            {
                "parameters": {"earth_alt": 400},
                "objectives": {"delta_v": 3200, "cost": 150e6},
            },
            {
                "parameters": {"earth_alt": 450},
                "objectives": {"delta_v": 3000, "cost": 180e6},
            },
            {
                "parameters": {"earth_alt": 350},
                "objectives": {"delta_v": 3500, "cost": 120e6},
            },
        ]

        pareto_front = analyzer.find_pareto_front(solutions)

        assert len(pareto_front) > 0
        assert len(pareto_front) <= len(solutions)


class TestRealIntegrationCore:
    """Fast real integration tests without mocks."""

    def setup_method(self):
        """Setup integration test components."""
        if INTEGRATION_AVAILABLE:
            self.cost_factors = CostFactors(
                launch_cost_per_kg=10000.0,
                operations_cost_per_day=100000.0,
                development_cost=500000000.0,
                contingency_percentage=15.0,
            )

            self.spacecraft_config = SpacecraftConfig(
                dry_mass=3000.0,
                max_propellant_mass=2000.0,
                specific_impulse=320.0,
                thrust=500.0,
            )

            self.fast_config = OptimizationConfig(
                population_size=8, num_generations=1, seed=42  # Ultra-minimal for speed
            )

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_optimizer_initialization_real(self):
        """Test real LunarHorizonOptimizer initialization."""
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Verify real components exist
        assert hasattr(optimizer, "cost_factors")
        assert hasattr(optimizer, "spacecraft_config")
        assert hasattr(optimizer, "mission_config")
        assert optimizer.cost_factors is not None
        assert optimizer.spacecraft_config is not None

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_mission_analysis_real(self):
        """Test real mission analysis with minimal parameters."""
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Run real analysis
        results = optimizer.analyze_mission(
            mission_name="Fast Real Test",
            optimization_config=self.fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Verify real results
        assert results is not None
        assert hasattr(results, "mission_name")
        assert results.mission_name == "Fast Real Test"
        assert hasattr(results, "trajectory_results")
        assert hasattr(results, "optimization_results")
        assert hasattr(results, "economic_analysis")


class TestRealEconomicsCore:
    """Fast real economics tests without mocks."""

    @pytest.mark.skipif(not ECONOMICS_AVAILABLE, reason="Economics not available")
    def test_financial_parameters_real(self):
        """Test real FinancialParameters validation."""
        params = FinancialParameters(
            discount_rate=0.08,
            inflation_rate=0.03,
            tax_rate=0.25,
            project_duration_years=10,
        )

        assert 0.0 < params.discount_rate < 1.0
        assert 0.0 < params.inflation_rate < 1.0
        assert 0.0 < params.tax_rate < 1.0
        assert params.project_duration_years > 0

    @pytest.mark.skipif(not ECONOMICS_AVAILABLE, reason="Economics not available")
    def test_roi_calculator_real(self):
        """Test real ROI calculator."""
        calculator = ROICalculator()

        # Test realistic scenarios
        roi = calculator.calculate_simple_roi(100e6, 150e6)
        assert abs(roi - 0.50) < 0.01, f"ROI calculation error: {roi}"

        roi_loss = calculator.calculate_simple_roi(200e6, 180e6)
        assert abs(roi_loss - (-0.10)) < 0.01, f"ROI loss calculation error: {roi_loss}"

    @pytest.mark.skipif(not ECONOMICS_AVAILABLE, reason="Economics not available")
    def test_isru_analyzer_real(self):
        """Test real ISRU benefits analyzer."""
        analyzer = ISRUBenefitAnalyzer()

        # Test water production savings
        savings = analyzer.calculate_savings("water", 1000, 30)

        assert savings > 0
        assert isinstance(savings, (int, float))


def test_performance_all_modules():
    """Test that all real implementations execute quickly."""
    start_time = time.time()

    test_count = 0

    # Run trajectory tests if available
    if TRAJECTORY_AVAILABLE:
        trajectory, delta_v = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="patched_conics",
        )
        assert delta_v > 0
        test_count += 1

    # Run optimization tests if available
    if OPTIMIZATION_AVAILABLE:
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=500000000.0,
            contingency_percentage=15.0,
        )
        problem = LunarMissionProblem(cost_factors=cost_factors)
        assert problem.get_nobj() == 3
        test_count += 1

    # Run integration tests if available
    if INTEGRATION_AVAILABLE:
        optimizer = LunarHorizonOptimizer()
        assert optimizer is not None
        test_count += 1

    # Run economics tests if available
    if ECONOMICS_AVAILABLE:
        calculator = ROICalculator()
        roi = calculator.calculate_simple_roi(100e6, 150e6)
        assert roi > 0
        test_count += 1

    execution_time = time.time() - start_time

    # All tests should complete quickly
    assert execution_time < 15.0, f"All tests too slow: {execution_time:.2f}s"
    assert test_count > 0, "No modules available for testing"

    print(
        f"✅ {test_count} modules tested in {execution_time:.2f}s - All real implementations!"
    )


def test_no_mocking_verification():
    """Verify that no mocking is used in any tests."""
    import inspect

    # Check this module for mock usage
    current_module = sys.modules[__name__]
    source = inspect.getsource(current_module)

    # Should not contain any mocking
    mock_terms = ["mock", "Mock", "patch", "@patch", "MagicMock"]
    for term in mock_terms:
        assert (
            term not in source
        ), f"Found mocking term '{term}' in real implementation tests"

    print("✅ No mocking detected - All tests use real implementations!")


if __name__ == "__main__":
    # Run comprehensive fast real tests
    pytest.main([__file__, "-v", "--tb=short"])
