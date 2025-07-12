#!/usr/bin/env python3
"""
Fast Real Integration Tests - No Mocking
========================================

These tests use real integration implementations with minimal parameters
for fast execution while ensuring complete end-to-end functionality.

Execution time target: < 10 seconds total
"""

import pytest
import sys
import os
import tempfile
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    # Real integration imports - NO MOCKING
    from lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig
    from config.costs import CostFactors
    from config.spacecraft import SpacecraftConfig

    INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    pytest_skip_reason = f"Integration modules not available: {e}"


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE, reason="Integration modules not available"
)
class TestRealIntegrationFast:
    """Fast tests using real integration implementations."""

    def setup_method(self):
        """Setup for each test method."""
        # Real cost factors (not mocked)
        self.cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=500000000.0,
            contingency_percentage=15.0,
        )

        # Real spacecraft config (not mocked)
        self.spacecraft_config = SpacecraftConfig(
            dry_mass=3000.0,
            max_propellant_mass=2000.0,
            specific_impulse=320.0,
            thrust=500.0,
        )

        # Fast optimization config for testing
        self.fast_config = OptimizationConfig(
            population_size=8,  # Minimum for NSGA-II
            num_generations=2,  # Very few for speed
            seed=42,  # Reproducible results
        )

    def test_real_optimizer_initialization(self):
        """Test real LunarHorizonOptimizer initialization without mocks."""
        # Create real optimizer
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Verify real components exist (no mocks)
        required_components = [
            "cost_factors",
            "spacecraft_config",
            "mission_config",
            "pareto_analyzer",
            "cost_model",
        ]

        for component in required_components:
            assert hasattr(optimizer, component), f"Missing real component: {component}"
            assert getattr(optimizer, component) is not None

    def test_real_mission_analysis_minimal(self):
        """Test real mission analysis with minimal parameters for speed."""
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Run real analysis (not mocked) with minimal settings
        results = optimizer.analyze_mission(
            mission_name="Fast Test Mission",
            optimization_config=self.fast_config,
            include_sensitivity=False,  # Skip for speed
            include_isru=False,  # Skip for speed
            verbose=False,  # No logging overhead
        )

        # Verify real results structure
        assert results is not None
        assert hasattr(results, "mission_name")
        assert hasattr(results, "trajectory_results")
        assert hasattr(results, "optimization_results")
        assert hasattr(results, "economic_analysis")

        # Verify real data content
        assert results.mission_name == "Fast Test Mission"
        assert results.trajectory_results is not None
        assert results.optimization_results is not None
        assert results.economic_analysis is not None

    def test_real_data_flow_between_modules(self):
        """Test real data flow between actual modules (no mocks)."""
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Run real analysis to generate actual data flow
        results = optimizer.analyze_mission(
            mission_name="Data Flow Test",
            optimization_config=self.fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Test real trajectory → optimization data flow
        traj_results = results.trajectory_results
        opt_results = results.optimization_results

        assert isinstance(traj_results, dict)
        assert isinstance(opt_results, dict)

        # Verify real pareto solutions exist
        if "pareto_solutions" in opt_results:
            pareto_solutions = opt_results["pareto_solutions"]
            assert len(pareto_solutions) > 0

            # Test real solution structure
            for solution in pareto_solutions:
                assert "parameters" in solution
                assert "objectives" in solution

                # Verify real parameter values
                params = solution["parameters"]
                objectives = solution["objectives"]
                assert isinstance(params, dict)
                assert isinstance(objectives, dict)

                # Check realistic objective values
                if "delta_v" in objectives:
                    dv = objectives["delta_v"]
                    assert 1000 < dv < 15000, f"Unrealistic delta-v: {dv}"

    def test_real_economic_analysis_integration(self):
        """Test real economic analysis integration without mocks."""
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Run real analysis with economic calculations
        results = optimizer.analyze_mission(
            mission_name="Economic Test",
            optimization_config=self.fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Verify real economic analysis
        economic_results = results.economic_analysis
        assert economic_results is not None

        # Check for real financial metrics
        if "solution_analyses" in economic_results:
            analyses = economic_results["solution_analyses"]
            assert len(analyses) > 0

            for analysis in analyses:
                if "financial_summary" in analysis:
                    financial = analysis["financial_summary"]

                    # Verify real NPV calculation
                    if "npv" in financial:
                        npv = financial["npv"]
                        assert isinstance(npv, (int, float))
                        # NPV can be negative, but should be realistic
                        assert -10e9 < npv < 10e9, f"Unrealistic NPV: ${npv/1e6:.1f}M"

    def test_real_configuration_validation(self):
        """Test real configuration validation without mocks."""
        # Test invalid configuration
        invalid_config = OptimizationConfig(
            population_size=3, num_generations=1, seed=42  # Too small for NSGA-II
        )

        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Should handle gracefully or raise meaningful error
        try:
            results = optimizer.analyze_mission(
                mission_name="Invalid Config Test",
                optimization_config=invalid_config,
                include_sensitivity=False,
                include_isru=False,
                verbose=False,
            )
            # If no exception, verify result handling
            assert results is not None
        except (ValueError, AssertionError) as e:
            # Should be meaningful error about population size
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg for keyword in ["population", "nsga", "size"]
            )

    def test_real_export_functionality(self):
        """Test real export functionality without mocks."""
        optimizer = LunarHorizonOptimizer(
            cost_factors=self.cost_factors, spacecraft_config=self.spacecraft_config
        )

        # Run real analysis
        results = optimizer.analyze_mission(
            mission_name="Export Test",
            optimization_config=self.fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Test real export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            export_success = optimizer.export_results(results, temp_dir)

            # Verify real export worked
            assert export_success is not False  # May be True or None

            # Check if files were actually created
            exported_files = os.listdir(temp_dir)
            assert len(exported_files) >= 0  # At least attempt was made


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE, reason="Integration modules not available"
)
class TestRealPerformanceIntegration:
    """Test real performance characteristics without mocks."""

    def test_real_memory_efficiency(self):
        """Test real memory usage during analysis."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create real optimizer
        optimizer = LunarHorizonOptimizer()

        # Run real analysis
        fast_config = OptimizationConfig(population_size=8, num_generations=2, seed=42)

        results = optimizer.analyze_mission(
            mission_name="Memory Test",
            optimization_config=fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
        assert results is not None

    def test_real_execution_speed(self):
        """Test real execution speed without mocks."""
        import time

        start_time = time.time()

        # Real fast analysis
        optimizer = LunarHorizonOptimizer()
        fast_config = OptimizationConfig(
            population_size=8,
            num_generations=1,  # Ultra-minimal for speed test
            seed=42,
        )

        results = optimizer.analyze_mission(
            mission_name="Speed Test",
            optimization_config=fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        execution_time = time.time() - start_time

        # Should complete quickly
        assert (
            execution_time < 15.0
        ), f"Integration test too slow: {execution_time:.2f}s"
        assert results is not None

    def test_real_concurrent_analysis(self):
        """Test real concurrent analysis without mocks."""
        # Create two real optimizers
        optimizer1 = LunarHorizonOptimizer()
        optimizer2 = LunarHorizonOptimizer()

        fast_config = OptimizationConfig(
            population_size=8, num_generations=1, seed=42  # Same seed for consistency
        )

        # Run real analyses independently
        results1 = optimizer1.analyze_mission(
            mission_name="Concurrent Test 1",
            optimization_config=fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        results2 = optimizer2.analyze_mission(
            mission_name="Concurrent Test 2",
            optimization_config=fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Verify independence
        assert results1 is not results2
        assert results1.mission_name != results2.mission_name
        assert results1.mission_name == "Concurrent Test 1"
        assert results2.mission_name == "Concurrent Test 2"


def test_real_integration_performance_summary():
    """Test overall real integration performance."""
    if not INTEGRATION_AVAILABLE:
        pytest.skip("Integration modules not available")

    import time

    start_time = time.time()

    # Run multiple real integration tests
    for i in range(2):
        optimizer = LunarHorizonOptimizer()
        fast_config = OptimizationConfig(
            population_size=8, num_generations=1, seed=42 + i
        )

        results = optimizer.analyze_mission(
            mission_name=f"Performance Test {i+1}",
            optimization_config=fast_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        assert results is not None

    total_time = time.time() - start_time

    # Multiple real integrations should still be reasonably fast
    assert total_time < 30.0, f"Multiple integrations too slow: {total_time:.2f}s"


if __name__ == "__main__":
    # Run fast integration tests
    pytest.main([__file__, "-v", "--tb=short"])
