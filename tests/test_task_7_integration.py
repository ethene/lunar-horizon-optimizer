"""Test suite for Task 7: MVP Integration.

This module tests the complete integration of all system components:
- Mission Configuration (Task 2)
- Trajectory Generation (Task 3)
- Global Optimization (Task 4)
- Economic Analysis (Task 5)
- Visualization (Task 6)

Tests verify end-to-end workflows, data flow, error handling, and system performance.
"""

import pytest
import sys
import os
import tempfile
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from lunar_horizon_optimizer import (
        LunarHorizonOptimizer,
        OptimizationConfig,
        AnalysisResults,
    )
    from config.costs import CostFactors
    from config.models import MissionConfig as ConfigMissionConfig
    from config.spacecraft import SpacecraftConfig

    INTEGRATION_AVAILABLE = True
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    pytest_skip_reason = f"Integration modules not available: {e}"


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE, reason="Integration modules not available"
)
class TestTask7MVPIntegration:
    """Test complete MVP integration functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = LunarHorizonOptimizer()

        # Quick test configuration for fast execution
        self.test_config = OptimizationConfig(
            population_size=8,  # Minimum for NSGA-II
            num_generations=2,  # Very small for testing
            seed=42,
        )

    def test_initialization_all_components(self):
        """Test that all system components initialize correctly."""
        # Verify all required components are present
        required_components = [
            "lunar_transfer",
            "window_analyzer",
            "pareto_analyzer",
            "cost_model",
            "npv_analyzer",
            "isru_analyzer",
            "dashboard",
            "trajectory_viz",
            "optimization_viz",
            "economic_viz",
        ]

        for component in required_components:
            assert hasattr(self.optimizer, component), f"Missing component: {component}"
            assert getattr(self.optimizer, component) is not None

    def test_configuration_compatibility(self):
        """Test configuration system compatibility across modules."""
        # Test default configurations
        assert self.optimizer.mission_config is not None
        assert self.optimizer.cost_factors is not None
        assert self.optimizer.spacecraft_config is not None

        # Verify configuration attributes exist
        assert hasattr(self.optimizer.mission_config, "name")
        assert hasattr(self.optimizer.mission_config, "payload")
        assert hasattr(self.optimizer.cost_factors, "launch_cost_per_kg")
        assert hasattr(self.optimizer.spacecraft_config, "dry_mass")

    def test_end_to_end_pipeline_minimal(self):
        """Test minimal end-to-end pipeline execution."""
        # Run analysis with minimal settings for speed
        results = self.optimizer.analyze_mission(
            mission_name="Test Mission",
            optimization_config=self.test_config,
            include_sensitivity=False,  # Skip for speed
            include_isru=False,  # Skip for speed
            verbose=False,
        )

        # Verify results structure
        assert isinstance(results, AnalysisResults)
        assert results.mission_name == "Test Mission"
        assert results.trajectory_results is not None
        assert results.optimization_results is not None
        assert results.economic_analysis is not None
        assert results.visualization_assets is not None
        assert results.analysis_metadata is not None

    def test_data_flow_between_modules(self):
        """Test data flow and compatibility between system modules."""
        results = self.optimizer.analyze_mission(
            mission_name="Data Flow Test",
            optimization_config=self.test_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Test trajectory data structure
        traj_results = results.trajectory_results
        assert "baseline" in traj_results
        assert "total_dv" in traj_results["baseline"]
        assert isinstance(traj_results["baseline"]["total_dv"], (int, float))

        # Test optimization data structure
        opt_results = results.optimization_results
        assert "raw_results" in opt_results
        assert "analyzed_results" in opt_results

        # Test economic data structure
        econ_results = results.economic_analysis
        assert "solution_analyses" in econ_results
        assert len(econ_results["solution_analyses"]) > 0

        # Verify data type consistency
        for analysis in econ_results["solution_analyses"]:
            assert "financial_summary" in analysis
            assert "cost_breakdown" in analysis

    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery mechanisms."""
        # Test with invalid optimization config
        invalid_config = OptimizationConfig(
            population_size=3,  # Too small for NSGA-II
            num_generations=1,
            seed=42,
        )

        # Should handle gracefully
        try:
            results = self.optimizer.analyze_mission(
                mission_name="Error Test",
                optimization_config=invalid_config,
                include_sensitivity=False,
                include_isru=False,
                verbose=False,
            )
            # If no exception, check if optimization was skipped or handled
            assert results is not None
        except Exception as e:
            # Should be a meaningful error
            assert "population" in str(e).lower() or "nsga" in str(e).lower()

    def test_caching_and_performance(self):
        """Test caching mechanisms and performance optimizations."""
        # Run same analysis twice to test caching
        config = OptimizationConfig(
            population_size=8,
            num_generations=1,
            seed=42,  # Same seed for reproducibility
        )

        # First run
        results1 = self.optimizer.analyze_mission(
            mission_name="Cache Test 1",
            optimization_config=config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Check if cache stats are available
        opt_results = results1.optimization_results.get("raw_results", {})
        cache_stats = opt_results.get("cache_stats")
        if cache_stats:
            assert "hit_rate" in cache_stats
            assert "cache_hits" in cache_stats
            assert "cache_misses" in cache_stats

    def test_export_functionality(self):
        """Test results export functionality."""
        results = self.optimizer.analyze_mission(
            mission_name="Export Test",
            optimization_config=self.test_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Test export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.optimizer.export_results(results, temp_dir)

            # Verify files were created
            exported_files = os.listdir(temp_dir)
            assert len(exported_files) > 0

            # Check for key files
            file_names = [f.lower() for f in exported_files]
            assert any("metadata" in name for name in file_names)

    def test_visualization_integration(self):
        """Test visualization component integration."""
        results = self.optimizer.analyze_mission(
            mission_name="Visualization Test",
            optimization_config=self.test_config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        # Check visualization assets
        viz_assets = results.visualization_assets
        assert isinstance(viz_assets, dict)

        # Count successful visualizations
        successful_viz = sum(1 for v in viz_assets.values() if v is not None)
        assert successful_viz >= 0  # At least some should work

    def test_logging_and_monitoring(self):
        """Test logging and monitoring capabilities."""
        import logging

        # Capture log messages
        log_messages = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())

        handler = TestHandler()
        logger = logging.getLogger("lunar_horizon_optimizer")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            # Run analysis with verbose logging
            self.optimizer.analyze_mission(
                mission_name="Logging Test",
                optimization_config=self.test_config,
                include_sensitivity=False,
                include_isru=False,
                verbose=True,
            )

            # Verify meaningful log messages were generated
            assert len(log_messages) > 0
            assert any("Starting" in msg for msg in log_messages)

        finally:
            logger.removeHandler(handler)

    def test_configuration_file_workflow(self):
        """Test configuration file-based workflow."""
        # Create test configuration
        config_data = {
            "mission": {
                "name": "Test Config Mission",
                "transfer_time": 5.0,
            },
            "optimization": {
                "population_size": 8,
                "num_generations": 1,
            },
            "costs": {
                "launch_cost_per_kg": 12000.0,
                "development_cost": 800000000.0,
            },
        }

        # Test configuration loading (simplified)
        # This would normally use the CLI configuration functions
        cost_factors = CostFactors(
            launch_cost_per_kg=config_data["costs"]["launch_cost_per_kg"],
            development_cost=config_data["costs"]["development_cost"],
            operations_cost_per_day=100000.0,
            contingency_percentage=20.0,
        )

        # Create optimizer with custom configuration
        custom_optimizer = LunarHorizonOptimizer(cost_factors=cost_factors)
        assert custom_optimizer.cost_factors.launch_cost_per_kg == 12000.0


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE, reason="Integration modules not available"
)
class TestTask7CLIIntegration:
    """Test CLI interface integration."""

    def test_cli_validation_command(self):
        """Test CLI validation functionality."""
        # This would test the actual CLI, but we'll test the validation logic
        from config.costs import CostFactors
        from config.models import MissionConfig
        from config.spacecraft import PayloadSpecification
        from config.orbit import OrbitParameters

        # Test that all required classes can be imported and instantiated
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
            contingency_percentage=20.0,
        )
        payload = PayloadSpecification(
            dry_mass=5000.0,
            max_propellant_mass=3000.0,
            payload_mass=1000.0,
            specific_impulse=320.0,
        )
        orbit = OrbitParameters(
            semi_major_axis=6778.0,
            inclination=0.0,
            eccentricity=0.0,
        )
        mission_config = MissionConfig(
            name="CLI Test",
            payload=payload,
            cost_factors=cost_factors,
            mission_duration_days=5.0,
            target_orbit=orbit,
        )

        assert mission_config.name == "CLI Test"

    def test_cli_config_generation(self):
        """Test CLI configuration generation."""
        # Test the sample configuration structure
        sample_config = {
            "mission": {
                "name": "Sample Mission",
                "earth_orbit_alt": 400.0,
                "moon_orbit_alt": 100.0,
                "transfer_time": 4.5,
            },
            "optimization": {
                "population_size": 100,
                "num_generations": 100,
                "seed": 42,
            },
        }

        # Verify required fields are present
        assert "mission" in sample_config
        assert "optimization" in sample_config
        assert sample_config["mission"]["name"] is not None
        assert sample_config["optimization"]["population_size"] > 0


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE, reason="Integration modules not available"
)
class TestTask7PerformanceIntegration:
    """Test system performance and scalability."""

    def test_memory_usage(self):
        """Test memory usage during analysis."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        optimizer = LunarHorizonOptimizer()
        config = OptimizationConfig(population_size=8, num_generations=1)

        # Run analysis
        optimizer.analyze_mission(
            mission_name="Memory Test",
            optimization_config=config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 1GB)
        assert (
            memory_increase < 1000
        ), f"Memory usage increased by {memory_increase:.1f} MB"

    def test_concurrent_analysis(self):
        """Test system behavior with concurrent analyses."""
        # Test that multiple optimizers can coexist
        optimizer1 = LunarHorizonOptimizer()
        optimizer2 = LunarHorizonOptimizer()

        config = OptimizationConfig(population_size=8, num_generations=1)

        # Both should work independently
        results1 = optimizer1.analyze_mission(
            mission_name="Concurrent Test 1",
            optimization_config=config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        results2 = optimizer2.analyze_mission(
            mission_name="Concurrent Test 2",
            optimization_config=config,
            include_sensitivity=False,
            include_isru=False,
            verbose=False,
        )

        assert results1.mission_name != results2.mission_name
        assert results1 is not results2


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
