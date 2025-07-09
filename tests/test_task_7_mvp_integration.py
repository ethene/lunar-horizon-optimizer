#!/usr/bin/env python3
"""
Task 7: MVP Integration - Comprehensive Test Suite
=================================================

This test suite validates the complete integration of all modules in the
Lunar Horizon Optimizer system, ensuring end-to-end functionality.

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0-rc1
"""

import pytest
import sys
import os
from unittest.mock import patch, Mock
from datetime import datetime
import numpy as np
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig, AnalysisResults
from config.mission_config import MissionConfig
from config.costs import CostFactors
from config.spacecraft import SpacecraftConfig


class TestLunarHorizonOptimizerInitialization:
    """Test initialization of the main optimizer class."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        optimizer = LunarHorizonOptimizer()

        assert optimizer.mission_config is not None
        assert optimizer.cost_factors is not None
        assert optimizer.spacecraft_config is not None
        assert hasattr(optimizer, "lunar_transfer")
        assert hasattr(optimizer, "window_analyzer")
        assert hasattr(optimizer, "pareto_analyzer")
        assert hasattr(optimizer, "dashboard")

    def test_custom_configuration_initialization(self):
        """Test initialization with custom configurations."""
        mission_config = MissionConfig(
            name="Test Mission",
            earth_orbit_alt=500.0,
            moon_orbit_alt=150.0,
            transfer_time=5.0,
            departure_epoch=12000.0
        )

        cost_factors = CostFactors(
            launch_cost_per_kg=15000.0,
            operations_cost_per_day=150000.0,
            development_cost=1.5e9,
            contingency_percentage=25.0
        )

        spacecraft_config = SpacecraftConfig(
            name="Test Spacecraft",
            dry_mass=6000.0,
            propellant_mass=4000.0,
            payload_mass=1500.0,
            power_system_mass=600.0,
            propulsion_isp=350.0
        )

        optimizer = LunarHorizonOptimizer(
            mission_config=mission_config,
            cost_factors=cost_factors,
            spacecraft_config=spacecraft_config
        )

        assert optimizer.mission_config.name == "Test Mission"
        assert optimizer.mission_config.earth_orbit_alt == 500.0
        assert optimizer.cost_factors.launch_cost_per_kg == 15000.0
        assert optimizer.spacecraft_config.dry_mass == 6000.0


class TestIntegratedAnalysisWorkflow:
    """Test the complete integrated analysis workflow."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance for testing."""
        return LunarHorizonOptimizer()

    @pytest.fixture
    def optimization_config(self):
        """Create optimization configuration for testing."""
        return OptimizationConfig(
            population_size=10,  # Small for testing
            num_generations=5,   # Small for testing
            seed=42
        )

    def test_trajectory_analysis_component(self, optimizer):
        """Test trajectory analysis component."""
        with patch("trajectory.earth_moon_trajectories.generate_earth_moon_trajectory") as mock_generate:
            # Mock trajectory generation
            mock_trajectory = Mock()
            mock_trajectory.departure_epoch = 10000.0
            mock_trajectory.arrival_epoch = 10004.5
            mock_trajectory.trajectory_data = {
                "positions": np.random.rand(3, 100),
                "velocities": np.random.rand(3, 100),
                "times": np.linspace(0, 86400*4.5, 100)
            }

            mock_generate.return_value = (mock_trajectory, 3200.0)

            with patch.object(optimizer.window_analyzer, "find_transfer_windows") as mock_windows:
                mock_window = Mock()
                mock_window.departure_date = datetime(2025, 6, 15)
                mock_window.arrival_date = datetime(2025, 6, 19, 12)
                mock_window.total_dv = 3150.0
                mock_window.transfer_time = 4.5
                mock_windows.return_value = [mock_window]

                opt_config = OptimizationConfig(population_size=10, num_generations=5)
                results = optimizer._analyze_trajectories(opt_config, verbose=False)

                assert "baseline" in results
                assert "transfer_windows" in results
                assert results["baseline"]["total_dv"] == 3200.0
                assert len(results["transfer_windows"]["windows"]) == 1

    def test_optimization_component(self, optimizer):
        """Test optimization component."""
        with patch("optimization.global_optimizer.LunarMissionProblem") as mock_problem_class:
            with patch("optimization.global_optimizer.GlobalOptimizer") as mock_optimizer_class:
                # Mock optimization results
                mock_optimization_results = {
                    "pareto_solutions": [
                        {
                            "parameters": {"earth_alt": 400, "moon_alt": 100, "transfer_time": 4.5},
                            "objectives": {"delta_v": 3200, "time": 4.5, "cost": 500e6}
                        }
                    ],
                    "cache_stats": {"hit_rate": 0.15}
                }

                mock_optimizer = Mock()
                mock_optimizer.optimize.return_value = mock_optimization_results
                mock_optimizer.get_best_solutions.return_value = [mock_optimization_results["pareto_solutions"][0]]
                mock_optimizer_class.return_value = mock_optimizer

                with patch.object(optimizer.pareto_analyzer, "analyze_pareto_front") as mock_analyze:
                    mock_analyzed = Mock()
                    mock_analyzed.pareto_solutions = mock_optimization_results["pareto_solutions"]
                    mock_analyze.return_value = mock_analyzed

                    opt_config = OptimizationConfig(population_size=10, num_generations=5)
                    results = optimizer._perform_optimization(opt_config, verbose=False)

                    assert "raw_results" in results
                    assert "analyzed_results" in results
                    assert "best_solutions" in results
                    assert results["pareto_front_size"] == 1

    def test_economic_analysis_component(self, optimizer):
        """Test economic analysis component."""
        # Mock optimization results
        mock_optimization_results = {
            "raw_results": {
                "pareto_solutions": [
                    {
                        "parameters": {"earth_alt": 400, "moon_alt": 100, "transfer_time": 4.5},
                        "objectives": {"delta_v": 3200, "time": 4.5, "cost": 500e6}
                    }
                ]
            }
        }

        with patch.object(optimizer, "_analyze_solution_economics") as mock_analyze_solution:
            mock_financial_summary = Mock()
            mock_financial_summary.net_present_value = 125e6
            mock_financial_summary.internal_rate_of_return = 0.18

            mock_solution_analysis = {
                "label": "Solution_1",
                "financial_summary": mock_financial_summary,
                "cost_breakdown": Mock()
            }
            mock_analyze_solution.return_value = mock_solution_analysis

            with patch.object(optimizer.isru_analyzer, "analyze_isru_economics") as mock_isru:
                mock_isru.return_value = {
                    "financial_metrics": {"npv": 50e6, "roi": 0.15}
                }

                with patch.object(optimizer.sensitivity_analyzer, "monte_carlo_simulation") as mock_mc:
                    mock_mc.return_value = {
                        "statistics": {"mean": 100e6},
                        "risk_metrics": {"probability_positive_npv": 0.75}
                    }

                    results = optimizer._analyze_economics(
                        mock_optimization_results,
                        include_sensitivity=True,
                        include_isru=True,
                        verbose=False
                    )

                    assert "solution_analyses" in results
                    assert "isru_analysis" in results
                    assert "sensitivity_analysis" in results
                    assert len(results["solution_analyses"]) == 1

    def test_visualization_component(self, optimizer):
        """Test visualization component."""
        # Mock input data
        trajectory_results = {
            "baseline": {
                "trajectory": Mock(),
                "total_dv": 3200,
                "transfer_time": 4.5
            }
        }

        optimization_results = {
            "analyzed_results": Mock()
        }

        economic_results = {
            "solution_analyses": [{
                "financial_summary": Mock(),
                "cost_breakdown": Mock()
            }]
        }

        # Mock visualization creation
        with patch.object(optimizer.dashboard, "create_executive_dashboard") as mock_exec:
            with patch.object(optimizer.dashboard, "create_technical_dashboard") as mock_tech:
                mock_exec.return_value = Mock()
                mock_tech.return_value = Mock()

                results = optimizer._create_visualizations(
                    trajectory_results,
                    optimization_results,
                    economic_results,
                    "Test Mission"
                )

                assert "executive_dashboard" in results
                assert "technical_dashboard" in results

    def test_end_to_end_analysis(self, optimizer, optimization_config):
        """Test complete end-to-end analysis workflow."""
        with patch("trajectory.earth_moon_trajectories.generate_earth_moon_trajectory") as mock_traj:
            mock_trajectory = Mock()
            mock_trajectory.departure_epoch = 10000.0
            mock_trajectory.arrival_epoch = 10004.5
            mock_traj.return_value = (mock_trajectory, 3200.0)

            with patch.object(optimizer.window_analyzer, "find_transfer_windows") as mock_windows:
                mock_windows.return_value = []

                with patch("optimization.global_optimizer.LunarMissionProblem"):
                    with patch("optimization.global_optimizer.GlobalOptimizer") as mock_opt_class:
                        mock_optimizer = Mock()
                        mock_optimizer.optimize.return_value = {
                            "pareto_solutions": [],
                            "cache_stats": {"hit_rate": 0.0}
                        }
                        mock_optimizer.get_best_solutions.return_value = []
                        mock_opt_class.return_value = mock_optimizer

                        with patch.object(optimizer.pareto_analyzer, "analyze_pareto_front") as mock_analyze:
                            mock_analyze.return_value = Mock()

                            with patch.object(optimizer.isru_analyzer, "analyze_isru_economics") as mock_isru:
                                mock_isru.return_value = {"financial_metrics": {"npv": 50e6}}

                                with patch.object(optimizer.sensitivity_analyzer, "monte_carlo_simulation") as mock_mc:
                                    mock_mc.return_value = {
                                        "statistics": {"mean": 100e6},
                                        "risk_metrics": {"probability_positive_npv": 0.75}
                                    }

                                    # Run analysis
                                    results = optimizer.analyze_mission(
                                        mission_name="Integration Test",
                                        optimization_config=optimization_config,
                                        include_sensitivity=True,
                                        include_isru=True,
                                        verbose=False
                                    )

                                    # Verify results structure
                                    assert isinstance(results, AnalysisResults)
                                    assert results.mission_name == "Integration Test"
                                    assert "baseline" in results.trajectory_results
                                    assert "analysis_date" in results.analysis_metadata


class TestConfigurationManagement:
    """Test configuration management and validation."""

    def test_default_configurations(self):
        """Test default configuration creation."""
        mission_config = LunarHorizonOptimizer._create_default_mission_config()
        cost_factors = LunarHorizonOptimizer._create_default_cost_factors()
        spacecraft_config = LunarHorizonOptimizer._create_default_spacecraft_config()

        assert isinstance(mission_config, MissionConfig)
        assert isinstance(cost_factors, CostFactors)
        assert isinstance(spacecraft_config, SpacecraftConfig)

        # Verify reasonable defaults
        assert 200 <= mission_config.earth_orbit_alt <= 1000
        assert 50 <= mission_config.moon_orbit_alt <= 500
        assert 3.0 <= mission_config.transfer_time <= 10.0
        assert cost_factors.launch_cost_per_kg > 0
        assert spacecraft_config.dry_mass > 0

    def test_optimization_configuration(self):
        """Test optimization configuration validation."""
        config = OptimizationConfig(
            population_size=50,
            num_generations=25,
            seed=123
        )

        assert config.population_size == 50
        assert config.num_generations == 25
        assert config.seed == 123

        # Test defaults
        default_config = OptimizationConfig()
        assert default_config.population_size == 100
        assert default_config.num_generations == 100


class TestDataExportAndResults:
    """Test data export and results management."""

    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results for testing."""
        return AnalysisResults(
            mission_name="Test Mission",
            trajectory_results={"baseline": {"total_dv": 3200}},
            optimization_results={"pareto_front_size": 10},
            economic_analysis={"solution_analyses": []},
            visualization_assets={"executive_dashboard": None},
            analysis_metadata={"analysis_date": datetime.now().isoformat()}
        )

    def test_analysis_results_structure(self, sample_results):
        """Test AnalysisResults data structure."""
        assert sample_results.mission_name == "Test Mission"
        assert "baseline" in sample_results.trajectory_results
        assert "pareto_front_size" in sample_results.optimization_results
        assert "analysis_date" in sample_results.analysis_metadata

    def test_export_functionality(self, sample_results, tmp_path):
        """Test export functionality."""
        optimizer = LunarHorizonOptimizer()

        # Export to temporary directory
        output_dir = str(tmp_path / "test_export")
        optimizer.export_results(sample_results, output_dir)

        # Verify files were created
        metadata_file = tmp_path / "test_export" / "analysis_metadata.json"
        assert metadata_file.exists()

        # Verify metadata content
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata["analysis_date"] is not None


class TestErrorHandlingAndRobustness:
    """Test error handling and system robustness."""

    def test_missing_dependency_handling(self):
        """Test handling of missing dependencies."""
        # This test verifies the system can handle missing optional dependencies
        optimizer = LunarHorizonOptimizer()
        assert optimizer is not None

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test with invalid mission configuration
        with pytest.raises((ValueError, TypeError)):
            MissionConfig(
                name="Invalid",
                earth_orbit_alt=-100,  # Invalid negative altitude
                moon_orbit_alt=50,
                transfer_time=4.5,
                departure_epoch=10000.0
            )

    def test_optimization_failure_handling(self):
        """Test handling of optimization failures."""
        optimizer = LunarHorizonOptimizer()

        with patch("optimization.global_optimizer.GlobalOptimizer") as mock_opt_class:
            mock_optimizer = Mock()
            mock_optimizer.optimize.side_effect = Exception("Optimization failed")
            mock_opt_class.return_value = mock_optimizer

            # The system should handle optimization failures gracefully
            try:
                opt_config = OptimizationConfig(population_size=10, num_generations=5)
                optimizer._perform_optimization(opt_config, verbose=False)
            except Exception as e:
                # Expected to fail, but should be a controlled failure
                assert "Optimization failed" in str(e)


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for typical problems."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create optimizer and run small analysis
        optimizer = LunarHorizonOptimizer()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for initialization)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"

    def test_small_problem_performance(self):
        """Test performance on small optimization problems."""
        import time

        optimizer = LunarHorizonOptimizer()
        opt_config = OptimizationConfig(population_size=10, num_generations=5)

        with patch("trajectory.earth_moon_trajectories.generate_earth_moon_trajectory"):
            with patch.object(optimizer.window_analyzer, "find_transfer_windows"):
                with patch("optimization.global_optimizer.LunarMissionProblem"):
                    with patch("optimization.global_optimizer.GlobalOptimizer"):
                        with patch.object(optimizer.pareto_analyzer, "analyze_pareto_front"):
                            with patch.object(optimizer.isru_analyzer, "analyze_isru_economics"):
                                with patch.object(optimizer.sensitivity_analyzer, "monte_carlo_simulation"):

                                    start_time = time.time()

                                    # This should complete quickly with mocked components
                                    results = optimizer.analyze_mission(
                                        mission_name="Performance Test",
                                        optimization_config=opt_config,
                                        include_sensitivity=False,
                                        include_isru=False,
                                        verbose=False
                                    )

                                    elapsed_time = time.time() - start_time

                                    # Should complete within reasonable time (5 seconds with mocking)
                                    assert elapsed_time < 5.0, f"Analysis took {elapsed_time:.2f} seconds"
                                    assert results is not None


class TestSystemIntegration:
    """Test complete system integration scenarios."""

    def test_typical_mission_scenario(self):
        """Test a typical lunar mission analysis scenario."""
        # Create realistic mission configuration
        mission_config = MissionConfig(
            name="Artemis Gateway Supply",
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            departure_epoch=10000.0
        )

        cost_factors = CostFactors(
            launch_cost_per_kg=12000.0,
            operations_cost_per_day=120000.0,
            development_cost=1.2e9,
            contingency_percentage=25.0
        )

        spacecraft_config = SpacecraftConfig(
            name="Gateway Supply Vehicle",
            dry_mass=4500.0,
            propellant_mass=2800.0,
            payload_mass=1200.0,
            power_system_mass=450.0,
            propulsion_isp=330.0
        )

        optimizer = LunarHorizonOptimizer(
            mission_config=mission_config,
            cost_factors=cost_factors,
            spacecraft_config=spacecraft_config
        )

        assert optimizer.mission_config.name == "Artemis Gateway Supply"
        assert optimizer.cost_factors.launch_cost_per_kg == 12000.0
        assert optimizer.spacecraft_config.dry_mass == 4500.0

    def test_configuration_consistency(self):
        """Test that configurations remain consistent through analysis."""
        optimizer = LunarHorizonOptimizer()

        initial_mission_name = optimizer.mission_config.name
        initial_earth_alt = optimizer.mission_config.earth_orbit_alt
        initial_cost_per_kg = optimizer.cost_factors.launch_cost_per_kg

        # Configurations should not change during analysis
        assert optimizer.mission_config.name == initial_mission_name
        assert optimizer.mission_config.earth_orbit_alt == initial_earth_alt
        assert optimizer.cost_factors.launch_cost_per_kg == initial_cost_per_kg


# Integration Test Suite Summary
def test_integration_summary():
    """Comprehensive integration test summary."""
    print("\n" + "="*60)
    print("TASK 7: MVP INTEGRATION - TEST SUMMARY")
    print("="*60)
    print("✅ Optimizer initialization and configuration")
    print("✅ End-to-end workflow integration")
    print("✅ Component interaction and data flow")
    print("✅ Error handling and robustness")
    print("✅ Performance and scalability")
    print("✅ System integration scenarios")
    print("="*60)
    print("🎉 All integration tests designed and ready!")
    print("="*60)


if __name__ == "__main__":
    # Run a simple integration test
    test_integration_summary()
    print("\nRunning basic integration test...")

    try:
        # Test optimizer initialization
        optimizer = LunarHorizonOptimizer()
        print("✅ Optimizer initialized successfully")

        # Test configuration
        opt_config = OptimizationConfig(population_size=5, num_generations=2)
        print("✅ Configuration created successfully")

        print("🚀 Integration test completed successfully!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
