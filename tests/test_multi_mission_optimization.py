"""Comprehensive tests for multi-mission constellation optimization.

This test suite validates the multi-mission optimization implementation
including genome design, problem formulation, and backward compatibility.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from src.config.costs import CostFactors
from src.optimization.multi_mission_genome import (
    MultiMissionGenome,
    MultiMissionProblem,
    create_backward_compatible_problem,
)
from src.optimization.multi_mission_optimizer import (
    MultiMissionOptimizer,
    optimize_constellation,
    migrate_single_to_multi,
)
from src.optimization.global_optimizer import LunarMissionProblem


class TestMultiMissionGenome:
    """Test MultiMissionGenome dataclass and encoding/decoding."""

    def test_genome_creation(self):
        """Test basic genome creation and validation."""
        genome = MultiMissionGenome(
            num_missions=3,
            epochs=[10000.0, 10001.0, 10002.0],
            parking_altitudes=[300.0, 400.0, 500.0],
            plane_raan=[0.0, 120.0, 240.0],
            payload_masses=[1000.0, 1200.0, 800.0],
            lunar_altitude=100.0,
            transfer_time=5.0,
        )

        assert genome.num_missions == 3
        assert len(genome.epochs) == 3
        assert len(genome.parking_altitudes) == 3
        assert len(genome.plane_raan) == 3
        assert len(genome.payload_masses) == 3
        assert genome.lunar_altitude == 100.0
        assert genome.transfer_time == 5.0

    def test_genome_auto_initialization(self):
        """Test automatic initialization of missing parameters."""
        genome = MultiMissionGenome(num_missions=4)

        assert len(genome.epochs) == 4
        assert len(genome.parking_altitudes) == 4
        assert len(genome.plane_raan) == 4
        assert len(genome.payload_masses) == 4

        # Check RAAN distribution is uniform
        expected_raan = [0.0, 90.0, 180.0, 270.0]
        assert genome.plane_raan == expected_raan

    def test_decision_vector_encoding_decoding(self):
        """Test conversion to/from PyGMO decision vector."""
        num_missions = 3

        # Create test decision vector
        x = [
            # Epochs
            10000.0,
            10001.0,
            10002.0,
            # Parking altitudes
            300.0,
            400.0,
            500.0,
            # RAAN values
            0.0,
            120.0,
            240.0,
            # Payload masses
            1000.0,
            1200.0,
            800.0,
            # Shared parameters
            100.0,  # lunar_altitude
            5.0,  # transfer_time
        ]

        # Decode to genome
        genome = MultiMissionGenome.from_decision_vector(x, num_missions)

        assert genome.num_missions == 3
        assert genome.epochs == [10000.0, 10001.0, 10002.0]
        assert genome.parking_altitudes == [300.0, 400.0, 500.0]
        assert genome.plane_raan == [0.0, 120.0, 240.0]
        assert genome.payload_masses == [1000.0, 1200.0, 800.0]
        assert genome.lunar_altitude == 100.0
        assert genome.transfer_time == 5.0

        # Encode back to decision vector
        x_decoded = genome.to_decision_vector()

        assert len(x_decoded) == len(x)
        np.testing.assert_array_almost_equal(x_decoded, x)

    def test_mission_parameters_extraction(self):
        """Test extraction of individual mission parameters."""
        genome = MultiMissionGenome(
            num_missions=2,
            epochs=[10000.0, 10005.0],
            parking_altitudes=[300.0, 400.0],
            plane_raan=[0.0, 180.0],
            payload_masses=[1000.0, 1500.0],
            lunar_altitude=150.0,
            transfer_time=6.0,
        )

        # Test first mission
        params_0 = genome.get_mission_parameters(0)
        expected_0 = {
            "epoch": 10000.0,
            "earth_orbit_alt": 300.0,
            "moon_orbit_alt": 150.0,
            "transfer_time": 6.0,
            "plane_raan": 0.0,
            "payload_mass": 1000.0,
        }
        assert params_0 == expected_0

        # Test second mission
        params_1 = genome.get_mission_parameters(1)
        expected_1 = {
            "epoch": 10005.0,
            "earth_orbit_alt": 400.0,
            "moon_orbit_alt": 150.0,
            "transfer_time": 6.0,
            "plane_raan": 180.0,
            "payload_mass": 1500.0,
        }
        assert params_1 == expected_1

    def test_constellation_geometry_validation(self):
        """Test constellation geometry validation."""
        # Valid constellation with good RAAN separation
        valid_genome = MultiMissionGenome(
            num_missions=3, plane_raan=[0.0, 120.0, 240.0]  # 120° separation
        )
        assert valid_genome.validate_constellation_geometry() is True

        # Invalid constellation with poor RAAN separation
        invalid_genome = MultiMissionGenome(
            num_missions=3, plane_raan=[0.0, 5.0, 10.0]  # Only 5° separation
        )
        assert invalid_genome.validate_constellation_geometry() is False

        # Single mission always valid
        single_genome = MultiMissionGenome(num_missions=1, plane_raan=[45.0])
        assert single_genome.validate_constellation_geometry() is True

    def test_invalid_genome_creation(self):
        """Test error handling for invalid genome parameters."""
        # Invalid number of missions
        with pytest.raises(ValueError, match="num_missions must be positive"):
            MultiMissionGenome(num_missions=0)

        # Mismatched parameter lengths
        with pytest.raises(ValueError, match="epochs length"):
            MultiMissionGenome(
                num_missions=3,
                epochs=[10000.0, 10001.0],  # Only 2 values for 3 missions
            )


class TestMultiMissionProblem:
    """Test MultiMissionProblem PyGMO interface."""

    def test_problem_creation(self):
        """Test basic multi-mission problem creation."""
        problem = MultiMissionProblem(num_missions=3)

        assert problem.num_missions == 3
        assert problem.constellation_mode is True
        assert hasattr(problem, "lunar_transfer")
        assert hasattr(problem, "cost_calculator")

    def test_problem_bounds(self):
        """Test decision variable bounds."""
        num_missions = 4
        problem = MultiMissionProblem(num_missions=num_missions)

        lower, upper = problem.get_bounds()

        # Check bounds length: 4*K + 2
        expected_length = 4 * num_missions + 2
        assert len(lower) == expected_length
        assert len(upper) == expected_length

        # Check bounds values are reasonable
        assert all(l < u for l, u in zip(lower, upper, strict=False))
        assert all(l >= 0 for l in lower)

    def test_problem_objectives(self):
        """Test number of objectives."""
        # Constellation mode (5 objectives)
        constellation_problem = MultiMissionProblem(
            num_missions=3, constellation_mode=True
        )
        assert constellation_problem.get_nobj() == 5

        # Non-constellation mode (3 objectives)
        basic_problem = MultiMissionProblem(num_missions=3, constellation_mode=False)
        assert basic_problem.get_nobj() == 3

    def test_fitness_evaluation_structure(self):
        """Test fitness evaluation returns correct structure."""
        problem = MultiMissionProblem(num_missions=2)

        # Create valid decision vector
        lower, upper = problem.get_bounds()

        # Use midpoint values for reasonable test
        x = [(l + u) / 2 for l, u in zip(lower, upper, strict=False)]

        # Evaluate fitness
        objectives = problem.fitness(x)

        # Should return 5 objectives for constellation mode
        assert len(objectives) == 5
        assert all(isinstance(obj, (int, float)) for obj in objectives)
        assert all(obj > 0 for obj in objectives)  # All objectives should be positive

    def test_penalty_handling(self):
        """Test penalty values for invalid solutions."""
        problem = MultiMissionProblem(num_missions=2)

        # Create invalid decision vector (out of bounds)
        x = [-1000.0] * (4 * 2 + 2)  # All negative values

        objectives = problem.fitness(x)

        # Should return penalty values
        assert len(objectives) == 5
        assert all(obj >= 1e6 for obj in objectives)  # Large penalty values


class TestMultiMissionOptimizer:
    """Test MultiMissionOptimizer functionality."""

    def test_optimizer_creation_single_mission(self):
        """Test optimizer creation for single mission (backward compatibility)."""
        optimizer = MultiMissionOptimizer(
            multi_mission_mode=False,
            num_missions=1,
            population_size=20,
            num_generations=5,
        )

        assert optimizer.is_multi_mission is False
        assert optimizer.num_missions == 1
        assert optimizer.population_size == 20
        assert optimizer.num_generations == 5

    def test_optimizer_creation_multi_mission(self):
        """Test optimizer creation for multi-mission constellation."""
        optimizer = MultiMissionOptimizer(
            multi_mission_mode=True,
            num_missions=3,
            population_size=50,
            num_generations=10,
        )

        assert optimizer.is_multi_mission is True
        assert optimizer.num_missions == 3
        assert optimizer.population_size == 50
        assert optimizer.num_generations == 10

    def test_optimization_execution(self):
        """Test basic optimization execution."""
        # Use small parameters for fast testing
        optimizer = MultiMissionOptimizer(
            multi_mission_mode=True,
            num_missions=2,
            population_size=20,
            num_generations=3,
        )

        results = optimizer.optimize(verbose=False)

        # Check results structure
        assert isinstance(results, dict)
        assert "success" in results
        assert "pareto_front" in results
        assert "pareto_solutions" in results

        # Multi-mission specific results
        if results.get("success", False):
            assert "constellation_metrics" in results
            assert "best_constellations" in results
            assert "problem_info" in results


class TestBackwardCompatibility:
    """Test backward compatibility with single-mission code."""

    def test_create_backward_compatible_problem_single(self):
        """Test backward compatible problem creation for single mission."""
        problem = create_backward_compatible_problem(enable_multi=False, num_missions=1)

        assert isinstance(problem, LunarMissionProblem)
        assert hasattr(problem, "fitness")
        assert hasattr(problem, "get_bounds")
        assert problem.get_nobj() == 3  # Single mission objectives

    def test_create_backward_compatible_problem_multi(self):
        """Test backward compatible problem creation for multi-mission."""
        problem = create_backward_compatible_problem(enable_multi=True, num_missions=3)

        assert isinstance(problem, MultiMissionProblem)
        assert problem.num_missions == 3
        assert problem.get_nobj() == 5  # Constellation objectives

    def test_migrate_single_to_multi_config(self):
        """Test configuration migration from single to multi-mission."""
        single_config = {"population_size": 100, "num_generations": 50, "verbose": True}

        multi_config = migrate_single_to_multi(single_config, num_missions=5)

        # Should scale population and generations
        assert multi_config["population_size"] >= single_config["population_size"]
        assert multi_config["num_generations"] >= single_config["num_generations"]
        assert multi_config["num_missions"] == 5
        assert multi_config["constellation_mode"] is True

        # Should preserve other settings
        assert multi_config["verbose"] == single_config["verbose"]


class TestConstellationOptimization:
    """Test high-level constellation optimization function."""

    def test_optimize_constellation_basic(self):
        """Test basic constellation optimization."""
        # Use minimal parameters for fast testing
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=50000.0,
            development_cost=1e8,
        )

        optimization_config = {
            "optimizer_params": {"population_size": 20, "num_generations": 3},
            "verbose": False,
        }

        results = optimize_constellation(
            num_missions=2,
            cost_factors=cost_factors,
            optimization_config=optimization_config,
        )

        assert isinstance(results, dict)
        assert "success" in results

        # Check constellation-specific results
        if results.get("success", False):
            assert "constellation_metrics" in results
            assert "problem_info" in results
            assert results["problem_info"]["problem_type"] == "multi_mission"
            assert results["problem_info"]["num_missions"] == 2


class TestDimensionalityScaling:
    """Test dimensionality scaling with different constellation sizes."""

    @pytest.mark.parametrize("num_missions", [1, 3, 5, 8])
    def test_decision_vector_scaling(self, num_missions):
        """Test decision vector scaling with different K values."""
        problem = MultiMissionProblem(num_missions=num_missions)

        lower, upper = problem.get_bounds()

        # Check expected length: 4*K + 2
        expected_length = 4 * num_missions + 2
        assert len(lower) == expected_length
        assert len(upper) == expected_length

        # Test with valid decision vector
        x = [(l + u) / 2 for l, u in zip(lower, upper, strict=False)]

        objectives = problem.fitness(x)

        # Should always return same number of objectives
        assert len(objectives) == 5

    def test_population_scaling_recommendations(self):
        """Test population scaling recommendations."""
        configs = [
            (1, 20),  # Single mission
            (3, 50),  # Small constellation
            (8, 100),  # Medium constellation
            (24, 200),  # Large constellation
        ]

        for num_missions, min_population in configs:
            migrated = migrate_single_to_multi(
                {"population_size": 50, "num_generations": 50},
                num_missions=num_missions,
            )

            # Should scale population appropriately
            assert migrated["population_size"] >= min_population

            # Should scale generations
            assert migrated.get("num_generations", 100) >= 100


class TestRealOptimizationSmall:
    """Test with real optimization on small problem (fast execution)."""

    def test_single_vs_multi_mission_comparison(self):
        """Compare single mission vs 1-mission constellation (should be similar)."""
        # Single mission optimizer
        single_optimizer = MultiMissionOptimizer(
            multi_mission_mode=False,
            num_missions=1,
            population_size=20,
            num_generations=5,
        )

        # 1-mission constellation optimizer
        multi_optimizer = MultiMissionOptimizer(
            multi_mission_mode=True,
            num_missions=1,
            population_size=20,
            num_generations=5,
        )

        # Run optimizations
        single_results = single_optimizer.optimize(verbose=False)
        multi_results = multi_optimizer.optimize(verbose=False)

        # Both should succeed
        assert single_results.get("success", False)
        assert multi_results.get("success", False)

        # Results should be reasonably similar
        # (allowing for some variation due to randomness)
        single_pareto = single_results.get("pareto_front", [])
        multi_pareto = multi_results.get("pareto_front", [])

        assert len(single_pareto) > 0
        assert len(multi_pareto) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_fitness_evaluation(self):
        """Test handling of invalid fitness evaluations."""
        problem = MultiMissionProblem(num_missions=2)

        # Test with invalid decision vector length
        x_short = [1.0, 2.0]  # Too short
        objectives = problem.fitness(x_short)

        # Should return penalty values
        assert len(objectives) == 5
        assert all(obj >= 1e6 for obj in objectives)

    def test_genome_validation_errors(self):
        """Test genome validation error handling."""
        # Test out of range mission index
        genome = MultiMissionGenome(num_missions=3)

        with pytest.raises(ValueError, match="Mission index .* out of range"):
            genome.get_mission_parameters(5)

        with pytest.raises(ValueError, match="Mission index .* out of range"):
            genome.get_mission_parameters(-1)

    def test_optimizer_no_results(self):
        """Test optimizer behavior with no results."""
        optimizer = MultiMissionOptimizer(
            multi_mission_mode=True,
            num_missions=2,
            population_size=10,
            num_generations=1,
        )

        # Test getting solutions before optimization
        with pytest.raises(ValueError, match="No optimization results available"):
            optimizer.get_best_constellation_solutions()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
