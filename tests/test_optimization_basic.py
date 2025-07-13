"""
Basic unit tests for optimization modules to improve coverage.
"""

import pytest
import numpy as np

from src.optimization.global_optimizer import LunarMissionProblem, GlobalOptimizer
from src.optimization.pareto_analysis import ParetoAnalyzer
from src.optimization.cost_integration import CostCalculator, EconomicObjectives
from src.config.costs import CostFactors


class TestLunarMissionProblem:
    """Test LunarMissionProblem class."""

    def test_problem_creation(self):
        """Test creating a lunar mission problem."""
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
        )

        problem = LunarMissionProblem(
            cost_factors=cost_factors,
            min_earth_alt=200,
            max_earth_alt=600,
            min_moon_alt=50,
            max_moon_alt=200,
        )

        assert problem is not None
        assert problem.min_earth_alt == 200
        assert problem.max_earth_alt == 600

    def test_problem_bounds(self):
        """Test problem bounds."""
        problem = LunarMissionProblem()
        bounds = problem.get_bounds()

        assert len(bounds) == 2  # lower and upper bounds
        assert len(bounds[0]) == 3  # 3 variables: earth_alt, moon_alt, transfer_time
        assert len(bounds[1]) == 3


class TestGlobalOptimizer:
    """Test GlobalOptimizer class."""

    def test_global_optimizer_creation(self):
        """Test creating a global optimizer."""
        problem = LunarMissionProblem()
        optimizer = GlobalOptimizer(
            problem=problem, population_size=20, num_generations=5
        )

        assert optimizer is not None
        assert optimizer.population_size == 20
        assert optimizer.num_generations == 5

    def test_optimizer_configuration(self):
        """Test optimizer configuration validation."""
        problem = LunarMissionProblem()

        # Test valid configuration
        optimizer = GlobalOptimizer(
            problem=problem, population_size=30, num_generations=10
        )

        assert optimizer.population_size > 0
        assert optimizer.num_generations > 0


class TestParetoAnalyzer:
    """Test ParetoAnalyzer class."""

    def test_pareto_analyzer_creation(self):
        """Test creating a Pareto analyzer."""
        analyzer = ParetoAnalyzer()
        assert analyzer is not None

    def test_pareto_front_extraction(self):
        """Test Pareto front extraction."""
        analyzer = ParetoAnalyzer()

        # Mock population with fitness values
        mock_population = [
            {"fitness": [1.0, 5.0, 100.0]},  # Good delta-v, bad time, bad cost
            {"fitness": [3.0, 3.0, 80.0]},  # Medium all
            {"fitness": [5.0, 1.0, 120.0]},  # Bad delta-v, good time, bad cost
            {"fitness": [2.0, 4.0, 60.0]},  # Good cost
        ]

        # Basic test - should return some results
        if hasattr(analyzer, "extract_pareto_front"):
            front = analyzer.extract_pareto_front(mock_population)
            assert isinstance(front, list)


class TestCostCalculator:
    """Test CostCalculator class."""

    def test_cost_calculator_creation(self):
        """Test creating a cost calculator."""
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
        )
        calculator = CostCalculator(cost_factors)

        assert calculator is not None

    def test_mission_cost_calculation(self):
        """Test mission cost calculation."""
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
        )
        calculator = CostCalculator(cost_factors)

        # Test basic cost calculation
        cost = calculator.calculate_mission_cost(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
        )

        # Cost should be positive
        assert cost > 0
        # Should be reasonable order of magnitude (millions to billions)
        assert 1e6 < cost < 1e12


class TestEconomicObjectives:
    """Test EconomicObjectives class."""

    def test_economic_objectives_creation(self):
        """Test creating economic objectives."""
        objectives = EconomicObjectives(
            delta_v=3500.0, transfer_time=5.0, total_cost=1e9
        )

        assert objectives.delta_v == 3500.0
        assert objectives.transfer_time == 5.0
        assert objectives.total_cost == 1e9

    def test_objectives_to_list(self):
        """Test converting objectives to list format."""
        objectives = EconomicObjectives(
            delta_v=3200.0, transfer_time=6.0, total_cost=8.5e8
        )

        obj_list = objectives.to_list()

        assert len(obj_list) == 3
        assert obj_list[0] == 3200.0  # delta_v
        assert obj_list[1] == 6.0  # transfer_time
        assert obj_list[2] == 8.5e8  # total_cost


class TestOptimizationIntegration:
    """Test integration between optimization modules."""

    def test_problem_fitness_evaluation(self):
        """Test fitness evaluation of lunar mission problem."""
        problem = LunarMissionProblem()

        # Test valid parameter vector
        x = [400.0, 100.0, 5.0]  # earth_alt, moon_alt, transfer_time

        fitness = problem.fitness(x)

        # Should return 3 objectives
        assert len(fitness) == 3

        # All objectives should be positive
        assert all(f > 0 for f in fitness)

        # Should be reasonable values
        assert fitness[0] > 1000  # delta_v > 1000 m/s
        assert fitness[1] > 1000  # transfer_time > 1000 seconds (converted from days)
        assert fitness[2] > 1e6  # cost > $1M

    def test_out_of_bounds_handling(self):
        """Test handling of out-of-bounds parameters."""
        problem = LunarMissionProblem()

        # Test out-of-bounds parameters
        x_invalid = [1500.0, 600.0, 15.0]  # All out of bounds

        fitness = problem.fitness(x_invalid)

        # Should return penalty values or handle gracefully
        assert len(fitness) == 3
        # Either very large penalty values or reasonable fallback
        assert all(f > 0 for f in fitness)
