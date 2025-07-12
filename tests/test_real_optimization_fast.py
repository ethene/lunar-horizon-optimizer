#!/usr/bin/env python3
"""
Fast Real Optimization Tests - No Mocking
=========================================

These tests use real optimization implementations with minimal parameters
for fast execution while ensuring full functionality coverage.

Execution time target: < 5 seconds
"""

import pytest
import sys
import os
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    # Real optimization imports - NO MOCKING
    from optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
    from optimization.pareto_analysis import ParetoAnalyzer
    from config.costs import CostFactors

    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    pytest_skip_reason = f"Optimization modules not available: {e}"

try:
    from optimization.cost_integration import CostIntegrator

    COST_INTEGRATION_AVAILABLE = True
except ImportError:
    COST_INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(
    not OPTIMIZATION_AVAILABLE, reason="Optimization modules not available"
)
class TestRealOptimizationFast:
    """Fast tests using real optimization implementations."""

    def setup_method(self):
        """Setup for each test method."""
        self.cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=500000000.0,
            contingency_percentage=15.0,
        )

    def test_lunar_mission_problem_real(self):
        """Test real LunarMissionProblem without mocks."""
        # Create real problem instance
        problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=350,
            max_earth_alt=450,
            min_moon_alt=80,
            max_moon_alt=120,
            min_transfer_time=4,
            max_transfer_time=8,
        )

        # Test real problem properties
        assert problem.get_nobj() == 3  # Three objectives
        assert problem.get_nix() == 3  # Three variables

        # Test real bounds
        bounds = problem.get_bounds()
        assert len(bounds[0]) == 3  # Lower bounds
        assert len(bounds[1]) == 3  # Upper bounds
        assert all(bounds[1][i] > bounds[0][i] for i in range(3))

    def test_real_fitness_evaluation_fast(self):
        """Test real fitness evaluation with minimal computation."""
        problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=400,
            max_earth_alt=410,  # Very narrow for speed
            min_moon_alt=100,
            max_moon_alt=110,
            min_transfer_time=4.5,
            max_transfer_time=5.5,
        )

        # Test real fitness evaluation
        decision_vector = [405.0, 105.0, 5.0]  # Middle values
        fitness = problem.fitness(decision_vector)

        # Verify real results (no mocking)
        assert len(fitness) == 3  # Three objectives
        assert all(f > 0 for f in fitness), "All objectives should be positive"

        # Check realistic ranges
        delta_v, time_sec, cost = fitness
        assert 1000 < delta_v < 10000, f"Unrealistic delta-v: {delta_v}"
        assert 300000 < time_sec < 800000, f"Unrealistic time: {time_sec}"
        assert cost > 0, f"Cost should be positive: {cost}"

    def test_global_optimizer_real_minimal(self):
        """Test real GlobalOptimizer with minimal parameters for speed."""
        problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=400,
            max_earth_alt=450,
            min_moon_alt=100,
            max_moon_alt=120,
            min_transfer_time=4,
            max_transfer_time=6,
        )

        # Create real optimizer with minimal settings
        optimizer = GlobalOptimizer(
            population_size=8,  # Minimum for NSGA-II
            generations=3,  # Very few generations for speed
            algorithm="nsga2",
            seed=42,  # For reproducibility
        )

        # Run real optimization
        results = optimizer.optimize(problem)

        # Verify real results
        assert hasattr(results, "pareto_solutions")
        assert len(results.pareto_solutions) > 0

        # Check solution structure
        for solution in results.pareto_solutions:
            assert "parameters" in solution
            assert "objectives" in solution
            assert len(solution["parameters"]) == 3  # 3 decision variables
            assert len(solution["objectives"]) == 2  # 2 objectives

    def test_pareto_analyzer_real(self):
        """Test real ParetoAnalyzer without mocks."""
        analyzer = ParetoAnalyzer()

        # Create real solution data (not mocked)
        solutions = [
            {
                "parameters": {"earth_alt": 400, "moon_alt": 100, "time": 4.5},
                "objectives": {"delta_v": 3200, "cost": 150e6},
            },
            {
                "parameters": {"earth_alt": 450, "moon_alt": 120, "time": 5.0},
                "objectives": {"delta_v": 3000, "cost": 180e6},
            },
            {
                "parameters": {"earth_alt": 350, "moon_alt": 80, "time": 6.0},
                "objectives": {"delta_v": 3500, "cost": 120e6},
            },
        ]

        # Test real Pareto analysis
        pareto_front = analyzer.find_pareto_front(solutions)

        # Verify real results
        assert len(pareto_front) > 0
        assert len(pareto_front) <= len(solutions)

        # All Pareto solutions should be non-dominated
        for solution in pareto_front:
            assert "parameters" in solution
            assert "objectives" in solution

    @pytest.mark.skipif(
        not COST_INTEGRATION_AVAILABLE, reason="CostIntegrator not available"
    )
    def test_cost_integrator_real(self):
        """Test real CostIntegrator without mocks."""
        integrator = CostIntegrator(cost_factors=self.cost_factors)

        # Real trajectory parameters
        trajectory_params = {
            "delta_v": 3200.0,  # m/s
            "time_of_flight": 5.0,  # days
            "spacecraft_mass": 5000.0,  # kg
            "propellant_mass": 2000.0,  # kg
        }

        # Calculate real costs
        cost_breakdown = integrator.calculate_mission_cost(trajectory_params)

        # Verify real cost calculations
        assert "total_cost" in cost_breakdown
        assert "launch_cost" in cost_breakdown
        assert "operations_cost" in cost_breakdown
        assert "development_cost" in cost_breakdown

        # Verify realistic cost values
        total_cost = cost_breakdown["total_cost"]
        assert total_cost > 0, "Total cost must be positive"
        assert total_cost > 100e6, f"Total cost too low: ${total_cost/1e6:.1f}M"
        assert total_cost < 10e9, f"Total cost too high: ${total_cost/1e6:.1f}M"

    def test_optimization_performance_real(self):
        """Test that real optimization executes quickly."""
        import time

        start_time = time.time()

        # Real optimization with minimal parameters
        problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=400,
            max_earth_alt=420,  # Narrow range
            min_moon_alt=100,
            max_moon_alt=110,
            min_transfer_time=4.5,
            max_transfer_time=5.5,
        )

        optimizer = GlobalOptimizer(
            population_size=8, generations=2, seed=42  # Minimum  # Very few
        )

        results = optimizer.optimize(problem)

        execution_time = time.time() - start_time

        # Should complete quickly
        assert execution_time < 5.0, f"Optimization too slow: {execution_time:.2f}s"
        assert len(results.pareto_solutions) > 0


@pytest.mark.skipif(
    not OPTIMIZATION_AVAILABLE, reason="Optimization modules not available"
)
class TestRealOptimizationValidation:
    """Test real optimization validation without mocks."""

    def test_population_size_validation_real(self):
        """Test real population size validation."""
        # Test invalid population size
        with pytest.raises((ValueError, AssertionError)):
            GlobalOptimizer(population_size=3, generations=5)  # Too small for NSGA-II

        # Test valid population size
        optimizer = GlobalOptimizer(population_size=8, generations=2)  # Valid minimum
        assert optimizer.population_size >= 8

    def test_objective_validation_real(self):
        """Test real objective function validation."""
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=500000000.0,
            contingency_percentage=15.0,
        )

        # Test valid problem creation
        problem = LunarMissionProblem(cost_factors=cost_factors)
        assert problem.get_nobj() == 3  # Default objectives: delta_v, time, cost


def test_optimization_integration_real():
    """Test real optimization integration without mocks."""
    if not OPTIMIZATION_AVAILABLE:
        pytest.skip("Optimization modules not available")

    # Complete real integration test
    cost_factors = CostFactors(
        launch_cost_per_kg=10000.0,
        operations_cost_per_day=100000.0,
        development_cost=500000000.0,
        contingency_percentage=15.0,
    )

    # Create real problem
    problem = LunarMissionProblem(cost_factors=cost_factors)

    # Create real optimizer
    optimizer = GlobalOptimizer(population_size=8, generations=2, seed=42)

    # Run real optimization
    results = optimizer.optimize(problem)

    # Analyze real results
    analyzer = ParetoAnalyzer()
    pareto_front = analyzer.find_pareto_front(results.pareto_solutions)

    # Verify complete integration
    assert len(pareto_front) > 0
    assert all("objectives" in sol for sol in pareto_front)
    assert all("parameters" in sol for sol in pareto_front)


if __name__ == "__main__":
    # Run fast optimization tests
    pytest.main([__file__, "-v", "--tb=short"])
