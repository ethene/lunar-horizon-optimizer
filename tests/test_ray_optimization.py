"""
Test suite for Ray-based parallel optimization.

Tests cover:
- Ray worker initialization and cleanup
- Parallel fitness evaluation correctness
- Performance characteristics
- Fallback behavior when Ray unavailable
- Integration with existing GlobalOptimizer

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import pytest
import numpy as np
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config.costs import CostFactors
from optimization.global_optimizer import LunarMissionProblem, GlobalOptimizer

# Ray imports with graceful handling
try:
    import ray
    from optimization.ray_optimizer import (
        RayParallelOptimizer,
        FitnessWorker,
        create_ray_optimizer,
        RAY_AVAILABLE,
    )

    RAY_AVAILABLE_FOR_TESTS = RAY_AVAILABLE
except ImportError:
    RAY_AVAILABLE_FOR_TESTS = False
    RayParallelOptimizer = None
    FitnessWorker = None
    create_ray_optimizer = None


class TestRayAvailability:
    """Test Ray availability and graceful fallback."""

    def test_ray_import(self):
        """Test Ray import status."""
        if RAY_AVAILABLE_FOR_TESTS:
            assert ray is not None
            assert RayParallelOptimizer is not None
            assert FitnessWorker is not None
        else:
            pytest.skip("Ray not available for testing")

    def test_create_ray_optimizer_fallback(self):
        """Test fallback to regular optimizer when Ray unavailable."""
        with patch("optimization.ray_optimizer.RAY_AVAILABLE", False):
            optimizer = create_ray_optimizer()
            assert isinstance(optimizer, GlobalOptimizer)
            assert not isinstance(optimizer, (RayParallelOptimizer, type(None)))


@pytest.mark.skipif(not RAY_AVAILABLE_FOR_TESTS, reason="Ray not available")
class TestFitnessWorker:
    """Test Ray fitness worker functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Initialize Ray for testing
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.cost_factors = {
            "launch_cost_per_kg": 10000.0,
            "operations_cost_per_day": 100000.0,
            "development_cost": 1e9,
        }

        self.worker_config = {
            "cost_factors": self.cost_factors,
            "min_earth_alt": 200,
            "max_earth_alt": 800,
            "min_moon_alt": 50,
            "max_moon_alt": 300,
            "min_transfer_time": 3.0,
            "max_transfer_time": 8.0,
            "reference_epoch": 10000.0,
            "worker_id": 0,
        }

    def teardown_method(self):
        """Cleanup after tests."""
        if ray.is_initialized():
            ray.shutdown()

    def test_worker_initialization(self):
        """Test fitness worker initialization."""
        worker = FitnessWorker.remote(**self.worker_config)

        # Test worker can be created and responds
        stats_future = worker.get_stats.remote()
        stats = ray.get(stats_future)

        assert stats["worker_id"] == 0
        assert stats["evaluations"] == 0
        assert stats["cache_size"] == 0

    def test_single_evaluation(self):
        """Test single fitness evaluation."""
        worker = FitnessWorker.remote(**self.worker_config)

        # Test single evaluation
        test_individual = [400.0, 100.0, 5.0]  # Earth alt, Moon alt, transfer time
        result_future = worker.evaluate_batch.remote([test_individual])
        results = ray.get(result_future)

        assert len(results) == 1
        fitness = results[0]
        assert len(fitness) == 3  # delta_v, time, cost
        assert all(f > 0 for f in fitness)  # All objectives should be positive
        assert all(f < 1e10 for f in fitness)  # Should not be penalty values

    def test_batch_evaluation(self):
        """Test batch fitness evaluation."""
        worker = FitnessWorker.remote(**self.worker_config)

        # Test batch evaluation
        test_batch = [[400.0, 100.0, 5.0], [600.0, 150.0, 6.0], [300.0, 80.0, 4.0]]

        result_future = worker.evaluate_batch.remote(test_batch)
        results = ray.get(result_future)

        assert len(results) == len(test_batch)
        for fitness in results:
            assert len(fitness) == 3
            assert all(f > 0 for f in fitness)
            assert all(f < 1e10 for f in fitness)

    def test_out_of_bounds_handling(self):
        """Test handling of out-of-bounds parameters."""
        worker = FitnessWorker.remote(**self.worker_config)

        # Test out-of-bounds individual
        out_of_bounds = [1500.0, 600.0, 15.0]  # All parameters out of bounds
        result_future = worker.evaluate_batch.remote([out_of_bounds])
        results = ray.get(result_future)

        assert len(results) == 1
        fitness = results[0]
        assert all(f >= 1e12 for f in fitness)  # Should return penalty values

    def test_worker_caching(self):
        """Test worker-level caching."""
        worker = FitnessWorker.remote(**self.worker_config)

        # Evaluate same individual twice
        test_individual = [400.0, 100.0, 5.0]

        # First evaluation
        result1_future = worker.evaluate_batch.remote([test_individual])
        result1 = ray.get(result1_future)

        # Second evaluation (should use cache)
        result2_future = worker.evaluate_batch.remote([test_individual])
        result2 = ray.get(result2_future)

        # Results should be identical
        assert result1 == result2

        # Check cache statistics
        stats_future = worker.get_stats.remote()
        stats = ray.get(stats_future)
        assert stats["cache_hits"] >= 1
        assert stats["cache_size"] >= 1


@pytest.mark.skipif(not RAY_AVAILABLE_FOR_TESTS, reason="Ray not available")
class TestRayParallelOptimizer:
    """Test Ray parallel optimizer functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
        )

        self.problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=200,
            max_earth_alt=600,
            min_moon_alt=50,
            max_moon_alt=200,
            min_transfer_time=3.0,
            max_transfer_time=7.0,
        )

    def teardown_method(self):
        """Cleanup after tests."""
        if ray.is_initialized():
            ray.shutdown()

    def test_optimizer_initialization(self):
        """Test Ray optimizer initialization."""
        optimizer = RayParallelOptimizer(
            problem=self.problem, population_size=20, num_generations=5, num_workers=2
        )

        assert optimizer.use_ray
        assert optimizer.num_workers == 2
        assert optimizer.population_size == 20
        assert optimizer.num_generations == 5

    def test_small_optimization_run(self):
        """Test small optimization run for correctness."""
        optimizer = RayParallelOptimizer(
            problem=self.problem,
            population_size=10,
            num_generations=3,
            num_workers=2,
            ray_config={"ignore_reinit_error": True},
        )

        start_time = time.time()
        results = optimizer.optimize(verbose=False)
        runtime = time.time() - start_time

        # Check basic results structure
        assert results["success"]
        assert len(results["pareto_front"]) > 0
        assert len(results["pareto_solutions"]) > 0
        assert results["generations"] == 3
        assert results["population_size"] == 10

        # Check Ray-specific results
        ray_stats = results.get("ray_stats", {})
        assert ray_stats["ray_used"]
        assert ray_stats["num_workers"] == 2
        assert (
            len(ray_stats["worker_stats"]) <= 2
        )  # May be fewer if workers not all used

        # Check performance is reasonable
        assert runtime < 60  # Should complete within a minute

        print(f"Small Ray optimization completed in {runtime:.2f}s")

    def test_worker_statistics(self):
        """Test collection of worker statistics."""
        optimizer = RayParallelOptimizer(
            problem=self.problem,
            population_size=8,
            num_generations=2,
            num_workers=2,
            ray_config={"ignore_reinit_error": True},
        )

        results = optimizer.optimize(verbose=False)

        ray_stats = results["ray_stats"]
        worker_stats = ray_stats["worker_stats"]

        # Check worker statistics
        assert len(worker_stats) <= 2
        for stats in worker_stats:
            assert "worker_id" in stats
            assert "evaluations" in stats
            assert "total_time" in stats
            assert "cache_hit_rate" in stats
            assert stats["evaluations"] >= 0
            assert stats["total_time"] >= 0

    def test_performance_vs_sequential(self):
        """Test performance comparison with sequential optimizer."""
        # Small test to avoid long runtime
        population_size = 20
        generations = 3

        # Sequential optimizer
        sequential_optimizer = GlobalOptimizer(
            problem=self.problem,
            population_size=population_size,
            num_generations=generations,
            seed=42,
        )

        start_time = time.time()
        sequential_results = sequential_optimizer.optimize(verbose=False)
        sequential_time = time.time() - start_time

        # Ray optimizer
        ray_optimizer = RayParallelOptimizer(
            problem=self.problem,
            population_size=population_size,
            num_generations=generations,
            num_workers=2,
            seed=42,
            ray_config={"ignore_reinit_error": True},
        )

        start_time = time.time()
        ray_results = ray_optimizer.optimize(verbose=False)
        ray_time = time.time() - start_time

        # Check both produce valid results
        assert sequential_results["success"]
        assert ray_results["success"]
        assert len(sequential_results["pareto_front"]) > 0
        assert len(ray_results["pareto_front"]) > 0

        # Performance comparison (Ray may not be faster for small problems)
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Ray time: {ray_time:.2f}s")

        # Both should complete in reasonable time
        assert sequential_time < 60
        assert ray_time < 60


class TestRayIntegration:
    """Test integration of Ray optimization with existing code."""

    def test_create_ray_optimizer_with_config(self):
        """Test creating Ray optimizer with configuration."""
        problem_config = {
            "min_earth_alt": 200,
            "max_earth_alt": 500,
            "min_moon_alt": 50,
            "max_moon_alt": 150,
        }

        optimizer_config = {
            "population_size": 10,
            "num_generations": 3,
            "num_workers": 2,
        }

        ray_config = {"ignore_reinit_error": True}

        optimizer = create_ray_optimizer(
            problem_config=problem_config,
            optimizer_config=optimizer_config,
            ray_config=ray_config,
        )

        # Should create appropriate optimizer type
        if RAY_AVAILABLE_FOR_TESTS:
            assert isinstance(optimizer, RayParallelOptimizer)
            assert optimizer.population_size == 10
            assert optimizer.num_generations == 3
        else:
            assert isinstance(optimizer, GlobalOptimizer)

    def test_optimization_correctness(self):
        """Test that Ray optimization produces correct results."""
        if not RAY_AVAILABLE_FOR_TESTS:
            pytest.skip("Ray not available")

        # Create a simple test problem
        problem = LunarMissionProblem(
            min_earth_alt=300,
            max_earth_alt=500,
            min_moon_alt=80,
            max_moon_alt=120,
            min_transfer_time=4.0,
            max_transfer_time=6.0,
        )

        # Test with very small population for speed
        optimizer = RayParallelOptimizer(
            problem=problem,
            population_size=6,
            num_generations=2,
            num_workers=2,
            ray_config={"ignore_reinit_error": True},
        )

        results = optimizer.optimize(verbose=False)

        # Check results make sense
        assert results["success"]
        pareto_solutions = results["pareto_solutions"]
        assert len(pareto_solutions) > 0

        # Check solution parameters are within bounds
        for solution in pareto_solutions:
            params = solution["parameters"]
            assert 300 <= params["earth_orbit_alt"] <= 500
            assert 80 <= params["moon_orbit_alt"] <= 120
            assert 4.0 <= params["transfer_time"] <= 6.0

            # Check objectives are reasonable
            objectives = solution["objectives"]
            assert objectives["delta_v"] > 0
            assert objectives["time"] > 0
            assert objectives["cost"] > 0
            assert objectives["delta_v"] < 50000  # Reasonable delta-v limit

    @pytest.mark.skipif(not RAY_AVAILABLE_FOR_TESTS, reason="Ray not available")
    def test_ray_cleanup(self):
        """Test proper Ray cleanup after optimization."""
        optimizer = RayParallelOptimizer(
            population_size=4,
            num_generations=1,
            num_workers=2,
            ray_config={"ignore_reinit_error": True},
        )

        # Should initialize Ray workers
        optimizer._initialize_ray_workers()
        assert len(optimizer.workers) == 2

        # Should clean up workers
        optimizer._shutdown_ray_workers()
        assert len(optimizer.workers) == 0
        assert len(optimizer.worker_stats) == 2
