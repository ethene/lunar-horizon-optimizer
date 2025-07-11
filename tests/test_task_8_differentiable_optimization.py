"""
Test suite for Task 8: JAX Differentiable Optimization Module

This test suite validates the implementation of the JAX-based differentiable
optimization module including trajectory models, economic models, and the
complete optimization pipeline.

Features tested:
- JAX infrastructure and environment validation
- Differentiable trajectory models with automatic differentiation
- Differentiable economic models with JIT compilation
- Combined optimization objective functions
- Gradient-based optimization with scipy integration
- End-to-end optimization demonstrations
- Performance and accuracy validation
"""

import unittest
import time
import numpy as np
from typing import Dict, List, Any

# JAX imports (with fallback)
try:
    import jax.numpy as jnp
    from jax import grad, jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None

# Local imports
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.differentiable import (
    JAX_AVAILABLE as MODULE_JAX_AVAILABLE,
    validate_jax_environment,
    get_jax_device_info,
)

if JAX_AVAILABLE:
    from src.optimization.differentiable.differentiable_models import (
        TrajectoryModel,
        EconomicModel,
        create_combined_model,
    )
    from src.optimization.differentiable.jax_optimizer import (
        DifferentiableOptimizer,
        OptimizationResult,
    )
    from src.optimization.differentiable.demo_optimization import (
        OptimizationDemonstration,
        run_quick_demo,
    )
    from src.optimization.differentiable.result_comparison import (
        ResultComparator,
        ComparisonResult,
        ConvergenceAnalysis,
        SolutionQuality,
        ComparisonMetric,
        compare_single_run,
        analyze_optimization_convergence,
        rank_optimization_results,
        evaluate_solution_quality,
    )
    from src.optimization.differentiable.comparison_demo import (
        ComparisonDemonstration,
        run_comparison_demo,
    )


class TestJAXInfrastructure(unittest.TestCase):
    """Test JAX infrastructure and environment setup."""

    def test_jax_availability(self):
        """Test that JAX is available and properly configured."""
        self.assertTrue(JAX_AVAILABLE, "JAX should be available for testing")
        self.assertTrue(MODULE_JAX_AVAILABLE, "Module should detect JAX availability")

    def test_jax_environment_validation(self):
        """Test JAX environment validation."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        # Test environment validation
        is_valid = validate_jax_environment()
        self.assertTrue(is_valid, "JAX environment should be valid")

    def test_jax_device_info(self):
        """Test JAX device information retrieval."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        device_info = get_jax_device_info()
        self.assertIn("available", device_info)
        self.assertTrue(device_info["available"])
        self.assertIn("devices", device_info)
        self.assertIn("default_device", device_info)

    def test_basic_jax_operations(self):
        """Test basic JAX operations work correctly."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        # Test array creation and operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x**2)

        self.assertIsInstance(x, jnp.ndarray)
        self.assertAlmostEqual(float(y), 14.0, places=6)

        # Test gradient computation
        grad_fn = grad(lambda x: jnp.sum(x**2))
        grad_result = grad_fn(x)

        expected_grad = 2 * x
        np.testing.assert_array_almost_equal(grad_result, expected_grad, decimal=6)

    def test_performance_optimization_imports(self):
        """Test that performance optimization module imports correctly."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        from src.optimization.differentiable.performance_optimization import (
            PerformanceConfig,
            JITOptimizer,
            BatchOptimizer,
            MemoryOptimizer,
            PerformanceBenchmark,
        )

        # Test basic instantiation
        config = PerformanceConfig()
        self.assertTrue(config.enable_jit)
        self.assertTrue(config.enable_vectorization)

        jit_optimizer = JITOptimizer(config)
        self.assertEqual(jit_optimizer.config, config)

    def test_jit_compilation(self):
        """Test JIT compilation functionality."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        # Test JIT compilation
        @jit
        def square_sum(x):
            return jnp.sum(x**2)

        x = jnp.array([1.0, 2.0, 3.0])
        result = square_sum(x)

        self.assertAlmostEqual(float(result), 14.0, places=6)


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestTrajectoryModel(unittest.TestCase):
    """Test JAX-based differentiable trajectory models."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TrajectoryModel(use_jit=True)
        self.test_params = jnp.array(
            [
                6.778e6,  # Earth departure radius (400 km altitude)
                1.937e6,  # Lunar orbit radius (200 km altitude)
                4.5 * 24 * 3600,  # Time of flight (4.5 days)
            ]
        )

    def test_model_initialization(self):
        """Test trajectory model initialization."""
        # Test with JIT enabled
        model_jit = TrajectoryModel(use_jit=True)
        self.assertTrue(model_jit.use_jit)

        # Test with JIT disabled
        model_no_jit = TrajectoryModel(use_jit=False)
        self.assertFalse(model_no_jit.use_jit)

    def test_orbital_velocity_calculation(self):
        """Test orbital velocity calculations."""
        radius = 6.778e6  # 400 km altitude
        velocity = self.model.orbital_velocity(radius)

        # Expected orbital velocity at 400 km altitude (~7670 m/s)
        expected_velocity = 7670.0
        self.assertAlmostEqual(float(velocity), expected_velocity, delta=100.0)

    def test_orbital_energy_calculation(self):
        """Test orbital energy calculations."""
        radius = 6.778e6
        velocity = 7670.0
        energy = self.model.orbital_energy(radius, velocity)

        # Energy should be negative for bound orbits
        self.assertLess(float(energy), 0.0)

    def test_hohmann_transfer_calculation(self):
        """Test Hohmann transfer calculations."""
        r1 = 6.778e6  # 400 km altitude
        r2 = 6.878e6  # 500 km altitude

        delta_v_total, delta_v1, delta_v2 = self.model.hohmann_transfer(r1, r2)

        # All delta-v values should be positive
        self.assertGreater(float(delta_v_total), 0.0)
        self.assertGreater(float(delta_v1), 0.0)
        self.assertGreater(float(delta_v2), 0.0)

        # Total should equal sum of components
        self.assertAlmostEqual(
            float(delta_v_total), float(delta_v1) + float(delta_v2), places=6
        )

    def test_lambert_solver_simple(self):
        """Test simplified Lambert solver."""
        r1 = jnp.array([6.778e6, 0.0, 0.0])  # Initial position
        r2 = jnp.array([0.0, 6.878e6, 0.0])  # Final position
        tof = 3600.0  # 1 hour

        v1, v2 = self.model.lambert_solver_simple(r1, r2, tof)

        # Velocities should be reasonable magnitude
        v1_mag = float(jnp.linalg.norm(v1))
        v2_mag = float(jnp.linalg.norm(v2))

        self.assertGreater(v1_mag, 1000.0)  # > 1 km/s
        self.assertLess(v1_mag, 20000.0)  # < 20 km/s
        self.assertGreater(v2_mag, 1000.0)
        self.assertLess(v2_mag, 20000.0)

    def test_trajectory_cost_calculation(self):
        """Test complete trajectory cost calculation."""
        result = self.model._trajectory_cost(self.test_params)

        # Verify result structure
        self.assertIn("delta_v", result)
        self.assertIn("time_of_flight", result)
        self.assertIn("final_position", result)
        self.assertIn("final_velocity", result)
        self.assertIn("energy", result)

        # Verify reasonable values
        delta_v = float(result["delta_v"])
        self.assertGreater(delta_v, 1000.0)  # > 1 km/s
        self.assertLess(delta_v, 20000.0)  # < 20 km/s

        tof = float(result["time_of_flight"])
        self.assertAlmostEqual(tof, 4.5 * 24 * 3600, places=0)

    def test_evaluate_trajectory(self):
        """Test trajectory evaluation interface."""
        result = self.model.evaluate_trajectory(self.test_params)

        # Verify result structure
        required_keys = ["delta_v", "time", "energy", "total_cost"]
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)

    def test_gradient_computation(self):
        """Test automatic differentiation of trajectory model."""

        # Define a simple objective function
        def objective(params):
            result = self.model._trajectory_cost(params)
            return result["delta_v"]

        # Compute gradient
        grad_fn = grad(objective)
        gradient = grad_fn(self.test_params)

        # Gradient should be finite and have correct shape
        self.assertEqual(gradient.shape, self.test_params.shape)
        self.assertTrue(jnp.all(jnp.isfinite(gradient)))

    def test_jit_compilation_performance(self):
        """Test JIT compilation performance benefits."""
        # Compare JIT vs non-JIT performance
        model_jit = TrajectoryModel(use_jit=True)
        model_no_jit = TrajectoryModel(use_jit=False)

        # Warm up JIT
        _ = model_jit._trajectory_cost(self.test_params)

        # Time JIT version
        start_time = time.time()
        for _ in range(10):
            _ = model_jit._trajectory_cost(self.test_params)
        jit_time = time.time() - start_time

        # Time non-JIT version
        start_time = time.time()
        for _ in range(10):
            _ = model_no_jit._trajectory_cost(self.test_params)
        no_jit_time = time.time() - start_time

        # JIT should be faster (though this test may be noisy)
        # At minimum, JIT should not be more than 10x slower
        self.assertLess(jit_time, no_jit_time * 10.0)


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestEconomicModel(unittest.TestCase):
    """Test JAX-based differentiable economic models."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = EconomicModel(use_jit=True)
        self.test_params = jnp.array(
            [3200.0, 4.5 * 24 * 3600]  # Delta-v (m/s)  # Time of flight (seconds)
        )

    def test_model_initialization(self):
        """Test economic model initialization."""
        # Test with JIT enabled
        model_jit = EconomicModel(use_jit=True)
        self.assertTrue(model_jit.use_jit)

        # Test with JIT disabled
        model_no_jit = EconomicModel(use_jit=False)
        self.assertFalse(model_no_jit.use_jit)

    def test_launch_cost_model(self):
        """Test launch cost calculations."""
        delta_v = 3200.0
        payload_mass = 1000.0

        cost = self.model.launch_cost_model(delta_v, payload_mass)

        # Cost should be positive and reasonable
        self.assertGreater(float(cost), 1e6)  # > $1M
        self.assertLess(float(cost), 1e11)  # < $100B

    def test_operations_cost_model(self):
        """Test operations cost calculations."""
        tof = 4.5 * 24 * 3600  # 4.5 days
        daily_cost = 100000.0  # $100k/day

        cost = self.model.operations_cost_model(tof, daily_cost)

        # Should be approximately 4.5 * daily_cost
        expected_cost = 4.5 * daily_cost
        self.assertAlmostEqual(float(cost), expected_cost, delta=10000.0)

    def test_npv_calculation(self):
        """Test Net Present Value calculations."""
        # Simple cash flow: -$100M initial, +$50M for 5 years
        cash_flows = jnp.array([-100e6, 50e6, 50e6, 50e6, 50e6, 50e6])
        discount_rate = 0.1

        npv = self.model.npv_calculation(cash_flows, discount_rate)

        # NPV should be positive for this profitable project
        self.assertGreater(float(npv), 0.0)

    def test_roi_calculation(self):
        """Test Return on Investment calculations."""
        total_cost = 100e6  # $100M
        annual_revenue = 50e6  # $50M/year

        roi = self.model.roi_calculation(total_cost, annual_revenue)

        # ROI should be approximately 0.5 (50M / 100M)
        self.assertAlmostEqual(float(roi), 0.5, delta=0.01)

    def test_economic_cost_calculation(self):
        """Test complete economic cost calculation."""
        result = self.model._economic_cost(self.test_params)

        # Verify result structure
        required_keys = ["total_cost", "launch_cost", "operations_cost", "npv", "roi"]
        for key in required_keys:
            self.assertIn(key, result)

        # Verify reasonable values
        total_cost = float(result["total_cost"])
        launch_cost = float(result["launch_cost"])
        ops_cost = float(result["operations_cost"])

        self.assertGreater(total_cost, 1e6)  # > $1M
        self.assertLess(total_cost, 1e11)  # < $100B
        self.assertAlmostEqual(total_cost, launch_cost + ops_cost, delta=1000.0)

    def test_evaluate_economics(self):
        """Test economics evaluation interface."""
        result = self.model.evaluate_economics(self.test_params)

        # Verify result structure
        required_keys = ["total_cost", "launch_cost", "operations_cost", "npv", "roi"]
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)

    def test_gradient_computation(self):
        """Test automatic differentiation of economic model."""

        # Define a simple objective function
        def objective(params):
            result = self.model._economic_cost(params)
            return result["total_cost"]

        # Compute gradient
        grad_fn = grad(objective)
        gradient = grad_fn(self.test_params)

        # Gradient should be finite and have correct shape
        self.assertEqual(gradient.shape, self.test_params.shape)
        self.assertTrue(jnp.all(jnp.isfinite(gradient)))


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestCombinedModel(unittest.TestCase):
    """Test combined trajectory-economic optimization models."""

    def setUp(self):
        """Set up test fixtures."""
        self.trajectory_model = TrajectoryModel(use_jit=True)
        self.economic_model = EconomicModel(use_jit=True)
        self.combined_model = create_combined_model(
            self.trajectory_model,
            self.economic_model,
            weights={"delta_v": 0.4, "cost": 0.4, "time": 0.2},
        )
        self.test_params = jnp.array(
            [
                6.778e6,  # Earth departure radius
                1.937e6,  # Lunar orbit radius
                4.5 * 24 * 3600,  # Time of flight
            ]
        )

    def test_combined_model_creation(self):
        """Test combined model creation with different weights."""
        # Test default weights
        model_default = create_combined_model(
            self.trajectory_model, self.economic_model
        )
        self.assertIsNotNone(model_default)

        # Test custom weights
        custom_weights = {"delta_v": 1.0, "cost": 2.0, "time": 0.5}
        model_custom = create_combined_model(
            self.trajectory_model, self.economic_model, weights=custom_weights
        )
        self.assertIsNotNone(model_custom)

    def test_combined_model_evaluation(self):
        """Test combined model evaluation."""
        result = self.combined_model(self.test_params)

        # Result should be a scalar
        self.assertIsInstance(result, jnp.ndarray)
        self.assertEqual(result.shape, ())
        self.assertTrue(jnp.isfinite(result))
        self.assertGreater(float(result), 0.0)

    def test_combined_model_gradient(self):
        """Test gradient computation of combined model."""
        grad_fn = grad(self.combined_model)
        gradient = grad_fn(self.test_params)

        # Gradient should be finite and have correct shape
        self.assertEqual(gradient.shape, self.test_params.shape)
        self.assertTrue(jnp.all(jnp.isfinite(gradient)))

    def test_weight_sensitivity(self):
        """Test sensitivity to different weight combinations."""
        base_result = float(self.combined_model(self.test_params))

        # Test with different weight emphasis
        cost_heavy_model = create_combined_model(
            self.trajectory_model,
            self.economic_model,
            weights={"delta_v": 0.1, "cost": 0.8, "time": 0.1},
        )
        cost_heavy_result = float(cost_heavy_model(self.test_params))

        time_heavy_model = create_combined_model(
            self.trajectory_model,
            self.economic_model,
            weights={"delta_v": 0.1, "cost": 0.1, "time": 0.8},
        )
        time_heavy_result = float(time_heavy_model(self.test_params))

        # Results should be different with different weight combinations
        self.assertNotAlmostEqual(base_result, cost_heavy_result, places=6)
        self.assertNotAlmostEqual(base_result, time_heavy_result, places=6)


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestDifferentiableOptimizer(unittest.TestCase):
    """Test JAX-based differentiable optimizer."""

    def setUp(self):
        """Set up test fixtures."""

        # Simple quadratic objective for testing
        def quadratic_objective(x):
            return jnp.sum((x - jnp.array([1.0, 2.0, 3.0])) ** 2)

        self.simple_optimizer = DifferentiableOptimizer(
            objective_function=quadratic_objective,
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            method="L-BFGS-B",
            use_jit=True,
            tolerance=1e-6,
            max_iterations=100,
            verbose=False,
        )

        # Combined trajectory-economic optimizer
        trajectory_model = TrajectoryModel(use_jit=True)
        economic_model = EconomicModel(use_jit=True)
        combined_model = create_combined_model(trajectory_model, economic_model)

        self.trajectory_optimizer = DifferentiableOptimizer(
            objective_function=combined_model,
            bounds=[
                (6.578e6, 6.978e6),  # Earth departure radius
                (1.837e6, 2.137e6),  # Lunar orbit radius
                (3.0 * 24 * 3600, 10.0 * 24 * 3600),  # Time of flight
            ],
            method="L-BFGS-B",
            use_jit=True,
            tolerance=1e-6,
            max_iterations=50,
            verbose=False,
        )

    def test_optimizer_initialization(self):
        """Test optimizer initialization with different configurations."""
        # Test with different methods
        methods = ["L-BFGS-B", "SLSQP"]
        for method in methods:
            optimizer = DifferentiableOptimizer(
                objective_function=lambda x: jnp.sum(x**2), method=method, use_jit=True
            )
            self.assertEqual(optimizer.method, method)

    def test_simple_optimization(self):
        """Test optimization of simple quadratic function."""
        # Initial guess
        x0 = jnp.array([0.0, 0.0, 0.0])

        # Run optimization
        result = self.simple_optimizer.optimize(x0)

        # Verify optimization result
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)

        # Solution should be close to [1, 2, 3]
        expected_solution = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.x, expected_solution, decimal=3)

        # Final objective should be near zero
        self.assertLess(result.fun, 1e-6)

    def test_trajectory_optimization(self):
        """Test optimization of trajectory-economic problem."""
        # Suboptimal initial guess
        x0 = jnp.array(
            [
                6.878e6,  # 500 km altitude (suboptimal)
                2.037e6,  # 300 km lunar altitude (suboptimal)
                6.0 * 24 * 3600,  # 6 days (suboptimal)
            ]
        )

        # Run optimization
        result = self.trajectory_optimizer.optimize(x0)

        # Verify optimization result
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)

        # Verify optimization improved the objective
        initial_obj = result.initial_objective
        final_obj = result.fun
        self.assertIsNotNone(initial_obj)
        self.assertLessEqual(final_obj, initial_obj)

        # Verify parameters are within bounds
        bounds = self.trajectory_optimizer.bounds
        for _i, (param, (lower, upper)) in enumerate(
            zip(result.x, bounds, strict=False)
        ):
            self.assertGreaterEqual(param, lower)
            self.assertLessEqual(param, upper)

    def test_optimization_result_structure(self):
        """Test optimization result structure and metrics."""
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = self.simple_optimizer.optimize(x0)

        # Verify required attributes
        required_attrs = [
            "x",
            "fun",
            "success",
            "message",
            "nit",
            "nfev",
            "njev",
            "optimization_time",
            "convergence_history",
            "gradient_norms",
            "objective_components",
            "constraint_violations",
            "initial_objective",
            "improvement_percentage",
        ]

        for attr in required_attrs:
            self.assertTrue(hasattr(result, attr))

        # Verify types
        self.assertIsInstance(result.x, np.ndarray)
        self.assertIsInstance(result.fun, float)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.optimization_time, float)
        self.assertIsInstance(result.convergence_history, list)

    def test_batch_optimization(self):
        """Test batch optimization with multiple initial points."""
        # Multiple initial guesses
        x0_batch = [
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([2.0, 2.0, 2.0]),
            jnp.array([-1.0, 1.0, 4.0]),
        ]

        # Run batch optimization
        results = self.simple_optimizer.batch_optimize(x0_batch)

        # Verify results
        self.assertEqual(len(results), len(x0_batch))
        for result in results:
            self.assertIsInstance(result, OptimizationResult)
            self.assertTrue(result.success)

    def test_optimization_comparison(self):
        """Test optimization result comparison analysis."""
        # Multiple initial guesses
        x0_batch = [jnp.array([0.0, 0.0, 0.0]), jnp.array([2.0, 2.0, 2.0])]

        # Run batch optimization
        results = self.simple_optimizer.batch_optimize(x0_batch)

        # Analyze results
        comparison = self.simple_optimizer.compare_with_initial(results)

        # Verify comparison structure
        required_keys = [
            "total_candidates",
            "successful_optimizations",
            "success_rate",
            "average_improvement_percentage",
            "best_improvement_percentage",
            "average_final_objective",
            "best_final_objective",
            "average_optimization_time",
            "total_function_evaluations",
        ]

        for key in required_keys:
            self.assertIn(key, comparison)

        # Verify reasonable values
        self.assertEqual(comparison["total_candidates"], len(x0_batch))
        self.assertGreaterEqual(comparison["success_rate"], 0.0)
        self.assertLessEqual(comparison["success_rate"], 1.0)


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestOptimizationDemonstration(unittest.TestCase):
    """Test complete optimization demonstration."""

    def test_demonstration_initialization(self):
        """Test optimization demonstration initialization."""
        demo = OptimizationDemonstration(use_jit=True, verbose=False)

        # Verify components are initialized
        self.assertIsNotNone(demo.trajectory_model)
        self.assertIsNotNone(demo.economic_model)
        self.assertIsNotNone(demo.combined_model)
        self.assertIsNotNone(demo.optimizer)

    def test_initial_guess_generation(self):
        """Test initial guess generation."""
        demo = OptimizationDemonstration(use_jit=True, verbose=False)

        # Test default guess
        initial_params_default = demo.generate_initial_guess(suboptimal=False)
        self.assertEqual(initial_params_default.shape, (3,))

        # Test suboptimal guess
        initial_params_suboptimal = demo.generate_initial_guess(suboptimal=True)
        self.assertEqual(initial_params_suboptimal.shape, (3,))

        # Suboptimal should be different from default
        self.assertFalse(
            jnp.allclose(initial_params_default, initial_params_suboptimal)
        )

    def test_solution_evaluation(self):
        """Test solution evaluation functionality."""
        demo = OptimizationDemonstration(use_jit=True, verbose=False)
        params = demo.generate_initial_guess(suboptimal=False)

        # Test initial solution evaluation
        initial_results = demo.evaluate_initial_solution(params)

        # Verify result structure
        required_keys = [
            "combined_objective",
            "delta_v",
            "time_of_flight_days",
            "total_cost_millions",
            "npv_millions",
            "roi",
        ]
        for key in required_keys:
            self.assertIn(key, initial_results)
            self.assertIsInstance(initial_results[key], float)

        # Test optimized solution evaluation
        optimized_results = demo.evaluate_optimized_solution(params)

        # Should have same structure plus physical parameters
        for key in required_keys:
            self.assertIn(key, optimized_results)

        self.assertIn("earth_altitude_km", optimized_results)
        self.assertIn("lunar_altitude_km", optimized_results)

    def test_complete_demonstration(self):
        """Test complete optimization demonstration workflow."""
        demo = OptimizationDemonstration(use_jit=True, verbose=False)

        # Run complete demonstration
        results = demo.run_complete_demonstration()

        # Verify result structure
        required_keys = [
            "initial_params",
            "initial_results",
            "optimization_results",
            "optimized_results",
            "comparison",
            "demonstration_success",
        ]
        for key in required_keys:
            self.assertIn(key, results)

        # Verify demonstration success
        self.assertIsInstance(results["demonstration_success"], bool)

        # Verify optimization was attempted
        opt_result = results["optimization_results"]["optimization_result"]
        self.assertIsInstance(opt_result, OptimizationResult)

    def test_solution_comparison(self):
        """Test solution comparison functionality."""
        demo = OptimizationDemonstration(use_jit=True, verbose=False)

        # Create mock results for comparison
        initial_results = {
            "combined_objective": 0.5,
            "delta_v": 3000.0,
            "total_cost_millions": 30.0,
        }

        optimized_results = {
            "combined_objective": 0.4,
            "delta_v": 2800.0,
            "total_cost_millions": 28.0,
        }

        # Compare solutions
        comparison = demo.compare_solutions(initial_results, optimized_results)

        # Verify comparison structure
        self.assertIn("combined_objective_improvement_pct", comparison)
        self.assertIn("delta_v_improvement_pct", comparison)
        self.assertIn("total_cost_millions_improvement_pct", comparison)

        # Verify improvement calculations
        obj_improvement = comparison["combined_objective_improvement_pct"]
        self.assertGreater(obj_improvement, 0.0)  # Should show improvement

    def test_quick_demo_function(self):
        """Test quick demonstration function."""
        # Run quick demo (should complete without errors)
        success = run_quick_demo()

        # Should return boolean success status
        self.assertIsInstance(success, bool)


class TestIntegrationAndPerformance(unittest.TestCase):
    """Test integration and performance aspects."""

    def setUp(self):
        """Set up test fixtures."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

    def test_module_import_performance(self):
        """Test module import performance."""
        start_time = time.time()

        # Import modules (these are already imported, so this is fast)
        import importlib
        import src.optimization.differentiable.differentiable_models
        import src.optimization.differentiable.jax_optimizer

        # Force reimport to test actual import time
        importlib.reload(src.optimization.differentiable.differentiable_models)
        importlib.reload(src.optimization.differentiable.jax_optimizer)

        import_time = time.time() - start_time

        # Import should be reasonably fast (< 10 seconds allowing for JAX compilation)
        self.assertLess(import_time, 10.0)

    def test_end_to_end_performance(self):
        """Test end-to-end optimization performance."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        # Create models
        trajectory_model = TrajectoryModel(use_jit=True)
        economic_model = EconomicModel(use_jit=True)
        combined_model = create_combined_model(trajectory_model, economic_model)

        # Create optimizer with reduced iterations for speed
        optimizer = DifferentiableOptimizer(
            objective_function=combined_model,
            bounds=[
                (6.578e6, 6.978e6),
                (1.837e6, 2.137e6),
                (3.0 * 24 * 3600, 10.0 * 24 * 3600),
            ],
            method="L-BFGS-B",
            max_iterations=10,  # Reduced for performance test
            verbose=False,
        )

        # Initial guess
        x0 = jnp.array([6.778e6, 1.937e6, 4.5 * 24 * 3600])

        # Time optimization
        start_time = time.time()
        result = optimizer.optimize(x0)
        optimization_time = time.time() - start_time

        # Optimization should complete quickly (< 10 seconds)
        self.assertLess(optimization_time, 10.0)
        self.assertTrue(result.success)

    def test_memory_usage(self):
        """Test reasonable memory usage."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        # Create multiple models to test memory scaling
        models = []
        for _i in range(10):
            trajectory_model = TrajectoryModel(use_jit=True)
            economic_model = EconomicModel(use_jit=True)
            models.append((trajectory_model, economic_model))

        # This test mainly ensures we don't crash with memory issues
        # More sophisticated memory testing would require additional tools
        self.assertEqual(len(models), 10)

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameter values."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available")

        trajectory_model = TrajectoryModel(use_jit=True)

        # Test with boundary values
        extreme_params = jnp.array(
            [
                6.578e6,  # Minimum Earth radius
                2.137e6,  # Maximum lunar radius
                10.0 * 24 * 3600,  # Maximum time of flight
            ]
        )

        try:
            # Should not crash or produce NaN/Inf
            result = trajectory_model._trajectory_cost(extreme_params)

            for key, value in result.items():
                # Allow very large values but not NaN/Inf
                self.assertFalse(jnp.isnan(value), f"NaN value in {key}: {value}")
                self.assertFalse(jnp.isinf(value), f"Infinite value in {key}: {value}")

        except Exception as e:
            # If calculation fails with extreme values, that's acceptable
            # as long as it's a controlled failure, not a crash
            self.assertIsInstance(e, (ValueError, RuntimeError, ArithmeticError))


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization features."""

    def test_jit_optimizer_functionality(self):
        """Test JIT optimizer function compilation."""
        from src.optimization.differentiable.performance_optimization import (
            JITOptimizer,
            PerformanceConfig,
        )

        config = PerformanceConfig(enable_jit=True)
        jit_optimizer = JITOptimizer(config)

        # Simple objective function
        def simple_objective(x):
            return jnp.sum(x**2)

        # Test compilation
        compiled_fn = jit_optimizer.compile_objective_function(simple_objective)

        # Test compiled function works
        test_input = jnp.array([1.0, 2.0, 3.0])
        result = compiled_fn(test_input)

        self.assertAlmostEqual(float(result), 14.0, places=6)
        # Check that a compiled function was cached
        self.assertGreater(len(jit_optimizer.compiled_functions), 0)
        # Check that the key contains "objective"
        objective_keys = [
            key for key in jit_optimizer.compiled_functions.keys() if "objective" in key
        ]
        self.assertGreater(len(objective_keys), 0)

    def test_batch_optimizer_functionality(self):
        """Test batch optimizer basic functionality."""
        from src.optimization.differentiable.performance_optimization import (
            BatchOptimizer,
            PerformanceConfig,
        )

        # Create models
        trajectory_model = TrajectoryModel(use_jit=True)
        economic_model = EconomicModel(use_jit=True)

        from src.optimization.differentiable.loss_functions import (
            create_balanced_loss_function,
        )

        loss_function = create_balanced_loss_function(trajectory_model, economic_model)

        # Create batch optimizer
        config = PerformanceConfig(enable_vectorization=True)
        batch_optimizer = BatchOptimizer(
            trajectory_model, economic_model, loss_function, config
        )

        # Test batch evaluation
        test_params = jnp.array(
            [[6.778e6, 1.937e6, 4.5 * 24 * 3600], [6.828e6, 1.987e6, 5.0 * 24 * 3600]]
        )

        results = batch_optimizer.evaluate_batch(test_params)

        self.assertIn("losses", results)
        self.assertIn("gradients", results)
        self.assertEqual(results["batch_size"], 2)
        self.assertEqual(len(results["losses"]), 2)

    def test_memory_optimizer_functionality(self):
        """Test memory optimizer basic functionality."""
        from src.optimization.differentiable.performance_optimization import (
            MemoryOptimizer,
            PerformanceConfig,
        )

        config = PerformanceConfig(enable_memory_efficiency=True)
        memory_optimizer = MemoryOptimizer(config)

        # Test workspace allocation
        memory_optimizer.preallocate_workspace(
            "test_workspace", [(10, 3), (5, 5)], jnp.float32
        )

        # Test workspace access
        array = memory_optimizer.get_workspace_array("test_workspace", 0)
        self.assertEqual(array.shape, (10, 3))
        self.assertEqual(array.dtype, jnp.float32)

    def test_performance_benchmark_functionality(self):
        """Test performance benchmark basic functionality."""
        from src.optimization.differentiable.performance_optimization import (
            PerformanceBenchmark,
            PerformanceConfig,
        )

        config = PerformanceConfig(benchmark_iterations=10, warmup_iterations=2)
        benchmark = PerformanceBenchmark(config)

        # Simple function to benchmark
        def test_function(x):
            return jnp.sum(x**2)

        test_input = jnp.array([1.0, 2.0, 3.0])

        # Test benchmarking
        metrics = benchmark.benchmark_function(
            test_function, test_input, "test_function"
        )

        self.assertGreater(metrics.execution_time, 0)
        self.assertGreater(metrics.throughput, 0)
        self.assertIn("test_function", benchmark.benchmark_results)

    def test_enhanced_jax_optimizer(self):
        """Test enhanced JAX optimizer with performance features."""
        from src.optimization.differentiable.loss_functions import (
            create_balanced_loss_function,
        )

        # Create models and loss function
        trajectory_model = TrajectoryModel(use_jit=True)
        economic_model = EconomicModel(use_jit=True)
        loss_function = create_balanced_loss_function(trajectory_model, economic_model)

        # Create optimizer with JIT enabled
        optimizer = DifferentiableOptimizer(
            objective_function=loss_function.compute_loss,
            use_jit=True,
            max_iterations=5,
            verbose=False,
        )

        # Test that batch functions are created
        self.assertTrue(hasattr(optimizer, "_compiled_batch_objective"))
        self.assertTrue(hasattr(optimizer, "_compiled_batch_grad"))

        # Test batch evaluation
        test_params_batch = jnp.array(
            [[6.778e6, 1.937e6, 4.5 * 24 * 3600], [6.828e6, 1.987e6, 5.0 * 24 * 3600]]
        )

        batch_objectives = optimizer.evaluate_batch_objectives(test_params_batch)
        self.assertEqual(len(batch_objectives), 2)
        self.assertTrue(jnp.all(jnp.isfinite(batch_objectives)))

        batch_gradients = optimizer.evaluate_batch_gradients(test_params_batch)
        self.assertEqual(batch_gradients.shape, (2, 3))
        self.assertTrue(jnp.all(jnp.isfinite(batch_gradients)))


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestResultComparison(unittest.TestCase):
    """Test result comparison and evaluation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create models
        self.trajectory_model = TrajectoryModel(use_jit=True)
        self.economic_model = EconomicModel(use_jit=True)

        # Create loss function
        from src.optimization.differentiable.loss_functions import (
            create_balanced_loss_function,
        )

        self.loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )

        # Create comparator
        self.comparator = ResultComparator(
            trajectory_model=self.trajectory_model,
            economic_model=self.economic_model,
            loss_function=self.loss_function,
        )

        # Test parameters
        self.test_params = jnp.array([6.778e6, 1.937e6, 4.5 * 24 * 3600])

    def test_result_comparator_initialization(self):
        """Test result comparator initialization."""
        self.assertIsNotNone(self.comparator.trajectory_model)
        self.assertIsNotNone(self.comparator.economic_model)
        self.assertIsNotNone(self.comparator.loss_function)
        self.assertEqual(len(self.comparator.comparison_history), 0)

    def test_optimization_result_comparison(self):
        """Test comparison of optimization results."""
        # Create mock global result
        global_result = {
            "x": np.array(self.test_params),
            "fun": 1000.0,
            "success": True,
            "nit": 500,
            "nfev": 5000,
            "time": 30.0,
            "message": "Global optimization completed",
        }

        # Create local optimizer and result
        optimizer = DifferentiableOptimizer(
            objective_function=self.loss_function.compute_loss,
            max_iterations=10,
            verbose=False,
        )
        local_result = optimizer.optimize(self.test_params)

        # Compare results
        comparison = self.comparator.compare_optimization_results(
            global_result, local_result
        )

        self.assertIsInstance(comparison, ComparisonResult)
        self.assertEqual(comparison.global_result, global_result)
        self.assertEqual(comparison.local_result, local_result)
        self.assertIsInstance(comparison.solution_quality, SolutionQuality)
        self.assertGreater(comparison.speedup_factor, 0)
        self.assertGreater(comparison.efficiency_ratio, 0)

    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        # Create optimizer with longer run
        optimizer = DifferentiableOptimizer(
            objective_function=self.loss_function.compute_loss,
            max_iterations=50,
            tolerance=1e-8,
            verbose=False,
        )

        # Run optimization
        result = optimizer.optimize(self.test_params)

        # Analyze convergence
        analysis = self.comparator.analyze_convergence(result, detailed_analysis=True)

        self.assertIsInstance(analysis, ConvergenceAnalysis)
        self.assertGreaterEqual(analysis.convergence_rate, 0.0)
        self.assertGreaterEqual(analysis.solution_stability, 0.0)
        self.assertGreaterEqual(analysis.time_to_convergence, 0.0)
        self.assertIsInstance(analysis.convergence_quality, SolutionQuality)

    def test_solution_ranking(self):
        """Test solution ranking functionality."""
        # Create multiple optimization results
        results = []
        for _i in range(3):
            # Add some variation to initial conditions
            noisy_params = self.test_params * (1.0 + 0.01 * np.random.randn(3))

            optimizer = DifferentiableOptimizer(
                objective_function=self.loss_function.compute_loss,
                max_iterations=20,
                verbose=False,
            )
            result = optimizer.optimize(noisy_params)
            results.append(result)

        # Rank solutions
        rankings = self.comparator.rank_solutions(results)

        self.assertEqual(len(rankings), len(results))
        self.assertIsInstance(rankings[0], tuple)
        self.assertEqual(len(rankings[0]), 2)  # (index, score)

        # Check that rankings are sorted by score (descending)
        scores = [score for _, score in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_pareto_front_computation(self):
        """Test Pareto front computation."""
        # Create results with different objective components
        results = []
        for i in range(4):
            result = OptimizationResult(
                x=np.array(self.test_params),
                fun=100.0 + i * 10,
                success=True,
                message="Test result",
                nit=50,
                nfev=500,
                njev=50,
                optimization_time=1.0,
                objective_components={
                    "trajectory": 50.0 + i * 5,
                    "economic": 50.0 - i * 2,
                },
            )
            results.append(result)

        # Compute Pareto front
        pareto_indices = self.comparator.compute_pareto_front(
            results, objectives=["trajectory", "economic"]
        )

        self.assertIsInstance(pareto_indices, list)
        self.assertLessEqual(len(pareto_indices), len(results))

        # All indices should be valid
        for idx in pareto_indices:
            self.assertIn(idx, range(len(results)))

    def test_utility_functions(self):
        """Test utility functions for result comparison."""
        # Create test results
        global_result = {
            "x": np.array(self.test_params),
            "fun": 1000.0,
            "success": True,
            "nit": 500,
            "nfev": 5000,
            "time": 30.0,
        }

        optimizer = DifferentiableOptimizer(
            objective_function=self.loss_function.compute_loss,
            max_iterations=10,
            verbose=False,
        )
        local_result = optimizer.optimize(self.test_params)

        # Test utility functions
        comparison = compare_single_run(global_result, local_result)
        self.assertIsInstance(comparison, ComparisonResult)

        convergence = analyze_optimization_convergence(local_result)
        self.assertIsInstance(convergence, ConvergenceAnalysis)

        rankings = rank_optimization_results([local_result])
        self.assertEqual(len(rankings), 1)

        quality = evaluate_solution_quality(local_result)
        self.assertIsInstance(quality, SolutionQuality)


@unittest.skipIf(not JAX_AVAILABLE, "JAX not available")
class TestComparisonDemonstration(unittest.TestCase):
    """Test comparison demonstration functionality."""

    def test_comparison_demo_initialization(self):
        """Test comparison demo initialization."""
        demo = ComparisonDemonstration(verbose=False)

        self.assertIsNotNone(demo.trajectory_model)
        self.assertIsNotNone(demo.economic_model)
        self.assertIsNotNone(demo.loss_function)
        self.assertIsNotNone(demo.comparator)
        self.assertEqual(demo.test_parameters.shape, (5, 3))

    def test_single_comparison_demo(self):
        """Test single comparison demonstration."""
        demo = ComparisonDemonstration(verbose=False)

        # Run single comparison
        comparison = demo.demonstrate_single_comparison()

        self.assertIsInstance(comparison, ComparisonResult)
        self.assertIsNotNone(comparison.global_result)
        self.assertIsNotNone(comparison.local_result)
        self.assertIsInstance(comparison.solution_quality, SolutionQuality)

    def test_convergence_analysis_demo(self):
        """Test convergence analysis demonstration."""
        demo = ComparisonDemonstration(verbose=False)

        # Run convergence analysis
        analysis = demo.demonstrate_convergence_analysis()

        self.assertIsInstance(analysis, ConvergenceAnalysis)
        self.assertGreaterEqual(analysis.convergence_rate, 0.0)
        self.assertIsInstance(analysis.convergence_quality, SolutionQuality)

    def test_solution_ranking_demo(self):
        """Test solution ranking demonstration."""
        demo = ComparisonDemonstration(verbose=False)

        # Run solution ranking
        rankings = demo.demonstrate_solution_ranking()

        self.assertIsInstance(rankings, list)
        self.assertEqual(len(rankings), 5)  # 5 test parameters

        # Check ranking structure
        for rank_info in rankings:
            self.assertIsInstance(rank_info, tuple)
            self.assertEqual(len(rank_info), 2)  # (index, score)

    def test_pareto_analysis_demo(self):
        """Test Pareto analysis demonstration."""
        demo = ComparisonDemonstration(verbose=False)

        # Run Pareto analysis
        pareto_indices = demo.demonstrate_pareto_analysis()

        self.assertIsInstance(pareto_indices, list)
        self.assertLessEqual(len(pareto_indices), 5)  # At most 5 solutions

        # All indices should be valid
        for idx in pareto_indices:
            self.assertIn(idx, range(5))

    def test_method_benchmark_demo(self):
        """Test method benchmarking demonstration."""
        demo = ComparisonDemonstration(verbose=False)

        # Run method benchmark
        benchmark_results = demo.demonstrate_method_benchmark()

        self.assertIn("methods", benchmark_results)
        self.assertIn("problems", benchmark_results)
        self.assertIn("detailed_results", benchmark_results)
        self.assertIn("summary", benchmark_results)

        # Check that all methods were tested
        methods = benchmark_results["methods"]
        for method in methods:
            self.assertIn(method, benchmark_results["detailed_results"])

    def test_comprehensive_analysis_demo(self):
        """Test comprehensive analysis demonstration."""
        demo = ComparisonDemonstration(verbose=False)

        # Run comprehensive analysis
        results = demo.demonstrate_comprehensive_analysis()

        self.assertIn("single_comparison", results)
        self.assertIn("convergence_analysis", results)
        self.assertIn("solution_ranking", results)
        self.assertIn("pareto_analysis", results)
        self.assertIn("method_benchmark", results)
        self.assertIn("summary", results)

        # Check that all components are correct types
        self.assertIsInstance(results["single_comparison"], ComparisonResult)
        self.assertIsInstance(results["convergence_analysis"], ConvergenceAnalysis)
        self.assertIsInstance(results["solution_ranking"], list)
        self.assertIsInstance(results["pareto_analysis"], list)
        self.assertIsInstance(results["method_benchmark"], dict)
        self.assertIsInstance(results["summary"], dict)

    def test_comparison_demo_runner(self):
        """Test comparison demo runner function."""
        # This test just ensures the demo can run without errors
        success = run_comparison_demo()
        self.assertTrue(success, "Comparison demo should complete successfully")


if __name__ == "__main__":
    # Configure test verbosity
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2)
