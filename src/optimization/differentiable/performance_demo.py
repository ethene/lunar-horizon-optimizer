"""
Performance Optimization Demonstration

This module demonstrates the performance improvements achieved through
various JIT compilation and optimization techniques in the JAX differentiable
optimization system.

Features:
- Performance comparison between non-JIT and JIT implementations
- Batch optimization performance analysis
- Memory usage optimization demonstrations
- Compilation time vs execution time trade-offs
- Real-world optimization scenario benchmarks

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

import time
import numpy as np
from typing import Dict, Any

# JAX imports
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Local imports
from .differentiable_models import TrajectoryModel, EconomicModel
from .loss_functions import create_balanced_loss_function
from .jax_optimizer import DifferentiableOptimizer
from .performance_optimization import (
    PerformanceConfig,
    JITOptimizer,
    BatchOptimizer,
    PerformanceBenchmark,
    optimize_differentiable_optimizer,
    create_performance_optimized_loss_function,
    benchmark_optimization_performance,
)


class PerformanceDemo:
    """
    Comprehensive performance demonstration for JAX optimization.

    This class showcases the performance improvements achieved through
    various optimization techniques including JIT compilation, vectorization,
    and memory optimization.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize performance demonstration.

        Args:
            verbose: Whether to print detailed results
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for performance demonstration")

        self.verbose = verbose

        # Setup models
        self.trajectory_model = TrajectoryModel(use_jit=False)  # Start without JIT
        self.economic_model = EconomicModel(use_jit=False)

        # Create test parameters
        self.single_test_params = jnp.array(
            [
                6.778e6,  # Earth departure radius
                1.937e6,  # Lunar orbit radius
                4.5 * 24 * 3600,  # Time of flight
            ]
        )

        # Create batch test parameters
        self.batch_test_params = jnp.array(
            [
                [6.778e6, 1.937e6, 4.5 * 24 * 3600],
                [6.828e6, 1.987e6, 5.0 * 24 * 3600],
                [6.878e6, 2.037e6, 5.5 * 24 * 3600],
                [6.928e6, 2.087e6, 6.0 * 24 * 3600],
                [6.978e6, 2.137e6, 6.5 * 24 * 3600],
            ]
        )

        # Performance configurations
        self.no_jit_config = PerformanceConfig(
            enable_jit=False, enable_vectorization=False, enable_memory_efficiency=False
        )

        self.basic_jit_config = PerformanceConfig(
            enable_jit=True, enable_vectorization=False, enable_memory_efficiency=False
        )

        self.optimized_config = PerformanceConfig(
            enable_jit=True,
            enable_vectorization=True,
            enable_memory_efficiency=True,
            enable_compilation_cache=True,
        )

    def demonstrate_jit_compilation_benefits(self) -> Dict[str, Any]:
        """
        Demonstrate the performance benefits of JIT compilation.

        Returns:
            Dictionary with JIT compilation benchmark results
        """
        if self.verbose:
            print("=" * 60)
            print("🚀 JIT Compilation Performance Demonstration")
            print("=" * 60)

        results = {}

        # Create loss functions with different JIT configurations
        loss_no_jit = create_balanced_loss_function(
            TrajectoryModel(use_jit=False), EconomicModel(use_jit=False)
        )
        loss_no_jit.use_jit = False
        loss_no_jit._setup_compiled_functions()

        loss_with_jit = create_balanced_loss_function(
            TrajectoryModel(use_jit=True), EconomicModel(use_jit=True)
        )

        # Benchmark different implementations
        benchmark = PerformanceBenchmark(self.optimized_config)

        implementations = {
            "No JIT": loss_no_jit.compute_loss,
            "With JIT": loss_with_jit.compute_loss,
        }

        comparison_results = benchmark.compare_implementations(
            implementations, self.single_test_params
        )

        results["single_parameter_comparison"] = comparison_results

        if self.verbose:
            print("\n📊 Single Parameter Evaluation Performance:")
            for name, metrics in comparison_results.items():
                print(f"  {name}:")
                print(f"    Execution time: {metrics['execution_time']:.6f}s")
                print(f"    Throughput: {metrics['throughput']:.1f} evals/s")
                print(f"    Speedup: {metrics['relative_speedup']:.2f}x")

        return results

    def demonstrate_vectorization_benefits(self) -> Dict[str, Any]:
        """
        Demonstrate the performance benefits of vectorization.

        Returns:
            Dictionary with vectorization benchmark results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("⚡ Vectorization Performance Demonstration")
            print("=" * 60)

        results = {}

        # Create optimized loss function
        loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )
        loss_function = create_performance_optimized_loss_function(
            loss_function, self.optimized_config
        )

        # Create batch optimizer
        batch_optimizer = BatchOptimizer(
            self.trajectory_model,
            self.economic_model,
            loss_function,
            self.optimized_config,
        )

        # Time individual evaluations
        base_loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )
        start_time = time.time()
        individual_results = []
        for params in self.batch_test_params:
            result = base_loss_function.compute_loss(params)
            individual_results.append(result)
        individual_time = time.time() - start_time

        # Time batch evaluation
        start_time = time.time()
        batch_results = batch_optimizer.evaluate_batch(self.batch_test_params)
        batch_time = batch_results["execution_time"]

        # Calculate speedup
        speedup = individual_time / batch_time if batch_time > 0 else 1.0

        results = {
            "batch_size": self.batch_test_params.shape[0],
            "individual_evaluation_time": individual_time,
            "batch_evaluation_time": batch_time,
            "vectorization_speedup": speedup,
            "throughput_individual": self.batch_test_params.shape[0] / individual_time,
            "throughput_batch": self.batch_test_params.shape[0] / batch_time,
        }

        if self.verbose:
            print("\n📊 Batch Evaluation Performance:")
            print(f"  Batch size: {results['batch_size']}")
            print(
                f"  Individual evaluation time: {results['individual_evaluation_time']:.6f}s"
            )
            print(f"  Batch evaluation time: {results['batch_evaluation_time']:.6f}s")
            print(f"  Vectorization speedup: {results['vectorization_speedup']:.2f}x")
            print(
                f"  Throughput improvement: {results['throughput_batch']/results['throughput_individual']:.2f}x"
            )

        return results

    def demonstrate_optimizer_performance(self) -> Dict[str, Any]:
        """
        Demonstrate performance improvements in optimization.

        Returns:
            Dictionary with optimizer performance results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🎯 Optimizer Performance Demonstration")
            print("=" * 60)

        results = {}

        # Create loss function
        loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )

        # Create standard optimizer
        standard_optimizer = DifferentiableOptimizer(
            objective_function=loss_function.compute_loss,
            bounds=[
                (6.578e6, 6.978e6),
                (1.837e6, 2.137e6),
                (3.0 * 24 * 3600, 10.0 * 24 * 3600),
            ],
            max_iterations=50,
            verbose=False,
        )

        # Create performance-optimized optimizer
        optimized_optimizer = optimize_differentiable_optimizer(
            standard_optimizer, self.optimized_config
        )

        # Test both optimizers
        test_point = self.single_test_params * (1.0 + 0.1 * np.random.randn(3))

        # Benchmark standard optimizer
        start_time = time.time()
        standard_result = standard_optimizer.optimize(test_point)
        standard_time = time.time() - start_time

        # Benchmark optimized optimizer
        start_time = time.time()
        optimized_result = optimized_optimizer.optimize(test_point)
        optimized_time = time.time() - start_time

        results = {
            "standard_optimizer": {
                "optimization_time": standard_time,
                "success": standard_result.success,
                "final_objective": standard_result.fun,
                "function_evaluations": standard_result.nfev,
                "iterations": standard_result.nit,
            },
            "optimized_optimizer": {
                "optimization_time": optimized_time,
                "success": optimized_result.success,
                "final_objective": optimized_result.fun,
                "function_evaluations": optimized_result.nfev,
                "iterations": optimized_result.nit,
            },
            "speedup": standard_time / optimized_time if optimized_time > 0 else 1.0,
        }

        if self.verbose:
            print("\n📊 Optimization Performance Comparison:")
            print("  Standard Optimizer:")
            print(
                f"    Time: {results['standard_optimizer']['optimization_time']:.4f}s"
            )
            print(f"    Success: {results['standard_optimizer']['success']}")
            print(
                f"    Final objective: {results['standard_optimizer']['final_objective']:.6e}"
            )
            print(
                f"    Function evaluations: {results['standard_optimizer']['function_evaluations']}"
            )

            print("  Optimized Optimizer:")
            print(
                f"    Time: {results['optimized_optimizer']['optimization_time']:.4f}s"
            )
            print(f"    Success: {results['optimized_optimizer']['success']}")
            print(
                f"    Final objective: {results['optimized_optimizer']['final_objective']:.6e}"
            )
            print(
                f"    Function evaluations: {results['optimized_optimizer']['function_evaluations']}"
            )
            print(f"    Speedup: {results['speedup']:.2f}x")

        return results

    def demonstrate_compilation_cache_benefits(self) -> Dict[str, Any]:
        """
        Demonstrate benefits of compilation caching.

        Returns:
            Dictionary with compilation cache benchmark results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("💾 Compilation Cache Demonstration")
            print("=" * 60)

        results = {}

        # Test without cache
        config_no_cache = PerformanceConfig(
            enable_jit=True, enable_compilation_cache=False
        )

        # Test with cache
        config_with_cache = PerformanceConfig(
            enable_jit=True, enable_compilation_cache=True
        )

        # Time function compilation without cache
        start_time = time.time()
        jit_optimizer_no_cache = JITOptimizer(config_no_cache)
        loss_fn = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )
        compiled_fn_no_cache = jit_optimizer_no_cache.compile_objective_function(
            loss_fn.compute_loss
        )

        # Trigger compilation
        _ = compiled_fn_no_cache(self.single_test_params)
        compilation_time_no_cache = time.time() - start_time

        # Time function compilation with cache
        start_time = time.time()
        jit_optimizer_with_cache = JITOptimizer(config_with_cache)
        compiled_fn_with_cache = jit_optimizer_with_cache.compile_objective_function(
            loss_fn.compute_loss
        )

        # Trigger compilation
        _ = compiled_fn_with_cache(self.single_test_params)
        compilation_time_with_cache = time.time() - start_time

        results = {
            "compilation_time_no_cache": compilation_time_no_cache,
            "compilation_time_with_cache": compilation_time_with_cache,
            "cache_benefit": max(
                0, compilation_time_no_cache - compilation_time_with_cache
            ),
            "cache_enabled": config_with_cache.enable_compilation_cache,
        }

        if self.verbose:
            print("\n📊 Compilation Cache Performance:")
            print(
                f"  Compilation time (no cache): {results['compilation_time_no_cache']:.4f}s"
            )
            print(
                f"  Compilation time (with cache): {results['compilation_time_with_cache']:.4f}s"
            )
            print(f"  Cache benefit: {results['cache_benefit']:.4f}s")

        return results

    def demonstrate_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive performance benchmark covering all optimization techniques.

        Returns:
            Dictionary with comprehensive benchmark results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔬 Comprehensive Performance Benchmark")
            print("=" * 60)

        # Run all individual demonstrations
        jit_results = self.demonstrate_jit_compilation_benefits()
        vectorization_results = self.demonstrate_vectorization_benefits()
        optimizer_results = self.demonstrate_optimizer_performance()
        cache_results = self.demonstrate_compilation_cache_benefits()

        # Run comprehensive system benchmark
        system_benchmark = benchmark_optimization_performance(
            self.trajectory_model,
            self.economic_model,
            self.batch_test_params,
            self.optimized_config,
        )

        results = {
            "jit_compilation": jit_results,
            "vectorization": vectorization_results,
            "optimizer_performance": optimizer_results,
            "compilation_cache": cache_results,
            "system_benchmark": system_benchmark,
            "summary": self._generate_performance_summary(
                jit_results, vectorization_results, optimizer_results, cache_results
            ),
        }

        if self.verbose:
            print("\n🎉 Performance Demonstration Complete!")
            print("\n📋 Performance Summary:")
            summary = results["summary"]
            print(f"  JIT Compilation Speedup: {summary['jit_speedup']:.2f}x")
            print(f"  Vectorization Speedup: {summary['vectorization_speedup']:.2f}x")
            print(f"  Optimizer Speedup: {summary['optimizer_speedup']:.2f}x")
            print(
                f"  Overall Performance Improvement: {summary['overall_improvement']:.2f}x"
            )

        return results

    def _generate_performance_summary(
        self,
        jit_results: Dict[str, Any],
        vectorization_results: Dict[str, Any],
        optimizer_results: Dict[str, Any],
        cache_results: Dict[str, Any],
    ) -> Dict[str, float]:
        """Generate summary of performance improvements."""

        # Extract speedups
        jit_speedup = 1.0
        if "single_parameter_comparison" in jit_results:
            jit_comparison = jit_results["single_parameter_comparison"]
            if "With JIT" in jit_comparison:
                jit_speedup = jit_comparison["With JIT"]["relative_speedup"]

        vectorization_speedup = vectorization_results.get("vectorization_speedup", 1.0)
        optimizer_speedup = optimizer_results.get("speedup", 1.0)

        # Calculate overall improvement (multiplicative)
        overall_improvement = jit_speedup * vectorization_speedup * optimizer_speedup

        return {
            "jit_speedup": jit_speedup,
            "vectorization_speedup": vectorization_speedup,
            "optimizer_speedup": optimizer_speedup,
            "overall_improvement": overall_improvement,
            "cache_benefit_seconds": cache_results.get("cache_benefit", 0.0),
        }


def run_performance_demo() -> bool:
    """
    Run the complete performance optimization demonstration.

    Returns:
        True if demonstration completed successfully
    """
    try:
        demo = PerformanceDemo(verbose=True)
        demo.demonstrate_comprehensive_benchmark()
        return True
    except Exception as e:
        print(f"❌ Performance demonstration failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the performance demonstration when script is executed directly
    print("Running JAX Performance Optimization Demonstration...")
    success = run_performance_demo()
    print(f"\nPerformance demo result: {'SUCCESS' if success else 'FAILED'}")
