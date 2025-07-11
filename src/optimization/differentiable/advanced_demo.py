"""
Advanced JAX Differentiable Optimization Demonstration

This module demonstrates the advanced features of the JAX differentiable
optimization system including multi-objective loss functions, constraint
handling, and PyGMO integration.

Features:
- Multi-objective loss function configurations
- Constraint handling methods comparison
- PyGMO-JAX hybrid optimization workflows
- Performance benchmarking and analysis
- Advanced optimization strategies

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

import time
from typing import Dict, Any

# JAX imports (with fallback)
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# PyGMO imports (with fallback)
try:
    import pygmo as pg

    PYGMO_AVAILABLE = True
except ImportError:
    PYGMO_AVAILABLE = False

# Local imports
from .differentiable_models import TrajectoryModel, EconomicModel
from .loss_functions import (
    MultiObjectiveLoss,
    create_balanced_loss_function,
    create_performance_focused_loss_function,
    create_economic_focused_loss_function,
)
from .constraints import (
    create_penalty_constraint_handler,
    create_barrier_constraint_handler,
    create_adaptive_constraint_handler,
)
from .integration import (
    create_standard_hybrid_optimizer,
    create_fast_hybrid_optimizer,
)


class AdvancedOptimizationDemo:
    """
    Advanced demonstration of JAX differentiable optimization capabilities.

    This class showcases sophisticated optimization techniques including
    multi-objective optimization, constraint handling, and hybrid workflows.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize advanced optimization demonstration.

        Args:
            verbose: Whether to print detailed progress
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for advanced optimization demonstration")

        self.verbose = verbose

        # Initialize core models
        self.trajectory_model = TrajectoryModel(use_jit=True)
        self.economic_model = EconomicModel(use_jit=True)

        # Test parameters
        self.test_params = jnp.array(
            [
                6.778e6,  # Earth departure radius (400 km altitude)
                1.937e6,  # Lunar orbit radius (200 km altitude)
                4.5 * 24 * 3600,  # Time of flight (4.5 days)
            ]
        )

        self.suboptimal_params = jnp.array(
            [
                6.878e6,  # Earth departure radius (500 km altitude)
                2.037e6,  # Lunar orbit radius (300 km altitude)
                6.0 * 24 * 3600,  # Time of flight (6.0 days)
            ]
        )

    def demonstrate_loss_functions(self) -> Dict[str, Any]:
        """
        Demonstrate different multi-objective loss function configurations.

        Returns:
            Dictionary with loss function demonstration results
        """
        if self.verbose:
            print("=" * 60)
            print("🎯 Multi-Objective Loss Function Demonstration")
            print("=" * 60)

        results = {}

        # Test different loss function configurations
        loss_functions = {
            "balanced": create_balanced_loss_function(
                self.trajectory_model, self.economic_model
            ),
            "performance_focused": create_performance_focused_loss_function(
                self.trajectory_model, self.economic_model
            ),
            "economic_focused": create_economic_focused_loss_function(
                self.trajectory_model, self.economic_model
            ),
        }

        for name, loss_fn in loss_functions.items():
            if self.verbose:
                print(f"\n📊 Testing {name} loss function...")

            # Evaluate loss function
            loss_value = float(loss_fn.compute_loss(self.test_params))
            objective_breakdown = loss_fn.get_objective_breakdown(self.test_params)

            # Update metrics for adaptive strategies
            loss_fn.update_metrics(self.test_params)

            results[name] = {
                "loss_value": loss_value,
                "objective_breakdown": objective_breakdown,
                "weighting_strategy": loss_fn.config.weighting_strategy.value,
                "normalization_method": loss_fn.config.normalization_method.value,
            }

            if self.verbose:
                print(f"  Loss value: {loss_value:.6f}")
                print(
                    f"  Total objective: {objective_breakdown['total_objective']:.6f}"
                )
                print(f"  Total penalty: {objective_breakdown['total_penalty']:.6f}")
                print(f"  Raw objectives: {objective_breakdown['raw_objectives']}")

        return results

    def demonstrate_constraint_handling(self) -> Dict[str, Any]:
        """
        Demonstrate different constraint handling methods.

        Returns:
            Dictionary with constraint handling demonstration results
        """
        if self.verbose:
            print("=" * 60)
            print("⚖️  Constraint Handling Methods Demonstration")
            print("=" * 60)

        results = {}

        # Create different constraint handlers
        constraint_handlers = {
            "penalty": create_penalty_constraint_handler(
                self.trajectory_model, self.economic_model, penalty_factor=1e6
            ),
            "barrier": create_barrier_constraint_handler(
                self.trajectory_model, self.economic_model, barrier_parameter=0.1
            ),
            "adaptive": create_adaptive_constraint_handler(
                self.trajectory_model, self.economic_model
            ),
        }

        for name, handler in constraint_handlers.items():
            if self.verbose:
                print(f"\n🛡️  Testing {name} constraint handling...")

            # Analyze constraint violations
            _ = handler.analyze_constraint_violations(self.suboptimal_params)
            constraint_function_value = float(
                handler.compute_constraint_function(self.suboptimal_params)
            )
            summary = handler.get_constraint_summary(self.suboptimal_params)

            # Update adaptive parameters if applicable
            handler.update_adaptive_parameters(self.suboptimal_params)

            results[name] = {
                "constraint_function_value": constraint_function_value,
                "total_violations": summary["total_violations"],
                "satisfaction_rate": summary["satisfaction_rate"],
                "max_violation": summary["max_violation"],
                "handling_method": summary["handling_method"],
                "violations_by_type": summary["violations_by_type"],
            }

            if self.verbose:
                print(f"  Constraint function value: {constraint_function_value:.2e}")
                print(f"  Total violations: {summary['total_violations']}")
                print(f"  Satisfaction rate: {summary['satisfaction_rate']:.2%}")
                print(f"  Max violation: {summary['max_violation']:.2e}")
                print(f"  Violations by type: {summary['violations_by_type']}")

        return results

    def demonstrate_hybrid_optimization(self) -> Dict[str, Any]:
        """
        Demonstrate PyGMO-JAX hybrid optimization.

        Returns:
            Dictionary with hybrid optimization demonstration results
        """
        if not PYGMO_AVAILABLE:
            if self.verbose:
                print(
                    "❌ PyGMO not available - skipping hybrid optimization demonstration"
                )
            return {"error": "PyGMO not available"}

        if self.verbose:
            print("=" * 60)
            print("🔄 PyGMO-JAX Hybrid Optimization Demonstration")
            print("=" * 60)

        results = {}

        # Create loss function and constraint handler for integration
        loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )
        constraint_handler = create_adaptive_constraint_handler(
            self.trajectory_model, self.economic_model
        )

        # Test different hybrid optimization configurations
        hybrid_optimizers = {
            "fast": create_fast_hybrid_optimizer(
                self.trajectory_model,
                self.economic_model,
                loss_function,
                constraint_handler,
            ),
            "standard": create_standard_hybrid_optimizer(
                self.trajectory_model,
                self.economic_model,
                loss_function,
                constraint_handler,
            ),
        }

        for name, optimizer in hybrid_optimizers.items():
            if self.verbose:
                print(f"\n🚀 Testing {name} hybrid optimization...")

            try:
                # Run hybrid optimization
                start_time = time.time()
                hybrid_results = optimizer.run_hybrid_optimization()
                optimization_time = time.time() - start_time

                results[name] = {
                    "success": True,
                    "best_objective": hybrid_results["best_objective"],
                    "improvement_over_global": hybrid_results["analysis"][
                        "improvement_over_global"
                    ],
                    "local_success_rate": hybrid_results["analysis"][
                        "local_success_rate"
                    ],
                    "total_time": optimization_time,
                    "global_time": hybrid_results["global_results"][
                        "optimization_time"
                    ],
                    "local_successful": hybrid_results["num_successful_local"],
                    "configuration": {
                        "population_size": optimizer.config.pygmo_population_size,
                        "generations": optimizer.config.pygmo_generations,
                        "local_starts": optimizer.config.num_local_starts,
                    },
                }

                if self.verbose:
                    print(
                        f"  ✅ Success! Best objective: {hybrid_results['best_objective']:.6f}"
                    )
                    print(
                        f"  📈 Improvement over global: {hybrid_results['analysis']['improvement_over_global']:.2f}%"
                    )
                    print(
                        f"  🎯 Local success rate: {hybrid_results['analysis']['local_success_rate']:.2%}"
                    )
                    print(f"  ⏱️  Total time: {optimization_time:.2f}s")
                    print(
                        f"  🌍 Global time: {hybrid_results['global_results']['optimization_time']:.2f}s"
                    )
                    print(
                        f"  🚀 Local successful: {hybrid_results['num_successful_local']}"
                    )

            except Exception as e:
                results[name] = {"success": False, "error": str(e)}

                if self.verbose:
                    print(f"  ❌ Failed: {str(e)}")

        return results

    def benchmark_optimization_methods(self) -> Dict[str, Any]:
        """
        Benchmark different optimization methods and configurations.

        Returns:
            Dictionary with benchmarking results
        """
        if self.verbose:
            print("=" * 60)
            print("⏱️  Optimization Methods Benchmarking")
            print("=" * 60)

        results = {}

        # Create test loss function
        loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )

        # Test configurations
        test_configs = [
            {
                "name": "basic_jax",
                "description": "Basic JAX optimization",
                "test_fn": lambda: self._benchmark_basic_jax(loss_function),
            },
            {
                "name": "multi_objective",
                "description": "Multi-objective loss comparison",
                "test_fn": lambda: self._benchmark_loss_functions(),
            },
            {
                "name": "constraint_methods",
                "description": "Constraint handling methods",
                "test_fn": lambda: self._benchmark_constraint_methods(),
            },
        ]

        for config in test_configs:
            if self.verbose:
                print(f"\n🔬 Benchmarking {config['description']}...")

            try:
                start_time = time.time()
                benchmark_result = config["test_fn"]()
                benchmark_time = time.time() - start_time

                results[config["name"]] = {
                    "success": True,
                    "benchmark_time": benchmark_time,
                    "results": benchmark_result,
                }

                if self.verbose:
                    print(f"  ✅ Completed in {benchmark_time:.2f}s")

            except Exception as e:
                results[config["name"]] = {"success": False, "error": str(e)}

                if self.verbose:
                    print(f"  ❌ Failed: {str(e)}")

        return results

    def _benchmark_basic_jax(self, loss_function: MultiObjectiveLoss) -> Dict[str, Any]:
        """Benchmark basic JAX optimization performance."""
        from .jax_optimizer import DifferentiableOptimizer

        optimizer = DifferentiableOptimizer(
            objective_function=loss_function.compute_loss,
            bounds=[
                (6.578e6, 6.978e6),
                (1.837e6, 2.137e6),
                (3.0 * 24 * 3600, 10.0 * 24 * 3600),
            ],
            method="L-BFGS-B",
            max_iterations=50,
            verbose=False,
        )

        # Test optimization
        result = optimizer.optimize(self.suboptimal_params)

        return {
            "initial_objective": result.initial_objective,
            "final_objective": result.fun,
            "improvement": result.improvement_percentage,
            "iterations": result.nit,
            "function_evaluations": result.nfev,
            "optimization_time": result.optimization_time,
            "success": result.success,
        }

    def _benchmark_loss_functions(self) -> Dict[str, Any]:
        """Benchmark different loss function configurations."""
        loss_functions = {
            "balanced": create_balanced_loss_function(
                self.trajectory_model, self.economic_model
            ),
            "performance": create_performance_focused_loss_function(
                self.trajectory_model, self.economic_model
            ),
            "economic": create_economic_focused_loss_function(
                self.trajectory_model, self.economic_model
            ),
        }

        results = {}
        for name, loss_fn in loss_functions.items():
            start_time = time.time()
            loss_value = float(loss_fn.compute_loss(self.test_params))
            eval_time = time.time() - start_time

            results[name] = {
                "loss_value": loss_value,
                "evaluation_time": eval_time,
                "strategy": loss_fn.config.weighting_strategy.value,
            }

        return results

    def _benchmark_constraint_methods(self) -> Dict[str, Any]:
        """Benchmark different constraint handling methods."""
        handlers = {
            "penalty": create_penalty_constraint_handler(
                self.trajectory_model, self.economic_model
            ),
            "barrier": create_barrier_constraint_handler(
                self.trajectory_model, self.economic_model
            ),
            "adaptive": create_adaptive_constraint_handler(
                self.trajectory_model, self.economic_model
            ),
        }

        results = {}
        for name, handler in handlers.items():
            start_time = time.time()
            constraint_value = float(
                handler.compute_constraint_function(self.suboptimal_params)
            )
            eval_time = time.time() - start_time

            summary = handler.get_constraint_summary(self.suboptimal_params)

            results[name] = {
                "constraint_value": constraint_value,
                "evaluation_time": eval_time,
                "total_violations": summary["total_violations"],
                "satisfaction_rate": summary["satisfaction_rate"],
            }

        return results

    def run_complete_demonstration(self) -> Dict[str, Any]:
        """
        Run the complete advanced optimization demonstration.

        Returns:
            Dictionary with all demonstration results
        """
        if self.verbose:
            print("🌙 Advanced JAX Differentiable Optimization Demonstration")
            print(
                "🚀 Showcasing multi-objective optimization, constraints, and hybrid workflows"
            )
            print()

        demo_results = {}

        # Run demonstrations
        demo_results["loss_functions"] = self.demonstrate_loss_functions()
        demo_results["constraint_handling"] = self.demonstrate_constraint_handling()
        demo_results["hybrid_optimization"] = self.demonstrate_hybrid_optimization()
        demo_results["benchmarks"] = self.benchmark_optimization_methods()

        # Summary
        if self.verbose:
            print("=" * 60)
            print("📋 Demonstration Summary")
            print("=" * 60)

            print(f"✅ Loss functions tested: {len(demo_results['loss_functions'])}")
            print(
                f"✅ Constraint methods tested: {len(demo_results['constraint_handling'])}"
            )

            if "error" not in demo_results["hybrid_optimization"]:
                successful_hybrid = sum(
                    1
                    for r in demo_results["hybrid_optimization"].values()
                    if r.get("success", False)
                )
                print(f"✅ Hybrid optimizations successful: {successful_hybrid}")
            else:
                print("⚠️  Hybrid optimization unavailable (PyGMO not installed)")

            successful_benchmarks = sum(
                1
                for r in demo_results["benchmarks"].values()
                if r.get("success", False)
            )
            print(f"✅ Benchmarks completed: {successful_benchmarks}")

            print("\n🎉 Advanced demonstration completed successfully!")

        return demo_results


def run_advanced_demo() -> bool:
    """
    Run the advanced JAX optimization demonstration.

    Returns:
        True if demonstration completed successfully
    """
    try:
        demo = AdvancedOptimizationDemo(verbose=True)
        _ = demo.run_complete_demonstration()
        return True
    except Exception as e:
        print(f"❌ Advanced demonstration failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the advanced demonstration when script is executed directly
    print("Running Advanced JAX Differentiable Optimization Demonstration...")
    success = run_advanced_demo()
    print(f"\nAdvanced demo result: {'SUCCESS' if success else 'FAILED'}")
