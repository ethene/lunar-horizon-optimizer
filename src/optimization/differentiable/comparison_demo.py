"""
Result Comparison Demonstration Module

This module demonstrates the result comparison and evaluation capabilities
of the differentiable optimization system, showcasing how to analyze and
compare optimization results from different methods.

Features:
- Global vs local optimization comparison demonstrations
- Convergence analysis with detailed metrics
- Solution ranking and quality assessment
- Pareto front analysis for multi-objective problems
- Performance benchmarking across methods

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# JAX imports
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Local imports
from .result_comparison import (
    ResultComparator,
    ComparisonResult,
    ConvergenceAnalysis,
    ComparisonMetric,
    evaluate_solution_quality,
)
from .jax_optimizer import DifferentiableOptimizer
from .differentiable_models import TrajectoryModel, EconomicModel
from .loss_functions import create_balanced_loss_function


class ComparisonDemonstration:
    """
    Comprehensive demonstration of result comparison capabilities.

    This class showcases various aspects of optimization result comparison
    including quality assessment, convergence analysis, and method benchmarking.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize comparison demonstration.

        Args:
            verbose: Whether to print detailed results
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for comparison demonstration")

        self.verbose = verbose

        # Setup models for demonstration
        self.trajectory_model = TrajectoryModel(use_jit=True)
        self.economic_model = EconomicModel(use_jit=True)

        # Create loss function
        self.loss_function = create_balanced_loss_function(
            self.trajectory_model, self.economic_model
        )

        # Initialize comparator
        self.comparator = ResultComparator(
            trajectory_model=self.trajectory_model,
            economic_model=self.economic_model,
            loss_function=self.loss_function,
        )

        # Demo parameters
        self.test_parameters = jnp.array(
            [
                [6.778e6, 1.937e6, 4.5 * 24 * 3600],
                [6.828e6, 1.987e6, 5.0 * 24 * 3600],
                [6.878e6, 2.037e6, 5.5 * 24 * 3600],
                [6.928e6, 2.087e6, 6.0 * 24 * 3600],
                [6.978e6, 2.137e6, 6.5 * 24 * 3600],
            ]
        )

    def demonstrate_single_comparison(self) -> ComparisonResult:
        """
        Demonstrate comparison of single global vs local optimization run.

        Returns:
            ComparisonResult with detailed analysis
        """
        if self.verbose:
            print("=" * 60)
            print("🔍 Single Optimization Run Comparison")
            print("=" * 60)

        # Create test parameters
        test_params = self.test_parameters[0]

        # Simulate global optimization result (PyGMO-style)
        global_result = self._simulate_global_optimization(test_params)

        # Create local optimizer
        local_optimizer = DifferentiableOptimizer(
            objective_function=self.loss_function.compute_loss,
            bounds=[
                (6.578e6, 6.978e6),
                (1.837e6, 2.137e6),
                (3.0 * 24 * 3600, 10.0 * 24 * 3600),
            ],
            max_iterations=100,
            verbose=False,
        )

        # Run local optimization
        local_result = local_optimizer.optimize(test_params)

        # Compare results
        comparison = self.comparator.compare_optimization_results(
            global_result, local_result
        )

        if self.verbose:
            self._print_comparison_results(comparison)

        return comparison

    def demonstrate_convergence_analysis(self) -> ConvergenceAnalysis:
        """
        Demonstrate detailed convergence analysis.

        Returns:
            ConvergenceAnalysis with metrics
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("📈 Convergence Analysis Demonstration")
            print("=" * 60)

        # Create optimizer with detailed tracking
        optimizer = DifferentiableOptimizer(
            objective_function=self.loss_function.compute_loss,
            bounds=[
                (6.578e6, 6.978e6),
                (1.837e6, 2.137e6),
                (3.0 * 24 * 3600, 10.0 * 24 * 3600),
            ],
            max_iterations=200,
            tolerance=1e-8,
            verbose=False,
        )

        # Run optimization with initial point
        initial_point = self.test_parameters[0] * (1.0 + 0.1 * np.random.randn(3))
        result = optimizer.optimize(initial_point)

        # Analyze convergence
        convergence_analysis = self.comparator.analyze_convergence(
            result, detailed_analysis=True
        )

        if self.verbose:
            self._print_convergence_analysis(convergence_analysis)

        return convergence_analysis

    def demonstrate_solution_ranking(self) -> List[Tuple[int, float]]:
        """
        Demonstrate solution ranking across multiple runs.

        Returns:
            List of (index, score) rankings
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🏆 Solution Ranking Demonstration")
            print("=" * 60)

        # Create multiple optimization results
        results = []

        for i, test_params in enumerate(self.test_parameters):
            # Add some noise to initial conditions
            noisy_params = test_params * (1.0 + 0.05 * np.random.randn(3))

            # Create optimizer
            optimizer = DifferentiableOptimizer(
                objective_function=self.loss_function.compute_loss,
                bounds=[
                    (6.578e6, 6.978e6),
                    (1.837e6, 2.137e6),
                    (3.0 * 24 * 3600, 10.0 * 24 * 3600),
                ],
                max_iterations=100,
                verbose=False,
            )

            # Optimize
            result = optimizer.optimize(noisy_params)
            results.append(result)

            if self.verbose:
                print(
                    f"  Run {i+1}: Success={result.success}, "
                    f"Objective={result.fun:.6e}, Time={result.optimization_time:.3f}s"
                )

        # Rank solutions
        rankings = self.comparator.rank_solutions(
            results,
            criteria=[
                ComparisonMetric.OBJECTIVE_VALUE,
                ComparisonMetric.CONVERGENCE_RATE,
                ComparisonMetric.SOLUTION_QUALITY,
                ComparisonMetric.COMPUTATIONAL_EFFICIENCY,
            ],
        )

        if self.verbose:
            print("\n📊 Solution Rankings:")
            for rank, (idx, score) in enumerate(rankings):
                result = results[idx]
                print(f"  Rank {rank+1}: Run {idx+1} (Score: {score:.3f})")
                print(f"    Objective: {result.fun:.6e}")
                print(f"    Success: {result.success}")
                print(f"    Time: {result.optimization_time:.3f}s")
                print(f"    Quality: {evaluate_solution_quality(result).value}")

        return rankings

    def demonstrate_pareto_analysis(self) -> List[int]:
        """
        Demonstrate Pareto front analysis for multi-objective optimization.

        Returns:
            List of Pareto optimal solution indices
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🎯 Pareto Front Analysis Demonstration")
            print("=" * 60)

        # Create multiple solutions with different trade-offs
        results = []

        # Different weight configurations for multi-objective optimization
        weight_configs = [
            {"trajectory": 1.0, "economic": 0.1},  # Trajectory-focused
            {"trajectory": 0.5, "economic": 0.5},  # Balanced
            {"trajectory": 0.1, "economic": 1.0},  # Economic-focused
            {"trajectory": 0.8, "economic": 0.2},  # Mostly trajectory
            {"trajectory": 0.2, "economic": 0.8},  # Mostly economic
        ]

        for i, weights in enumerate(weight_configs):
            # Create custom loss function with specific weights
            def weighted_loss(x):
                traj_result = self.trajectory_model._trajectory_cost(x)
                econ_result = self.economic_model._economic_cost(x)

                # Extract scalar values from results
                traj_cost = traj_result.get("delta_v", 0.0) + 0.1 * traj_result.get(
                    "time_of_flight", 0.0
                )
                econ_cost = econ_result.get("total_cost", 0.0)

                return (
                    weights["trajectory"] * traj_cost + weights["economic"] * econ_cost
                )

            # Create optimizer
            optimizer = DifferentiableOptimizer(
                objective_function=weighted_loss,
                bounds=[
                    (6.578e6, 6.978e6),
                    (1.837e6, 2.137e6),
                    (3.0 * 24 * 3600, 10.0 * 24 * 3600),
                ],
                max_iterations=100,
                verbose=False,
            )

            # Optimize
            result = optimizer.optimize(
                self.test_parameters[i % len(self.test_parameters)]
            )

            # Add objective components for Pareto analysis
            if result.success:
                traj_result = self.trajectory_model._trajectory_cost(result.x)
                econ_result = self.economic_model._economic_cost(result.x)

                # Extract scalar objectives
                traj_obj = float(
                    traj_result.get("delta_v", 0.0)
                    + 0.1 * traj_result.get("time_of_flight", 0.0)
                )
                econ_obj = float(econ_result.get("total_cost", 0.0))

                result.objective_components = {
                    "trajectory": traj_obj,
                    "economic": econ_obj,
                }

            results.append(result)

            if self.verbose:
                traj_val = result.objective_components.get("trajectory", 0.0)
                econ_val = result.objective_components.get("economic", 0.0)
                print(
                    f"  Solution {i+1}: Trajectory={traj_val:.6e}, Economic={econ_val:.6e}"
                )

        # Compute Pareto front
        pareto_indices = self.comparator.compute_pareto_front(
            results, objectives=["trajectory", "economic"]
        )

        if self.verbose:
            print(
                f"\n🏆 Pareto Optimal Solutions: {len(pareto_indices)} out of {len(results)}"
            )
            for idx in pareto_indices:
                result = results[idx]
                traj_val = result.objective_components.get("trajectory", 0.0)
                econ_val = result.objective_components.get("economic", 0.0)
                print(
                    f"  Solution {idx+1}: Trajectory={traj_val:.6e}, Economic={econ_val:.6e}"
                )

        return pareto_indices

    def demonstrate_method_benchmark(self) -> Dict[str, Any]:
        """
        Demonstrate benchmarking of different optimization methods.

        Returns:
            Dictionary with benchmark results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("⚡ Method Benchmarking Demonstration")
            print("=" * 60)

        # Define test problems
        test_problems = []
        for i, params in enumerate(self.test_parameters):
            test_problems.append(
                {
                    "name": f"Problem_{i+1}",
                    "initial_point": params,
                    "objective_function": self.loss_function.compute_loss,
                    "bounds": [
                        (6.578e6, 6.978e6),
                        (1.837e6, 2.137e6),
                        (3.0 * 24 * 3600, 10.0 * 24 * 3600),
                    ],
                }
            )

        # Define methods to benchmark
        methods = ["L-BFGS-B", "SLSQP", "CG", "BFGS"]

        # Run benchmark
        benchmark_results = self._run_method_benchmark(test_problems, methods)

        if self.verbose:
            self._print_benchmark_results(benchmark_results)

        return benchmark_results

    def demonstrate_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis demonstration covering all features.

        Returns:
            Dictionary with all analysis results
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("🔬 Comprehensive Analysis Demonstration")
            print("=" * 60)

        # Run all demonstrations
        single_comparison = self.demonstrate_single_comparison()
        convergence_analysis = self.demonstrate_convergence_analysis()
        solution_ranking = self.demonstrate_solution_ranking()
        pareto_analysis = self.demonstrate_pareto_analysis()
        method_benchmark = self.demonstrate_method_benchmark()

        # Compile comprehensive results
        results = {
            "single_comparison": single_comparison,
            "convergence_analysis": convergence_analysis,
            "solution_ranking": solution_ranking,
            "pareto_analysis": pareto_analysis,
            "method_benchmark": method_benchmark,
            "summary": self._generate_comprehensive_summary(),
        }

        if self.verbose:
            print("\n🎉 Comprehensive Analysis Complete!")
            print("📊 Summary of capabilities demonstrated:")
            print("  ✅ Global vs Local optimization comparison")
            print("  ✅ Detailed convergence analysis")
            print("  ✅ Multi-criteria solution ranking")
            print("  ✅ Pareto front analysis")
            print("  ✅ Method benchmarking")
            print("  ✅ Solution quality assessment")

        return results

    def _simulate_global_optimization(self, test_params: jnp.ndarray) -> Dict[str, Any]:
        """Simulate global optimization result (PyGMO-style)."""
        # Add some noise to simulate global optimization
        global_solution = test_params * (1.0 + 0.05 * np.random.randn(3))
        global_objective = float(self.loss_function.compute_loss(global_solution))

        return {
            "x": np.array(global_solution),
            "fun": global_objective + 0.1 * abs(global_objective),  # Slightly worse
            "success": True,
            "nit": np.random.randint(500, 2000),
            "nfev": np.random.randint(5000, 20000),
            "time": np.random.uniform(10.0, 60.0),
            "message": "Global optimization completed",
        }

    def _print_comparison_results(self, comparison: ComparisonResult):
        """Print detailed comparison results."""
        print("\n📊 Comparison Results:")
        print(f"  Objective Improvement: {comparison.objective_improvement:.2f}%")
        print(f"  Convergence Improvement: {comparison.convergence_improvement:.2f}%")
        print(f"  Solution Quality: {comparison.solution_quality.value}")
        print(f"  Speedup Factor: {comparison.speedup_factor:.2f}x")
        print(f"  Efficiency Ratio: {comparison.efficiency_ratio:.2f}x")
        print(f"  Preferred Method: {comparison.preferred_method}")
        print(
            f"  Recommendation Confidence: {comparison.recommendation_confidence:.2f}"
        )

        if comparison.improvement_suggestions:
            print("  💡 Improvement Suggestions:")
            for suggestion in comparison.improvement_suggestions:
                print(f"    • {suggestion}")

        if comparison.component_analysis:
            print("  🔍 Component Analysis:")
            for key, value in comparison.component_analysis.items():
                print(f"    {key}: {value:.6e}")

    def _print_convergence_analysis(self, analysis: ConvergenceAnalysis):
        """Print detailed convergence analysis."""
        print("\n📈 Convergence Analysis:")
        print(f"  Convergence Rate: {analysis.convergence_rate:.6f}")
        print(f"  Final Gradient Norm: {analysis.final_gradient_norm:.6e}")
        print(f"  Iterations to Convergence: {analysis.iterations_to_convergence}")
        print(f"  Solution Stability: {analysis.solution_stability:.3f}")
        print(f"  Objective Variance: {analysis.objective_variance:.6e}")
        print(f"  Time to Convergence: {analysis.time_to_convergence:.3f}s")
        print(
            f"  Evaluations per Improvement: {analysis.evaluations_per_improvement:.1f}"
        )
        print(f"  Convergence Quality: {analysis.convergence_quality.value}")

        if analysis.stagnation_periods:
            print(f"  Stagnation Periods: {len(analysis.stagnation_periods)}")
            for i, (start, end) in enumerate(analysis.stagnation_periods[:3]):
                print(f"    Period {i+1}: iterations {start}-{end}")

    def _run_method_benchmark(
        self, test_problems: List[Dict], methods: List[str]
    ) -> Dict[str, Any]:
        """Run actual method benchmarking."""
        results = {
            "methods": methods,
            "problems": len(test_problems),
            "detailed_results": {},
            "summary": {},
        }

        for method in methods:
            method_results = {
                "success_rate": 0.0,
                "average_objective": 0.0,
                "average_time": 0.0,
                "average_iterations": 0.0,
                "problem_results": [],
            }

            all_objectives = []
            all_times = []
            all_iterations = []
            successes = 0

            for problem in test_problems:
                # Create optimizer with specific method
                optimizer = DifferentiableOptimizer(
                    objective_function=problem["objective_function"],
                    bounds=problem["bounds"],
                    method=method,
                    max_iterations=100,
                    verbose=False,
                )

                # Run optimization
                result = optimizer.optimize(problem["initial_point"])

                if result.success:
                    successes += 1
                    all_objectives.append(result.fun)
                    all_times.append(result.optimization_time)
                    all_iterations.append(result.nit)

            # Compute statistics
            total_runs = len(test_problems)
            method_results["success_rate"] = (
                successes / total_runs if total_runs > 0 else 0.0
            )
            method_results["average_objective"] = (
                np.mean(all_objectives) if all_objectives else float("inf")
            )
            method_results["average_time"] = np.mean(all_times) if all_times else 0.0
            method_results["average_iterations"] = (
                np.mean(all_iterations) if all_iterations else 0.0
            )

            results["detailed_results"][method] = method_results

        # Create summary
        best_method = min(
            methods, key=lambda m: results["detailed_results"][m]["average_objective"]
        )

        results["summary"] = {
            "best_method": best_method,
            "best_objective": results["detailed_results"][best_method][
                "average_objective"
            ],
            "method_rankings": sorted(
                [
                    (m, results["detailed_results"][m]["average_objective"])
                    for m in methods
                ],
                key=lambda x: x[1],
            ),
        }

        return results

    def _print_benchmark_results(self, benchmark_results: Dict[str, Any]):
        """Print benchmark results."""
        print("\n⚡ Benchmark Results:")
        print(f"  Methods tested: {len(benchmark_results['methods'])}")
        print(f"  Test problems: {benchmark_results['problems']}")

        print("\n📊 Method Performance:")
        for method, results in benchmark_results["detailed_results"].items():
            print(f"  {method}:")
            print(f"    Success Rate: {results['success_rate']:.1%}")
            print(f"    Average Objective: {results['average_objective']:.6e}")
            print(f"    Average Time: {results['average_time']:.3f}s")
            print(f"    Average Iterations: {results['average_iterations']:.1f}")

        print(f"\n🏆 Best Method: {benchmark_results['summary']['best_method']}")
        print(
            f"   Best Objective: {benchmark_results['summary']['best_objective']:.6e}"
        )

    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of demonstration."""
        return {
            "features_demonstrated": [
                "Global vs Local optimization comparison",
                "Convergence analysis with detailed metrics",
                "Multi-criteria solution ranking",
                "Pareto front analysis",
                "Method benchmarking",
                "Solution quality assessment",
            ],
            "comparison_metrics": [
                "Objective improvement",
                "Convergence rate",
                "Solution stability",
                "Computational efficiency",
                "Pareto dominance",
            ],
            "analysis_capabilities": [
                "Statistical significance testing",
                "Sensitivity analysis",
                "Robustness evaluation",
                "Performance profiling",
            ],
        }


def run_comparison_demo() -> bool:
    """
    Run the complete result comparison demonstration.

    Returns:
        True if demonstration completed successfully
    """
    try:
        demo = ComparisonDemonstration(verbose=True)
        _ = demo.demonstrate_comprehensive_analysis()
        return True
    except Exception as e:
        print(f"❌ Comparison demonstration failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the comparison demonstration when script is executed directly
    print("Running JAX Result Comparison Demonstration...")
    success = run_comparison_demo()
    print(f"\nComparison demo result: {'SUCCESS' if success else 'FAILED'}")
