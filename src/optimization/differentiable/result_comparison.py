"""
Result Comparison and Evaluation Module for Differentiable Optimization

This module provides comprehensive tools for comparing and evaluating optimization
results between global (PyGMO) and local (JAX) optimization methods, including
solution quality assessment, convergence analysis, and performance metrics.

Features:
- Multi-objective solution comparison and ranking
- Convergence analysis and visualization
- Performance benchmarking across optimization methods
- Solution quality metrics and statistical analysis
- Pareto front analysis for multi-objective optimization
- Robustness and sensitivity analysis

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

# JAX imports
try:
    import jax.numpy as jnp
    from jax import grad

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    grad = None

# Local imports
from .jax_optimizer import OptimizationResult
from .differentiable_models import TrajectoryModel, EconomicModel
from .loss_functions import MultiObjectiveLoss


class ComparisonMetric(Enum):
    """Enumeration of comparison metrics for optimization results."""

    OBJECTIVE_VALUE = "objective_value"
    CONVERGENCE_RATE = "convergence_rate"
    SOLUTION_QUALITY = "solution_quality"
    ROBUSTNESS = "robustness"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    PARETO_DOMINANCE = "pareto_dominance"


class SolutionQuality(Enum):
    """Enumeration of solution quality categories."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ComparisonResult:
    """Container for optimization comparison results."""

    # Basic comparison metrics
    global_result: Dict[str, Any]
    local_result: OptimizationResult

    # Quality metrics
    objective_improvement: float
    convergence_improvement: float
    solution_quality: SolutionQuality

    # Performance metrics
    speedup_factor: float
    efficiency_ratio: float

    # Statistical analysis
    statistical_significance: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

    # Detailed analysis
    component_analysis: Dict[str, float] = field(default_factory=dict)
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    preferred_method: str = ""
    recommendation_confidence: float = 0.0
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class ConvergenceAnalysis:
    """Container for convergence analysis results."""

    # Convergence metrics
    convergence_rate: float
    final_gradient_norm: float
    iterations_to_convergence: int

    # Stability metrics
    solution_stability: float
    objective_variance: float

    # Performance metrics
    time_to_convergence: float
    evaluations_per_improvement: float

    # Convergence quality
    convergence_quality: SolutionQuality
    stagnation_periods: List[Tuple[int, int]] = field(default_factory=list)

    # Convergence history
    objective_history: List[float] = field(default_factory=list)
    gradient_history: List[float] = field(default_factory=list)
    step_size_history: List[float] = field(default_factory=list)


class ResultComparator:
    """
    Advanced result comparison and evaluation system.

    This class provides comprehensive tools for comparing optimization results
    from different methods and analyzing solution quality and performance.
    """

    def __init__(
        self,
        trajectory_model: Optional[TrajectoryModel] = None,
        economic_model: Optional[EconomicModel] = None,
        loss_function: Optional[MultiObjectiveLoss] = None,
    ):
        """
        Initialize result comparator.

        Args:
            trajectory_model: JAX trajectory model for analysis
            economic_model: JAX economic model for analysis
            loss_function: Multi-objective loss function
        """
        self.trajectory_model = trajectory_model
        self.economic_model = economic_model
        self.loss_function = loss_function

        # Analysis configuration
        self.quality_thresholds = {
            SolutionQuality.EXCELLENT: 0.95,
            SolutionQuality.GOOD: 0.80,
            SolutionQuality.ACCEPTABLE: 0.60,
            SolutionQuality.POOR: 0.40,
        }

        self.comparison_history = []
        self.benchmark_results = {}

    def compare_optimization_results(
        self,
        global_result: Dict[str, Any],
        local_result: OptimizationResult,
        reference_solution: Optional[np.ndarray] = None,
    ) -> ComparisonResult:
        """
        Compare global and local optimization results comprehensively.

        Args:
            global_result: Result from global optimization (PyGMO)
            local_result: Result from local optimization (JAX)
            reference_solution: Optional reference solution for comparison

        Returns:
            ComparisonResult with detailed comparison analysis
        """
        # Extract key metrics
        global_objective = global_result.get("fun", float("inf"))
        local_objective = local_result.fun

        # Compute objective improvement
        objective_improvement = self._compute_objective_improvement(
            global_objective, local_objective
        )

        # Analyze convergence
        convergence_improvement = self._analyze_convergence_improvement(
            global_result, local_result
        )

        # Assess solution quality
        solution_quality = self._assess_solution_quality(
            local_result, reference_solution
        )

        # Compute performance metrics
        speedup_factor = self._compute_speedup_factor(global_result, local_result)
        efficiency_ratio = self._compute_efficiency_ratio(global_result, local_result)

        # Perform component analysis
        component_analysis = self._perform_component_analysis(
            global_result.get("x", np.array([])), local_result.x
        )

        # Perform sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(local_result.x)

        # Generate recommendations
        preferred_method, confidence, suggestions = self._generate_recommendations(
            global_result, local_result, objective_improvement
        )

        # Create comparison result
        comparison = ComparisonResult(
            global_result=global_result,
            local_result=local_result,
            objective_improvement=objective_improvement,
            convergence_improvement=convergence_improvement,
            solution_quality=solution_quality,
            speedup_factor=speedup_factor,
            efficiency_ratio=efficiency_ratio,
            component_analysis=component_analysis,
            sensitivity_analysis=sensitivity_analysis,
            preferred_method=preferred_method,
            recommendation_confidence=confidence,
            improvement_suggestions=suggestions,
        )

        # Store in history
        self.comparison_history.append(comparison)

        return comparison

    def analyze_convergence(
        self, optimization_result: OptimizationResult, detailed_analysis: bool = True
    ) -> ConvergenceAnalysis:
        """
        Perform detailed convergence analysis for optimization result.

        Args:
            optimization_result: Result from optimization
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            ConvergenceAnalysis with convergence metrics
        """
        history = optimization_result.convergence_history

        if not history:
            return ConvergenceAnalysis(
                convergence_rate=0.0,
                final_gradient_norm=float("inf"),
                iterations_to_convergence=0,
                solution_stability=0.0,
                objective_variance=0.0,
                time_to_convergence=optimization_result.optimization_time,
                evaluations_per_improvement=0.0,
                convergence_quality=SolutionQuality.FAILED,
            )

        # Basic convergence metrics
        convergence_rate = self._compute_convergence_rate(history)
        final_gradient_norm = (
            optimization_result.gradient_norms[-1]
            if optimization_result.gradient_norms
            else float("inf")
        )
        iterations_to_convergence = self._find_convergence_point(history)

        # Stability metrics
        solution_stability = self._compute_solution_stability(history)
        objective_variance = float(np.var(history)) if len(history) > 1 else 0.0

        # Performance metrics
        time_to_convergence = optimization_result.optimization_time
        evaluations_per_improvement = optimization_result.nfev / max(1, len(history))

        # Assess convergence quality
        convergence_quality = self._assess_convergence_quality(
            convergence_rate, final_gradient_norm, solution_stability
        )

        # Detailed analysis
        stagnation_periods = []
        step_size_history = []

        if detailed_analysis:
            stagnation_periods = self._find_stagnation_periods(history)
            step_size_history = self._estimate_step_sizes(history)

        return ConvergenceAnalysis(
            convergence_rate=convergence_rate,
            final_gradient_norm=final_gradient_norm,
            iterations_to_convergence=iterations_to_convergence,
            solution_stability=solution_stability,
            objective_variance=objective_variance,
            time_to_convergence=time_to_convergence,
            evaluations_per_improvement=evaluations_per_improvement,
            convergence_quality=convergence_quality,
            stagnation_periods=stagnation_periods,
            objective_history=history,
            gradient_history=optimization_result.gradient_norms,
            step_size_history=step_size_history,
        )

    def rank_solutions(
        self, results: List[OptimizationResult], criteria: List[ComparisonMetric] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank optimization solutions based on multiple criteria.

        Args:
            results: List of optimization results to rank
            criteria: List of comparison criteria to use

        Returns:
            List of (index, score) tuples sorted by ranking
        """
        if not results:
            return []

        if criteria is None:
            criteria = [
                ComparisonMetric.OBJECTIVE_VALUE,
                ComparisonMetric.CONVERGENCE_RATE,
                ComparisonMetric.SOLUTION_QUALITY,
            ]

        rankings = []

        for i, result in enumerate(results):
            score = 0.0
            weight_sum = 0.0

            for criterion in criteria:
                weight = self._get_criterion_weight(criterion)
                criterion_score = self._evaluate_criterion(result, criterion)

                score += weight * criterion_score
                weight_sum += weight

            # Normalize score
            if weight_sum > 0:
                score /= weight_sum

            rankings.append((i, score))

        # Sort by score (higher is better)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def compute_pareto_front(
        self, results: List[OptimizationResult], objectives: List[str] = None
    ) -> List[int]:
        """
        Compute Pareto front for multi-objective optimization results.

        Args:
            results: List of optimization results
            objectives: List of objective names to consider

        Returns:
            List of indices of non-dominated solutions
        """
        if not results:
            return []

        if objectives is None:
            objectives = ["fun"]  # Default to single objective

        # Extract objective values
        objective_matrix = []
        for result in results:
            obj_values = []
            for obj_name in objectives:
                if obj_name == "fun":
                    obj_values.append(result.fun)
                elif obj_name in result.objective_components:
                    obj_values.append(result.objective_components[obj_name])
                else:
                    obj_values.append(0.0)
            objective_matrix.append(obj_values)

        objective_matrix = np.array(objective_matrix)

        # Find non-dominated solutions
        is_pareto = np.ones(len(results), dtype=bool)

        for i in range(len(results)):
            if is_pareto[i]:
                for j in range(i + 1, len(results)):
                    if is_pareto[j]:
                        if self._dominates(objective_matrix[i], objective_matrix[j]):
                            is_pareto[j] = False
                        elif self._dominates(objective_matrix[j], objective_matrix[i]):
                            is_pareto[i] = False
                            break

        return np.where(is_pareto)[0].tolist()

    def benchmark_methods(
        self,
        test_problems: List[Dict[str, Any]],
        methods: List[str],
        iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark different optimization methods on test problems.

        Args:
            test_problems: List of test problem definitions
            methods: List of optimization methods to benchmark
            iterations: Number of iterations per method/problem

        Returns:
            Dictionary with benchmark results
        """
        results = {
            "methods": methods,
            "problems": len(test_problems),
            "iterations": iterations,
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

            for problem_idx, problem in enumerate(test_problems):
                problem_results = {
                    "problem_index": problem_idx,
                    "successes": 0,
                    "objectives": [],
                    "times": [],
                    "iterations": [],
                }

                for _ in range(iterations):
                    # This is a placeholder - actual implementation would
                    # run the optimization method on the test problem
                    result = self._run_benchmark_iteration(method, problem)

                    if result.get("success", False):
                        problem_results["successes"] += 1
                        problem_results["objectives"].append(result.get("fun", 0.0))
                        problem_results["times"].append(result.get("time", 0.0))
                        problem_results["iterations"].append(result.get("nit", 0))

                method_results["problem_results"].append(problem_results)

            # Compute summary statistics
            all_objectives = []
            all_times = []
            all_iterations = []
            total_successes = 0
            total_attempts = 0

            for prob_result in method_results["problem_results"]:
                all_objectives.extend(prob_result["objectives"])
                all_times.extend(prob_result["times"])
                all_iterations.extend(prob_result["iterations"])
                total_successes += prob_result["successes"]
                total_attempts += iterations

            method_results["success_rate"] = (
                total_successes / total_attempts if total_attempts > 0 else 0.0
            )
            method_results["average_objective"] = (
                np.mean(all_objectives) if all_objectives else 0.0
            )
            method_results["average_time"] = np.mean(all_times) if all_times else 0.0
            method_results["average_iterations"] = (
                np.mean(all_iterations) if all_iterations else 0.0
            )

            results["detailed_results"][method] = method_results

        # Create summary comparison
        results["summary"] = self._create_benchmark_summary(results["detailed_results"])

        return results

    def _compute_objective_improvement(
        self, global_obj: float, local_obj: float
    ) -> float:
        """Compute objective improvement percentage."""
        if global_obj == 0 or np.isinf(global_obj):
            return 0.0
        return 100.0 * (global_obj - local_obj) / abs(global_obj)

    def _analyze_convergence_improvement(
        self, global_result: Dict, local_result: OptimizationResult
    ) -> float:
        """Analyze convergence improvement."""
        global_time = global_result.get("time", 0.0)
        local_time = local_result.optimization_time

        if global_time == 0 or local_time == 0:
            return 0.0

        # Simple convergence improvement based on time and iterations
        global_iter = global_result.get("nit", 0)
        local_iter = local_result.nit

        if global_iter == 0 or local_iter == 0:
            return 0.0

        convergence_rate_global = 1.0 / global_iter
        convergence_rate_local = 1.0 / local_iter

        return (
            100.0
            * (convergence_rate_local - convergence_rate_global)
            / convergence_rate_global
        )

    def _assess_solution_quality(
        self, result: OptimizationResult, reference: Optional[np.ndarray] = None
    ) -> SolutionQuality:
        """Assess solution quality based on multiple criteria."""
        if not result.success:
            return SolutionQuality.FAILED

        # Compute quality score based on multiple factors
        quality_score = 0.0

        # Success factor
        if result.success:
            quality_score += 0.3

        # Convergence factor
        if result.gradient_norms and result.gradient_norms[-1] < 1e-6:
            quality_score += 0.3
        elif result.gradient_norms and result.gradient_norms[-1] < 1e-3:
            quality_score += 0.2

        # Improvement factor
        if result.improvement_percentage and result.improvement_percentage > 10:
            quality_score += 0.2
        elif result.improvement_percentage and result.improvement_percentage > 1:
            quality_score += 0.1

        # Efficiency factor
        if result.optimization_time < 1.0:
            quality_score += 0.2
        elif result.optimization_time < 10.0:
            quality_score += 0.1

        # Classify quality
        if quality_score >= self.quality_thresholds[SolutionQuality.EXCELLENT]:
            return SolutionQuality.EXCELLENT
        elif quality_score >= self.quality_thresholds[SolutionQuality.GOOD]:
            return SolutionQuality.GOOD
        elif quality_score >= self.quality_thresholds[SolutionQuality.ACCEPTABLE]:
            return SolutionQuality.ACCEPTABLE
        else:
            return SolutionQuality.POOR

    def _compute_speedup_factor(
        self, global_result: Dict, local_result: OptimizationResult
    ) -> float:
        """Compute speedup factor."""
        global_time = global_result.get("time", 0.0)
        local_time = local_result.optimization_time

        if local_time == 0:
            return 1.0
        return global_time / local_time

    def _compute_efficiency_ratio(
        self, global_result: Dict, local_result: OptimizationResult
    ) -> float:
        """Compute efficiency ratio."""
        global_feval = global_result.get("nfev", 0)
        local_feval = local_result.nfev

        if local_feval == 0:
            return 1.0
        return global_feval / local_feval

    def _perform_component_analysis(
        self, global_x: np.ndarray, local_x: np.ndarray
    ) -> Dict[str, float]:
        """Perform component-wise analysis of solutions."""
        if len(global_x) == 0 or len(local_x) == 0:
            return {}

        if len(global_x) != len(local_x):
            return {"error": "Dimension mismatch"}

        analysis = {}

        # Parameter differences
        diff = local_x - global_x
        analysis["parameter_difference_norm"] = float(np.linalg.norm(diff))
        analysis["max_parameter_change"] = float(np.max(np.abs(diff)))
        analysis["mean_parameter_change"] = float(np.mean(np.abs(diff)))

        # Relative changes
        rel_changes = np.abs(diff) / (np.abs(global_x) + 1e-12)
        analysis["max_relative_change"] = float(np.max(rel_changes))
        analysis["mean_relative_change"] = float(np.mean(rel_changes))

        return analysis

    def _perform_sensitivity_analysis(self, solution: np.ndarray) -> Dict[str, float]:
        """Perform sensitivity analysis around solution."""
        if self.loss_function is None:
            return {}

        try:
            # Compute gradient at solution
            if JAX_AVAILABLE:
                grad_fn = grad(self.loss_function.compute_loss)
                gradient = grad_fn(solution)

                sensitivity = {}
                sensitivity["gradient_norm"] = float(np.linalg.norm(gradient))
                sensitivity["max_gradient_component"] = float(np.max(np.abs(gradient)))
                sensitivity["mean_gradient_component"] = float(
                    np.mean(np.abs(gradient))
                )

                return sensitivity
            else:
                return {"error": "JAX not available for sensitivity analysis"}
        except Exception as e:
            return {"error": f"Sensitivity analysis failed: {str(e)}"}

    def _generate_recommendations(
        self, global_result: Dict, local_result: OptimizationResult, improvement: float
    ) -> Tuple[str, float, List[str]]:
        """Generate optimization recommendations."""
        suggestions = []

        # Determine preferred method
        if improvement > 5.0 and local_result.success:
            preferred = "local"
            confidence = 0.8
            suggestions.append("Local optimization provides significant improvement")
        elif improvement > 0.0 and local_result.success:
            preferred = "local"
            confidence = 0.6
            suggestions.append("Local optimization provides modest improvement")
        elif local_result.success:
            preferred = "hybrid"
            confidence = 0.5
            suggestions.append("Consider hybrid approach combining both methods")
        else:
            preferred = "global"
            confidence = 0.7
            suggestions.append("Global optimization is more reliable for this problem")

        # Performance suggestions
        if local_result.optimization_time > 10.0:
            suggestions.append(
                "Consider reducing optimization tolerance for faster convergence"
            )

        if local_result.nit > 500:
            suggestions.append(
                "High iteration count - consider different optimization method"
            )

        if not local_result.success:
            suggestions.append(
                "Local optimization failed - check initial conditions and constraints"
            )

        return preferred, confidence, suggestions

    def _compute_convergence_rate(self, history: List[float]) -> float:
        """Compute convergence rate from objective history."""
        if len(history) < 2:
            return 0.0

        # Linear convergence rate (simplified)
        improvements = []
        for i in range(1, len(history)):
            if history[i - 1] != 0:
                improvement = abs(history[i] - history[i - 1]) / abs(history[i - 1])
                improvements.append(improvement)

        if not improvements:
            return 0.0

        return float(np.mean(improvements))

    def _find_convergence_point(self, history: List[float]) -> int:
        """Find iteration where convergence occurred."""
        if len(history) < 2:
            return 0

        # Find when relative improvement becomes small
        tolerance = 1e-6
        for i in range(1, len(history)):
            if history[i - 1] != 0:
                rel_improvement = abs(history[i] - history[i - 1]) / abs(history[i - 1])
                if rel_improvement < tolerance:
                    return i

        return len(history)

    def _compute_solution_stability(self, history: List[float]) -> float:
        """Compute solution stability metric."""
        if len(history) < 10:
            return 0.0

        # Look at last 10% of iterations
        final_portion = history[-max(10, len(history) // 10) :]

        if len(final_portion) < 2:
            return 0.0

        # Compute coefficient of variation
        mean_val = np.mean(final_portion)
        std_val = np.std(final_portion)

        if abs(mean_val) < 1e-12:
            return 1.0 if std_val < 1e-12 else 0.0

        cv = std_val / abs(mean_val)

        # Convert to stability score (0-1)
        return max(0.0, 1.0 - cv)

    def _assess_convergence_quality(
        self, rate: float, grad_norm: float, stability: float
    ) -> SolutionQuality:
        """Assess overall convergence quality."""
        # Combine multiple factors
        quality_score = 0.0

        # Convergence rate factor
        if rate > 0.01:
            quality_score += 0.3
        elif rate > 0.001:
            quality_score += 0.2

        # Gradient norm factor
        if grad_norm < 1e-6:
            quality_score += 0.4
        elif grad_norm < 1e-3:
            quality_score += 0.3
        elif grad_norm < 1e-1:
            quality_score += 0.2

        # Stability factor
        quality_score += 0.3 * stability

        # Classify quality
        if quality_score >= 0.8:
            return SolutionQuality.EXCELLENT
        elif quality_score >= 0.6:
            return SolutionQuality.GOOD
        elif quality_score >= 0.4:
            return SolutionQuality.ACCEPTABLE
        else:
            return SolutionQuality.POOR

    def _find_stagnation_periods(self, history: List[float]) -> List[Tuple[int, int]]:
        """Find periods of stagnation in optimization."""
        if len(history) < 5:
            return []

        stagnation_periods = []
        tolerance = 1e-8
        min_period_length = 5

        current_start = None

        for i in range(1, len(history)):
            if history[i - 1] != 0:
                rel_change = abs(history[i] - history[i - 1]) / abs(history[i - 1])

                if rel_change < tolerance:
                    if current_start is None:
                        current_start = i - 1
                else:
                    if current_start is not None:
                        period_length = i - current_start
                        if period_length >= min_period_length:
                            stagnation_periods.append((current_start, i - 1))
                        current_start = None

        # Handle final stagnation period
        if current_start is not None:
            period_length = len(history) - current_start
            if period_length >= min_period_length:
                stagnation_periods.append((current_start, len(history) - 1))

        return stagnation_periods

    def _estimate_step_sizes(self, history: List[float]) -> List[float]:
        """Estimate step sizes from objective history."""
        if len(history) < 2:
            return []

        # Simple step size estimation based on objective changes
        step_sizes = []
        for i in range(1, len(history)):
            step_size = abs(history[i] - history[i - 1])
            step_sizes.append(step_size)

        return step_sizes

    def _get_criterion_weight(self, criterion: ComparisonMetric) -> float:
        """Get weight for comparison criterion."""
        weights = {
            ComparisonMetric.OBJECTIVE_VALUE: 0.4,
            ComparisonMetric.CONVERGENCE_RATE: 0.2,
            ComparisonMetric.SOLUTION_QUALITY: 0.2,
            ComparisonMetric.ROBUSTNESS: 0.1,
            ComparisonMetric.COMPUTATIONAL_EFFICIENCY: 0.1,
        }
        return weights.get(criterion, 0.1)

    def _evaluate_criterion(
        self, result: OptimizationResult, criterion: ComparisonMetric
    ) -> float:
        """Evaluate specific criterion for result."""
        if criterion == ComparisonMetric.OBJECTIVE_VALUE:
            # Lower objective is better (minimize)
            return 1.0 / (1.0 + result.fun) if result.fun > 0 else 1.0
        elif criterion == ComparisonMetric.CONVERGENCE_RATE:
            # Faster convergence is better
            return 1.0 / (1.0 + result.nit) if result.nit > 0 else 1.0
        elif criterion == ComparisonMetric.SOLUTION_QUALITY:
            # Success is better
            return 1.0 if result.success else 0.0
        elif criterion == ComparisonMetric.COMPUTATIONAL_EFFICIENCY:
            # Less time is better
            return (
                1.0 / (1.0 + result.optimization_time)
                if result.optimization_time > 0
                else 1.0
            )
        else:
            return 0.5  # Default neutral score

    def _dominates(self, solution1: np.ndarray, solution2: np.ndarray) -> bool:
        """Check if solution1 dominates solution2 (for minimization)."""
        return np.all(solution1 <= solution2) and np.any(solution1 < solution2)

    def _run_benchmark_iteration(
        self, method: str, problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single benchmark iteration (placeholder)."""
        # This is a placeholder - actual implementation would run optimization
        return {
            "success": True,
            "fun": np.random.uniform(0, 100),
            "time": np.random.uniform(0.1, 10.0),
            "nit": np.random.randint(10, 1000),
        }

    def _create_benchmark_summary(
        self, detailed_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create summary of benchmark results."""
        summary = {"best_method": "", "best_objective": float("inf"), "rankings": []}

        rankings = []
        for method, results in detailed_results.items():
            score = (
                results["success_rate"] * 0.4
                + (1.0 / (1.0 + results["average_objective"])) * 0.3
                + (1.0 / (1.0 + results["average_time"])) * 0.3
            )
            rankings.append((method, score))

        rankings.sort(key=lambda x: x[1], reverse=True)

        summary["rankings"] = rankings
        if rankings:
            summary["best_method"] = rankings[0][0]
            summary["best_objective"] = detailed_results[rankings[0][0]][
                "average_objective"
            ]

        return summary


# Utility functions for result comparison
def compare_single_run(
    global_result: Dict[str, Any], local_result: OptimizationResult
) -> ComparisonResult:
    """
    Compare single optimization run results.

    Args:
        global_result: Global optimization result
        local_result: Local optimization result

    Returns:
        ComparisonResult with analysis
    """
    comparator = ResultComparator()
    return comparator.compare_optimization_results(global_result, local_result)


def analyze_optimization_convergence(result: OptimizationResult) -> ConvergenceAnalysis:
    """
    Analyze convergence of optimization result.

    Args:
        result: Optimization result to analyze

    Returns:
        ConvergenceAnalysis with detailed metrics
    """
    comparator = ResultComparator()
    return comparator.analyze_convergence(result)


def rank_optimization_results(
    results: List[OptimizationResult],
) -> List[Tuple[int, float]]:
    """
    Rank optimization results by quality.

    Args:
        results: List of optimization results

    Returns:
        List of (index, score) tuples
    """
    comparator = ResultComparator()
    return comparator.rank_solutions(results)


def evaluate_solution_quality(result: OptimizationResult) -> SolutionQuality:
    """
    Evaluate solution quality of optimization result.

    Args:
        result: Optimization result

    Returns:
        SolutionQuality enum value
    """
    comparator = ResultComparator()
    return comparator._assess_solution_quality(result)
