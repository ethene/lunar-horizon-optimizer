"""Pareto analysis and result processing for global optimization.

This module provides analysis tools for multi-objective optimization results,
including Pareto front analysis, solution ranking, and visualization support.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results and analysis.

    This class encapsulates the complete results from a multi-objective
    optimization run, including Pareto solutions, statistics, and metadata.
    """

    pareto_solutions: list[dict[str, Any]]
    optimization_stats: dict[str, Any]
    problem_config: dict[str, Any]
    algorithm_config: dict[str, Any]
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # Ensure timestamp is set if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now()

    @property
    def num_pareto_solutions(self) -> int:
        """Number of Pareto-optimal solutions."""
        return len(self.pareto_solutions)

    @property
    def objective_ranges(self) -> dict[str, tuple[float, float]]:
        """Get min/max ranges for each objective."""
        if not self.pareto_solutions:
            return {}

        delta_v_values = [sol["objectives"]["delta_v"] for sol in self.pareto_solutions]
        time_values = [sol["objectives"]["time"] for sol in self.pareto_solutions]
        cost_values = [sol["objectives"]["cost"] for sol in self.pareto_solutions]

        return {
            "delta_v": (min(delta_v_values), max(delta_v_values)),
            "time": (min(time_values), max(time_values)),
            "cost": (min(cost_values), max(cost_values)),
        }

    def get_best_solutions(self, objective: str, num_solutions: int = 5) -> list[dict[str, Any]]:
        """Get best solutions for a specific objective.

        Args:
            objective: Objective name ('delta_v', 'time', 'cost')
            num_solutions: Number of solutions to return

        Returns
        -------
            List of best solutions for the specified objective
        """
        if objective not in ["delta_v", "time", "cost"]:
            msg = f"Invalid objective: {objective}"
            raise ValueError(msg)

        # Sort by objective value (ascending - all objectives are minimized)
        sorted_solutions = sorted(
            self.pareto_solutions,
            key=lambda sol: sol["objectives"][objective],
        )

        return sorted_solutions[:num_solutions]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pareto_solutions": self.pareto_solutions,
            "optimization_stats": self.optimization_stats,
            "problem_config": self.problem_config,
            "algorithm_config": self.algorithm_config,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "num_pareto_solutions": self.num_pareto_solutions,
            "objective_ranges": self.objective_ranges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary representation."""
        return cls(
            pareto_solutions=data["pareto_solutions"],
            optimization_stats=data["optimization_stats"],
            problem_config=data["problem_config"],
            algorithm_config=data["algorithm_config"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class ParetoAnalyzer:
    """Analysis tools for Pareto fronts and multi-objective optimization results.

    This class provides methods to analyze, rank, and process Pareto-optimal
    solutions from multi-objective optimization.
    """

    def __init__(self) -> None:
        """Initialize Pareto analyzer."""
        logger.info("Initialized ParetoAnalyzer")

    def analyze_pareto_front(self, optimization_result: dict[str, Any]) -> OptimizationResult:
        """Analyze optimization results and create structured result object.

        Args:
            optimization_result: Raw optimization results from GlobalOptimizer

        Returns
        -------
            Structured optimization result with analysis
        """
        pareto_solutions = optimization_result.get("pareto_solutions", [])

        # Calculate optimization statistics
        stats = self._calculate_optimization_stats(optimization_result)

        # Extract configuration information
        problem_config = self._extract_problem_config(optimization_result)
        algorithm_config = optimization_result.get("algorithm_info", {})

        result = OptimizationResult(
            pareto_solutions=pareto_solutions,
            optimization_stats=stats,
            problem_config=problem_config,
            algorithm_config=algorithm_config,
            timestamp=datetime.now(),
        )

        logger.info(f"Analyzed Pareto front: {result.num_pareto_solutions} solutions")
        return result

    def rank_solutions_by_preference(self,
                                   solutions: list[dict[str, Any]],
                                   preference_weights: list[float],
                                   normalization_method: str = "minmax") -> list[tuple[float, dict[str, Any]]]:
        """Rank solutions by user preferences using weighted objectives.

        Args:
            solutions: List of Pareto solutions
            preference_weights: Weights for [delta_v, time, cost] objectives
            normalization_method: Normalization method ('minmax', 'zscore')

        Returns
        -------
            List of (score, solution) tuples sorted by preference score
        """
        if len(preference_weights) != 3:
            msg = "preference_weights must have 3 elements for [delta_v, time, cost]"
            raise ValueError(msg)

        if not solutions:
            return []

        # Extract objective values
        objectives_matrix = np.array([
            [sol["objectives"]["delta_v"], sol["objectives"]["time"], sol["objectives"]["cost"]]
            for sol in solutions
        ])

        # Normalize objectives
        if normalization_method == "minmax":
            normalized_objectives = self._minmax_normalize(objectives_matrix)
        elif normalization_method == "zscore":
            normalized_objectives = self._zscore_normalize(objectives_matrix)
        else:
            msg = f"Unknown normalization method: {normalization_method}"
            raise ValueError(msg)

        # Calculate weighted scores
        weights = np.array(preference_weights)
        scores = np.dot(normalized_objectives, weights)

        # Create ranked list (lower scores are better)
        ranked_solutions = sorted(
            zip(scores, solutions, strict=False),
            key=lambda x: x[0],
        )

        logger.info(f"Ranked {len(solutions)} solutions by preference weights {preference_weights}")
        return ranked_solutions

    def find_knee_solutions(self, solutions: list[dict[str, Any]], num_knees: int = 3) -> list[dict[str, Any]]:
        """Find knee points in the Pareto front (best trade-off solutions).

        Args:
            solutions: List of Pareto solutions
            num_knees: Number of knee points to find

        Returns
        -------
            List of knee point solutions
        """
        if len(solutions) < 3:
            return solutions

        # Extract normalized objective values
        objectives_matrix = np.array([
            [sol["objectives"]["delta_v"], sol["objectives"]["time"], sol["objectives"]["cost"]]
            for sol in solutions
        ])

        normalized_objectives = self._minmax_normalize(objectives_matrix)

        # Calculate knee points using perpendicular distance method
        knee_indices = self._find_knee_points(normalized_objectives, num_knees)

        knee_solutions = [solutions[i] for i in knee_indices]

        logger.info(f"Found {len(knee_solutions)} knee point solutions")
        return knee_solutions

    def compare_optimization_runs(self,
                                results: list[OptimizationResult]) -> dict[str, Any]:
        """Compare multiple optimization runs.

        Args:
            results: List of optimization results to compare

        Returns
        -------
            Comparison analysis
        """
        if not results:
            return {}

        comparison: dict[str, Any] = {
            "num_runs": len(results),
            "pareto_sizes": [r.num_pareto_solutions for r in results],
            "objective_ranges_comparison": {},
            "convergence_analysis": {},
            "best_solutions_overall": {},
        }

        # Compare objective ranges across runs
        for objective in ["delta_v", "time", "cost"]:
            ranges = [r.objective_ranges.get(objective, (float("inf"), float("-inf")))
                     for r in results]
            comparison["objective_ranges_comparison"][objective] = {
                "min_values": [r[0] for r in ranges],
                "max_values": [r[1] for r in ranges],
                "overall_min": min(r[0] for r in ranges if r[0] != float("inf")),
                "overall_max": max(r[1] for r in ranges if r[1] != float("-inf")),
            }

        # Find overall best solutions
        all_solutions = []
        for result in results:
            all_solutions.extend(result.pareto_solutions)

        if all_solutions:
            for objective in ["delta_v", "time", "cost"]:
                best_sol = min(all_solutions, key=lambda sol: sol["objectives"][objective])
                comparison["best_solutions_overall"][objective] = best_sol

        logger.info(f"Compared {len(results)} optimization runs")
        return comparison

    def calculate_hypervolume(self,
                            solutions: list[dict[str, Any]],
                            reference_point: list[float] | None = None) -> float:
        """Calculate hypervolume indicator for Pareto front quality.

        Args:
            solutions: List of Pareto solutions
            reference_point: Reference point for hypervolume calculation

        Returns
        -------
            Hypervolume value
        """
        if not solutions:
            return 0.0

        # Extract objective values
        objectives = np.array([
            [sol["objectives"]["delta_v"], sol["objectives"]["time"], sol["objectives"]["cost"]]
            for sol in solutions
        ])

        # Use worst point as reference if not provided
        if reference_point is None:
            reference_point_array = np.max(objectives, axis=0) * 1.1  # 10% worse than worst
        else:
            reference_point_array = np.array(reference_point)

        # Simple hypervolume calculation for 3D case
        # This is a simplified implementation - for production use, consider pygmo's hypervolume
        hypervolume = self._calculate_hypervolume_3d(objectives, reference_point_array)

        logger.info(f"Calculated hypervolume: {hypervolume:.2e}")
        return hypervolume

    def export_results(self, result: OptimizationResult, filepath: str) -> None:
        """Export optimization results to JSON file.

        Args:
            result: Optimization result to export
            filepath: Output file path
        """
        try:
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            logger.info(f"Exported optimization results to {filepath}")
        except Exception as e:
            logger.exception(f"Failed to export results: {e}")
            raise

    def _calculate_optimization_stats(self, optimization_result: dict[str, Any]) -> dict[str, Any]:
        """Calculate optimization statistics."""
        pareto_solutions = optimization_result.get("pareto_solutions", [])

        stats = {
            "num_pareto_solutions": len(pareto_solutions),
            "convergence_info": optimization_result.get("optimization_history", []),
            "cache_efficiency": optimization_result.get("cache_stats", {}),
            "algorithm_performance": {
                "generations": optimization_result.get("generations", 0),
                "population_size": optimization_result.get("population_size", 0),
            },
        }

        if pareto_solutions:
            # Calculate objective statistics
            delta_v_values = [sol["objectives"]["delta_v"] for sol in pareto_solutions]
            time_values = [sol["objectives"]["time"] for sol in pareto_solutions]
            cost_values = [sol["objectives"]["cost"] for sol in pareto_solutions]

            stats["objective_statistics"] = {
                "delta_v": {"min": min(delta_v_values), "max": max(delta_v_values), "mean": np.mean(delta_v_values)},
                "time": {"min": min(time_values), "max": max(time_values), "mean": np.mean(time_values)},
                "cost": {"min": min(cost_values), "max": max(cost_values), "mean": np.mean(cost_values)},
            }

        return stats

    def _extract_problem_config(self, optimization_result: dict[str, Any]) -> dict[str, Any]:
        """Extract problem configuration from optimization result."""
        return {
            "objectives": ["delta_v", "time", "cost"],
            "problem_type": "lunar_mission_optimization",
            "optimization_type": "multi_objective",
        }

    def _minmax_normalize(self, objectives: np.ndarray[np.float64, np.dtype[np.float64]]) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        """Min-max normalization of objectives."""
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)

        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        normalized_result: np.ndarray[np.float64, np.dtype[np.float64]] = (objectives - min_vals) / ranges
        return normalized_result

    def _zscore_normalize(self, objectives: np.ndarray[np.float64, np.dtype[np.float64]]) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        """Z-score normalization of objectives."""
        means = np.mean(objectives, axis=0)
        stds = np.std(objectives, axis=0)

        # Avoid division by zero
        stds[stds == 0] = 1.0

        normalized_result: np.ndarray[np.float64, np.dtype[np.float64]] = (objectives - means) / stds
        return normalized_result

    def _find_knee_points(self, normalized_objectives: np.ndarray[np.float64, np.dtype[np.float64]], num_knees: int) -> list[int]:
        """Find knee points using perpendicular distance method."""
        if len(normalized_objectives) <= num_knees:
            return list(range(len(normalized_objectives)))

        # Simple knee point detection using angle method
        distances = []
        for i in range(len(normalized_objectives)):
            # Calculate distance from origin
            distance = np.linalg.norm(normalized_objectives[i])
            distances.append((distance, i))

        # Sort by distance and select evenly distributed points
        distances.sort()
        step = len(distances) // num_knees
        return [distances[i * step][1] for i in range(num_knees)]


    def _calculate_hypervolume_3d(self, objectives: np.ndarray[np.float64, np.dtype[np.float64]], reference_point: np.ndarray[np.float64, np.dtype[np.float64]]) -> float:
        """Simple hypervolume calculation for 3D objectives."""
        # This is a simplified implementation
        # For production, consider using pygmo.hypervolume or similar

        # Sort objectives by first dimension
        sorted_objectives = objectives[np.argsort(objectives[:, 0])]

        volume = 0.0
        prev_point = reference_point.copy()

        for point in sorted_objectives:
            if np.all(point <= reference_point):
                # Calculate contribution
                contribution = np.prod(prev_point - point)
                volume += max(0, contribution)

                # Update previous point
                prev_point = np.minimum(prev_point, point)

        return volume


def create_pareto_analyzer() -> ParetoAnalyzer:
    """Create a Pareto analyzer instance.

    Returns
    -------
        Configured ParetoAnalyzer instance
    """
    return ParetoAnalyzer()
