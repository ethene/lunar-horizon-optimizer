"""
PyGMO Integration Module

This module provides seamless integration between PyGMO global optimization
and JAX differentiable local optimization, enabling hybrid optimization
workflows that combine the best of both approaches.

Features:
- PyGMO population initialization from global optimization results
- JAX local refinement of PyGMO solutions
- Hybrid optimization workflows (global → local)
- Solution comparison and validation
- Performance benchmarking between methods
- Multi-start local optimization from diverse global solutions

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import time
import numpy as np

# JAX imports
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# PyGMO imports
try:
    import pygmo as pg

    PYGMO_AVAILABLE = True
except ImportError:
    PYGMO_AVAILABLE = False

# Local imports
from .jax_optimizer import DifferentiableOptimizer, OptimizationResult
from .differentiable_models import TrajectoryModel, EconomicModel
from .loss_functions import MultiObjectiveLoss
from .constraints import ConstraintHandler


@dataclass
class HybridOptimizationConfig:
    """Configuration for hybrid PyGMO-JAX optimization."""

    # Global optimization configuration
    pygmo_algorithm: str = "nsga2"
    pygmo_population_size: int = 100
    pygmo_generations: int = 50

    # Local optimization configuration
    local_method: str = "L-BFGS-B"
    local_max_iterations: int = 100
    local_tolerance: float = 1e-6

    # Hybrid workflow configuration
    num_local_starts: int = 10
    selection_strategy: str = "best_diverse"  # "best", "diverse", "best_diverse"
    diversity_threshold: float = 0.1

    # Performance configuration
    time_limit_global: Optional[float] = None
    time_limit_local: Optional[float] = None
    parallel_local: bool = False

    # Validation configuration
    validate_solutions: bool = True
    comparison_metrics: List[str] = field(
        default_factory=lambda: [
            "objective_value",
            "constraint_violation",
            "feasibility",
        ]
    )


@dataclass
class SolutionComparison:
    """Container for comparing optimization solutions."""

    global_solution: Dict[str, Any]
    local_solution: Dict[str, Any]
    improvement_percentage: float
    constraint_improvement: float
    optimization_times: Dict[str, float]
    convergence_info: Dict[str, Any]


class PyGMOProblem:
    """
    PyGMO-compatible problem wrapper for JAX differentiable models.

    This class wraps JAX trajectory and economic models to be compatible
    with PyGMO's optimization interface.
    """

    def __init__(
        self,
        trajectory_model: TrajectoryModel,
        economic_model: EconomicModel,
        loss_function: MultiObjectiveLoss,
        constraint_handler: Optional[ConstraintHandler] = None,
    ):
        """
        Initialize PyGMO problem wrapper.

        Args:
            trajectory_model: JAX trajectory model
            economic_model: JAX economic model
            loss_function: Multi-objective loss function
            constraint_handler: Optional constraint handler
        """
        if not PYGMO_AVAILABLE:
            raise ImportError("PyGMO is required for global optimization integration")

        self.trajectory_model = trajectory_model
        self.economic_model = economic_model
        self.loss_function = loss_function
        self.constraint_handler = constraint_handler

        # Problem dimensions
        self.n_objectives = 1  # Single objective for now (can be extended)
        self.n_constraints = 0
        if constraint_handler:
            # Count active constraints
            dummy_params = jnp.array([6.778e6, 1.937e6, 4.5 * 24 * 3600])
            constraints = constraint_handler.compute_constraint_values(dummy_params)
            self.n_constraints = len(constraints)

    def fitness(self, x: List[float]) -> List[float]:
        """
        Compute fitness for PyGMO optimization.

        Args:
            x: Parameter vector [r1, r2, tof]

        Returns:
            List containing objective value(s) and constraint violations
        """
        try:
            # Convert to JAX array
            params = jnp.array(x)

            # Compute objective
            objective_value = float(self.loss_function.compute_loss(params))

            # Compute constraints if handler is provided
            constraint_violations = []
            if self.constraint_handler:
                constraints = self.constraint_handler.compute_constraint_values(params)
                # Convert inequality constraints g(x) <= 0 to violations
                for name, value in constraints.items():
                    if "equality" in name.lower():
                        # Equality constraint: |g(x)|
                        constraint_violations.append(abs(float(value)))
                    else:
                        # Inequality constraint: max(0, g(x))
                        constraint_violations.append(max(0.0, float(value)))

            # Return [objective] + [constraint_violations]
            return [objective_value] + constraint_violations

        except Exception:
            # Return large penalty for invalid solutions
            penalty = 1e12
            return [penalty] + [penalty] * self.n_constraints

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get parameter bounds for PyGMO.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Earth departure radius bounds (200-600 km altitude)
        earth_radius = 6.378e6
        r1_min, r1_max = earth_radius + 200e3, earth_radius + 600e3

        # Lunar orbit radius bounds (100-400 km altitude)
        moon_radius = 1.737e6
        r2_min, r2_max = moon_radius + 100e3, moon_radius + 400e3

        # Time of flight bounds (3-10 days)
        tof_min, tof_max = 3.0 * 24 * 3600, 10.0 * 24 * 3600

        lower_bounds = [r1_min, r2_min, tof_min]
        upper_bounds = [r1_max, r2_max, tof_max]

        return lower_bounds, upper_bounds

    def get_nobj(self) -> int:
        """Get number of objectives."""
        return self.n_objectives

    def get_nec(self) -> int:
        """Get number of equality constraints."""
        if not self.constraint_handler:
            return 0

        # Count equality constraints
        dummy_params = jnp.array([6.778e6, 1.937e6, 4.5 * 24 * 3600])
        constraints = self.constraint_handler.compute_constraint_values(dummy_params)
        return len([name for name in constraints.keys() if "equality" in name.lower()])

    def get_nic(self) -> int:
        """Get number of inequality constraints."""
        if not self.constraint_handler:
            return 0

        # Count inequality constraints
        dummy_params = jnp.array([6.778e6, 1.937e6, 4.5 * 24 * 3600])
        constraints = self.constraint_handler.compute_constraint_values(dummy_params)
        return len(
            [name for name in constraints.keys() if "equality" not in name.lower()]
        )


class PyGMOIntegration:
    """
    Integration interface between PyGMO global optimization and JAX local optimization.

    This class provides hybrid optimization workflows that leverage the strengths
    of both global and local optimization methods.
    """

    def __init__(
        self,
        trajectory_model: TrajectoryModel,
        economic_model: EconomicModel,
        loss_function: MultiObjectiveLoss,
        constraint_handler: Optional[ConstraintHandler] = None,
        config: Optional[HybridOptimizationConfig] = None,
    ):
        """
        Initialize PyGMO-JAX integration.

        Args:
            trajectory_model: JAX trajectory model
            economic_model: JAX economic model
            loss_function: Multi-objective loss function
            constraint_handler: Optional constraint handler
            config: Hybrid optimization configuration
        """
        if not PYGMO_AVAILABLE:
            raise ImportError("PyGMO is required for global optimization integration")

        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for local optimization integration")

        self.trajectory_model = trajectory_model
        self.economic_model = economic_model
        self.loss_function = loss_function
        self.constraint_handler = constraint_handler
        self.config = config or HybridOptimizationConfig()

        # Create PyGMO problem
        self.pygmo_problem = PyGMOProblem(
            trajectory_model, economic_model, loss_function, constraint_handler
        )

        # Create JAX optimizer
        self.jax_optimizer = DifferentiableOptimizer(
            objective_function=loss_function.compute_loss,
            constraint_functions=(
                self._create_constraint_functions() if constraint_handler else None
            ),
            bounds=self._get_jax_bounds(),
            method=self.config.local_method,
            tolerance=self.config.local_tolerance,
            max_iterations=self.config.local_max_iterations,
            use_jit=True,
            verbose=False,
        )

        # Optimization history
        self.global_optimization_history = []
        self.local_optimization_history = []
        self.hybrid_optimization_history = []

    def _create_constraint_functions(self) -> List[Callable]:
        """Create constraint functions for JAX optimizer."""
        if not self.constraint_handler:
            return []

        def constraint_function(x):
            constraints = self.constraint_handler.compute_constraint_values(x)
            # Return constraint violations as array
            violations = []
            for name, value in constraints.items():
                if "equality" in name.lower():
                    violations.append(value)  # Equality: g(x) = 0
                else:
                    violations.append(value)  # Inequality: g(x) <= 0
            return jnp.array(violations)

        return [constraint_function]

    def _get_jax_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for JAX optimizer."""
        lower_bounds, upper_bounds = self.pygmo_problem.get_bounds()
        return list(zip(lower_bounds, upper_bounds, strict=False))

    def run_global_optimization(self) -> Dict[str, Any]:
        """
        Run PyGMO global optimization.

        Returns:
            Dictionary with global optimization results
        """
        start_time = time.time()

        # Create PyGMO problem and algorithm
        problem = pg.problem(self.pygmo_problem)

        if self.config.pygmo_algorithm.lower() == "nsga2":
            algorithm = pg.algorithm(pg.nsga2(gen=self.config.pygmo_generations))
        elif self.config.pygmo_algorithm.lower() == "de":
            algorithm = pg.algorithm(pg.de(gen=self.config.pygmo_generations))
        elif self.config.pygmo_algorithm.lower() == "pso":
            algorithm = pg.algorithm(pg.pso(gen=self.config.pygmo_generations))
        else:
            # Default to differential evolution
            algorithm = pg.algorithm(pg.de(gen=self.config.pygmo_generations))

        # Create and evolve population
        population = pg.population(problem, self.config.pygmo_population_size)

        if self.config.time_limit_global:
            # Time-limited optimization (simplified)
            end_time = start_time + self.config.time_limit_global
            while time.time() < end_time:
                population = algorithm.evolve(population)
        else:
            population = algorithm.evolve(population)

        optimization_time = time.time() - start_time

        # Extract results
        best_idx = population.best_idx()
        best_solution = population.get_x()[best_idx]
        best_fitness = population.get_f()[best_idx]

        results = {
            "best_solution": best_solution,
            "best_fitness": best_fitness,
            "population": population,
            "algorithm": algorithm,
            "optimization_time": optimization_time,
            "final_population_size": len(population),
            "convergence_info": {
                "generations": self.config.pygmo_generations,
                "final_best_fitness": (
                    best_fitness[0] if len(best_fitness) > 0 else None
                ),
            },
        }

        self.global_optimization_history.append(results)
        return results

    def select_local_start_points(
        self, global_results: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Select starting points for local optimization from global results.

        Args:
            global_results: Results from global optimization

        Returns:
            List of starting points for local optimization
        """
        population = global_results["population"]
        solutions = population.get_x()
        fitnesses = population.get_f()

        if self.config.selection_strategy == "best":
            # Select best solutions
            sorted_indices = np.argsort([f[0] for f in fitnesses])
            selected_indices = sorted_indices[: self.config.num_local_starts]

        elif self.config.selection_strategy == "diverse":
            # Select diverse solutions using clustering or distance-based selection
            selected_indices = self._select_diverse_solutions(
                solutions, self.config.num_local_starts
            )

        elif self.config.selection_strategy == "best_diverse":
            # Combine best and diverse selection
            num_best = self.config.num_local_starts // 2
            num_diverse = self.config.num_local_starts - num_best

            # Get best solutions
            sorted_indices = np.argsort([f[0] for f in fitnesses])
            best_indices = sorted_indices[:num_best]

            # Get diverse solutions from remaining
            remaining_solutions = [solutions[i] for i in sorted_indices[num_best:]]
            diverse_indices_relative = self._select_diverse_solutions(
                remaining_solutions, num_diverse
            )
            diverse_indices = [
                sorted_indices[num_best + i] for i in diverse_indices_relative
            ]

            selected_indices = list(best_indices) + diverse_indices
        else:
            # Default to best selection
            sorted_indices = np.argsort([f[0] for f in fitnesses])
            selected_indices = sorted_indices[: self.config.num_local_starts]

        return [np.array(solutions[i]) for i in selected_indices]

    def _select_diverse_solutions(
        self, solutions: List[List[float]], num_select: int
    ) -> List[int]:
        """
        Select diverse solutions using simple distance-based selection.

        Args:
            solutions: List of solution vectors
            num_select: Number of solutions to select

        Returns:
            List of selected solution indices
        """
        if len(solutions) <= num_select:
            return list(range(len(solutions)))

        selected_indices = []
        remaining_indices = list(range(len(solutions)))

        # Start with random solution
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Greedily select most distant solutions
        for _ in range(num_select - 1):
            if not remaining_indices:
                break

            max_min_distance = -1
            best_idx = None

            for candidate_idx in remaining_indices:
                # Compute minimum distance to already selected solutions
                min_distance = float("inf")
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(
                        np.array(solutions[candidate_idx])
                        - np.array(solutions[selected_idx])
                    )
                    min_distance = min(min_distance, distance)

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = candidate_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return selected_indices

    def run_local_optimization(
        self, start_points: List[np.ndarray]
    ) -> List[OptimizationResult]:
        """
        Run JAX local optimization from multiple starting points.

        Args:
            start_points: List of starting points

        Returns:
            List of local optimization results
        """
        local_results = []

        for i, start_point in enumerate(start_points):
            start_time = time.time()

            try:
                # Convert to JAX array
                x0 = jnp.array(start_point)

                # Run optimization with time limit if specified
                if self.config.time_limit_local:
                    # Simplified time limit implementation
                    result = self.jax_optimizer.optimize(
                        x0,
                        options={"maxiter": min(self.config.local_max_iterations, 50)},
                    )
                else:
                    result = self.jax_optimizer.optimize(x0)

                optimization_time = time.time() - start_time

                # Add metadata
                result.start_point = start_point
                result.local_optimization_index = i
                result.local_optimization_time = optimization_time

                local_results.append(result)

            except Exception as e:
                # Create failed result
                failed_result = OptimizationResult(
                    x=start_point,
                    fun=float("inf"),
                    success=False,
                    message=f"Local optimization failed: {str(e)}",
                    nit=0,
                    nfev=0,
                    njev=0,
                    optimization_time=time.time() - start_time,
                )
                failed_result.start_point = start_point
                failed_result.local_optimization_index = i
                local_results.append(failed_result)

        self.local_optimization_history.append(local_results)
        return local_results

    def run_hybrid_optimization(self) -> Dict[str, Any]:
        """
        Run complete hybrid optimization workflow.

        Returns:
            Dictionary with hybrid optimization results
        """
        start_time = time.time()

        # Step 1: Global optimization
        print("🌍 Running global optimization with PyGMO...")
        global_results = self.run_global_optimization()

        # Step 2: Select starting points for local optimization
        print(
            f"🎯 Selecting {self.config.num_local_starts} starting points for local optimization..."
        )
        start_points = self.select_local_start_points(global_results)

        # Step 3: Local optimization
        print("🚀 Running local optimization with JAX...")
        local_results = self.run_local_optimization(start_points)

        # Step 4: Analyze results
        print("📊 Analyzing hybrid optimization results...")
        analysis = self.analyze_hybrid_results(global_results, local_results)

        total_time = time.time() - start_time

        hybrid_results = {
            "global_results": global_results,
            "local_results": local_results,
            "analysis": analysis,
            "total_optimization_time": total_time,
            "config": self.config,
            "num_successful_local": len([r for r in local_results if r.success]),
            "best_solution": (
                analysis["best_local_solution"]
                if analysis["best_local_solution"]
                else global_results["best_solution"]
            ),
            "best_objective": (
                analysis["best_local_objective"]
                if analysis["best_local_objective"] is not None
                else global_results["best_fitness"][0]
            ),
        }

        self.hybrid_optimization_history.append(hybrid_results)
        return hybrid_results

    def analyze_hybrid_results(
        self, global_results: Dict[str, Any], local_results: List[OptimizationResult]
    ) -> Dict[str, Any]:
        """
        Analyze and compare global vs local optimization results.

        Args:
            global_results: Global optimization results
            local_results: Local optimization results

        Returns:
            Dictionary with analysis results
        """
        successful_local = [r for r in local_results if r.success]

        if not successful_local:
            return {
                "best_local_solution": None,
                "best_local_objective": None,
                "improvement_over_global": 0.0,
                "local_success_rate": 0.0,
                "average_improvement": 0.0,
                "solution_comparisons": [],
            }

        # Find best local solution
        best_local = min(successful_local, key=lambda r: r.fun)
        best_global_objective = global_results["best_fitness"][0]

        # Calculate improvements
        improvement = (
            100 * (best_global_objective - best_local.fun) / abs(best_global_objective)
        )

        # Calculate statistics
        local_success_rate = len(successful_local) / len(local_results)

        improvements = []
        for result in successful_local:
            if (
                hasattr(result, "initial_objective")
                and result.initial_objective is not None
            ):
                local_improvement = (
                    100
                    * (result.initial_objective - result.fun)
                    / abs(result.initial_objective)
                )
                improvements.append(local_improvement)

        average_improvement = np.mean(improvements) if improvements else 0.0

        # Create solution comparisons
        comparisons = []
        for _i, result in enumerate(local_results):
            if result.success:
                comparison = SolutionComparison(
                    global_solution={
                        "parameters": result.start_point,
                        "objective": float(
                            self.loss_function.compute_loss(
                                jnp.array(result.start_point)
                            )
                        ),
                    },
                    local_solution={"parameters": result.x, "objective": result.fun},
                    improvement_percentage=result.improvement_percentage or 0.0,
                    constraint_improvement=0.0,  # Would need constraint analysis
                    optimization_times={
                        "local": getattr(
                            result, "local_optimization_time", result.optimization_time
                        )
                    },
                    convergence_info={
                        "iterations": result.nit,
                        "function_evaluations": result.nfev,
                        "success": result.success,
                    },
                )
                comparisons.append(comparison)

        return {
            "best_local_solution": best_local.x,
            "best_local_objective": best_local.fun,
            "improvement_over_global": improvement,
            "local_success_rate": local_success_rate,
            "average_local_improvement": average_improvement,
            "solution_comparisons": comparisons,
            "optimization_times": {
                "global": global_results["optimization_time"],
                "local_total": sum(
                    getattr(r, "local_optimization_time", r.optimization_time)
                    for r in local_results
                ),
                "local_average": np.mean(
                    [
                        getattr(r, "local_optimization_time", r.optimization_time)
                        for r in local_results
                    ]
                ),
            },
            "convergence_statistics": {
                "local_iterations_avg": np.mean([r.nit for r in successful_local]),
                "local_function_evals_avg": np.mean([r.nfev for r in successful_local]),
                "local_convergence_rate": len(successful_local) / len(local_results),
            },
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all optimization runs.

        Returns:
            Dictionary with optimization summary
        """
        if not self.hybrid_optimization_history:
            return {"message": "No hybrid optimization runs completed"}

        latest_run = self.hybrid_optimization_history[-1]

        return {
            "total_runs": len(self.hybrid_optimization_history),
            "latest_results": {
                "best_objective": latest_run["best_objective"],
                "total_time": latest_run["total_optimization_time"],
                "global_time": latest_run["global_results"]["optimization_time"],
                "local_success_rate": latest_run["analysis"]["local_success_rate"],
                "improvement_over_global": latest_run["analysis"][
                    "improvement_over_global"
                ],
            },
            "configuration": {
                "global_algorithm": self.config.pygmo_algorithm,
                "local_method": self.config.local_method,
                "population_size": self.config.pygmo_population_size,
                "local_starts": self.config.num_local_starts,
            },
            "performance_metrics": {
                "average_total_time": np.mean(
                    [
                        h["total_optimization_time"]
                        for h in self.hybrid_optimization_history
                    ]
                ),
                "average_improvement": np.mean(
                    [
                        h["analysis"]["improvement_over_global"]
                        for h in self.hybrid_optimization_history
                    ]
                ),
                "best_ever_objective": min(
                    h["best_objective"] for h in self.hybrid_optimization_history
                ),
            },
        }


# Utility functions for creating integration workflows
def create_standard_hybrid_optimizer(
    trajectory_model: TrajectoryModel,
    economic_model: EconomicModel,
    loss_function: MultiObjectiveLoss,
    constraint_handler: Optional[ConstraintHandler] = None,
) -> PyGMOIntegration:
    """
    Create a standard hybrid optimizer with balanced configuration.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        loss_function: Multi-objective loss function
        constraint_handler: Optional constraint handler

    Returns:
        Configured PyGMOIntegration instance
    """
    config = HybridOptimizationConfig(
        pygmo_algorithm="nsga2",
        pygmo_population_size=50,
        pygmo_generations=25,
        num_local_starts=5,
        selection_strategy="best_diverse",
    )

    return PyGMOIntegration(
        trajectory_model, economic_model, loss_function, constraint_handler, config
    )


def create_fast_hybrid_optimizer(
    trajectory_model: TrajectoryModel,
    economic_model: EconomicModel,
    loss_function: MultiObjectiveLoss,
    constraint_handler: Optional[ConstraintHandler] = None,
) -> PyGMOIntegration:
    """
    Create a fast hybrid optimizer with reduced computational requirements.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        loss_function: Multi-objective loss function
        constraint_handler: Optional constraint handler

    Returns:
        Configured PyGMOIntegration instance
    """
    config = HybridOptimizationConfig(
        pygmo_algorithm="de",
        pygmo_population_size=20,
        pygmo_generations=10,
        num_local_starts=3,
        local_max_iterations=50,
        selection_strategy="best",
    )

    return PyGMOIntegration(
        trajectory_model, economic_model, loss_function, constraint_handler, config
    )


def create_thorough_hybrid_optimizer(
    trajectory_model: TrajectoryModel,
    economic_model: EconomicModel,
    loss_function: MultiObjectiveLoss,
    constraint_handler: Optional[ConstraintHandler] = None,
) -> PyGMOIntegration:
    """
    Create a thorough hybrid optimizer for high-quality solutions.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        loss_function: Multi-objective loss function
        constraint_handler: Optional constraint handler

    Returns:
        Configured PyGMOIntegration instance
    """
    config = HybridOptimizationConfig(
        pygmo_algorithm="nsga2",
        pygmo_population_size=100,
        pygmo_generations=50,
        num_local_starts=10,
        local_max_iterations=200,
        selection_strategy="best_diverse",
        diversity_threshold=0.05,
    )

    return PyGMOIntegration(
        trajectory_model, economic_model, loss_function, constraint_handler, config
    )
