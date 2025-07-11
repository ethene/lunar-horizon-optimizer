"""
JAX-based Differentiable Optimizer

This module implements the main differentiable optimizer using JAX for automatic
differentiation and gradient-based optimization methods.

Features:
- Gradient-based optimization using scipy.optimize
- JIT-compiled objective and gradient functions
- Integration with PyGMO global optimization results
- Multi-objective optimization with customizable weights
- Performance monitoring and convergence tracking

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import logging
from dataclasses import dataclass, field

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, value_and_grad

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Standard imports
import numpy as np
from scipy.optimize import minimize

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of differentiable optimization."""

    # Optimization results
    x: np.ndarray  # Optimized parameters
    fun: float  # Final objective value
    success: bool  # Whether optimization succeeded
    message: str  # Optimization status message
    nit: int  # Number of iterations
    nfev: int  # Number of function evaluations
    njev: int  # Number of jacobian evaluations

    # Performance metrics
    optimization_time: float  # Wall clock time for optimization
    convergence_history: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)

    # Objective breakdown
    objective_components: Dict[str, float] = field(default_factory=dict)
    constraint_violations: Dict[str, float] = field(default_factory=dict)

    # Comparison with initial solution
    initial_objective: Optional[float] = None
    improvement_percentage: Optional[float] = None


class DifferentiableOptimizer:
    """
    JAX-based differentiable optimizer for trajectory and economic optimization.

    This optimizer uses automatic differentiation to compute gradients and
    applies gradient-based optimization methods to refine solutions from
    global optimization.
    """

    def __init__(
        self,
        objective_function: Callable,
        constraint_functions: Optional[List[Callable]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        use_jit: bool = True,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize the differentiable optimizer.

        Args:
            objective_function: JAX-compatible objective function to minimize
            constraint_functions: List of JAX-compatible constraint functions
            bounds: Parameter bounds as list of (min, max) tuples
            method: Scipy optimization method ('L-BFGS-B', 'SLSQP', etc.)
            use_jit: Whether to JIT-compile functions for performance
            tolerance: Optimization tolerance
            max_iterations: Maximum number of iterations
            verbose: Whether to print optimization progress
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for differentiable optimization")

        self.objective_function = objective_function
        self.constraint_functions = constraint_functions or []
        self.bounds = bounds
        self.method = method
        self.use_jit = use_jit
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Setup compiled functions
        self._setup_compiled_functions()

        # Optimization tracking
        self.optimization_history = []
        self.current_iteration = 0
        self.start_time = None

    def _setup_compiled_functions(self):
        """Setup JIT-compiled objective and gradient functions with optimization."""
        if self.use_jit:
            # JIT-compile objective function and its gradient with performance optimizations
            self._compiled_objective = jit(
                self.objective_function,
                static_argnums=(),  # Can be customized for specific use cases
                donate_argnums=(),  # Memory optimization for large arrays
            )
            self._compiled_grad = jit(
                grad(self.objective_function), static_argnums=(), donate_argnums=()
            )
            self._compiled_value_and_grad = jit(
                value_and_grad(self.objective_function),
                static_argnums=(),
                donate_argnums=(),
            )

            # JIT-compile constraint functions if provided
            if self.constraint_functions:
                self._compiled_constraints = [
                    jit(cf, static_argnums=(), donate_argnums=())
                    for cf in self.constraint_functions
                ]
                self._compiled_constraint_grads = [
                    jit(grad(cf), static_argnums=(), donate_argnums=())
                    for cf in self.constraint_functions
                ]

            # Setup vectorized functions for batch operations
            self._compiled_batch_objective = jit(
                vmap(self.objective_function, in_axes=0)
            )
            self._compiled_batch_grad = jit(
                vmap(grad(self.objective_function), in_axes=0)
            )

        else:
            self._compiled_objective = self.objective_function
            self._compiled_grad = grad(self.objective_function)
            self._compiled_value_and_grad = value_and_grad(self.objective_function)

            if self.constraint_functions:
                self._compiled_constraints = self.constraint_functions
                self._compiled_constraint_grads = [
                    grad(cf) for cf in self.constraint_functions
                ]

            # Non-JIT vectorized functions
            self._compiled_batch_objective = vmap(self.objective_function, in_axes=0)
            self._compiled_batch_grad = vmap(grad(self.objective_function), in_axes=0)

    def _callback(self, x: np.ndarray):
        """Callback function for optimization progress tracking."""
        self.current_iteration += 1

        # Compute current objective value
        obj_val = float(self._compiled_objective(x))
        grad_norm = float(np.linalg.norm(self._compiled_grad(x)))

        # Store history
        self.optimization_history.append(obj_val)

        if self.verbose and self.current_iteration % 10 == 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(
                f"Iteration {self.current_iteration}: "
                f"obj={obj_val:.6e}, grad_norm={grad_norm:.6e}, "
                f"time={elapsed:.2f}s"
            )

    def optimize(
        self, x0: Union[np.ndarray, jnp.ndarray], **optimizer_kwargs
    ) -> OptimizationResult:
        """
        Perform gradient-based optimization starting from initial point x0.

        Args:
            x0: Initial parameter values
            **optimizer_kwargs: Additional arguments for scipy.optimize.minimize

        Returns:
            OptimizationResult containing optimization results and metrics
        """
        # Convert JAX array to numpy if needed
        x0 = np.asarray(x0)

        # Record initial objective value
        initial_obj = float(self._compiled_objective(x0))

        # Setup optimization tracking
        self.optimization_history = []
        self.current_iteration = 0
        self.start_time = time.time()

        # Define objective function for scipy (returns numpy scalars)
        def scipy_objective(x):
            return float(self._compiled_objective(x))

        # Define gradient function for scipy (returns numpy arrays)
        def scipy_gradient(x):
            return np.asarray(self._compiled_grad(x))

        # Setup constraints for scipy if provided
        constraints = []
        if self.constraint_functions:
            for _i, (constraint_func, grad_func) in enumerate(
                zip(
                    self._compiled_constraints,
                    self._compiled_constraint_grads,
                    strict=False,
                )
            ):
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x, cf=constraint_func: float(cf(x)),
                        "jac": lambda x, gf=grad_func: np.asarray(gf(x)),
                    }
                )

        # Merge optimization parameters
        opt_params = {
            "method": self.method,
            "jac": scipy_gradient,
            "bounds": self.bounds,
            "callback": self._callback,
            "options": {
                "ftol": self.tolerance,
                "gtol": self.tolerance,
                "maxiter": self.max_iterations,
                **optimizer_kwargs.get("options", {}),
            },
        }

        if constraints:
            opt_params["constraints"] = constraints

        # Remove callback if not supported by method
        if self.method in ["Nelder-Mead", "Powell"]:
            opt_params.pop("jac", None)

        # Perform optimization
        try:
            if self.verbose:
                logger.info(f"Starting optimization with method: {self.method}")
                logger.info(f"Initial objective: {initial_obj:.6e}")

            result = minimize(scipy_objective, x0, **opt_params)
            optimization_time = time.time() - self.start_time

            if self.verbose:
                logger.info(f"Optimization completed in {optimization_time:.2f}s")
                logger.info(f"Final objective: {result.fun:.6e}")
                logger.info(f"Success: {result.success}")

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return OptimizationResult(
                x=x0,
                fun=initial_obj,
                success=False,
                message=f"Optimization failed: {str(e)}",
                nit=0,
                nfev=0,
                njev=0,
                optimization_time=0.0,
                initial_objective=initial_obj,
            )

        # Compute improvement percentage
        improvement = None
        if initial_obj != 0:
            improvement = 100 * (initial_obj - result.fun) / abs(initial_obj)

        # Analyze objective components and constraints
        obj_components = self._analyze_objective_components(result.x)
        constraint_violations = self._analyze_constraint_violations(result.x)

        # Extract convergence history
        convergence_history = self.optimization_history
        gradient_norms = []
        if result.success and len(convergence_history) > 1:
            # Compute gradient norms throughout optimization (approximate)
            try:
                grad_norm = float(np.linalg.norm(self._compiled_grad(result.x)))
                gradient_norms = [grad_norm]  # Simplified for now
            except:
                gradient_norms = []

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            success=result.success,
            message=result.message,
            nit=result.nit,
            nfev=result.nfev,
            njev=result.get("njev", 0),
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            gradient_norms=gradient_norms,
            objective_components=obj_components,
            constraint_violations=constraint_violations,
            initial_objective=initial_obj,
            improvement_percentage=improvement,
        )

    def _analyze_objective_components(self, x: np.ndarray) -> Dict[str, float]:
        """Analyze individual components of the objective function."""
        # This is a placeholder - specific implementations will override
        # to provide detailed breakdown of objective components
        try:
            total_obj = float(self._compiled_objective(x))
            return {"total": total_obj}
        except:
            return {}

    def _analyze_constraint_violations(self, x: np.ndarray) -> Dict[str, float]:
        """Analyze constraint violations at the solution."""
        violations = {}
        try:
            for i, constraint_func in enumerate(self._compiled_constraints):
                violation = float(constraint_func(x))
                violations[f"constraint_{i}"] = violation
        except:
            pass
        return violations

    def batch_optimize(
        self,
        x0_batch: List[Union[np.ndarray, jnp.ndarray]],
        parallel: bool = True,
        **optimizer_kwargs,
    ) -> List[OptimizationResult]:
        """
        Optimize multiple initial points in batch with performance optimizations.

        Args:
            x0_batch: List of initial parameter arrays
            parallel: Whether to use parallel processing
            **optimizer_kwargs: Additional arguments for optimization

        Returns:
            List of OptimizationResult objects
        """
        results = []

        # Convert batch to JAX array for potential vectorized operations
        try:
            x0_array = jnp.array(x0_batch)
            batch_size = x0_array.shape[0]

            if self.verbose:
                logger.info(f"Starting batch optimization for {batch_size} candidates")

            # For small batches, can attempt vectorized evaluation for initial assessment
            if batch_size <= 10 and hasattr(self, "_compiled_batch_objective"):
                initial_objectives = self._compiled_batch_objective(x0_array)
                if self.verbose:
                    logger.info(
                        f"Initial objectives range: {float(jnp.min(initial_objectives)):.6e} to {float(jnp.max(initial_objectives)):.6e}"
                    )
        except:
            # Fall back to individual processing if batch conversion fails
            pass

        for i, x0 in enumerate(x0_batch):
            if self.verbose and batch_size > 5:
                logger.info(f"Optimizing candidate {i+1}/{len(x0_batch)}")

            result = self.optimize(x0, **optimizer_kwargs)
            results.append(result)

        return results

    def evaluate_batch_objectives(self, parameter_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate objective function for a batch of parameters efficiently.

        Args:
            parameter_batch: Batch of parameter vectors [batch_size, n_params]

        Returns:
            Array of objective values [batch_size]
        """
        if hasattr(self, "_compiled_batch_objective"):
            return self._compiled_batch_objective(parameter_batch)
        else:
            # Fall back to individual evaluations
            return jnp.array([self._compiled_objective(x) for x in parameter_batch])

    def evaluate_batch_gradients(self, parameter_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate gradients for a batch of parameters efficiently.

        Args:
            parameter_batch: Batch of parameter vectors [batch_size, n_params]

        Returns:
            Array of gradient vectors [batch_size, n_params]
        """
        if hasattr(self, "_compiled_batch_grad"):
            return self._compiled_batch_grad(parameter_batch)
        else:
            # Fall back to individual evaluations
            return jnp.array([self._compiled_grad(x) for x in parameter_batch])

    def compare_with_initial(self, results: List[OptimizationResult]) -> Dict[str, Any]:
        """
        Compare optimization results with initial solutions.

        Args:
            results: List of optimization results

        Returns:
            Dictionary with comparison statistics
        """
        if not results:
            return {}

        successful_results = [r for r in results if r.success]

        if not successful_results:
            return {
                "total_candidates": len(results),
                "successful_optimizations": 0,
                "success_rate": 0.0,
            }

        improvements = [
            r.improvement_percentage
            for r in successful_results
            if r.improvement_percentage is not None
        ]

        final_objectives = [r.fun for r in successful_results]
        [
            r.initial_objective
            for r in successful_results
            if r.initial_objective is not None
        ]

        return {
            "total_candidates": len(results),
            "successful_optimizations": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "average_improvement_percentage": (
                np.mean(improvements) if improvements else 0.0
            ),
            "best_improvement_percentage": (
                np.max(improvements) if improvements else 0.0
            ),
            "average_final_objective": np.mean(final_objectives),
            "best_final_objective": np.min(final_objectives),
            "average_optimization_time": np.mean(
                [r.optimization_time for r in successful_results]
            ),
            "total_function_evaluations": sum(r.nfev for r in successful_results),
        }


# Utility functions for creating optimizers with common configurations
def create_trajectory_optimizer(
    trajectory_model: Callable,
    economic_model: Callable,
    weights: Dict[str, float] = None,
    **kwargs,
) -> DifferentiableOptimizer:
    """
    Create a differentiable optimizer for trajectory optimization.

    Args:
        trajectory_model: JAX function for trajectory evaluation
        economic_model: JAX function for economic evaluation
        weights: Weights for multi-objective optimization
        **kwargs: Additional arguments for DifferentiableOptimizer

    Returns:
        Configured DifferentiableOptimizer instance
    """
    if weights is None:
        weights = {"delta_v": 1.0, "time": 1.0, "cost": 1.0}

    def combined_objective(x):
        """Combined trajectory and economic objective."""
        traj_result = trajectory_model(x)
        econ_result = economic_model(x)

        return (
            weights.get("delta_v", 1.0) * traj_result.get("delta_v", 0.0)
            + weights.get("time", 1.0) * traj_result.get("time", 0.0)
            + weights.get("cost", 1.0) * econ_result.get("total_cost", 0.0)
        )

    return DifferentiableOptimizer(objective_function=combined_objective, **kwargs)


def create_economic_optimizer(
    economic_model: Callable, objective: str = "minimize_cost", **kwargs
) -> DifferentiableOptimizer:
    """
    Create a differentiable optimizer focused on economic objectives.

    Args:
        economic_model: JAX function for economic evaluation
        objective: Type of economic objective ("minimize_cost", "maximize_npv", etc.)
        **kwargs: Additional arguments for DifferentiableOptimizer

    Returns:
        Configured DifferentiableOptimizer instance
    """

    def economic_objective(x):
        """Economic-focused objective function."""
        result = economic_model(x)

        if objective == "minimize_cost":
            return result.get("total_cost", 0.0)
        elif objective == "maximize_npv":
            return -result.get("npv", 0.0)  # Negative for minimization
        elif objective == "maximize_roi":
            return -result.get("roi", 0.0)  # Negative for minimization
        else:
            return result.get("total_cost", 0.0)

    return DifferentiableOptimizer(objective_function=economic_objective, **kwargs)
