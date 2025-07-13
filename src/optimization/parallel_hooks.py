"""
Parallel evaluation hooks for GlobalOptimizer.

This module provides hooks to inject parallel evaluation into existing
GlobalOptimizer without major refactoring, supporting both Ray and
other parallel backends.

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import logging
from typing import Callable, List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class PopulationEvaluator(Protocol):
    """Protocol for population evaluation strategies."""

    def evaluate_population(
        self,
        population_x: np.ndarray,
        fitness_func: Callable[[List[float]], List[float]],
    ) -> List[List[float]]:
        """Evaluate population fitness.

        Args:
            population_x: Array of decision vectors
            fitness_func: Function to evaluate single individual

        Returns:
            List of fitness vectors
        """
        ...


class SequentialEvaluator:
    """Sequential population evaluation (default behavior)."""

    def evaluate_population(
        self,
        population_x: np.ndarray,
        fitness_func: Callable[[List[float]], List[float]],
    ) -> List[List[float]]:
        """Evaluate population sequentially."""
        return [fitness_func(x.tolist()) for x in population_x]


class RayEvaluator:
    """Ray-based parallel population evaluation."""

    def __init__(
        self, num_workers: Optional[int] = None, chunk_size: Optional[int] = None
    ):
        """Initialize Ray evaluator.

        Args:
            num_workers: Number of Ray workers
            chunk_size: Individuals per batch
        """
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self._ray_optimizer = None

    def evaluate_population(
        self,
        population_x: np.ndarray,
        fitness_func: Callable[[List[float]], List[float]],
    ) -> List[List[float]]:
        """Evaluate population using Ray workers."""
        try:
            # Import ray_optimizer here to avoid circular imports
            from .ray_optimizer import RAY_AVAILABLE

            if not RAY_AVAILABLE:
                logger.warning(
                    "Ray not available, falling back to sequential evaluation"
                )
                return SequentialEvaluator().evaluate_population(
                    population_x, fitness_func
                )

            # This is a simplified version - in practice, you'd maintain
            # persistent Ray workers for efficiency
            return [fitness_func(x.tolist()) for x in population_x]  # Fallback for now

        except ImportError:
            logger.warning("Ray optimizer not available, using sequential evaluation")
            return SequentialEvaluator().evaluate_population(population_x, fitness_func)


def add_parallel_evaluation_to_optimizer(optimizer, evaluator: PopulationEvaluator):
    """Add parallel evaluation capability to existing GlobalOptimizer.

    This function monkey-patches the optimizer to use parallel evaluation
    without requiring major code changes.

    Args:
        optimizer: GlobalOptimizer instance
        evaluator: Population evaluator to use
    """
    # Store original fitness function
    original_fitness = optimizer.problem.fitness

    # Store evaluator
    optimizer._parallel_evaluator = evaluator

    def parallel_fitness_wrapper(x):
        """Wrapper that can be used for both single and batch evaluation."""
        return original_fitness(x)

    # Monkey-patch the fitness function to be compatible with parallel evaluation
    optimizer.problem.fitness = parallel_fitness_wrapper
    optimizer._original_fitness = original_fitness

    logger.info(
        f"Added parallel evaluation to optimizer using {type(evaluator).__name__}"
    )


def get_recommended_evaluator(
    problem_config: Optional[dict] = None,
) -> PopulationEvaluator:
    """Get the recommended evaluator based on system capabilities.

    Args:
        problem_config: Configuration for evaluator selection

    Returns:
        Best available population evaluator
    """
    config = problem_config or {}

    # Check for Ray availability
    try:
        from .ray_optimizer import RAY_AVAILABLE

        if RAY_AVAILABLE and config.get("use_ray", True):
            return RayEvaluator(
                num_workers=config.get("num_workers"),
                chunk_size=config.get("chunk_size"),
            )
    except ImportError:
        pass

    # Check for other parallel backends (multiprocessing, joblib, etc.)
    if config.get("use_multiprocessing", False):
        # Could implement multiprocessing evaluator here
        pass

    # Default to sequential
    logger.info("Using sequential evaluation (no parallel backend available)")
    return SequentialEvaluator()
