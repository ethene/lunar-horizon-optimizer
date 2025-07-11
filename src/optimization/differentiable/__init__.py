"""
Differentiable Optimization Module

This module implements local gradient-based optimization using JAX and Diffrax
to refine trajectory and economic solutions from global optimization.

Key Features:
- JAX-based differentiable trajectory models
- Automatic differentiation through economic calculations
- Gradient-based local optimization
- Integration with PyGMO global optimization results
- JIT-compiled performance optimization

Modules:
- jax_optimizer: Main differentiable optimizer implementation
- differentiable_models: JAX versions of trajectory and economic models
- loss_functions: Multi-objective loss functions combining physical and economic objectives
- constraints: Differentiable constraint handling
- integration: Interface with PyGMO global optimization results

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Any, Dict, Optional, Tuple, Union

# JAX and scientific computing imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    import diffrax

    JAX_AVAILABLE = True

    # Set JAX configuration for optimization
    jax.config.update("jax_enable_x64", True)  # Use 64-bit precision

except ImportError as e:
    JAX_AVAILABLE = False
    _jax_import_error = str(e)

# Standard imports
import numpy as np
import scipy.optimize
from dataclasses import dataclass

# Module version
__version__ = "1.0.0"

# Export key classes and functions
__all__ = [
    "JAX_AVAILABLE",
    "DifferentiableOptimizer",
    "TrajectoryModel",
    "EconomicModel",
    "MultiObjectiveLoss",
    "ConstraintHandler",
    "PyGMOIntegration",
    # Performance optimization
    "PerformanceConfig",
    "JITOptimizer",
    "BatchOptimizer",
    "MemoryOptimizer",
    "PerformanceBenchmark",
    "optimize_differentiable_optimizer",
    "create_performance_optimized_loss_function",
    "benchmark_optimization_performance",
    # Loss functions
    "LossFunctionConfig",
    "WeightingStrategy",
    "NormalizationMethod",
    "create_balanced_loss_function",
    "create_performance_focused_loss_function",
    "create_economic_focused_loss_function",
    "create_adaptive_loss_function",
    # Constraints
    "ConstraintConfig",
    "ConstraintType",
    "ConstraintHandlingMethod",
    "create_penalty_constraint_handler",
    "create_barrier_constraint_handler",
    "create_adaptive_constraint_handler",
    # Integration
    "HybridOptimizationConfig",
    "PyGMOProblem",
    "create_standard_hybrid_optimizer",
    "create_fast_hybrid_optimizer",
    "create_thorough_hybrid_optimizer",
    # Result comparison
    "ResultComparator",
    "ComparisonResult",
    "ConvergenceAnalysis",
    "SolutionQuality",
    "ComparisonMetric",
    "compare_single_run",
    "analyze_optimization_convergence",
    "rank_optimization_results",
    "evaluate_solution_quality",
    "ComparisonDemonstration",
    "run_comparison_demo",
]


# Check JAX availability and provide helpful error message
def _check_jax_availability():
    """Check if JAX is available and raise informative error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            f"JAX/Diffrax not available: {_jax_import_error}\n"
            "Please install JAX and Diffrax:\n"
            "  conda install -c conda-forge jax\n"
            "  pip install diffrax"
        )


# Lazy imports to avoid circular dependencies
def _lazy_import():
    """Lazy import of module components."""
    if not JAX_AVAILABLE:
        return None

    try:
        from .jax_optimizer import DifferentiableOptimizer
        from .differentiable_models import TrajectoryModel, EconomicModel
        from .loss_functions import (
            MultiObjectiveLoss,
            LossFunctionConfig,
            WeightingStrategy,
            NormalizationMethod,
            create_balanced_loss_function,
            create_performance_focused_loss_function,
            create_economic_focused_loss_function,
            create_adaptive_loss_function,
        )
        from .constraints import (
            ConstraintHandler,
            ConstraintConfig,
            ConstraintType,
            ConstraintHandlingMethod,
            create_penalty_constraint_handler,
            create_barrier_constraint_handler,
            create_adaptive_constraint_handler,
        )
        from .integration import (
            PyGMOIntegration,
            HybridOptimizationConfig,
            PyGMOProblem,
            create_standard_hybrid_optimizer,
            create_fast_hybrid_optimizer,
            create_thorough_hybrid_optimizer,
        )
        from .performance_optimization import (
            PerformanceConfig,
            JITOptimizer,
            BatchOptimizer,
            MemoryOptimizer,
            PerformanceBenchmark,
            optimize_differentiable_optimizer,
            create_performance_optimized_loss_function,
            benchmark_optimization_performance,
        )
        from .result_comparison import (
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
        from .comparison_demo import ComparisonDemonstration, run_comparison_demo

        return {
            "DifferentiableOptimizer": DifferentiableOptimizer,
            "TrajectoryModel": TrajectoryModel,
            "EconomicModel": EconomicModel,
            "MultiObjectiveLoss": MultiObjectiveLoss,
            "ConstraintHandler": ConstraintHandler,
            "PyGMOIntegration": PyGMOIntegration,
            # Performance optimization
            "PerformanceConfig": PerformanceConfig,
            "JITOptimizer": JITOptimizer,
            "BatchOptimizer": BatchOptimizer,
            "MemoryOptimizer": MemoryOptimizer,
            "PerformanceBenchmark": PerformanceBenchmark,
            "optimize_differentiable_optimizer": optimize_differentiable_optimizer,
            "create_performance_optimized_loss_function": create_performance_optimized_loss_function,
            "benchmark_optimization_performance": benchmark_optimization_performance,
            # Loss functions
            "LossFunctionConfig": LossFunctionConfig,
            "WeightingStrategy": WeightingStrategy,
            "NormalizationMethod": NormalizationMethod,
            "create_balanced_loss_function": create_balanced_loss_function,
            "create_performance_focused_loss_function": create_performance_focused_loss_function,
            "create_economic_focused_loss_function": create_economic_focused_loss_function,
            "create_adaptive_loss_function": create_adaptive_loss_function,
            # Constraints
            "ConstraintConfig": ConstraintConfig,
            "ConstraintType": ConstraintType,
            "ConstraintHandlingMethod": ConstraintHandlingMethod,
            "create_penalty_constraint_handler": create_penalty_constraint_handler,
            "create_barrier_constraint_handler": create_barrier_constraint_handler,
            "create_adaptive_constraint_handler": create_adaptive_constraint_handler,
            # Integration
            "HybridOptimizationConfig": HybridOptimizationConfig,
            "PyGMOProblem": PyGMOProblem,
            "create_standard_hybrid_optimizer": create_standard_hybrid_optimizer,
            "create_fast_hybrid_optimizer": create_fast_hybrid_optimizer,
            "create_thorough_hybrid_optimizer": create_thorough_hybrid_optimizer,
            # Result comparison
            "ResultComparator": ResultComparator,
            "ComparisonResult": ComparisonResult,
            "ConvergenceAnalysis": ConvergenceAnalysis,
            "SolutionQuality": SolutionQuality,
            "ComparisonMetric": ComparisonMetric,
            "compare_single_run": compare_single_run,
            "analyze_optimization_convergence": analyze_optimization_convergence,
            "rank_optimization_results": rank_optimization_results,
            "evaluate_solution_quality": evaluate_solution_quality,
            "ComparisonDemonstration": ComparisonDemonstration,
            "run_comparison_demo": run_comparison_demo,
        }
    except ImportError:
        return None


# Module configuration
@dataclass
class DifferentiableOptimizationConfig:
    """Configuration for differentiable optimization."""

    # Optimization parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
    learning_rate: float = 0.01

    # JAX configuration
    use_jit: bool = True
    precision: str = "float64"

    # Multi-objective weights
    delta_v_weight: float = 1.0
    time_weight: float = 1.0
    cost_weight: float = 1.0

    # Constraint parameters
    constraint_penalty: float = 1e6
    constraint_tolerance: float = 1e-3

    # Performance settings
    batch_size: int = 32
    parallel_evaluations: bool = True


# Module utilities
def get_jax_device_info() -> Dict[str, Any]:
    """Get information about available JAX devices."""
    if not JAX_AVAILABLE:
        return {"available": False, "error": _jax_import_error}

    try:
        devices = jax.devices()
        return {
            "available": True,
            "devices": [{"type": d.device_kind, "id": d.id} for d in devices],
            "default_device": str(jax.devices()[0]),
            "x64_enabled": jax.config.x64_enabled,
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def validate_jax_environment() -> bool:
    """Validate that JAX environment is properly configured."""
    if not JAX_AVAILABLE:
        return False

    try:
        # Test basic JAX operations
        x = jnp.array([1.0, 2.0, 3.0])
        _ = jnp.sum(x**2)
        grad_fn = grad(lambda x: jnp.sum(x**2))
        _ = grad_fn(x)

        # Test JIT compilation
        jit_fn = jit(lambda x: jnp.sum(x**2))
        _ = jit_fn(x)

        return True
    except Exception:
        return False


# Initialize module
def initialize_module():
    """Initialize the differentiable optimization module."""
    if JAX_AVAILABLE:
        print(f"✅ JAX Differentiable Optimization Module v{__version__} initialized")
        device_info = get_jax_device_info()
        print(f"✅ JAX device: {device_info.get('default_device', 'unknown')}")

        if validate_jax_environment():
            print("✅ JAX environment validation passed")
        else:
            print("⚠️  JAX environment validation failed")
    else:
        print(f"❌ JAX not available: {_jax_import_error}")
        print("   Differentiable optimization features disabled")


# Auto-initialize when module is imported
initialize_module()
