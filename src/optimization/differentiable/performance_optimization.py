"""
Performance Optimization Module for JAX Differentiable Optimization

This module implements performance optimizations for the JAX-based differentiable
optimization system, focusing on JIT compilation strategies, memory efficiency,
and computational optimization techniques.

Features:
- Advanced JIT compilation patterns and caching
- Vectorized operations for batch optimization
- Memory-efficient gradient computations
- Performance profiling and benchmarking tools
- Adaptive compilation strategies
- Function composition and fusion optimizations

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import time

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, pmap, lax, tree_util
    from jax.experimental import optimizers
    from jax.experimental.compilation_cache import initialize_cache

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Local imports
from .differentiable_models import TrajectoryModel, EconomicModel
from .loss_functions import MultiObjectiveLoss
from .jax_optimizer import DifferentiableOptimizer


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""

    # JIT compilation settings
    enable_jit: bool = True
    static_argnums: Tuple[int, ...] = ()
    donate_argnums: Tuple[int, ...] = ()

    # Memory optimization
    enable_memory_efficiency: bool = True
    gradient_checkpointing: bool = False
    preallocate_arrays: bool = True

    # Vectorization settings
    enable_vectorization: bool = True
    batch_size: int = 32
    parallel_evaluations: bool = True

    # Compilation cache
    enable_compilation_cache: bool = True
    cache_directory: Optional[str] = None

    # Performance monitoring
    enable_profiling: bool = False
    benchmark_iterations: int = 100
    warmup_iterations: int = 10


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    compilation_time: float = 0.0
    execution_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0
    speedup_factor: float = 1.0
    cache_hit_rate: float = 0.0
    function_call_counts: Dict[str, int] = field(default_factory=dict)


class JITOptimizer:
    """
    Advanced JIT compilation optimizer for differentiable optimization functions.

    This class provides sophisticated JIT compilation strategies including
    function composition, fusion optimizations, and adaptive compilation.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        Initialize JIT optimizer.

        Args:
            config: Performance optimization configuration
        """
        # Import JAX here to get the updated availability status
        try:
            import jax
            import jax.numpy as jnp
            from jax import grad, jit, vmap, pmap, lax, tree_util

            self._jax_available = True
        except ImportError:
            self._jax_available = False

        if not self._jax_available:
            raise ImportError("JAX is required for JIT optimization")

        self.config = config or PerformanceConfig()
        self.compiled_functions: Dict[str, Callable] = {}
        self.compilation_cache: Dict[str, Any] = {}
        self.metrics = PerformanceMetrics()

        # Initialize compilation cache if enabled
        if self.config.enable_compilation_cache:
            self._setup_compilation_cache()

    def _setup_compilation_cache(self):
        """Setup JAX compilation cache for faster recompilation."""
        try:
            from jax.experimental.compilation_cache import initialize_cache

            if self.config.cache_directory:
                initialize_cache(self.config.cache_directory)
            else:
                initialize_cache("/tmp/jax_cache")
        except Exception as e:
            print(f"Warning: Could not initialize compilation cache: {e}")

    def compile_objective_function(
        self, objective_fn: Callable, constraint_fn: Optional[Callable] = None
    ) -> Callable:
        """
        Compile objective function with advanced JIT optimizations.

        Args:
            objective_fn: Objective function to compile
            constraint_fn: Optional constraint function

        Returns:
            Compiled objective function
        """
        cache_key = f"objective_{id(objective_fn)}"

        if cache_key in self.compiled_functions:
            self.metrics.cache_hit_rate += 1
            return self.compiled_functions[cache_key]

        start_time = time.time()

        if constraint_fn is not None:
            # Compose objective and constraint functions
            def combined_function(x):
                obj_val = objective_fn(x)
                constraint_val = constraint_fn(x)
                return obj_val + constraint_val

            compiled_fn = jax.jit(
                combined_function,
                static_argnums=self.config.static_argnums,
                donate_argnums=self.config.donate_argnums,
            )
        else:
            compiled_fn = jax.jit(
                objective_fn,
                static_argnums=self.config.static_argnums,
                donate_argnums=self.config.donate_argnums,
            )

        compilation_time = time.time() - start_time
        self.metrics.compilation_time += compilation_time

        self.compiled_functions[cache_key] = compiled_fn
        return compiled_fn

    def compile_gradient_function(self, objective_fn: Callable) -> Callable:
        """
        Compile gradient function with optimization.

        Args:
            objective_fn: Objective function

        Returns:
            Compiled gradient function
        """
        cache_key = f"gradient_{id(objective_fn)}"

        if cache_key in self.compiled_functions:
            return self.compiled_functions[cache_key]

        gradient_fn = jax.grad(objective_fn)
        compiled_gradient = jax.jit(
            gradient_fn,
            static_argnums=self.config.static_argnums,
            donate_argnums=self.config.donate_argnums,
        )

        self.compiled_functions[cache_key] = compiled_gradient
        return compiled_gradient

    def compile_hessian_function(self, objective_fn: Callable) -> Callable:
        """
        Compile Hessian function with optimization.

        Args:
            objective_fn: Objective function

        Returns:
            Compiled Hessian function
        """
        cache_key = f"hessian_{id(objective_fn)}"

        if cache_key in self.compiled_functions:
            return self.compiled_functions[cache_key]

        hessian_fn = jax.hessian(objective_fn)
        compiled_hessian = jax.jit(
            hessian_fn, static_argnums=self.config.static_argnums
        )

        self.compiled_functions[cache_key] = compiled_hessian
        return compiled_hessian

    def create_vectorized_function(
        self, base_fn: Callable, in_axes: Union[int, Tuple, None] = 0
    ) -> Callable:
        """
        Create vectorized version of function for batch processing.

        Args:
            base_fn: Base function to vectorize
            in_axes: Vectorization axes specification

        Returns:
            Vectorized function
        """
        cache_key = f"vectorized_{id(base_fn)}_{in_axes}"

        if cache_key in self.compiled_functions:
            return self.compiled_functions[cache_key]

        vectorized_fn = jax.vmap(base_fn, in_axes=in_axes)
        compiled_vectorized = jax.jit(vectorized_fn)

        self.compiled_functions[cache_key] = compiled_vectorized
        return compiled_vectorized

    def create_parallel_function(
        self, base_fn: Callable, axis_name: str = "batch"
    ) -> Callable:
        """
        Create parallel version of function for multi-device execution.

        Args:
            base_fn: Base function to parallelize
            axis_name: Name of the parallel axis

        Returns:
            Parallel function
        """
        if len(jax.devices()) <= 1:
            # Fall back to vectorized version if only one device
            return self.create_vectorized_function(base_fn)

        cache_key = f"parallel_{id(base_fn)}_{axis_name}"

        if cache_key in self.compiled_functions:
            return self.compiled_functions[cache_key]

        parallel_fn = jax.pmap(base_fn, axis_name=axis_name)

        self.compiled_functions[cache_key] = parallel_fn
        return parallel_fn


class BatchOptimizer:
    """
    Batch optimization utilities for efficient parameter space exploration.

    This class provides vectorized operations for optimizing multiple
    parameter sets simultaneously.
    """

    def __init__(
        self,
        trajectory_model: TrajectoryModel,
        economic_model: EconomicModel,
        loss_function: MultiObjectiveLoss,
        config: Optional[PerformanceConfig] = None,
    ):
        """
        Initialize batch optimizer.

        Args:
            trajectory_model: JAX trajectory model
            economic_model: JAX economic model
            loss_function: Multi-objective loss function
            config: Performance configuration
        """
        self.trajectory_model = trajectory_model
        self.economic_model = economic_model
        self.loss_function = loss_function
        self.config = config or PerformanceConfig()

        self.jit_optimizer = JITOptimizer(config)
        self._setup_batch_functions()

    def _setup_batch_functions(self):
        """Setup vectorized batch processing functions."""
        # Vectorized loss function
        self.batch_loss_fn = self.jit_optimizer.create_vectorized_function(
            self.loss_function.compute_loss, in_axes=0
        )

        # Vectorized gradient function
        gradient_fn = jax.grad(self.loss_function.compute_loss)
        self.batch_gradient_fn = self.jit_optimizer.create_vectorized_function(
            gradient_fn, in_axes=0
        )

        # Vectorized objective breakdown
        self.batch_objectives_fn = self.jit_optimizer.create_vectorized_function(
            self.loss_function.compute_raw_objectives, in_axes=0
        )

    def evaluate_batch(self, parameter_batch: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Evaluate loss function for a batch of parameters.

        Args:
            parameter_batch: Batch of parameter vectors [batch_size, 3]

        Returns:
            Dictionary with batch evaluation results
        """
        start_time = time.time()

        # Compute batch loss values
        batch_losses = self.batch_loss_fn(parameter_batch)

        # Compute batch gradients
        batch_gradients = self.batch_gradient_fn(parameter_batch)

        # Compute batch objectives
        batch_objectives = self.batch_objectives_fn(parameter_batch)

        execution_time = time.time() - start_time
        self.jit_optimizer.metrics.execution_time += execution_time

        return {
            "losses": batch_losses,
            "gradients": batch_gradients,
            "objectives": batch_objectives,
            "execution_time": execution_time,
            "batch_size": parameter_batch.shape[0],
        }

    def optimize_batch(
        self,
        initial_parameters: jnp.ndarray,
        num_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Perform batch gradient descent optimization.

        Args:
            initial_parameters: Initial parameter batch [batch_size, 3]
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent

        Returns:
            Batch optimization results
        """
        current_params = initial_parameters
        loss_history = []

        # Setup optimizer
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_state = opt_init(current_params)

        for i in range(num_iterations):
            # Compute gradients
            gradients = self.batch_gradient_fn(current_params)

            # Update parameters
            opt_state = opt_update(i, gradients, opt_state)
            current_params = get_params(opt_state)

            # Compute current losses
            current_losses = self.batch_loss_fn(current_params)
            loss_history.append(float(jnp.mean(current_losses)))

            # Early stopping check
            if i > 10 and abs(loss_history[-1] - loss_history[-10]) < 1e-8:
                break

        return {
            "final_parameters": current_params,
            "final_losses": self.batch_loss_fn(current_params),
            "loss_history": loss_history,
            "iterations": i + 1,
            "converged": i < num_iterations - 1,
        }


class MemoryOptimizer:
    """
    Memory optimization utilities for large-scale differentiable optimization.

    This class provides memory-efficient implementations of gradient computations
    and optimization routines.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        Initialize memory optimizer.

        Args:
            config: Performance configuration
        """
        self.config = config or PerformanceConfig()
        self.memory_pools: Dict[str, List[jnp.ndarray]] = {}

    def create_memory_efficient_gradient(
        self, objective_fn: Callable, chunk_size: int = 32
    ) -> Callable:
        """
        Create memory-efficient gradient function using chunking.

        Args:
            objective_fn: Objective function
            chunk_size: Size of chunks for gradient computation

        Returns:
            Memory-efficient gradient function
        """

        def chunked_gradient(parameters):
            if parameters.ndim == 1:
                # Single parameter vector
                return jax.grad(objective_fn)(parameters)

            # Batch of parameters - process in chunks
            batch_size = parameters.shape[0]
            gradients = []

            for i in range(0, batch_size, chunk_size):
                chunk_end = min(i + chunk_size, batch_size)
                chunk_params = parameters[i:chunk_end]

                chunk_grads = jax.vmap(jax.grad(objective_fn))(chunk_params)
                gradients.append(chunk_grads)

            return jnp.concatenate(gradients, axis=0)

        return jax.jit(chunked_gradient)

    def preallocate_workspace(
        self,
        workspace_name: str,
        shapes: List[Tuple[int, ...]],
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        Preallocate memory workspace for optimization.

        Args:
            workspace_name: Name of the workspace
            shapes: List of array shapes to preallocate
            dtype: Data type for arrays
        """
        if workspace_name not in self.memory_pools:
            self.memory_pools[workspace_name] = []

        for shape in shapes:
            array = jnp.zeros(shape, dtype=dtype)
            self.memory_pools[workspace_name].append(array)

    def get_workspace_array(self, workspace_name: str, index: int) -> jnp.ndarray:
        """
        Get preallocated array from workspace.

        Args:
            workspace_name: Name of the workspace
            index: Index of array in workspace

        Returns:
            Preallocated array
        """
        if workspace_name in self.memory_pools and index < len(
            self.memory_pools[workspace_name]
        ):
            return self.memory_pools[workspace_name][index]
        else:
            raise ValueError(f"Workspace {workspace_name} or index {index} not found")


class PerformanceBenchmark:
    """
    Performance benchmarking and profiling utilities.

    This class provides tools for measuring and analyzing the performance
    of differentiable optimization functions.
    """

    def __init__(self, config: Optional[PerformanceConfig] = None):
        """
        Initialize performance benchmark.

        Args:
            config: Performance configuration
        """
        self.config = config or PerformanceConfig()
        self.benchmark_results: Dict[str, PerformanceMetrics] = {}

    def benchmark_function(
        self,
        function: Callable,
        inputs: jnp.ndarray,
        function_name: str,
        warmup_iterations: Optional[int] = None,
        benchmark_iterations: Optional[int] = None,
    ) -> PerformanceMetrics:
        """
        Benchmark function performance.

        Args:
            function: Function to benchmark
            inputs: Input data for function
            function_name: Name for storing results
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations

        Returns:
            Performance metrics
        """
        warmup_iter = warmup_iterations or self.config.warmup_iterations
        bench_iter = benchmark_iterations or self.config.benchmark_iterations

        # Warmup phase
        for _ in range(warmup_iter):
            _ = function(inputs)

        # Benchmark phase
        start_time = time.time()

        for _ in range(bench_iter):
            result = function(inputs)

        # Ensure computation is complete
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / bench_iter
        throughput = bench_iter / total_time

        metrics = PerformanceMetrics(execution_time=avg_time, throughput=throughput)

        self.benchmark_results[function_name] = metrics
        return metrics

    def compare_implementations(
        self, implementations: Dict[str, Callable], inputs: jnp.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of different implementations.

        Args:
            implementations: Dictionary of implementation functions
            inputs: Input data for comparison

        Returns:
            Performance comparison results
        """
        results = {}
        baseline_time = None

        for name, impl in implementations.items():
            metrics = self.benchmark_function(impl, inputs, name)

            results[name] = {
                "execution_time": metrics.execution_time,
                "throughput": metrics.throughput,
                "relative_speedup": 1.0,
            }

            if baseline_time is None:
                baseline_time = metrics.execution_time
            else:
                speedup = baseline_time / metrics.execution_time
                results[name]["relative_speedup"] = speedup

        return results

    def profile_memory_usage(
        self, function: Callable, inputs: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Profile memory usage of function.

        Args:
            function: Function to profile
            inputs: Input data

        Returns:
            Memory usage statistics
        """
        # JAX memory profiling is limited, so this is a simplified implementation
        # In practice, you would use tools like JAX's memory profiler

        # Get initial memory state
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run function
            result = function(inputs)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()

            # Get final memory state
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            return {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": final_memory - initial_memory,
            }
        except ImportError:
            return {"error": "psutil not available for memory profiling"}


# Utility functions for performance optimization
def optimize_differentiable_optimizer(
    optimizer: DifferentiableOptimizer, config: Optional[PerformanceConfig] = None
) -> DifferentiableOptimizer:
    """
    Apply performance optimizations to a DifferentiableOptimizer instance.

    Args:
        optimizer: DifferentiableOptimizer to optimize
        config: Performance optimization configuration

    Returns:
        Optimized DifferentiableOptimizer instance
    """
    perf_config = config or PerformanceConfig()
    jit_optimizer = JITOptimizer(perf_config)

    # Compile objective function
    if hasattr(optimizer, "objective_function"):
        optimizer.objective_function = jit_optimizer.compile_objective_function(
            optimizer.objective_function
        )

    # Compile gradient function if available
    if hasattr(optimizer, "_gradient_function"):
        optimizer._gradient_function = jit_optimizer.compile_gradient_function(
            optimizer.objective_function
        )

    # Add performance tracking
    optimizer._performance_metrics = jit_optimizer.metrics

    return optimizer


def create_performance_optimized_loss_function(
    loss_function: MultiObjectiveLoss, config: Optional[PerformanceConfig] = None
) -> MultiObjectiveLoss:
    """
    Create performance-optimized version of loss function.

    Args:
        loss_function: MultiObjectiveLoss to optimize
        config: Performance optimization configuration

    Returns:
        Performance-optimized MultiObjectiveLoss instance
    """
    perf_config = config or PerformanceConfig()
    jit_optimizer = JITOptimizer(perf_config)

    # Compile core functions
    loss_function.compute_loss = jit_optimizer.compile_objective_function(
        loss_function.compute_loss
    )

    # Enable batch processing if configured
    if perf_config.enable_vectorization:
        batch_optimizer = BatchOptimizer(
            loss_function.trajectory_model,
            loss_function.economic_model,
            loss_function,
            perf_config,
        )
        loss_function._batch_optimizer = batch_optimizer

    return loss_function


def benchmark_optimization_performance(
    trajectory_model: TrajectoryModel,
    economic_model: EconomicModel,
    test_parameters: jnp.ndarray,
    config: Optional[PerformanceConfig] = None,
) -> Dict[str, Any]:
    """
    Comprehensive performance benchmark for optimization components.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        test_parameters: Test parameter sets
        config: Performance configuration

    Returns:
        Comprehensive benchmark results
    """
    perf_config = config or PerformanceConfig()
    benchmark = PerformanceBenchmark(perf_config)

    # Create different implementations
    implementations = {}

    # Standard implementation
    implementations["standard"] = trajectory_model._trajectory_cost

    # JIT-compiled implementation
    jit_optimizer = JITOptimizer(perf_config)
    implementations["jit_compiled"] = jit_optimizer.compile_objective_function(
        trajectory_model._trajectory_cost
    )

    # Vectorized implementation
    implementations["vectorized"] = jit_optimizer.create_vectorized_function(
        trajectory_model._trajectory_cost, in_axes=0
    )

    # Benchmark all implementations
    single_param = test_parameters[0] if test_parameters.ndim > 1 else test_parameters
    batch_params = (
        test_parameters if test_parameters.ndim > 1 else test_parameters[None, :]
    )

    results = {
        "single_parameter_benchmark": benchmark.compare_implementations(
            {k: v for k, v in implementations.items() if k != "vectorized"},
            single_param,
        ),
        "batch_parameter_benchmark": benchmark.compare_implementations(
            {"vectorized": implementations["vectorized"]}, batch_params
        ),
        "memory_profile": benchmark.profile_memory_usage(
            implementations["jit_compiled"], single_param
        ),
    }

    return results
