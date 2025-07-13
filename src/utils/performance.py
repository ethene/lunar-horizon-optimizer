#!/usr/bin/env python3
"""
Performance optimization utilities for Lunar Horizon Optimizer.

This module provides JIT compilation decorators, parallel processing utilities,
and performance monitoring tools to accelerate real calculations.
"""

import functools
import time
import warnings
from typing import Any, Callable, Optional

# Optional imports for speed-up packages
try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available - JIT compilation disabled", stacklevel=2)

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("Joblib not available - parallel processing limited", stacklevel=2)

try:
    import dask

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    warnings.warn("Dask not available - large-scale parallelism disabled", stacklevel=2)


def jit_compile(
    func: Optional[Callable] = None, *, nopython: bool = True, cache: bool = True
):
    """
    JIT compilation decorator using Numba when available.

    Args:
        func: Function to compile
        nopython: Use nopython mode for best performance
        cache: Enable caching of compiled functions

    Returns:
        Compiled function or original function if Numba unavailable
    """

    def decorator(f):
        if HAS_NUMBA:
            try:
                return numba.jit(nopython=nopython, cache=cache)(f)
            except Exception as e:
                warnings.warn(
                    f"Numba compilation failed for {f.__name__}: {e}", stacklevel=2
                )
                return f
        return f

    if func is None:
        return decorator
    return decorator(func)


def vectorize(
    func: Optional[Callable] = None, *, nopython: bool = True, cache: bool = True
):
    """
    Vectorization decorator using Numba when available.

    Args:
        func: Function to vectorize
        nopython: Use nopython mode for best performance
        cache: Enable caching of compiled functions

    Returns:
        Vectorized function or original function if Numba unavailable
    """

    def decorator(f):
        if HAS_NUMBA:
            try:
                return numba.vectorize(nopython=nopython, cache=cache)(f)
            except Exception as e:
                warnings.warn(
                    f"Numba vectorization failed for {f.__name__}: {e}", stacklevel=2
                )
                return f
        return f

    if func is None:
        return decorator
    return decorator(func)


def parallel_map(func: Callable, iterable: Any, n_jobs: int = -1, verbose: int = 0):
    """
    Parallel map using joblib when available.

    Args:
        func: Function to apply
        iterable: Input data
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level

    Returns:
        Results list
    """
    if HAS_JOBLIB:
        try:
            return joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
                joblib.delayed(func)(item) for item in iterable
            )
        except Exception as e:
            warnings.warn(f"Joblib parallel execution failed: {e}", stacklevel=2)

    # Fallback to standard map
    return list(map(func, iterable))


class PerformanceMonitor:
    """Monitor performance of critical operations."""

    def __init__(self):
        self.timings = {}
        self.call_counts = {}

    def time_function(self, name: str):
        """Decorator to time function execution."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                if name not in self.timings:
                    self.timings[name] = []
                    self.call_counts[name] = 0

                self.timings[name].append(elapsed)
                self.call_counts[name] += 1

                return result

            return wrapper

        return decorator

    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "call_count": self.call_counts[name],
                    "calls_per_second": (
                        self.call_counts[name] / sum(times) if sum(times) > 0 else 0
                    ),
                }
        return stats

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()

    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print("\n📊 Performance Statistics:")
        print("=" * 50)
        for name, data in stats.items():
            print(f"{name}:")
            print(f"  Total time: {data['total_time']:.3f}s")
            print(f"  Average: {data['avg_time']:.3f}s")
            print(f"  Calls: {data['call_count']}")
            print(f"  Rate: {data['calls_per_second']:.1f} calls/sec")
            print()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def enable_performance_optimizations():
    """
    Enable all available performance optimizations.

    Returns:
        dict: Status of optimization packages
    """
    status = {
        "numba_jit": HAS_NUMBA,
        "joblib_parallel": HAS_JOBLIB,
        "dask_distributed": HAS_DASK,
    }

    if HAS_NUMBA:
        # Set number of threads for Numba
        import os

        if "NUMBA_NUM_THREADS" not in os.environ:
            import multiprocessing

            os.environ["NUMBA_NUM_THREADS"] = str(multiprocessing.cpu_count())

    return status


def get_optimization_status() -> dict:
    """Get status of all optimization packages."""
    return {
        "numba": {
            "available": HAS_NUMBA,
            "version": numba.__version__ if HAS_NUMBA else None,
            "threads": numba.config.NUMBA_NUM_THREADS if HAS_NUMBA else None,
        },
        "joblib": {
            "available": HAS_JOBLIB,
            "version": joblib.__version__ if HAS_JOBLIB else None,
        },
        "dask": {
            "available": HAS_DASK,
            "version": dask.__version__ if HAS_DASK else None,
        },
    }


# Optimized mathematical functions
@jit_compile(nopython=True, cache=True)
def fast_norm(vector):
    """Fast vector norm calculation."""
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5


@jit_compile(nopython=True, cache=True)
def fast_dot(a, b):
    """Fast dot product calculation."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit_compile(nopython=True, cache=True)
def fast_cross(a, b):
    """Fast cross product calculation."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@vectorize(nopython=True, cache=True)
def fast_sqrt(x):
    """Fast square root for arrays."""
    return x**0.5


# Export commonly used functions
__all__ = [
    "jit_compile",
    "vectorize",
    "parallel_map",
    "PerformanceMonitor",
    "performance_monitor",
    "enable_performance_optimizations",
    "get_optimization_status",
    "fast_norm",
    "fast_dot",
    "fast_cross",
    "fast_sqrt",
]
