# Ray Parallelization for Global Optimization

## Overview

The Lunar Horizon Optimizer now includes Ray-based parallelization for the global optimization module, enabling efficient multi-core utilization during PyGMO population evaluation. This significantly improves performance for computationally intensive optimization runs.

**Status**: ✅ **Production Ready**  
**Performance Improvement**: 2-8x speedup on multi-core systems  
**Ray Version**: 2.8.0+  
**Fallback**: Graceful degradation to sequential optimization when Ray unavailable

## Architecture

### Ray Actor System

The parallelization uses Ray actors to distribute fitness evaluations across multiple CPU cores:

```
GlobalOptimizer.optimize()
├── RayParallelOptimizer (extends GlobalOptimizer)
│   ├── FitnessWorker actors (N workers)
│   │   ├── Pre-loaded resources (SPICE kernels, trajectory models)
│   │   ├── Local caching for efficiency
│   │   └── Batch evaluation methods
│   └── Population chunking and result aggregation
└── Standard PyGMO evolution with parallel fitness evaluation
```

### Key Components

1. **FitnessWorker Actor**: Ray remote actor that pre-loads heavy resources and evaluates fitness in batches
2. **RayParallelOptimizer**: Extended GlobalOptimizer with Ray-based population evaluation
3. **Batch Processing**: Chunks population into batches to minimize Ray overhead
4. **Graceful Fallback**: Automatically falls back to sequential when Ray unavailable

## Quick Start

### Installation

```bash
# Install Ray with default components
pip install ray[default]>=2.8.0

# Or install from requirements file
pip install -r requirements-ray.txt
```

### Basic Usage

```python
from src.optimization.ray_optimizer import RayParallelOptimizer, create_ray_optimizer
from src.optimization.global_optimizer import LunarMissionProblem

# Option 1: Direct instantiation
problem = LunarMissionProblem()
optimizer = RayParallelOptimizer(
    problem=problem,
    population_size=100,
    num_generations=50,
    num_workers=8  # Use 8 CPU cores
)

results = optimizer.optimize()

# Option 2: Factory function with automatic fallback
optimizer = create_ray_optimizer(
    optimizer_config={
        'population_size': 100,
        'num_generations': 50,
        'num_workers': 8
    }
)

results = optimizer.optimize()
```

### Advanced Configuration

```python
# Ray-specific configuration
ray_config = {
    'ignore_reinit_error': True,
    'num_cpus': 8,
    'object_store_memory': 2_000_000_000  # 2GB object store
}

# Optimization configuration
optimizer_config = {
    'population_size': 200,
    'num_generations': 100,
    'num_workers': 8,
    'chunk_size': 25  # 25 individuals per batch
}

optimizer = RayParallelOptimizer(
    population_size=optimizer_config['population_size'],
    num_generations=optimizer_config['num_generations'],
    num_workers=optimizer_config['num_workers'],
    chunk_size=optimizer_config['chunk_size'],
    ray_config=ray_config
)

results = optimizer.optimize(verbose=True)

# Access Ray-specific statistics
ray_stats = results['ray_stats']
print(f"Ray used: {ray_stats['ray_used']}")
print(f"Workers: {ray_stats['num_workers']}")
print(f"Setup time: {ray_stats['setup_time']:.2f}s")

for worker_stat in ray_stats['worker_stats']:
    print(f"Worker {worker_stat['worker_id']}: "
          f"{worker_stat['evaluations']} evaluations, "
          f"{worker_stat['cache_hit_rate']:.1%} cache hit rate")
```

## Performance Optimization

### Worker Configuration

**Number of Workers**: 
- Default: `os.cpu_count()` 
- Recommended: Leave 1-2 cores free for system processes
- Example: 8-core system → use 6-7 workers

**Chunk Size**:
- Default: `population_size // (num_workers * 2)`
- Smaller chunks: Better load balancing, higher overhead
- Larger chunks: Lower overhead, potential load imbalance
- Recommended: 10-50 individuals per chunk

### Memory Optimization

**Resource Pre-loading**: Workers pre-load heavy resources (SPICE kernels, trajectory models) during initialization to avoid repeated loading.

**Local Caching**: Each worker maintains its own cache of fitness evaluations to minimize redundant calculations.

**Memory Management**:
```python
ray_config = {
    'object_store_memory': 2_000_000_000,  # 2GB for shared objects
    'num_cpus': 8,
    'ignore_reinit_error': True
}
```

### Performance Tuning

```python
# High-performance configuration for large problems
optimizer = RayParallelOptimizer(
    population_size=500,        # Large population
    num_generations=200,        # Many generations
    num_workers=14,            # Use most cores (16-core system)
    chunk_size=20,             # Optimal chunk size
    ray_config={
        'object_store_memory': 4_000_000_000,  # 4GB object store
        'num_cpus': 16,
        'ignore_reinit_error': True
    }
)
```

## Benchmarking

### Running Benchmarks

```bash
# Basic benchmark: 100 individuals, 8 workers
cd /path/to/lunar-horizon-optimizer
python benchmarks/ray_optimization_benchmark.py --individuals 100 --workers 8

# Comprehensive comparison
python benchmarks/ray_optimization_benchmark.py --compare-all --runs 5

# Profile fitness function performance
python benchmarks/ray_optimization_benchmark.py --profile --profile-evals 1000

# Large-scale benchmark
python benchmarks/ray_optimization_benchmark.py \
    --individuals 500 --generations 50 --workers 12 --runs 3

# Save results to file
python benchmarks/ray_optimization_benchmark.py \
    --compare-all --output results/ray_benchmark.json
```

### Expected Performance

**Typical Speedups** (8-core system):
- Small problems (pop=50, gen=20): 1.5-2.5x
- Medium problems (pop=100, gen=50): 2.5-4.0x  
- Large problems (pop=500, gen=100): 3.5-6.0x

**Efficiency Factors**:
- Fitness function complexity (higher = better speedup)
- Population size (larger = better speedup)
- Cache hit rate (higher = better overall performance)
- System load and memory availability

### Performance Analysis

```python
# Analyze benchmark results
results = optimizer.optimize()
ray_stats = results['ray_stats']

# Calculate parallel efficiency
total_worker_time = sum(ws['total_time'] for ws in ray_stats['worker_stats'])
wall_clock_time = results['optimization_time']
parallel_efficiency = (total_worker_time / wall_clock_time) / ray_stats['num_workers']

print(f"Parallel efficiency: {parallel_efficiency:.1%}")
print(f"Speedup estimate: {parallel_efficiency * ray_stats['num_workers']:.1f}x")
```

## Integration Examples

### Drop-in Replacement

```python
# Before: Sequential optimization
from src.optimization.global_optimizer import GlobalOptimizer

optimizer = GlobalOptimizer(population_size=100, num_generations=50)
results = optimizer.optimize()

# After: Ray parallel optimization
from src.optimization.ray_optimizer import create_ray_optimizer

optimizer = create_ray_optimizer(
    optimizer_config={'population_size': 100, 'num_generations': 50}
)
results = optimizer.optimize()  # Same interface, parallel execution
```

### Conditional Ray Usage

```python
from src.optimization.ray_optimizer import RAY_AVAILABLE, create_ray_optimizer
from src.optimization.global_optimizer import GlobalOptimizer

def create_optimizer(use_ray=True, **config):
    """Create optimizer with optional Ray parallelization."""
    if use_ray and RAY_AVAILABLE:
        return create_ray_optimizer(optimizer_config=config)
    else:
        return GlobalOptimizer(**config)

# Usage
optimizer = create_optimizer(
    use_ray=True,  # Will fall back to sequential if Ray unavailable
    population_size=200,
    num_generations=100
)
```

### Batch Processing Workflow

```python
# Process multiple optimization runs in parallel
import ray
from src.optimization.ray_optimizer import RayParallelOptimizer

@ray.remote
def run_optimization_scenario(scenario_config):
    """Run single optimization scenario."""
    optimizer = RayParallelOptimizer(**scenario_config)
    return optimizer.optimize()

# Define multiple scenarios
scenarios = [
    {'population_size': 100, 'num_generations': 50, 'seed': 42},
    {'population_size': 150, 'num_generations': 75, 'seed': 43},
    {'population_size': 200, 'num_generations': 100, 'seed': 44},
]

# Run scenarios in parallel
if not ray.is_initialized():
    ray.init()

futures = [run_optimization_scenario.remote(scenario) for scenario in scenarios]
results = ray.get(futures)

ray.shutdown()
```

## Troubleshooting

### Common Issues

**1. Ray Initialization Errors**
```python
# Solution: Use ignore_reinit_error
ray_config = {'ignore_reinit_error': True}
optimizer = RayParallelOptimizer(ray_config=ray_config)
```

**2. Memory Issues**
```python
# Solution: Increase object store memory
ray_config = {
    'object_store_memory': 4_000_000_000,  # 4GB
    'ignore_reinit_error': True
}
```

**3. Worker Startup Failures**
```bash
# Check Ray cluster status
ray status

# View Ray logs
ray logs cluster

# Check system resources
htop  # or top on macOS
```

**4. Performance Issues**
```python
# Debug worker statistics
results = optimizer.optimize()
worker_stats = results['ray_stats']['worker_stats']

for stats in worker_stats:
    print(f"Worker {stats['worker_id']}:")
    print(f"  Evaluations: {stats['evaluations']}")
    print(f"  Avg time: {stats['avg_time_per_eval']:.4f}s")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

### Fallback Behavior

The system gracefully handles Ray unavailability:

1. **Import Failure**: Falls back to `GlobalOptimizer`
2. **Initialization Failure**: Disables Ray, uses sequential evaluation
3. **Runtime Errors**: Logs warnings, continues with available workers
4. **Worker Failures**: Redistributes work to remaining workers

### Environment Verification

```bash
# Check Ray installation
python -c "import ray; print(f'Ray version: {ray.__version__}')"

# Test Ray functionality
python -c "
import ray
ray.init()
print('Ray initialized successfully')
print(f'Available CPUs: {ray.available_resources()["CPU"]}')
ray.shutdown()
"

# Run Ray optimizer test
python -m pytest tests/test_ray_optimization.py -v
```

## API Reference

### RayParallelOptimizer

**Constructor Parameters:**
- `problem`: LunarMissionProblem instance
- `population_size`: Population size for NSGA-II
- `num_generations`: Number of optimization generations
- `seed`: Random seed for reproducibility
- `num_workers`: Number of Ray workers (default: CPU count)
- `chunk_size`: Individuals per worker batch (default: auto)
- `ray_config`: Ray initialization configuration

**Methods:**
- `optimize(verbose=True)`: Run optimization with Ray parallelization
- `_initialize_ray_workers()`: Initialize Ray actor pool
- `_shutdown_ray_workers()`: Clean up Ray workers and collect statistics

### FitnessWorker (Ray Actor)

**Constructor Parameters:**
- `cost_factors`: Cost model parameters
- `min_earth_alt`, `max_earth_alt`: Earth orbit bounds [km]
- `min_moon_alt`, `max_moon_alt`: Moon orbit bounds [km]
- `min_transfer_time`, `max_transfer_time`: Transfer time bounds [days]
- `reference_epoch`: Reference epoch [days since J2000]
- `worker_id`: Unique worker identifier

**Remote Methods:**
- `evaluate_batch(population_chunk)`: Evaluate fitness for batch of individuals
- `get_stats()`: Get worker performance statistics

### Utility Functions

- `create_ray_optimizer(problem_config, optimizer_config, ray_config)`: Factory function with fallback
- `RAY_AVAILABLE`: Boolean indicating Ray availability

---

For additional examples and advanced usage, see:
- `/benchmarks/ray_optimization_benchmark.py`
- `/tests/test_ray_optimization.py`
- `/src/optimization/ray_optimizer.py`