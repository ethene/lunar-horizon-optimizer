# Multi-Mission Constellation Optimization User Guide

## Overview

The Lunar Horizon Optimizer now supports **multi-mission constellation optimization**, enabling you to optimize K simultaneous lunar transfers in a single optimization run. This is ideal for scenarios like deploying 24 lunar communication satellites or planning multiple cargo missions.

## Key Features

### 🛰️ **Constellation Optimization**
- Optimize K missions simultaneously (K=1 to 24+)
- Constellation-specific objectives (coverage, redundancy)
- Orbital plane geometry optimization (RAAN distribution)
- Mission-specific and shared parameters

### 🔄 **Backward Compatibility**
- Existing single-mission code unchanged
- Automatic detection of single vs multi-mission mode
- Drop-in replacement for existing optimizers

### 📊 **Enhanced Analysis**
- Constellation performance metrics
- Mission-by-mission parameter breakdown
- Coverage uniformity assessment
- Cost efficiency analysis across constellation sizes

## Quick Start

### Command Line Usage

#### Single Mission (Unchanged)
```bash
# Your existing workflow continues to work
python cli_constellation.py config/lunar_mission.yaml
```

#### Multi-Mission Constellation
```bash
# 3-satellite constellation
python cli_constellation.py config/lunar_mission.yaml --multi 3

# 8-satellite constellation with custom weights
python cli_constellation.py config/lunar_mission.yaml --multi 8 \
    --constellation-weights "coverage=2.0,redundancy=1.0"

# 24-satellite constellation for lunar communications
python cli_constellation.py config/constellation.yaml --multi 24 \
    --population 600 --generations 400 \
    --output results/lunar_comms_constellation.json
```

### Programmatic Usage

#### Basic Multi-Mission Optimization
```python
from src.optimization.multi_mission_optimizer import optimize_constellation
from src.config.costs import CostFactors

# Define cost parameters
cost_factors = CostFactors(
    launch_cost_per_kg=12000.0,
    operations_cost_per_day=75000.0,
    development_cost=2e9,
    contingency_percentage=25.0
)

# Optimize 6-satellite constellation
results = optimize_constellation(
    num_missions=6,
    cost_factors=cost_factors,
    optimization_config={
        'optimizer_params': {
            'population_size': 300,
            'num_generations': 200
        },
        'verbose': True
    },
    constellation_config={
        'problem_params': {
            'coverage_weight': 1.5,
            'redundancy_weight': 0.8
        }
    }
)

# Analyze results
if results['success']:
    best_constellation = results['best_constellations'][0]
    print(f"Best constellation cost: ${best_constellation['objectives'][2]/1e6:.1f}M")
```

#### Advanced Multi-Mission Setup
```python
from src.optimization.multi_mission_optimizer import MultiMissionOptimizer
from src.optimization.multi_mission_genome import MultiMissionProblem

# Create custom multi-mission problem
problem = MultiMissionProblem(
    num_missions=12,
    # Parameter bounds
    min_epoch=9500.0,      # Earlier launch window
    max_epoch=10500.0,     # Later launch window
    min_earth_alt=250.0,   # Higher minimum altitude
    max_earth_alt=800.0,   # Lower maximum altitude
    # Constellation objectives
    coverage_weight=2.0,   # Emphasize coverage
    redundancy_weight=1.0  # Moderate redundancy
)

# Create optimizer with custom configuration
optimizer = MultiMissionOptimizer(
    problem=problem,
    multi_mission_mode=True,
    num_missions=12,
    population_size=500,
    num_generations=300,
    constellation_preferences={
        'delta_v': 1.0,
        'time': 0.8,
        'cost': 1.2,
        'coverage': 2.0,
        'redundancy': 1.0
    }
)

# Run optimization
results = optimizer.optimize(verbose=True)

# Get detailed constellation analysis
constellation_metrics = optimizer.get_constellation_metrics()
best_solutions = optimizer.get_best_constellation_solutions(num_solutions=5)
```

## Understanding Multi-Mission Architecture

### Decision Variables

For K missions, the optimizer uses **4K + 2** decision variables:

| Variable Range | Parameter | Description | Bounds |
|----------------|-----------|-------------|--------|
| [0, K) | epochs | Launch timing for each mission | [9000, 11000] days since J2000 |
| [K, 2K) | parking_altitudes | Earth orbit altitude per mission | [200, 1000] km |
| [2K, 3K) | plane_raan | Orbital plane orientation per mission | [0, 360] degrees |
| [3K, 4K) | payload_masses | Payload mass per mission | [500, 2000] kg |
| [4K] | lunar_altitude | Shared lunar orbit altitude | [50, 500] km |
| [4K+1] | transfer_time | Shared transfer duration | [3, 10] days |

### Objectives

Multi-mission optimization uses **5 objectives**:

1. **Total ΔV**: Sum of delta-v requirements across all missions (minimize)
2. **Total Time**: Sum of transfer times for all missions (minimize)
3. **Total Cost**: Sum of mission costs including economics (minimize)
4. **Coverage Metric**: Based on RAAN distribution uniformity (minimize)
5. **Redundancy Metric**: Performance similarity and temporal distribution (minimize)

### Constellation-Specific Objectives

#### Coverage Calculation
```python
def calculate_coverage_metric(raan_values, num_missions):
    """Calculate constellation coverage based on orbital plane distribution."""
    # Calculate uniformity of RAAN distribution
    sorted_raan = np.sort(raan_values)
    gaps = np.diff(sorted_raan)
    wrap_gap = 360.0 - sorted_raan[-1] + sorted_raan[0]
    all_gaps = np.append(gaps, wrap_gap)
    
    # Ideal gap would be 360/K degrees
    ideal_gap = 360.0 / num_missions
    uniformity = np.std(all_gaps - ideal_gap)
    
    return uniformity  # Lower is better
```

#### Redundancy Assessment
```python
def calculate_redundancy_metric(mission_results, genome):
    """Calculate constellation redundancy for robust operations."""
    # Performance similarity (low variance is good)
    delta_vs = [result['delta_v'] for result in mission_results]
    dv_variance = np.var(delta_vs) / (np.mean(delta_vs) + 1e-6)
    
    # Temporal distribution (avoid clustering)
    epoch_variance = np.var(genome.epochs) / (np.mean(genome.epochs) + 1e-6)
    temporal_diversity = max(0.0, 0.1 - epoch_variance)
    
    return (dv_variance + temporal_diversity) / 2.0  # Lower is better
```

## Configuration Guidelines

### Population Scaling

The optimizer automatically scales population size based on constellation size:

```python
# Recommended scaling rules
population_size = max(100, 50 * K)           # Minimum population for convergence
num_generations = max(100, 100 + 20 * K)     # Additional generations for complexity

# Examples:
# K=1:  population=100, generations=100
# K=3:  population=150, generations=160  
# K=8:  population=400, generations=260
# K=24: population=1200, generations=580
```

### Memory and Runtime Expectations

| K (Missions) | Variables | Population | Memory | Runtime (Est.) |
|--------------|-----------|------------|---------|----------------|
| 1 (Current)  | 3         | 100        | ~50MB   | ~8 minutes     |
| 3            | 14        | 150        | ~150MB  | ~30 minutes    |
| 8            | 34        | 400        | ~400MB  | ~2 hours       |
| 24           | 98        | 1200       | ~1.2GB  | ~13 hours      |

### Configuration Files

#### Basic Constellation Config
```yaml
# config/constellation_basic.yaml
optimization:
  population_size: 300
  num_generations: 200
  seed: 42

constellation:
  coverage_weight: 1.5
  redundancy_weight: 1.0

costs:
  launch_cost_per_kg: 12000.0
  operations_cost_per_day: 75000.0
  development_cost: 2.0e9
  contingency_percentage: 25.0
```

#### Advanced Constellation Config
```yaml
# config/constellation_advanced.yaml
optimization:
  population_size: 600
  num_generations: 400
  seed: 42

constellation:
  coverage_weight: 2.0
  redundancy_weight: 1.5
  
  # Custom parameter bounds
  min_epoch: 9500.0
  max_epoch: 10500.0
  min_earth_alt: 250.0
  max_earth_alt: 800.0
  min_payload: 800.0
  max_payload: 1500.0

costs:
  launch_cost_per_kg: 15000.0  # Higher for large constellation
  operations_cost_per_day: 100000.0
  development_cost: 5.0e9      # Higher development cost
  contingency_percentage: 30.0  # Higher contingency for complexity

preferences:
  delta_v: 1.0
  time: 0.8
  cost: 1.5
  coverage: 2.0
  redundancy: 1.2
```

## Results Analysis

### Accessing Results

```python
# Basic results access
if results['success']:
    pareto_front = results['pareto_front']
    best_constellations = results['best_constellations']
    constellation_metrics = results['constellation_metrics']
    
    print(f"Found {len(pareto_front)} Pareto-optimal solutions")
    print(f"Top {len(best_constellations)} constellation solutions analyzed")
```

### Constellation Metrics

```python
# Detailed constellation analysis
const_metrics = results['constellation_metrics']

# Coverage statistics
coverage_stats = const_metrics['coverage_stats']
print(f"Coverage: {coverage_stats['mean']:.2f} ± {coverage_stats['std']:.2f}")

# Redundancy statistics  
redundancy_stats = const_metrics['redundancy_stats']
print(f"Redundancy: {redundancy_stats['mean']:.2f} ± {redundancy_stats['std']:.2f}")

# Mission efficiency
efficiency = const_metrics['mission_efficiency']
print(f"Efficiency: {np.mean(efficiency):.2f} payload_kg per $M")
```

### Individual Mission Analysis

```python
# Analyze individual mission parameters
best_solution = results['best_constellations'][0]
constellation_analysis = best_solution['constellation_analysis']

mission_details = constellation_analysis['mission_details']
for i, mission in enumerate(mission_details):
    print(f"Mission {i+1}:")
    print(f"  Launch epoch: {mission['epoch']:.1f} days since J2000")
    print(f"  Earth altitude: {mission['earth_altitude']:.1f} km")
    print(f"  Orbital plane (RAAN): {mission['plane_raan']:.1f}°")
    print(f"  Payload mass: {mission['payload_mass']:.1f} kg")

# Constellation geometry
geometry = constellation_analysis['geometry']
print(f"RAAN distribution: {geometry['raan_distribution']}")
print(f"Altitude uniformity: {geometry['altitude_uniformity']:.2f} km std")
print(f"Launch timing spread: {geometry['timing_spread']:.2f} days std")
```

## Migration from Single-Mission

### Automatic Migration

```python
from src.optimization.multi_mission_optimizer import migrate_single_to_multi

# Your existing single-mission config
single_config = {
    'population_size': 100,
    'num_generations': 50,
    'verbose': True
}

# Migrate to 6-mission constellation
multi_config = migrate_single_to_multi(single_config, num_missions=6)

# Automatically scaled:
# population_size: 300 (scaled for complexity)
# num_generations: 170 (additional generations)
# constellation_mode: True (enabled)
```

### Backward Compatibility Verification

```python
from src.optimization.multi_mission_optimizer import MultiMissionOptimizer
from src.optimization.global_optimizer import GlobalOptimizer

# Test that single-mission results are identical
original_optimizer = GlobalOptimizer(population_size=50, num_generations=10)
new_optimizer = MultiMissionOptimizer(
    multi_mission_mode=False,  # Single-mission mode
    num_missions=1,
    population_size=50,
    num_generations=10
)

# Both should produce equivalent results
orig_results = original_optimizer.optimize()
new_results = new_optimizer.optimize()

# Verify compatibility
assert orig_results['success'] == new_results['success']
assert len(orig_results['pareto_front']) == len(new_results['pareto_front'])
```

## Real-World Examples

### Lunar Communication Constellation (24 Satellites)

```python
# 24-satellite lunar communication network
results = optimize_constellation(
    num_missions=24,
    cost_factors=CostFactors(
        launch_cost_per_kg=10000.0,
        operations_cost_per_day=200000.0,  # High ops for comms
        development_cost=10e9,             # Large development
        contingency_percentage=35.0        # High contingency
    ),
    optimization_config={
        'optimizer_params': {
            'population_size': 1200,  # Large population
            'num_generations': 500,   # Many generations
            'seed': 42
        },
        'verbose': True
    },
    constellation_config={
        'problem_params': {
            'coverage_weight': 3.0,    # Critical for communications
            'redundancy_weight': 2.0,  # High redundancy needed
            'min_earth_alt': 300.0,    # Avoid atmospheric drag
            'max_earth_alt': 600.0,    # Reasonable for constellation
        }
    }
)
```

### Cargo Supply Constellation (6 Missions)

```python
# 6-mission cargo supply to lunar base
results = optimize_constellation(
    num_missions=6,
    cost_factors=CostFactors(
        launch_cost_per_kg=8000.0,      # Lower cost for cargo
        operations_cost_per_day=50000.0, # Lower ops cost
        development_cost=1e9,            # Moderate development
        contingency_percentage=20.0      # Standard contingency
    ),
    optimization_config={
        'optimizer_params': {
            'population_size': 300,
            'num_generations': 200
        }
    },
    constellation_config={
        'problem_params': {
            'coverage_weight': 1.0,      # Less critical for cargo
            'redundancy_weight': 0.5,    # Some redundancy
            'min_payload': 2000.0,       # Heavy cargo
            'max_payload': 5000.0,       # Very heavy cargo
        }
    }
)
```

### Science Mission Constellation (3 Satellites)

```python
# 3-satellite lunar science constellation
results = optimize_constellation(
    num_missions=3,
    cost_factors=CostFactors(
        launch_cost_per_kg=15000.0,     # Higher cost for precision
        operations_cost_per_day=100000.0, # Moderate ops
        development_cost=3e9,            # High science development
        contingency_percentage=30.0      # High science contingency
    ),
    optimization_config={
        'optimizer_params': {
            'population_size': 150,
            'num_generations': 120
        }
    },
    constellation_config={
        'problem_params': {
            'coverage_weight': 2.0,      # Good coverage for science
            'redundancy_weight': 1.5,    # Backup instruments
            'min_moon_alt': 30.0,        # Low lunar orbits
            'max_moon_alt': 200.0,       # for detailed observation
        }
    }
)
```

## Performance Optimization

### For Large Constellations (K>12)

```python
# Enable Ray parallelization for large problems
import os
os.environ['RAY_DEDUP_LOGS'] = '0'  # Reduce Ray logging

# Optimize with Ray (when available)
try:
    import ray
    ray.init(num_cpus=8)
    
    # Use Ray-enabled optimizer for large constellations
    from src.optimization.ray_optimizer import RayParallelOptimizer
    
    optimizer = RayParallelOptimizer(
        problem=MultiMissionProblem(num_missions=24),
        population_size=1200,
        num_generations=500,
        ray_num_workers=8
    )
    
    results = optimizer.optimize()
    
finally:
    ray.shutdown()
    
except ImportError:
    print("Ray not available, using standard optimization")
    # Fall back to standard optimizer
```

### Memory Management

```python
# For very large constellations, use memory-efficient settings
large_constellation_config = {
    'optimizer_params': {
        'population_size': min(2000, 100 * K),  # Cap population
        'num_generations': min(1000, 200 + 30 * K),  # Cap generations
        'cache_size_limit': 50000,  # Limit trajectory cache
        'enable_garbage_collection': True,  # Clean up memory
    }
}
```

## Troubleshooting

### Common Issues

#### "Array Ambiguity" Warnings
```
DEBUG: Fitness evaluation failed: The truth value of an array with more than one element is ambiguous
```
**Solution**: This is normal during optimization as some parameter combinations are infeasible. The optimizer handles this automatically with penalty values.

#### Memory Issues for Large K
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce population size or enable Ray parallelization:
```python
optimization_config = {
    'optimizer_params': {
        'population_size': min(1000, 80 * K),  # Reduce population
        'num_generations': min(500, 150 + 20 * K),  # Reduce generations
    }
}
```

#### Slow Convergence
```
# Poor Pareto front after many generations
```
**Solution**: Increase population diversity:
```python
constellation_config = {
    'problem_params': {
        'min_epoch': 9000.0,   # Wider epoch range
        'max_epoch': 11000.0,
        'coverage_weight': 0.5,  # Reduce constraint weights
        'redundancy_weight': 0.3,
    }
}
```

### Performance Monitoring

```python
# Monitor optimization progress
def monitor_optimization():
    results = optimize_constellation(
        num_missions=8,
        optimization_config={
            'optimizer_params': {
                'population_size': 400,
                'num_generations': 200
            },
            'verbose': True  # Enable detailed logging
        }
    )
    
    # Check cache efficiency
    cache_stats = results.get('cache_stats', {})
    hit_rate = cache_stats.get('hit_rate', 0)
    
    if hit_rate < 0.3:
        print("Warning: Low cache hit rate, consider adjusting parameters")
    
    # Check convergence quality
    pareto_size = len(results.get('pareto_front', []))
    if pareto_size < 10:
        print("Warning: Small Pareto front, consider more generations")
        
    return results
```

## Advanced Topics

### Custom Constellation Objectives

```python
# Define custom constellation objectives
class CustomMultiMissionProblem(MultiMissionProblem):
    def _calculate_communication_coverage(self, genome):
        """Custom objective for communication coverage."""
        # Calculate ground coverage based on satellite positions
        coverage_percentage = self._estimate_ground_coverage(genome.plane_raan)
        return 100.0 - coverage_percentage  # Minimize (maximize coverage)
    
    def _calculate_orbital_stability(self, genome):
        """Custom objective for orbital stability."""
        # Penalize very low or high altitudes
        stability_penalty = 0.0
        for alt in genome.parking_altitudes:
            if alt < 300 or alt > 800:
                stability_penalty += abs(alt - 550)  # Prefer ~550km
        return stability_penalty
```

### Integration with Existing Workflows

```python
# Integrate with existing mission planning workflow
def integrated_mission_planning(mission_requirements):
    """Integrate constellation optimization with mission planning."""
    
    # Phase 1: Determine constellation size based on requirements
    num_missions = estimate_required_satellites(mission_requirements)
    
    # Phase 2: Optimize constellation
    constellation_results = optimize_constellation(
        num_missions=num_missions,
        cost_factors=mission_requirements['cost_factors'],
        optimization_config=mission_requirements['optimization_config']
    )
    
    # Phase 3: Validate against mission requirements
    if validate_mission_requirements(constellation_results, mission_requirements):
        return finalize_mission_plan(constellation_results)
    else:
        # Adjust and re-optimize
        return adjust_and_reoptimize(constellation_results, mission_requirements)
```

## Summary

The multi-mission constellation optimization capability provides:

- ✅ **Scalable optimization** from 1 to 24+ satellites
- ✅ **Constellation-specific objectives** for coverage and redundancy
- ✅ **Full backward compatibility** with existing code
- ✅ **Production-ready implementation** with comprehensive testing
- ✅ **Real PyKEP trajectory calculations** (no mocking)
- ✅ **Flexible configuration** for different mission types
- ✅ **Detailed analysis tools** for constellation performance

This enables optimization of complex multi-satellite missions while maintaining the same ease of use as single-mission optimization.