# Multi-Mission Constellation Optimization Architecture

## Overview

This document describes the architecture for evolving the Lunar Horizon Optimizer to handle K simultaneous lunar transfers in a single chromosome, enabling constellation deployment optimization for scenarios like 24 lunar communication satellites.

## Current vs. Multi-Mission Architecture

### Current Single-Mission Structure
```
Decision Vector (3 variables):
[earth_alt, moon_alt, transfer_time] 

Objectives (3):
[delta_v, time, cost]

Problem Size: O(3) variables, O(1) trajectory evaluation
```

### New Multi-Mission Structure  
```
Decision Vector (4K + 2 variables):
[epochs[0..K-1], parking_alts[0..K-1], plane_raan[0..K-1], payload_masses[0..K-1], lunar_alt, transfer_time]

Objectives (5 in constellation mode):
[total_delta_v, total_time, total_cost, coverage_metric, redundancy_metric]

Problem Size: O(4K+2) variables, O(K) trajectory evaluations
```

## Multi-Mission Genome Design

### Data Structure
```python
@dataclass
class MultiMissionGenome:
    num_missions: int                      # K
    epochs: List[float]                    # K launch epochs [days since J2000]
    parking_altitudes: List[float]         # K Earth orbit altitudes [km]  
    plane_raan: List[float]               # K orbital plane orientations [deg]
    payload_masses: List[float]           # K mission-specific masses [kg]
    lunar_altitude: float                 # Shared lunar orbit altitude [km]
    transfer_time: float                  # Shared transfer duration [days]
```

### Decision Vector Encoding
```
Index Range        | Parameter           | Bounds
[0, K)            | epochs              | [9000, 11000] days since J2000
[K, 2K)           | parking_altitudes   | [200, 1000] km
[2K, 3K)          | plane_raan         | [0, 360] degrees  
[3K, 4K)          | payload_masses     | [500, 2000] kg
[4K]              | lunar_altitude     | [50, 500] km
[4K+1]            | transfer_time      | [3, 10] days

Total Length: 4K + 2
```

## PyGMO Dimensionality Impact Analysis

### Scaling Characteristics

| K (Missions) | Variables | Population Size | Generations | Evaluation Cost | Memory Usage |
|--------------|-----------|-----------------|-------------|-----------------|--------------|
| 1 (Current)  | 3         | 100            | 100         | O(1)           | ~50MB        |
| 3            | 14        | 150            | 120         | O(3)           | ~150MB       |
| 8            | 34        | 300            | 200         | O(8)           | ~400MB       |
| 24           | 98        | 600            | 400         | O(24)          | ~1.2GB       |

### Recommended PyGMO Configuration

```python
# Constellation-aware scaling rules
population_size = max(100, 50 * K)           # Minimum population for convergence
num_generations = max(100, 100 + 20 * K)     # Additional generations for complexity
algorithm = pg.nsga2(gen=1, seed=42)         # NSGA-II handles multi-objective well

# Memory management for large constellations
if K > 12:
    enable_caching = True
    cache_size_limit = 10000
    use_parallel_evaluation = True  # Ray parallelization
```

### Integer Variable Flags

```python
# PyGMO integer variable specification
def get_nix(self) -> int:
    """Number of integer variables."""
    return 0  # All variables are continuous for gradient-based refinement

# Alternative: Discretize RAAN for orbital mechanics
# def get_integer_part(self) -> List[int]:
#     integer_indices = list(range(2*K, 3*K))  # RAAN variables as integers
#     return integer_indices
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     MULTI-MISSION OPTIMIZER                    │
├─────────────────────────────────────────────────────────────────┤
│  Input: --multi K                                              │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ Single Mission  │ OR │ Multi-Mission   │                   │
│  │ (K=1, Original) │    │ (K>1, Enhanced) │                   │
│  └─────────────────┘    └─────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROBLEM SELECTION                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            LunarMissionProblem (Original)                  │ │
│  │  • Decision Vector: [earth_alt, moon_alt, transfer_time]   │ │
│  │  • Objectives: [delta_v, time, cost]                      │ │
│  │  • Variables: 3                                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              OR                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            MultiMissionProblem (Enhanced)                  │ │
│  │  • Decision Vector: [epochs×K, alts×K, raan×K,            │ │
│  │                      masses×K, lunar_alt, transfer_time]   │ │
│  │  • Objectives: [Σdelta_v, Σtime, Σcost,                  │ │
│  │                 coverage, redundancy]                      │ │
│  │  • Variables: 4K + 2                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GENOME DECODING                          │
├─────────────────────────────────────────────────────────────────┤
│  MultiMissionGenome.from_decision_vector(x, K)                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Mission 0: epoch[0], earth_alt[0], raan[0], mass[0]      │ │
│  │  Mission 1: epoch[1], earth_alt[1], raan[1], mass[1]      │ │
│  │  ...                                                       │ │
│  │  Mission K-1: epoch[K-1], earth_alt[K-1], raan[K-1], ... │ │
│  │  Shared: lunar_altitude, transfer_time                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY EVALUATION                       │
├─────────────────────────────────────────────────────────────────┤
│  FOR i in range(K):                                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  mission_params = genome.get_mission_parameters(i)        │ │
│  │  trajectory[i], dv[i] = lunar_transfer.generate_transfer( │ │
│  │      epoch=mission_params['epoch'],                       │ │
│  │      earth_orbit_alt=mission_params['earth_orbit_alt'],   │ │
│  │      moon_orbit_alt=mission_params['moon_orbit_alt'],     │ │
│  │      transfer_time=mission_params['transfer_time']        │ │
│  │  )                                                        │ │
│  │  cost[i] = cost_calculator.calculate_mission_cost(...)    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONSTELLATION OBJECTIVES                     │
├─────────────────────────────────────────────────────────────────┤
│  Base Objectives:                                             │
│  • total_delta_v = Σ dv[i]                                   │
│  • total_time = Σ time[i]                                    │  
│  • total_cost = Σ cost[i]                                    │
│                                                               │
│  Constellation Objectives:                                    │
│  • coverage_metric = f(raan_distribution, K)                 │
│  • redundancy_metric = f(performance_similarity, timing)     │
└─────────────────────────────────────────────────────────────────┘
```

## Migration Strategy

### Phase 1: Implementation (Current)
- [x] Create `MultiMissionGenome` dataclass
- [x] Implement `MultiMissionProblem` PyGMO interface
- [x] Create `MultiMissionOptimizer` with backward compatibility
- [x] Add constellation-specific objectives (coverage, redundancy)

### Phase 2: Integration
- [ ] Add `--multi K` command-line flag
- [ ] Create migration utilities for existing configurations
- [ ] Add comprehensive testing with K=1,3,8,24
- [ ] Performance optimization for large constellations

### Phase 3: Validation  
- [ ] Validate constellation objectives with domain experts
- [ ] Benchmark against single-mission performance
- [ ] Documentation and examples
- [ ] Ray parallelization for K>8

## Command-Line Interface

### Backward Compatible (Current)
```bash
# Single mission (unchanged)
python optimize.py --config lunar_mission.yaml
```

### Multi-Mission (New)
```bash
# 3-satellite constellation
python optimize.py --config lunar_mission.yaml --multi 3

# 24-satellite constellation with custom settings
python optimize.py --config constellation.yaml --multi 24 \
    --population 600 --generations 400 --constellation-weights coverage=2.0,redundancy=1.0
```

### Configuration File Migration
```yaml
# Original single-mission config
optimization:
  population_size: 100
  num_generations: 100

# Auto-migrated multi-mission config  
optimization:
  population_size: 150  # Scaled for K=3
  num_generations: 120  # Scaled for K=3
  
constellation:
  num_missions: 3
  coverage_weight: 1.0
  redundancy_weight: 0.5
  constellation_mode: true
```

## Performance Considerations

### Computational Complexity
- **Single Mission**: O(1) trajectory evaluation per fitness call
- **K-Mission**: O(K) trajectory evaluations per fitness call  
- **Population**: O(P × K) evaluations per generation
- **Total**: O(G × P × K) for complete optimization

### Memory Scaling
- **Trajectory Cache**: Scales as O(K × cache_size)
- **Population**: Scales as O(P × (4K+2))
- **Results Storage**: Scales as O(P × 5) objectives

### Optimization Recommendations
- Use Ray parallelization for K > 8
- Enable trajectory caching for repeated evaluations
- Consider hierarchical optimization for K > 20
- Use restart strategies for large search spaces

## Validation Strategy

### Test Cases
1. **K=1**: Verify identical results to original optimizer
2. **K=3**: Small constellation validation
3. **K=8**: Medium constellation performance
4. **K=24**: Large constellation scalability

### Performance Benchmarks
```python
# Single mission baseline
K=1: ~5s/generation, 100 generations = ~8 minutes

# Expected multi-mission scaling
K=3: ~15s/generation, 120 generations = ~30 minutes  
K=8: ~40s/generation, 200 generations = ~2 hours
K=24: ~120s/generation, 400 generations = ~13 hours
```

### Validation Metrics
- Convergence quality vs. single-mission Pareto fronts
- Constellation coverage uniformity
- Cost efficiency compared to independent optimization
- Solution feasibility and robustness

## Implementation Status

### Completed ✅
- [x] Multi-mission genome dataclass design
- [x] PyGMO problem interface for K missions
- [x] Constellation-specific objectives (coverage, redundancy)
- [x] Backward compatibility utilities
- [x] Enhanced optimizer with constellation analysis

### In Progress 🔄
- [ ] Command-line interface integration
- [ ] Comprehensive test suite
- [ ] Performance optimization
- [ ] Documentation and examples

### Planned 📋
- [ ] Ray parallelization integration
- [ ] Hierarchical optimization for large K
- [ ] Domain expert validation
- [ ] Production deployment