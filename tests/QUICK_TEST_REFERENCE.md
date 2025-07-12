# Quick Test Reference Guide

## 🚀 Essential Test Files for Developers

### Must-Run Before Commits
```bash
make test              # 38 production tests (100% must pass)
make test-real-fast    # 6 real implementation demos (<5s)
```

### Core Test Files by Module

#### 🌍 **Environment Setup**
- `test_environment.py` - Validates all dependencies (PyKEP, PyGMO, JAX, etc.)
  - Tests: 8 | Pass Rate: 100% | Time: <2s
  - Real Functions: Library imports, basic operations

#### 💰 **Economics** 
- `test_economics_modules.py` - Financial calculations and ISRU analysis
  - Tests: 64 | Pass Rate: 100% | Time: ~4s
  - Real Functions: NPV, IRR, ROI, cost models, sensitivity analysis

#### 🚀 **Trajectory**
- `test_trajectory_modules.py` - Orbital mechanics and transfers
  - Tests: ~130 | Pass Rate: 77% | Time: ~20s
  - Real Functions: Lambert solver, N-body propagation, transfer windows
  - Subdirectory: `trajectory/` with 12 specialized test files

#### 🎯 **Optimization**
- `test_optimization_modules.py` - Multi-objective optimization
  - Tests: 30 | Pass Rate: 93% | Time: ~10s
  - Real Functions: PyGMO NSGA-II, Pareto analysis, cost integration

#### 🔧 **Configuration**
- `test_config_loader.py` - Configuration file handling
- `test_config_models.py` - Data model validation
- `test_config_manager.py` - Configuration management
  - Combined: ~20 tests | Pass Rate: 95% | Time: <3s

#### 🔗 **Integration**
- `test_final_functionality.py` - Core system validation (PRODUCTION GATE)
  - Tests: 15 | Pass Rate: 100% | Time: <3s
  - Real Functions: End-to-end workflows

### Real Implementation Test Files (NO MOCKING)
1. **test_real_working_demo.py** - Ultra-fast demo (6 tests, <5s)
2. **test_real_trajectory_fast.py** - Fast trajectory tests
3. **test_real_optimization_fast.py** - Fast optimization tests
4. **test_real_integration_fast.py** - Fast integration tests
5. **test_real_fast_comprehensive.py** - All modules combined

### Task-Specific Test Files
- `test_task_3_trajectory_generation.py` - Trajectory module completion
- `test_task_4_global_optimization.py` - PyGMO integration
- `test_task_5_economic_analysis.py` - Economic models
- `test_task_6_visualization.py` - Plotting and dashboards
- `test_task_7_integration.py` - MVP integration
- `test_task_8_differentiable_optimization.py` - JAX/Diffrax
- `test_task_9_enhanced_economics.py` - Advanced economics
- `test_task_10_extensibility.py` - Plugin system

## Key Real Functions Being Tested

### PyKEP Functions
- `pykep.lambert_problem()` - Two-body orbital transfers
- `pykep.planet.jpl_lp()` - Ephemeris data
- `pykep.epoch()` - Time conversions
- `pykep.ic2par()` - Orbital element conversions

### PyGMO Functions
- `pygmo.problem()` - Problem formulation
- `pygmo.algorithm(pygmo.nsga2())` - Multi-objective optimization
- `pygmo.population()` - Solution population management
- `pygmo.hypervolume()` - Solution quality metrics

### JAX Functions
- `jax.grad()` - Automatic differentiation
- `jax.jit()` - Just-in-time compilation
- `jax.vmap()` - Vectorization
- `diffrax.diffeqsolve()` - ODE integration

### Economic Functions
- `NPVAnalyzer.calculate_npv()` - Net present value
- `ROICalculator.calculate_simple_roi()` - Return on investment
- `ISRUBenefitAnalyzer.analyze_isru_economics()` - Resource utilization
- `EconomicSensitivityAnalyzer.monte_carlo_simulation()` - Risk analysis

### Trajectory Functions
- `generate_earth_moon_trajectory()` - Complete transfer calculation
- `LambertSolver.solve_lambert()` - Two-point boundary value problem
- `EarthMoonNBodyPropagator.propagate_spacecraft()` - Numerical integration
- `TrajectoryWindowAnalyzer.find_transfer_windows()` - Launch opportunities

## Test Execution Tips

1. **Quick Validation**: `make test-real-fast` (6 tests, <5s)
2. **Pre-Commit**: `make test` (38 tests, must pass 100%)
3. **Module Testing**: `make test-<module>` for specific areas
4. **Full Suite**: `make test-all` (415 tests, ~60s)
5. **With Coverage**: `make coverage` for detailed reports

## Common Test Patterns

### Real Implementation Pattern
```python
# NO MOCKING - Use real implementations
trajectory, delta_v = generate_earth_moon_trajectory(
    departure_epoch=10000.0,
    earth_orbit_alt=400.0,
    moon_orbit_alt=100.0,
    transfer_time=4.5,
    method="patched_conics"
)
assert 1000 < delta_v < 10000  # Realistic bounds
```

### Fast Execution Pattern
```python
# Use minimal parameters for speed
optimizer = GlobalOptimizer(
    population_size=8,   # Minimum for NSGA-II
    generations=2,       # Very few for speed
    seed=42
)
```

### Validation Pattern
```python
# Always validate realistic ranges
assert 0.0 < roi < 2.0  # Reasonable ROI
assert 3000 < delta_v < 4000  # LEO-Moon transfer
assert cost > 100e6  # Mission costs in millions
```