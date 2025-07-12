# Lunar Horizon Optimizer Test Suite

## Overview

This directory contains the comprehensive test suite for the Lunar Horizon Optimizer project. The test suite consists of **415 tests** across **46 test files**, providing extensive coverage of all system components with a strict **"NO MOCKING RULE"** policy.

## 🚀 Quick Start

```bash
# Run production tests (required before commits)
make test              # 38 core tests, 100% must pass

# Run ultra-fast real implementation tests
make test-real-fast    # 6 tests, <5 seconds

# Run module-specific tests
make test-economics    # Economic analysis tests
make test-trajectory   # Orbital mechanics tests
make test-optimization # Multi-objective optimization tests
```

## Test Environment Requirements

**CRITICAL**: Use ONLY the pre-configured conda py312 environment
```bash
conda activate py312
# Pre-installed: PyKEP 2.6, PyGMO 2.19.6, Python 3.12+, all dependencies
# NEVER run tests outside this environment - PyKEP/PyGMO require specific setup
```

## Testing Philosophy

### **NO MOCKING POLICY**
- **NEVER** use mocks when real functionality exists in the codebase
- **ALWAYS** examine existing code first before writing tests
- **USE** real classes: `LunarTransfer`, `SimpleOptimizationProblem`, `OrbitState`, etc.
- **ONLY** mock external I/O, APIs, or genuinely unavailable resources
- **PREFER** integration tests over mocked unit tests

## 📁 Directory Structure

```
tests/
├── README.md                          # This file
├── TEST_SUITE_DOCUMENTATION.md        # Comprehensive test documentation
├── QUICK_TEST_REFERENCE.md           # Quick reference guide
├── conftest.py                       # Pytest configuration
├── test_helpers.py                   # Shared test utilities
│
├── test_environment.py               # Environment validation (8 tests)
├── test_final_functionality.py       # Core production tests (15 tests)
│
├── test_economics_modules.py         # Economics tests (64 tests)
├── test_trajectory_modules.py        # Trajectory tests (~130 tests)
├── test_optimization_modules.py      # Optimization tests (30 tests)
├── test_config_*.py                  # Configuration tests (~20 tests)
│
├── test_real_*.py                    # Real implementation tests (NO MOCKING)
├── test_task_*.py                    # Task-specific validation tests
│
└── trajectory/                       # Trajectory module test subdirectory
    ├── test_lambert_solver.py        # Lambert problem physics
    ├── test_lunar_transfer.py        # Earth-Moon transfers
    └── ... (10 more specialized tests)
```

## Test Files Structure

### ✅ **PRODUCTION TEST FILES** (Must Pass 100%)

#### **Core Functionality Tests**
- **`test_final_functionality.py`** - **PRIMARY TEST SUITE**
  - **Tests**: 15 | **Pass Rate**: 100% ✅
  - **Coverage**: Complete real functionality testing without mocking
  - **Purpose**: Validates PyKEP, PyGMO, and economic analysis integration
  - **Runtime**: ~1.0 seconds
  - **Command**: `pytest tests/test_final_functionality.py -v`

### 📊 **MODULE TEST FILES**

#### **Economics Module** (100% Pass Rate)
- **`test_economics_modules.py`** - Comprehensive economic analysis
  - **Tests**: 64 | **Pass Rate**: 100% ✅ | **Time**: ~4s
  - **Coverage**: NPV, IRR, ROI, ISRU benefits, sensitivity analysis
  - **Real Functions**: Financial calculations, cost models

#### **Trajectory Module** (77% Pass Rate)
- **`test_trajectory_modules.py`** - Orbital mechanics and transfers
  - **Tests**: ~130 | **Pass Rate**: 77% ⚠️ | **Time**: ~20s
  - **Coverage**: Lambert solver, N-body propagation, transfer windows
  - **Known Issues**: Some edge cases produce NaN velocities

#### **Optimization Module** (93% Pass Rate)
- **`test_optimization_modules.py`** - Multi-objective optimization
  - **Tests**: 30 | **Pass Rate**: 93% ✅ | **Time**: ~10s
  - **Coverage**: PyGMO NSGA-II, Pareto analysis, cost integration
  - **Real Functions**: Problem formulation, optimization algorithms

#### **Configuration Module** (95% Pass Rate)
- **`test_config_loader.py`** - Configuration file handling
- **`test_config_models.py`** - Data model validation
- **`test_config_manager.py`** - Configuration management
  - **Combined**: ~20 tests | **Pass Rate**: 95% ✅ | **Time**: <3s

### 🚀 **REAL IMPLEMENTATION TEST FILES** (NO MOCKING)

- **`test_real_working_demo.py`** - Ultra-fast demonstration
  - **Tests**: 6 | **Time**: <5s | **Purpose**: Showcase real implementations
  
- **`test_real_trajectory_fast.py`** - Fast trajectory tests
  - **Time**: <3s | **Purpose**: Real PyKEP trajectory calculations
  
- **`test_real_optimization_fast.py`** - Fast optimization tests
  - **Time**: <5s | **Purpose**: Real PyGMO optimization runs
  
- **`test_real_integration_fast.py`** - Fast integration tests
  - **Time**: <5s | **Purpose**: End-to-end real workflows
  
- **`test_real_fast_comprehensive.py`** - All modules combined
  - **Time**: <15s | **Purpose**: Comprehensive real testing

### 📋 **TASK-SPECIFIC TEST FILES**

- **`test_task_3_trajectory_generation.py`** - Trajectory module completion
- **`test_task_4_global_optimization.py`** - PyGMO integration
- **`test_task_5_economic_analysis.py`** - Economic models
- **`test_task_6_visualization.py`** - Plotting and dashboards
- **`test_task_7_integration.py`** - MVP integration
- **`test_task_8_differentiable_optimization.py`** - JAX/Diffrax
- **`test_task_9_enhanced_economics.py`** - Advanced economics
- **`test_task_10_extensibility.py`** - Plugin system

### 📂 **TRAJECTORY SUBDIRECTORY** (12 specialized tests)

```
trajectory/
├── test_lambert_solver.py      # Lambert problem physics validation
├── test_hohmann_transfer.py    # Classical two-impulse transfers
├── test_lunar_transfer.py      # Earth-Moon specific trajectories
├── test_orbit_state.py         # Orbital state representations
├── test_propagator.py          # Trajectory propagation methods
├── test_elements.py            # Keplerian element conversions
├── test_celestial_bodies.py    # Planet/moon properties
├── test_unit_conversions.py    # Unit system handling
├── test_input_validation.py    # Parameter validation
├── test_epoch_conversions.py   # Time system conversions
├── test_trajectory_models.py   # Data structure validation
└── test_validator.py           # Physics constraint checking
```

### 📚 **DOCUMENTATION FILES**

- **`TEST_SUITE_DOCUMENTATION.md`** - Comprehensive test documentation
- **`QUICK_TEST_REFERENCE.md`** - Quick reference for developers
- **`README.md`** - This overview file
- **`run_tests_conda_py312.md`** - Environment setup guide
- **`test_validation_summary.md`** - Validation results

## 📊 Test Statistics

| Category | Tests | Pass Rate | Execution Time |
|----------|-------|-----------|----------------|
| Production Core | 38 | 100% ✅ | ~5s |
| Environment | 9 | 100% ✅ | ~2s |
| Economics | 64 | 100% ✅ | ~4s |
| Configuration | ~20 | 95% ✅ | ~3s |
| Optimization | ~49 | 93% ✅ | ~10s |
| Trajectory | ~130 | 77% ⚠️ | ~20s |
| Visualization | ~37 | 62% ⚠️ | ~15s |
| **Total** | **415** | **~85%** | **~60s** |

## 🔧 Real Functions Tested

### Core Libraries
- **PyKEP**: Lambert solver, ephemeris, orbital elements, epoch handling
- **PyGMO**: NSGA-II optimization, population management, hypervolume
- **JAX**: Automatic differentiation, JIT compilation, vectorization
- **Diffrax**: ODE solving for trajectory propagation

### System Modules
- **Economics**: NPV, IRR, ROI, ISRU benefits, Monte Carlo simulation
- **Trajectory**: Earth-Moon transfers, Lambert problems, N-body propagation
- **Optimization**: Multi-objective optimization, Pareto analysis, convergence
- **Configuration**: Data validation, file I/O, model management

## How to Run Tests

### **Primary Test Execution**

```bash
# CRITICAL: Always activate py312 environment first
conda activate py312

# RECOMMENDED: Use Makefile commands
make test              # Production tests (38 tests, must pass 100%)
make test-real-fast    # Ultra-fast real tests (6 tests, <5s)
make test-economics    # Economics module tests
make test-trajectory   # Trajectory module tests
make test-all          # Complete test suite (415 tests)

# ALTERNATIVE: Direct pytest execution
export PYTHONPATH=src
pytest tests/test_final_functionality.py -v
pytest tests/test_economics_modules.py -v
```

### **Module-Specific Execution**

```bash
# Test specific modules
make test-economics      # 64 tests for economic analysis
make test-trajectory     # ~130 tests for orbital mechanics  
make test-optimization   # ~49 tests for multi-objective optimization
make test-config         # ~20 tests for configuration management

# Test real implementations (NO MOCKING)
make test-real           # Fast real implementation tests
make test-real-comprehensive  # Complete real implementation suite
```

### **Coverage Analysis**

```bash
# Generate coverage report
make coverage            # Runs tests with coverage analysis

# Coverage report will be in htmlcov/index.html
```

## 🎯 Key Features

### NO MOCKING RULE
All tests use real implementations of PyKEP, PyGMO, JAX, and other libraries. This ensures tests validate actual functionality rather than mocked interfaces.

### Fast Execution
Production tests complete in under 5 seconds, enabling rapid development iteration.

### Comprehensive Coverage
Every major module has dedicated test files with detailed validation of core functionality.

### Production Gate
38 core tests must pass 100% before any commit is allowed.

## 🛠️ Development Workflow

1. **Before starting work**: `make test-real-fast` (quick validation)
2. **During development**: Run module-specific tests
3. **Before committing**: `make test` (must pass 100%)
4. **For detailed analysis**: `make coverage`

## ⚠️ Known Issues

1. **Trajectory Tests**: Some failures due to edge cases in Lambert solver (NaN velocities)
2. **Visualization Tests**: Dashboard tests may fail due to Plotly version compatibility
3. **Integration Tests**: Pass rate varies based on component availability

## 🚀 Best Practices

1. **Use Real Implementations**: Never mock when real functionality exists
2. **Keep Tests Fast**: Use minimal parameters for quick execution
3. **Validate Physics**: Check conservation laws and realistic bounds
4. **Document Purpose**: Clearly state what each test validates
5. **Maintain Coverage**: Add tests for all new features

## 📚 Documentation

### Main Documentation Files
- **[TEST_SUITE_DOCUMENTATION.md](TEST_SUITE_DOCUMENTATION.md)** - Comprehensive documentation of all test files
- **[QUICK_TEST_REFERENCE.md](QUICK_TEST_REFERENCE.md)** - Quick reference for common testing tasks

### Additional Resources
- Individual test file docstrings for specific details
- CLAUDE.md in project root for development standards
- run_tests_conda_py312.md for environment setup

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH=src` is set
2. **Missing Dependencies**: Activate conda py312 with PyKEP/PyGMO
3. **Test Failures**: Check environment setup with `test_environment.py`

### Environment Verification

```bash
python -c "
import pykep as pk
import pygmo as pg
import jax
import diffrax
print(f'✅ PyKEP: {pk.__version__}')
print(f'✅ PyGMO: {pg.__version__}')
print(f'✅ JAX: {jax.__version__}')
print(f'✅ Diffrax: {diffrax.__version__}')
"
```

## 📦 Summary

The Lunar Horizon Optimizer test suite provides comprehensive validation of all system components with a focus on real implementations. With 415 tests across 46 files, the suite ensures robust functionality while maintaining fast execution times and clear documentation.