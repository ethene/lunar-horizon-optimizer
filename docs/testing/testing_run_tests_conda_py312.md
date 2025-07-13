# Running Tests in Conda py312 Environment

## Overview

This guide provides instructions for running the comprehensive test suite in the proper conda py312 environment where PyKEP, PyGMO, and all required dependencies are installed.

## Environment Setup

### 1. Create and Activate Conda py312 Environment

```bash
# Create conda py312 environment
conda create -n py312 python=3.12 -y

# Activate environment
conda activate py312

# Install specialized dependencies from conda-forge
conda install -c conda-forge pykep pygmo astropy spiceypy -y

# Install Python packages
pip install numpy scipy matplotlib plotly poliastro pytest black flake8 mypy
```

### 2. Verify Environment Setup

```bash
# Change to project directory
cd "/Users/dmitrystakhin/Library/CloudStorage/Dropbox/work/Lunar Horizon Optimizer"

# Verify dependencies
python scripts/verify_dependencies.py

# Expected output should show:
# ✓ PyKEP 2.6 available
# ✓ PyGMO 2.19.6 available  
# ✓ All core dependencies available
```

## Running the Comprehensive Test Suite

### 1. Quick Test Execution

```bash
# Ensure you're in the project root and conda py312 is activated
conda activate py312
cd "/Users/dmitrystakhin/Library/CloudStorage/Dropbox/work/Lunar Horizon Optimizer"

# Run comprehensive test suite
python tests/run_comprehensive_tests.py
```

### 2. Individual Test Suite Execution

```bash
# Task 3: Trajectory Generation Tests
pytest tests/test_task_3_trajectory_generation.py -v

# Task 4: Global Optimization Tests  
pytest tests/test_task_4_global_optimization.py -v

# Task 5: Economic Analysis Tests
pytest tests/test_task_5_economic_analysis.py -v

# Integration Tests
pytest tests/test_integration_tasks_3_4_5.py -v
```

### 3. Specific Test Categories

```bash
# Run only tests that require PyKEP
pytest tests/test_task_3_trajectory_generation.py::TestLambertSolver -v

# Run only tests that require PyGMO
pytest tests/test_task_4_global_optimization.py::TestGlobalOptimizer -v

# Run only basic tests (no external dependencies)
pytest tests/test_task_5_economic_analysis.py::TestFinancialModels -v
```

## Expected Test Results in Conda py312 Environment

### Projected Success Rates

Based on the test framework design:

#### Task 3: Enhanced Trajectory Generation
- **Expected**: 20-23 tests passing out of 25 total (80-92% success rate)
- **Key Features**: Lambert solvers, N-body dynamics, trajectory I/O
- **Dependencies**: PyKEP, SciPy, NumPy

#### Task 4: Global Optimization Module
- **Expected**: 25-28 tests passing out of 30 total (83-93% success rate)  
- **Key Features**: PyGMO NSGA-II, Pareto analysis, cost integration
- **Dependencies**: PyGMO, PyKEP (via integration), NumPy

#### Task 5: Basic Economic Analysis Module
- **Expected**: 35-37 tests passing out of 38 total (92-97% success rate)
- **Key Features**: Financial modeling, cost estimation, ISRU analysis
- **Dependencies**: SciPy, NumPy (minimal external dependencies)

#### Integration Tests
- **Expected**: 12-14 tests passing out of 15 total (80-93% success rate)
- **Key Features**: End-to-end workflows, cross-module integration
- **Dependencies**: All above dependencies

### Overall Expected Results
- **Total Tests**: 103 tests
- **Expected Passing**: 92-102 tests (89-99% success rate)
- **Expected Skipped**: 0-5 tests (environment-specific skips)
- **Expected Failed**: 1-6 tests (edge cases, environmental issues)

## Test Execution Commands

### Standard Test Run

```bash
# Activate environment and run all tests
conda activate py312
cd "/Users/dmitrystakhin/Library/CloudStorage/Dropbox/work/Lunar Horizon Optimizer"

# Run comprehensive test suite with detailed reporting
python tests/run_comprehensive_tests.py

# Alternative: Direct pytest execution
pytest tests/ -v --tb=short --disable-warnings
```

### Performance Testing

```bash
# Run tests with timing information
pytest tests/ -v --durations=10

# Run tests with coverage analysis (if coverage installed)
pytest tests/ --cov=src --cov-report=html
```

### Debug Mode Testing

```bash
# Run tests with maximum verbosity and debugging
pytest tests/ -vvv --tb=long --capture=no

# Run specific failing test with debugging
pytest tests/test_task_3_trajectory_generation.py::TestLambertSolver::test_lambert_solution_earth_orbit -vvv --tb=long
```

## Troubleshooting

### Common Issues and Solutions

#### 1. PyKEP Import Errors
```bash
# Check PyKEP installation
conda list pykep

# Reinstall if needed
conda install -c conda-forge pykep=2.6 --force-reinstall
```

#### 2. PyGMO Import Errors
```bash
# Check PyGMO installation
conda list pygmo

# Reinstall if needed
conda install -c conda-forge pygmo=2.19.6 --force-reinstall
```

#### 3. Module Import Errors
```bash
# Ensure project structure is correct
python -c "import sys; sys.path.insert(0, 'src'); import trajectory; print('✓ Trajectory module imports')"
python -c "import sys; sys.path.insert(0, 'src'); import optimization; print('✓ Optimization module imports')"
python -c "import sys; sys.path.insert(0, 'src'); import economics; print('✓ Economics module imports')"
```

#### 4. Test Data Issues
```bash
# Verify test data directory exists
ls -la tests/
ls -la data/  # If test data files needed
```

## Test Validation Checklist

### Pre-Test Checklist
- [ ] Conda py312 environment activated
- [ ] All dependencies installed and verified
- [ ] Project directory set correctly
- [ ] No import errors when loading modules

### Post-Test Validation
- [ ] Overall pass rate > 85%
- [ ] Task 3 tests demonstrate PyKEP integration
- [ ] Task 4 tests demonstrate PyGMO optimization
- [ ] Task 5 tests demonstrate economic analysis
- [ ] Integration tests show cross-module functionality
- [ ] Performance metrics within acceptable ranges

### Success Criteria
- **Minimum Acceptable**: 85% overall pass rate (87+ tests passing)
- **Target Success**: 90% overall pass rate (92+ tests passing)  
- **Excellent Performance**: 95% overall pass rate (97+ tests passing)

## Automated Test Execution Script

You can also create a shell script for automated testing:

```bash
#!/bin/bash
# File: run_tests_automated.sh

echo "🚀 Starting Lunar Horizon Optimizer Test Suite"
echo "Environment: conda py312"

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py312

# Change to project directory
cd "/Users/dmitrystakhin/Library/CloudStorage/Dropbox/work/Lunar Horizon Optimizer"

# Verify environment
echo "📋 Verifying environment..."
python scripts/verify_dependencies.py

# Run comprehensive tests
echo "🧪 Running comprehensive test suite..."
python tests/run_comprehensive_tests.py

# Check results
if [ $? -eq 0 ]; then
    echo "✅ All tests completed successfully!"
else
    echo "❌ Some tests failed. Check output above."
fi
```

## Next Steps

After running tests in conda py312 environment:

1. **Analyze Results**: Review test report for any failures or issues
2. **Fix Issues**: Address any failing tests or dependency problems  
3. **Validate Coverage**: Ensure all critical functionality is tested
4. **Document Results**: Update test validation documentation
5. **Proceed to Task 6**: Begin visualization module development

---

**Important**: Always run tests in the conda py312 environment to get accurate results, as the test framework is specifically designed for the PyKEP and PyGMO dependencies available in that environment.