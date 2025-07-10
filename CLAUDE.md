# CLAUDE.md - Project Working Rules

## Project Overview

The **Lunar Horizon Optimizer** is an integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions. This document establishes the working rules, coding standards, and development practices for this project when working with Claude Code.

## Project Structure & Architecture

### Core Components
```
src/
├── config/          # Mission configuration and parameter management
├── trajectory/      # Orbital mechanics and trajectory calculations  
├── optimization/    # Global optimization with PyGMO (Task 4) ✅
├── economics/       # Economic analysis and financial modeling (Task 5) ✅
├── utils/           # Utility functions and unit conversions
└── constants/       # Physical constants and parameters

tests/               # Comprehensive test suite (83% success rate)
docs/                # Project documentation (comprehensive)
tasks/               # Development task management
scripts/             # Utility scripts and PRD
```

### Key Technologies
- **PyKEP 2.6** - High-fidelity orbital mechanics
- **PyGMO 2.19.6** - Global optimization algorithms (NSGA-II)
- **JAX 0.5.3** + **Diffrax 0.7.0** - Differentiable programming
- **SciPy 1.13.1** - Scientific computing foundation
- **Plotly 5.24.1** - Interactive visualization
- **Pydantic** - Configuration validation
- **SpiceyPy** - NASA SPICE toolkit for ephemeris calculations

### Development Environment Requirements
- **Python Environment**: conda py312 environment (required - ALREADY CONFIGURED)
- **Local Environment**: The py312 environment is pre-configured with PyKEP, PyGMO, and all dependencies
- **Testing Environment**: ALWAYS use `conda activate py312` for all testing operations
- **Environment Setup** (if needed on new systems): 
  ```bash
  conda create -n py312 python=3.12 -y
  conda activate py312
  conda install -c conda-forge pykep pygmo astropy spiceypy -y
  pip install -r requirements.txt
  ```
- **CRITICAL**: All testing must be done in the py312 environment to ensure PyKEP/PyGMO compatibility

## Development Standards

### Code Quality Requirements

#### 1. **Testing is Mandatory**
- **Coverage**: Maintain >90% test coverage for all new code
- **Test Types**: Unit tests, integration tests, and validation tests
- **Test Location**: All tests in `tests/` directory with clear naming
- **Before Committing**: Always run `pytest tests/` in conda py312 environment and ensure all tests pass
- **NO MOCKING RULE**: NEVER use mocks when real functionality exists - always examine existing code first
- **Real Implementation First**: Use actual classes and functions instead of mocks for all testing

#### 2. **Code Style & Formatting** 
- **Linting**: Run `flake8` for code quality checks
- **Formatting**: Use `black` for consistent code formatting  
- **Type Hints**: All functions must include type hints
- **Docstrings**: All public functions/classes require comprehensive docstrings

#### 3. **Architecture Principles**
- **Single Responsibility**: Each module has one clear purpose
- **Unit Consistency**: Always use PyKEP native units (meters, m/s, radians)
- **Error Handling**: Comprehensive validation and meaningful error messages
- **Logging**: Use structured logging for debugging and monitoring

### Makefile Commands Management

#### Makefile Command Policy
- **Comprehensive Documentation**: Always document all available make commands
- **Consistency**: Maintain a standard format for command naming and usage
- **Testing Requirement**: Every command should have an associated testing mechanism
- **Commit Policy**: After every fix or change, run tests and commit immediately to maintain rollback capability

#### Available Make Commands

##### **Main Development Commands**
- **`make help`**: Display comprehensive help with all available commands
- **`make pipeline`**: Run complete development pipeline (format, lint, type-check, security, etc.)
- **`make test`**: Run production test suite (38 curated tests) in conda py312 environment
- **`make coverage`**: Run tests with comprehensive coverage reporting and HTML output

##### **Code Quality Commands**
- **`make format`**: Format code with Black (line length 88, Python 3.12 target)
- **`make lint`**: Comprehensive linting with Ruff (flake8 + pylint rules)
- **`make complexity`**: Analyze code complexity with Radon and Xenon
- **`make type-check`**: Static type analysis with MyPy in strict mode
- **`make refactor`**: AI-based refactor suggestions with Sourcery
- **`make security`**: Security vulnerability scan with Bandit

##### **Utility Commands**
- **`make install-dev`**: Install all development dependencies
- **`make clean`**: Remove temporary files (__pycache__, .pytest_cache, etc.)

#### Command Usage Guidelines

##### **Pre-Commit Workflow (MANDATORY)**
```bash
# Always run before committing changes:
conda activate py312       # Activate required environment
make test                 # Ensure all tests pass
make pipeline            # Run complete quality checks
git add .               # Stage changes
git commit -m "descriptive message"  # Commit with clear message
```

##### **Environment Requirements**
- **CRITICAL**: Always activate `conda py312` environment before running ANY make commands
- All make commands automatically use `conda run -n py312` to ensure environment consistency
- Never run make commands without proper conda environment activation

##### **Command Execution Order**
1. **Development**: `make format` → `make lint` → `make test`
2. **Pre-Commit**: `make pipeline` (runs all quality checks)
3. **Coverage Analysis**: `make coverage` (generates HTML reports)
4. **Cleanup**: `make clean` (when needed)

#### Git Commit Policy and Rollback Strategy

##### **Mandatory Commit Policy**
- **After Every Major Change**: ALWAYS commit immediately after completing any significant modification
- **Test Before Commit**: Run `make test` and ensure 100% pass rate before committing
- **Quality Checks**: Run `make pipeline` to verify code quality standards
- **Descriptive Messages**: Use clear, descriptive commit messages following conventional commit format

##### **Rollback and State Management**
- **Frequent Commits**: Enable easy rollback to previous working states
- **State Comparison**: Use `git diff` and `git log` to compare with previous states
- **Rollback Commands**:
  ```bash
  git log --oneline -10        # View recent commits
  git diff HEAD~1              # Compare with previous commit
  git reset --hard HEAD~1      # Rollback to previous commit (DESTRUCTIVE)
  git revert <commit-hash>     # Safe rollback creating new commit
  git stash                    # Temporarily save uncommitted changes
  ```

##### **Branch Protection Strategy**
- **Main Branch**: Always keep main branch in working state
- **Feature Branches**: Use for experimental changes
- **Testing**: Ensure all tests pass before merging to main
- **Backup**: Regular commits provide restore points

#### Error Handling and Recovery

##### **Test Failures**
```bash
# If make test fails:
conda activate py312                    # Ensure correct environment
python -m pytest tests/ -v --tb=short  # Run with detailed output
# Fix issues, then re-run make test
```

##### **Pipeline Failures**
```bash
# If make pipeline fails:
make format     # Fix formatting issues
make lint       # Address linting errors
make type-check # Resolve type issues
make security   # Fix security vulnerabilities
```

##### **Environment Issues**
```bash
# If conda environment issues:
conda activate py312
conda install -c conda-forge pykep pygmo astropy spiceypy -y
pip install -r requirements.txt
```

#### Testing Strategy

##### **Production Test Suite (38 tests)**
- **Command**: `make test`
- **Coverage**: Core functionality, physics validation, economics analysis
- **Requirement**: 100% pass rate for commits
- **Time**: ~30 seconds execution

##### **Comprehensive Test Suite (445 tests)**
```bash
# Full test suite (development only):
conda activate py312
python -m pytest tests/ --tb=short
```

##### **Coverage Analysis**
- **Command**: `make coverage`
- **Output**: Terminal report + HTML in `htmlcov/`
- **Threshold**: 80% minimum coverage required
- **Review**: Check `htmlcov/index.html` for detailed analysis

#### Performance and Monitoring

##### **Execution Times**
- `make test`: ~30 seconds
- `make pipeline`: ~2-3 minutes
- `make coverage`: ~45 seconds
- `make lint`: ~30 seconds

##### **Resource Usage**
- Memory: ~500MB peak during testing
- Disk: ~100MB for coverage reports
- CPU: Multi-core utilization during parallel tests

#### Troubleshooting Guide

##### **Common Issues**
1. **Environment not activated**: Always run `conda activate py312`
2. **Missing dependencies**: Run `make install-dev`
3. **Cache issues**: Run `make clean` to clear temporary files
4. **Test failures**: Check specific test output and fix underlying issues
5. **Type errors**: Review MyPy output and add proper type annotations

##### **Debug Commands**
```bash
# Verbose test output:
conda activate py312
python -m pytest tests/test_final_functionality.py -v -s

# Check environment:
conda info --envs
conda list -n py312

# Verify tools:
conda run -n py312 python --version
conda run -n py312 pytest --version
```