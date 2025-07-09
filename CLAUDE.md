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
- **Commit Policy**: After every fix or change, run tests and commit immediately

#### Available Make Commands
- **make test**: Run full test suite in py312 environment
- **make lint**: Run code linting and formatting checks
- **make deps**: Verify and install project dependencies
- **make clean**: Clean up temporary files and reset environment
- **make docs**: Generate and update project documentation
- **make verify**: Run comprehensive project verification checks

#### Command Usage Guidelines
- Always activate conda py312 environment before running make commands
- Use `make test` before committing any changes to ensure code quality
- Document any new make commands in the project documentation
- Ensure each command has a clear, single responsibility

(Rest of the document remains the same as in the original file)