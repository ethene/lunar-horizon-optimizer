# 🌙 Lunar Horizon Optimizer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-243%2F243%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-27%25-yellow.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pipeline](https://img.shields.io/badge/pipeline-clean-brightgreen.svg)](Makefile)

<!-- Core Dependencies -->
[![PyKEP](https://img.shields.io/badge/PyKEP-2.6-orange.svg)](https://esa.github.io/pykep/)
[![PyGMO](https://img.shields.io/badge/PyGMO-2.19.6-orange.svg)](https://esa.github.io/pygmo2/)
[![JAX](https://img.shields.io/badge/JAX-0.6.0-red.svg)](https://jax.readthedocs.io/)
[![Diffrax](https://img.shields.io/badge/Diffrax-0.7.0-red.svg)](https://docs.kidger.site/diffrax/)
[![SciPy](https://img.shields.io/badge/SciPy-1.13.1-blue.svg)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24.1-3F4F75.svg)](https://plotly.com/)

<!-- Analysis & ML -->
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.2-150458.svg)](https://pandas.pydata.org/)
[![Astropy](https://img.shields.io/badge/Astropy-6.1.2-306998.svg)](https://www.astropy.org/)
[![SpiceyPy](https://img.shields.io/badge/SpiceyPy-6.0.0-darkgreen.svg)](https://spiceypy.readthedocs.io/)

<!-- Development & Quality -->
[![Click](https://img.shields.io/badge/CLI-Click-blue.svg)](https://click.palletsprojects.com/)
[![Pydantic](https://img.shields.io/badge/validation-Pydantic-E92063.svg)](https://pydantic.dev/)
[![Pytest](https://img.shields.io/badge/testing-Pytest-0A9EDC.svg)](https://pytest.org/)
[![Ruff](https://img.shields.io/badge/linting-Ruff-FCC21B.svg)](https://docs.astral.sh/ruff/)
[![MyPy](https://img.shields.io/badge/typing-MyPy-00B4AB.svg)](https://mypy-lang.org/)

<!-- Features -->
[![No Mocking](https://img.shields.io/badge/testing-No%20Mocking-brightgreen.svg)](tests/)
[![JAX Compatible](https://img.shields.io/badge/optimization-JAX%20Compatible-red.svg)](src/optimization/differentiable/)
[![Orbital Mechanics](https://img.shields.io/badge/physics-PyKEP%20Integration-orange.svg)](src/trajectory/)
[![Multi Objective](https://img.shields.io/badge/optimization-NSGA--II-orange.svg)](src/optimization/)

An integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions.

## 🚀 Overview

The Lunar Horizon Optimizer is a comprehensive platform that enables real-world lunar mission design and analysis. From basic Earth-Moon transfers to complex economic modeling of lunar resource utilization, this tool supports the full spectrum of space mission engineering challenges.

**Core Technologies:**
- **High-fidelity orbital mechanics** using PyKEP 2.6
- **Global optimization** with PyGMO 2.19.6 (NSGA-II)
- **Differentiable programming** with JAX 0.5.3 + Diffrax 0.7.0
- **Economic analysis** with ISRU modeling and sensitivity analysis
- **Interactive visualization** with Plotly dashboards
- **Extensible plugin architecture** for custom components

**Applications**: Mission architecture trade studies, lunar mining business cases, real-time trajectory optimization, constellation design, investment analysis, and policy planning.

## 📈 Project Status

**Tasks Completed**: 10/10

🎉 **Project is FEATURE-COMPLETE!** All core tasks have been successfully implemented.

### 🧪 Test Status
- **Production Tests**: 243/243 passing (100% success rate)
- **Core Modules**: 161/161 tests passing across 11 key modules
- **CLI Tests**: All 10 scenarios tested and working
- **No Mocking Policy**: 100% real implementations (PyKEP, PyGMO, JAX)
- **Pipeline Status**: Clean with only 2 acceptable complexity warnings

## ✨ Key Features

- 🔄 **Differentiable Optimization**: JAX-based gradient optimization
- 💰 **Economic Analysis**: NPV, IRR, ROI calculations with ISRU benefits
- ⚡ **Global Optimization**: Multi-objective optimization with Pareto front analysis
- 📈 **Interactive Visualization**: 3D trajectory plots and economic dashboards
- 📊 **Cost Modeling**: Wright's law learning curves and environmental costs
- 🛸 **Trajectory Generation**: Lambert solvers, N-body integration, transfer window analysis, powered descent modeling

## 🆕 Recent Updates

### Powered Descent & Test Suite Excellence (Latest)
- ✅ **Powered Descent Module**: JAX/Diffrax-based lunar landing trajectory modeling
- ✅ **Continuous Thrust**: Moon-centered inertial frame with optimal braking control
- ✅ **Gradient-Compatible**: Full JAX integration for differentiable optimization
- ✅ **100% Test Success**: 243/243 production tests passing with real implementations
- ✅ **No Mocking Policy**: All tests use real PyKEP/PyGMO/JAX/Diffrax implementations
- ✅ **Modern CLI Complete**: Comprehensive Click-based interface with rich progress tracking
- ✅ **Production Ready**: Clean pipeline with comprehensive error handling

### Core Implementation Complete
- ✅ **Differentiable Optimization**: JAX-based gradient optimization with real calculations
- ✅ **Economic Analysis**: NPV, IRR, ROI with ISRU modeling and sensitivity analysis
- ✅ **Global Optimization**: Multi-objective PyGMO NSGA-II with Pareto front analysis
- ✅ **Wright's Law Integration**: Launch cost learning curves and environmental costs
- ✅ **Interactive Visualization**: 3D trajectory plots and economic dashboards
- ✅ **Extensible Architecture**: Plugin system for custom components and extensions

## 🏗️ Project Structure

**Codebase Scale**: 150 Python files, 326 classes, 340 functions

```
📁 Lunar Horizon Optimizer/
├── 📁 src/                    # Source code (main implementation)
│   ├── config/                # Mission configuration management
│   ├── trajectory/            # Orbital mechanics & PyKEP integration
│   ├── optimization/          # PyGMO global optimization & JAX differentiable
│   ├── economics/             # Economic analysis & ISRU modeling
│   ├── visualization/         # Interactive dashboards & plotting
│   ├── extensibility/         # Plugin system & extension framework
│   └── utils/                 # Utility functions & performance optimizations
├── 📁 docs/                   # Complete documentation suite
│   ├── USER_GUIDE.md         # Getting started guide
│   ├── INDEX.md              # Documentation index
│   ├── PROJECT_STATUS.md     # Implementation status
│   └── archive/              # Historical documentation
├── 📁 tests/                  # Comprehensive test suite (415 tests)
├── 📁 scenarios/              # Mission configuration examples
├── 📁 examples/               # Usage examples and demos
├── 📁 scripts/                # Development and utility scripts
│   └── utilities/            # Maintenance scripts
├── 📁 results/                # Analysis outputs (gitignored)
└── 📁 archive/                # Historical project files
```

## 🚀 Quick Start

### Prerequisites
```bash
# Create conda environment
conda create -n py312 python=3.12 -y
conda activate py312

# Install dependencies
conda install -c conda-forge pykep pygmo astropy spiceypy -y
pip install -r requirements.txt
```

### Modern CLI Usage (Recommended)
```bash
# Make CLI executable
chmod +x lunar_opt.py

# Validate environment
./lunar_opt.py validate

# List available scenarios
./lunar_opt.py run list

# Run basic lunar cargo mission
./lunar_opt.py run scenario 01_basic_transfer

# Run comprehensive ISRU economics analysis
./lunar_opt.py run scenario 06_isru_economics --risk --export-pdf

# Get detailed help
./lunar_opt.py --help
./lunar_opt.py run scenario --help
```

### Legacy Usage
```bash
# Run production test suite
make test

# Run optimization with learning curves
python src/cli.py analyze --config examples/config_after_upgrade.json \
  --learning-rate 0.88 --carbon-price 75.0

# Run cost comparison demo
python examples/cost_comparison_demo.py
```

## ⚡ Quick Start - Reproduce the Analysis

### Option 1: Use the Examples Script
```bash
# Quick 30-second test
python run_analysis_examples.py quick

# Production 3-4 minute analysis  
python run_analysis_examples.py production

# See all options
python run_analysis_examples.py
```

### Option 2: Manual Commands
```bash
# Environment setup
conda activate py312

# Quick test (30 seconds)
python src/cli.py analyze --config scenarios/01_basic_transfer.json \
  --output quick_test --population-size 8 --generations 5 --no-sensitivity

# Production analysis (3-4 minutes) 
python src/cli.py analyze --config scenarios/01_basic_transfer.json \
  --output production_test --population-size 52 --generations 30

# Expected results: Delta-V ~22,446 m/s, NPV ~$374M
```

## 📚 Documentation

### 🚀 CLI Documentation (NEW)
- 🌟 **[CLI Help Reference](docs/CLI_HELP_REFERENCE.md)**: Complete command reference and examples
- 📖 **[CLI User Guide](docs/guides/NEW_CLI_USER_GUIDE.md)**: Comprehensive usage guide with tutorials
- 🎯 **[CLI Overview](CLI_README.md)**: Feature overview and quick start guide

### 🚀 Getting Started
- ⏱️ **[Progress Tracking Guide](docs/PROGRESS_TRACKING_GUIDE.md)**: How to run analyses and track progress
- 🎯 **[Use Cases & Scenarios](docs/USE_CASES.md)**: Real-world applications and problem scenarios
- 📊 **[Analysis Examples](scripts/utilities/run_analysis_examples.py)**: Executable script for testing different analysis types

### 📖 Technical Reference  
- 📖 **[Complete Capabilities](docs/CAPABILITIES.md)**: Comprehensive API reference
- 💰 **[Cost Model Upgrade](docs/COST_MODEL_UPGRADE.md)**: Wright's law and environmental costs
- 🧪 **[Testing Guide](tests/TEST_SUITE_DOCUMENTATION.md)**: Test suite documentation and coverage
- 🔧 **[Development Guide](CLAUDE.md)**: Project working rules and standards

### 📋 Implementation Status
- 🏆 **[Final Implementation Status](docs/FINAL_IMPLEMENTATION_STATUS.md)**: Production-ready real optimizer
- 🚫 **[Real Optimizer Only](docs/REAL_OPTIMIZER_ONLY.md)**: No mocks policy documentation
- 📋 **[Project Audit](docs/PROJECT_AUDIT.md)**: Repository structure analysis
- 🛠️ **[Cleanup Report](docs/CLEANUP_REPORT.md)**: Recent organizational improvements

## 🛠️ Development

### Available Commands
```bash
make help          # Show all available commands
make pipeline      # Run complete development pipeline
make test          # Run production test suite (243 tests, 100% pass rate)
make coverage      # Generate coverage report
make lint          # Run code quality checks
```

### Code Quality Standards
- ✅ **NO MOCKING RULE**: All tests use real PyKEP, PyGMO, JAX implementations
- ✅ **100% Test Pass Rate**: 243/243 production tests passing (415 total tests)
- ✅ **Clean Pipeline**: 0 linting errors, formatted code
- ✅ **Type Safety**: Comprehensive type hints and MyPy validation

## 🤝 Contributing

1. Follow the [development guide](CLAUDE.md)
2. Review [documentation index](docs/INDEX.md) for project overview
3. Ensure all tests pass with `make test`
4. Run quality checks with `make pipeline`
5. Commit with descriptive messages

## 📁 Documentation Index

See **[docs/INDEX.md](docs/INDEX.md)** for complete documentation navigation including:
- User guides and getting started
- API reference and technical documentation
- Task implementation details
- Testing guidelines and status

## 📄 License

This project is part of the Lunar Horizon Optimizer development.

---
*Last updated: 2025-07-13*