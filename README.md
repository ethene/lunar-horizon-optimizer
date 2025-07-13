# 🌙 Lunar Horizon Optimizer

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

## ✨ Key Features

- 🔄 **Differentiable Optimization**: JAX-based gradient optimization
- 💰 **Economic Analysis**: NPV, IRR, ROI calculations with ISRU benefits
- ⚡ **Global Optimization**: Multi-objective optimization with Pareto front analysis
- 📈 **Interactive Visualization**: 3D trajectory plots and economic dashboards
- 📊 **Cost Modeling**: Wright's law learning curves and environmental costs
- 🛸 **Trajectory Generation**: Lambert solvers, N-body integration, transfer window analysis

## 🆕 Recent Updates

### Modern CLI Complete (Latest)
- ✅ **Modern Click-based CLI**: Comprehensive command-line interface with rich help
- ✅ **Scenario-Based Workflows**: 10 predefined scenarios with automatic discovery
- ✅ **Rich Progress Tracking**: Real-time optimization monitoring with live updates
- ✅ **All Scenarios Tested**: 100% success rate across all 10 lunar mission scenarios
- ✅ **Executable Tool**: Direct execution with `./lunar_opt.py` command
- ✅ **Comprehensive Documentation**: Complete CLI help reference and user guides

### Previous Major Updates
- ✅ **No Mocks Policy**: 100% real calculations using PyKEP/PyGMO/JAX
- ✅ **Progress Tracking**: Real-time analysis progress with accurate time estimates  
- ✅ **Performance Optimization**: JIT compilation and parallel processing
- ✅ **Production Results**: Delta-V 22,446 m/s, NPV $374M for Apollo-class missions
- ✅ **Wright's Law Learning Curves**: Launch costs reduce over time with production scaling
- ✅ **Environmental Cost Integration**: CO₂ emissions pricing and carbon cost accounting

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