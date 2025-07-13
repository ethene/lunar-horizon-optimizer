# 🌙 Lunar Horizon Optimizer

An integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions.

## 🚀 Overview

The Lunar Horizon Optimizer is a comprehensive platform that combines:
- **High-fidelity orbital mechanics** using PyKEP 2.6
- **Global optimization** with PyGMO 2.19.6 (NSGA-II)
- **Differentiable programming** with JAX 0.5.3 + Diffrax 0.7.0
- **Economic analysis** with ISRU modeling and sensitivity analysis
- **Interactive visualization** with Plotly dashboards
- **Extensible plugin architecture** for custom components

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

### Cost Model Upgrade (Latest)
- ✅ **Wright's Law Learning Curves**: Launch costs reduce over time with production scaling
- ✅ **Environmental Cost Integration**: CO₂ emissions pricing and carbon cost accounting
- ✅ **CLI Enhancement**: `--learning-rate` and `--carbon-price` flags for parameter control
- ✅ **Comprehensive Testing**: 21 new unit tests with real implementation (NO MOCKING)
- ✅ **Production Ready**: 243/243 tests passing, clean pipeline

## 🏗️ Architecture

**Codebase Scale**: 150 Python files, 326 classes, 340 functions

```
src/
tests/           # Comprehensive test suite
docs/            # Documentation
examples/        # Usage examples
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

### Basic Usage
```bash
# Run production test suite
make test

# Run optimization with learning curves
python src/cli.py analyze --config examples/config_after_upgrade.json \
  --learning-rate 0.88 --carbon-price 75.0

# Run cost comparison demo
python examples/cost_comparison_demo.py
```

## 📚 Documentation

- 📖 **[Complete Capabilities](CAPABILITIES.md)**: Comprehensive API reference
- 💰 **[Cost Model Upgrade](docs/COST_MODEL_UPGRADE.md)**: Wright's law and environmental costs
- 🧪 **[Testing Guide](tests/README.md)**: Test suite documentation
- 🔧 **[Development Guide](CLAUDE.md)**: Project working rules and standards

## 🛠️ Development

### Available Commands
```bash
make help          # Show all available commands
make pipeline      # Run complete development pipeline
make test          # Run production test suite (243 tests)
make coverage      # Generate coverage report
make lint          # Run code quality checks
```

### Code Quality Standards
- ✅ **NO MOCKING RULE**: All tests use real PyKEP, PyGMO, JAX implementations
- ✅ **100% Test Pass Rate**: 243/243 production tests passing
- ✅ **Clean Pipeline**: 0 linting errors, formatted code
- ✅ **Type Safety**: Comprehensive type hints and MyPy validation

## 🤝 Contributing

1. Follow the [development guide](CLAUDE.md)
2. Ensure all tests pass with `make test`
3. Run quality checks with `make pipeline`
4. Commit with descriptive messages

## 📄 License

This project is part of the Lunar Horizon Optimizer development.

---
*Last updated: 2025-07-13*