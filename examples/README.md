# Lunar Horizon Optimizer Examples

This directory contains comprehensive examples demonstrating the Lunar Horizon Optimizer's capabilities. Each example is designed to showcase different aspects of the system, from basic usage to advanced integration scenarios.

## 🚀 Quick Start

**New to the project?** Start with [`quick_start.py`](#quick_startpy) for a comprehensive introduction.

## 📁 Example Files Overview

### Core Examples

| File | Purpose | Complexity | Runtime |
|------|---------|------------|---------|
| [`quick_start.py`](#quick_startpy) | Complete system demonstration | ⭐⭐⭐ | ~30s |
| [`working_example.py`](#working_examplepy) | Basic working example | ⭐⭐ | ~15s |
| [`integration_test.py`](#integration_testpy) | PRD compliance validation | ⭐⭐⭐⭐ | ~60s |

### Advanced Integration Tests

| File | Purpose | Complexity | Runtime |
|------|---------|------------|---------|
| [`simple_trajectory_test.py`](#simple_trajectory_testpy) | Trajectory integration validation | ⭐⭐⭐ | ~10s |
| [`advanced_trajectory_test.py`](#advanced_trajectory_testpy) | Comprehensive trajectory testing | ⭐⭐⭐⭐ | ~120s |
| [`final_integration_test.py`](#final_integration_testpy) | Complete system integration | ⭐⭐⭐⭐⭐ | ~30s |
| [`differentiable_optimization_demo.py`](#differentiable_optimization_demopy) | JAX/Diffrax differentiable optimization | ⭐⭐⭐⭐ | ~45s |

### Configuration Examples

| File | Purpose | Description |
|------|---------|-------------|
| [`configs/basic_mission.yaml`](#configsbasic_missionyaml) | Mission configuration | Sample mission parameters |

## 📖 Detailed Example Documentation

### `quick_start.py`

**🎯 Purpose**: Complete introduction to the Lunar Horizon Optimizer system

**Features Demonstrated**:
- Mission configuration setup
- Trajectory generation with Lambert solvers
- Multi-objective optimization with PyGMO
- Economic analysis (NPV, IRR, ROI)
- Interactive visualization dashboards
- Complete workflow integration

**Usage**:
```bash
conda activate py312
python examples/quick_start.py
```

**Expected Output**:
- Mission configuration validation
- Trajectory optimization results
- Economic analysis summary
- Interactive Plotly dashboards
- Complete system performance metrics

**Key Learning Points**:
- How to configure mission parameters
- Basic trajectory optimization workflow
- Economic analysis integration
- Visualization system usage

---

### `working_example.py`

**🎯 Purpose**: Simple, focused example of core functionality

**Features Demonstrated**:
- Basic trajectory generation
- Simple optimization
- Core visualization
- Streamlined workflow

**Usage**:
```bash
conda activate py312
python examples/working_example.py
```

**Expected Output**:
- Trajectory calculation results
- Basic optimization output
- Simple visualization plots

**Key Learning Points**:
- Minimal setup requirements
- Core system functionality
- Basic API usage patterns

---

### `integration_test.py`

**🎯 Purpose**: Comprehensive PRD compliance validation

**Features Demonstrated**:
- All 5 PRD user workflows
- Cross-module integration testing
- System capability validation
- Performance benchmarking

**Usage**:
```bash
conda activate py312
python examples/integration_test.py
```

**Expected Output**:
- PRD compliance report
- Integration test results
- Performance metrics
- Capability validation summary

**Key Learning Points**:
- Complete system capabilities
- PRD requirement fulfillment
- Integration patterns
- Performance characteristics

---

### `simple_trajectory_test.py`

**🎯 Purpose**: Focused trajectory generation integration testing

**Features Demonstrated**:
- Lambert solver integration
- Trajectory data structures
- Visualization compatibility
- Error handling patterns

**Usage**:
```bash
conda activate py312
python examples/simple_trajectory_test.py
```

**Expected Output**:
- Trajectory generation validation
- Data structure verification
- Integration test results
- Visualization compatibility confirmation

**Key Learning Points**:
- Trajectory generation API
- Data structure requirements
- Integration patterns
- Error handling approaches

---

### `advanced_trajectory_test.py`

**🎯 Purpose**: Comprehensive trajectory system testing

**Features Demonstrated**:
- Lambert solver advanced features
- Patched conics approximation
- Optimal timing calculations
- Multi-revolution solutions
- Complex trajectory scenarios

**Usage**:
```bash
conda activate py312
python examples/advanced_trajectory_test.py
```

**Expected Output**:
- Advanced trajectory calculations
- Multi-method comparisons
- Timing optimization results
- Complex scenario handling

**Key Learning Points**:
- Advanced trajectory techniques
- Multi-method trajectory generation
- Timing optimization strategies
- Complex scenario handling

---

### `final_integration_test.py`

**🎯 Purpose**: Complete system integration validation

**Features Demonstrated**:
- All integration improvements
- Cross-module workflow automation
- Configuration system validation
- Performance optimization
- PRD compliance measurement

**Usage**:
```bash
conda activate py312
python examples/final_integration_test.py
```

**Expected Output**:
- Complete integration test results
- PRD compliance improvement metrics
- System performance summary
- Integration validation report

**Key Learning Points**:
- Complete system integration
- Cross-module communication
- Performance optimization
- Compliance validation

---

### `differentiable_optimization_demo.py`

**🎯 Purpose**: Comprehensive demonstration of JAX/Diffrax differentiable optimization

**Features Demonstrated**:
- JAX and Diffrax availability verification
- Differentiable trajectory and economic models
- Gradient-based optimization with automatic differentiation
- JIT compilation for performance optimization
- Batch optimization for multiple starting points
- PyGMO-JAX integration for hybrid optimization
- Performance comparison vs numerical methods
- Advanced JAX features (grad, vmap, jit)

**Usage**:
```bash
conda activate py312
python examples/differentiable_optimization_demo.py
```

**Expected Output**:
- JAX/Diffrax system verification
- Differentiable model demonstrations
- Gradient-based optimization results
- Batch optimization performance metrics
- PyGMO-JAX hybrid optimization
- Performance comparison analysis

**Key Learning Points**:
- Understanding JAX automatic differentiation
- Implementing gradient-based optimization
- Leveraging JIT compilation for performance
- Integrating global and local optimization methods
- Using advanced JAX features for optimization

---

### `configs/basic_mission.yaml`

**🎯 Purpose**: Sample mission configuration file

**Contents**:
- Mission parameters
- Spacecraft specifications
- Cost factors
- Orbital parameters
- Economic assumptions

**Usage**:
```python
from src.config.loader import ConfigLoader
config = ConfigLoader.load_yaml('examples/configs/basic_mission.yaml')
```

**Key Learning Points**:
- Configuration file structure
- Parameter organization
- YAML formatting requirements
- Configuration validation

## 🛠️ Running Examples

### Prerequisites

Ensure you have the correct environment setup:

```bash
# Activate the required conda environment
conda activate py312

# Verify installation
python scripts/verify_dependencies.py
```

### Environment Requirements

- **Python 3.12** (conda py312 environment)
- **PyKEP 2.6** (conda-forge installation required)
- **PyGMO 2.19.6** (conda-forge installation required)
- **JAX 0.5.3** (with CPU or GPU support)
- **All dependencies from requirements.txt**

### Execution Order

For new users, we recommend running examples in this order:

1. **Start Here**: `quick_start.py` - Complete system overview
2. **Basic Usage**: `working_example.py` - Simple functionality
3. **Integration**: `simple_trajectory_test.py` - Core integration
4. **Advanced**: `advanced_trajectory_test.py` - Advanced features
5. **Validation**: `integration_test.py` - PRD compliance
6. **Complete**: `final_integration_test.py` - Full system test

### Expected Runtime

- **Quick examples** (working_example.py): ~15 seconds
- **Standard examples** (quick_start.py): ~30 seconds
- **Advanced examples** (advanced_trajectory_test.py): ~2 minutes
- **Integration tests** (integration_test.py): ~1 minute

## 🔧 Troubleshooting

### Common Issues

1. **Environment Issues**
   ```bash
   # Solution: Ensure py312 environment is active
   conda activate py312
   ```

2. **Missing Dependencies**
   ```bash
   # Solution: Install conda dependencies first
   conda install -c conda-forge pykep pygmo astropy spiceypy -y
   pip install -r requirements.txt
   ```

3. **Import Errors**
   ```bash
   # Solution: Run from project root
   cd /path/to/lunar-horizon-optimizer
   python examples/quick_start.py
   ```

4. **Performance Issues**
   ```bash
   # Solution: Reduce complexity for testing
   # Edit example files to use smaller population_size and num_generations
   ```

### Debug Mode

For detailed debugging, set logging level in examples:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

- **Documentation**: See [`docs/`](../docs/) directory
- **User Guide**: [`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)
- **API Reference**: [`docs/API_REFERENCE.md`](../docs/API_REFERENCE.md)
- **Integration Guide**: [`docs/INTEGRATION_GUIDE.md`](../docs/INTEGRATION_GUIDE.md)

## 📊 Performance Benchmarks

### System Performance Metrics

| Example | Trajectory Gen | Optimization | Economic Analysis | Total Runtime |
|---------|---------------|--------------|------------------|---------------|
| quick_start.py | ~2s | ~20s | ~1s | ~30s |
| working_example.py | ~1s | ~10s | ~1s | ~15s |
| integration_test.py | ~5s | ~40s | ~5s | ~60s |

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Optimal**: 32GB RAM, 16 CPU cores, GPU support

## 🎯 Learning Path

### Beginner Path
1. Read [`docs/USER_GUIDE.md`](../docs/USER_GUIDE.md)
2. Run `quick_start.py`
3. Examine `working_example.py`
4. Modify parameters and re-run

### Intermediate Path
1. Study `simple_trajectory_test.py`
2. Understand `configs/basic_mission.yaml`
3. Run `advanced_trajectory_test.py`
4. Create custom configurations

### Advanced Path
1. Analyze `integration_test.py`
2. Study `final_integration_test.py`
3. Develop custom extensions
4. Contribute to the project

## 🔗 Related Documentation

- **[Main README](../README.md)** - Project overview
- **[User Guide](../docs/USER_GUIDE.md)** - Comprehensive usage guide
- **[API Reference](../docs/API_REFERENCE.md)** - Complete API documentation
- **[Integration Guide](../docs/INTEGRATION_GUIDE.md)** - Cross-module integration
- **[CLAUDE.md](../CLAUDE.md)** - Development guidelines

---

*Last Updated: December 2024 - All examples tested and validated*
*Environment: Python 3.12 with conda py312 environment*