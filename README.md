# Lunar Horizon Optimizer

An integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions. This platform combines high-fidelity n-body dynamics with economic performance metrics to optimize lunar mission trajectories for both physical feasibility and financial returns.

**Project Status**: ✅ **Feature Complete** - All 10 planned tasks implemented  
**Getting Started**: See [USER_GUIDE.md](docs/USER_GUIDE.md) for complete user guide  
**Documentation**: See [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for detailed status  
**Version**: 1.0.0

## Features

- **Trajectory Simulation & Optimization**
  - Global optimization using PyGMO (NSGA-II)
  - Local differentiable optimization with JAX and Diffrax
  - High-fidelity n-body dynamics using PyKEP

- **Economic Analysis**
  - ROI and NPV calculations
  - Cost modeling for mission phases
  - ISRU benefit analysis

- **Visualization**
  - Interactive 3D trajectory visualization
  - Economic metrics dashboards
  - Multi-objective Pareto front exploration

## Requirements

- **Python 3.12** (conda py312 environment required)
- GPU support recommended for JAX acceleration (currently running on CPU)

### Core Dependencies
- SciPy 1.13.1 (required for PyKEP compatibility)
- PyKEP 2.6 - Orbital mechanics and trajectory calculations (conda-forge required)
- PyGMO 2.19.6 - Global optimization algorithms
- JAX 0.5.3 - Differentiable programming and local optimization
- Diffrax 0.7.0 - Differentiable ordinary differential equation solvers
- Plotly 5.24.1 - Interactive visualization
- Poliastro 0.17.0 - Orbital mechanics utilities and visualization
- SpiceyPy - NASA SPICE toolkit for ephemeris calculations

### Development Dependencies
- pytest >= 7.4.0 - Testing framework
- black >= 23.7.0 - Code formatting
- flake8 >= 6.1.0 - Code linting
- mypy >= 1.5.1 - Static type checking

### Optional Dependencies
- jupyter >= 1.0.0 - For running notebooks
- ipykernel >= 6.25.0 - Jupyter notebook kernel

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/lunar-horizon-optimizer.git
   cd lunar-horizon-optimizer
   ```

2. Create and activate conda environment (required for PyKEP):
```bash
   conda create -n py312 python=3.12 -y
   conda activate py312
   ```

3. Install core dependencies via conda:
```bash
   conda install -c conda-forge pykep pygmo astropy spiceypy -y
   ```

4. Install remaining dependencies:
```bash
   pip install -r requirements.txt
   ```

5. Verify installation:
```bash
   python scripts/verify_dependencies.py
   ```

## Project Structure

After comprehensive refactoring, the codebase now follows a clean, modular architecture:

```
.
├── src/                    # Source code (refactored for maintainability)
│   ├── config/            # Mission configuration and parameters
│   │   ├── management/    # Modular config management (Phase 2B)
│   │   │   ├── config_manager.py     # Core configuration orchestration
│   │   │   ├── template_manager.py   # Template-based config creation
│   │   │   └── file_operations.py    # Config file I/O operations
│   │   ├── models.py      # Consolidated configuration models
│   │   ├── costs.py       # Cost modeling parameters
│   │   ├── isru.py        # ISRU configuration
│   │   └── spacecraft.py  # Spacecraft specifications
│   ├── trajectory/        # Orbital mechanics and calculations
│   │   ├── validation/    # Modular validation package (Phase 2A)
│   │   │   ├── physics_validation.py    # Orbital mechanics validation
│   │   │   ├── constraint_validation.py # Trajectory constraints
│   │   │   └── vector_validation.py     # Vector operations
│   │   ├── lunar_transfer.py        # Refactored with extracted methods (Phase 2C)
│   │   ├── celestial_bodies.py      # Celestial body state calculations
│   │   ├── propagator.py            # Trajectory propagation
│   │   └── trajectory_validator.py  # Consolidated validation class
│   ├── utils/             # Utility functions and conversions
│   └── constants/         # Physical constants
├── tests/                 # Comprehensive test suite
├── docs/                  # Project documentation
│   ├── refactoring_plan.md         # Completed refactoring documentation
│   ├── trajectory_modules.md       # Module documentation
│   └── trajectory_tests.md         # Testing strategy
├── tasks/                 # Development task management
├── scripts/               # Utility scripts and PRD
└── CLAUDE.md              # Development guidelines (conda py312 requirements)
```

### Key Refactoring Improvements

- **🔧 Modular Architecture**: Large files split into focused, maintainable modules
- **📦 Package Organization**: Related functionality grouped into cohesive packages
- **🔄 Backward Compatibility**: All legacy interfaces maintained with deprecation warnings
- **✅ Comprehensive Testing**: All refactored modules validated and functional
- **📚 Clear Separation of Concerns**: Each module has a single, well-defined responsibility

## Development Setup

1. Install development dependencies:
```bash
   pip install -r requirements.txt
   ```

2. Set up pre-commit hooks (coming soon)

3. Run tests:
```bash
   pytest tests/
   ```

## Documentation

### Quick Start
- **[CLAUDE.md](CLAUDE.md)** - Essential development guidelines and project rules
- **[docs/README.md](docs/README.md)** - Complete documentation overview

### Development Resources  
- **[Refactoring Plan](docs/refactoring_plan.md)** - Code restructuring strategy
- **[Trajectory Modules](docs/trajectory_modules.md)** - Module implementation details
- **[Test Documentation](docs/trajectory_tests.md)** - Testing approach and coverage
- **[Task Management](tasks/)** - Current development priorities

## License

[License information to be added]

## Contributing

Contribution guidelines will be added soon.