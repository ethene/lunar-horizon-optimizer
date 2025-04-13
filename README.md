# Lunar Horizon Optimizer

An integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions. This platform combines high-fidelity n-body dynamics with economic performance metrics to optimize lunar mission trajectories for both physical feasibility and financial returns.

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

- Python 3.8+
- GPU support recommended for JAX acceleration

### Core Dependencies
- PyKEP >= 2.6 - Orbital mechanics and trajectory calculations
- PyGMO >= 2.19 - Global optimization algorithms
- JAX >= 0.4.13 - Differentiable programming and local optimization
- Diffrax >= 0.4.0 - Differentiable ordinary differential equation solvers
- Plotly >= 5.18.0 - Interactive visualization
- Poliastro >= 0.17.0 - Orbital mechanics utilities and visualization

### Development Dependencies
- pytest >= 7.4.0 - Testing framework
- black >= 23.3.0 - Code formatting
- flake8 >= 6.0.0 - Code linting
- mypy >= 1.4.1 - Static type checking

### Optional Dependencies
- jupyter >= 1.0.0 - For running notebooks
- ipykernel >= 6.25.0 - Jupyter notebook kernel

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lunar-horizon-optimizer.git
   cd lunar-horizon-optimizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python scripts/verify_dependencies.py
   ```

## Project Structure

```
.
├── src/            # Source code
├── tests/          # Test files
├── docs/           # Documentation
├── notebooks/      # Jupyter notebooks
└── scripts/        # Utility scripts
```

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

## License

[License information to be added]

## Contributing

Contribution guidelines will be added soon.