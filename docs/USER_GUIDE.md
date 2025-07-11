# Lunar Horizon Optimizer - User Guide

Welcome to the Lunar Horizon Optimizer! This guide will help you get started with optimizing lunar mission trajectories and analyzing their economic feasibility.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start Examples](#quick-start-examples)
3. [Major Workflows](#major-workflows)
4. [Understanding Results](#understanding-results)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Getting Started

### Prerequisites

Before using the Lunar Horizon Optimizer, ensure you have:
- Python 3.12 environment (conda py312)
- All dependencies installed (`make install-dev`)
- Activated the conda environment (`conda activate py312`)

### Basic Concepts

The optimizer works with four main concepts:
1. **Mission Configuration** - Your mission parameters (payload, budget, timeline)
2. **Trajectory Optimization** - Finding the best path from Earth to Moon
3. **Economic Analysis** - Calculating costs, ROI, and financial viability
4. **Trade-off Analysis** - Balancing competing objectives (cost vs. time vs. fuel)

---

## Quick Start Examples

### Example 1: Basic Trajectory Optimization

```python
from src.lunar_horizon_optimizer import LunarHorizonOptimizer
from src.config.models import MissionParameters

# Create optimizer
optimizer = LunarHorizonOptimizer()

# Configure mission
config = MissionParameters(
    payload_mass=1000,  # kg
    launch_date="2025-12-01",
    mission_duration_days=10,
    budget_millions=50
)

# Run optimization
results = optimizer.optimize(config)

# Display results
print(f"Optimal Delta-V: {results.delta_v:.1f} m/s")
print(f"Total Cost: ${results.total_cost/1e6:.1f}M")
print(f"ROI: {results.roi:.1%}")
```

### Example 2: Economic Analysis Only

```python
from src.economics.financial_models import MissionEconomics
from src.economics.isru_benefits import ISRUBenefitAnalyzer

# Analyze mission economics
economics = MissionEconomics()

# Define mission costs
costs = {
    "launch": 50_000_000,
    "spacecraft": 30_000_000,
    "operations": 20_000_000,
    "payload": 10_000_000
}

# Calculate with ISRU benefits
isru = ISRUBenefitAnalyzer()
isru_savings = isru.calculate_savings(
    resource="water",
    quantity_kg=1000,
    mission_duration_days=30
)

# Get financial metrics
metrics = economics.analyze(
    costs=costs,
    revenues=150_000_000,
    isru_savings=isru_savings,
    mission_years=5
)

print(f"NPV: ${metrics['npv']:,.0f}")
print(f"IRR: {metrics['irr']:.1%}")
print(f"Payback Period: {metrics['payback_years']:.1f} years")
```

### Example 3: Visualization

```python
from src.visualization.integrated_dashboard import create_mission_dashboard

# Create interactive dashboard
dashboard = create_mission_dashboard(
    trajectory_results=results.trajectory,
    economic_results=results.economics,
    title="Lunar Mission Analysis"
)

# Save to HTML
dashboard.write_html("mission_analysis.html")

# Or display in Jupyter notebook
dashboard.show()
```

---

## Major Workflows

### Workflow 1: Complete Mission Analysis

This workflow combines trajectory optimization with economic analysis:

```python
# Step 1: Load configuration
from src.config.config_loader import load_mission_config
config = load_mission_config("configs/example_mission.yaml")

# Step 2: Run global optimization
from src.optimization.global_optimizer import GlobalOptimizer
global_opt = GlobalOptimizer(config)
pareto_front = global_opt.optimize(generations=50)

# Step 3: Select and refine best solution
best_solution = pareto_front.get_best_compromise()

# Step 4: Refine with local optimization
from src.optimization.differentiable.jax_optimizer import refine_trajectory
refined = refine_trajectory(best_solution)

# Step 5: Comprehensive economic analysis
from src.economics.integrated_analyzer import IntegratedAnalyzer
analyzer = IntegratedAnalyzer()
economics = analyzer.analyze_mission(refined, config)

# Step 6: Generate report
from src.economics.reporting import generate_mission_report
report = generate_mission_report(refined, economics, config)
report.save("mission_report.pdf")
```

### Workflow 2: Trade Study Analysis

Compare multiple mission scenarios:

```python
from src.economics.scenario_comparison import ScenarioComparison

# Define scenarios
scenarios = [
    {"name": "Baseline", "payload": 1000, "duration": 10},
    {"name": "Heavy Cargo", "payload": 2000, "duration": 15},
    {"name": "Fast Transit", "payload": 500, "duration": 5},
]

# Run comparison
comparison = ScenarioComparison()
results = comparison.analyze_scenarios(scenarios)

# Visualize trade-offs
comparison.plot_trade_space(results)
```

### Workflow 3: ISRU-Enhanced Mission

Analyze missions with In-Situ Resource Utilization:

```python
from src.economics.advanced_isru_models import TimeBasedISRUModel
from datetime import datetime

# Create ISRU model
isru_model = TimeBasedISRUModel()

# Add production facility
isru_model.add_facility(
    name="Lunar Water Extractor",
    startup_date=datetime(2026, 1, 1),
    peak_capacity_kg_per_day=100,
    capital_cost=50_000_000
)

# Calculate production over mission
production = isru_model.calculate_cumulative_production(
    resource="water",
    start_date=datetime(2026, 1, 1),
    end_date=datetime(2031, 1, 1)
)

print(f"Total Water Produced: {production['cumulative_production']:,.0f} kg")
print(f"Cost Savings: ${production['cost_savings']:,.0f}")
```

---

## Understanding Results

### Trajectory Results

The optimizer returns trajectory information including:
- **Delta-V**: Total velocity change required (m/s)
- **Time of Flight**: Duration from Earth to Moon (days)
- **Departure/Arrival Dates**: Optimal launch windows
- **Trajectory Path**: 3D coordinates over time

### Economic Metrics

Key financial indicators:
- **NPV (Net Present Value)**: Total value in today's dollars
- **IRR (Internal Rate of Return)**: Effective investment return rate
- **ROI (Return on Investment)**: Percentage return on costs
- **Payback Period**: Years to recover investment

### Pareto Front

The multi-objective optimization produces a Pareto front showing trade-offs:
- Each point represents a different mission design
- No single "best" solution - depends on priorities
- Interactive plots let you explore options

---

## Advanced Features

### Custom Objectives

Define your own optimization objectives:

```python
from src.optimization.differentiable.loss_functions import create_custom_loss

def safety_objective(trajectory):
    """Minimize radiation exposure."""
    van_allen_time = calculate_van_allen_transit_time(trajectory)
    return van_allen_time * radiation_dose_rate

# Add to optimization
optimizer.add_objective(safety_objective, weight=0.3)
```

### Plugin Extensions

Extend functionality with plugins:

```python
from src.extensibility.extension_manager import ExtensionManager

# Load custom extension
manager = ExtensionManager()
manager.register_extension("my_custom_analysis", MyAnalysisPlugin())

# Use in optimization
results = optimizer.optimize(config, extensions=["my_custom_analysis"])
```

### Sensitivity Analysis

Understand how changes affect results:

```python
from src.economics.sensitivity_analysis import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(baseline_results)

# Vary parameters
sensitivity = analyzer.analyze(
    parameters={
        "launch_cost": (-20, +20),  # ±20% variation
        "payload_mass": (-10, +10),  # ±10% variation
        "isru_efficiency": (-30, +30)  # ±30% variation
    }
)

# Plot tornado diagram
analyzer.plot_tornado_diagram(sensitivity)
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'pykep'`

**Solution**: Ensure you're in the correct conda environment:
```bash
conda activate py312
python -m src.lunar_horizon_optimizer
```

#### 2. Trajectory Convergence Issues
**Problem**: Optimization doesn't converge or produces NaN values

**Solution**: Adjust optimization parameters:
```python
# Increase generations for difficult problems
results = optimizer.optimize(config, generations=100)

# Or relax constraints
config.constraints.max_delta_v = 15000  # m/s
```

#### 3. Memory Issues
**Problem**: Out of memory during optimization

**Solution**: Reduce population size or use batch processing:
```python
# Smaller population
optimizer.set_population_size(50)  # Default is 100

# Or process in batches
results = optimizer.optimize_batch(configs, batch_size=10)
```

#### 4. Visualization Not Displaying
**Problem**: Plots don't appear or save incorrectly

**Solution**: Check backend and use appropriate method:
```python
# For scripts
import matplotlib
matplotlib.use('Agg')  # For headless environments

# For Jupyter notebooks
%matplotlib inline

# Save instead of show
fig.write_html("output.html")  # For Plotly
plt.savefig("output.png")  # For Matplotlib
```

### Performance Issues

#### Slow Optimization
- Enable JAX JIT compilation: `optimizer.enable_jit()`
- Use GPU if available: `optimizer.use_gpu()`
- Reduce problem complexity initially, then refine

#### Large Result Files
- Use compression: `results.save("output.pkl.gz", compress=True)`
- Save only essential data: `results.save_summary("summary.json")`

### Data Issues

#### Invalid Configuration
- Use validation: `config.validate()` before optimization
- Check units (PyKEP uses SI units: meters, seconds, radians)
- Ensure dates are within valid ephemeris range

#### Numerical Instabilities
- Use robust algorithms: `optimizer.use_robust_mode()`
- Check condition numbers: `results.check_numerical_stability()`
- Scale variables appropriately

---

## FAQ

### Q: What's the difference between global and local optimization?
**A**: Global optimization (PyGMO) explores the entire solution space to find multiple good solutions. Local optimization (JAX) refines a specific solution using gradients for maximum performance.

### Q: How do I choose between different Pareto solutions?
**A**: Use the preference articulation tools:
```python
# Weight-based selection
best = pareto_front.select_by_weights({"cost": 0.4, "time": 0.3, "fuel": 0.3})

# Or constraint-based
best = pareto_front.select_with_constraints(max_cost=100e6, max_time_days=15)
```

### Q: Can I optimize for missions beyond Earth-Moon?
**A**: The current version focuses on LEO-Moon trajectories. The extensibility interface allows adding other destinations:
```python
from src.extensibility.examples.mars_extension import MarsTrajectoryExtension
optimizer.register_extension(MarsTrajectoryExtension())
```

### Q: How accurate are the economic predictions?
**A**: The models use industry-standard financial calculations with configurable uncertainty:
```python
# Add uncertainty analysis
results = economics.analyze_with_uncertainty(
    monte_carlo_runs=1000,
    cost_uncertainty=0.2,  # ±20%
    revenue_uncertainty=0.3  # ±30%
)
```

### Q: What's the recommended workflow for beginners?
**A**: Start with:
1. Run example configurations in `examples/`
2. Modify parameters gradually
3. Use visualization to understand results
4. Progress to custom objectives once comfortable

---

## Getting Help

### Resources
- **API Documentation**: See `docs/api_reference.md`
- **Examples**: Check `examples/` directory
- **Tests**: Review `tests/` for usage patterns

### Support Channels
- GitHub Issues: Report bugs or request features
- Documentation: `docs/` directory
- Code Comments: Extensive inline documentation

### Best Practices
1. Start with validated example configurations
2. Visualize results to verify correctness
3. Run sensitivity analysis on critical parameters
4. Document your mission configurations
5. Use version control for configurations

---

*Happy optimizing! May your trajectories be efficient and your missions profitable.* 🚀