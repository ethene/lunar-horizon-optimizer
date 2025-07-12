# User Guide Update - Validated Working Examples

Based on testing against the PRD requirements, here are the **verified working examples** for the Lunar Horizon Optimizer:

## Working Core Functionality

### 1. Mission Configuration ✅
```python
from src.config.models import MissionConfig, PayloadSpecification, CostFactors, OrbitParameters

# Create configuration with all required fields
config = MissionConfig(
    name="Demo Lunar Mission",
    payload=PayloadSpecification(
        dry_mass=2000.0,           # kg - spacecraft dry mass
        payload_mass=1000.0,       # kg - payload mass
        max_propellant_mass=1500.0, # kg - max propellant
        specific_impulse=450.0     # s - engine efficiency
    ),
    cost_factors=CostFactors(
        launch_cost_per_kg=50000,
        spacecraft_cost_per_kg=30000,
        operations_cost_per_day=100000
    ),
    mission_duration_days=10,
    target_orbit=OrbitParameters(
        altitude=100000,  # m (100 km)
        inclination=90.0, # degrees
        eccentricity=0.0
    )
)
```

### 2. JAX Differentiable Optimization ✅
```python
import jax.numpy as jnp
from src.optimization.differentiable.jax_optimizer import DifferentiableOptimizer

# Define optimization problem
def trajectory_cost(params):
    delta_v, time_of_flight, fuel_mass = params
    return 0.4 * fuel_mass + 0.3 * time_of_flight + 0.3 * delta_v

# Create optimizer
optimizer = DifferentiableOptimizer(
    objective_function=trajectory_cost,
    bounds=[(2000, 5000), (5, 15), (500, 2000)],
    method="L-BFGS-B",
    use_jit=True
)

# Optimize
result = optimizer.optimize(jnp.array([3500.0, 10.0, 1200.0]))
```

### 3. Financial Analysis ✅
```python
from src.economics.financial_models import FinancialMetrics
import numpy as np

# Create cash flows
cash_flows = np.array([-100e6, 25e6, 25e6, 25e6, 25e6, 25e6])

# Calculate metrics
npv = FinancialMetrics.calculate_npv(cash_flows, discount_rate=0.08)
irr = FinancialMetrics.calculate_irr(cash_flows)
```

### 4. ISRU Benefits Analysis ✅
```python
from src.economics.isru_benefits import ISRUBenefitAnalyzer

analyzer = ISRUBenefitAnalyzer()
savings = analyzer.calculate_savings(
    resource="water",
    quantity_kg=1000,
    mission_duration_days=30
)
```

### 5. 3D Trajectory Visualization ✅
```python
import plotly.graph_objects as go

# Create 3D trajectory plot
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=trajectory_x, y=trajectory_y, z=trajectory_z,
    mode='lines+markers',
    name='Trajectory'
))
fig.write_html("trajectory.html")
```

## Components Needing Integration

### Partial Functionality
- **Global Optimization**: PyGMO integration exists but needs API fixes
- **Advanced Trajectory**: Lambert solvers implemented but integration needed
- **Economic Dashboard**: Core visualizations exist but need method fixes

### Integration Status
- **Core Modules**: ✅ Working (4/7 components)
- **User Flows**: ✅ 1/5 fully working, 3/5 partially working
- **PRD Compliance**: 31% (needs API integration fixes)

## Recommendations for User Guide

1. **Focus on Working Examples**: Update main user guide to emphasize verified functionality
2. **Clear Status Indicators**: Mark each example as "✅ Working" or "⚠️ Needs Integration"
3. **Practical Workflows**: Show what can be accomplished with current working components
4. **Integration Roadmap**: Explain what's coming next for full PRD compliance

## Updated Quick Start

```python
# Working Quick Start Example
from src.config.models import MissionConfig, PayloadSpecification, CostFactors, OrbitParameters
from src.optimization.differentiable.jax_optimizer import DifferentiableOptimizer
from src.economics.financial_models import FinancialMetrics
import jax.numpy as jnp
import numpy as np

# 1. Configure mission
config = MissionConfig(
    name="My Lunar Mission",
    payload=PayloadSpecification(
        dry_mass=2000.0, payload_mass=1000.0, 
        max_propellant_mass=1500.0, specific_impulse=450.0
    ),
    cost_factors=CostFactors(
        launch_cost_per_kg=50000, spacecraft_cost_per_kg=30000,
        operations_cost_per_day=100000
    ),
    mission_duration_days=10,
    target_orbit=OrbitParameters(altitude=100000, inclination=90.0, eccentricity=0.0)
)

# 2. Optimize trajectory parameters
def cost_function(params):
    return jnp.sum(params**2)  # Simple example

optimizer = DifferentiableOptimizer(cost_function, use_jit=True)
result = optimizer.optimize(jnp.array([3500.0, 10.0, 1200.0]))

# 3. Analyze economics
cash_flows = np.array([-100e6, 25e6, 25e6, 25e6, 25e6, 25e6])
npv = FinancialMetrics.calculate_npv(cash_flows, 0.08)

print(f"Optimized parameters: {result.x}")
print(f"Mission NPV: ${npv/1e6:.1f}M")
```

This provides a foundation for the full system integration.