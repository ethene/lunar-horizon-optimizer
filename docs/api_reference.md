# API Reference - Lunar Horizon Optimizer

## Overview

This API reference provides comprehensive documentation for the Lunar Horizon Optimizer modules, including usage examples, parameter descriptions, and integration guidelines.

**Last Updated**: July 2025  
**Status**: Tasks 3, 4, 5, 6 Complete & Tested  
**Environment**: conda py312 with PyKEP, PyGMO, Plotly

## Task 3: Enhanced Trajectory Generation

### Core Classes

#### `LambertSolver`
Solves Lambert problems for two-body trajectory transfers.

```python
from trajectory.earth_moon_trajectories import LambertSolver

solver = LambertSolver(central_body_mu=3.986004418e14)
```

**Methods**:

##### `solve_lambert(r1, r2, time_of_flight, direction=0, max_revolutions=0)`
Solve Lambert problem for given position vectors and time.

**Parameters**:
- `r1` (np.ndarray): Initial position vector [m]
- `r2` (np.ndarray): Final position vector [m]
- `time_of_flight` (float): Time of flight [s]
- `direction` (int): Transfer direction (0=auto, 1=prograde, -1=retrograde)
- `max_revolutions` (int): Maximum number of revolutions

**Returns**:
- `Tuple[np.ndarray, np.ndarray]`: (initial_velocity, final_velocity) [m/s]

**Example**:
```python
import numpy as np

r1 = np.array([7000000, 0, 0])      # 7000 km altitude
r2 = np.array([0, 42000000, 0])     # Moon vicinity
tof = 4.5 * 86400                   # 4.5 days

v1, v2 = solver.solve_lambert(r1, r2, tof)
print(f"Initial velocity: {np.linalg.norm(v1):.0f} m/s")
```

#### `TrajectoryWindowAnalyzer`
Analyzes Earth-Moon transfer windows for optimal launch opportunities.

```python
from trajectory.transfer_window_analysis import TrajectoryWindowAnalyzer

analyzer = TrajectoryWindowAnalyzer(
    min_earth_alt=200,   # km
    max_earth_alt=1000,  # km
    min_moon_alt=50,     # km
    max_moon_alt=500     # km
)
```

##### `find_transfer_windows(start_date, end_date, **kwargs)`
Find optimal transfer windows in a given time period.

**Parameters**:
- `start_date` (datetime): Start of analysis period
- `end_date` (datetime): End of analysis period
- `earth_orbit_alt` (float): Earth parking orbit altitude [km]
- `moon_orbit_alt` (float): Target lunar orbit altitude [km]
- `min_transfer_time` (float): Minimum transfer time [days]
- `max_transfer_time` (float): Maximum transfer time [days]
- `time_step` (float): Time step for analysis [days]

**Returns**:
- `List[TransferWindow]`: List of viable transfer windows sorted by delta-v

**Example**:
```python
from datetime import datetime

start_date = datetime(2025, 6, 1)
end_date = datetime(2025, 7, 1)

windows = analyzer.find_transfer_windows(
    start_date=start_date,
    end_date=end_date,
    earth_orbit_alt=400.0,
    moon_orbit_alt=100.0,
    time_step=1.0
)

print(f"Found {len(windows)} transfer windows")
for window in windows[:3]:
    print(f"  {window.departure_date}: {window.total_dv:.0f} m/s")
```

#### `EarthMoonNBodyPropagator`
Complete Earth-Moon n-body propagator with solar effects.

```python
from trajectory.nbody_integration import EarthMoonNBodyPropagator

propagator = EarthMoonNBodyPropagator(
    include_sun=True,
    include_perturbations=False,
    integrator_method='DOP853'
)
```

##### `propagate_spacecraft(initial_position, initial_velocity, reference_epoch, propagation_time, num_points=1000)`
Propagate spacecraft trajectory in Earth-Moon system.

**Parameters**:
- `initial_position` (np.ndarray): Initial position in Earth-centered frame [m]
- `initial_velocity` (np.ndarray): Initial velocity in Earth-centered frame [m/s]
- `reference_epoch` (float): Reference epoch [days since J2000]
- `propagation_time` (float): Propagation time [s]
- `num_points` (int): Number of output points

**Returns**:
- `Dict[str, np.ndarray]`: Propagation results with positions, velocities, energies

**Example**:
```python
import numpy as np

# 400 km circular orbit
initial_position = np.array([6778000.0, 0.0, 0.0])  # m
initial_velocity = np.array([0.0, 7669.0, 0.0])     # m/s

result = propagator.propagate_spacecraft(
    initial_position=initial_position,
    initial_velocity=initial_velocity,
    reference_epoch=10000.0,
    propagation_time=86400.0,  # 1 day
    num_points=1440
)

print(f"Final position: {result['positions'][:, -1] / 1000:.1f} km")
print(f"Energy conservation: {np.std(result['total_energy']):.2e}")
```

### Convenience Functions

#### `generate_earth_moon_trajectory(departure_epoch, earth_orbit_alt, moon_orbit_alt, transfer_time, method='lambert')`
Generate Earth-Moon trajectory using specified method.

**Parameters**:
- `departure_epoch` (float): Departure epoch [days since J2000]
- `earth_orbit_alt` (float): Earth orbit altitude [km]
- `moon_orbit_alt` (float): Moon orbit altitude [km]
- `transfer_time` (float): Transfer time [days]
- `method` (str): Generation method ('lambert', 'patched_conics')

**Returns**:
- `Tuple[Trajectory, float]`: (trajectory, total_deltav)

**Example**:
```python
from trajectory.earth_moon_trajectories import generate_earth_moon_trajectory

trajectory, total_dv = generate_earth_moon_trajectory(
    departure_epoch=10000.0,
    earth_orbit_alt=400.0,
    moon_orbit_alt=100.0,
    transfer_time=4.5,
    method='lambert'
)

print(f"Total delta-v: {total_dv:.0f} m/s")
print(f"Transfer time: {trajectory.arrival_epoch - trajectory.departure_epoch:.1f} days")
```

## Task 4: Global Optimization Module

### Core Classes

#### `LunarMissionProblem`
PyGMO problem implementation for lunar mission optimization.

```python
from optimization.global_optimizer import LunarMissionProblem
from config.costs import CostFactors

cost_factors = CostFactors(
    launch_cost_per_kg=10000.0,
    operations_cost_per_day=100000.0,
    development_cost=1e9,
    contingency_percentage=20.0
)

problem = LunarMissionProblem(
    cost_factors=cost_factors,
    min_earth_alt=200,
    max_earth_alt=1000,
    min_moon_alt=50,
    max_moon_alt=500,
    min_transfer_time=3.0,
    max_transfer_time=10.0,
    reference_epoch=10000.0
)
```

##### `fitness(x)`
Evaluate fitness for multi-objective optimization.

**Parameters**:
- `x` (List[float]): Decision vector [earth_alt, moon_alt, transfer_time]

**Returns**:
- `List[float]`: Objective values [delta_v, time, cost]

##### `get_bounds()`
Get optimization bounds for decision variables.

**Returns**:
- `Tuple[List[float], List[float]]`: (lower_bounds, upper_bounds)

#### `GlobalOptimizer`
PyGMO-based global optimizer using NSGA-II algorithm.

```python
from optimization.global_optimizer import GlobalOptimizer

optimizer = GlobalOptimizer(
    problem=problem,
    population_size=100,
    num_generations=100,
    seed=42
)
```

##### `optimize(verbose=True)`
Run multi-objective optimization.

**Parameters**:
- `verbose` (bool): Enable detailed logging

**Returns**:
- `Dict[str, Any]`: Optimization results with Pareto front and statistics

**Example**:
```python
results = optimizer.optimize(verbose=True)

print(f"Found {len(results['pareto_front'])} Pareto solutions")
print(f"Cache efficiency: {results['cache_stats']['hit_rate']:.1%}")

# Get best solutions
best_solutions = optimizer.get_best_solutions(
    num_solutions=5,
    preference_weights=[0.4, 0.3, 0.3]  # Prefer delta-v
)

for i, solution in enumerate(best_solutions, 1):
    params = solution['parameters']
    objectives = solution['objectives']
    print(f"Solution {i}: ΔV={objectives['delta_v']:.0f} m/s, "
          f"Cost=${objectives['cost']/1e6:.1f}M")
```

#### `ParetoAnalyzer`
Analysis tools for Pareto fronts and solution ranking.

```python
from optimization.pareto_analysis import ParetoAnalyzer

analyzer = ParetoAnalyzer()
```

##### `analyze_pareto_front(optimization_result)`
Analyze optimization results and create structured result object.

**Parameters**:
- `optimization_result` (Dict[str, Any]): Raw optimization results

**Returns**:
- `OptimizationResult`: Structured optimization result with analysis

##### `rank_solutions_by_preference(solutions, preference_weights, normalization_method='minmax')`
Rank solutions by user preferences using weighted objectives.

**Parameters**:
- `solutions` (List[Dict]): List of Pareto solutions
- `preference_weights` (List[float]): Weights for [delta_v, time, cost] objectives
- `normalization_method` (str): Normalization method ('minmax', 'zscore')

**Returns**:
- `List[Tuple[float, Dict]]`: List of (score, solution) tuples sorted by preference

### Convenience Functions

#### `optimize_lunar_mission(cost_factors=None, optimization_config=None)`
Convenience function for complete lunar mission optimization.

**Parameters**:
- `cost_factors` (CostFactors): Economic cost parameters
- `optimization_config` (Dict): Configuration for optimization parameters

**Returns**:
- `Dict[str, Any]`: Complete optimization results

**Example**:
```python
from optimization.global_optimizer import optimize_lunar_mission

config = {
    'problem_params': {
        'min_earth_alt': 200,
        'max_earth_alt': 800,
    },
    'optimizer_params': {
        'population_size': 150,
        'num_generations': 150
    },
    'verbose': True
}

results = optimize_lunar_mission(
    cost_factors=cost_factors,
    optimization_config=config
)
```

## Task 5: Basic Economic Analysis Module

### Core Classes

#### `CashFlowModel`
Cash flow modeling for lunar mission economics.

```python
from economics.financial_models import CashFlowModel, FinancialParameters
from datetime import datetime, timedelta

params = FinancialParameters(
    discount_rate=0.08,
    inflation_rate=0.03,
    tax_rate=0.25,
    project_duration_years=10
)

cash_model = CashFlowModel(params)
```

##### `add_development_costs(total_cost, start_date, duration_months)`
Add development costs spread over development period.

**Parameters**:
- `total_cost` (float): Total development cost
- `start_date` (datetime): Development start date
- `duration_months` (int): Development duration in months

##### `add_launch_costs(cost_per_launch, launch_dates)`
Add launch costs for multiple launches.

**Parameters**:
- `cost_per_launch` (float): Cost per launch
- `launch_dates` (List[datetime]): List of launch dates

**Example**:
```python
start_date = datetime(2025, 1, 1)

# Add mission cash flows
cash_model.add_development_costs(100e6, start_date, 24)
cash_model.add_launch_costs(50e6, [start_date + timedelta(days=730)])
cash_model.add_operational_costs(5e6, start_date + timedelta(days=730), 36)
cash_model.add_revenue_stream(8e6, start_date + timedelta(days=760), 36)

print(f"Total cash flows: {len(cash_model.cash_flows)}")
```

#### `NPVAnalyzer`
Net Present Value analysis for lunar missions.

```python
from economics.financial_models import NPVAnalyzer

npv_analyzer = NPVAnalyzer(params)
```

##### `calculate_npv(cash_flow_model, reference_date=None)`
Calculate Net Present Value of cash flows.

**Parameters**:
- `cash_flow_model` (CashFlowModel): Cash flow model containing all cash flows
- `reference_date` (datetime): Reference date for NPV calculation

**Returns**:
- `float`: Net Present Value

##### `calculate_irr(cash_flow_model, reference_date=None)`
Calculate Internal Rate of Return.

**Parameters**:
- `cash_flow_model` (CashFlowModel): Cash flow model
- `reference_date` (datetime): Reference date for IRR calculation

**Returns**:
- `float`: Internal Rate of Return (as decimal)

**Example**:
```python
npv = npv_analyzer.calculate_npv(cash_model)
irr = npv_analyzer.calculate_irr(cash_model)
payback = npv_analyzer.calculate_payback_period(cash_model)

print(f"NPV: ${npv/1e6:.1f}M")
print(f"IRR: {irr:.1%}")
print(f"Payback: {payback:.1f} years")
```

#### `MissionCostModel`
Comprehensive mission cost model with parametric scaling.

```python
from economics.cost_models import MissionCostModel

cost_model = MissionCostModel()
```

##### `estimate_total_mission_cost(spacecraft_mass, mission_duration_years, technology_readiness=3, complexity='moderate', schedule='nominal')`
Estimate total mission cost with detailed breakdown.

**Parameters**:
- `spacecraft_mass` (float): Spacecraft mass [kg]
- `mission_duration_years` (float): Mission duration [years]
- `technology_readiness` (int): Technology readiness level (1-4 scale)
- `complexity` (str): Mission complexity ('simple', 'moderate', 'complex', 'flagship')
- `schedule` (str): Schedule pressure ('relaxed', 'nominal', 'aggressive', 'crash')

**Returns**:
- `CostBreakdown`: Detailed cost breakdown

**Example**:
```python
cost_breakdown = cost_model.estimate_total_mission_cost(
    spacecraft_mass=5000,
    mission_duration_years=5,
    technology_readiness=3,
    complexity='moderate',
    schedule='nominal'
)

print(f"Total cost: ${cost_breakdown.total:.1f}M")
print(f"Development: ${cost_breakdown.development:.1f}M")
print(f"Launch: ${cost_breakdown.launch:.1f}M")
print(f"Operations: ${cost_breakdown.operations:.1f}M")
```

#### `ISRUBenefitAnalyzer`
Comprehensive ISRU benefit analysis for lunar missions.

```python
from economics.isru_benefits import ISRUBenefitAnalyzer

analyzer = ISRUBenefitAnalyzer()
```

##### `analyze_isru_economics(resource_name, facility_scale='commercial', operation_duration_months=60, discount_rate=0.08)`
Perform comprehensive ISRU economic analysis.

**Parameters**:
- `resource_name` (str): Primary resource to extract
- `facility_scale` (str): Scale of ISRU facility
- `operation_duration_months` (int): Duration of operations
- `discount_rate` (float): Discount rate for NPV calculation

**Returns**:
- `Dict[str, Any]`: Complete ISRU economic analysis

**Example**:
```python
analysis = analyzer.analyze_isru_economics(
    resource_name='water_ice',
    facility_scale='commercial',
    operation_duration_months=60
)

print(f"ISRU NPV: ${analysis['financial_metrics']['npv']/1e6:.1f}M")
print(f"ISRU ROI: {analysis['financial_metrics']['roi']:.1%}")
print(f"Break-even: {analysis['break_even_analysis']['payback_period_months']:.1f} months")
```

#### `EconomicSensitivityAnalyzer`
Economic sensitivity and scenario analysis.

```python
from economics.sensitivity_analysis import EconomicSensitivityAnalyzer

def economic_model(params):
    # Your economic model implementation
    return {'npv': calculated_npv}

analyzer = EconomicSensitivityAnalyzer(economic_model)
```

##### `monte_carlo_simulation(base_parameters, variable_distributions, num_simulations=10000, confidence_levels=None)`
Perform Monte Carlo simulation for risk analysis.

**Parameters**:
- `base_parameters` (Dict[str, float]): Base case parameters
- `variable_distributions` (Dict): Parameter distributions
- `num_simulations` (int): Number of Monte Carlo simulations
- `confidence_levels` (List[float]): Confidence levels for analysis

**Returns**:
- `Dict[str, Any]`: Monte Carlo simulation results

**Example**:
```python
base_params = {'cost_multiplier': 1.0, 'revenue_multiplier': 1.0}

distributions = {
    'cost_multiplier': {'type': 'triangular', 'min': 0.8, 'mode': 1.0, 'max': 1.5},
    'revenue_multiplier': {'type': 'normal', 'mean': 1.0, 'std': 0.2}
}

mc_results = analyzer.monte_carlo_simulation(
    base_params, distributions, 10000
)

print(f"Mean NPV: ${mc_results['statistics']['mean']/1e6:.1f}M")
print(f"P(NPV > 0): {mc_results['risk_metrics']['probability_positive_npv']:.1%}")
print(f"VaR (5%): ${mc_results['risk_metrics']['value_at_risk_5%']/1e6:.1f}M")
```

#### `EconomicReporter`
Professional economic reporting and data export.

```python
from economics.reporting import EconomicReporter, FinancialSummary

reporter = EconomicReporter('reports')
```

##### `generate_executive_summary(financial_summary, analysis_results=None)`
Generate executive summary report.

**Parameters**:
- `financial_summary` (FinancialSummary): Financial summary data
- `analysis_results` (Dict): Additional analysis results

**Returns**:
- `str`: Executive summary as formatted string

**Example**:
```python
summary = FinancialSummary(
    total_investment=200e6,
    total_revenue=350e6,
    net_present_value=75e6,
    internal_rate_of_return=0.18,
    return_on_investment=0.25,
    payback_period_years=6.5,
    mission_duration_years=8,
    probability_of_success=0.75
)

exec_summary = reporter.generate_executive_summary(summary)
print(exec_summary)

# Export data
json_path = reporter.export_to_json(summary, 'mission_summary')
csv_path = reporter.export_to_csv(summary, 'mission_summary')
```

## Data Structures

### `TransferWindow`
Represents a transfer window opportunity.

**Attributes**:
- `departure_date` (datetime): Departure date
- `arrival_date` (datetime): Arrival date
- `total_dv` (float): Total delta-v [m/s]
- `c3_energy` (float): Characteristic energy [m²/s²]
- `trajectory` (Trajectory): Trajectory object
- `transfer_time` (float): Transfer time [days]

### `CostBreakdown`
Detailed cost breakdown for mission components.

**Attributes**:
- `development` (float): Development cost
- `launch` (float): Launch cost
- `spacecraft` (float): Spacecraft cost
- `operations` (float): Operations cost
- `ground_systems` (float): Ground systems cost
- `contingency` (float): Contingency cost
- `total` (float): Total cost

### `FinancialSummary`
Financial summary data structure for lunar mission economics.

**Attributes**:
- `total_investment` (float): Total investment required
- `total_revenue` (float): Total expected revenue
- `net_present_value` (float): Net Present Value
- `internal_rate_of_return` (float): Internal Rate of Return
- `return_on_investment` (float): Return on Investment
- `payback_period_years` (float): Payback period in years
- `probability_of_success` (float): Probability of mission success

## Error Handling

### Common Exceptions

#### `TrajectoryGenerationError`
Raised when trajectory generation fails.

#### `OptimizationError`
Raised when optimization fails to converge or encounters errors.

#### `EconomicAnalysisError`
Raised when economic analysis encounters invalid parameters or calculation errors.

### Example Error Handling

```python
from trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
from optimization.global_optimizer import OptimizationError
from economics.financial_models import EconomicAnalysisError

try:
    trajectory, dv = generate_earth_moon_trajectory(
        departure_epoch=10000.0,
        earth_orbit_alt=400.0,
        moon_orbit_alt=100.0,
        transfer_time=4.5
    )
except Exception as e:
    print(f"Trajectory generation failed: {e}")

try:
    results = optimizer.optimize()
except OptimizationError as e:
    print(f"Optimization failed: {e}")

try:
    npv = npv_analyzer.calculate_npv(cash_model)
except EconomicAnalysisError as e:
    print(f"Economic analysis failed: {e}")
```

## Performance Tips

### Optimization Performance
1. **Use Caching**: Enable trajectory caching for repeated evaluations
2. **Parallel Processing**: Utilize PyGMO's parallel capabilities
3. **Population Size**: Balance between solution quality and computation time
4. **Convergence Criteria**: Monitor convergence to avoid unnecessary generations

### Memory Management
1. **Large Datasets**: Use numpy arrays efficiently
2. **Propagation**: Limit number of points for long propagations
3. **Monte Carlo**: Consider batch processing for very large simulations

### Example Performance Optimization

```python
# Optimized optimization setup
problem = LunarMissionProblem(
    cost_factors=cost_factors,
    # Enable caching for better performance
    cache_size_limit=10000
)

optimizer = GlobalOptimizer(
    problem=problem,
    population_size=100,    # Balanced size
    num_generations=50,     # Monitor convergence
    seed=42
)

# Monitor progress
results = optimizer.optimize(verbose=True)

# Check cache performance
cache_stats = problem.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1%}")
```

## Task 6: Visualization Module

### Core Classes

#### `TrajectoryVisualizer`
Interactive 3D trajectory visualization using Plotly.

```python
from visualization.trajectory_visualization import TrajectoryVisualizer, TrajectoryPlotConfig

config = TrajectoryPlotConfig(width=1200, height=800, title="Lunar Transfer")
viz = TrajectoryVisualizer(config)
```

**Methods**:

##### `create_3d_trajectory_plot(trajectory_data, **kwargs)`
Create interactive 3D trajectory visualization.

**Parameters**:
- `trajectory_data` (Dict): Trajectory data with positions, velocities, times
- `**kwargs`: Additional plot customization options

**Returns**:
- `plotly.graph_objects.Figure`: Interactive 3D plot

**Example**:
```python
trajectory_data = {
    'positions': positions,  # 3×N array [m]
    'velocities': velocities, # 3×N array [m/s]
    'times': times           # N-element array [s]
}

fig = viz.create_3d_trajectory_plot(trajectory_data)
fig.show()
```

##### `create_transfer_window_plot(start_date, end_date, **kwargs)`
Visualize transfer window analysis results.

**Parameters**:
- `start_date` (datetime): Analysis start date
- `end_date` (datetime): Analysis end date
- `**kwargs`: Additional plot options

**Returns**:
- `plotly.graph_objects.Figure`: Transfer window porkchop plot

#### `OptimizationVisualizer`
Pareto front and optimization results visualization.

```python
from visualization.optimization_visualization import OptimizationVisualizer

viz = OptimizationVisualizer()
```

**Methods**:

##### `create_pareto_front_plot(optimization_result, objective_names=None, **kwargs)`
Visualize Pareto front from optimization results.

**Parameters**:
- `optimization_result` (OptimizationResult): Optimization results
- `objective_names` (List[str]): Names for objectives
- `**kwargs`: Plot customization options

**Returns**:
- `plotly.graph_objects.Figure`: Interactive Pareto front plot

**Example**:
```python
objective_names = ["Delta-V (m/s)", "Transfer Time (days)", "Cost ($)"]
fig = viz.create_pareto_front_plot(
    optimization_result,
    objective_names=objective_names,
    show_dominated=False
)
fig.show()
```

##### `create_solution_comparison_plot(solutions, solution_labels, **kwargs)`
Compare multiple optimization solutions.

**Parameters**:
- `solutions` (List[Dict]): List of solution dictionaries
- `solution_labels` (List[str]): Labels for solutions
- `**kwargs`: Plot customization options

**Returns**:
- `plotly.graph_objects.Figure`: Solution comparison chart

#### `EconomicVisualizer`
Economic analysis dashboards and financial visualization.

```python
from visualization.economic_visualization import EconomicVisualizer, DashboardConfig

config = DashboardConfig(width=1400, height=1000)
viz = EconomicVisualizer(config)
```

**Methods**:

##### `create_financial_dashboard(financial_summary, **kwargs)`
Create comprehensive financial analysis dashboard.

**Parameters**:
- `financial_summary` (FinancialSummary): Financial analysis results
- `cash_flow_model` (CashFlowModel, optional): Detailed cash flow data
- `cost_breakdown` (CostBreakdown, optional): Cost breakdown data

**Returns**:
- `plotly.graph_objects.Figure`: Multi-panel financial dashboard

**Example**:
```python
fig = viz.create_financial_dashboard(
    financial_summary=financial_results,
    cost_breakdown=cost_analysis
)
fig.show()
```

##### `create_sensitivity_analysis_dashboard(sensitivity_results, **kwargs)`
Visualize sensitivity analysis and risk assessment.

**Parameters**:
- `sensitivity_results` (Dict): Sensitivity analysis results
- `monte_carlo_results` (Dict, optional): Monte Carlo simulation results

**Returns**:
- `plotly.graph_objects.Figure`: Sensitivity analysis dashboard

#### `MissionVisualizer`
Mission timeline and milestone visualization.

```python
from visualization.mission_visualization import MissionVisualizer, MissionPhase, MissionMilestone

viz = MissionVisualizer()
```

**Methods**:

##### `create_mission_timeline(phases, milestones=None, **kwargs)`
Create Gantt-style mission timeline.

**Parameters**:
- `phases` (List[MissionPhase]): List of mission phases
- `milestones` (List[MissionMilestone], optional): Mission milestones
- `**kwargs`: Timeline customization options

**Returns**:
- `plotly.graph_objects.Figure`: Interactive mission timeline

**Example**:
```python
from datetime import datetime, timedelta

base_date = datetime(2025, 1, 1)
phases = [
    MissionPhase(
        name="Development",
        start_date=base_date,
        end_date=base_date + timedelta(days=730),
        category="Development",
        risk_level="Medium"
    )
]

fig = viz.create_mission_timeline(phases)
fig.show()
```

#### `ComprehensiveDashboard`
Integrated dashboard combining all analysis modules.

```python
from visualization.dashboard import ComprehensiveDashboard, MissionAnalysisData

dashboard = ComprehensiveDashboard()
```

**Methods**:

##### `create_executive_dashboard(mission_data)`
Create executive summary dashboard with key metrics.

**Parameters**:
- `mission_data` (MissionAnalysisData): Complete mission analysis data

**Returns**:
- `plotly.graph_objects.Figure`: Executive summary dashboard

**Example**:
```python
mission_data = MissionAnalysisData(
    mission_name="Artemis Lunar Base",
    trajectory_data=trajectory_results,
    optimization_results=optimization_results,
    financial_summary=financial_summary,
    cost_breakdown=cost_breakdown
)

fig = dashboard.create_executive_dashboard(mission_data)
fig.show()
```

##### `create_technical_dashboard(mission_data)`
Create detailed technical analysis dashboard.

**Parameters**:
- `mission_data` (MissionAnalysisData): Complete mission analysis data

**Returns**:
- `plotly.graph_objects.Figure`: Technical analysis dashboard

### Quick Functions

#### `create_quick_trajectory_plot(earth_orbit_alt, moon_orbit_alt, transfer_time, departure_epoch)`
Quick 3D trajectory plot generation.

**Parameters**:
- `earth_orbit_alt` (float): Earth orbit altitude [km]
- `moon_orbit_alt` (float): Moon orbit altitude [km]
- `transfer_time` (float): Transfer time [days]
- `departure_epoch` (float): Departure epoch [days since J2000]

**Returns**:
- `plotly.graph_objects.Figure`: 3D trajectory plot

#### `create_quick_financial_dashboard(npv, irr, roi, payback_years, total_investment, total_revenue)`
Quick financial dashboard creation.

**Parameters**:
- `npv` (float): Net Present Value [$]
- `irr` (float): Internal Rate of Return [fraction]
- `roi` (float): Return on Investment [fraction]
- `payback_years` (float): Payback period [years]
- `total_investment` (float): Total investment [$]
- `total_revenue` (float): Total revenue [$]

**Returns**:
- `plotly.graph_objects.Figure`: Financial dashboard

### Data Models

#### `TrajectoryPlotConfig`
Configuration for trajectory visualization.

**Attributes**:
- `width` (int): Plot width [pixels]
- `height` (int): Plot height [pixels]
- `title` (str): Plot title
- `show_earth` (bool): Show Earth sphere
- `show_moon` (bool): Show Moon sphere
- `trajectory_color` (str): Trajectory line color
- `enable_animation` (bool): Enable animation controls

#### `MissionPhase`
Mission phase data model.

**Attributes**:
- `name` (str): Phase name
- `start_date` (datetime): Phase start date
- `end_date` (datetime): Phase end date
- `category` (str): Phase category
- `cost` (float, optional): Phase cost [$]
- `risk_level` (str, optional): Risk level ("Low", "Medium", "High")
- `dependencies` (List[str], optional): Dependent phases

#### `MissionMilestone`
Mission milestone data model.

**Attributes**:
- `name` (str): Milestone name
- `date` (datetime): Milestone date
- `category` (str): Milestone category
- `description` (str, optional): Milestone description
- `importance` (str, optional): Importance level

### Visualization Best Practices

#### Performance Optimization
1. **Data Size**: Limit trajectory points to ~10,000 for smooth interaction
2. **Dashboard Complexity**: Use subplots judiciously to maintain responsiveness
3. **Memory Usage**: Consider data decimation for very large datasets

#### User Experience
1. **Interactivity**: Enable zoom, pan, and selection for exploration
2. **Color Schemes**: Use consistent color schemes across visualizations
3. **Annotations**: Provide clear labels and units for all quantities

#### Integration
1. **Data Flow**: Ensure consistent data formats between modules
2. **Error Handling**: Implement graceful fallbacks for missing data
3. **Export**: Support high-resolution export for presentations

### Example Integrated Workflow

```python
from trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
from optimization.global_optimizer import optimize_lunar_mission
from economics.reporting import generate_financial_summary
from visualization.dashboard import ComprehensiveDashboard, MissionAnalysisData

# Generate trajectory
trajectory_result = generate_earth_moon_trajectory(
    earth_orbit_alt=400.0,
    moon_orbit_alt=100.0,
    transfer_time=4.5,
    departure_epoch=10000.0
)

# Perform optimization
optimization_result = optimize_lunar_mission(
    trajectory_config=trajectory_result.config,
    cost_factors=cost_factors
)

# Economic analysis
financial_summary = generate_financial_summary(
    optimization_result=optimization_result,
    cost_factors=cost_factors
)

# Comprehensive visualization
mission_data = MissionAnalysisData(
    mission_name="Integrated Analysis",
    trajectory_data=trajectory_result,
    optimization_results=optimization_result,
    financial_summary=financial_summary
)

dashboard = ComprehensiveDashboard()
fig = dashboard.create_executive_dashboard(mission_data)
fig.show()
```

---

**Last Updated**: July 2025  
**Version**: 1.0.0-rc1  
**Environment**: conda py312 with PyKEP, PyGMO, Plotly, SciPy