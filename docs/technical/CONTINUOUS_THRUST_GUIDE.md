# Continuous-Thrust Propagator Guide

## Overview

The Lunar Horizon Optimizer includes a comprehensive continuous-thrust propagator using JAX/Diffrax for optimal trajectory design. This implementation provides differentiable trajectory computation for electric propulsion, ion thruster missions, and powered descent lunar landing scenarios.

## Core Implementation

### Edelbaum Planar Model

The propagator uses an Edelbaum-like planar model with the following formulation:

**State Vector**: `[r, θ, v, m]`
- `r`: Radius from central body [m]
- `θ`: Angular position [rad]  
- `v`: Velocity magnitude [m/s]
- `m`: Spacecraft mass [kg]

**Control**: `α` - Thrust angle relative to velocity vector [rad]

**Dynamics**: 
```python
ṙ = v sin(α)
θ̇ = v cos(α) / r  
v̇ = -μ/r² sin(α) + T/m
ṁ = -T/(Isp × g₀)
```

### Basic Usage

```python
from src.trajectory.continuous_thrust import low_thrust_transfer

# Initial state: [radius, angle, velocity, mass]
start_state = jnp.array([6.778e6, 0.0, 7.7e3, 1000.0])  # Earth orbit
target_state = jnp.array([1.937e6 + 100e3, 0.0, 1.6e3, 0.0])  # Lunar orbit

# Thruster parameters
T = 1000.0    # Thrust [N]
Isp = 3000.0  # Specific impulse [s]
tf = 15 * 24 * 3600  # 15-day transfer [s]
alpha = 0.1   # Thrust angle [rad]

# Compute transfer
delta_v, trajectory = low_thrust_transfer(
    start_state, target_state, T, Isp, tf, alpha
)

print(f"Delta-v equivalent: {delta_v:.0f} m/s")
print(f"Final radius: {trajectory[-1, 0]/1e6:.3f} Mm")
```

## Powered Descent Modeling

### 3D Lunar Landing Trajectories

The propagator includes specialized powered descent functionality for lunar landing:

```python
import jax.numpy as jnp
from src.trajectory.continuous_thrust import powered_descent

# Initial state in Moon-centered inertial frame: [x, y, z, vx, vy, vz, m]
moon_radius = 1.737e6  # m
orbit_radius = moon_radius + 100e3  # 100 km altitude
orbital_velocity = jnp.sqrt(4.9048695e12 / orbit_radius)  # ~1634 m/s

start_state = jnp.array([
    orbit_radius, 0.0, 0.0,        # Position [m]
    0.0, orbital_velocity, 0.0,    # Velocity [m/s]
    5000.0                         # Mass [kg]
])

# Spacecraft parameters
thrust = 15000.0  # N (15 kN thrust)
isp = 300.0       # s (chemical propulsion)
burn_time = 300.0 # s (5 minutes)
steps = 100       # Integration steps

# Run powered descent
states, times, total_delta_v = powered_descent(
    start_state, thrust, isp, burn_time, steps
)

# Extract landing results
final_altitude = jnp.linalg.norm(states[-1, :3]) - moon_radius
final_speed = jnp.linalg.norm(states[-1, 3:6])
fuel_consumed = start_state[6] - states[-1, 6]

print(f"Landing altitude: {final_altitude/1000:.1f} km")
print(f"Landing speed: {final_speed:.1f} m/s")
print(f"Fuel consumed: {fuel_consumed:.0f} kg")
print(f"Total delta-v: {total_delta_v:.0f} m/s")
```

### JIT Compilation for Performance

```python
from src.trajectory.continuous_thrust import create_jitted_powered_descent

# Create JIT-compiled version for repeated use
jitted_descent = create_jitted_powered_descent(steps=100)

# Fast repeated calls
for thrust_level in [10000, 15000, 20000]:  # Test different thrust levels
    states, times, delta_v = jitted_descent(
        start_state, thrust_level, 300.0, 300.0
    )
    print(f"Thrust {thrust_level/1000:.0f}kN: Δv = {delta_v:.0f} m/s")
```

### Optimization Integration

```python
from src.optimization.differentiable.continuous_thrust_integration import (
    optimize_continuous_thrust_transfer
)

# Optimize Earth-Moon transfer
results = optimize_continuous_thrust_transfer(
    T=500.0,      # 500N ion thruster
    Isp=3500.0,   # High specific impulse
    target_radius=1.937e6 + 100e3  # 100km lunar orbit
)

if results['success']:
    params = results['optimal_parameters']
    print(f"Optimal thrust angle: {params['thrust_angle']:.3f} rad")
    print(f"Transfer time: {params['transfer_time']/24/3600:.1f} days")
```

## Integration with Differentiable Optimization

### JAX Compatibility

The continuous-thrust propagator is fully JAX-compatible for gradient-based optimization:

```python
import jax
from src.trajectory.continuous_thrust import optimize_thrust_angle

# Gradient computation
gradient_fn = jax.grad(optimize_thrust_angle)
start_state = jnp.array([6.778e6, 0.0, 7.7e3, 1000.0])
grad = gradient_fn(start_state, 1.5e7, 1000.0, 3000.0, 20*24*3600)
```

### Multi-Objective Optimization

The framework supports multi-objective optimization combining:
- **Delta-v minimization**: Propellant efficiency
- **Transfer time**: Mission duration
- **Economic cost**: Launch and operations costs
- **Target accuracy**: Orbital insertion precision

```python
from src.optimization.differentiable.continuous_thrust_integration import (
    ContinuousThrustLoss
)

# Create multi-objective loss function
loss_function = ContinuousThrustLoss(
    trajectory_model=ContinuousThrustModel(T=800.0, Isp=3200.0),
    economic_model=EconomicModel(),
    target_radius=1.937e6 + 100e3
)

# Optimization weights: 30% delta-v, 20% time, 40% cost, 10% accuracy
```

## Performance Characteristics

### Computational Efficiency

- **Integration**: Diffrax Tsit5 adaptive solver (~1ms per trajectory)
- **Gradient computation**: JAX automatic differentiation
- **Optimization**: L-BFGS-B convergence in ~50 iterations

### Typical Results

| Mission | Thrust [N] | Isp [s] | Transfer Time | Delta-v [m/s] |
|---------|------------|---------|---------------|---------------|
| Ion thruster | 500 | 3500 | 20-30 days | 6,000-8,000 |
| Hall thruster | 1000 | 2500 | 15-25 days | 7,000-9,000 |
| Chemical (reference) | 50000 | 450 | 3-5 days | 10,000-12,000 |

## Accuracy Caveats and Limitations

### Model Limitations

• **Planar approximation** - No inclination changes or 3D orbital mechanics
• **Constant thrust magnitude** - Assumes fixed thruster performance  
• **No third-body perturbations** - Moon/Sun gravity effects neglected
• **Circular target orbit assumption** - No elliptical target orbits
• **No spacecraft attitude dynamics** - Thrust always in orbital plane

### Numerical Considerations

• **Stiff dynamics** when thrust >> gravitational acceleration can cause instability
• **Integration tolerance** affects solution accuracy vs computational speed trade-off
• **Event detection precision** for orbital insertions may require refinement
• **Gradient computation stability** depends on parameter conditioning for optimization

### Recommended Use Cases

**Suitable For**:
- Preliminary mission design and trade studies
- Electric propulsion trajectory optimization
- Multi-objective optimization studies
- Educational and research applications

**Not Suitable For**:
- High-fidelity mission operations planning
- Complex multi-body dynamics (e.g., Lagrange points)
- Attitude-constrained maneuvers
- Real-time trajectory control

## Advanced Usage

### Custom Thrust Profiles

```python
def time_varying_thrust_angle(t, params):
    """Time-varying thrust angle profile."""
    alpha0, alpha1, period = params
    return alpha0 + alpha1 * jnp.sin(2 * jnp.pi * t / period)

# Modify dynamics function for time-varying control
```

### Multi-Phase Transfers

```python
def multi_phase_transfer(waypoints, thrust_schedule):
    """Multi-phase transfer through waypoints."""
    total_delta_v = 0.0
    current_state = waypoints[0]
    
    for i in range(len(waypoints) - 1):
        phase_dv, trajectory = low_thrust_transfer(
            current_state, waypoints[i+1], 
            thrust_schedule[i]['T'], 
            thrust_schedule[i]['Isp'],
            thrust_schedule[i]['tf'],
            thrust_schedule[i]['alpha']
        )
        total_delta_v += phase_dv
        current_state = trajectory[-1]
    
    return total_delta_v
```

### Constraint Handling

```python
def constrained_optimization():
    """Add trajectory constraints to optimization."""
    
    def trajectory_constraints(params):
        """Custom constraint functions."""
        _, trajectory = low_thrust_transfer(start_state, None, T, Isp, params[3], params[2])
        
        # Maximum distance constraint (avoid Van Allen belts)
        max_radius = jnp.max(trajectory[:, 0])
        distance_constraint = max_radius - 50e6  # 50 Mm limit
        
        # Minimum velocity constraint
        min_velocity = jnp.min(trajectory[:, 2])
        velocity_constraint = 1e3 - min_velocity  # 1 km/s minimum
        
        return jnp.array([distance_constraint, velocity_constraint])
    
    # Add to optimizer constraints
    return trajectory_constraints
```

## Examples and Demonstrations

### Basic Transfer Example

```python
from src.optimization.differentiable.continuous_thrust_integration import (
    demonstrate_continuous_thrust_optimization
)

# Run demonstration
results = demonstrate_continuous_thrust_optimization()
```

### Comparison with Chemical Propulsion

```python
def compare_propulsion_systems():
    """Compare electric vs chemical propulsion."""
    
    # Electric propulsion
    electric_results = optimize_continuous_thrust_transfer(
        T=500.0, Isp=3500.0
    )
    
    # Chemical propulsion (for reference)
    chemical_dv = 10500.0  # Typical chemical delta-v
    chemical_time = 3.5 * 24 * 3600  # 3.5 days
    
    if electric_results['success']:
        electric_dv = electric_results['trajectory_results']['delta_v']
        electric_time = electric_results['optimal_parameters']['transfer_time']
        
        print(f"Electric:  {electric_dv:.0f} m/s in {electric_time/24/3600:.1f} days")
        print(f"Chemical:  {chemical_dv:.0f} m/s in {chemical_time/24/3600:.1f} days")
        print(f"Delta-v savings: {(chemical_dv-electric_dv)/chemical_dv*100:.1f}%")
```

## Future Enhancements

### Planned Improvements

1. **3D orbital mechanics** - Full 3D state propagation
2. **Variable thrust magnitude** - Throttleable engines
3. **Multi-body dynamics** - Moon/Sun perturbations
4. **Attitude coupling** - Thrust vector control
5. **Advanced control laws** - Optimal steering profiles

### Integration Opportunities

- **Multi-mission optimization** - Electric propulsion constellations  
- **Hybrid transfers** - Chemical + electric phases
- **Trajectory optimization** - Integration with existing PyKEP solvers
- **Economic analysis** - Electric vs chemical trade studies

The continuous-thrust propagator provides a solid foundation for low-thrust mission optimization while maintaining compatibility with the existing Lunar Horizon Optimizer architecture.