#!/usr/bin/env python3
"""
Working Example for Lunar Horizon Optimizer

This example demonstrates the core functionality that actually works
based on the PRD validation test results.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def example_1_configuration():
    """Working configuration example."""
    print("\n" + "="*50)
    print("Example 1: Mission Configuration")
    print("="*50)
    
    from src.config.models import MissionConfig, PayloadSpecification, CostFactors, OrbitParameters
    
    # Create proper configuration with all required fields
    config = MissionConfig(
        name="Demo Lunar Mission",
        payload=PayloadSpecification(
            type="cargo",
            mass=1000.0,
            volume=10.0,
            power_requirement=2.0,
            data_rate=1.0
        ),
        cost_factors=CostFactors(
            launch_cost_per_kg=50000,
            spacecraft_cost_per_kg=30000,
            operations_cost_per_day=100000
        ),
        mission_duration_days=10,
        target_orbit=OrbitParameters(
            altitude=100000,  # m
            inclination=90.0,  # degrees
            eccentricity=0.0
        )
    )
    
    print(f"✅ Mission: {config.name}")
    print(f"   Payload: {config.payload.mass} kg {config.payload.type}")
    print(f"   Duration: {config.mission_duration_days} days")
    print(f"   Target orbit: {config.target_orbit.altitude/1000:.0f} km")
    
    return config

def example_2_jax_optimization():
    """Working JAX optimization example."""
    print("\n" + "="*50)
    print("Example 2: JAX Differentiable Optimization")
    print("="*50)
    
    import jax.numpy as jnp
    from jax import grad, jit
    from src.optimization.differentiable.jax_optimizer import DifferentiableOptimizer
    
    # Define a simple trajectory optimization problem
    def trajectory_cost(params):
        """Simple trajectory cost function."""
        delta_v = params[0]
        time_of_flight = params[1]
        fuel_mass = params[2]
        
        # Multi-objective cost (minimize fuel, time, and delta-v)
        return 0.4 * fuel_mass + 0.3 * time_of_flight + 0.3 * delta_v
    
    # Create optimizer
    optimizer = DifferentiableOptimizer(
        objective_function=trajectory_cost,
        bounds=[(2000, 5000), (5, 15), (500, 2000)],  # delta_v, time, fuel
        method="L-BFGS-B",
        use_jit=True
    )
    
    # Initial guess
    x0 = jnp.array([3500.0, 10.0, 1200.0])
    
    # Optimize
    result = optimizer.optimize(x0)
    
    print(f"✅ Optimization completed")
    print(f"   Success: {result.success}")
    print(f"   Optimal delta-v: {result.x[0]:.0f} m/s")
    print(f"   Optimal time: {result.x[1]:.1f} days")
    print(f"   Optimal fuel: {result.x[2]:.0f} kg")
    print(f"   Final cost: {result.fun:.2f}")
    
    return result

def example_3_economics():
    """Working economics example."""
    print("\n" + "="*50)
    print("Example 3: Economic Analysis")
    print("="*50)
    
    from src.economics.financial_models import FinancialMetrics
    
    # Mission cash flows
    initial_investment = 100e6  # $100M
    annual_returns = 25e6       # $25M per year
    mission_years = 5
    
    # Create cash flow array
    cash_flows = np.array([-initial_investment] + [annual_returns] * mission_years)
    
    # Calculate financial metrics
    npv = FinancialMetrics.calculate_npv(cash_flows, discount_rate=0.08)
    irr = FinancialMetrics.calculate_irr(cash_flows)
    
    # Simple ROI calculation
    total_returns = annual_returns * mission_years
    roi = (total_returns - initial_investment) / initial_investment
    
    print(f"✅ Financial Analysis:")
    print(f"   Initial Investment: ${initial_investment/1e6:.0f}M")
    print(f"   Annual Returns: ${annual_returns/1e6:.0f}M")
    print(f"   NPV (8%): ${npv/1e6:.1f}M")
    print(f"   IRR: {irr:.1%}")
    print(f"   ROI: {roi:.1%}")
    
    return {'npv': npv, 'irr': irr, 'roi': roi}

def example_4_isru_analysis():
    """Working ISRU analysis example."""
    print("\n" + "="*50)
    print("Example 4: ISRU Benefits Analysis")
    print("="*50)
    
    from src.economics.isru_benefits import ISRUBenefitAnalyzer
    
    # Create analyzer
    analyzer = ISRUBenefitAnalyzer()
    
    # Calculate savings for water production
    water_savings = analyzer.calculate_savings(
        resource="water",
        quantity_kg=1000,
        mission_duration_days=30
    )
    
    # Calculate savings for oxygen production
    oxygen_savings = analyzer.calculate_savings(
        resource="oxygen",
        quantity_kg=500,
        mission_duration_days=30
    )
    
    total_savings = water_savings + oxygen_savings
    
    print(f"✅ ISRU Analysis:")
    print(f"   Water savings: ${water_savings/1e6:.1f}M")
    print(f"   Oxygen savings: ${oxygen_savings/1e6:.1f}M")
    print(f"   Total ISRU savings: ${total_savings/1e6:.1f}M")
    
    return total_savings

def example_5_visualization():
    """Working visualization example."""
    print("\n" + "="*50)
    print("Example 5: 3D Trajectory Visualization")
    print("="*50)
    
    import plotly.graph_objects as go
    
    # Create sample trajectory data
    t = np.linspace(0, 10, 100)
    
    # Earth to Moon trajectory (simplified)
    earth_radius = 6371e3  # m
    moon_distance = 384400e3  # m
    
    # Spiral trajectory from Earth to Moon
    r = earth_radius + (moon_distance - earth_radius) * t / 10
    theta = 2 * np.pi * t
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 0.1 * r * np.sin(t)  # Small out-of-plane component
    
    # Create 3D plot
    fig = go.Figure()
    
    # Add trajectory
    fig.add_trace(go.Scatter3d(
        x=x/1e6, y=y/1e6, z=z/1e6,  # Convert to Mm
        mode='lines+markers',
        name='Trajectory',
        line=dict(color='blue', width=4),
        marker=dict(size=2)
    ))
    
    # Add Earth
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='Earth',
        marker=dict(color='green', size=10)
    ))
    
    # Add Moon (approximate final position)
    fig.add_trace(go.Scatter3d(
        x=[x[-1]/1e6], y=[y[-1]/1e6], z=[z[-1]/1e6],
        mode='markers',
        name='Moon',
        marker=dict(color='gray', size=8)
    ))
    
    fig.update_layout(
        title="Earth-Moon Trajectory",
        scene=dict(
            xaxis_title="X (Mm)",
            yaxis_title="Y (Mm)",
            zaxis_title="Z (Mm)",
            aspectmode='cube'
        ),
        showlegend=True
    )
    
    # Save plot
    fig.write_html("trajectory_demo.html")
    print(f"✅ 3D visualization created and saved as 'trajectory_demo.html'")
    
    return fig

def main():
    """Run all working examples."""
    print("🚀 Lunar Horizon Optimizer - Working Examples")
    print("=" * 60)
    print("These examples demonstrate verified working functionality.")
    
    try:
        # Example 1: Configuration
        config = example_1_configuration()
        
        # Example 2: JAX Optimization
        opt_result = example_2_jax_optimization()
        
        # Example 3: Economics
        econ_result = example_3_economics()
        
        # Example 4: ISRU
        isru_savings = example_4_isru_analysis()
        
        # Example 5: Visualization
        fig = example_5_visualization()
        
        # Summary
        print("\n" + "="*60)
        print("Working Examples Summary")
        print("="*60)
        print("\n✅ All examples completed successfully!")
        print("\nKey Results:")
        print(f"  - Mission: {config.name}")
        print(f"  - Optimal delta-v: {opt_result.x[0]:.0f} m/s")
        print(f"  - NPV: ${econ_result['npv']/1e6:.1f}M")
        print(f"  - ISRU savings: ${isru_savings/1e6:.1f}M")
        print(f"  - 3D plot saved: trajectory_demo.html")
        
        print("\n🔧 What Works:")
        print("  ✅ Mission configuration with validation")
        print("  ✅ JAX differentiable optimization")
        print("  ✅ Financial metrics (NPV, IRR, ROI)")
        print("  ✅ ISRU benefits analysis")
        print("  ✅ 3D trajectory visualization")
        
        print("\n⚠️  What Needs Integration:")
        print("  - Global optimization (PyGMO)")
        print("  - Advanced trajectory generation")
        print("  - Economic dashboard visualizations")
        print("  - Integrated workflow automation")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check environment: conda activate py312")

if __name__ == "__main__":
    main()