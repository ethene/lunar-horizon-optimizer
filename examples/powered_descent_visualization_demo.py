#!/usr/bin/env python3
"""
Powered Descent Visualization Demo

This script demonstrates the comprehensive visualization capabilities
of the Lunar Horizon Optimizer for powered descent scenarios without
running a full optimization (for quick demonstration purposes).
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.dashboard import ComprehensiveDashboard, MissionAnalysisData
from src.visualization.economic_visualization import EconomicVisualizer
from src.visualization.trajectory_visualization import TrajectoryVisualizer
from src.visualization.optimization_visualization import OptimizationVisualizer
from src.economics.reporting import FinancialSummary
from src.economics.cost_models import CostBreakdown
from src.cli.scenario_manager import ScenarioManager


def create_demo_trajectory_data():
    """Create demo trajectory data for visualization."""
    # Generate synthetic Earth-Moon trajectory points
    t = np.linspace(0, 4.5, 100)  # 4.5 days
    
    # Earth orbit to Moon trajectory (simplified)
    x = 6378 + 200 + t * 80000  # km - increasing distance from Earth
    y = np.sin(t * 0.5) * 10000  # km - some orbital motion
    z = np.cos(t * 0.3) * 5000   # km - out of plane motion
    
    trajectory_data = {
        'time': t,
        'position': {'x': x, 'y': y, 'z': z},
        'velocity': {'vx': np.gradient(x), 'vy': np.gradient(y), 'vz': np.gradient(z)},
        'total_dv': 3250.0,
        'transfer_time': 4.5,
        'earth_orbit_alt': 400.0,
        'moon_orbit_alt': 100.0
    }
    
    return trajectory_data


def create_demo_optimization_results():
    """Create demo multi-objective optimization results."""
    # Generate synthetic Pareto front data
    n_solutions = 50
    
    # Create realistic trade-offs between delta-v, time, and cost
    delta_v = np.random.uniform(3000, 3500, n_solutions)
    time = np.random.uniform(3.5, 7.0, n_solutions)
    
    # Cost should increase with delta-v and decrease with time (more fuel needed for faster missions)
    base_cost = 150e6  # $150M base cost
    cost = base_cost + (delta_v - 3000) * 20000 + (7.0 - time) * 5e6
    
    # Add some powered descent cost variation (5-8% of total)
    descent_fraction = np.random.uniform(0.05, 0.08, n_solutions)
    descent_cost = cost * descent_fraction
    
    optimization_results = {
        'pareto_solutions': [
            {
                'objectives': [dv, t, c],
                'parameters': [400.0, 100.0, t],
                'descent_cost': dc,
                'descent_fraction': df
            }
            for dv, t, c, dc, df in zip(delta_v, time, cost, descent_cost, descent_fraction)
        ],
        'convergence_history': {
            'generation': list(range(25)),
            'best_fitness': np.random.exponential(0.5, 25) + 1.0,
            'mean_fitness': np.random.exponential(0.8, 25) + 2.0
        },
        'statistics': {
            'total_evaluations': 1000,
            'pareto_front_size': n_solutions,
            'optimization_time': 45.2
        }
    }
    
    return optimization_results


def create_demo_financial_data():
    """Create demo financial analysis data."""
    # Create realistic financial summary for powered descent mission
    financial_summary = FinancialSummary(
        total_investment=180e6,      # $180M total investment
        total_revenue=420e6,         # $420M total revenue over project life
        net_present_value=85e6,      # $85M NPV
        internal_rate_of_return=0.18, # 18% IRR
        return_on_investment=0.22,   # 22% ROI
        payback_period_years=6.2,    # 6.2 years payback
        mission_duration_years=8.0,  # 8 year project
        probability_of_success=0.85  # 85% success probability
    )
    
    # Create cost breakdown including descent costs
    cost_breakdown = CostBreakdown(
        development=95e6,    # $95M development
        launch=35e6,         # $35M launch
        operations=38e6,     # $38M operations
        contingency=12e6,    # $12M contingency
        total=180e6          # $180M total
    )
    
    # Add descent-specific costs
    descent_costs = {
        'propellant_cost': 2.8e6,      # $2.8M propellant
        'lander_hardware_cost': 10e6,   # $10M lander hardware
        'total_descent_cost': 12.8e6,   # $12.8M total descent
        'descent_fraction': 0.071       # 7.1% of total mission cost
    }
    
    return financial_summary, cost_breakdown, descent_costs


def generate_powered_descent_visualizations():
    """Generate comprehensive powered descent visualizations."""
    print("🚀 Generating Powered Descent Visualizations Demo")
    print("=" * 60)
    
    # Load one of our powered descent scenarios for realistic parameters
    scenario_manager = ScenarioManager()
    config = scenario_manager.get_scenario_config('13_powered_descent_quick')
    
    if not config:
        print("❌ Could not load powered descent scenario configuration")
        return
    
    descent_params = config.get('descent_parameters', {})
    economics = config.get('economics', {})
    
    print(f"📊 Using scenario: {config['mission']['name']}")
    print(f"   - Thrust: {descent_params.get('thrust', 'N/A')} N")
    print(f"   - ISP: {descent_params.get('isp', 'N/A')} s")
    print(f"   - Burn time: {descent_params.get('burn_time', 'N/A')} s")
    print(f"   - Propellant cost: ${economics.get('propellant_unit_cost', 'N/A')}/kg")
    print()
    
    # Generate demo data
    print("📈 Creating demo data...")
    trajectory_data = create_demo_trajectory_data()
    optimization_results = create_demo_optimization_results()
    financial_summary, cost_breakdown, descent_costs = create_demo_financial_data()
    
    # Initialize visualizers
    dashboard = ComprehensiveDashboard()
    econ_viz = EconomicVisualizer()
    traj_viz = TrajectoryVisualizer()
    opt_viz = OptimizationVisualizer()
    
    # Create mission analysis data
    mission_data = MissionAnalysisData(
        mission_name=config['mission']['name'],
        trajectory_data=trajectory_data,
        optimization_results=optimization_results,
        financial_summary=financial_summary,
        cost_breakdown=cost_breakdown
    )
    
    # Create output directory
    output_dir = Path("powered_descent_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating visualizations...")
    
    # 1. Executive Dashboard
    print("   - Creating executive dashboard...")
    try:
        exec_dashboard = dashboard.create_executive_dashboard(mission_data)
        exec_dashboard.write_html(output_dir / "executive_dashboard.html")
        print("     ✅ Executive dashboard saved")
    except Exception as e:
        print(f"     ❌ Executive dashboard failed: {e}")
    
    # 2. Technical Dashboard
    print("   - Creating technical dashboard...")
    try:
        tech_dashboard = dashboard.create_technical_dashboard(mission_data)
        tech_dashboard.write_html(output_dir / "technical_dashboard.html")
        print("     ✅ Technical dashboard saved")
    except Exception as e:
        print(f"     ❌ Technical dashboard failed: {e}")
    
    # 3. Economic Visualization
    print("   - Creating economic analysis...")
    try:
        econ_dashboard = econ_viz.create_financial_dashboard(financial_summary)
        econ_dashboard.write_html(output_dir / "economic_dashboard.html")
        print("     ✅ Economic dashboard saved")
    except Exception as e:
        print(f"     ❌ Economic dashboard failed: {e}")
    
    # 4. Powered Descent Cost Analysis
    print("   - Creating descent cost breakdown...")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create descent cost breakdown visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Descent Cost Breakdown",
                "Cost vs Mission Parameters",
                "Propellant Mass Calculation",
                "Economic Impact"
            ),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Pie chart of descent costs
        fig.add_trace(
            go.Pie(
                labels=["Propellant", "Hardware", "Other Mission Costs"],
                values=[descent_costs['propellant_cost'], 
                       descent_costs['lander_hardware_cost'],
                       cost_breakdown.total - descent_costs['total_descent_cost']],
                name="Descent Costs"
            ),
            row=1, col=1
        )
        
        # Scatter plot of cost vs parameters
        thrust_range = np.linspace(8000, 20000, 20)
        cost_vs_thrust = 150e6 + (thrust_range - 10000) * 1000  # Cost increases with thrust
        
        fig.add_trace(
            go.Scatter(
                x=thrust_range,
                y=cost_vs_thrust/1e6,
                mode='lines+markers',
                name="Total Mission Cost",
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        
        # Bar chart of propellant mass calculation
        propellant_mass = descent_params.get('thrust', 10000) / (descent_params.get('isp', 320) * 9.81) * descent_params.get('burn_time', 31.4)
        
        fig.add_trace(
            go.Bar(
                x=["Calculated", "Target", "Upper Bound"],
                y=[propellant_mass, 100, 150],
                name="Propellant Mass (kg)",
                marker_color=['green', 'blue', 'orange']
            ),
            row=2, col=1
        )
        
        # Indicator for descent fraction
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=descent_costs['descent_fraction'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Descent Cost Fraction (%)"},
                delta={'reference': 8},
                gauge={
                    'axis': {'range': [None, 12]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 10], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 10}
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Powered Descent Cost Analysis",
            height=800,
            showlegend=True
        )
        
        fig.write_html(output_dir / "descent_cost_analysis.html")
        print("     ✅ Descent cost analysis saved")
        
    except Exception as e:
        print(f"     ❌ Descent cost analysis failed: {e}")
    
    # 5. 3D Trajectory Visualization
    print("   - Creating 3D trajectory plot...")
    try:
        traj_fig = traj_viz.create_3d_trajectory_plot(trajectory_data)
        traj_fig.write_html(output_dir / "trajectory_3d.html")
        print("     ✅ 3D trajectory plot saved")
    except Exception as e:
        print(f"     ❌ 3D trajectory plot failed: {e}")
    
    # 6. Pareto Front Visualization
    print("   - Creating Pareto front analysis...")
    try:
        pareto_fig = opt_viz.create_pareto_front_plot(
            optimization_results,
            objective_names=["Delta-V (m/s)", "Transfer Time (days)", "Total Cost ($M)"]
        )
        pareto_fig.write_html(output_dir / "pareto_front.html")
        print("     ✅ Pareto front analysis saved")
    except Exception as e:
        print(f"     ❌ Pareto front analysis failed: {e}")
    
    # 7. Save demo data as JSON
    print("   - Saving demo data...")
    demo_data = {
        'scenario_config': config,
        'trajectory_data': trajectory_data,
        'optimization_results': optimization_results,
        'financial_data': {
            'summary': financial_summary.__dict__,
            'cost_breakdown': cost_breakdown.__dict__,
            'descent_costs': descent_costs
        },
        'generated_timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "demo_data.json", 'w') as f:
        json.dump(demo_data, f, indent=2, default=str)
    
    print("     ✅ Demo data saved")
    
    # Create summary
    print()
    print("📋 Visualization Summary")
    print("-" * 40)
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Generated files:")
    for file in sorted(output_dir.glob("*.html")):
        print(f"   📊 {file.name}")
    for file in sorted(output_dir.glob("*.json")):
        print(f"   📄 {file.name}")
    
    print()
    print("🌐 To view dashboards:")
    print(f"   open {output_dir.absolute()}/executive_dashboard.html")
    print(f"   open {output_dir.absolute()}/technical_dashboard.html")
    print(f"   open {output_dir.absolute()}/descent_cost_analysis.html")
    
    print()
    print("✅ Powered descent visualization demo completed!")
    
    return output_dir


if __name__ == "__main__":
    generate_powered_descent_visualizations()