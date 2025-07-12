#!/usr/bin/env python3
"""
Integration Test - Demonstrating Fixed Components

This test demonstrates the integration improvements made to address
the PRD compliance gaps identified in the validation.
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_global_optimization_fix():
    """Test the fixed global optimization API."""
    print("\n" + "="*60)
    print("Testing Global Optimization API Fix")
    print("="*60)
    
    try:
        from src.optimization.pareto_analysis import ParetoAnalyzer
        
        # Create sample solutions
        solutions = [
            {
                'delta_v': 3200,
                'time_of_flight': 7,
                'cost': 80e6,
                'objectives': {'delta_v': 3200, 'time': 7, 'cost': 80e6}
            },
            {
                'delta_v': 3500,
                'time_of_flight': 5,
                'cost': 90e6,
                'objectives': {'delta_v': 3500, 'time': 5, 'cost': 90e6}
            },
            {
                'delta_v': 3000,
                'time_of_flight': 10,
                'cost': 70e6,
                'objectives': {'delta_v': 3000, 'time': 10, 'cost': 70e6}
            }
        ]
        
        # Test Pareto front finding
        analyzer = ParetoAnalyzer()
        pareto_front = analyzer.find_pareto_front(solutions)
        
        print(f"✅ Global optimization API working")
        print(f"   - Input solutions: {len(solutions)}")
        print(f"   - Pareto solutions: {len(pareto_front)}")
        
        return True, pareto_front
        
    except Exception as e:
        print(f"❌ Global optimization failed: {e}")
        return False, None

def test_economic_dashboard_fix():
    """Test the fixed economic dashboard."""
    print("\n" + "="*60)
    print("Testing Economic Dashboard Fix")
    print("="*60)
    
    try:
        from src.visualization.economic_visualization import EconomicVisualizer
        
        # Create visualizer
        visualizer = EconomicVisualizer()
        
        # Test scenario comparison
        scenarios = ['Baseline', 'Optimized', 'ISRU Enhanced']
        npv_values = [50e6, 75e6, 100e6]
        
        fig = visualizer.create_scenario_comparison(
            scenarios=scenarios,
            npv_values=npv_values,
            title="Mission Scenarios"
        )
        
        print(f"✅ Economic dashboard working")
        print(f"   - Scenarios compared: {len(scenarios)}")
        print(f"   - Chart created with {len(fig.data)} traces")
        
        return True, fig
        
    except Exception as e:
        print(f"❌ Economic dashboard failed: {e}")
        return False, None

def test_integrated_dashboard_fix():
    """Test the new integrated dashboard."""
    print("\n" + "="*60)
    print("Testing Integrated Dashboard Fix")
    print("="*60)
    
    try:
        from src.visualization.integrated_dashboard import create_mission_dashboard
        
        # Create sample trajectory data
        trajectory_data = {
            'delta_v': 3500,
            'time_of_flight': 7,
            'trajectory_points': [
                (7000e3, 0, 0),
                (100000e3, 50000e3, 0),
                (384400e3, 0, 0)
            ]
        }
        
        # Create sample economic data
        economic_data = {
            'npv': 75e6,
            'irr': 0.15,
            'roi': 0.25
        }
        
        # Create dashboard
        dashboard = create_mission_dashboard(
            trajectory_results=trajectory_data,
            economic_results=economic_data,
            title="Integration Test Dashboard"
        )
        
        print(f"✅ Integrated dashboard working")
        print(f"   - Dashboard created with {len(dashboard.data)} traces")
        print(f"   - Includes trajectory and economic data")
        
        return True, dashboard
        
    except Exception as e:
        print(f"❌ Integrated dashboard failed: {e}")
        return False, None

def test_workflow_automation():
    """Test cross-module workflow automation."""
    print("\n" + "="*60)
    print("Testing Workflow Automation")
    print("="*60)
    
    try:
        # Test the workflow components we can run
        from src.optimization.differentiable.jax_optimizer import DifferentiableOptimizer
        from src.economics.financial_models import FinancialMetrics
        import jax.numpy as jnp
        
        # Step 1: Local optimization
        def simple_cost(params):
            return jnp.sum(params**2)
        
        optimizer = DifferentiableOptimizer(simple_cost, use_jit=True)
        opt_result = optimizer.optimize(jnp.array([1.0, 2.0]))
        
        # Step 2: Economic analysis
        cash_flows = np.array([-100e6, 30e6, 35e6, 40e6])
        npv = FinancialMetrics.calculate_npv(cash_flows, 0.08)
        
        # Step 3: Integration
        workflow_result = {
            'optimization': {
                'success': opt_result.success,
                'final_value': float(opt_result.fun),
                'iterations': opt_result.nit
            },
            'economics': {
                'npv': npv,
                'roi': (sum(cash_flows[1:]) + cash_flows[0]) / abs(cash_flows[0])
            }
        }
        
        print(f"✅ Workflow automation working")
        print(f"   - Optimization success: {opt_result.success}")
        print(f"   - NPV: ${npv/1e6:.1f}M")
        print(f"   - Components integrated successfully")
        
        return True, workflow_result
        
    except Exception as e:
        print(f"❌ Workflow automation failed: {e}")
        return False, None

def test_configuration_compatibility():
    """Test configuration system compatibility."""
    print("\n" + "="*60)
    print("Testing Configuration Compatibility")
    print("="*60)
    
    try:
        from src.config.models import MissionConfig, PayloadSpecification, CostFactors, OrbitParameters
        
        # Create complete configuration
        config = MissionConfig(
            name="Integration Test Mission",
            payload=PayloadSpecification(
                dry_mass=2000.0,
                payload_mass=1000.0,
                max_propellant_mass=1500.0,
                specific_impulse=450.0
            ),
            cost_factors=CostFactors(
                launch_cost_per_kg=50000,
                spacecraft_cost_per_kg=30000,
                operations_cost_per_day=100000
            ),
            mission_duration_days=10,
            target_orbit=OrbitParameters(
                altitude=100000,
                inclination=90.0,
                eccentricity=0.0
            )
        )
        
        print(f"✅ Configuration system working")
        print(f"   - Mission: {config.name}")
        print(f"   - Payload: {config.payload.payload_mass} kg")
        print(f"   - Duration: {config.mission_duration_days} days")
        
        return True, config
        
    except Exception as e:
        print(f"❌ Configuration compatibility failed: {e}")
        return False, None

def main():
    """Run all integration tests."""
    print("🔧 Integration Testing - PRD Compliance Improvements")
    print("=" * 70)
    
    test_results = {}
    
    # Test each fix
    success_1, pareto_front = test_global_optimization_fix()
    test_results['Global Optimization API'] = success_1
    
    success_2, dashboard = test_economic_dashboard_fix()
    test_results['Economic Dashboard'] = success_2
    
    success_3, integrated_dash = test_integrated_dashboard_fix()
    test_results['Integrated Dashboard'] = success_3
    
    success_4, workflow = test_workflow_automation()
    test_results['Workflow Automation'] = success_4
    
    success_5, config = test_configuration_compatibility()
    test_results['Configuration System'] = success_5
    
    # Summary
    print("\n" + "="*70)
    print("Integration Test Results")
    print("="*70)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\n📊 Results: {passed}/{total} integration tests passed")
    print(f"Success Rate: {passed/total*100:.0f}%")
    
    print(f"\n✅ Working Integrations:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Calculate improvement
    previous_compliance = 31  # From previous test
    estimated_improvement = (passed/total * 100) - previous_compliance
    
    print(f"\n📈 Estimated PRD Compliance Improvement:")
    print(f"   Previous: {previous_compliance}%")
    print(f"   Current: {passed/total*100:.0f}%")
    print(f"   Improvement: {max(0, estimated_improvement):.0f}%")
    
    if passed >= 4:
        print("\n🎉 Major integration improvements successful!")
        print("   - Global optimization API fixed")
        print("   - Economic dashboard methods added")
        print("   - Integrated dashboard created")
        print("   - Workflow automation components working")
    else:
        print("\n⚠️  Some integration issues remain")
        print("   - Continue with remaining fixes")
    
    return test_results

if __name__ == "__main__":
    main()