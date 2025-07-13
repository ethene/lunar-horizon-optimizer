#!/usr/bin/env python3
"""Cost Comparison Demo: Before vs After Learning Curves and Environmental Costs.

This example demonstrates the cost impact of implementing Wright's law learning curves
and CO₂ environmental costs in the Lunar Horizon Optimizer.

Usage:
    conda activate py312
    python examples/cost_comparison_demo.py

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.costs import CostFactors
from src.optimization.cost_integration import CostCalculator, launch_price, co2_cost


def demo_learning_curve_impact():
    """Demonstrate the impact of Wright's law learning curves."""
    print("🔬 Wright's Law Learning Curve Impact")
    print("=" * 45)
    
    base_price = 10000.0  # $10,000/kg base launch cost
    learning_rate = 0.88  # 12% reduction per doubling
    base_year = 2024
    
    print(f"Base launch cost (2024): ${base_price:,.0f}/kg")
    print(f"Learning rate: {learning_rate:.2f} (12% reduction per doubling)")
    print()
    
    years = [2024, 2026, 2028, 2030, 2035]
    
    for year in years:
        adjusted_price = launch_price(
            year=year,
            base_price=base_price,
            learning_rate=learning_rate,
            base_year=base_year,
            cumulative_units_base=10
        )
        
        savings = base_price - adjusted_price
        savings_percent = (savings / base_price) * 100
        
        print(f"{year}: ${adjusted_price:,.0f}/kg "
              f"(${savings:,.0f} savings, {savings_percent:.1f}% reduction)")
    
    print("\n✅ Learning curve reduces costs significantly over time!")


def demo_environmental_cost_impact():
    """Demonstrate the impact of CO₂ environmental costs."""
    print("\n🌍 CO₂ Environmental Cost Impact")
    print("=" * 35)
    
    payload_mass = 1000.0  # kg
    print(f"Payload mass: {payload_mass:,.0f} kg")
    print()
    
    scenarios = [
        ("Low carbon price", 25.0, 2.0),
        ("Medium carbon price", 50.0, 2.5),
        ("High carbon price", 100.0, 3.0),
        ("Very high carbon price", 200.0, 3.5),
    ]
    
    for scenario_name, carbon_price, co2_per_kg in scenarios:
        env_cost = co2_cost(
            payload_mass_kg=payload_mass,
            co2_per_kg=co2_per_kg,
            price_per_ton=carbon_price
        )
        
        cost_per_kg = env_cost / payload_mass
        
        print(f"{scenario_name:20s}: ${env_cost:7,.0f} "
              f"(${cost_per_kg:4.0f}/kg, {co2_per_kg}tCO₂/kg @ ${carbon_price}/tCO₂)")
    
    print("\n✅ Environmental costs add significant mission cost!")


def demo_cost_calculator_comparison():
    """Compare old vs new cost calculation methodology."""
    print("\n💰 Mission Cost Comparison: Before vs After Upgrade")
    print("=" * 55)
    
    # Mission parameters
    mission_params = {
        "total_dv": 3200.0,
        "transfer_time": 4.5,
        "earth_orbit_alt": 400.0,
        "moon_orbit_alt": 100.0
    }
    
    print("Mission parameters:")
    print(f"  Total Δv: {mission_params['total_dv']:,.0f} m/s")
    print(f"  Transfer time: {mission_params['transfer_time']:.1f} days")
    print(f"  Earth orbit: {mission_params['earth_orbit_alt']:.0f} km")
    print(f"  Lunar orbit: {mission_params['moon_orbit_alt']:.0f} km")
    print()
    
    # BEFORE: Traditional cost calculation (2024 baseline)
    old_cost_factors = CostFactors(
        launch_cost_per_kg=10000.0,
        operations_cost_per_day=100000.0,
        development_cost=1000000000.0,
        contingency_percentage=20.0,
        # Use defaults for new parameters
    )
    
    old_calculator = CostCalculator(cost_factors=old_cost_factors, mission_year=2024)
    old_cost = old_calculator.calculate_mission_cost(**mission_params)
    old_breakdown = old_calculator.calculate_cost_breakdown(**mission_params)
    
    # AFTER: Advanced cost calculation with learning curves and environmental costs (2028)
    new_cost_factors = CostFactors(
        launch_cost_per_kg=10000.0,
        operations_cost_per_day=100000.0,
        development_cost=1000000000.0,
        contingency_percentage=20.0,
        learning_rate=0.88,  # Aggressive learning
        carbon_price_per_ton_co2=75.0,  # High carbon price
        co2_emissions_per_kg_payload=2.2,  # Efficient launcher
        environmental_compliance_factor=1.15
    )
    
    new_calculator = CostCalculator(cost_factors=new_cost_factors, mission_year=2028)
    new_cost = new_calculator.calculate_mission_cost(**mission_params)
    new_breakdown = new_calculator.calculate_cost_breakdown(**mission_params)
    
    # Cost comparison
    cost_difference = new_cost - old_cost
    percent_change = (cost_difference / old_cost) * 100
    
    print("📊 COST COMPARISON RESULTS")
    print("-" * 30)
    print(f"Before upgrade (2024): ${old_cost:12,.0f}")
    print(f"After upgrade (2028):  ${new_cost:12,.0f}")
    print(f"Difference:            ${cost_difference:12,.0f} ({percent_change:+.1f}%)")
    print()
    
    # Detailed breakdown comparison
    print("📋 DETAILED COST BREAKDOWN")
    print("-" * 30)
    
    components = [
        ("Propellant", "propellant_cost"),
        ("Launch", "launch_cost"),
        ("Operations", "operations_cost"),
        ("Development", "development_cost"),
        ("Altitude", "altitude_cost"),
        ("Contingency", "contingency_cost"),
    ]
    
    for component_name, key in components:
        old_value = old_breakdown.get(key, 0)
        new_value = new_breakdown.get(key, 0)
        difference = new_value - old_value
        
        print(f"{component_name:12s}: ${old_value:10,.0f} → ${new_value:10,.0f} "
              f"({difference:+10,.0f})")
    
    # New components
    if "environmental_cost" in new_breakdown:
        env_cost = new_breakdown["environmental_cost"]
        print(f"{'Environmental':12s}: ${0:10,.0f} → ${env_cost:10,.0f} "
              f"({env_cost:+10,.0f}) ✨ NEW")
    
    if "learning_curve_savings" in new_breakdown:
        lc_savings = new_breakdown["learning_curve_savings"]
        print(f"{'LC Savings':12s}: ${0:10,.0f} → ${lc_savings:10,.0f} "
              f"({lc_savings:+10,.0f}) ✨ NEW")
    
    print(f"{'TOTAL':12s}: ${old_cost:10,.0f} → ${new_cost:10,.0f} "
          f"({cost_difference:+10,.0f})")
    
    # Analysis
    print()
    print("📈 ANALYSIS")
    print("-" * 10)
    
    if cost_difference < 0:
        print(f"✅ Net cost REDUCTION of ${abs(cost_difference):,.0f} ({abs(percent_change):.1f}%)")
        print("   Learning curve benefits outweigh environmental costs")
    else:
        print(f"⚠️ Net cost INCREASE of ${cost_difference:,.0f} ({percent_change:.1f}%)")
        print("   Environmental costs exceed learning curve benefits")
    
    # Fractions
    if "environmental_fraction" in new_breakdown:
        env_fraction = new_breakdown["environmental_fraction"]
        print(f"   Environmental costs: {env_fraction:.1%} of total mission cost")
    
    if "learning_curve_adjustment" in new_breakdown:
        lc_adjustment = new_breakdown["learning_curve_adjustment"]
        lc_reduction = (1 - lc_adjustment) * 100
        print(f"   Learning curve reduction: {lc_reduction:.1f}% in launch costs")


def demo_cli_flag_examples():
    """Demonstrate CLI flag usage examples."""
    print("\n🖥️ CLI Usage Examples")
    print("=" * 22)
    
    examples = [
        {
            "description": "Conservative learning, low carbon price",
            "flags": "--learning-rate 0.92 --carbon-price 25.0"
        },
        {
            "description": "Aggressive learning, medium carbon price", 
            "flags": "--learning-rate 0.85 --carbon-price 50.0"
        },
        {
            "description": "Moderate learning, high carbon price",
            "flags": "--learning-rate 0.90 --carbon-price 100.0"
        },
        {
            "description": "Very aggressive learning, very high carbon price",
            "flags": "--learning-rate 0.80 --carbon-price 150.0"
        }
    ]
    
    print("python src/cli.py analyze --config examples/config_after_upgrade.json \\")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['flags']}")
    
    print("\n✅ CLI flags provide flexible cost modeling!")


def main():
    """Run all cost comparison demonstrations."""
    print("🌙 Lunar Horizon Optimizer - Cost Model Upgrade Demo")
    print("=" * 60)
    print("Comparing costs BEFORE and AFTER learning curves + environmental costs")
    print()
    
    try:
        # Run all demonstrations
        demo_learning_curve_impact()
        demo_environmental_cost_impact()
        demo_cost_calculator_comparison()
        demo_cli_flag_examples()
        
        print("\n🎉 Cost comparison demo completed successfully!")
        print("\nKey Takeaways:")
        print("• Learning curves reduce launch costs over time (Wright's law)")
        print("• Environmental costs add CO₂ pricing to mission analysis")
        print("• Net impact depends on mission year and carbon pricing")
        print("• CLI flags provide easy parameter adjustment")
        print("• Cost breakdown shows detailed impact of each factor")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)