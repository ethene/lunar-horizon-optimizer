#!/usr/bin/env python3
"""
Quick Start Example for Lunar Horizon Optimizer

This script demonstrates basic usage of the optimizer with a simple mission configuration.
Run this to verify your installation and see example outputs.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from src.config.models import MissionConfig, PayloadSpecification
from src.economics.financial_models import FinancialMetrics
import numpy as np


def example_1_basic_configuration():
    """Example 1: Create and validate a basic mission configuration."""
    print("\n" + "="*60)
    print("Example 1: Basic Mission Configuration")
    print("="*60)
    
    # Create basic mission configuration
    config = MissionConfig()
    
    # Create payload specification
    payload = PayloadSpecification(
        type="cargo",
        mass=1000.0,  # kg
        volume=10.0,  # m³
        power_requirement=2.0,  # kW
        data_rate=1.0  # Mbps
    )
    
    # Basic spacecraft parameters (simplified for demo)
    dry_mass = 2000.0  # kg
    propellant_mass = 1500.0  # kg
    wet_mass = dry_mass + propellant_mass + payload.mass
    
    # Display configuration
    print(f"\nMission Configuration Created")
    print(f"Payload Type: {payload.type}")
    print(f"Payload Mass: {payload.mass:.0f} kg")
    print(f"Duration: 10 days (example)")
    print(f"\nSpacecraft Mass Breakdown:")
    print(f"  Dry Mass: {dry_mass:.0f} kg")
    print(f"  Propellant: {propellant_mass:.0f} kg")
    print(f"  Payload: {payload.mass:.0f} kg")
    print(f"  Total Mass: {wet_mass:.0f} kg")
    print(f"  Payload Fraction: {payload.mass/wet_mass:.1%}")
    
    return config, payload


def example_2_economic_analysis():
    """Example 2: Perform basic economic analysis."""
    print("\n" + "="*60)
    print("Example 2: Economic Analysis")
    print("="*60)
    
    # Define mission costs and revenues
    costs = {
        "launch": 50_000_000,
        "spacecraft": 30_000_000,
        "operations": 20_000_000,
    }
    
    total_cost = sum(costs.values())
    revenue = 150_000_000
    
    # Create cash flow projection (simplified)
    # Year 0: Initial investment
    # Years 1-5: Revenue from payload delivery
    cash_flows = [-total_cost] + [revenue/5] * 5
    
    # Calculate financial metrics
    npv = FinancialMetrics.calculate_npv(cash_flows, discount_rate=0.08)
    irr = FinancialMetrics.calculate_irr(cash_flows)
    roi = (revenue - total_cost) / total_cost
    
    # Display results
    print(f"\nTotal Investment: ${total_cost:,.0f}")
    print(f"Expected Revenue: ${revenue:,.0f}")
    print(f"\nFinancial Metrics:")
    print(f"  NPV (8% discount): ${npv:,.0f}")
    print(f"  IRR: {irr:.1%}")
    print(f"  ROI: {roi:.1%}")
    
    return npv, irr, roi


def example_3_trajectory_calculations():
    """Example 3: Basic trajectory calculations."""
    print("\n" + "="*60)
    print("Example 3: Trajectory Calculations")
    print("="*60)
    
    # Simplified delta-V calculations for Earth-Moon transfer
    # These are approximations for demonstration
    
    # LEO to Trans-Lunar Injection (TLI)
    delta_v_tli = 3150  # m/s
    
    # Lunar Orbit Insertion (LOI)
    delta_v_loi = 900  # m/s
    
    # Total delta-V
    total_delta_v = delta_v_tli + delta_v_loi
    
    # Calculate propellant requirements
    isp = 450  # seconds
    g0 = 9.81  # m/s
    exhaust_velocity = isp * g0
    
    # Initial mass (with payload)
    m_initial = 4500  # kg
    
    # Use Tsiolkovsky rocket equation
    mass_ratio = np.exp(total_delta_v / exhaust_velocity)
    m_final = m_initial / mass_ratio
    propellant_used = m_initial - m_final
    
    print(f"\nTrajectory Requirements:")
    print(f"  Earth Departure (TLI): {delta_v_tli} m/s")
    print(f"  Lunar Insertion (LOI): {delta_v_loi} m/s")
    print(f"  Total Delta-V: {total_delta_v} m/s")
    print(f"\nPropellant Requirements:")
    print(f"  Initial Mass: {m_initial:.0f} kg")
    print(f"  Propellant Used: {propellant_used:.0f} kg")
    print(f"  Final Mass: {m_final:.0f} kg")
    
    return total_delta_v, propellant_used


def example_4_isru_benefits():
    """Example 4: Calculate ISRU benefits."""
    print("\n" + "="*60)
    print("Example 4: ISRU Benefits Analysis")
    print("="*60)
    
    # ISRU can produce propellant on the Moon
    # This reduces the mass that needs to be launched from Earth
    
    # Propellant needed for return trip
    return_propellant = 800  # kg
    
    # Cost to deliver 1 kg to lunar surface
    delivery_cost_per_kg = 100_000  # USD/kg
    
    # ISRU production cost per kg
    isru_cost_per_kg = 10_000  # USD/kg
    
    # Calculate savings
    earth_delivery_cost = return_propellant * delivery_cost_per_kg
    isru_production_cost = return_propellant * isru_cost_per_kg
    savings = earth_delivery_cost - isru_production_cost
    
    print(f"\nReturn Propellant Required: {return_propellant} kg")
    print(f"\nCost Comparison:")
    print(f"  Deliver from Earth: ${earth_delivery_cost:,.0f}")
    print(f"  Produce via ISRU: ${isru_production_cost:,.0f}")
    print(f"  Savings: ${savings:,.0f} ({savings/earth_delivery_cost:.0%})")
    
    return savings


def main():
    """Run all examples."""
    print("\n🚀 Lunar Horizon Optimizer - Quick Start Examples")
    print("=" * 70)
    print("This demonstrates basic functionality of the optimizer.")
    print("For full optimization runs, use the complete API.")
    
    # Run examples
    try:
        # Example 1: Configuration
        config, payload = example_1_basic_configuration()
        
        # Example 2: Economics
        npv, irr, roi = example_2_economic_analysis()
        
        # Example 3: Trajectory
        delta_v, propellant = example_3_trajectory_calculations()
        
        # Example 4: ISRU
        isru_savings = example_4_isru_benefits()
        
        # Summary
        print("\n" + "="*60)
        print("Quick Start Summary")
        print("="*60)
        print("\n✅ All examples completed successfully!")
        print("\nKey Results:")
        print(f"  - Mission Duration: 10 days")
        print(f"  - Total Delta-V: {delta_v} m/s")
        print(f"  - Financial NPV: ${npv:,.0f}")
        print(f"  - ISRU Savings: ${isru_savings:,.0f}")
        print("\n📚 Next Steps:")
        print("  1. Review the User Guide (docs/USER_GUIDE.md)")
        print("  2. Explore example configurations (examples/configs/)")
        print("  3. Run full optimization with your parameters")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you're in the correct environment:")
        print("  conda activate py312")
        print("  cd <project_root>")
        print("  python examples/quick_start.py")


if __name__ == "__main__":
    main()