#!/usr/bin/env python3
"""Test script to isolate Monte Carlo simulation hanging issue."""

import logging
import sys
from src.economics.sensitivity_analysis import EconomicSensitivityAnalyzer

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('monte_carlo_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def simple_economic_model(params):
    """Simple economic model for testing."""
    base_cost = 500e6
    base_revenue = 750e6
    
    total_cost = base_cost * params.get("cost_multiplier", 1.0)
    total_revenue = base_revenue * params.get("revenue_multiplier", 1.0)
    
    npv = total_revenue - total_cost
    return {"npv": npv}

def test_monte_carlo():
    """Test Monte Carlo simulation to identify hanging point."""
    logger.info("🧪 Testing Monte Carlo simulation with enhanced logging")
    
    # Create analyzer
    analyzer = EconomicSensitivityAnalyzer(simple_economic_model)
    
    # Test parameters
    base_params = {
        "cost_multiplier": 1.0,
        "revenue_multiplier": 1.0,
    }
    
    distributions = {
        "cost_multiplier": {
            "type": "triang",
            "min": 0.8,
            "mode": 1.0,
            "max": 1.5,
        },
        "revenue_multiplier": {
            "type": "normal", 
            "mean": 1.0, 
            "std": 0.2
        },
    }
    
    logger.info("🚀 Starting Monte Carlo simulation with 1000 iterations...")
    
    try:
        results = analyzer.monte_carlo_simulation(
            base_params,
            distributions,
            num_simulations=1000
        )
        
        logger.info("✅ Monte Carlo simulation completed successfully!")
        logger.info(f"Valid simulations: {results['valid_simulations']}")
        logger.info(f"Mean NPV: ${results['statistics']['mean']/1e6:.1f}M")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Monte Carlo simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔬 Monte Carlo Simulation Debug Test")
    print("This isolates the Monte Carlo simulation to identify hanging issues")
    print("Log output is saved to monte_carlo_debug.log")
    print("-" * 60)
    
    results = test_monte_carlo()
    
    if results:
        print("\n✅ Test completed successfully")
        print("📋 Check monte_carlo_debug.log for detailed analysis")
    else:
        print("\n❌ Test failed")
        print("📋 Check monte_carlo_debug.log for error details")