#!/usr/bin/env python3
"""Test script to identify exactly where the analysis hangs."""

import logging
import sys
import time
from src.lunar_horizon_optimizer import LunarHorizonOptimizer
from src.config.management.config_manager import ConfigManager

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hang_location_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def test_analysis_phases():
    """Test analysis phases individually to identify hanging point."""
    logger.info("🔍 Testing analysis phases to identify hang location")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config_data = config_manager.load_config("scenarios/01_basic_transfer.json")
        mission_config = config_data.mission_config
        cost_factors = config_data.cost_factors  
        optimization_config = config_data.optimization_config
        
        logger.info("✅ Configuration loaded successfully")
        
        # Create optimizer
        optimizer = LunarHorizonOptimizer(mission_config, cost_factors)
        logger.info("✅ Optimizer created successfully")
        
        # Test with minimal parameters to speed up testing
        optimization_config.population_size = 10  # Reduced from 52
        optimization_config.num_generations = 5   # Reduced from 30
        logger.info(f"Using reduced parameters: pop={optimization_config.population_size}, gen={optimization_config.num_generations}")
        
        start_time = time.time()
        
        # Run analysis with detailed tracking
        logger.info("🚀 Starting analysis with enhanced debug logging...")
        
        results = optimizer.analyze_mission(
            mission_name="Debug Test Mission",
            optimization_config=optimization_config,
            include_sensitivity=True,  # This includes Monte Carlo
            include_isru=True,
            verbose=True,
        )
        
        end_time = time.time()
        logger.info(f"✅ Analysis completed successfully in {end_time - start_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 Analysis Phase Location Debug Test")
    print("This runs a reduced analysis to identify where hanging occurs")
    print("Log output is saved to hang_location_debug.log")
    print("-" * 60)
    
    results = test_analysis_phases()
    
    if results:
        print("\n✅ Test completed successfully")
        print("📋 Check hang_location_debug.log for detailed phase timing")
    else:
        print("\n❌ Test failed or hung")
        print("📋 Check hang_location_debug.log for last completed phase")