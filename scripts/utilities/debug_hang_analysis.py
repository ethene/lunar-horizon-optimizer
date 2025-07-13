#!/usr/bin/env python3
"""Debug script to identify where the analysis hangs at 95%."""

import logging
import sys
import os

# Configure logging for maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug_hang.log')
    ]
)

# Set all relevant loggers to DEBUG
loggers = [
    'src.cli',
    'src.lunar_horizon_optimizer', 
    'src.economics.sensitivity_analysis',
    'src.optimization.global_optimizer',
    'src.trajectory.lunar_transfer'
]

for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

print("🔧 Debug Hang Analysis - Enhanced Logging Enabled")
print("This will run a production analysis with maximum verbosity to identify the hang point")
print("Log output is saved to debug_hang.log")
print("-" * 60)

# Import and run analysis
from src.cli import main

# Run with production parameters and verbose output
sys.argv = [
    'debug_hang_analysis.py',
    'analyze',
    '--config', 'scenarios/01_basic_transfer.json',
    '--output', 'debug_hang_output',
    '--verbose'  # Enable verbose output to see PyGMO output
]

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        print("📋 Check debug_hang.log for detailed log analysis")
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        print("📋 Check debug_hang.log for detailed error analysis")