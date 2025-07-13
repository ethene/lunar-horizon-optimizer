#!/usr/bin/env python3
"""
Test script to verify all scenarios work correctly with REAL LunarHorizonOptimizer.
Runs quick analysis on each scenario with small parameters for fast validation.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def run_scenario(scenario_file, output_dir, extra_args=None):
    """Run a single scenario analysis with real optimizer."""
    # Use conda run to ensure correct environment
    cmd = [
        "conda", "run", "-n", "py312", 
        "python", "src/cli.py", "analyze",
        "--config", scenario_file,
        "--output", output_dir,
        "--population-size", "8",  # Small for quick testing
        "--generations", "5",      # Small for quick testing
        "--no-sensitivity"         # Skip sensitivity for speed
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"🔄 Testing {scenario_file}...")
    start_time = time.time()
    
    try:
        # Increased timeout for real calculations
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ SUCCESS ({elapsed:.1f}s)")
            return True
        else:
            print(f"   ❌ FAILED ({elapsed:.1f}s)")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"   💥 EXCEPTION: {e}")
        return False

def main():
    """Test all scenarios with REAL LunarHorizonOptimizer."""
    print("🧪 Testing All Scenarios with REAL PyKEP/PyGMO Calculations")
    print("=" * 55)
    print("⚙️  Using: 8 population × 5 generations for quick validation")
    print("🏃 Expected: 1-3 minutes per scenario")
    print()
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    # Test scenarios
    scenarios = [
        ("scenarios/01_basic_transfer.json", "test_results/scenario_01"),
        ("scenarios/02_launch_windows.json", "test_results/scenario_02"),
        ("scenarios/03_propulsion_comparison.json", "test_results/scenario_03"),
        ("scenarios/04_pareto_optimization.json", "test_results/scenario_04"),
        ("scenarios/05_constellation_optimization.json", "test_results/scenario_05"),
        ("scenarios/06_isru_economics.json", "test_results/scenario_06"),
        ("scenarios/07_environmental_economics.json", "test_results/scenario_07", ["--learning-rate", "0.85", "--carbon-price", "100"]),
        ("scenarios/08_risk_analysis.json", "test_results/scenario_08"),
        ("scenarios/09_complete_mission.json", "test_results/scenario_09"),
        ("scenarios/10_multi_mission_campaign.json", "test_results/scenario_10"),
    ]
    
    results = []
    total_start = time.time()
    
    for scenario_data in scenarios:
        scenario_file = scenario_data[0]
        output_dir = scenario_data[1]
        extra_args = scenario_data[2] if len(scenario_data) > 2 else None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run test
        success = run_scenario(scenario_file, output_dir, extra_args)
        results.append((scenario_file, success))
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for scenario_file, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {scenario_file}")
    
    print(f"\n📈 Results: {passed}/{total} scenarios passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds")
    
    if passed == total:
        print("\n🎉 All scenarios working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} scenarios failed - check logs above")
        return 1

if __name__ == "__main__":
    sys.exit(main())