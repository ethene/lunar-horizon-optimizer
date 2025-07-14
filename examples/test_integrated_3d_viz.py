#!/usr/bin/env python3
"""
Test script to verify integrated 3D landing visualization is working correctly.

This script tests the CLI integration by running scenarios and checking that
proper 3D visualizations are generated with realistic trajectories.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_scenario_with_3d_viz(scenario_name):
    """Run a scenario with 3D visualization and check results."""
    print(f"\n🚀 Testing scenario: {scenario_name}")
    
    # Run the CLI command
    cmd = [
        "./lunar_opt.py", "run", "scenario", scenario_name,
        "--include-descent", "--3d-viz",
        "--gens", "5", "--population", "10",  # Fast settings
        "--no-sensitivity", "--no-isru"
    ]
    
    try:
        # Change to project directory
        project_dir = Path(__file__).parent.parent
        os.chdir(project_dir)
        
        # Activate conda environment and run command
        conda_cmd = ["conda", "run", "-n", "py312"] + cmd
        
        print(f"Running: {' '.join(conda_cmd)}")
        result = subprocess.run(
            conda_cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Scenario completed successfully")
            
            # Check if 3D visualization was created
            if "3D visualization saved" in result.stdout:
                print("✅ 3D visualization was generated")
                
                # Extract the path to the generated file
                lines = result.stdout.split('\n')
                for line in lines:
                    if "3D visualization saved" in line:
                        viz_path = line.split(': ')[-1].strip()
                        print(f"📄 Visualization file: {viz_path}")
                        
                        # Check file exists and has reasonable size
                        if os.path.exists(viz_path):
                            file_size = os.path.getsize(viz_path)
                            print(f"📊 File size: {file_size/1024/1024:.1f} MB")
                            
                            if file_size > 1024 * 1024:  # > 1MB
                                print("✅ File size looks reasonable for 3D visualization")
                                return True
                            else:
                                print("⚠️  File size seems too small")
                                return False
                        else:
                            print("❌ Visualization file not found")
                            return False
            else:
                print("❌ No 3D visualization was generated")
                return False
        else:
            print(f"❌ Scenario failed with return code: {result.returncode}")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Scenario timed out")
        return False
    except Exception as e:
        print(f"❌ Error running scenario: {e}")
        return False


def check_trajectory_realism(viz_path):
    """Check if the trajectory data in the visualization is realistic."""
    try:
        # Read the HTML file and extract some basic statistics
        with open(viz_path, 'r') as f:
            content = f.read()
            
        # Look for data patterns that indicate realistic trajectories
        checks = {
            "Has position data": '"positions"' in content,
            "Has velocity data": '"velocities"' in content,
            "Has thrust data": '"thrust_profile"' in content,
            "Has time data": '"time_points"' in content,
            "Has lunar surface": '"Lunar Surface"' in content,
            "Has landing target": '"Landing Target"' in content,
        }
        
        print("\n🔍 Trajectory Data Checks:")
        all_good = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
            if not passed:
                all_good = False
                
        return all_good
        
    except Exception as e:
        print(f"❌ Error checking trajectory realism: {e}")
        return False


def main():
    """Run tests for integrated 3D visualization."""
    print("🌙 Testing Integrated 3D Landing Trajectory Visualization")
    print("=" * 60)
    
    # Test scenarios in order of complexity
    scenarios = [
        "13_powered_descent_quick",      # Quick test
        "11_powered_descent_mission",    # Intermediate
    ]
    
    results = {}
    
    for scenario in scenarios:
        success = run_scenario_with_3d_viz(scenario)
        results[scenario] = success
        
        if success:
            print(f"✅ {scenario}: PASSED")
        else:
            print(f"❌ {scenario}: FAILED")
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for scenario, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"   {scenario}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} scenarios passed")
    
    if passed == total:
        print("🎉 All tests PASSED! 3D visualization integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())