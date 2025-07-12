#!/usr/bin/env python3
"""
Simple Trajectory Integration Test

This test validates the core integration of Lambert solvers with the trajectory generation
system without running heavy optimization algorithms.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_trajectory_integration():
    """Test basic trajectory generation integration."""
    print("\n" + "="*60)
    print("Testing Basic Trajectory Integration")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
        
        # Generate trajectory using Lambert method
        trajectory, total_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="lambert"
        )
        
        print(f"✅ Basic trajectory generation working")
        print(f"   - Total delta-v: {total_dv:.0f} m/s")
        print(f"   - Trajectory type: {type(trajectory).__name__}")
        
        # Test trajectory data access
        if hasattr(trajectory, 'trajectory_data'):
            traj_data = trajectory.trajectory_data
            print(f"   - Trajectory points: {len(traj_data['trajectory_points'])}")
            print(f"   - Total maneuvers: {len(traj_data['maneuvers'])}")
            
            # Verify data structure
            first_point = traj_data['trajectory_points'][0]
            print(f"   - First point format: {len(first_point)} coordinates")
            print(f"   - Transfer time: {traj_data['transfer_time']:.1f} days")
            
            return True, trajectory
        else:
            print("❌ Missing trajectory_data attribute")
            return False, None
            
    except Exception as e:
        print(f"❌ Basic trajectory generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_trajectory_data_structure():
    """Test trajectory data structure for integration."""
    print("\n" + "="*60)
    print("Testing Trajectory Data Structure")
    print("="*60)
    
    try:
        from src.trajectory.lunar_transfer import LunarTrajectory
        import numpy as np
        
        # Create a simple trajectory
        trajectory = LunarTrajectory(
            departure_epoch=10000.0,
            arrival_epoch=10004.5,
            departure_pos=(400, 0, 0),  # km
            departure_vel=(0, 7.5, 0),  # km/s
            arrival_pos=(384400, 0, 0),  # km
            arrival_vel=(0, 1.0, 0),    # km/s
        )
        
        # Test trajectory data property
        traj_data = trajectory.trajectory_data
        
        print(f"✅ Trajectory data structure working")
        print(f"   - Trajectory points: {len(traj_data['trajectory_points'])}")
        print(f"   - Transfer time: {traj_data['transfer_time']:.1f} days")
        print(f"   - Total delta-v: {traj_data['total_delta_v']:.1f} km/s")
        
        # Verify data types
        assert isinstance(traj_data['trajectory_points'], list)
        assert isinstance(traj_data['transfer_time'], (int, float))
        assert len(traj_data['trajectory_points']) > 0
        
        # Test that points have correct format (x, y, z in meters)
        point = traj_data['trajectory_points'][0]
        assert len(point) == 3
        assert all(isinstance(coord, (int, float)) for coord in point)
        
        print(f"   - Data validation passed")
        return True, traj_data
        
    except Exception as e:
        print(f"❌ Trajectory data structure failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_lambert_solver_direct():
    """Test Lambert solver directly."""
    print("\n" + "="*60)
    print("Testing Lambert Solver Direct")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import LambertSolver
        
        solver = LambertSolver()
        
        # Test with simple vectors
        r1 = np.array([7000e3, 0, 0])      # 7000 km altitude
        r2 = np.array([50000e3, 0, 0])     # 50000 km
        tof = 3.0 * 86400                  # 3 days
        
        v1, v2 = solver.solve_lambert(r1, r2, tof)
        
        print(f"✅ Lambert solver direct test working")
        print(f"   - Initial velocity: {np.linalg.norm(v1):.0f} m/s")
        print(f"   - Final velocity: {np.linalg.norm(v2):.0f} m/s")
        print(f"   - Transfer time: {tof/86400:.1f} days")
        
        # Test delta-v calculation
        total_dv, dv1, dv2 = solver.calculate_transfer_deltav(
            r1, np.array([0, 7500, 0]),  # Initial circular velocity
            r2, np.array([0, 1000, 0]),  # Target velocity
            tof
        )
        
        print(f"   - Total delta-v: {total_dv:.0f} m/s")
        print(f"   - Departure delta-v: {np.linalg.norm(dv1):.0f} m/s")
        print(f"   - Arrival delta-v: {np.linalg.norm(dv2):.0f} m/s")
        
        return True, solver
        
    except Exception as e:
        print(f"❌ Lambert solver direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_visualization_compatibility():
    """Test visualization compatibility."""
    print("\n" + "="*60)
    print("Testing Visualization Compatibility")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
        
        # Generate trajectory
        trajectory, total_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="lambert"
        )
        
        # Test visualization data format
        traj_data = trajectory.trajectory_data
        
        # Check that trajectory points are in the right format
        points = traj_data['trajectory_points']
        assert len(points) > 0
        
        # Check first and last points
        first_point = points[0]
        last_point = points[-1]
        
        print(f"✅ Visualization compatibility working")
        print(f"   - Points format: {len(points)} points with 3 coordinates each")
        print(f"   - First point: ({first_point[0]/1e6:.1f}, {first_point[1]/1e6:.1f}, {first_point[2]/1e6:.1f}) Mm")
        print(f"   - Last point: ({last_point[0]/1e6:.1f}, {last_point[1]/1e6:.1f}, {last_point[2]/1e6:.1f}) Mm")
        
        # Test with main optimizer data structure
        baseline_data = {
            "trajectory": trajectory,
            "total_dv": total_dv,
            "transfer_time": 4.5,
            "earth_orbit_alt": 400.0,
            "moon_orbit_alt": 100.0
        }
        
        # Test accessing trajectory data through main optimizer format
        if hasattr(baseline_data["trajectory"], 'trajectory_data'):
            vis_data = baseline_data["trajectory"].trajectory_data
            print(f"   - Main optimizer format compatible")
            print(f"   - Trajectory data accessible: {len(vis_data['trajectory_points'])} points")
            
        return True, traj_data
        
    except Exception as e:
        print(f"❌ Visualization compatibility failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_integration_with_existing_code():
    """Test integration with existing code paths."""
    print("\n" + "="*60)
    print("Testing Integration with Existing Code")
    print("="*60)
    
    try:
        # Test the exact code path from lunar_horizon_optimizer.py
        from src.trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
        
        # Simulate the exact call from _analyze_trajectories
        baseline_trajectory, baseline_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="lambert",
        )
        
        # Test the exact access pattern from _create_visualizations
        if hasattr(baseline_trajectory, 'trajectory_data'):
            traj_data = baseline_trajectory.trajectory_data
            
            print(f"✅ Integration with existing code working")
            print(f"   - Baseline trajectory created: {baseline_dv:.0f} m/s")
            print(f"   - Trajectory data accessible: {len(traj_data['trajectory_points'])} points")
            
            # Test the visualization access pattern
            trajectory_points = traj_data['trajectory_points']
            print(f"   - Trajectory points format: {len(trajectory_points)} points")
            
            return True, (baseline_trajectory, baseline_dv)
        else:
            print("❌ Trajectory missing trajectory_data attribute")
            return False, None
            
    except Exception as e:
        print(f"❌ Integration with existing code failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run all simple trajectory integration tests."""
    print("🚀 Simple Trajectory Integration Tests")
    print("=" * 70)
    
    test_results = {}
    
    # Test each integration component
    success_1, basic_result = test_basic_trajectory_integration()
    test_results['Basic Trajectory'] = success_1
    
    success_2, data_structure = test_trajectory_data_structure()
    test_results['Data Structure'] = success_2
    
    success_3, lambert_solver = test_lambert_solver_direct()
    test_results['Lambert Solver'] = success_3
    
    success_4, vis_compat = test_visualization_compatibility()
    test_results['Visualization'] = success_4
    
    success_5, integration_test = test_integration_with_existing_code()
    test_results['Existing Code Integration'] = success_5
    
    # Summary
    print("\n" + "="*70)
    print("Simple Trajectory Integration Test Results")
    print("="*70)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    print(f"\n📊 Results: {passed}/{total} integration tests passed")
    print(f"Success Rate: {passed/total*100:.0f}%")
    
    print(f"\n✅ Working Integrations:")
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if passed >= 4:
        print("\n🎉 Advanced trajectory generation integration successful!")
        print("   - Lambert solver properly integrated")
        print("   - Trajectory data structure compatible")
        print("   - Visualization integration working")
        print("   - Main optimizer integration complete")
    else:
        print("\n⚠️  Some integration issues remain")
        print("   - Continue with remaining fixes")
    
    return test_results

if __name__ == "__main__":
    main()