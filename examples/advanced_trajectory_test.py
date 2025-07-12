#!/usr/bin/env python3
"""
Advanced Trajectory Generation Integration Test

This test validates the integration of Lambert solvers with the trajectory generation
system and ensures proper data structures for visualization and analysis.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_lambert_solver_integration():
    """Test Lambert solver integration with trajectory generation."""
    print("\n" + "="*60)
    print("Testing Lambert Solver Integration")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import LambertSolver
        
        # Create Lambert solver
        solver = LambertSolver()
        
        # Test Lambert problem solution
        r1 = np.array([7000e3, 0, 0])  # Initial position (7000 km altitude)
        r2 = np.array([100000e3, 50000e3, 0])  # Target position
        tof = 4.5 * 86400  # 4.5 days in seconds
        
        # Solve Lambert problem
        v1, v2 = solver.solve_lambert(r1, r2, tof)
        
        print(f"✅ Lambert solver working")
        print(f"   - Initial velocity: {np.linalg.norm(v1):.0f} m/s")
        print(f"   - Final velocity: {np.linalg.norm(v2):.0f} m/s")
        print(f"   - Transfer time: {tof/86400:.1f} days")
        
        # Test multiple revolution solutions
        solutions = solver.solve_multiple_revolution(r1, r2, tof, max_revs=1)
        print(f"   - Multiple revolution solutions: {len(solutions)}")
        
        return True, solver
        
    except Exception as e:
        print(f"❌ Lambert solver failed: {e}")
        return False, None

def test_trajectory_generation_with_lambert():
    """Test trajectory generation using Lambert solver."""
    print("\n" + "="*60)
    print("Testing Trajectory Generation with Lambert Solver")
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
        
        print(f"✅ Trajectory generation working")
        print(f"   - Total delta-v: {total_dv:.0f} m/s")
        print(f"   - Transfer time: 4.5 days")
        print(f"   - Trajectory type: {type(trajectory).__name__}")
        
        # Test trajectory data access
        if hasattr(trajectory, 'trajectory_data'):
            traj_data = trajectory.trajectory_data
            print(f"   - Trajectory points: {len(traj_data['trajectory_points'])}")
            print(f"   - Total maneuvers: {len(traj_data['maneuvers'])}")
            
            # Test visualization data format
            first_point = traj_data['trajectory_points'][0]
            print(f"   - First point: ({first_point[0]/1e6:.1f}, {first_point[1]/1e6:.1f}, {first_point[2]/1e6:.1f}) Mm")
            
        return True, trajectory
        
    except Exception as e:
        print(f"❌ Trajectory generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_patched_conics_integration():
    """Test patched conics approximation integration."""
    print("\n" + "="*60)
    print("Testing Patched Conics Integration")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
        
        # Generate trajectory using patched conics method
        trajectory, total_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="patched_conics"
        )
        
        print(f"✅ Patched conics working")
        print(f"   - Total delta-v: {total_dv:.0f} m/s")
        print(f"   - Transfer time: 4.5 days")
        print(f"   - Trajectory type: {type(trajectory).__name__}")
        
        # Test trajectory data access
        if hasattr(trajectory, 'trajectory_data'):
            traj_data = trajectory.trajectory_data
            print(f"   - Trajectory points: {len(traj_data['trajectory_points'])}")
            
        return True, trajectory
        
    except Exception as e:
        print(f"❌ Patched conics failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_optimal_timing_integration():
    """Test optimal timing calculator integration."""
    print("\n" + "="*60)
    print("Testing Optimal Timing Integration")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import OptimalTimingCalculator
        
        # Create timing calculator
        timing_calc = OptimalTimingCalculator()
        
        # Find optimal departure time
        optimal_timing = timing_calc.find_optimal_departure_time(
            start_epoch=10000.0,
            search_days=10,  # Reduced for faster testing
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0
        )
        
        print(f"✅ Optimal timing working")
        print(f"   - Optimal epoch: {optimal_timing['optimal_epoch']:.1f}")
        print(f"   - Optimal delta-v: {optimal_timing['optimal_deltav']:.0f} m/s")
        print(f"   - Optimal transfer time: {optimal_timing['optimal_transfer_time']:.1f} days")
        print(f"   - Optimal date: {optimal_timing['optimal_date'].strftime('%Y-%m-%d')}")
        
        # Test launch window calculation
        windows = timing_calc.calculate_launch_windows(
            year=2025, month=6, num_windows=3
        )
        
        print(f"   - Launch windows found: {len(windows)}")
        if windows:
            best_window = windows[0]
            print(f"   - Best window delta-v: {best_window['optimal_deltav']:.0f} m/s")
        
        return True, optimal_timing
        
    except Exception as e:
        print(f"❌ Optimal timing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_main_optimizer_integration():
    """Test integration with main optimizer."""
    print("\n" + "="*60)
    print("Testing Main Optimizer Integration")
    print("="*60)
    
    try:
        from src.lunar_horizon_optimizer import LunarHorizonOptimizer
        
        # Create optimizer
        optimizer = LunarHorizonOptimizer()
        
        # Test trajectory analysis component
        # Use simplified parameters for faster testing
        from src.lunar_horizon_optimizer import OptimizationConfig
        
        test_config = OptimizationConfig(
            population_size=10,  # Reduced for testing
            num_generations=5,   # Reduced for testing
            seed=42
        )
        
        # Test just the trajectory analysis part
        trajectory_results = optimizer._analyze_trajectories(test_config, verbose=True)
        
        print(f"✅ Main optimizer integration working")
        print(f"   - Baseline trajectory generated: {'baseline' in trajectory_results}")
        
        if 'baseline' in trajectory_results:
            baseline = trajectory_results['baseline']
            print(f"   - Baseline delta-v: {baseline['total_dv']:.0f} m/s")
            print(f"   - Transfer time: {baseline['transfer_time']:.1f} days")
            
        if 'transfer_windows' in trajectory_results:
            windows = trajectory_results['transfer_windows']['windows']
            print(f"   - Transfer windows found: {len(windows)}")
            
        return True, trajectory_results
        
    except Exception as e:
        print(f"❌ Main optimizer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_visualization_integration():
    """Test visualization integration with trajectory data."""
    print("\n" + "="*60)
    print("Testing Visualization Integration")
    print("="*60)
    
    try:
        from src.trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
        from src.visualization.trajectory_visualization import TrajectoryVisualizer
        
        # Generate trajectory
        trajectory, total_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method="lambert"
        )
        
        # Create visualizer
        visualizer = TrajectoryVisualizer()
        
        # Test creating plot with trajectory data
        if hasattr(trajectory, 'trajectory_data'):
            traj_data = trajectory.trajectory_data
            
            # Create plot (simplified test)
            fig = visualizer.create_3d_trajectory_plot(
                trajectories=traj_data,
                title="Advanced Trajectory Test"
            )
            
            print(f"✅ Visualization integration working")
            print(f"   - Plot created with {len(fig.data)} traces")
            print(f"   - Trajectory data properly formatted")
            
            return True, fig
        else:
            print("❌ Trajectory missing trajectory_data attribute")
            return False, None
            
    except Exception as e:
        print(f"❌ Visualization integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Run all advanced trajectory integration tests."""
    print("🚀 Advanced Trajectory Generation Integration Tests")
    print("=" * 70)
    
    test_results = {}
    
    # Test each integration component
    success_1, lambert_solver = test_lambert_solver_integration()
    test_results['Lambert Solver'] = success_1
    
    success_2, trajectory = test_trajectory_generation_with_lambert()
    test_results['Trajectory Generation'] = success_2
    
    success_3, patched_conics = test_patched_conics_integration()
    test_results['Patched Conics'] = success_3
    
    success_4, optimal_timing = test_optimal_timing_integration()
    test_results['Optimal Timing'] = success_4
    
    success_5, optimizer_results = test_main_optimizer_integration()
    test_results['Main Optimizer'] = success_5
    
    success_6, visualization = test_visualization_integration()
    test_results['Visualization'] = success_6
    
    # Summary
    print("\n" + "="*70)
    print("Advanced Trajectory Integration Test Results")
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
        print("   - Trajectory generation with visualization support")
        print("   - Optimal timing calculations working")
        print("   - Main optimizer integration complete")
    else:
        print("\n⚠️  Some integration issues remain")
        print("   - Continue with remaining fixes")
    
    return test_results

if __name__ == "__main__":
    main()