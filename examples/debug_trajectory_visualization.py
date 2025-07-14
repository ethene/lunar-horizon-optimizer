#!/usr/bin/env python3
"""
Debug script to check trajectory visualization issues.

This script investigates why the trajectory path is not visible in the 3D visualization.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.real_landing_3d_visualization import create_realistic_powered_descent_trajectory
from src.visualization.landing_3d_visualization import Landing3DVisualizer, LandingVisualizationConfig


def debug_trajectory_generation():
    """Debug the trajectory generation to see what's wrong."""
    print("🔍 Debugging trajectory generation...")
    
    # Use the same parameters as the scenario
    scenario_config = {
        'descent_parameters': {
            'thrust': 10000.0,
            'isp': 320.0,
            'burn_time': 31.4,
            'landing_site': 'Oceanus Procellarum'
        },
        'mission': {
            'name': 'Quick Lunar Descent Test'
        }
    }
    
    # Generate trajectory
    trajectory_data = create_realistic_powered_descent_trajectory(
        scenario_config=scenario_config,
        moon_orbit_altitude=100.0,  # 100 km
        optimization_results={}
    )
    
    positions = trajectory_data['positions']
    times = trajectory_data['time_points']
    
    print(f"\n📊 Trajectory Analysis:")
    print(f"   Number of points: {len(positions)}")
    print(f"   Time range: {times[0]:.1f} to {times[-1]:.1f} seconds")
    print(f"   Duration: {(times[-1] - times[0])/60:.1f} minutes")
    
    # Analyze position ranges
    x_range = [positions[:, 0].min(), positions[:, 0].max()]
    y_range = [positions[:, 1].min(), positions[:, 1].max()]  
    z_range = [positions[:, 2].min(), positions[:, 2].max()]
    
    print(f"\n📍 Position Ranges:")
    print(f"   X: {x_range[0]/1000:.1f} to {x_range[1]/1000:.1f} km")
    print(f"   Y: {y_range[0]/1000:.1f} to {y_range[1]/1000:.1f} km") 
    print(f"   Z: {z_range[0]/1000:.1f} to {z_range[1]/1000:.1f} km")
    
    # Check lunar radius
    lunar_radius = 1737.4e3  # meters
    print(f"\n🌙 Lunar Reference:")
    print(f"   Lunar radius: {lunar_radius/1000:.1f} km")
    print(f"   Surface Z: {lunar_radius/1000:.1f} km")
    
    # Calculate altitudes
    altitudes = []
    for pos in positions:
        altitude = np.linalg.norm(pos) - lunar_radius
        altitudes.append(altitude)
    
    print(f"\n🚁 Altitude Analysis:")
    print(f"   Start altitude: {altitudes[0]/1000:.1f} km")
    print(f"   End altitude: {altitudes[-1]/1000:.1f} km")
    print(f"   Min altitude: {min(altitudes)/1000:.1f} km")
    print(f"   Max altitude: {max(altitudes)/1000:.1f} km")
    
    # Check trajectory visibility
    trajectory_size = np.linalg.norm([x_range[1] - x_range[0], 
                                     y_range[1] - y_range[0],
                                     z_range[1] - z_range[0]])
    
    print(f"\n👁️ Visibility Analysis:")
    print(f"   Trajectory span: {trajectory_size/1000:.1f} km")
    print(f"   Moon diameter: {2*lunar_radius/1000:.1f} km")
    print(f"   Trajectory/Moon ratio: {trajectory_size/(2*lunar_radius)*100:.2f}%")
    
    if trajectory_size/(2*lunar_radius) < 0.1:
        print("   ⚠️  ISSUE: Trajectory is <10% of Moon size - may be too small to see!")
    
    return trajectory_data


def create_fixed_trajectory():
    """Create a more visible trajectory for debugging."""
    print("\n🔧 Creating fixed, more visible trajectory...")
    
    lunar_radius = 1737.4e3  # meters
    
    # Create a simple but visible descent trajectory
    n_points = 100
    times = np.linspace(0, 600, n_points)  # 10 minutes
    
    positions = np.zeros((n_points, 3))
    velocities = np.zeros((n_points, 3))
    thrust_profile = np.zeros(n_points)
    
    for i, t in enumerate(times):
        # Progress from 0 to 1
        progress = t / times[-1]
        
        # Start from 50km altitude, land at surface
        start_altitude = 50000  # 50 km
        current_altitude = start_altitude * (1 - progress)
        
        # Horizontal approach from 20km away
        horizontal_distance = 20000 * (1 - progress)  # 20 km approach
        
        positions[i] = np.array([
            horizontal_distance,
            0,
            lunar_radius + current_altitude
        ])
        
        # Simple velocity calculation
        if i > 0:
            dt = times[i] - times[i-1]
            velocities[i] = (positions[i] - positions[i-1]) / dt
        
        # Variable thrust
        thrust_profile[i] = 15000 * (0.5 + 0.5 * progress)  # Increasing thrust
    
    # Analysis
    trajectory_size = np.linalg.norm([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ])
    
    print(f"   Fixed trajectory span: {trajectory_size/1000:.1f} km")
    print(f"   Fixed trajectory/Moon ratio: {trajectory_size/(2*lunar_radius)*100:.2f}%")
    
    return {
        "positions": positions,
        "velocities": velocities,
        "time_points": times,
        "thrust_profile": thrust_profile
    }


def test_visualization_with_fixed_trajectory():
    """Test visualization with the fixed trajectory."""
    print("\n🎨 Testing visualization with fixed trajectory...")
    
    # Create fixed trajectory
    trajectory_data = create_fixed_trajectory()
    
    # Create enhanced visualization config
    config = LandingVisualizationConfig(
        width=1400,
        height=1000,
        title="DEBUG: Fixed Lunar Landing Trajectory",
        
        # Enhance visibility
        trajectory_width=8,  # Thicker line
        trajectory_color="#00ff00",
        show_thrust_vectors=True,
        thrust_vector_scale=0.01,  # Larger thrust vectors
        
        # Better initial view
        background_color="#000011",
        stars_background=True,
        star_count=200,  # Fewer stars for clarity
        
        # Lunar surface
        show_lunar_surface=True,
        lunar_surface_resolution=30,  # Lower resolution for performance
        lunar_surface_opacity=0.6,
        
        # Landing zone
        show_landing_zone=True,
        landing_zone_radius=5000,  # 5km radius for visibility
        
        theme="plotly_dark"
    )
    
    # Create visualizer
    visualizer = Landing3DVisualizer(config)
    
    # Generate visualization
    landing_site = (0, 0, 1737400)
    fig = visualizer.create_landing_trajectory_plot(
        trajectory_data=trajectory_data,
        landing_site=landing_site,
        title="DEBUG: Fixed Trajectory Test"
    )
    
    # Set better initial camera position
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=0.8, y=0.8, z=0.8),  # Closer initial view
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )
    )
    
    # Save visualization
    debug_file = project_root / "debug_fixed_trajectory.html"
    fig.write_html(str(debug_file))
    
    print(f"✅ Debug visualization saved: {debug_file}")
    return str(debug_file)


def main():
    """Run trajectory debugging."""
    print("🐛 Trajectory Visualization Debug Tool")
    print("=" * 50)
    
    # Debug current trajectory
    original_trajectory = debug_trajectory_generation()
    
    # Create and test fixed trajectory
    fixed_viz_file = test_visualization_with_fixed_trajectory()
    
    print(f"\n🎯 Debug Summary:")
    print(f"   Original trajectory may be too small relative to Moon size")
    print(f"   Created fixed trajectory with better visibility")
    print(f"   Test file: {fixed_viz_file}")
    print(f"\n💡 Recommended fixes:")
    print(f"   1. Increase trajectory scale relative to Moon")
    print(f"   2. Improve initial camera positioning")
    print(f"   3. Use thicker trajectory lines")
    print(f"   4. Better color contrast")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())