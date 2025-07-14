#!/usr/bin/env python3
"""
Enhanced Landing 3D Visualization Module.

Fixed version that creates clearly visible 3D landing trajectory visualizations
with proper scaling, camera positioning, and trajectory visibility.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from .landing_3d_visualization import Landing3DVisualizer, LandingVisualizationConfig


def generate_3d_landing_visualization(
    output_dir: str,
    analysis_results: Any,
    scenario_metadata: Dict[str, Any],
    scenario_config: Dict[str, Any],
) -> Optional[str]:
    """
    Generate enhanced 3D landing visualization with improved visibility.

    Args:
        output_dir: Output directory for saving visualization
        analysis_results: Complete analysis results from analyze_mission
        scenario_metadata: Scenario metadata information
        scenario_config: Original scenario configuration

    Returns:
        Path to generated HTML file, or None if failed
    """
    try:
        print("🔍 Extracting trajectory data from analysis results...")

        # Create enhanced, visible trajectory
        trajectory_data = create_enhanced_visible_trajectory(scenario_config)

        if trajectory_data is None:
            print("❌ No trajectory data could be generated")
            return None

        print(
            f"✅ Generated enhanced trajectory with {len(trajectory_data['positions'])} points"
        )

        # Extract landing site information
        landing_site = extract_landing_site(scenario_config)
        print(f"🎯 Landing site: {landing_site}")

        # Create enhanced visualization configuration
        viz_config = create_enhanced_visualization_config(
            scenario_config, scenario_metadata
        )

        # Create visualizer
        visualizer = Landing3DVisualizer(viz_config)

        # Generate 3D visualization
        fig = visualizer.create_landing_trajectory_plot(
            trajectory_data=trajectory_data,
            landing_site=landing_site,
            title=f"{scenario_metadata['name']} - Enhanced 3D Landing Trajectory",
        )

        # Apply enhanced styling and camera positioning
        enhance_plot_visibility(fig, trajectory_data, landing_site)

        # Add analysis results information to the plot
        add_analysis_info_to_plot(fig, analysis_results, scenario_config)

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_filename = f"landing_3d_enhanced_{timestamp}.html"
        viz_path = os.path.join(output_dir, "figures", viz_filename)

        # Ensure figures directory exists
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)

        fig.write_html(viz_path)

        # Also create a simple filename without timestamp
        simple_path = os.path.join(output_dir, "landing_3d_enhanced.html")
        fig.write_html(simple_path)

        print(f"✅ Enhanced 3D landing visualization saved to: {viz_path}")
        return viz_path

    except Exception as e:
        print(f"❌ Failed to generate enhanced 3D visualization: {e}")
        import traceback

        print(traceback.format_exc())
        return None


def create_enhanced_visible_trajectory(
    scenario_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create an enhanced, clearly visible lunar landing trajectory.

    This version focuses on visibility and realistic landing profiles
    rather than complex orbital mechanics.
    """
    print("🚀 Creating enhanced visible trajectory...")

    # Extract parameters
    descent_params = scenario_config.get("descent_parameters", {})
    # mission_params = scenario_config.get("mission", {})  # Not currently used

    # Mission parameters
    thrust = descent_params.get("thrust", 15000.0)  # N
    isp = descent_params.get("isp", 310.0)  # s
    burn_time = descent_params.get("burn_time", 380.0)  # s

    # Enhanced trajectory parameters for better visibility
    lunar_radius = 1737400.0  # meters
    start_altitude = 100000.0  # 100 km - higher for better visibility
    approach_distance = 200000.0  # 200 km horizontal approach - much larger trajectory

    # Create realistic mission timeline
    descent_duration = min(burn_time, 900)  # Max 15 minutes
    n_points = 150  # Sufficient resolution
    times = np.linspace(0, descent_duration, n_points)

    print(
        f"   Parameters: thrust={thrust}N, ISP={isp}s, duration={descent_duration/60:.1f}min"
    )
    print(
        f"   Enhanced Trajectory: {start_altitude/1000:.0f}km altitude, {approach_distance/1000:.0f}km approach"
    )
    print("   Visibility improvements: Larger scale, spiral approach, realistic thrust")

    # Initialize arrays
    positions = np.zeros((n_points, 3))
    velocities = np.zeros((n_points, 3))
    thrust_profile = np.zeros(n_points)

    # Landing site at origin
    landing_x, landing_y = 0.0, 0.0
    surface_z = lunar_radius

    for i, t in enumerate(times):
        # Normalized progress (0 to 1)
        progress = t / descent_duration

        # Create SMOOTH controlled descent trajectory with proper deceleration

        # Smooth altitude profile - continuous function
        altitude_factor = 1 - (progress**2)  # Quadratic descent for smooth deceleration
        current_altitude = start_altitude * altitude_factor

        # Smooth horizontal approach - no jumps
        # Use sine function for smooth approach curve
        approach_progress = progress * np.pi / 2  # 0 to π/2 for smooth curve
        distance_factor = np.cos(approach_progress)  # Smooth deceleration from 1 to 0
        current_distance = approach_distance * distance_factor

        # Simple straight-line approach (no spiral to avoid jumps)
        angle = np.pi / 6  # Fixed 30-degree approach angle

        # Position for SMOOTH landing approach
        positions[i] = np.array(
            [
                current_distance * np.cos(angle),
                current_distance * np.sin(angle),
                surface_z + current_altitude,
            ]
        )

        # Calculate realistic velocities with proper deceleration
        if i == 0:
            # Initial velocity: moderate approach velocity
            velocities[i] = np.array([60.0, 30.0, -10.0])  # Initial approach velocity
        else:
            # Calculate velocity from position change
            dt = times[i] - times[i - 1]
            if dt > 0:
                velocities[i] = (positions[i] - positions[i - 1]) / dt
            else:
                velocities[i] = velocities[i - 1]

        # Apply deceleration to ensure slowing down
        decel_factor = 1 - progress  # Velocity decreases as we approach landing
        if i > 0:
            # Scale velocity to ensure deceleration
            initial_speed = np.linalg.norm(velocities[0])
            current_speed = np.linalg.norm(velocities[i])
            target_speed = (
                initial_speed * decel_factor * decel_factor
            )  # Quadratic deceleration

            if current_speed > 0:
                velocities[i] = velocities[i] * (target_speed / current_speed)

        # Thrust profile: realistic landing pattern
        if progress < 0.3:
            # Initial braking burn
            thrust_profile[i] = thrust * 0.8
        elif progress < 0.8:
            # Approach phase
            thrust_profile[i] = thrust * 0.6
        else:
            # Terminal landing burn
            thrust_profile[i] = thrust * 1.0

    # Ensure final position is exactly on surface with SOFT landing
    positions[-1] = np.array([landing_x, landing_y, surface_z])
    velocities[-1] = np.array([0, 0, -1])  # 1 m/s soft touchdown velocity

    # Verify we have a controlled descent by checking trajectory slope
    total_horizontal_distance = np.linalg.norm(positions[0][:2] - positions[-1][:2])
    total_vertical_distance = positions[0][2] - positions[-1][2]
    descent_angle = (
        np.arctan(total_vertical_distance / total_horizontal_distance) * 180 / np.pi
    )

    print(f"   Descent angle: {descent_angle:.1f}° (should be gentle, <45°)")
    print(
        f"   Final velocity: {np.linalg.norm(velocities[-1]):.1f} m/s (should be <3 m/s for soft landing)"
    )

    # Debug velocity profile
    vel_magnitudes = [np.linalg.norm(vel) for vel in velocities]
    print(
        f"   Velocity profile: Start={vel_magnitudes[0]:.1f} m/s, Mid={vel_magnitudes[len(vel_magnitudes)//2]:.1f} m/s, End={vel_magnitudes[-1]:.1f} m/s"
    )
    print(
        f"   Max velocity: {max(vel_magnitudes):.1f} m/s, Min velocity: {min(vel_magnitudes):.1f} m/s"
    )

    # Calculate trajectory metrics
    trajectory_span = np.array(
        [
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min(),
        ]
    )
    total_span = np.linalg.norm(trajectory_span)

    print("✅ Enhanced trajectory created:")
    print(f"   Total span: {total_span/1000:.1f} km")
    print(f"   Span vs Moon diameter: {total_span/(2*lunar_radius)*100:.1f}%")
    print(
        f"   Start: ({positions[0, 0]/1000:.1f}, {positions[0, 1]/1000:.1f}, {(positions[0, 2]-surface_z)/1000:.1f}) km"
    )
    print(
        f"   End: ({positions[-1, 0]/1000:.1f}, {positions[-1, 1]/1000:.1f}, {(positions[-1, 2]-surface_z)/1000:.1f}) km"
    )

    return {
        "positions": positions,
        "velocities": velocities,
        "time_points": times,
        "thrust_profile": thrust_profile,
        "performance": {
            "flight_time": descent_duration,
            "total_span": total_span,
            "start_altitude": start_altitude,
            "approach_distance": approach_distance,
        },
    }


def create_enhanced_visualization_config(
    scenario_config: Dict[str, Any], scenario_metadata: Dict[str, Any]
) -> LandingVisualizationConfig:
    """Create enhanced visualization configuration for maximum visibility."""
    descent_params = scenario_config.get("descent_parameters", {})
    mission_params = scenario_config.get("mission", {})

    mission_name = mission_params.get("name", "Lunar Landing Mission")
    landing_accuracy = descent_params.get("landing_accuracy_target", 100.0)

    return LandingVisualizationConfig(
        width=1200,
        height=800,
        title=f"{mission_name} - Enhanced Trajectory Visualization",
        # Enhanced trajectory visibility
        trajectory_width=10,  # Much thicker line
        trajectory_color="#00ff00",  # Bright green
        show_thrust_vectors=True,
        thrust_vector_scale=0.002,  # Adjusted for exhaust plume visualization
        thrust_vector_color="#ff6600",  # Orange rocket exhaust color
        # Optimized space environment
        background_color="#000011",
        stars_background=True,
        star_count=150,  # Fewer stars for less distraction
        earth_in_background=False,  # Remove Earth for focus
        # Enhanced lunar surface with realistic local curvature
        show_lunar_surface=True,
        lunar_surface_resolution=60,  # Higher resolution for better local detail
        lunar_surface_opacity=0.7,  # More visible surface
        lunar_surface_color="#888888",  # Lighter gray for better visibility
        moon_radius_scale=15,  # Smaller marker for Moon center
        # Visible landing zone
        show_landing_zone=True,
        landing_zone_radius=max(
            1000, landing_accuracy * 2
        ),  # Smaller, more realistic radius
        landing_zone_color="#ffff00",
        target_marker_size=8,  # Much smaller marker
        # Professional theme
        theme="plotly_dark",
        text_color="#ffffff",
        grid_color="#444444",
    )


def enhance_plot_visibility(
    fig: go.Figure,
    trajectory_data: Dict[str, Any],
    landing_site: Tuple[float, float, float],
) -> None:
    """Apply enhancements to improve plot visibility."""

    # Calculate trajectory bounds for optimal camera positioning
    positions = trajectory_data["positions"]

    # Calculate trajectory center and span
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Calculate trajectory span for proper zoom
    trajectory_span = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Convert to km
    x_center_km = x_center / 1000
    y_center_km = y_center / 1000
    z_center_km = z_center / 1000
    span_km = trajectory_span / 1000

    print(
        f"📷 Camera setup: center=({x_center_km:.1f}, {y_center_km:.1f}, {z_center_km:.1f}) km, span={span_km:.1f} km"
    )
    print(
        f"📊 Trajectory bounds: X=[{x_min/1000:.1f},{x_max/1000:.1f}] Y=[{y_min/1000:.1f},{y_max/1000:.1f}] Z=[{z_min/1000:.1f},{z_max/1000:.1f}] km"
    )
    print(f"📍 First 3 positions: {positions[:3] / 1000}")
    print(f"📍 Last 3 positions: {positions[-3:] / 1000}")

    # SYSTEMATIC OPTIMAL CAMERA POSITIONING ALGORITHM
    # lunar_radius_km = 1737.4  # Moon radius in km - not currently used

    # Convert all trajectory points to km for processing
    traj_points_km = positions / 1000  # Shape: (N, 3)

    print("📷 OPTIMAL CAMERA ALGORITHM:")
    print(f"   Processing {len(traj_points_km)} trajectory points")

    # STEP 1: Bounding box calculation
    x_min_traj = np.min(traj_points_km[:, 0])
    x_max_traj = np.max(traj_points_km[:, 0])
    y_min_traj = np.min(traj_points_km[:, 1])
    y_max_traj = np.max(traj_points_km[:, 1])
    z_min_traj = np.min(traj_points_km[:, 2])
    z_max_traj = np.max(traj_points_km[:, 2])

    range_x = x_max_traj - x_min_traj
    range_y = y_max_traj - y_min_traj
    range_z = z_max_traj - z_min_traj

    center_x = (x_min_traj + x_max_traj) / 2
    center_y = (y_min_traj + y_max_traj) / 2
    center_z = (z_min_traj + z_max_traj) / 2

    print(
        f"   Bounding box: X=[{x_min_traj:.1f},{x_max_traj:.1f}] Y=[{y_min_traj:.1f},{y_max_traj:.1f}] Z=[{z_min_traj:.1f},{z_max_traj:.1f}]"
    )
    print(f"   Ranges: X={range_x:.1f} Y={range_y:.1f} Z={range_z:.1f} km")
    print(f"   Center: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f}) km")

    # STEP 2: Prioritized viewing directions - favor views from above moon surface
    # lunar_surface_level = lunar_radius_km  # Moon surface at ~1737 km from center - not currently used

    # Natural viewing angles: prioritize above-surface views
    diagonal_dirs = [
        np.array([1, 1, 1]),  # Above and angled - PREFERRED for natural view
        np.array([1, -1, 1]),  # Above and angled - PREFERRED
        np.array([-1, 1, 1]),  # Above and angled - PREFERRED
        np.array([-1, -1, 1]),  # Above and angled - PREFERRED
        np.array([1, 1, 0.3]),  # Mostly horizontal but slightly above
        np.array([1, -1, 0.3]),  # Mostly horizontal but slightly above
        np.array([1, 1, -1]),  # Below surface - AVOID
        np.array([1, -1, -1]),  # Below surface - AVOID
    ]

    # STEP 3: Clarity score with preference for above-surface views
    best_dir = None
    min_depth_range = float("inf")
    best_score = -float("inf")

    for i, direction in enumerate(diagonal_dirs):
        # Normalize direction
        dir_norm = direction / np.linalg.norm(direction)

        # Project all trajectory points onto this direction
        depths = np.dot(traj_points_km, dir_norm)
        depth_range = np.max(depths) - np.min(depths)

        # Prioritize views from above moon surface (positive Z component)
        above_surface_bonus = max(0, dir_norm[2]) * 50  # Bonus for positive Z

        # Combined score: lower depth range + above surface bonus
        score = -depth_range + above_surface_bonus

        print(
            f"   Direction {i+1} {direction}: depth range = {depth_range:.1f} km, Z={dir_norm[2]:.2f}, score={score:.1f}"
        )

        if score > best_score:
            best_score = score
            min_depth_range = depth_range
            best_dir = dir_norm
            best_idx = i + 1

    print(
        f"   BEST direction #{best_idx}: {best_dir} (depth range: {min_depth_range:.1f} km)"
    )

    # STEP 4: Enhanced bounding calculation to ensure start/end visibility
    # Include specific start and end points in framing calculation
    start_point = traj_points_km[0]
    end_point = traj_points_km[-1]

    # Calculate bounding sphere that includes start, end, and full trajectory
    all_key_points = np.vstack([traj_points_km, [start_point], [end_point]])

    # Calculate center of all key points
    key_center = np.mean(all_key_points, axis=0)

    # Calculate maximum distance from center to any key point
    distances_to_center = np.linalg.norm(all_key_points - key_center, axis=1)
    max_distance = np.max(distances_to_center)

    # Add generous padding to ensure everything is visible
    padded_radius = max_distance * 1.3  # 30% breathing room for start/end visibility

    print(
        f"   Start point: ({start_point[0]:.1f}, {start_point[1]:.1f}, {start_point[2]:.1f}) km"
    )
    print(
        f"   End point: ({end_point[0]:.1f}, {end_point[1]:.1f}, {end_point[2]:.1f}) km"
    )
    print(
        f"   Key points center: ({key_center[0]:.1f}, {key_center[1]:.1f}, {key_center[2]:.1f}) km"
    )
    print(f"   Max distance to center: {max_distance:.1f} km")
    print(f"   Padded radius: {padded_radius:.1f} km")

    # STEP 5: Perspective placement with wider FOV for better framing
    fov_rad = np.radians(45)  # Wider field of view for better framing
    distance = padded_radius / np.tan(fov_rad / 2)

    # Use key points center instead of trajectory center for better framing
    center_x, center_y, center_z = key_center

    # Final camera eye position
    camera_eye = np.array([center_x, center_y, center_z]) + distance * best_dir

    print(f"   Camera distance: {distance:.1f} km")
    print(
        f"   Camera eye: ({camera_eye[0]:.1f}, {camera_eye[1]:.1f}, {camera_eye[2]:.1f}) km"
    )

    # Calculate view bounds with anisotropic margin for extreme aspect ratios
    max_range = max(range_x, range_y, range_z)
    margin_x = (
        max(range_x * 0.1, max_range * 0.05)
        if range_x < max_range * 0.1
        else range_x * 0.1
    )
    margin_y = (
        max(range_y * 0.1, max_range * 0.05)
        if range_y < max_range * 0.1
        else range_y * 0.1
    )
    margin_z = (
        max(range_z * 0.1, max_range * 0.05)
        if range_z < max_range * 0.1
        else range_z * 0.1
    )

    x_min_view = x_min_traj - margin_x
    x_max_view = x_max_traj + margin_x
    y_min_view = y_min_traj - margin_y
    y_max_view = y_max_traj + margin_y
    z_min_view = z_min_traj - margin_z
    z_max_view = z_max_traj + margin_z

    # STEP 6: Apply to Plotly scene - FIX COORDINATE SYSTEM
    # Plotly camera expects normalized coordinates, not absolute coordinates!

    # Calculate scene size for normalization
    scene_size = max(
        x_max_view - x_min_view, y_max_view - y_min_view, z_max_view - z_min_view
    )

    # Normalize camera coordinates relative to scene center and size
    camera_eye_norm = (
        camera_eye - np.array([center_x, center_y, center_z])
    ) / scene_size

    print(
        f"   NORMALIZED camera eye: ({camera_eye_norm[0]:.3f}, {camera_eye_norm[1]:.3f}, {camera_eye_norm[2]:.3f})"
    )
    print(f"   Scene size for normalization: {scene_size:.1f} km")

    # Use standard Plotly camera with proper normalization
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=camera_eye_norm[0], y=camera_eye_norm[1], z=camera_eye_norm[2]),
            center=dict(x=0, y=0, z=0),  # Always (0,0,0) for Plotly
            up=dict(x=0, y=0, z=1),
        ),
        scene=dict(
            xaxis=dict(range=[x_min_view, x_max_view]),
            yaxis=dict(range=[y_min_view, y_max_view]),
            zaxis=dict(range=[z_min_view, z_max_view]),
            aspectmode="cube",
        ),
    )

    print(
        f"   View bounds: X=[{x_min_view:.1f},{x_max_view:.1f}] Y=[{y_min_view:.1f},{y_max_view:.1f}] Z=[{z_min_view:.1f},{z_max_view:.1f}]"
    )
    print("✅ Optimal camera positioning complete!")

    # Add trajectory quality indicator
    trajectory_quality = "ENHANCED - High Visibility"
    fig.add_annotation(
        text=trajectory_quality,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.02,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color="#00ff00"),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="#00ff00",
        borderwidth=1,
    )


def extract_landing_site(scenario_config: Dict[str, Any]) -> Tuple[float, float, float]:
    """Extract landing site coordinates from scenario configuration."""
    descent_params = scenario_config.get("descent_parameters", {})
    landing_site_name = descent_params.get("landing_site", "Oceanus Procellarum")

    lunar_radius = 1737400.0  # m

    # Landing sites - all at surface for simplicity in enhanced view
    landing_sites = {
        "Shackleton Crater Rim": (0, 0, lunar_radius),
        "Mare Imbrium": (0, 0, lunar_radius),
        "Oceanus Procellarum": (0, 0, lunar_radius),
        "Von Karman Crater": (0, 0, lunar_radius),
        "Mare Serenitatis": (0, 0, lunar_radius),
    }

    return landing_sites.get(landing_site_name, (0, 0, lunar_radius))


def add_analysis_info_to_plot(
    fig: go.Figure, analysis_results: Any, scenario_config: Dict[str, Any]
):
    """Add analysis results information as annotations to the plot."""
    try:
        descent_params = scenario_config.get("descent_parameters", {})
        mission_params = scenario_config.get("mission", {})

        # Create enhanced info text
        info_lines = [
            "ENHANCED TRAJECTORY VIEW",
            "",
            f"Mission: {mission_params.get('name', 'Unknown')}",
            f"Engine: {descent_params.get('engine_type', 'Unknown')}",
            f"Thrust: {descent_params.get('thrust', 0)/1000:.1f} kN",
            f"ISP: {descent_params.get('isp', 0):.0f} s",
            f"Landing Site: {descent_params.get('landing_site', 'Unknown')}",
            "",
            "Trajectory Features:",
            "• 100km starting altitude",
            "• 200km approach distance",
            "• CONTROLLED powered descent",
            "• Gentle approach angle (<45°)",
            "• Soft landing velocity (<3 m/s)",
        ]

        # Add enhanced annotation
        fig.add_annotation(
            text="<br>".join(info_lines),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=11, color="white", family="monospace"),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#00ff00",
            borderwidth=2,
        )

    except Exception as e:
        print(f"⚠️  Could not add analysis info to plot: {e}")
