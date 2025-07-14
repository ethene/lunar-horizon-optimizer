"""
Enhanced 3D Landing Trajectory Visualization Module.

Provides specialized 3D visualization capabilities for lunar landing trajectories,
building on the existing trajectory visualization framework with enhanced features
for powered descent, lunar surface rendering, and landing zone visualization.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import plotly.express as px  # Not currently used

from .trajectory_visualization import TrajectoryVisualizer, TrajectoryPlotConfig


@dataclass
class LandingVisualizationConfig(TrajectoryPlotConfig):
    """Extended configuration for landing trajectory visualization."""

    # Lunar surface rendering
    show_lunar_surface: bool = True
    lunar_surface_resolution: int = 50
    lunar_surface_color: str = "#666666"
    lunar_surface_opacity: float = 0.3

    # Landing zone visualization
    show_landing_zone: bool = True
    landing_zone_radius: float = 1000.0  # meters
    landing_zone_color: str = "#ffff00"
    target_marker_size: int = 15

    # Powered descent specific
    show_thrust_vectors: bool = True
    thrust_vector_scale: float = 0.001  # Scale factor for thrust magnitude
    thrust_vector_color: str = "#ff6600"
    show_velocity_profile: bool = True

    # Animation and timeline
    enable_descent_animation: bool = True
    animation_speed: float = 1.0  # Speed multiplier
    show_altitude_profile: bool = True

    # Enhanced styling for space environment
    stars_background: bool = True
    star_count: int = 500
    earth_in_background: bool = True
    earth_position_scale: float = 0.1  # Scale factor for Earth distance


class Landing3DVisualizer(TrajectoryVisualizer):
    """
    Enhanced 3D visualizer for lunar landing trajectories.

    Extends the base TrajectoryVisualizer with specialized features for
    powered descent visualization including lunar surface rendering,
    landing zone visualization, and thrust vector display.
    """

    def __init__(self, config: Optional[LandingVisualizationConfig] = None) -> None:
        """
        Initialize landing trajectory visualizer.

        Args:
            config: Landing visualization configuration
        """
        self.landing_config = config or LandingVisualizationConfig()
        super().__init__(self.landing_config)

        # Lunar-specific constants
        self.lunar_radius = 1737.4e3  # meters
        self.lunar_surface_gravity = 1.62  # m/s^2

    def create_landing_trajectory_plot(
        self,
        trajectory_data: Dict[str, Any],
        landing_site: Optional[Tuple[float, float, float]] = None,
        title: str = "Lunar Landing Trajectory",
    ) -> go.Figure:
        """
        Create comprehensive 3D landing trajectory visualization.

        Args:
            trajectory_data: Trajectory data with positions, velocities, thrusts
            landing_site: Target landing coordinates (x, y, z) in meters
            title: Plot title

        Returns:
            Plotly Figure with enhanced 3D landing visualization
        """
        fig = go.Figure()

        # Add space environment background
        if self.landing_config.stars_background:
            self._add_star_field(fig)

        # Add Earth in background if enabled
        if self.landing_config.earth_in_background:
            self._add_earth_background(fig)

        # Add lunar surface
        if self.landing_config.show_lunar_surface:
            self._add_lunar_surface(fig)

        # Add Moon body
        self._add_moon_body(fig)

        # Add landing zone if specified
        if landing_site and self.landing_config.show_landing_zone:
            self._add_landing_zone(fig, landing_site)

        # Add main trajectory path
        print(
            f"🛤️  Adding trajectory path with {len(trajectory_data.get('positions', []))} points"
        )
        self._add_landing_trajectory_path(fig, trajectory_data)
        print(f"📊 Figure now has {len(fig.data)} total traces")

        # Add thrust vectors if available
        if (
            self.landing_config.show_thrust_vectors
            and "thrust_profile" in trajectory_data
        ):
            self._add_thrust_vectors(fig, trajectory_data)

        # Add velocity indicators along trajectory
        if "velocities" in trajectory_data:
            self._add_velocity_indicators(fig, trajectory_data)

        # Configure enhanced 3D layout
        self._configure_landing_layout(fig, title)

        return fig

    def create_descent_analysis_dashboard(
        self,
        trajectory_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        landing_site: Optional[Tuple[float, float, float]] = None,
    ) -> go.Figure:
        """
        Create comprehensive descent analysis dashboard.

        Args:
            trajectory_data: Complete trajectory data
            performance_metrics: Performance and cost metrics
            landing_site: Target landing coordinates

        Returns:
            Multi-panel dashboard with 3D visualization and analysis plots
        """
        # Create subplot layout
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene", "colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
            ],
            subplot_titles=[
                "3D Landing Trajectory",
                "Altitude vs Time",
                "Velocity Profile",
            ],
            vertical_spacing=0.1,
        )

        # Main 3D plot (top panel)
        landing_fig = self.create_landing_trajectory_plot(
            trajectory_data, landing_site, "3D Landing Trajectory"
        )

        # Add 3D traces to subplot
        for trace in landing_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Add altitude profile (bottom left)
        if "positions" in trajectory_data and "time_points" in trajectory_data:
            positions = trajectory_data["positions"]
            times = trajectory_data["time_points"]

            # Calculate altitudes
            altitudes = []
            for pos in positions:
                altitude = np.linalg.norm(pos) - self.lunar_radius
                altitudes.append(altitude / 1000)  # Convert to km

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=altitudes,
                    mode="lines",
                    name="Altitude",
                    line={"color": "#00ff00", "width": 3},
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Add velocity profile (bottom right)
        if "velocities" in trajectory_data:
            velocities = trajectory_data["velocities"]

            # Calculate velocity magnitudes
            vel_magnitudes = [np.linalg.norm(vel) for vel in velocities]

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=vel_magnitudes,
                    mode="lines",
                    name="Velocity Magnitude",
                    line={"color": "#ff6600", "width": 3},
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Altitude (km)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)

        # Apply 3D scene configuration from landing plot
        fig.update_scenes(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            bgcolor=self.landing_config.background_color,
            xaxis_gridcolor=self.landing_config.grid_color,
            yaxis_gridcolor=self.landing_config.grid_color,
            zaxis_gridcolor=self.landing_config.grid_color,
            aspectmode="data",
        )

        fig.update_layout(
            title="Lunar Landing Analysis Dashboard",
            template=self.landing_config.theme,
            height=1000,
            width=1400,
            font={"color": self.landing_config.text_color},
        )

        return fig

    def create_animated_landing_sequence(
        self,
        trajectory_data: Dict[str, Any],
        landing_site: Optional[Tuple[float, float, float]] = None,
        animation_frames: int = 100,
    ) -> go.Figure:
        """
        Create animated landing sequence visualization.

        Args:
            trajectory_data: Complete trajectory data
            landing_site: Target landing coordinates
            animation_frames: Number of animation frames

        Returns:
            Animated Plotly Figure showing landing progression
        """
        if "positions" not in trajectory_data or "time_points" not in trajectory_data:
            raise ValueError("Trajectory data must include positions and time_points")

        positions = trajectory_data["positions"]
        times = trajectory_data["time_points"]

        # Create base figure
        fig = self.create_landing_trajectory_plot(
            trajectory_data, landing_site, "Animated Lunar Landing"
        )

        # Prepare animation frames
        n_points = len(positions)
        frame_indices = np.linspace(0, n_points - 1, animation_frames, dtype=int)

        frames = []
        for i, idx in enumerate(frame_indices):
            # Current position
            current_pos = positions[idx] / 1000  # Convert to km

            # Trajectory up to current point
            traj_positions = positions[: idx + 1] / 1000

            frame_data = []

            # Add trajectory path up to current point
            if len(traj_positions) > 1:
                frame_data.append(
                    go.Scatter3d(
                        x=traj_positions[:, 0],
                        y=traj_positions[:, 1],
                        z=traj_positions[:, 2],
                        mode="lines",
                        line={"color": "#00ff00", "width": 4},
                        name="Trajectory",
                        showlegend=False,
                    )
                )

            # Add current spacecraft position
            frame_data.append(
                go.Scatter3d(
                    x=[current_pos[0]],
                    y=[current_pos[1]],
                    z=[current_pos[2]],
                    mode="markers",
                    marker={"size": 8, "color": "#ff0000", "symbol": "diamond"},
                    name="Spacecraft",
                    showlegend=False,
                    hovertemplate=f"Time: {times[idx]:.1f}s<br>Alt: {(np.linalg.norm(positions[idx]) - self.lunar_radius)/1000:.1f} km<extra></extra>",
                )
            )

            frames.append(go.Frame(data=frame_data, name=str(i)))

        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 50},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f"{times[frame_indices[int(f.name)]]:.1f}s",
                            "method": "animate",
                        }
                        for f in frames
                    ],
                    "active": 0,
                    "len": 0.9,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
        )

        return fig

    def _add_star_field(self, fig: go.Figure) -> None:
        """Add random star field background."""
        # Generate random star positions
        n_stars = self.landing_config.star_count
        star_distance = 500000  # km (far background)

        # Random spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi, n_stars)
        phi = np.random.uniform(0, np.pi, n_stars)

        x_stars = star_distance * np.sin(phi) * np.cos(theta)
        y_stars = star_distance * np.sin(phi) * np.sin(theta)
        z_stars = star_distance * np.cos(phi)

        fig.add_trace(
            go.Scatter3d(
                x=x_stars,
                y=y_stars,
                z=z_stars,
                mode="markers",
                marker={"size": 1, "color": "white", "opacity": 0.6},
                name="Stars",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    def _add_earth_background(self, fig: go.Figure) -> None:
        """Add Earth in the background."""
        earth_distance = 384400  # km (Earth-Moon distance)
        earth_scale = self.landing_config.earth_position_scale

        fig.add_trace(
            go.Scatter3d(
                x=[-earth_distance * earth_scale],
                y=[0],
                z=[0],
                mode="markers",
                marker={"size": 20, "color": "blue", "opacity": 0.7},
                name="Earth",
                hovertemplate="Earth<br>Distance: 384,400 km<extra></extra>",
            )
        )

    def _add_lunar_surface(self, fig: go.Figure) -> None:
        """Add lunar surface mesh."""
        resolution = self.landing_config.lunar_surface_resolution

        # Create spherical surface mesh
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        x_surface = (self.lunar_radius / 1000) * np.outer(np.cos(u), np.sin(v))
        y_surface = (self.lunar_radius / 1000) * np.outer(np.sin(u), np.sin(v))
        z_surface = (self.lunar_radius / 1000) * np.outer(
            np.ones(np.size(u)), np.cos(v)
        )

        fig.add_trace(
            go.Surface(
                x=x_surface,
                y=y_surface,
                z=z_surface,
                colorscale=[
                    [0, self.landing_config.lunar_surface_color],
                    [1, self.landing_config.lunar_surface_color],
                ],
                opacity=self.landing_config.lunar_surface_opacity,
                showscale=False,
                name="Lunar Surface",
                hoverinfo="skip",
            )
        )

    def _add_moon_body(self, fig: go.Figure) -> None:
        """Add Moon as central body."""
        fig.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker={
                    "size": self.landing_config.moon_radius_scale,
                    "color": "gray",
                    "opacity": 0.8,
                },
                name="Moon",
                hovertemplate="Moon<br>Radius: 1,737 km<extra></extra>",
            )
        )

    def _add_landing_zone(
        self, fig: go.Figure, landing_site: Tuple[float, float, float]
    ) -> None:
        """Add landing zone visualization."""
        x, y, z = landing_site
        radius_km = self.landing_config.landing_zone_radius / 1000

        # Create landing zone circle on surface
        theta = np.linspace(0, 2 * np.pi, 50)

        # Approximate circle on lunar surface
        circle_x = (x / 1000) + radius_km * np.cos(theta)
        circle_y = (y / 1000) + radius_km * np.sin(theta)
        circle_z = np.full_like(circle_x, z / 1000)

        fig.add_trace(
            go.Scatter3d(
                x=circle_x,
                y=circle_y,
                z=circle_z,
                mode="lines",
                line={"color": self.landing_config.landing_zone_color, "width": 3},
                name="Landing Zone",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add target marker
        fig.add_trace(
            go.Scatter3d(
                x=[x / 1000],
                y=[y / 1000],
                z=[z / 1000],
                mode="markers",
                marker={
                    "size": self.landing_config.target_marker_size,
                    "color": self.landing_config.landing_zone_color,
                    "symbol": "x",
                },
                name="Landing Target",
                hovertemplate=f"Target Site<br>Coordinates: ({x:.0f}, {y:.0f}, {z:.0f}) m<extra></extra>",
            )
        )

    def _add_landing_trajectory_path(
        self, fig: go.Figure, trajectory_data: Dict[str, Any]
    ) -> None:
        """Add landing trajectory path with enhanced features."""
        if "positions" not in trajectory_data:
            raise ValueError("Trajectory data must contain positions")

        positions = trajectory_data["positions"]
        times = trajectory_data.get("time_points", range(len(positions)))

        # Convert positions to km
        x_km = positions[:, 0] / 1000
        y_km = positions[:, 1] / 1000
        z_km = positions[:, 2] / 1000

        # Calculate altitudes for color coding
        altitudes = []
        for pos in positions:
            alt = np.linalg.norm(pos) - self.lunar_radius
            altitudes.append(alt)

        # Create hover text with detailed info
        hover_text = []
        for i, (t, alt) in enumerate(zip(times, altitudes, strict=True)):
            hover_text.append(
                f"Time: {t:.1f}s<br>"
                f"Altitude: {alt/1000:.2f} km<br>"
                f"Position: ({x_km[i]:.1f}, {y_km[i]:.1f}, {z_km[i]:.1f}) km"
            )

        # Main trajectory path
        fig.add_trace(
            go.Scatter3d(
                x=x_km,
                y=y_km,
                z=z_km,
                mode="lines+markers",
                line={
                    "width": self.landing_config.trajectory_width,
                    "color": altitudes,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Altitude (m)", "x": 1.02},
                },
                marker={
                    "size": 3,
                    "color": altitudes,
                    "colorscale": "Viridis",
                    "showscale": False,
                },
                name="Landing Trajectory",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

        # Highlight start and end points
        fig.add_trace(
            go.Scatter3d(
                x=[x_km[0]],
                y=[y_km[0]],
                z=[z_km[0]],
                mode="markers",
                marker={"size": 12, "color": "green", "symbol": "circle"},
                name="Start",
                hovertemplate=f"Start Point<br>Altitude: {altitudes[0]/1000:.2f} km<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[x_km[-1]],
                y=[y_km[-1]],
                z=[z_km[-1]],
                mode="markers",
                marker={"size": 12, "color": "red", "symbol": "square"},
                name="Landing",
                hovertemplate=f"Landing Point<br>Altitude: {altitudes[-1]/1000:.2f} km<extra></extra>",
            )
        )

    def _add_thrust_vectors(
        self, fig: go.Figure, trajectory_data: Dict[str, Any]
    ) -> None:
        """Add realistic thrust vector visualization."""
        positions = trajectory_data["positions"]
        velocities = trajectory_data.get("velocities", None)
        thrust_profile = trajectory_data["thrust_profile"]
        times = trajectory_data.get("time_points", range(len(positions)))

        print(
            f"🚀 Adding thrust vectors: {len(positions)} positions, {len(thrust_profile)} thrust values"
        )

        # Sample thrust vectors (every 3rd point for better visibility)
        sample_indices = range(0, len(positions), 3)
        vectors_added = 0

        for i in sample_indices:
            if i >= len(thrust_profile):
                continue

            thrust_mag = thrust_profile[i]
            print(f"   Point {i}: thrust={thrust_mag:.0f}N")

            if thrust_mag < 500:  # Lower threshold to show more vectors
                continue

            pos = positions[i] / 1000  # Convert to km
            thrust_mag = thrust_profile[i]

            # Calculate thrust direction for physics (where thrust should point)
            physics_thrust_direction = self._calculate_realistic_thrust_direction(
                positions, velocities, i, times
            )

            # VISUAL REPRESENTATION: Show rocket exhaust coming OUT of spacecraft
            # Exhaust direction = OPPOSITE of thrust direction (Newton's 3rd law)
            exhaust_direction = -physics_thrust_direction

            # Scale exhaust plume length based on thrust magnitude - MAKE IT MUCH MORE VISIBLE
            max_thrust = max(thrust_profile) if len(thrust_profile) > 0 else thrust_mag
            thrust_scale = thrust_mag / max_thrust  # 0 to 1 scale
            exhaust_length = (
                thrust_scale * 50.0
            )  # Much larger scale for visibility (50km for max thrust)

            exhaust_vector = exhaust_direction * exhaust_length

            # Color and width based on thrust magnitude
            phase_color = self._get_thrust_phase_color(positions, i)
            exhaust_width = max(2, int(thrust_scale * 8))  # Thicker for higher thrust

            # Add rocket exhaust plume visualization
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0], pos[0] + exhaust_vector[0]],
                    y=[pos[1], pos[1] + exhaust_vector[1]],
                    z=[pos[2], pos[2] + exhaust_vector[2]],
                    mode="lines",
                    line={"color": phase_color, "width": exhaust_width},
                    name="Rocket Exhaust" if vectors_added == 0 else "",
                    showlegend=(vectors_added == 0),
                    hovertemplate=f"Thrust: {thrust_mag:.0f} N ({thrust_scale*100:.0f}%)<br>Time: {times[i]:.1f}s<br>Phase: {self._get_flight_phase(positions, i)}<br>Velocity: {np.linalg.norm(velocities[i]) if velocities is not None and i < len(velocities) else 'N/A':.1f} m/s<extra></extra>",
                )
            )
            vectors_added += 1

        print(f"✅ Added {vectors_added} thrust vectors to visualization")

    def _add_velocity_indicators(
        self, fig: go.Figure, trajectory_data: Dict[str, Any]
    ) -> None:
        """Add velocity indicators along the trajectory."""
        positions = trajectory_data["positions"]
        velocities = trajectory_data["velocities"]
        times = trajectory_data.get("time_points", range(len(positions)))

        # Sample every 10th point for velocity indicators
        sample_indices = range(0, len(positions), 10)

        for i in sample_indices:
            if i >= len(velocities):
                continue

            pos = positions[i] / 1000  # Convert to km
            vel = velocities[i]
            vel_magnitude = np.linalg.norm(vel)

            if vel_magnitude < 1.0:  # Skip very low velocities
                continue

            # Velocity vector visualization (scaled down)
            vel_direction = vel / vel_magnitude
            vel_scale = min(vel_magnitude / 10, 20)  # Scale velocity vector (max 20km)
            vel_vector = vel_direction * vel_scale

            # Color based on velocity magnitude
            if vel_magnitude > 50:
                vel_color = "#ff0000"  # Red for high velocity
            elif vel_magnitude > 20:
                vel_color = "#ff8800"  # Orange for medium velocity
            else:
                vel_color = "#00ff00"  # Green for low velocity

            # Add velocity vector
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0], pos[0] + vel_vector[0]],
                    y=[pos[1], pos[1] + vel_vector[1]],
                    z=[pos[2], pos[2] + vel_vector[2]],
                    mode="lines",
                    line={"color": vel_color, "width": 2},
                    name="Velocity Vector" if i == sample_indices[0] else "",
                    showlegend=(i == sample_indices[0]),
                    hovertemplate=f"Velocity: {vel_magnitude:.1f} m/s<br>Time: {times[i]:.1f}s<extra></extra>",
                )
            )

    def _calculate_realistic_thrust_direction(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray],
        index: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """Calculate realistic thrust direction based on flight phase and dynamics."""
        pos = positions[index]
        total_time = times[-1] if len(times) > 1 else 1.0
        time_progress = times[index] / total_time if total_time > 0 else 0.0

        # Calculate altitude
        # altitude = np.linalg.norm(pos) - self.lunar_radius  # Not currently used

        # Calculate thrust direction for DECELERATION and landing
        # Key principle: Thrust must OPPOSE velocity to slow down spacecraft

        # Get current velocity at this point
        if velocities is not None and index < len(velocities):
            vel = velocities[index]
            vel_magnitude = np.linalg.norm(vel)
        else:
            # Estimate velocity from position change
            if index > 0:
                vel = (positions[index] - positions[index - 1]) / (
                    times[index] - times[index - 1]
                )
                vel_magnitude = np.linalg.norm(vel)
            else:
                vel = np.array([60.0, 30.0, -10.0])  # Initial velocity estimate
                vel_magnitude = np.linalg.norm(vel)

        # Gravity direction (toward Moon center)
        gravity_dir = -pos / np.linalg.norm(pos)

        # MAIN PRINCIPLE: Thrust opposes velocity for deceleration
        if vel_magnitude > 1.0:
            # Primary thrust: OPPOSITE to velocity direction (for braking)
            deceleration_thrust = -vel / vel_magnitude

            # Secondary thrust: Counter gravity to maintain controlled descent
            gravity_compensation = -gravity_dir  # Upward

            # Blend based on flight phase
            if time_progress < 0.5:
                # Early phase: More deceleration, some gravity compensation
                thrust_direction = (
                    0.8 * deceleration_thrust + 0.2 * gravity_compensation
                )
            else:
                # Late phase: More gravity compensation, some deceleration
                thrust_direction = (
                    0.5 * deceleration_thrust + 0.5 * gravity_compensation
                )
        else:
            # Low velocity: Mainly gravity compensation
            thrust_direction = -gravity_dir

        # Normalize the thrust direction
        return thrust_direction / np.linalg.norm(thrust_direction)

    def _get_thrust_phase_color(self, positions: np.ndarray, index: int) -> str:
        """Get color for rocket exhaust based on flight phase and thrust intensity."""
        total_points = len(positions)
        progress = index / max(1, total_points - 1)

        # Rocket exhaust colors based on thrust intensity and phase
        if progress < 0.3:
            return "#ff6600"  # Bright orange for high-thrust braking
        elif progress < 0.8:
            return "#ff9900"  # Orange-yellow for descent thrust
        else:
            return "#ffcc00"  # Yellow for terminal landing thrust

    def _get_flight_phase(self, positions: np.ndarray, index: int) -> str:
        """Get flight phase name for hover info."""
        total_points = len(positions)
        progress = index / max(1, total_points - 1)

        if progress < 0.3:
            return "Approach/Braking"
        elif progress < 0.8:
            return "Powered Descent"
        else:
            return "Terminal Landing"

    def _configure_landing_layout(self, fig: go.Figure, title: str) -> None:
        """Configure 3D layout optimized for landing visualization."""
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": "X (km)",
                "yaxis_title": "Y (km)",
                "zaxis_title": "Z (km)",
                "bgcolor": self.landing_config.background_color,
                "xaxis": {
                    "gridcolor": self.landing_config.grid_color,
                    "showgrid": True,
                    "zeroline": False,
                },
                "yaxis": {
                    "gridcolor": self.landing_config.grid_color,
                    "showgrid": True,
                    "zeroline": False,
                },
                "zaxis": {
                    "gridcolor": self.landing_config.grid_color,
                    "showgrid": True,
                    "zeroline": False,
                },
                "aspectmode": "data",
                "camera": {
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.5},  # Good viewing angle
                    "up": {"x": 0, "y": 0, "z": 1},
                },
            },
            template=self.landing_config.theme,
            width=self.landing_config.width,
            height=self.landing_config.height,
            font={"color": self.landing_config.text_color},
            showlegend=True,
            legend={"x": 0.02, "y": 0.98, "bgcolor": "rgba(0,0,0,0.5)"},
        )


# Convenience functions for quick visualization


def create_quick_landing_plot(
    trajectory_positions: np.ndarray,
    trajectory_times: np.ndarray,
    landing_target: Optional[Tuple[float, float, float]] = None,
    thrust_profile: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Quick function to create a landing trajectory plot.

    Args:
        trajectory_positions: Position array (N, 3) in meters
        trajectory_times: Time array (N,) in seconds
        landing_target: Target coordinates (x, y, z) in meters
        thrust_profile: Optional thrust profile (N,) in Newtons

    Returns:
        Plotly Figure with 3D landing visualization
    """
    # Prepare trajectory data
    trajectory_data = {
        "positions": trajectory_positions,
        "time_points": trajectory_times,
    }

    if thrust_profile is not None:
        trajectory_data["thrust_profile"] = thrust_profile

    # Create visualizer and plot
    visualizer = Landing3DVisualizer()

    return visualizer.create_landing_trajectory_plot(
        trajectory_data, landing_target, "Lunar Landing Trajectory - Quick View"
    )


def create_descent_dashboard_from_scenario(
    scenario_name: str, output_directory: str = "landing_visualizations"
) -> str:
    """
    Create landing visualization dashboard from scenario results.

    Args:
        scenario_name: Name of the powered descent scenario
        output_directory: Directory to save visualizations

    Returns:
        Path to the generated HTML dashboard
    """
    import os
    from datetime import datetime

    # This would integrate with the existing scenario system
    # Placeholder for now - would load scenario results and create visualization

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_directory, f"{scenario_name}_landing_3d_{timestamp}.html"
    )

    # Create sample visualization for demonstration
    # In real implementation, this would load actual scenario data
    sample_positions = np.array(
        [
            [0, 0, 100000],  # 100 km altitude
            [0, 0, 50000],  # 50 km
            [0, 0, 10000],  # 10 km
            [0, 0, 1000],  # 1 km
            [0, 0, 0],  # Surface
        ]
    )

    sample_times = np.array([0, 30, 60, 90, 120])

    fig = create_quick_landing_plot(
        sample_positions, sample_times, landing_target=(0, 0, 0)
    )

    # Save HTML
    os.makedirs(output_directory, exist_ok=True)
    fig.write_html(output_file)

    return output_file
