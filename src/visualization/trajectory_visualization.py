"""
Interactive 3D Trajectory Visualization Module.

Provides comprehensive 3D visualization capabilities for lunar trajectories using Plotly,
including Earth-Moon system rendering, trajectory paths, orbital mechanics visualization,
and interactive analysis tools.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Any
from dataclasses import dataclass
from datetime import datetime
import pykep as pk

from src.trajectory.earth_moon_trajectories import LambertSolver, generate_earth_moon_trajectory
from src.trajectory.nbody_integration import EarthMoonNBodyPropagator
from src.trajectory.transfer_window_analysis import TrajectoryWindowAnalyzer


@dataclass
class TrajectoryPlotConfig:
    """Configuration for trajectory visualization plots."""

    # Plot dimensions and layout
    width: int = 1200
    height: int = 800
    title: str = "Lunar Mission Trajectory"

    # Celestial body visualization
    show_earth: bool = True
    show_moon: bool = True
    show_sun: bool = False
    earth_radius_scale: float = 10.0  # Scale factor for visibility
    moon_radius_scale: float = 20.0

    # Trajectory visualization
    trajectory_color: str = "#00ff00"
    trajectory_width: int = 4
    show_transfer_arcs: bool = True
    show_orbital_elements: bool = True

    # Animation and interactivity
    enable_animation: bool = True
    animation_frame_count: int = 100
    show_velocity_vectors: bool = False
    show_acceleration_vectors: bool = False

    # Background and styling
    background_color: str = "#000011"
    grid_color: str = "#333333"
    text_color: str = "#ffffff"
    theme: str = "plotly_dark"


class TrajectoryVisualizer:
    """
    Interactive 3D trajectory visualization using Plotly.

    Provides comprehensive visualization of Earth-Moon trajectories including:
    - 3D trajectory paths with celestial bodies
    - Transfer window analysis visualization
    - Orbital mechanics parameter visualization
    - Interactive trajectory comparison tools
    """

    def __init__(self, config: TrajectoryPlotConfig | None = None) -> None:
        """
        Initialize trajectory visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or TrajectoryPlotConfig()
        self.lambert_solver = LambertSolver()
        self.propagator = EarthMoonNBodyPropagator()
        self.window_analyzer = TrajectoryWindowAnalyzer()

        # Physical constants
        self.earth_radius = 6378137.0  # meters
        self.moon_radius = 1737400.0   # meters
        self.earth_moon_distance = 384400000.0  # meters (average)

    def create_3d_trajectory_plot(
        self,
        trajectories: dict[str, Any] | list[dict[str, Any]],
        title: str | None = None
    ) -> go.Figure:
        """
        Create comprehensive 3D trajectory visualization.

        Args:
            trajectories: Single trajectory or list of trajectories to plot
            title: Optional plot title override

        Returns
        -------
            Plotly Figure object with 3D trajectory visualization
        """
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        fig = go.Figure()

        # Add celestial bodies
        self._add_celestial_bodies(fig)

        # Add trajectory paths
        for i, trajectory in enumerate(trajectories):
            self._add_trajectory_path(fig, trajectory, f"Trajectory {i+1}")

        # Configure layout
        self._configure_3d_layout(fig, title or self.config.title)

        return fig

    def create_transfer_window_plot(
        self,
        start_date: datetime,
        end_date: datetime,
        earth_orbit_alt: float = 400.0,
        moon_orbit_alt: float = 100.0,
        max_windows: int = 50
    ) -> go.Figure:
        """
        Create transfer window opportunity visualization.

        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            earth_orbit_alt: Earth parking orbit altitude [km]
            moon_orbit_alt: Target lunar orbit altitude [km]
            max_windows: Maximum number of windows to display

        Returns
        -------
            Plotly Figure with transfer window analysis
        """
        # Find transfer windows
        windows = self.window_analyzer.find_transfer_windows(
            start_date=start_date,
            end_date=end_date,
            earth_orbit_alt=earth_orbit_alt,
            moon_orbit_alt=moon_orbit_alt,
            time_step=1.0
        )

        if not windows:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="No transfer windows found in specified period",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font={"size": 16, "color": self.config.text_color}
            )
            return fig

        # Limit to max_windows
        windows = windows[:max_windows]

        # Create subplot with multiple views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Transfer Windows (Delta-V vs Time)",
                "C3 Energy vs Transfer Time",
                "Departure Date vs Delta-V",
                "Transfer Duration Distribution"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ]
        )

        # Extract data
        departure_dates = [w.departure_date for w in windows]
        transfer_times = [w.transfer_time for w in windows]
        delta_vs = [w.total_dv for w in windows]
        c3_energies = [w.c3_energy for w in windows]

        # Plot 1: Delta-V vs Transfer Time
        fig.add_trace(
            go.Scatter(
                x=transfer_times,
                y=delta_vs,
                mode="markers+lines",
                name="Transfer Windows",
                text=[f"Departure: {d.strftime('%Y-%m-%d')}" for d in departure_dates],
                hovertemplate="Transfer Time: %{x:.1f} days<br>Delta-V: %{y:.0f} m/s<br>%{text}<extra></extra>",
                marker={
                    "size": 8,
                    "color": delta_vs,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Delta-V (m/s)", "x": 0.45}
                }
            ),
            row=1, col=1
        )

        # Plot 2: C3 Energy vs Transfer Time
        fig.add_trace(
            go.Scatter(
                x=transfer_times,
                y=c3_energies,
                mode="markers",
                name="C3 Energy",
                text=[f"Departure: {d.strftime('%Y-%m-%d')}" for d in departure_dates],
                hovertemplate="Transfer Time: %{x:.1f} days<br>C3: %{y:.0f} m²/s²<br>%{text}<extra></extra>",
                marker={"size": 6, "color": "orange"},
                showlegend=False
            ),
            row=1, col=2
        )

        # Plot 3: Departure Date vs Delta-V
        fig.add_trace(
            go.Scatter(
                x=departure_dates,
                y=delta_vs,
                mode="markers+lines",
                name="Departure Opportunities",
                hovertemplate="Date: %{x}<br>Delta-V: %{y:.0f} m/s<extra></extra>",
                marker={"size": 6, "color": "cyan"},
                line={"width": 2},
                showlegend=False
            ),
            row=2, col=1
        )

        # Plot 4: Transfer Duration Distribution
        fig.add_trace(
            go.Histogram(
                x=transfer_times,
                nbinsx=20,
                name="Duration Distribution",
                marker={"color": "lightblue", "opacity": 0.7},
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Transfer Time (days)", row=1, col=1)
        fig.update_yaxes(title_text="Delta-V (m/s)", row=1, col=1)

        fig.update_xaxes(title_text="Transfer Time (days)", row=1, col=2)
        fig.update_yaxes(title_text="C3 Energy (m²/s²)", row=1, col=2)

        fig.update_xaxes(title_text="Departure Date", row=2, col=1)
        fig.update_yaxes(title_text="Delta-V (m/s)", row=2, col=1)

        fig.update_xaxes(title_text="Transfer Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        fig.update_layout(
            title=f"Transfer Window Analysis ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            template=self.config.theme,
            height=800,
            width=1400,
            showlegend=True
        )

        return fig

    def create_orbital_elements_plot(
        self,
        trajectory_data: dict[str, Any],
        show_evolution: bool = True
    ) -> go.Figure:
        """
        Create orbital elements evolution visualization.

        Args:
            trajectory_data: Trajectory data with positions and velocities
            show_evolution: Whether to show parameter evolution over time

        Returns
        -------
            Plotly Figure with orbital elements visualization
        """
        if "positions" not in trajectory_data or "velocities" not in trajectory_data:
            msg = "Trajectory data must contain 'positions' and 'velocities' arrays"
            raise ValueError(msg)

        positions = trajectory_data["positions"]  # Shape: (3, N)
        velocities = trajectory_data["velocities"]  # Shape: (3, N)
        times = trajectory_data.get("times", np.linspace(0, 1, positions.shape[1]))

        # Calculate orbital elements at each time step
        elements = self._calculate_orbital_elements_evolution(positions, velocities)

        if show_evolution:
            # Create time evolution plots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    "Semi-major Axis", "Eccentricity",
                    "Inclination", "RAAN",
                    "Argument of Periapsis", "True Anomaly"
                ]
            )

            # Plot each orbital element
            element_names = ["a", "e", "i", "raan", "argp", "nu"]
            element_units = ["km", "", "deg", "deg", "deg", "deg"]

            for idx, (name, unit) in enumerate(zip(element_names, element_units, strict=False)):
                row = (idx // 2) + 1
                col = (idx % 2) + 1

                y_data = elements[name]
                if name == "a":
                    y_data = y_data / 1000  # Convert to km
                elif name in ["i", "raan", "argp", "nu"]:
                    y_data = np.degrees(y_data)  # Convert to degrees

                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=y_data,
                        mode="lines",
                        name=f"{name.upper()} ({unit})",
                        line={"width": 2},
                        showlegend=False
                    ),
                    row=row, col=col
                )

                fig.update_yaxes(title_text=f"{name.upper()} ({unit})", row=row, col=col)
                fig.update_xaxes(title_text="Time", row=row, col=col)
        else:
            # Create single orbital elements summary
            fig = go.Figure()

            # Take initial orbital elements
            initial_elements = {name: elements[name][0] for name in elements}

            # Create summary table
            summary_data = []
            for name, value in initial_elements.items():
                if name == "a":
                    summary_data.append([name.upper(), f"{value/1000:.1f} km"])
                elif name == "e":
                    summary_data.append([name.upper(), f"{value:.4f}"])
                else:
                    summary_data.append([name.upper(), f"{np.degrees(value):.1f}°"])

            fig.add_trace(
                go.Table(
                    header={
                        "values": ["Orbital Element", "Value"],
                        "fill_color": "darkblue",
                        "align": "left",
                        "font": {"color": "white", "size": 14}
                    },
                    cells={
                        "values": list(zip(*summary_data, strict=False)),
                        "fill_color": "lightblue",
                        "align": "left",
                        "font": {"size": 12}
                    }
                )
            )

        fig.update_layout(
            title="Orbital Elements Analysis",
            template=self.config.theme,
            height=800,
            width=1200
        )

        return fig

    def create_trajectory_comparison(
        self,
        trajectories: list[dict[str, Any]],
        labels: list[str] | None = None
    ) -> go.Figure:
        """
        Create comparative visualization of multiple trajectories.

        Args:
            trajectories: List of trajectory data dictionaries
            labels: Optional labels for each trajectory

        Returns
        -------
            Plotly Figure with trajectory comparison
        """
        if not trajectories:
            msg = "At least one trajectory required for comparison"
            raise ValueError(msg)

        if labels is None:
            labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]

        # Create main 3D plot
        fig = go.Figure()

        # Add celestial bodies
        self._add_celestial_bodies(fig)

        # Color palette for trajectories
        colors = px.colors.qualitative.Set1[:len(trajectories)]

        # Add each trajectory
        for i, (trajectory, label) in enumerate(zip(trajectories, labels, strict=False)):
            self._add_trajectory_path(fig, trajectory, label, colors[i % len(colors)])

        # Configure layout
        self._configure_3d_layout(fig, "Trajectory Comparison")

        return fig

    def _add_celestial_bodies(self, fig: go.Figure) -> None:
        """Add Earth and Moon to 3D plot."""
        if self.config.show_earth:
            # Earth at origin
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode="markers",
                    marker={
                        "size": self.config.earth_radius_scale,
                        "color": "blue",
                        "opacity": 0.8
                    },
                    name="Earth",
                    hovertemplate="Earth<br>Radius: 6,378 km<extra></extra>"
                )
            )

        if self.config.show_moon:
            # Moon at average distance
            moon_x = self.earth_moon_distance / 1000  # Convert to km for display
            fig.add_trace(
                go.Scatter3d(
                    x=[moon_x], y=[0], z=[0],
                    mode="markers",
                    marker={
                        "size": self.config.moon_radius_scale,
                        "color": "gray",
                        "opacity": 0.8
                    },
                    name="Moon",
                    hovertemplate="Moon<br>Radius: 1,737 km<extra></extra>"
                )
            )

    def _add_trajectory_path(
        self,
        fig: go.Figure,
        trajectory: dict[str, Any],
        name: str,
        color: str | None = None
    ) -> None:
        """Add trajectory path to 3D plot."""
        if "positions" not in trajectory:
            msg = "Trajectory must contain 'positions' data"
            raise ValueError(msg)

        positions = trajectory["positions"]  # Shape: (3, N)

        # Convert to km for display
        x_km = positions[0, :] / 1000
        y_km = positions[1, :] / 1000
        z_km = positions[2, :] / 1000

        # Create hover text
        if "times" in trajectory:
            times = trajectory["times"]
            hover_text = [f"Time: {t:.2f}<br>Position: ({x:.0f}, {y:.0f}, {z:.0f}) km"
                         for t, x, y, z in zip(times, x_km, y_km, z_km, strict=False)]
        else:
            hover_text = [f"Position: ({x:.0f}, {y:.0f}, {z:.0f}) km"
                         for x, y, z in zip(x_km, y_km, z_km, strict=False)]

        fig.add_trace(
            go.Scatter3d(
                x=x_km, y=y_km, z=z_km,
                mode="lines+markers",
                line={
                    "width": self.config.trajectory_width,
                    "color": color or self.config.trajectory_color
                },
                marker={"size": 2},
                name=name,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>"
            )
        )

    def _configure_3d_layout(self, fig: go.Figure, title: str) -> None:
        """Configure 3D plot layout."""
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": "X (km)",
                "yaxis_title": "Y (km)",
                "zaxis_title": "Z (km)",
                "bgcolor": self.config.background_color,
                "xaxis": {"gridcolor": self.config.grid_color},
                "yaxis": {"gridcolor": self.config.grid_color},
                "zaxis": {"gridcolor": self.config.grid_color},
                "aspectmode": "data"
            },
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            font={"color": self.config.text_color}
        )

    def _calculate_orbital_elements_evolution(
        self,
        positions: np.ndarray,
        velocities: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Calculate orbital elements evolution from position/velocity data.

        Args:
            positions: Position vectors (3, N)
            velocities: Velocity vectors (3, N)

        Returns
        -------
            Dictionary of orbital elements arrays
        """
        mu = pk.MU_EARTH
        n_points = positions.shape[1]

        # Initialize arrays
        elements = {
            "a": np.zeros(n_points),      # Semi-major axis
            "e": np.zeros(n_points),      # Eccentricity
            "i": np.zeros(n_points),      # Inclination
            "raan": np.zeros(n_points),   # Right ascension of ascending node
            "argp": np.zeros(n_points),   # Argument of periapsis
            "nu": np.zeros(n_points)      # True anomaly
        }

        for idx in range(n_points):
            r = positions[:, idx]
            v = velocities[:, idx]

            try:
                # Convert to PyKEP format and calculate elements
                kep_elements = pk.ic2par(r.tolist(), v.tolist(), mu)

                elements["a"][idx] = kep_elements[0]
                elements["e"][idx] = kep_elements[1]
                elements["i"][idx] = kep_elements[2]
                elements["raan"][idx] = kep_elements[3]
                elements["argp"][idx] = kep_elements[4]
                elements["nu"][idx] = kep_elements[5]

            except Exception:
                # Handle edge cases with NaN
                for key in elements:
                    elements[key][idx] = np.nan

        return elements


def create_quick_trajectory_plot(
    earth_orbit_alt: float = 400.0,
    moon_orbit_alt: float = 100.0,
    transfer_time: float = 4.5,
    departure_epoch: float = 10000.0
) -> go.Figure:
    """
    Quick function to create a simple trajectory plot.

    Args:
        earth_orbit_alt: Earth orbit altitude [km]
        moon_orbit_alt: Moon orbit altitude [km]
        transfer_time: Transfer time [days]
        departure_epoch: Departure epoch [days since J2000]

    Returns
    -------
        Plotly Figure with trajectory visualization
    """
    try:
        # Generate trajectory
        trajectory, total_dv = generate_earth_moon_trajectory(
            departure_epoch=departure_epoch,
            earth_orbit_alt=earth_orbit_alt,
            moon_orbit_alt=moon_orbit_alt,
            transfer_time=transfer_time
        )

        # Create visualizer and plot
        visualizer = TrajectoryVisualizer()

        # Convert trajectory to visualization format
        trajectory_data = {
            "positions": trajectory.position_history,
            "velocities": trajectory.velocity_history,
            "times": trajectory.time_history
        }

        return visualizer.create_3d_trajectory_plot(
            trajectory_data,
            title=f"Earth-Moon Transfer (ΔV: {total_dv:.0f} m/s)"
        )


    except Exception as e:
        # Create error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Trajectory generation failed: {e!s}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16, "color": "red"}
        )
        return fig
