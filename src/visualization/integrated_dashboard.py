"""
Integrated Dashboard for Mission Analysis

This module provides unified visualization combining trajectory analysis,
economic metrics, and optimization results into comprehensive dashboards.
"""

from typing import Any, Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_mission_dashboard(
    trajectory_results: Dict[str, Any],
    economic_results: Dict[str, Any],
    title: str = "Integrated Mission Analysis",
    show_3d: bool = True,
    show_economics: bool = True,
    **kwargs,
) -> go.Figure:
    """
    Create integrated mission analysis dashboard.

    Args:
        trajectory_results: Dictionary with trajectory data including:
            - delta_v: Total delta-v (m/s)
            - time_of_flight: Mission duration (days)
            - trajectory_points: List of (x, y, z) coordinates
        economic_results: Dictionary with economic data including:
            - npv: Net Present Value
            - irr: Internal Rate of Return
            - roi: Return on Investment
        title: Dashboard title
        show_3d: Whether to include 3D trajectory plot
        show_economics: Whether to include economic metrics
        **kwargs: Additional styling options

    Returns:
        Plotly Figure with integrated dashboard
    """
    # Create subplot structure
    if show_3d and show_economics:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "3D Trajectory",
                "Economic Metrics",
                "Mission Summary",
                "Performance",
            ),
            specs=[[{"type": "scatter3d"}, {"type": "bar"}], [{"colspan": 2}, None]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )
    elif show_3d:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("3D Trajectory", "Mission Summary"),
            specs=[[{"type": "scatter3d"}, {"type": "bar"}]],
        )
    elif show_economics:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Economic Metrics", "Mission Summary"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )
    else:
        fig = go.Figure()

    # Add 3D trajectory if requested
    if show_3d and "trajectory_points" in trajectory_results:
        trajectory_points = trajectory_results["trajectory_points"]
        if trajectory_points:
            x_coords = [p[0] / 1e6 for p in trajectory_points]  # Convert to Mm
            y_coords = [p[1] / 1e6 for p in trajectory_points]
            z_coords = [p[2] / 1e6 for p in trajectory_points]

            fig.add_trace(
                go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines+markers",
                    name="Trajectory",
                    line=dict(color="blue", width=4),
                    marker=dict(size=2),
                ),
                row=1,
                col=1,
            )

    # Add economic metrics if requested
    if show_economics and economic_results:
        metrics = ["NPV", "IRR", "ROI"]
        values = [
            economic_results.get("npv", 0) / 1e6,  # Convert to millions
            economic_results.get("irr", 0) * 100,  # Convert to percentage
            economic_results.get("roi", 0) * 100,  # Convert to percentage
        ]

        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name="Economic Metrics",
                marker_color=["green" if v > 0 else "red" for v in values],
                text=[
                    f"${v:.1f}M" if i == 0 else f"{v:.1f}%"
                    for i, v in enumerate(values)
                ],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

    # Add mission summary
    if show_3d and show_economics:
        # Mission parameters summary
        summary_metrics = ["Delta-V", "Time of Flight", "Total Cost"]
        summary_values = [
            trajectory_results.get("delta_v", 0),
            trajectory_results.get("time_of_flight", 0),
            abs(economic_results.get("npv", 0)) / 1e6,  # Rough cost estimate
        ]

        fig.add_trace(
            go.Bar(
                x=summary_metrics,
                y=summary_values,
                name="Mission Parameters",
                marker_color="lightblue",
                text=[
                    (
                        f"{v:.0f} m/s"
                        if i == 0
                        else f"{v:.1f} days" if i == 1 else f"${v:.1f}M"
                    )
                    for i, v in enumerate(summary_values)
                ],
                textposition="auto",
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800 if show_3d and show_economics else 600,
        showlegend=True,
        font=dict(family="Arial, sans-serif", size=12),
    )

    return fig


def create_optimization_dashboard(
    optimization_results: Dict[str, Any], title: str = "Optimization Results Dashboard"
) -> go.Figure:
    """
    Create optimization results dashboard.

    Args:
        optimization_results: Dictionary with optimization data
        title: Dashboard title

    Returns:
        Plotly Figure with optimization dashboard
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Convergence",
            "Pareto Front",
            "Objective Values",
            "Solution Quality",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # Add convergence plot
    if "convergence_history" in optimization_results:
        history = optimization_results["convergence_history"]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(history))),
                y=history,
                mode="lines",
                name="Convergence",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

    # Add Pareto front
    if "pareto_solutions" in optimization_results:
        solutions = optimization_results["pareto_solutions"]
        if solutions:
            delta_v = [s["objectives"]["delta_v"] for s in solutions]
            cost = [s["objectives"]["cost"] for s in solutions]

            fig.add_trace(
                go.Scatter(
                    x=delta_v,
                    y=cost,
                    mode="markers",
                    name="Pareto Front",
                    marker=dict(color="red", size=8),
                ),
                row=1,
                col=2,
            )

    # Update layout
    fig.update_layout(title=title, template="plotly_white", height=800, showlegend=True)

    return fig


def create_comparison_dashboard(
    scenarios: List[Dict[str, Any]], title: str = "Mission Scenarios Comparison"
) -> go.Figure:
    """
    Create multi-scenario comparison dashboard.

    Args:
        scenarios: List of scenario dictionaries
        title: Dashboard title

    Returns:
        Plotly Figure with comparison dashboard
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cost Comparison",
            "Performance Comparison",
            "Risk Analysis",
            "ROI Comparison",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
    )

    # Extract data for comparison
    scenario_names = [s.get("name", f"Scenario {i+1}") for i, s in enumerate(scenarios)]
    costs = [s.get("cost", 0) for s in scenarios]
    delta_vs = [s.get("delta_v", 0) for s in scenarios]
    times = [s.get("time", 0) for s in scenarios]
    rois = [s.get("roi", 0) for s in scenarios]

    # Cost comparison
    fig.add_trace(
        go.Bar(x=scenario_names, y=costs, name="Cost", marker_color="lightcoral"),
        row=1,
        col=1,
    )

    # Performance comparison (delta-v vs time)
    fig.add_trace(
        go.Scatter(
            x=delta_vs,
            y=times,
            mode="markers+text",
            text=scenario_names,
            textposition="top center",
            name="Performance",
            marker=dict(size=10, color="blue"),
        ),
        row=1,
        col=2,
    )

    # ROI comparison
    fig.add_trace(
        go.Bar(x=scenario_names, y=rois, name="ROI", marker_color="lightgreen"),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(title=title, template="plotly_white", height=800, showlegend=True)

    return fig
