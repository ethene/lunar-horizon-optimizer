"""
Visualization Package.

Interactive visualization modules for the Lunar Horizon Optimizer:
- 3D trajectory visualization with Plotly
- Pareto front visualization for optimization results
- Economic analysis dashboards
- Mission timeline and milestone visualization
- Comprehensive mission analysis dashboard

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

from .dashboard import ComprehensiveDashboard, DashboardTheme
from .economic_visualization import DashboardConfig, EconomicVisualizer
from .mission_visualization import MissionVisualizer, TimelineConfig
from .optimization_visualization import OptimizationVisualizer, ParetoPlotConfig
from .trajectory_visualization import TrajectoryPlotConfig, TrajectoryVisualizer

__all__ = [
    "ComprehensiveDashboard",
    "DashboardConfig",
    "DashboardTheme",
    "EconomicVisualizer",
    "MissionVisualizer",
    "OptimizationVisualizer",
    "ParetoPlotConfig",
    "TimelineConfig",
    "TrajectoryPlotConfig",
    "TrajectoryVisualizer",
]

__version__ = "0.9.0"
