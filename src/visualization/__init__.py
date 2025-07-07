"""
Visualization Package

Interactive visualization modules for the Lunar Horizon Optimizer:
- 3D trajectory visualization with Plotly
- Pareto front visualization for optimization results
- Economic analysis dashboards
- Mission timeline and milestone visualization
- Comprehensive mission analysis dashboard

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

from .trajectory_visualization import TrajectoryVisualizer, TrajectoryPlotConfig
from .optimization_visualization import OptimizationVisualizer, ParetoPlotConfig
from .economic_visualization import EconomicVisualizer, DashboardConfig
from .mission_visualization import MissionVisualizer, TimelineConfig
from .dashboard import ComprehensiveDashboard, DashboardTheme

__all__ = [
    'TrajectoryVisualizer',
    'TrajectoryPlotConfig', 
    'OptimizationVisualizer',
    'ParetoPlotConfig',
    'EconomicVisualizer',
    'DashboardConfig',
    'MissionVisualizer',
    'TimelineConfig',
    'ComprehensiveDashboard',
    'DashboardTheme'
]

__version__ = "0.9.0"