"""Global Optimization Module for Task 4 implementation.

This package provides PyGMO-based multi-objective optimization capabilities
for lunar mission trajectory design, implementing NSGA-II algorithms to
generate Pareto fronts balancing delta-v, time, and cost objectives.
"""

from src.optimization.cost_integration import CostCalculator, EconomicObjectives
from src.optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
from src.optimization.pareto_analysis import OptimizationResult, ParetoAnalyzer

__all__ = [
    "CostCalculator",
    "EconomicObjectives",
    "GlobalOptimizer",
    "LunarMissionProblem",
    "OptimizationResult",
    "ParetoAnalyzer",
]
