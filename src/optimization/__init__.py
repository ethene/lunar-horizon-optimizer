"""Global Optimization Module for Task 4 implementation.

This package provides PyGMO-based multi-objective optimization capabilities
for lunar mission trajectory design, implementing NSGA-II algorithms to 
generate Pareto fronts balancing delta-v, time, and cost objectives.
"""

from optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
from optimization.pareto_analysis import ParetoAnalyzer, OptimizationResult
from optimization.cost_integration import CostCalculator, EconomicObjectives

__all__ = [
    'GlobalOptimizer',
    'LunarMissionProblem', 
    'ParetoAnalyzer',
    'OptimizationResult',
    'CostCalculator',
    'EconomicObjectives'
]