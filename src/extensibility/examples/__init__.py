"""Example extensions for the Task 10 extensibility framework.

This module contains example implementations of extensions to demonstrate
how to create new flight stages, cost models, and other extension types.
"""

from .lunar_descent_extension import LunarDescentExtension
from .custom_cost_model import CustomCostModel

__all__ = [
    "LunarDescentExtension",
    "CustomCostModel",
]
