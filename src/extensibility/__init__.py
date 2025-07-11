"""Extensibility framework for the Lunar Horizon Optimizer.

This module provides a standardized interface for extending the system with
new flight stages, analysis modules, and optimization algorithms.
"""

from .base_extension import BaseExtension, ExtensionMetadata
from .extension_manager import ExtensionManager
from .plugin_interface import PluginInterface
from .data_transform import DataTransformLayer
from .registry import ExtensionRegistry

__all__ = [
    "BaseExtension",
    "ExtensionMetadata",
    "ExtensionManager",
    "PluginInterface",
    "DataTransformLayer",
    "ExtensionRegistry",
]
