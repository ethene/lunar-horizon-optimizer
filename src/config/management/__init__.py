"""Configuration management package.

This package provides specialized configuration management functionality,
split into focused modules for maintainability:

- config_manager: Core ConfigManager class for orchestration
- template_manager: Template-related functionality
- file_operations: Load/save operations and file handling
"""

from .config_manager import ConfigManager
from .file_operations import FileOperations
from .template_manager import TemplateManager

__all__ = [
    "ConfigManager",
    "FileOperations",
    "TemplateManager",
]
