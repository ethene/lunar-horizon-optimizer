"""Legacy configuration manager module - DEPRECATED.

This module is maintained for backward compatibility only.
Please import from the new management package modules:
- src.config.management.config_manager
- src.config.management.template_manager
- src.config.management.file_operations

This module will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "manager.py is deprecated. Import from src.config.management package instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new management modules for backward compatibility
from .management.config_manager import ConfigManager
from .management.file_operations import FileOperations
from .management.template_manager import TemplateManager

__all__ = [
    "ConfigManager",
    "FileOperations",
    "TemplateManager",
]
