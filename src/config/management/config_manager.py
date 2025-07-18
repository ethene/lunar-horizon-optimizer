"""Core configuration manager using composition pattern.

This module provides the main ConfigManager class that orchestrates
configuration operations by delegating to specialized components.
"""

from pathlib import Path
from typing import Any

from src.config.loader import ConfigurationError
from src.config.models import MissionConfig

from .file_operations import FileOperations
from .template_manager import TemplateManager


class ConfigManager:
    """Manager for handling mission configuration using composition.

    This class provides a high-level interface for managing mission configurations,
    delegating specialized operations to focused components:
    - FileOperations: Handle file I/O operations
    - TemplateManager: Handle template-based configuration creation

    The manager maintains state for the currently active configuration
    and ensures all operations maintain validation.

    Attributes
    ----------
        file_ops (FileOperations): Handler for file operations
        template_manager (TemplateManager): Handler for template operations
        _active_config (Optional[MissionConfig]): Currently active configuration
    """

    def __init__(
        self,
        file_ops: FileOperations | None = None,
        template_manager: TemplateManager | None = None,
    ) -> None:
        """Initialize the configuration manager.

        Args:
            file_ops: Optional FileOperations instance. Creates default if not provided.
            template_manager: Optional TemplateManager instance. Creates default if not provided.
        """
        self.file_ops = file_ops or FileOperations()
        self.template_manager = template_manager or TemplateManager()
        self._active_config: MissionConfig | None = None

    @property
    def active_config(self) -> MissionConfig | None:
        """Get the currently active configuration.

        Returns
        -------
            Currently active MissionConfig or None if not set.
        """
        return self._active_config

    @property
    def registry(self):
        """Get the configuration registry from the template manager.

        Returns
        -------
            ConfigRegistry instance used by the template manager.
        """
        return self.template_manager.registry

    def load_config(self, file_path: Path) -> MissionConfig:
        """Load a configuration from file and set as active.

        Delegates to FileOperations for actual loading operation.

        Args:
            file_path: Path to the configuration file.

        Returns
        -------
            Loaded and validated configuration.

        Raises
        ------
            ConfigurationError: If file cannot be loaded or validation fails.
        """
        config = self.file_ops.load_from_file(file_path)
        self._active_config = config
        return config

    def save_config(self, file_path: Path) -> None:
        """Save the active configuration to a file.

        Delegates to FileOperations for actual saving operation.

        Args:
            file_path: Path where to save the configuration.

        Raises
        ------
            ConfigurationError: If no active configuration or save fails.
        """
        if not self._active_config:
            msg = "No active configuration to save"
            raise ConfigurationError(msg)

        self.file_ops.save_to_file(self._active_config, file_path)

    def create_from_template(self, template_name: str, **overrides) -> MissionConfig:
        """Create a new configuration from a template.

        Delegates to TemplateManager for template operations.

        Args:
            template_name: Name of the template to use.
            **overrides: Values to override in the template.

        Returns
        -------
            New configuration based on the template.

        Raises
        ------
            KeyError: If template does not exist.
            ValidationError: If overrides are invalid.
        """
        config = self.template_manager.create_from_template(template_name, **overrides)
        self._active_config = config
        return config

    def validate_config(self, config: dict[str, Any]) -> MissionConfig:
        """Validate a configuration dictionary against the mission config model.

        Args:
            config: Configuration dictionary to validate.

        Returns
        -------
            Validated MissionConfig instance.

        Raises
        ------
            ValidationError: If configuration is invalid.
        """
        return MissionConfig.model_validate(config)

    def update_config(self, updates: dict[str, Any]) -> MissionConfig:
        """Update the active configuration with new values.

        Updates the active configuration with new values, performing a deep
        update that preserves nested structure. The updated configuration
        is automatically validated.

        Args:
            updates: Dictionary of values to update.

        Returns
        -------
            Updated configuration.

        Raises
        ------
            ConfigurationError: If no active configuration.
            ValidationError: If updates are invalid.
        """
        if not self._active_config:
            msg = "No active configuration to update"
            raise ConfigurationError(msg)

        # Create updated configuration dictionary
        config_dict = self._active_config.model_dump()

        # Deep update nested dictionaries
        config_dict = self._deep_update(config_dict, updates)

        # Validate and apply updates
        updated_config = MissionConfig.model_validate(config_dict)
        self._active_config = updated_config
        return updated_config

    def get_available_templates(self) -> list[str]:
        """Get list of available template names.

        Delegates to TemplateManager.

        Returns
        -------
            List of available template names.
        """
        return self.template_manager.get_available_templates()

    def get_supported_file_formats(self) -> list[str]:
        """Get list of supported file formats.

        Delegates to FileOperations.

        Returns
        -------
            List of supported file extensions.
        """
        return self.file_ops.get_supported_formats()

    def _deep_update(self, d: dict, u: dict) -> dict:
        """Perform deep update of nested dictionaries.

        Args:
            d: Base dictionary.
            u: Updates to apply.

        Returns
        -------
            Updated dictionary.
        """
        d = d.copy()  # Don't modify original
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
