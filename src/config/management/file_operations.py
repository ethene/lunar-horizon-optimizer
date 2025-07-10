"""File operations for configuration management.

This module handles all file I/O operations for configuration loading and saving,
supporting both JSON and YAML formats with proper error handling.
"""

from pathlib import Path

from src.config.loader import ConfigLoader
from src.config.models import MissionConfig


class FileOperations:
    """Handles file operations for configuration management.

    This class encapsulates all file I/O operations, including loading
    configurations from files and saving configurations to files.
    Supports both JSON and YAML formats.
    """

    def __init__(self) -> None:
        """Initialize file operations handler."""
        self.loader = ConfigLoader()

    def load_from_file(self, file_path: Path) -> MissionConfig:
        """Load a configuration from file.

        Supports both JSON (.json) and YAML (.yml, .yaml) file formats.
        The loaded configuration is automatically validated against the
        mission configuration schema.

        Args:
            file_path: Path to the configuration file.

        Returns
        -------
            Loaded and validated configuration.

        Raises
        ------
            ConfigurationError: If file cannot be loaded or validation fails.
        """
        return self.loader.load_file(file_path)

    def save_to_file(self, config: MissionConfig, file_path: Path) -> None:
        """Save a configuration to file.

        Supports both JSON (.json) and YAML (.yml, .yaml) file formats.
        The file format is determined by the file extension.

        Args:
            config: Configuration to save.
            file_path: Path where to save the configuration.

        Raises
        ------
            ConfigurationError: If save operation fails.
        """
        self.loader.save_config(config, file_path)

    def validate_file_path(self, file_path: Path) -> bool:
        """Validate if a file path is suitable for configuration operations.

        Args:
            file_path: Path to validate.

        Returns
        -------
            True if path is valid for configuration operations.
        """
        # Check if file extension is supported
        supported_extensions = {".json", ".yml", ".yaml"}
        return file_path.suffix.lower() in supported_extensions

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats.

        Returns
        -------
            List of supported file extensions.
        """
        return [".json", ".yml", ".yaml"]
