"""Configuration loader module.

This module provides functionality for loading and validating mission configurations
from JSON and YAML files.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .models import MissionConfig

class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""


class ConfigLoader:
    """Configuration loader with validation support."""

    SUPPORTED_FORMATS = {".json", ".yml", ".yaml"}

    def __init__(self, default_config: dict[str, Any] | None = None) -> None:
        """Initialize the configuration loader.

        Args:
            default_config: Optional dictionary containing default configuration values.
        """
        self.default_config = default_config or {}

    def load_file(self, file_path: str | Path) -> MissionConfig:
        """Load and validate a configuration file.

        Args:
            file_path: Path to the configuration file.

        Returns
        -------
            Validated MissionConfig object.

        Raises
        ------
            ConfigurationError: If the file cannot be loaded or validation fails.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            msg = f"Configuration file not found: {file_path}"
            raise ConfigurationError(msg)

        if file_path.suffix not in self.SUPPORTED_FORMATS:
            msg = (
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
            raise ConfigurationError(
                msg
            )

        try:
            config_dict = self._read_config_file(file_path)
            merged_config = self._merge_with_defaults(config_dict)
            return self._validate_config(merged_config)

        except yaml.YAMLError as e:
            msg = f"Invalid YAML format in {file_path}: {e!s}"
            raise ConfigurationError(msg)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON format in {file_path}: {e!s}"
            raise ConfigurationError(msg)
        except Exception as e:
            msg = f"Error loading configuration: {e!s}"
            raise ConfigurationError(msg)

    def _read_config_file(self, file_path: Path) -> dict[str, Any]:
        """Read configuration from a file.

        Args:
            file_path: Path to the configuration file.

        Returns
        -------
            Dictionary containing the configuration data.
        """
        with open(file_path, encoding="utf-8") as f:
            if file_path.suffix == ".json":
                return json.load(f)
            return yaml.safe_load(f)

    def _merge_with_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge configuration with default values.

        Args:
            config: Configuration dictionary to merge.

        Returns
        -------
            Merged configuration dictionary.
        """
        merged = self.default_config.copy()
        merged.update(config)
        return merged

    def _validate_config(self, config: dict[str, Any]) -> MissionConfig:
        """Validate configuration data using Pydantic model.

        Args:
            config: Configuration dictionary to validate.

        Returns
        -------
            Validated MissionConfig object.

        Raises
        ------
            ValidationError: If validation fails.
        """
        try:
            return MissionConfig.model_validate(config)
        except ValidationError as e:
            # Format validation errors
            error_messages = []
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                error_messages.append(f"{location}: {error['msg']}")

            raise ConfigurationError(
                "Configuration validation failed:\n" + "\n".join(error_messages)
            )

    def save_config(self, config: MissionConfig, file_path: str | Path) -> None:
        """Save configuration to a file.

        Args:
            config: MissionConfig object to save.
            file_path: Path where to save the configuration.

        Raises
        ------
            ConfigurationError: If the file cannot be saved.
        """
        file_path = Path(file_path)

        if file_path.suffix not in self.SUPPORTED_FORMATS:
            msg = (
                f"Unsupported output format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
            raise ConfigurationError(
                msg
            )

        try:
            config_dict = config.model_dump()

            with open(file_path, "w", encoding="utf-8") as f:
                if file_path.suffix == ".json":
                    json.dump(config_dict, f, indent=2)
                else:
                    yaml.safe_dump(config_dict, f, default_flow_style=False)
        except Exception as e:
            msg = f"Failed to save configuration to {file_path}: {e!s}"
            raise ConfigurationError(msg)

    @classmethod
    def load_default_config(cls) -> "ConfigLoader":
        """Create a ConfigLoader with default lunar mission configuration.

        Returns
        -------
            ConfigLoader initialized with default configuration.
        """
        default_config = {
            "name": "Default Lunar Mission",
            "description": "Default configuration for lunar payload delivery mission",
            "payload": {
                "dry_mass": 2000.0,
                "payload_mass": 1000.0,
                "max_propellant_mass": 3000.0,
                "specific_impulse": 320.0
            },
            "cost_factors": {
                "launch_cost_per_kg": 10000.0,
                "operations_cost_per_day": 50000.0,
                "development_cost": 10000000.0,
                "contingency_percentage": 20.0
            },
            "mission_duration_days": 180.0,
            "target_orbit": {
                "semi_major_axis": 384400.0,
                "eccentricity": 0.0,
                "inclination": 0.0
            }
        }
        return cls(default_config=default_config)
