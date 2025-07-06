"""Configuration management module.

This module provides functionality for managing mission configurations,
including loading, saving, and validation. It supports both JSON and YAML
formats and provides template-based configuration creation.

Example Usage:
    ```python
    from pathlib import Path
    from src.config.manager import ConfigManager
    
    # Initialize manager
    manager = ConfigManager()
    
    # Create from template
    config = manager.create_from_template("lunar_delivery", 
        name="Custom Mission",
        payload={
            "dry_mass": 2000.0,
            "payload_mass": 1000.0,
            "max_propellant_mass": 3000.0,
            "specific_impulse": 320.0
        }
    )
    
    # Save configuration
    manager.save_config(Path("mission_config.yaml"))
    
    # Load existing configuration
    config = manager.load_config(Path("mission_config.yaml"))
    
    # Update configuration
    manager.update_config({
        "mission_duration_days": 200,
        "cost_factors": {
            "contingency_percentage": 25.0
        }
    })
    ```

Configuration Schema:
    ```yaml
    name: str  # Required: Mission name
    description: str | None  # Optional: Detailed description
    
    payload:  # Required: Spacecraft specifications
      dry_mass: float  # kg, > 0
      payload_mass: float  # kg, > 0, < dry_mass
      max_propellant_mass: float  # kg, > 0
      specific_impulse: float  # seconds, > 0
    
    cost_factors:  # Required: Economic parameters
      launch_cost_per_kg: float  # USD/kg, > 0
      operations_cost_per_day: float  # USD/day, > 0
      development_cost: float  # USD, > 0
      contingency_percentage: float | None  # %, 0-100, default: 20.0
    
    mission_duration_days: float  # Required: days, > 0
    
    target_orbit:  # Required: Orbital parameters
      semi_major_axis: float  # km, > 0
      eccentricity: float  # 0-1
      inclination: float  # degrees, 0-180
    
    isru_targets:  # Optional: List of ISRU production targets
      - resource_type: water | oxygen | hydrogen | methane | metals
        target_rate: float  # kg/day, > 0
        setup_time_days: float  # days, >= 0
        market_value_per_kg: float  # USD/kg, > 0
    ```

Available Templates:
    - lunar_delivery: Basic lunar payload delivery mission
    - lunar_isru: Lunar mission with ISRU capabilities

Notes:
    - All mass values are in kilograms
    - All cost values are in USD
    - Time values are in days
    - Orbital parameters use kilometers and degrees
    - ISRU production rates are in kg/day
"""

from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel

from .models import MissionConfig
from .loader import ConfigLoader, ConfigurationError
from .registry import ConfigRegistry

class ConfigManager:
    """Manager for handling mission configuration.
    
    This class provides a high-level interface for managing mission configurations,
    including loading from files, saving to files, creating from templates, and
    updating configurations. It ensures all configurations are properly validated
    against the mission configuration schema.
    
    Attributes:
        registry (ConfigRegistry): Registry containing configuration templates
        loader (ConfigLoader): Loader for reading/writing configuration files
        _active_config (Optional[MissionConfig]): Currently active configuration
        
    Example:
        ```python
        manager = ConfigManager()
        
        # Load from file
        config = manager.load_config("config.yaml")
        
        # Create from template
        config = manager.create_from_template("lunar_delivery")
        
        # Update configuration
        manager.update_config({
            "name": "Updated Mission",
            "mission_duration_days": 200
        })
        
        # Save to file
        manager.save_config("updated_config.yaml")
        ```
    """
    
    def __init__(self, registry: Optional[ConfigRegistry] = None):
        """Initialize the configuration manager.
        
        Args:
            registry: Optional ConfigRegistry instance for templates.
                     If not provided, creates a new registry with default templates.
        """
        self.registry = registry or ConfigRegistry()
        self.loader = ConfigLoader()
        self._active_config: Optional[MissionConfig] = None
    
    @property
    def active_config(self) -> Optional[MissionConfig]:
        """Get the currently active configuration.
        
        Returns:
            Currently active MissionConfig or None if not set.
            
        Example:
            ```python
            manager = ConfigManager()
            config = manager.active_config  # None initially
            
            manager.load_config("config.yaml")
            config = manager.active_config  # Returns loaded config
            ```
        """
        return self._active_config
    
    def load_config(self, file_path: Path) -> MissionConfig:
        """Load a configuration from file and set as active.
        
        Supports both JSON (.json) and YAML (.yml, .yaml) file formats.
        The loaded configuration is automatically validated against the
        mission configuration schema.
        
        Args:
            file_path: Path to the configuration file.
            
        Returns:
            Loaded and validated configuration.
            
        Raises:
            ConfigurationError: If file cannot be loaded or validation fails.
            
        Example:
            ```python
            manager = ConfigManager()
            
            # Load JSON config
            config = manager.load_config("mission.json")
            
            # Load YAML config
            config = manager.load_config("mission.yaml")
            ```
        """
        config = self.loader.load_file(file_path)
        self._active_config = config
        return config
    
    def save_config(self, file_path: Path) -> None:
        """Save the active configuration to a file.
        
        Supports both JSON (.json) and YAML (.yml, .yaml) file formats.
        The file format is determined by the file extension.
        
        Args:
            file_path: Path where to save the configuration.
            
        Raises:
            ConfigurationError: If no active configuration or save fails.
            
        Example:
            ```python
            manager = ConfigManager()
            config = manager.create_from_template("lunar_delivery")
            
            # Save as JSON
            manager.save_config("mission.json")
            
            # Save as YAML
            manager.save_config("mission.yaml")
            ```
        """
        if not self._active_config:
            raise ConfigurationError("No active configuration to save")
            
        self.loader.save_config(self._active_config, file_path)
    
    def create_from_template(self, template_name: str, **overrides) -> MissionConfig:
        """Create a new configuration from a template.
        
        Creates a new configuration based on a named template, with optional
        overrides for any configuration values. The resulting configuration
        is automatically validated.
        
        Args:
            template_name: Name of the template to use.
            **overrides: Values to override in the template.
            
        Returns:
            New configuration based on the template.
            
        Raises:
            KeyError: If template does not exist.
            ValidationError: If overrides are invalid.
            
        Example:
            ```python
            manager = ConfigManager()
            
            # Create from template with overrides
            config = manager.create_from_template("lunar_delivery",
                name="Custom Mission",
                mission_duration_days=200,
                payload={"dry_mass": 2500.0}
            )
            ```
        """
        template = self.registry.get_template(template_name)
        
        # Apply overrides
        config_dict = template.model_dump()
        config_dict.update(overrides)
        
        config = MissionConfig.model_validate(config_dict)
        self._active_config = config
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> MissionConfig:
        """Validate a configuration dictionary against the mission config model.
        
        Validates the provided configuration dictionary against the MissionConfig
        schema, ensuring all required fields are present and valid.
        
        Args:
            config: Configuration dictionary to validate.
            
        Returns:
            Validated MissionConfig instance.
            
        Raises:
            ValidationError: If configuration is invalid.
            
        Example:
            ```python
            manager = ConfigManager()
            
            config_dict = {
                "name": "Test Mission",
                "payload": {
                    "dry_mass": 1000.0,
                    "payload_mass": 500.0,
                    "max_propellant_mass": 2000.0,
                    "specific_impulse": 300.0
                },
                # ... other required fields ...
            }
            
            try:
                validated = manager.validate_config(config_dict)
            except ValidationError as e:
                print(f"Invalid configuration: {e}")
            ```
        """
        return MissionConfig.model_validate(config)
    
    def update_config(self, updates: Dict[str, Any]) -> MissionConfig:
        """Update the active configuration with new values.
        
        Updates the active configuration with new values, performing a deep
        update that preserves nested structure. The updated configuration
        is automatically validated.
        
        Args:
            updates: Dictionary of values to update.
            
        Returns:
            Updated configuration.
            
        Raises:
            ConfigurationError: If no active configuration.
            ValidationError: If updates are invalid.
            
        Example:
            ```python
            manager = ConfigManager()
            config = manager.create_from_template("lunar_delivery")
            
            # Update multiple fields
            updated = manager.update_config({
                "name": "Updated Mission",
                "mission_duration_days": 200,
                "payload": {
                    "dry_mass": 2500.0
                }
            })
            ```
        """
        if not self._active_config:
            raise ConfigurationError("No active configuration to update")
            
        # Create updated configuration dictionary
        config_dict = self._active_config.model_dump()
        
        # Deep update nested dictionaries
        def deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        config_dict = deep_update(config_dict, updates)
        
        # Validate and apply updates
        updated_config = MissionConfig.model_validate(config_dict)
        self._active_config = updated_config
        return updated_config 