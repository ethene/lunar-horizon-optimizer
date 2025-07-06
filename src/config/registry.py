"""Configuration registry module.

This module provides functionality for managing and accessing configuration templates
and default configurations for different mission types.
"""

from pathlib import Path
from typing import Dict, Optional, List, Any
import json
import yaml
import logging

from .models import MissionConfig
from .loader import ConfigLoader, ConfigurationError

logger = logging.getLogger(__name__)

class ConfigRegistry:
    """Registry for managing configuration templates and defaults."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the configuration registry.
        
        Args:
            templates_dir: Optional directory containing configuration templates.
                         If not provided, uses default templates.
        """
        self.templates_dir = templates_dir
        self._templates: Dict[str, MissionConfig] = {}
        self._default_templates: Dict[str, MissionConfig] = {}
        self._load_default_templates()
        
        # Load custom templates if directory provided
        if templates_dir:
            self.load_templates_dir(templates_dir)
    
    def _load_default_templates(self) -> None:
        """Load built-in default configuration templates."""
        # Basic lunar delivery mission
        self._default_templates["lunar_delivery"] = MissionConfig.model_validate({
            "name": "Lunar Payload Delivery",
            "description": "Standard lunar payload delivery mission",
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
        })
        
        # ISRU mission template
        self._default_templates["lunar_isru"] = MissionConfig.model_validate({
            "name": "Lunar ISRU Mission",
            "description": "Lunar mission with ISRU capabilities",
            "payload": {
                "dry_mass": 3000.0,
                "payload_mass": 1500.0,
                "max_propellant_mass": 4000.0,
                "specific_impulse": 320.0
            },
            "cost_factors": {
                "launch_cost_per_kg": 10000.0,
                "operations_cost_per_day": 75000.0,
                "development_cost": 15000000.0,
                "contingency_percentage": 25.0
            },
            "mission_duration_days": 365.0,
            "target_orbit": {
                "semi_major_axis": 384400.0,
                "eccentricity": 0.0,
                "inclination": 0.0
            },
            "isru_targets": [
                {
                    "resource_type": "water",
                    "target_rate": 5.0,
                    "setup_time_days": 30.0,
                    "market_value_per_kg": 1000.0
                }
            ]
        })
        
        # Copy default templates to main templates dict
        self._templates.update(self._default_templates)
    
    def register_template(self, name: str, config: MissionConfig) -> None:
        """Register a new configuration template.
        
        Args:
            name: Unique name for the template.
            config: MissionConfig object to use as template.
            
        Raises:
            ValueError: If template name already exists in default templates.
        """
        if name in self._default_templates:
            raise ValueError(f"Cannot override default template '{name}'")
        self._templates[name] = config
    
    def get_template(self, name: str) -> MissionConfig:
        """Get a configuration template by name.
        
        Args:
            name: Name of the template to retrieve.
            
        Returns:
            Copy of the requested template configuration.
            
        Raises:
            KeyError: If template does not exist.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        return self._templates[name].model_copy(deep=True)
    
    def list_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of template names.
        """
        return list(self._templates.keys())
    
    def load_template_file(self, file_path: Path) -> None:
        """Load a template configuration from a file.
        
        The template name will be derived from the file name without extension.
        If a template with the same name already exists, it will be overwritten.
        
        Args:
            file_path: Path to the template file (JSON or YAML).
            
        Raises:
            ConfigurationError: If file cannot be loaded.
        """
        loader = ConfigLoader()
        try:
            config = loader.load_file(file_path)
            name = file_path.stem
            # Store the template without overwriting defaults
            self._templates[name] = config
            logger.info(f"Loaded template '{name}' from {file_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load template from {file_path}: {str(e)}")
    
    def load_templates_dir(self, directory: Path) -> None:
        """Load all template configurations from a directory.
        
        Args:
            directory: Path to directory containing template files.
            
        Raises:
            ConfigurationError: If directory cannot be processed.
        """
        if not directory.is_dir():
            raise ConfigurationError(f"Template directory not found: {directory}")
            
        for file_path in directory.glob("*.{json,yml,yaml}"):
            try:
                self.load_template_file(file_path)
                logger.info(f"Loaded template from {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load template {file_path}: {str(e)}")
    
    def save_template(self, name: str, file_path: Path) -> None:
        """Save a template configuration to a file.
        
        Args:
            name: Name of the template to save.
            file_path: Path where to save the template.
            
        Raises:
            KeyError: If template does not exist.
            ConfigurationError: If file cannot be saved.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
            
        loader = ConfigLoader()
        try:
            loader.save_config(self._templates[name], file_path)
        except Exception as e:
            raise ConfigurationError(f"Failed to save template to {file_path}: {str(e)}") 