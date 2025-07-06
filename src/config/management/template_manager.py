"""Template management for configuration creation.

This module handles template-based configuration creation,
providing functionality to create configurations from predefined templates
with custom overrides.
"""

from typing import Dict, Any, Optional

from ..models import MissionConfig
from ..registry import ConfigRegistry
from pydantic import ValidationError


class TemplateManager:
    """Manages configuration templates and template-based creation.
    
    This class handles all template-related operations, including
    retrieving templates, applying overrides, and creating new
    configurations from templates.
    """
    
    def __init__(self, registry: Optional[ConfigRegistry] = None):
        """Initialize template manager.
        
        Args:
            registry: Optional ConfigRegistry instance for templates.
                     If not provided, creates a new registry with default templates.
        """
        self.registry = registry or ConfigRegistry()
    
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
        """
        template = self.registry.get_template(template_name)
        
        # Apply overrides
        config_dict = template.model_dump()
        config_dict = self._apply_overrides(config_dict, overrides)
        
        return MissionConfig.model_validate(config_dict)
    
    def get_available_templates(self) -> list[str]:
        """Get list of available template names.
        
        Returns:
            List of available template names.
        """
        return self.registry.list_templates()
    
    def get_template(self, template_name: str) -> MissionConfig:
        """Get a specific template by name.
        
        Args:
            template_name: Name of the template to retrieve.
            
        Returns:
            Template configuration.
            
        Raises:
            KeyError: If template does not exist.
        """
        return self.registry.get_template(template_name)
    
    def validate_template_overrides(self, template_name: str, overrides: Dict[str, Any]) -> bool:
        """Validate that template overrides will produce a valid configuration.
        
        Args:
            template_name: Name of the template to use.
            overrides: Override values to validate.
            
        Returns:
            True if overrides are valid.
            
        Raises:
            KeyError: If template does not exist.
            ValidationError: If overrides are invalid.
        """
        try:
            self.create_from_template(template_name, **overrides)
            return True
        except ValidationError:
            return False
    
    def _apply_overrides(self, config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply override values to a configuration dictionary.
        
        Performs deep update of nested dictionaries to preserve structure
        while applying override values.
        
        Args:
            config_dict: Base configuration dictionary.
            overrides: Override values to apply.
            
        Returns:
            Updated configuration dictionary.
        """
        # Deep update nested dictionaries
        def deep_update(d: dict, u: dict) -> dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        return deep_update(config_dict.copy(), overrides)