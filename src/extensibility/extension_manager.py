"""Extension manager for the Task 10 extensibility framework.

This module provides centralized management of all extensions,
including loading, registration, lifecycle management, and coordination.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from collections import defaultdict

from .base_extension import BaseExtension, ExtensionMetadata, ExtensionType
from .registry import ExtensionRegistry
from .data_transform import DataTransformLayer

logger = logging.getLogger(__name__)


class ExtensionLoadError(Exception):
    """Exception raised when extension loading fails."""

    pass


class ExtensionManager:
    """Centralized manager for all system extensions.

    The ExtensionManager handles the complete lifecycle of extensions,
    from discovery and loading through execution and shutdown.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the extension manager.

        Args:
            config_path: Path to extension configuration file
        """
        self.registry = ExtensionRegistry()
        self.data_transform = DataTransformLayer()
        self.extensions: Dict[str, BaseExtension] = {}
        self.extension_types: Dict[ExtensionType, List[str]] = defaultdict(list)
        self.config_path = config_path
        self.enabled = True

        logger.info("Extension manager initialized")

    def load_extensions_from_config(self, config_path: Optional[str] = None) -> int:
        """Load extensions from configuration file.

        Args:
            config_path: Path to configuration file (overrides instance config)

        Returns:
            Number of extensions successfully loaded
        """
        config_file = Path(config_path or self.config_path or "extensions.json")

        if not config_file.exists():
            logger.warning(f"Extension config file not found: {config_file}")
            return 0

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            loaded_count = 0
            for ext_config in config.get("extensions", []):
                if self.load_extension_from_config(ext_config):
                    loaded_count += 1

            logger.info(f"Loaded {loaded_count} extensions from {config_file}")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load extensions from config: {e}")
            raise ExtensionLoadError(f"Config loading failed: {e}") from e

    def load_extension_from_config(self, config: Dict[str, Any]) -> bool:
        """Load a single extension from configuration.

        Args:
            config: Extension configuration dictionary

        Returns:
            True if extension was loaded successfully
        """
        try:
            name = config["name"]

            if name in self.extensions:
                logger.warning(f"Extension {name} already loaded, skipping")
                return False

            # Create extension metadata
            metadata = ExtensionMetadata(
                name=name,
                version=config["version"],
                description=config["description"],
                author=config["author"],
                extension_type=ExtensionType(config["type"]),
                required_dependencies=config.get("required_dependencies", []),
                optional_dependencies=config.get("optional_dependencies", []),
                api_version=config.get("api_version", "1.0"),
                enabled=config.get("enabled", True),
                configuration_schema=config.get("configuration_schema", {}),
            )

            # Get extension class from registry or create generic
            extension_class = self.registry.get_extension_class(name)
            if extension_class is None:
                logger.warning(
                    f"No registered class for {name}, using generic extension"
                )
                extension_class = self._create_generic_extension_class(
                    metadata.extension_type
                )

            # Create and register extension
            extension = extension_class(metadata, config.get("config", {}))
            return self.register_extension(extension)

        except Exception as e:
            logger.error(
                f"Failed to load extension {config.get('name', 'unknown')}: {e}"
            )
            return False

    def register_extension(self, extension: BaseExtension) -> bool:
        """Register an extension with the manager.

        Args:
            extension: Extension instance to register

        Returns:
            True if registration was successful
        """
        try:
            name = extension.metadata.name

            if name in self.extensions:
                logger.error(f"Extension {name} already registered")
                return False

            # Validate dependencies
            if not self._check_dependencies(extension):
                logger.error(f"Dependency check failed for {name}")
                return False

            # Validate configuration
            if not extension.validate_configuration():
                logger.error(f"Configuration validation failed for {name}")
                return False

            # Initialize extension
            if not extension.initialize():
                logger.error(f"Initialization failed for {name}")
                return False

            # Register with system
            self.extensions[name] = extension
            self.extension_types[extension.metadata.extension_type].append(name)

            logger.info(f"Successfully registered extension: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register extension: {e}")
            return False

    def unregister_extension(self, name: str) -> bool:
        """Unregister an extension.

        Args:
            name: Name of extension to unregister

        Returns:
            True if unregistration was successful
        """
        if name not in self.extensions:
            logger.warning(f"Extension {name} not found for unregistration")
            return False

        try:
            extension = self.extensions[name]

            # Shutdown extension
            extension.shutdown()

            # Remove from tracking
            del self.extensions[name]
            ext_type = extension.metadata.extension_type
            if name in self.extension_types[ext_type]:
                self.extension_types[ext_type].remove(name)

            logger.info(f"Successfully unregistered extension: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister extension {name}: {e}")
            return False

    def get_extension(self, name: str) -> Optional[BaseExtension]:
        """Get an extension by name.

        Args:
            name: Name of the extension

        Returns:
            Extension instance or None if not found
        """
        return self.extensions.get(name)

    def get_extensions_by_type(
        self, extension_type: ExtensionType
    ) -> List[BaseExtension]:
        """Get all extensions of a specific type.

        Args:
            extension_type: Type of extensions to retrieve

        Returns:
            List of extensions of the specified type
        """
        names = self.extension_types.get(extension_type, [])
        return [self.extensions[name] for name in names if name in self.extensions]

    def list_extensions(self) -> List[Dict[str, Any]]:
        """List all registered extensions.

        Returns:
            List of extension status dictionaries
        """
        return [ext.get_status() for ext in self.extensions.values()]

    def enable_extension(self, name: str) -> bool:
        """Enable an extension.

        Args:
            name: Name of extension to enable

        Returns:
            True if successful
        """
        extension = self.get_extension(name)
        if extension:
            extension.enable()
            return True
        return False

    def disable_extension(self, name: str) -> bool:
        """Disable an extension.

        Args:
            name: Name of extension to disable

        Returns:
            True if successful
        """
        extension = self.get_extension(name)
        if extension:
            extension.disable()
            return True
        return False

    def shutdown_all_extensions(self) -> None:
        """Shutdown all registered extensions."""
        logger.info("Shutting down all extensions")

        for extension in self.extensions.values():
            try:
                extension.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {extension.metadata.name}: {e}")

        self.extensions.clear()
        self.extension_types.clear()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.

        Returns:
            Dictionary with system status information
        """
        return {
            "manager_enabled": self.enabled,
            "total_extensions": len(self.extensions),
            "extensions_by_type": {
                ext_type.value: len(names)
                for ext_type, names in self.extension_types.items()
            },
            "enabled_extensions": [
                name for name, ext in self.extensions.items() if ext.is_enabled
            ],
            "disabled_extensions": [
                name for name, ext in self.extensions.items() if not ext.is_enabled
            ],
        }

    def _check_dependencies(self, extension: BaseExtension) -> bool:
        """Check if extension dependencies are satisfied.

        Args:
            extension: Extension to check

        Returns:
            True if all dependencies are satisfied
        """
        for dep in extension.metadata.required_dependencies:
            if dep not in self.extensions:
                logger.error(
                    f"Required dependency {dep} not found for {extension.metadata.name}"
                )
                return False

            if not self.extensions[dep].is_enabled:
                logger.error(
                    f"Required dependency {dep} is disabled for {extension.metadata.name}"
                )
                return False

        return True

    def _create_generic_extension_class(
        self, extension_type: ExtensionType
    ) -> Type[BaseExtension]:
        """Create a generic extension class for unknown extensions.

        Args:
            extension_type: Type of extension to create

        Returns:
            Generic extension class
        """

        class GenericExtension(BaseExtension):
            def initialize(self) -> bool:
                self._initialized = True
                self.logger.info(
                    f"Generic {extension_type.value} extension initialized"
                )
                return True

            def validate_configuration(self) -> bool:
                return True

            def get_capabilities(self) -> Dict[str, Any]:
                return {
                    "type": extension_type.value,
                    "generic": True,
                    "provides_basic_functionality": True,
                }

        return GenericExtension

    def execute_extension_method(
        self, extension_name: str, method_name: str, *args, **kwargs
    ) -> Any:
        """Execute a method on a specific extension.

        Args:
            extension_name: Name of the extension
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Method return value

        Raises:
            ValueError: If extension not found or method doesn't exist
        """
        extension = self.get_extension(extension_name)
        if not extension:
            raise ValueError(f"Extension {extension_name} not found")

        if not extension.is_enabled:
            raise ValueError(f"Extension {extension_name} is disabled")

        if not hasattr(extension, method_name):
            raise ValueError(f"Method {method_name} not found on {extension_name}")

        method = getattr(extension, method_name)
        return method(*args, **kwargs)
