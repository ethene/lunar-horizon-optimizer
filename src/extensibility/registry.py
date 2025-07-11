"""Extension registry for the Task 10 extensibility framework.

This module provides centralized registration and discovery of extension classes,
allowing dynamic loading and management of extensions at runtime.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from collections import defaultdict

from .base_extension import BaseExtension, ExtensionType

logger = logging.getLogger(__name__)


class ExtensionRegistry:
    """Registry for managing extension class definitions and discovery.

    The registry maintains a mapping of extension names to their implementation
    classes, enabling dynamic loading and instantiation of extensions.
    """

    def __init__(self):
        """Initialize the extension registry."""
        self._extensions: Dict[str, Type[BaseExtension]] = {}
        self._extensions_by_type: Dict[ExtensionType, List[str]] = defaultdict(list)
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("Extension registry initialized")

    def register_extension_class(
        self,
        name: str,
        extension_class: Type[BaseExtension],
        extension_type: ExtensionType,
    ) -> bool:
        """Register an extension class with the registry.

        Args:
            name: Unique name for the extension
            extension_class: Extension class to register
            extension_type: Type of extension

        Returns:
            True if registration was successful
        """
        if not issubclass(extension_class, BaseExtension):
            logger.error(f"Extension {name} must inherit from BaseExtension")
            return False

        if name in self._extensions:
            logger.warning(f"Extension {name} already registered, overwriting")

        self._extensions[name] = extension_class

        # Update type mapping
        if name not in self._extensions_by_type[extension_type]:
            self._extensions_by_type[extension_type].append(name)

        logger.info(f"Registered extension class: {name} ({extension_type.value})")
        return True

    def unregister_extension_class(self, name: str) -> bool:
        """Unregister an extension class.

        Args:
            name: Name of extension to unregister

        Returns:
            True if unregistration was successful
        """
        if name not in self._extensions:
            logger.warning(f"Extension {name} not found for unregistration")
            return False

        # Remove from type mappings
        for _ext_type, names in self._extensions_by_type.items():
            if name in names:
                names.remove(name)

        # Remove from main registry
        del self._extensions[name]

        # Clear metadata cache
        if name in self._metadata_cache:
            del self._metadata_cache[name]

        logger.info(f"Unregistered extension class: {name}")
        return True

    def get_extension_class(self, name: str) -> Optional[Type[BaseExtension]]:
        """Get an extension class by name.

        Args:
            name: Name of the extension

        Returns:
            Extension class or None if not found
        """
        return self._extensions.get(name)

    def get_extension_classes_by_type(
        self, extension_type: ExtensionType
    ) -> List[Type[BaseExtension]]:
        """Get all extension classes of a specific type.

        Args:
            extension_type: Type of extensions to retrieve

        Returns:
            List of extension classes
        """
        names = self._extensions_by_type.get(extension_type, [])
        return [self._extensions[name] for name in names if name in self._extensions]

    def list_registered_extensions(self) -> List[Dict[str, Any]]:
        """List all registered extension classes.

        Returns:
            List of extension information dictionaries
        """
        extensions = []

        for name, ext_class in self._extensions.items():
            # Find extension type
            ext_type = None
            for etype, names in self._extensions_by_type.items():
                if name in names:
                    ext_type = etype
                    break

            extensions.append(
                {
                    "name": name,
                    "class": ext_class.__name__,
                    "module": ext_class.__module__,
                    "type": ext_type.value if ext_type else "unknown",
                    "docstring": ext_class.__doc__ or "",
                }
            )

        return extensions

    def get_extension_count(self) -> int:
        """Get total number of registered extensions."""
        return len(self._extensions)

    def get_extension_count_by_type(self) -> Dict[str, int]:
        """Get extension counts by type.

        Returns:
            Dictionary mapping extension type to count
        """
        return {
            ext_type.value: len(names)
            for ext_type, names in self._extensions_by_type.items()
        }

    def validate_extension_class(self, extension_class: Type[BaseExtension]) -> bool:
        """Validate that an extension class meets requirements.

        Args:
            extension_class: Extension class to validate

        Returns:
            True if class is valid
        """
        try:
            # Check inheritance
            if not issubclass(extension_class, BaseExtension):
                logger.error("Extension must inherit from BaseExtension")
                return False

            # Check required methods are implemented
            required_methods = [
                "initialize",
                "validate_configuration",
                "get_capabilities",
            ]
            for method in required_methods:
                if not hasattr(extension_class, method):
                    logger.error(f"Extension missing required method: {method}")
                    return False

                # Check method is not abstract
                method_obj = getattr(extension_class, method)
                if getattr(method_obj, "__isabstractmethod__", False):
                    logger.error(
                        f"Extension has unimplemented abstract method: {method}"
                    )
                    return False

            return True

        except Exception as e:
            logger.error(f"Extension validation failed: {e}")
            return False

    def find_extensions_by_capability(self, capability: str) -> List[str]:
        """Find extensions that provide a specific capability.

        Args:
            capability: Capability to search for

        Returns:
            List of extension names that provide the capability
        """
        matching_extensions = []

        for name, ext_class in self._extensions.items():
            try:
                # Create temporary instance to check capabilities
                # This is not ideal but needed for capability checking
                # In practice, capabilities should be class-level metadata
                if hasattr(ext_class, "get_static_capabilities"):
                    capabilities = ext_class.get_static_capabilities()
                    if capability in capabilities:
                        matching_extensions.append(name)

            except Exception as e:
                logger.warning(f"Could not check capabilities for {name}: {e}")

        return matching_extensions

    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status.

        Returns:
            Dictionary with registry status information
        """
        return {
            "total_extensions": len(self._extensions),
            "extensions_by_type": {
                ext_type.value: len(names)
                for ext_type, names in self._extensions_by_type.items()
            },
            "registered_extensions": list(self._extensions.keys()),
            "cache_size": len(self._metadata_cache),
        }

    def clear_registry(self) -> None:
        """Clear all registered extensions."""
        self._extensions.clear()
        self._extensions_by_type.clear()
        self._metadata_cache.clear()

        logger.info("Extension registry cleared")

    def import_extension_module(self, module_path: str) -> int:
        """Import an extension module and auto-register discovered extensions.

        Args:
            module_path: Python module path to import

        Returns:
            Number of extensions registered from the module
        """
        try:
            import importlib

            module = importlib.import_module(module_path)

            registered_count = 0

            # Look for extension classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's an extension class
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseExtension)
                    and attr != BaseExtension
                ):

                    # Try to determine extension type from class
                    extension_type = getattr(attr, "EXTENSION_TYPE", None)
                    if extension_type and isinstance(extension_type, ExtensionType):
                        if self.register_extension_class(
                            attr_name, attr, extension_type
                        ):
                            registered_count += 1
                    else:
                        logger.warning(
                            f"Extension {attr_name} missing EXTENSION_TYPE attribute"
                        )

            logger.info(f"Imported {registered_count} extensions from {module_path}")
            return registered_count

        except Exception as e:
            logger.error(f"Failed to import extension module {module_path}: {e}")
            return 0


# Global registry instance
_global_registry = ExtensionRegistry()


def get_global_registry() -> ExtensionRegistry:
    """Get the global extension registry instance."""
    return _global_registry


def register_extension(
    name: str, extension_class: Type[BaseExtension], extension_type: ExtensionType
) -> bool:
    """Convenience function to register an extension with the global registry."""
    return _global_registry.register_extension_class(
        name, extension_class, extension_type
    )


def get_extension_class(name: str) -> Optional[Type[BaseExtension]]:
    """Convenience function to get an extension class from the global registry."""
    return _global_registry.get_extension_class(name)
