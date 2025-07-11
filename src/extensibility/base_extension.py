"""Base extension interface for Task 10 extensibility framework.

This module defines the foundational classes and interfaces that all
extensions must implement to integrate with the Lunar Horizon Optimizer.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ExtensionType(Enum):
    """Types of extensions supported by the system."""

    FLIGHT_STAGE = "flight_stage"
    TRAJECTORY_ANALYZER = "trajectory_analyzer"
    COST_MODEL = "cost_model"
    OPTIMIZER = "optimizer"
    VISUALIZER = "visualizer"
    DATA_PROCESSOR = "data_processor"


@dataclass
class ExtensionMetadata:
    """Metadata describing an extension.

    This class contains all the information needed to register,
    load, and manage an extension within the system.
    """

    name: str
    version: str
    description: str
    author: str
    extension_type: ExtensionType
    required_dependencies: List[str] = None
    optional_dependencies: List[str] = None
    api_version: str = "1.0"
    enabled: bool = True
    configuration_schema: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.required_dependencies is None:
            self.required_dependencies = []
        if self.optional_dependencies is None:
            self.optional_dependencies = []
        if self.configuration_schema is None:
            self.configuration_schema = {}


class BaseExtension(ABC):
    """Abstract base class for all extensions.

    All extensions must inherit from this class and implement
    the required methods to integrate with the system.
    """

    def __init__(
        self, metadata: ExtensionMetadata, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the extension.

        Args:
            metadata: Extension metadata information
            config: Optional configuration dictionary
        """
        self.metadata = metadata
        self.config = config or {}
        self.logger = logging.getLogger(f"extension.{metadata.name}")
        self._initialized = False
        self._enabled = metadata.enabled

        logger.info(f"Created extension: {metadata.name} v{metadata.version}")

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the extension.

        This method is called during system startup to prepare
        the extension for use.

        Returns:
            True if initialization was successful, False otherwise
        """
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate the extension configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get the capabilities provided by this extension.

        Returns:
            Dictionary describing the extension's capabilities
        """
        pass

    def enable(self) -> None:
        """Enable the extension."""
        self._enabled = True
        self.logger.info(f"Extension {self.metadata.name} enabled")

    def disable(self) -> None:
        """Disable the extension."""
        self._enabled = False
        self.logger.info(f"Extension {self.metadata.name} disabled")

    @property
    def is_enabled(self) -> bool:
        """Check if the extension is enabled."""
        return self._enabled

    @property
    def is_initialized(self) -> bool:
        """Check if the extension is initialized."""
        return self._initialized

    def shutdown(self) -> None:
        """Shutdown the extension.

        This method is called during system shutdown to clean up
        any resources used by the extension.
        """
        self._initialized = False
        self.logger.info(f"Extension {self.metadata.name} shutdown")

    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update the extension configuration.

        Args:
            new_config: New configuration dictionary

        Returns:
            True if update was successful, False otherwise
        """
        if self.validate_configuration():
            self.config.update(new_config)
            self.logger.info(f"Configuration updated for {self.metadata.name}")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the extension.

        Returns:
            Dictionary containing status information
        """
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "type": self.metadata.extension_type.value,
            "enabled": self._enabled,
            "initialized": self._initialized,
            "configuration_valid": self.validate_configuration(),
        }

    def __str__(self) -> str:
        """String representation of the extension."""
        return f"{self.metadata.name} v{self.metadata.version} ({self.metadata.extension_type.value})"

    def __repr__(self) -> str:
        """Detailed string representation of the extension."""
        return (
            f"BaseExtension(name='{self.metadata.name}', "
            f"version='{self.metadata.version}', "
            f"type='{self.metadata.extension_type.value}', "
            f"enabled={self._enabled})"
        )


class FlightStageExtension(BaseExtension):
    """Specialized extension for flight stages.

    Flight stage extensions provide trajectory planning and analysis
    for specific phases of a mission (e.g., launch, lunar descent).
    """

    @abstractmethod
    def plan_trajectory(
        self,
        initial_state: Dict[str, Any],
        target_state: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Plan a trajectory for this flight stage.

        Args:
            initial_state: Initial state vector and parameters
            target_state: Target state vector and parameters
            constraints: Optional constraints for trajectory planning

        Returns:
            Dictionary containing trajectory data and performance metrics
        """
        pass

    @abstractmethod
    def calculate_delta_v(self, trajectory: Dict[str, Any]) -> float:
        """Calculate delta-v requirements for the trajectory.

        Args:
            trajectory: Trajectory data from plan_trajectory

        Returns:
            Total delta-v requirement in m/s
        """
        pass

    @abstractmethod
    def estimate_cost(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """Estimate costs for this flight stage.

        Args:
            trajectory: Trajectory data from plan_trajectory

        Returns:
            Dictionary of cost estimates by category
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Get flight stage capabilities."""
        return {
            "type": "flight_stage",
            "provides_trajectory_planning": True,
            "provides_delta_v_calculation": True,
            "provides_cost_estimation": True,
            "stage_name": getattr(self, "stage_name", "unknown"),
        }


def create_extension_from_config(config: Dict[str, Any]) -> Optional[BaseExtension]:
    """Factory function to create extensions from configuration.

    Args:
        config: Extension configuration dictionary

    Returns:
        Extension instance or None if creation failed
    """
    try:
        metadata = ExtensionMetadata(
            name=config["name"],
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

        # For now, return a generic BaseExtension
        # In practice, this would use a registry to find the appropriate class
        class GenericExtension(BaseExtension):
            def initialize(self) -> bool:
                self._initialized = True
                return True

            def validate_configuration(self) -> bool:
                return True

            def get_capabilities(self) -> Dict[str, Any]:
                return {"type": "generic"}

        return GenericExtension(metadata, config.get("config", {}))

    except Exception as e:
        logger.error(f"Failed to create extension from config: {e}")
        return None
