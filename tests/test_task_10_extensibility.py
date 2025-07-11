"""
Comprehensive test suite for Task 10 - Extensibility Interface.

This module tests the complete extensibility framework including:
- Base extension functionality
- Extension manager and registry
- Data transformation layer
- Plugin interface
- Example extensions
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

from src.extensibility.base_extension import (
    BaseExtension,
    ExtensionMetadata,
    ExtensionType,
    FlightStageExtension,
)
from src.extensibility.extension_manager import ExtensionManager, ExtensionLoadError
from src.extensibility.registry import ExtensionRegistry, get_global_registry
from src.extensibility.data_transform import DataTransformLayer, DataFormat
from src.extensibility.plugin_interface import PluginInterface, PluginResult
from src.extensibility.examples.lunar_descent_extension import LunarDescentExtension
from src.extensibility.examples.custom_cost_model import CustomCostModel


class TestExtensionMetadata:
    """Test extension metadata functionality."""

    def test_extension_metadata_creation(self):
        """Test basic extension metadata creation."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.FLIGHT_STAGE,
        )

        assert metadata.name == "test_extension"
        assert metadata.version == "1.0.0"
        assert metadata.extension_type == ExtensionType.FLIGHT_STAGE
        assert metadata.enabled is True
        assert metadata.required_dependencies == []
        assert metadata.optional_dependencies == []

    def test_extension_metadata_with_dependencies(self):
        """Test extension metadata with dependencies."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.OPTIMIZER,
            required_dependencies=["numpy", "scipy"],
            optional_dependencies=["matplotlib"],
        )

        assert metadata.required_dependencies == ["numpy", "scipy"]
        assert metadata.optional_dependencies == ["matplotlib"]


class MockExtension(BaseExtension):
    """Mock extension for testing."""

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def validate_configuration(self) -> bool:
        return True

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock", "test": True}


class TestBaseExtension:
    """Test base extension functionality."""

    def test_base_extension_creation(self):
        """Test base extension creation and properties."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.VISUALIZER,
        )

        extension = MockExtension(metadata)

        assert extension.metadata.name == "test_extension"
        assert extension.is_enabled is True
        assert extension.is_initialized is False

        # Test initialization
        assert extension.initialize() is True
        assert extension.is_initialized is True

    def test_extension_enable_disable(self):
        """Test extension enable/disable functionality."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.DATA_PROCESSOR,
        )

        extension = MockExtension(metadata)

        assert extension.is_enabled is True

        extension.disable()
        assert extension.is_enabled is False

        extension.enable()
        assert extension.is_enabled is True

    def test_extension_status(self):
        """Test extension status reporting."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.TRAJECTORY_ANALYZER,
        )

        extension = MockExtension(metadata)
        extension.initialize()

        status = extension.get_status()

        assert status["name"] == "test_extension"
        assert status["version"] == "1.0.0"
        assert status["type"] == "trajectory_analyzer"
        assert status["enabled"] is True
        assert status["initialized"] is True
        assert status["configuration_valid"] is True


class TestExtensionRegistry:
    """Test extension registry functionality."""

    def setup_method(self):
        """Set up test registry."""
        self.registry = ExtensionRegistry()

    def test_registry_initialization(self):
        """Test registry initialization."""
        assert self.registry.get_extension_count() == 0
        assert len(self.registry.list_registered_extensions()) == 0

    def test_register_extension_class(self):
        """Test extension class registration."""
        result = self.registry.register_extension_class(
            "mock_extension", MockExtension, ExtensionType.VISUALIZER
        )

        assert result is True
        assert self.registry.get_extension_count() == 1

        ext_class = self.registry.get_extension_class("mock_extension")
        assert ext_class == MockExtension

    def test_get_extensions_by_type(self):
        """Test getting extensions by type."""
        self.registry.register_extension_class(
            "mock1", MockExtension, ExtensionType.VISUALIZER
        )
        self.registry.register_extension_class(
            "mock2", MockExtension, ExtensionType.OPTIMIZER
        )

        visualizers = self.registry.get_extension_classes_by_type(
            ExtensionType.VISUALIZER
        )
        optimizers = self.registry.get_extension_classes_by_type(
            ExtensionType.OPTIMIZER
        )

        assert len(visualizers) == 1
        assert len(optimizers) == 1
        assert visualizers[0] == MockExtension

    def test_unregister_extension(self):
        """Test extension unregistration."""
        self.registry.register_extension_class(
            "mock_extension", MockExtension, ExtensionType.COST_MODEL
        )

        assert self.registry.get_extension_count() == 1

        result = self.registry.unregister_extension_class("mock_extension")
        assert result is True
        assert self.registry.get_extension_count() == 0

    def test_list_registered_extensions(self):
        """Test listing registered extensions."""
        self.registry.register_extension_class(
            "mock_extension", MockExtension, ExtensionType.DATA_PROCESSOR
        )

        extensions = self.registry.list_registered_extensions()

        assert len(extensions) == 1
        assert extensions[0]["name"] == "mock_extension"
        assert extensions[0]["class"] == "MockExtension"
        assert extensions[0]["type"] == "data_processor"


class TestExtensionManager:
    """Test extension manager functionality."""

    def setup_method(self):
        """Set up test manager."""
        self.manager = ExtensionManager()

    def test_manager_initialization(self):
        """Test manager initialization."""
        status = self.manager.get_system_status()

        assert status["manager_enabled"] is True
        assert status["total_extensions"] == 0
        assert len(status["enabled_extensions"]) == 0

    def test_register_extension(self):
        """Test extension registration with manager."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.FLIGHT_STAGE,
        )

        extension = MockExtension(metadata)
        result = self.manager.register_extension(extension)

        assert result is True
        assert self.manager.get_extension("test_extension") == extension

        status = self.manager.get_system_status()
        assert status["total_extensions"] == 1

    def test_unregister_extension(self):
        """Test extension unregistration."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.OPTIMIZER,
        )

        extension = MockExtension(metadata)
        self.manager.register_extension(extension)

        result = self.manager.unregister_extension("test_extension")
        assert result is True
        assert self.manager.get_extension("test_extension") is None

    def test_get_extensions_by_type(self):
        """Test getting extensions by type from manager."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.TRAJECTORY_ANALYZER,
        )

        extension = MockExtension(metadata)
        self.manager.register_extension(extension)

        analyzers = self.manager.get_extensions_by_type(
            ExtensionType.TRAJECTORY_ANALYZER
        )
        assert len(analyzers) == 1
        assert analyzers[0] == extension

    def test_enable_disable_extension(self):
        """Test enabling/disabling extensions through manager."""
        metadata = ExtensionMetadata(
            name="test_extension",
            version="1.0.0",
            description="Test extension",
            author="Test Author",
            extension_type=ExtensionType.COST_MODEL,
        )

        extension = MockExtension(metadata)
        self.manager.register_extension(extension)

        # Test disable
        result = self.manager.disable_extension("test_extension")
        assert result is True
        assert not extension.is_enabled

        # Test enable
        result = self.manager.enable_extension("test_extension")
        assert result is True
        assert extension.is_enabled


class TestDataTransformLayer:
    """Test data transformation layer."""

    def setup_method(self):
        """Set up test transformer."""
        self.transformer = DataTransformLayer()

    def test_normalize_trajectory_state(self):
        """Test trajectory state normalization."""
        input_state = {
            "position": [1000.0, 2000.0, 3000.0],
            "velocity": [1.0, 2.0, 3.0],
            "time": 12345.0,
            "mass": 5000.0,
        }

        normalized = self.transformer.normalize_trajectory_state(input_state)

        assert normalized["position"] == [1000.0, 2000.0, 3000.0]
        assert normalized["velocity"] == [1.0, 2.0, 3.0]
        assert normalized["time"] == 12345.0
        assert normalized["mass"] == 5000.0

    def test_normalize_optimization_result(self):
        """Test optimization result normalization."""
        input_result = {
            "objectives": {"delta_v": 3000, "time": 4 * 86400, "cost": 240e6},
            "parameters": {"earth_orbit_alt": 400, "moon_orbit_alt": 100},
            "constraints": {"max_mass": 5000},
        }

        normalized = self.transformer.normalize_optimization_result(input_result)

        assert normalized["objectives"] == [3000, 4 * 86400, 240e6]
        assert normalized["parameters"]["earth_orbit_alt"] == 400
        assert normalized["constraints"]["max_mass"] == 5000

    def test_normalize_cost_breakdown(self):
        """Test cost breakdown normalization."""
        input_costs = {
            "development": 100.0,
            "launch": 200.0,
            "spacecraft": 150.0,
            "operations": 50.0,
            "ground_systems": 30.0,
            "contingency": 20.0,
        }

        normalized = self.transformer.normalize_cost_breakdown(input_costs)

        assert normalized["development"] == 100.0
        assert normalized["total"] == 550.0  # Sum of all components

    def test_unit_conversion(self):
        """Test unit conversion functionality."""
        # Test length conversion
        km_to_m = self.transformer.convert_units(1.0, "km", "m")
        assert km_to_m == 1000.0

        # Test velocity conversion
        kmh_to_ms = self.transformer.convert_units(36.0, "km/h", "m/s")
        assert abs(kmh_to_ms - 10.0) < 0.01

        # Test same unit
        same_unit = self.transformer.convert_units(100.0, "kg", "kg")
        assert same_unit == 100.0

    def test_validate_extension_data(self):
        """Test extension data validation."""
        valid_data = {
            "initial_state": {"position": [0, 0, 0], "velocity": [0, 0, 0]},
            "target_state": {"position": [1000, 0, 0], "velocity": [0, 0, 0]},
        }

        result = self.transformer.validate_extension_data(valid_data, "flight_stage")
        assert result is True


class TestPluginInterface:
    """Test plugin interface functionality."""

    def setup_method(self):
        """Set up test plugin interface."""
        self.interface = PluginInterface()

    def test_plugin_interface_initialization(self):
        """Test plugin interface initialization."""
        assert self.interface.data_transform is not None
        assert len(self.interface._registered_plugins) == 0

    def test_register_plugin(self):
        """Test plugin registration."""
        metadata = ExtensionMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            extension_type=ExtensionType.FLIGHT_STAGE,
        )

        plugin = MockExtension(metadata)
        result = self.interface.register_plugin(plugin)

        assert result is True
        assert len(self.interface._registered_plugins) == 1

    def test_get_trajectory_planners(self):
        """Test getting trajectory planners."""
        metadata = ExtensionMetadata(
            name="trajectory_plugin",
            version="1.0.0",
            description="Trajectory plugin",
            author="Test Author",
            extension_type=ExtensionType.FLIGHT_STAGE,
        )

        plugin = MockExtension(metadata)
        self.interface.register_plugin(plugin)

        planners = self.interface.get_trajectory_planners()
        assert "trajectory_plugin" in planners

    def test_plugin_result_creation(self):
        """Test plugin result creation."""
        result = PluginResult(
            success=True,
            data={"test": "data"},
            metadata={"plugin": "test"},
            errors=[],
            warnings=["test warning"],
        )

        assert result.success is True
        assert result.data["test"] == "data"
        assert result.warnings == ["test warning"]
        assert result.errors == []


class TestLunarDescentExtension:
    """Test lunar descent extension example."""

    def setup_method(self):
        """Set up test extension."""
        self.extension = LunarDescentExtension()

    def test_lunar_descent_initialization(self):
        """Test lunar descent extension initialization."""
        result = self.extension.initialize()
        assert result is True
        assert self.extension.is_initialized is True

    def test_lunar_descent_validation(self):
        """Test configuration validation."""
        result = self.extension.validate_configuration()
        assert result is True

    def test_lunar_descent_capabilities(self):
        """Test getting capabilities."""
        capabilities = self.extension.get_capabilities()

        assert capabilities["type"] == "flight_stage"
        assert capabilities["stage_name"] == "lunar_descent"
        assert capabilities["provides_trajectory_planning"] is True
        assert capabilities["provides_delta_v_calculation"] is True
        assert capabilities["provides_cost_estimation"] is True

    def test_lunar_descent_trajectory_planning(self):
        """Test trajectory planning functionality."""
        self.extension.initialize()

        initial_state = {
            "position": [0, 0, 1737400 + 100000],  # 100 km altitude
            "velocity": [1000, 0, 0],  # Orbital velocity
            "mass": 5000.0,
        }

        target_state = {
            "position": [0, 0, 1737400],  # Surface
            "velocity": [0, 0, 0],
        }

        constraints = {
            "max_thrust": 45000.0,
            "specific_impulse": 320.0,
        }

        result = self.extension.plan_trajectory(
            initial_state, target_state, constraints
        )

        assert result["success"] is True
        assert "trajectory" in result
        assert "performance_metrics" in result
        assert "delta_v" in result["performance_metrics"]

    def test_lunar_descent_cost_estimation(self):
        """Test cost estimation."""
        self.extension.initialize()

        # Mock trajectory data
        trajectory = {
            "performance_metrics": {
                "fuel_consumption": 1000.0,  # kg
                "flight_time": 1800.0,  # seconds
            }
        }

        costs = self.extension.estimate_cost(trajectory)

        assert "total" in costs
        assert costs["total"] > 0
        assert "fuel" in costs
        assert "operations" in costs
        assert "hardware" in costs


class TestCustomCostModel:
    """Test custom cost model extension example."""

    def setup_method(self):
        """Set up test extension."""
        self.extension = CustomCostModel()

    def test_custom_cost_model_initialization(self):
        """Test custom cost model initialization."""
        result = self.extension.initialize()
        assert result is True
        assert self.extension.is_initialized is True

    def test_custom_cost_model_validation(self):
        """Test configuration validation."""
        result = self.extension.validate_configuration()
        assert result is True

    def test_custom_cost_model_capabilities(self):
        """Test getting capabilities."""
        capabilities = self.extension.get_capabilities()

        assert capabilities["type"] == "cost_model"
        assert capabilities["provides_cost_estimation"] is True
        assert capabilities["provides_risk_analysis"] is True
        assert "parametric_scaling" in capabilities["estimation_methods"]

    def test_mission_cost_estimation(self):
        """Test mission cost estimation."""
        self.extension.initialize()

        mission_params = {
            "mission_type": "lunar_transport",
            "payload_mass": 1500.0,
            "mission_duration": 30.0,
            "complexity_factor": 1.2,
            "technology_readiness": 6,
        }

        result = self.extension.estimate_mission_cost(mission_params)

        assert "total_cost" in result
        assert result["total_cost"] > 0
        assert "cost_breakdown" in result
        assert "confidence_intervals" in result
        assert "cost_drivers" in result

    def test_cost_breakdown_analysis(self):
        """Test detailed cost breakdown."""
        self.extension.initialize()

        cost_estimate = {
            "cost_breakdown": {
                "hardware": 100.0,
                "software": 20.0,
                "labor": 50.0,
                "operations": 30.0,
            }
        }

        breakdown = self.extension.breakdown_costs(cost_estimate)

        assert "propulsion_hardware" in breakdown
        assert "flight_software" in breakdown
        assert "engineering_labor" in breakdown
        assert "mission_operations" in breakdown

    def test_cost_driver_analysis(self):
        """Test cost driver analysis."""
        self.extension.initialize()

        mission_params = {
            "payload_mass": 3000.0,  # High payload
            "mission_duration": 48.0,  # Long duration
            "technology_readiness": 5,  # Low TRL
            "complexity_factor": 2.0,  # High complexity
        }

        drivers = self.extension.analyze_cost_drivers(mission_params)

        assert len(drivers) > 0
        assert any(d["driver"] == "payload_mass" for d in drivers)
        assert any(d["driver"] == "technology_readiness" for d in drivers)

        # Check that high impact drivers are first
        high_impact_drivers = [d for d in drivers if d["impact"] == "high"]
        assert len(high_impact_drivers) > 0


class TestExtensibilityIntegration:
    """Test integration between extensibility components."""

    def setup_method(self):
        """Set up integration test components."""
        self.manager = ExtensionManager()
        self.interface = PluginInterface()

    def test_end_to_end_extension_workflow(self):
        """Test complete extension workflow."""
        # Create and register lunar descent extension
        lunar_descent = LunarDescentExtension()
        lunar_descent.initialize()

        # Register with manager
        manager_result = self.manager.register_extension(lunar_descent)
        assert manager_result is True

        # Register with plugin interface
        interface_result = self.interface.register_plugin(lunar_descent)
        assert interface_result is True

        # Test trajectory planning through interface
        initial_state = {
            "position": [0, 0, 1737400 + 50000],
            "velocity": [800, 0, 0],
            "mass": 4000.0,
        }

        target_state = {
            "position": [0, 0, 1737400],
            "velocity": [0, 0, 0],
        }

        result = self.interface.plan_trajectory(
            "lunar_descent", initial_state, target_state
        )

        assert result.success is True
        assert "trajectory" in result.data

    def test_multiple_extension_types(self):
        """Test managing multiple extension types."""
        # Register different extension types
        lunar_descent = LunarDescentExtension()
        lunar_descent.initialize()

        cost_model = CustomCostModel()
        cost_model.initialize()

        self.manager.register_extension(lunar_descent)
        self.manager.register_extension(cost_model)

        # Test getting extensions by type
        flight_stages = self.manager.get_extensions_by_type(ExtensionType.FLIGHT_STAGE)
        cost_models = self.manager.get_extensions_by_type(ExtensionType.COST_MODEL)

        assert len(flight_stages) == 1
        assert len(cost_models) == 1

        # Test system status
        status = self.manager.get_system_status()
        assert status["total_extensions"] == 2
        assert status["extensions_by_type"]["flight_stage"] == 1
        assert status["extensions_by_type"]["cost_model"] == 1

    def test_data_transformation_integration(self):
        """Test data transformation with extensions."""
        transformer = DataTransformLayer()

        # Test trajectory state transformation
        raw_state = {
            "position": np.array([1000, 2000, 3000]),
            "velocity": np.array([10, 20, 30]),
            "time": 12345.0,
        }

        normalized = transformer.normalize_trajectory_state(raw_state)

        assert isinstance(normalized["position"], list)
        assert isinstance(normalized["velocity"], list)
        assert normalized["position"] == [1000.0, 2000.0, 3000.0]

        # Test with lunar descent extension
        lunar_descent = LunarDescentExtension()
        lunar_descent.initialize()

        # Extension should work with normalized data
        initial_state = normalized.copy()
        initial_state["mass"] = 5000.0

        target_state = {
            "position": [0, 0, 1737400],
            "velocity": [0, 0, 0],
        }

        result = lunar_descent.plan_trajectory(initial_state, target_state)
        assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
