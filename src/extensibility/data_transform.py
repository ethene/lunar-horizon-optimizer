"""Data transformation layer for the Task 10 extensibility framework.

This module provides standardized data transformation and validation
between different modules and extensions, ensuring data compatibility
and consistent interfaces across the system.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats for transformations."""

    TRAJECTORY_STATE = "trajectory_state"
    OPTIMIZATION_RESULT = "optimization_result"
    COST_BREAKDOWN = "cost_breakdown"
    MISSION_CONFIG = "mission_config"
    PERFORMANCE_METRICS = "performance_metrics"


@dataclass
class TransformationRule:
    """Definition of a data transformation rule."""

    source_format: DataFormat
    target_format: DataFormat
    field_mappings: Dict[str, str]
    validation_rules: Dict[str, Any]
    transformation_function: Optional[str] = None
    required_fields: List[str] = None
    optional_fields: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.required_fields is None:
            self.required_fields = []
        if self.optional_fields is None:
            self.optional_fields = []


class DataTransformLayer:
    """Data transformation and validation layer for extensibility.

    This class provides standardized data transformation between different
    modules and extensions, ensuring compatibility and consistent interfaces.
    """

    def __init__(self):
        """Initialize the data transformation layer."""
        self._transformation_rules: Dict[str, TransformationRule] = {}
        self._format_schemas: Dict[DataFormat, Dict[str, Any]] = {}
        self._unit_conversions: Dict[str, Dict[str, float]] = {}

        # Initialize default transformations
        self._register_default_transformations()
        self._register_default_schemas()
        self._register_unit_conversions()

        logger.info("Data transformation layer initialized")

    def register_transformation_rule(
        self, rule_name: str, rule: TransformationRule
    ) -> bool:
        """Register a new transformation rule.

        Args:
            rule_name: Unique name for the transformation rule
            rule: Transformation rule definition

        Returns:
            True if registration was successful
        """
        try:
            # Validate rule
            if not self._validate_transformation_rule(rule):
                return False

            self._transformation_rules[rule_name] = rule
            logger.info(f"Registered transformation rule: {rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register transformation rule {rule_name}: {e}")
            return False

    def transform_data(
        self, data: Dict[str, Any], source_format: DataFormat, target_format: DataFormat
    ) -> Optional[Dict[str, Any]]:
        """Transform data from one format to another.

        Args:
            data: Input data dictionary
            source_format: Source data format
            target_format: Target data format

        Returns:
            Transformed data dictionary or None if transformation failed
        """
        try:
            # Find appropriate transformation rule
            rule = self._find_transformation_rule(source_format, target_format)
            if not rule:
                logger.error(
                    f"No transformation rule found: {source_format.value} -> {target_format.value}"
                )
                return None

            # Validate input data
            if not self._validate_data_format(data, source_format):
                logger.error(
                    f"Input data validation failed for format: {source_format.value}"
                )
                return None

            # Apply transformation
            transformed_data = self._apply_transformation(data, rule)

            # Validate output data
            if not self._validate_data_format(transformed_data, target_format):
                logger.error(
                    f"Output data validation failed for format: {target_format.value}"
                )
                return None

            logger.debug(
                f"Successfully transformed data: {source_format.value} -> {target_format.value}"
            )
            return transformed_data

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            return None

    def normalize_trajectory_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize trajectory state to standard format.

        Args:
            state: Trajectory state dictionary

        Returns:
            Normalized trajectory state
        """
        normalized = {
            "position": self._normalize_vector(state.get("position", [0, 0, 0])),
            "velocity": self._normalize_vector(state.get("velocity", [0, 0, 0])),
            "time": float(state.get("time", 0.0)),
            "mass": float(state.get("mass", 1000.0)),
        }

        # Add optional fields if present
        optional_fields = [
            "acceleration",
            "quaternion",
            "angular_velocity",
            "reference_frame",
        ]
        for field in optional_fields:
            if field in state:
                if field in ["acceleration", "quaternion", "angular_velocity"]:
                    normalized[field] = self._normalize_vector(state[field])
                else:
                    normalized[field] = state[field]

        return normalized

    def normalize_optimization_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize optimization result to standard format.

        Args:
            result: Optimization result dictionary

        Returns:
            Normalized optimization result
        """
        normalized = {
            "objectives": self._normalize_objectives(result.get("objectives", [])),
            "parameters": dict(result.get("parameters", {})),
            "constraints": dict(result.get("constraints", {})),
            "convergence_info": dict(result.get("convergence_info", {})),
            "metadata": dict(result.get("metadata", {})),
        }

        # Ensure objectives are in standard format
        if isinstance(normalized["objectives"], dict):
            # Convert dict to list if needed
            obj_dict = normalized["objectives"]
            normalized["objectives"] = [
                obj_dict.get("delta_v", 0),
                obj_dict.get("time", 0),
                obj_dict.get("cost", 0),
            ]

        return normalized

    def normalize_cost_breakdown(self, cost_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize cost breakdown to standard format.

        Args:
            cost_data: Cost breakdown dictionary

        Returns:
            Normalized cost breakdown
        """
        normalized = {
            "development": float(cost_data.get("development", 0.0)),
            "launch": float(cost_data.get("launch", 0.0)),
            "spacecraft": float(cost_data.get("spacecraft", 0.0)),
            "operations": float(cost_data.get("operations", 0.0)),
            "ground_systems": float(cost_data.get("ground_systems", 0.0)),
            "contingency": float(cost_data.get("contingency", 0.0)),
        }

        # Calculate total
        normalized["total"] = sum(normalized.values())

        # Add optional fields
        optional_fields = ["currency", "year", "confidence_level", "risk_factors"]
        for field in optional_fields:
            if field in cost_data:
                normalized[field] = cost_data[field]

        return normalized

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between different units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        if from_unit == to_unit:
            return value

        # Find conversion factor
        conversion_factor = self._get_unit_conversion_factor(from_unit, to_unit)
        if conversion_factor is None:
            logger.warning(f"No conversion available: {from_unit} -> {to_unit}")
            return value

        return value * conversion_factor

    def validate_extension_data(
        self, data: Dict[str, Any], extension_type: str
    ) -> bool:
        """Validate data for a specific extension type.

        Args:
            data: Data to validate
            extension_type: Type of extension

        Returns:
            True if data is valid
        """
        try:
            # Get validation schema for extension type
            schema = self._get_extension_schema(extension_type)
            if not schema:
                logger.warning(
                    f"No validation schema for extension type: {extension_type}"
                )
                return True  # Allow if no schema defined

            # Validate required fields
            required_fields = schema.get("required_fields", [])
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Validate field types
            field_types = schema.get("field_types", {})
            for field, expected_type in field_types.items():
                if field in data:
                    if not self._validate_field_type(data[field], expected_type):
                        logger.error(
                            f"Invalid type for field {field}: expected {expected_type}"
                        )
                        return False

            return True

        except Exception as e:
            logger.error(f"Extension data validation failed: {e}")
            return False

    def _register_default_transformations(self) -> None:
        """Register default transformation rules."""
        # Trajectory state transformations
        self.register_transformation_rule(
            "trajectory_to_optimization",
            TransformationRule(
                source_format=DataFormat.TRAJECTORY_STATE,
                target_format=DataFormat.OPTIMIZATION_RESULT,
                field_mappings={
                    "position": "initial_state.position",
                    "velocity": "initial_state.velocity",
                    "time": "parameters.time",
                    "mass": "parameters.mass",
                },
                validation_rules={
                    "position": {"type": "list", "length": 3},
                    "velocity": {"type": "list", "length": 3},
                },
                required_fields=["position", "velocity", "time"],
            ),
        )

        # Optimization result transformations
        self.register_transformation_rule(
            "optimization_to_cost",
            TransformationRule(
                source_format=DataFormat.OPTIMIZATION_RESULT,
                target_format=DataFormat.COST_BREAKDOWN,
                field_mappings={
                    "objectives": "total_cost",
                    "parameters.delta_v": "launch",
                    "parameters.mass": "spacecraft",
                },
                validation_rules={
                    "objectives": {"type": "list", "min_length": 1},
                },
                required_fields=["objectives"],
            ),
        )

    def _register_default_schemas(self) -> None:
        """Register default data format schemas."""
        self._format_schemas[DataFormat.TRAJECTORY_STATE] = {
            "required_fields": ["position", "velocity", "time"],
            "optional_fields": ["mass", "acceleration", "reference_frame"],
            "field_types": {
                "position": "list",
                "velocity": "list",
                "time": "float",
                "mass": "float",
            },
        }

        self._format_schemas[DataFormat.OPTIMIZATION_RESULT] = {
            "required_fields": ["objectives", "parameters"],
            "optional_fields": ["constraints", "convergence_info", "metadata"],
            "field_types": {
                "objectives": "list",
                "parameters": "dict",
                "constraints": "dict",
            },
        }

        self._format_schemas[DataFormat.COST_BREAKDOWN] = {
            "required_fields": ["total"],
            "optional_fields": ["development", "launch", "spacecraft", "operations"],
            "field_types": {
                "total": "float",
                "development": "float",
                "launch": "float",
                "spacecraft": "float",
                "operations": "float",
            },
        }

    def _register_unit_conversions(self) -> None:
        """Register unit conversion factors."""
        # Length conversions (base: meters)
        self._unit_conversions["length"] = {
            "m": 1.0,
            "km": 1000.0,
            "au": 149597870700.0,
            "ft": 0.3048,
            "mi": 1609.34,
        }

        # Velocity conversions (base: m/s)
        self._unit_conversions["velocity"] = {
            "m/s": 1.0,
            "km/s": 1000.0,
            "km/h": 1.0 / 3.6,
            "ft/s": 0.3048,
            "mph": 0.44704,
        }

        # Mass conversions (base: kg)
        self._unit_conversions["mass"] = {
            "kg": 1.0,
            "g": 0.001,
            "t": 1000.0,
            "lb": 0.453592,
            "oz": 0.0283495,
        }

        # Time conversions (base: seconds)
        self._unit_conversions["time"] = {
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
            "day": 86400.0,
            "year": 31557600.0,
        }

    def _validate_transformation_rule(self, rule: TransformationRule) -> bool:
        """Validate a transformation rule."""
        if not rule.field_mappings:
            logger.error("Transformation rule must have field mappings")
            return False

        if not rule.validation_rules:
            rule.validation_rules = {}

        return True

    def _find_transformation_rule(
        self, source: DataFormat, target: DataFormat
    ) -> Optional[TransformationRule]:
        """Find transformation rule for format conversion."""
        for rule in self._transformation_rules.values():
            if rule.source_format == source and rule.target_format == target:
                return rule
        return None

    def _validate_data_format(
        self, data: Dict[str, Any], data_format: DataFormat
    ) -> bool:
        """Validate data against format schema."""
        schema = self._format_schemas.get(data_format)
        if not schema:
            return True  # No schema means no validation

        # Check required fields
        for field in schema.get("required_fields", []):
            if field not in data:
                return False

        # Check field types
        field_types = schema.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in data and not self._validate_field_type(
                data[field], expected_type
            ):
                return False

        return True

    def _apply_transformation(
        self, data: Dict[str, Any], rule: TransformationRule
    ) -> Dict[str, Any]:
        """Apply transformation rule to data."""
        transformed = {}

        for source_field, target_field in rule.field_mappings.items():
            if source_field in data:
                # Handle nested field paths
                if "." in target_field:
                    self._set_nested_field(
                        transformed, target_field, data[source_field]
                    )
                else:
                    transformed[target_field] = data[source_field]

        return transformed

    def _normalize_vector(self, vector: Union[List, np.ndarray]) -> List[float]:
        """Normalize vector to standard list format."""
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return [float(x) for x in vector]

    def _normalize_objectives(self, objectives: Union[List, Dict]) -> List[float]:
        """Normalize objectives to standard list format."""
        if isinstance(objectives, dict):
            # Convert dict to standard order: [delta_v, time, cost]
            return [
                float(objectives.get("delta_v", 0)),
                float(objectives.get("time", 0)),
                float(objectives.get("cost", 0)),
            ]
        return [float(x) for x in objectives]

    def _get_unit_conversion_factor(
        self, from_unit: str, to_unit: str
    ) -> Optional[float]:
        """Get conversion factor between units."""
        for _unit_type, conversions in self._unit_conversions.items():
            if from_unit in conversions and to_unit in conversions:
                return conversions[from_unit] / conversions[to_unit]
        return None

    def _get_extension_schema(self, extension_type: str) -> Optional[Dict[str, Any]]:
        """Get validation schema for extension type."""
        # This would be extended with actual extension schemas
        schemas = {
            "flight_stage": {
                "required_fields": ["initial_state", "target_state"],
                "field_types": {
                    "initial_state": "dict",
                    "target_state": "dict",
                    "constraints": "dict",
                },
            },
            "trajectory_analyzer": {
                "required_fields": ["trajectory_data"],
                "field_types": {
                    "trajectory_data": "dict",
                    "analysis_type": "str",
                },
            },
        }

        return schemas.get(extension_type)

    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type."""
        type_validators = {
            "str": lambda x: isinstance(x, str),
            "int": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "bool": lambda x: isinstance(x, bool),
            "list": lambda x: isinstance(x, (list, tuple)),
            "dict": lambda x: isinstance(x, dict),
        }

        validator = type_validators.get(expected_type)
        return validator(value) if validator else True

    def _set_nested_field(
        self, data: Dict[str, Any], field_path: str, value: Any
    ) -> None:
        """Set value in nested dictionary using dot notation."""
        keys = field_path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
