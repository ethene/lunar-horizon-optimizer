"""Example custom cost model extension.

This module demonstrates how to implement a custom cost analysis extension
that provides alternative cost modeling approaches for mission analysis.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_extension import BaseExtension, ExtensionMetadata, ExtensionType

logger = logging.getLogger(__name__)


class CustomCostModel(BaseExtension):
    """Example custom cost model extension.

    This extension demonstrates how to implement alternative cost modeling
    approaches, including parametric cost models, learning curve effects,
    and risk-adjusted costing methodologies.
    """

    # Class-level extension type for registry
    EXTENSION_TYPE = ExtensionType.COST_MODEL

    def __init__(
        self,
        metadata: Optional[ExtensionMetadata] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the custom cost model extension.

        Args:
            metadata: Extension metadata (auto-generated if None)
            config: Extension configuration
        """
        if metadata is None:
            metadata = ExtensionMetadata(
                name="custom_cost_model",
                version="1.0.0",
                description="Advanced parametric cost modeling with risk adjustment",
                author="Lunar Horizon Optimizer Team",
                extension_type=ExtensionType.COST_MODEL,
                required_dependencies=[],
                optional_dependencies=["scipy", "sklearn"],
                api_version="1.0",
                enabled=True,
                configuration_schema={
                    "type": "object",
                    "properties": {
                        "cost_model_type": {"type": "string", "default": "parametric"},
                        "learning_curve_enabled": {"type": "boolean", "default": True},
                        "risk_adjustment_factor": {"type": "number", "default": 1.2},
                        "inflation_rate": {"type": "number", "default": 0.03},
                        "currency": {"type": "string", "default": "USD"},
                    },
                },
            )

        super().__init__(metadata, config)

        # Configuration parameters
        self.cost_model_type = self.config.get("cost_model_type", "parametric")
        self.learning_curve_enabled = self.config.get("learning_curve_enabled", True)
        self.risk_adjustment_factor = self.config.get("risk_adjustment_factor", 1.2)
        self.inflation_rate = self.config.get("inflation_rate", 0.03)
        self.currency = self.config.get("currency", "USD")

        # Cost model parameters
        self._cost_estimating_relationships = {}
        self._learning_curve_parameters = {}
        self._risk_factors = {}

    def initialize(self) -> bool:
        """Initialize the custom cost model extension."""
        try:
            # Initialize cost estimating relationships
            self._initialize_cost_relationships()

            # Initialize learning curve parameters
            self._initialize_learning_curves()

            # Initialize risk factors
            self._initialize_risk_factors()

            self._initialized = True
            self.logger.info("Custom cost model extension initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize custom cost model: {e}")
            return False

    def validate_configuration(self) -> bool:
        """Validate the extension configuration."""
        try:
            # Validate cost model type
            valid_types = ["parametric", "analogical", "bottom_up", "hybrid"]
            if self.cost_model_type not in valid_types:
                self.logger.error(f"Invalid cost model type: {self.cost_model_type}")
                return False

            # Validate risk adjustment factor
            if self.risk_adjustment_factor < 1.0 or self.risk_adjustment_factor > 3.0:
                self.logger.error("Risk adjustment factor must be between 1.0 and 3.0")
                return False

            # Validate inflation rate
            if self.inflation_rate < 0 or self.inflation_rate > 0.2:
                self.logger.error("Inflation rate must be between 0 and 20%")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def get_capabilities(self) -> Dict[str, Any]:
        """Get custom cost model capabilities."""
        return {
            "type": "cost_model",
            "model_name": "custom_parametric_cost_model",
            "provides_cost_estimation": True,
            "provides_risk_analysis": True,
            "provides_learning_curve_modeling": self.learning_curve_enabled,
            "supported_cost_types": [
                "development",
                "manufacturing",
                "operations",
                "testing",
                "integration",
                "risk_adjusted",
            ],
            "cost_breakdown_categories": [
                "hardware",
                "software",
                "labor",
                "materials",
                "facilities",
                "overhead",
                "profit",
                "risk_reserve",
            ],
            "estimation_methods": [
                "parametric_scaling",
                "analogical_comparison",
                "bottom_up_aggregation",
                "monte_carlo_simulation",
            ],
            "risk_modeling": [
                "technical_risk",
                "schedule_risk",
                "cost_growth_risk",
                "market_risk",
            ],
        }

    def estimate_mission_cost(self, mission_params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate total mission cost using custom model.

        Args:
            mission_params: Mission parameters for cost estimation

        Returns:
            Dictionary containing cost estimates and analysis
        """
        try:
            # Extract mission parameters
            mission_type = mission_params.get("mission_type", "lunar_transport")
            payload_mass = mission_params.get("payload_mass", 1000.0)  # kg
            mission_duration = mission_params.get("mission_duration", 24.0)  # months
            complexity_factor = mission_params.get("complexity_factor", 1.0)
            technology_readiness = mission_params.get("technology_readiness", 6)

            # Calculate base costs using parametric model
            base_costs = self._calculate_parametric_costs(
                mission_type, payload_mass, mission_duration, complexity_factor
            )

            # Apply technology readiness adjustments
            trl_adjusted_costs = self._apply_trl_adjustments(
                base_costs, technology_readiness
            )

            # Apply learning curve effects if enabled
            if self.learning_curve_enabled:
                learning_adjusted_costs = self._apply_learning_curve(
                    trl_adjusted_costs, mission_params
                )
            else:
                learning_adjusted_costs = trl_adjusted_costs

            # Apply risk adjustments
            risk_adjusted_costs = self._apply_risk_adjustments(learning_adjusted_costs)

            # Generate cost breakdown
            cost_breakdown = self._generate_cost_breakdown(risk_adjusted_costs)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(cost_breakdown)

            # Generate cost drivers analysis
            cost_drivers = self._analyze_cost_drivers(mission_params, cost_breakdown)

            result = {
                "total_cost": cost_breakdown["total"],
                "cost_breakdown": cost_breakdown,
                "confidence_intervals": confidence_intervals,
                "cost_drivers": cost_drivers,
                "methodology": {
                    "model_type": self.cost_model_type,
                    "learning_curve_applied": self.learning_curve_enabled,
                    "risk_adjustment_factor": self.risk_adjustment_factor,
                    "currency": self.currency,
                },
                "assumptions": self._get_cost_assumptions(),
                "sensitivity_parameters": self._get_sensitivity_parameters(),
            }

            self.logger.info(f"Mission cost estimated: ${cost_breakdown['total']:.1f}M")
            return result

        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            return {
                "total_cost": 0.0,
                "error": str(e),
                "cost_breakdown": {},
                "methodology": {"model_type": self.cost_model_type},
            }

    def breakdown_costs(self, cost_estimate: Dict[str, Any]) -> Dict[str, float]:
        """Break down costs by detailed categories.

        Args:
            cost_estimate: Cost estimate from estimate_mission_cost

        Returns:
            Detailed cost breakdown dictionary
        """
        try:
            base_breakdown = cost_estimate.get("cost_breakdown", {})

            # Create detailed breakdown
            detailed_breakdown = {}

            # Hardware breakdown
            if "hardware" in base_breakdown:
                hardware_cost = base_breakdown["hardware"]
                detailed_breakdown.update(
                    {
                        "propulsion_hardware": hardware_cost * 0.35,
                        "avionics_hardware": hardware_cost * 0.25,
                        "structure_hardware": hardware_cost * 0.20,
                        "thermal_hardware": hardware_cost * 0.10,
                        "power_hardware": hardware_cost * 0.10,
                    }
                )

            # Software breakdown
            if "software" in base_breakdown:
                software_cost = base_breakdown["software"]
                detailed_breakdown.update(
                    {
                        "flight_software": software_cost * 0.40,
                        "ground_software": software_cost * 0.30,
                        "simulation_software": software_cost * 0.20,
                        "test_software": software_cost * 0.10,
                    }
                )

            # Labor breakdown
            if "labor" in base_breakdown:
                labor_cost = base_breakdown["labor"]
                detailed_breakdown.update(
                    {
                        "engineering_labor": labor_cost * 0.50,
                        "manufacturing_labor": labor_cost * 0.25,
                        "test_labor": labor_cost * 0.15,
                        "management_labor": labor_cost * 0.10,
                    }
                )

            # Operations breakdown
            if "operations" in base_breakdown:
                ops_cost = base_breakdown["operations"]
                detailed_breakdown.update(
                    {
                        "mission_operations": ops_cost * 0.40,
                        "ground_operations": ops_cost * 0.30,
                        "maintenance": ops_cost * 0.20,
                        "training": ops_cost * 0.10,
                    }
                )

            return detailed_breakdown

        except Exception as e:
            self.logger.error(f"Cost breakdown failed: {e}")
            return {}

    def analyze_cost_drivers(
        self, mission_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify and analyze primary cost drivers.

        Args:
            mission_params: Mission parameters

        Returns:
            List of cost driver analysis results
        """
        try:
            cost_drivers = []

            # Payload mass driver
            payload_mass = mission_params.get("payload_mass", 1000.0)
            if payload_mass > 2000:
                cost_drivers.append(
                    {
                        "driver": "payload_mass",
                        "impact": "high",
                        "value": payload_mass,
                        "cost_coefficient": 0.05,  # $50K per kg above 2000 kg
                        "description": "High payload mass drives launch vehicle and spacecraft costs",
                    }
                )

            # Mission duration driver
            duration = mission_params.get("mission_duration", 24.0)
            if duration > 36:
                cost_drivers.append(
                    {
                        "driver": "mission_duration",
                        "impact": "medium",
                        "value": duration,
                        "cost_coefficient": 0.02,  # $20K per month above 36 months
                        "description": "Extended mission duration increases operations costs",
                    }
                )

            # Technology readiness driver
            trl = mission_params.get("technology_readiness", 6)
            if trl < 7:
                cost_drivers.append(
                    {
                        "driver": "technology_readiness",
                        "impact": "high",
                        "value": trl,
                        "cost_coefficient": 0.15,  # 15% increase per TRL below 7
                        "description": "Low TRL increases development risk and cost",
                    }
                )

            # Complexity driver
            complexity = mission_params.get("complexity_factor", 1.0)
            if complexity > 1.5:
                cost_drivers.append(
                    {
                        "driver": "complexity_factor",
                        "impact": "high",
                        "value": complexity,
                        "cost_coefficient": 0.3,  # 30% increase per complexity unit above 1.5
                        "description": "High complexity increases all development costs",
                    }
                )

            # Sort by impact
            impact_order = {"high": 3, "medium": 2, "low": 1}
            cost_drivers.sort(
                key=lambda x: impact_order.get(x["impact"], 0), reverse=True
            )

            return cost_drivers

        except Exception as e:
            self.logger.error(f"Cost driver analysis failed: {e}")
            return []

    def _initialize_cost_relationships(self) -> None:
        """Initialize cost estimating relationships (CERs)."""
        # Parametric cost relationships
        self._cost_estimating_relationships = {
            "spacecraft_mass": {
                "coefficient": 0.05,  # $50K per kg
                "exponent": 1.0,
                "description": "Spacecraft cost vs mass relationship",
            },
            "payload_mass": {
                "coefficient": 0.08,  # $80K per kg payload
                "exponent": 1.0,
                "description": "Cost scaling with payload mass",
            },
            "mission_duration": {
                "coefficient": 2.0,  # $2M per month
                "exponent": 0.8,
                "description": "Operations cost vs mission duration",
            },
            "complexity_factor": {
                "coefficient": 50.0,  # $50M base cost
                "exponent": 1.5,
                "description": "Cost scaling with mission complexity",
            },
        }

    def _initialize_learning_curves(self) -> None:
        """Initialize learning curve parameters."""
        self._learning_curve_parameters = {
            "hardware_manufacturing": {
                "learning_rate": 0.85,  # 15% cost reduction per doubling
                "first_unit_cost": 1.0,
                "applicable_categories": ["hardware", "manufacturing"],
            },
            "software_development": {
                "learning_rate": 0.90,  # 10% cost reduction per doubling
                "first_unit_cost": 1.0,
                "applicable_categories": ["software", "labor"],
            },
            "operations": {
                "learning_rate": 0.95,  # 5% cost reduction per doubling
                "first_unit_cost": 1.0,
                "applicable_categories": ["operations"],
            },
        }

    def _initialize_risk_factors(self) -> None:
        """Initialize risk factors for different cost categories."""
        self._risk_factors = {
            "technical_risk": {
                "low": 1.1,
                "medium": 1.2,
                "high": 1.4,
            },
            "schedule_risk": {
                "low": 1.05,
                "medium": 1.15,
                "high": 1.3,
            },
            "cost_growth_risk": {
                "low": 1.1,
                "medium": 1.25,
                "high": 1.5,
            },
        }

    def _calculate_parametric_costs(
        self,
        mission_type: str,
        payload_mass: float,
        mission_duration: float,
        complexity_factor: float,
    ) -> Dict[str, float]:
        """Calculate base costs using parametric relationships."""
        # Base cost calculations
        spacecraft_base_cost = 100.0  # $100M base spacecraft cost
        payload_cost = (
            self._cost_estimating_relationships["payload_mass"]["coefficient"]
            * payload_mass
        )

        # Hardware costs
        hardware_cost = spacecraft_base_cost + payload_cost * 0.5

        # Software costs (typically 15-25% of hardware)
        software_cost = hardware_cost * 0.20

        # Labor costs (typically 30-40% of hardware + software)
        labor_cost = (hardware_cost + software_cost) * 0.35

        # Operations costs
        ops_cer = self._cost_estimating_relationships["mission_duration"]
        operations_cost = ops_cer["coefficient"] * (
            mission_duration ** ops_cer["exponent"]
        )

        # Apply complexity scaling
        complexity_multiplier = complexity_factor**1.2

        base_costs = {
            "hardware": hardware_cost * complexity_multiplier,
            "software": software_cost * complexity_multiplier,
            "labor": labor_cost * complexity_multiplier,
            "operations": operations_cost,
            "materials": hardware_cost * 0.15,  # 15% of hardware cost
            "facilities": hardware_cost * 0.10,  # 10% of hardware cost
        }

        return base_costs

    def _apply_trl_adjustments(
        self, base_costs: Dict[str, float], trl: int
    ) -> Dict[str, float]:
        """Apply technology readiness level adjustments."""
        # TRL risk multipliers
        trl_multipliers = {
            1: 3.0,
            2: 2.5,
            3: 2.0,
            4: 1.8,
            5: 1.5,
            6: 1.3,
            7: 1.1,
            8: 1.05,
            9: 1.0,
        }

        multiplier = trl_multipliers.get(trl, 1.0)

        # Apply primarily to development costs
        adjusted_costs = base_costs.copy()
        adjusted_costs["hardware"] *= multiplier
        adjusted_costs["software"] *= multiplier
        adjusted_costs["labor"] *= multiplier

        return adjusted_costs

    def _apply_learning_curve(
        self, costs: Dict[str, float], mission_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply learning curve effects."""
        # Assume this is the nth unit in a series
        unit_number = mission_params.get("unit_number", 1)

        if unit_number <= 1:
            return costs

        adjusted_costs = costs.copy()

        # Apply learning curves to applicable categories
        for _category, params in self._learning_curve_parameters.items():
            learning_rate = params["learning_rate"]
            applicable_cats = params["applicable_categories"]

            # Calculate learning curve factor
            learning_factor = unit_number ** np.log(learning_rate) / np.log(2)

            # Apply to applicable cost categories
            for cat in applicable_cats:
                if cat in adjusted_costs:
                    adjusted_costs[cat] *= learning_factor

        return adjusted_costs

    def _apply_risk_adjustments(self, costs: Dict[str, float]) -> Dict[str, float]:
        """Apply risk adjustments to costs."""
        adjusted_costs = costs.copy()

        # Apply overall risk adjustment factor
        for category in adjusted_costs:
            adjusted_costs[category] *= self.risk_adjustment_factor

        return adjusted_costs

    def _generate_cost_breakdown(self, costs: Dict[str, float]) -> Dict[str, float]:
        """Generate comprehensive cost breakdown."""
        # Calculate overhead and profit
        subtotal = sum(costs.values())
        overhead = subtotal * 0.15  # 15% overhead
        profit = subtotal * 0.08  # 8% profit

        breakdown = costs.copy()
        breakdown["overhead"] = overhead
        breakdown["profit"] = profit
        breakdown["total"] = subtotal + overhead + profit

        return breakdown

    def _calculate_confidence_intervals(
        self, cost_breakdown: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for cost estimates."""
        intervals = {}

        for category, cost in cost_breakdown.items():
            if category == "total":
                continue

            # Assume ±25% uncertainty for most categories
            uncertainty = 0.25

            # Adjust uncertainty based on category
            if category in ["operations", "software"]:
                uncertainty = 0.35  # Higher uncertainty
            elif category in ["materials", "facilities"]:
                uncertainty = 0.15  # Lower uncertainty

            intervals[category] = {
                "low": cost * (1 - uncertainty),
                "high": cost * (1 + uncertainty),
                "uncertainty": uncertainty,
            }

        return intervals

    def _analyze_cost_drivers(
        self, mission_params: Dict[str, Any], cost_breakdown: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Analyze primary cost drivers."""
        return self.analyze_cost_drivers(mission_params)

    def _get_cost_assumptions(self) -> List[str]:
        """Get list of cost model assumptions."""
        return [
            "Costs are in 2025 USD",
            f"Risk adjustment factor of {self.risk_adjustment_factor} applied",
            f"Inflation rate of {self.inflation_rate:.1%} assumed",
            "Learning curves applied for repeat units",
            "Technology readiness levels affect development costs",
            "Overhead includes G&A, facilities, and indirect costs",
            "Profit margins based on commercial space industry standards",
        ]

    def _get_sensitivity_parameters(self) -> List[str]:
        """Get list of parameters for sensitivity analysis."""
        return [
            "payload_mass",
            "mission_duration",
            "complexity_factor",
            "technology_readiness",
            "unit_number",
            "risk_adjustment_factor",
            "learning_curve_rates",
        ]
