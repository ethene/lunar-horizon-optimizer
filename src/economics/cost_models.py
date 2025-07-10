"""Cost models for lunar mission economic analysis - Task 5 implementation.

This module provides detailed cost modeling for different mission phases
including development, launch, and operational costs with parametric scaling.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config.costs import CostFactors

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for mission components."""

    development: float = 0.0
    launch: float = 0.0
    spacecraft: float = 0.0
    operations: float = 0.0
    ground_systems: float = 0.0
    contingency: float = 0.0
    total: float = 0.0

    def __post_init__(self):
        """Calculate total if not provided."""
        if self.total == 0.0:
            self.total = (
                self.development
                + self.launch
                + self.spacecraft
                + self.operations
                + self.ground_systems
                + self.contingency
            )


class MissionCostModel:
    """Comprehensive mission cost model for lunar missions.

    This class provides parametric cost estimation for complete lunar missions
    based on mission characteristics and historical cost data.
    """

    def __init__(self, cost_factors: CostFactors = None) -> None:
        """Initialize mission cost model.

        Args:
            cost_factors: Cost factors for calculations
        """
        self.cost_factors = cost_factors or CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
            contingency_percentage=20.0,
        )

        # Cost escalation factors (based on historical space mission data)
        self.escalation_factors: dict[str, dict[str | int, float]] = {
            "technology_readiness": {
                1: 3.0,  # Very early technology (TRL 1-3)
                2: 2.5,  # Early technology (TRL 4-5)
                3: 2.0,  # Mature technology (TRL 6-7)
                4: 1.5,  # Flight-proven technology (TRL 8-9)
            },
            "mission_complexity": {
                "simple": 1.0,  # Single spacecraft, simple operations
                "moderate": 1.5,  # Multiple systems, moderate complexity
                "complex": 2.5,  # Complex systems, high integration
                "flagship": 4.0,  # Flagship-class mission
            },
            "schedule_pressure": {
                "relaxed": 1.0,  # Normal development timeline
                "nominal": 1.2,  # Slight schedule pressure
                "aggressive": 1.8,  # Aggressive schedule
                "crash": 3.0,  # Crash program
            },
        }

        logger.info("Initialized MissionCostModel with parametric scaling")

    def estimate_total_mission_cost(
        self,
        spacecraft_mass: float,
        mission_duration_years: float,
        technology_readiness: int = 3,
        complexity: str = "moderate",
        schedule: str = "nominal",
    ) -> CostBreakdown:
        """Estimate total mission cost with detailed breakdown.

        Args:
            spacecraft_mass: Spacecraft mass [kg]
            mission_duration_years: Mission duration [years]
            technology_readiness: Technology readiness level (1-4 scale)
            complexity: Mission complexity ('simple', 'moderate', 'complex', 'flagship')
            schedule: Schedule pressure ('relaxed', 'nominal', 'aggressive', 'crash')

        Returns
        -------
            Detailed cost breakdown
        """
        logger.info(
            f"Estimating mission cost for {spacecraft_mass:.0f} kg spacecraft, "
            f"{mission_duration_years:.1f} year mission"
        )

        # Base cost estimates (in millions USD, 2024 dollars)
        base_costs = self._calculate_base_costs(spacecraft_mass, mission_duration_years)

        # Apply scaling factors
        tech_factors = self.escalation_factors["technology_readiness"]
        tech_factor = (
            tech_factors[technology_readiness]
            if technology_readiness in tech_factors
            else 2.0
        )

        complexity_factors = self.escalation_factors["mission_complexity"]
        complexity_factor = (
            complexity_factors[complexity] if complexity in complexity_factors else 1.5
        )

        schedule_factors = self.escalation_factors["schedule_pressure"]
        schedule_factor = (
            schedule_factors[schedule] if schedule in schedule_factors else 1.2
        )

        overall_factor = tech_factor * complexity_factor * schedule_factor

        # Scale costs
        scaled_costs = CostBreakdown(
            development=base_costs.development * overall_factor,
            launch=base_costs.launch,  # Launch costs don't scale with development factors
            spacecraft=base_costs.spacecraft * tech_factor * complexity_factor,
            operations=base_costs.operations * complexity_factor,
            ground_systems=base_costs.ground_systems * complexity_factor,
            contingency=0.0,  # Will be calculated below
        )

        # Add contingency
        base_total = (
            scaled_costs.development
            + scaled_costs.launch
            + scaled_costs.spacecraft
            + scaled_costs.operations
            + scaled_costs.ground_systems
        )

        scaled_costs.contingency = base_total * (
            self.cost_factors.contingency_percentage / 100
        )
        scaled_costs.total = base_total + scaled_costs.contingency

        logger.info(
            f"Total mission cost estimate: ${scaled_costs.total:.1f}M "
            f"(factor: {overall_factor:.1f}x)"
        )

        return scaled_costs

    def _calculate_base_costs(
        self, spacecraft_mass: float, mission_duration: float
    ) -> CostBreakdown:
        """Calculate base costs using parametric relationships."""
        # Development costs (based on spacecraft mass and complexity)
        # Typical range: $50-200M for lunar missions
        development_cost = 50 + 0.1 * spacecraft_mass  # $M

        # Launch costs (based on mass and launch vehicle)
        launch_cost = self._estimate_launch_cost(spacecraft_mass)

        # Spacecraft costs (based on mass and subsystem complexity)
        spacecraft_cost = 20 + 0.05 * spacecraft_mass  # $M

        # Operations costs (based on mission duration)
        operations_cost = 10 * mission_duration  # $M per year

        # Ground systems costs (typically 10-15% of development)
        ground_systems_cost = 0.12 * development_cost

        return CostBreakdown(
            development=development_cost,
            launch=launch_cost,
            spacecraft=spacecraft_cost,
            operations=operations_cost,
            ground_systems=ground_systems_cost,
        )

    def _estimate_launch_cost(self, spacecraft_mass: float) -> float:
        """Estimate launch cost based on spacecraft mass."""
        # Assumes Falcon Heavy class launcher
        # Cost per kg varies with total payload mass

        if spacecraft_mass <= 1000:  # Small payload
            cost_per_kg = 15000  # $15k/kg
        elif spacecraft_mass <= 5000:  # Medium payload
            cost_per_kg = 12000  # $12k/kg
        else:  # Large payload
            cost_per_kg = 10000  # $10k/kg

        return (spacecraft_mass * cost_per_kg) / 1e6  # Convert to $M

    def cost_sensitivity_analysis(
        self,
        base_params: dict[str, Any],
        sensitivity_ranges: dict[str, tuple[float, float]],
    ) -> dict[str, Any]:
        """Perform cost sensitivity analysis.

        Args:
            base_params: Base mission parameters
            sensitivity_ranges: Parameter ranges for sensitivity analysis

        Returns
        -------
            Sensitivity analysis results
        """
        logger.info("Performing cost sensitivity analysis")

        base_cost = self.estimate_total_mission_cost(**base_params)
        results: dict[str, Any] = {
            "base_cost": base_cost.total,
            "sensitivities": {},
        }

        for param, (min_val, max_val) in sensitivity_ranges.items():
            costs = []
            param_values = np.linspace(min_val, max_val, 10)

            for value in param_values:
                modified_params = base_params.copy()
                modified_params[param] = value

                cost_breakdown = self.estimate_total_mission_cost(**modified_params)
                costs.append(cost_breakdown.total)

            # Calculate sensitivity (% change in cost per % change in parameter)
            base_param_value = base_params[param]
            param_change = (max_val - min_val) / base_param_value
            cost_change = (max(costs) - min(costs)) / base_cost.total

            sensitivity = cost_change / param_change if param_change != 0 else 0

            if "sensitivities" not in results:
                results["sensitivities"] = {}
            results["sensitivities"][param] = {
                "parameter_values": param_values.tolist(),
                "costs": costs,
                "sensitivity_ratio": sensitivity,
                "cost_range": (min(costs), max(costs)),
            }

        return results


class LaunchCostModel:
    """Specialized launch cost model with vehicle-specific calculations.

    This class provides detailed launch cost analysis including different
    launch vehicle options and payload optimization.
    """

    def __init__(self) -> None:
        """Initialize launch cost model."""
        # Launch vehicle database (costs in $M, capacities in kg)
        self.launch_vehicles = {
            "Falcon 9": {
                "cost": 67,
                "leo_capacity": 22800,
                "gto_capacity": 8300,
                "tml_capacity": 4500,  # Trans-lunar injection
                "reusable_discount": 0.7,
            },
            "Falcon Heavy": {
                "cost": 150,
                "leo_capacity": 63800,
                "gto_capacity": 26700,
                "tml_capacity": 16800,
                "reusable_discount": 0.75,
            },
            "SLS Block 1": {
                "cost": 2000,  # Estimated cost per launch
                "leo_capacity": 95000,
                "gto_capacity": 37000,
                "tml_capacity": 26000,
                "reusable_discount": 1.0,  # Not reusable
            },
            "Starship": {
                "cost": 50,  # Projected cost
                "leo_capacity": 150000,
                "gto_capacity": 100000,
                "tml_capacity": 100000,  # With refueling
                "reusable_discount": 0.5,
            },
        }

        logger.info(
            f"Initialized LaunchCostModel with {len(self.launch_vehicles)} vehicle options"
        )

    def find_optimal_launch_vehicle(
        self, payload_mass: float, destination: str = "tml", use_reusable: bool = True
    ) -> dict[str, Any]:
        """Find optimal launch vehicle for given payload.

        Args:
            payload_mass: Payload mass [kg]
            destination: Destination ('leo', 'gto', 'tml')
            use_reusable: Use reusable vehicle option if available

        Returns
        -------
            Optimal launch vehicle analysis
        """
        logger.info(
            f"Finding optimal launch vehicle for {payload_mass:.0f} kg to {destination}"
        )

        capacity_key = f"{destination}_capacity"
        viable_vehicles = []

        for vehicle_name, specs in self.launch_vehicles.items():
            if payload_mass <= specs[capacity_key]:
                # Calculate effective cost
                base_cost = specs["cost"]
                if use_reusable:
                    effective_cost = base_cost * specs["reusable_discount"]
                else:
                    effective_cost = base_cost

                # Calculate cost efficiency
                cost_per_kg = effective_cost * 1e6 / specs[capacity_key]  # $/kg
                utilization = payload_mass / specs[capacity_key]

                viable_vehicles.append(
                    {
                        "name": vehicle_name,
                        "cost": effective_cost,
                        "capacity": specs[capacity_key],
                        "utilization": utilization,
                        "cost_per_kg": cost_per_kg,
                        "reusable": use_reusable and specs["reusable_discount"] < 1.0,
                    }
                )

        if not viable_vehicles:
            logger.warning("No suitable launch vehicles found for payload requirements")
            return {"error": "No suitable launch vehicles found"}

        # Sort by cost (considering utilization penalty for underutilization)
        for vehicle in viable_vehicles:
            # Penalty for low utilization (encourages efficient use)
            utilization_penalty = 1.0 if vehicle["utilization"] > 0.5 else 1.2
            vehicle["adjusted_cost"] = vehicle["cost"] * utilization_penalty

        optimal_vehicle = min(viable_vehicles, key=lambda v: v["adjusted_cost"])

        result = {
            "optimal_vehicle": optimal_vehicle,
            "all_options": viable_vehicles,
            "payload_mass": payload_mass,
            "destination": destination,
        }

        logger.info(
            f"Optimal vehicle: {optimal_vehicle['name']} at ${optimal_vehicle['cost']:.1f}M "
            f"({optimal_vehicle['utilization']:.1%} utilization)"
        )

        return result

    def calculate_multi_launch_strategy(
        self,
        total_payload: float,
        max_single_payload: float | None = None,
        destination: str = "tml",
    ) -> dict[str, Any]:
        """Calculate optimal multi-launch strategy for large payloads.

        Args:
            total_payload: Total payload mass [kg]
            max_single_payload: Maximum single launch payload [kg]
            destination: Destination orbit

        Returns
        -------
            Multi-launch strategy analysis
        """
        logger.info(
            f"Calculating multi-launch strategy for {total_payload:.0f} kg total payload"
        )

        strategies = []

        # Single launch options
        single_launch = self.find_optimal_launch_vehicle(total_payload, destination)
        if "optimal_vehicle" in single_launch:
            strategies.append(
                {
                    "type": "single_launch",
                    "num_launches": 1,
                    "vehicle": single_launch["optimal_vehicle"]["name"],
                    "total_cost": single_launch["optimal_vehicle"]["cost"],
                    "payload_per_launch": total_payload,
                    "total_utilization": single_launch["optimal_vehicle"][
                        "utilization"
                    ],
                }
            )

        # Multi-launch options
        for vehicle_name, specs in self.launch_vehicles.items():
            capacity = specs[f"{destination}_capacity"]

            if max_single_payload:
                effective_capacity = min(capacity, max_single_payload)
            else:
                effective_capacity = capacity

            num_launches = int(np.ceil(total_payload / effective_capacity))
            payload_per_launch = total_payload / num_launches

            if payload_per_launch <= capacity:
                total_cost = num_launches * specs["cost"] * specs["reusable_discount"]
                utilization = payload_per_launch / capacity

                strategies.append(
                    {
                        "type": "multi_launch",
                        "num_launches": num_launches,
                        "vehicle": vehicle_name,
                        "total_cost": total_cost,
                        "payload_per_launch": payload_per_launch,
                        "total_utilization": utilization,
                    }
                )

        # Sort strategies by total cost
        strategies.sort(key=lambda s: s["total_cost"])

        optimal_strategy = strategies[0] if strategies else None

        result = {
            "optimal_strategy": optimal_strategy,
            "all_strategies": strategies,
            "total_payload": total_payload,
            "destination": destination,
        }

        if optimal_strategy:
            logger.info(
                f"Optimal strategy: {optimal_strategy['num_launches']} x {optimal_strategy['vehicle']} "
                f"at ${optimal_strategy['total_cost']:.1f}M total"
            )

        return result


class OperationalCostModel:
    """Operational cost model for lunar mission operations.

    This class provides detailed operational cost estimation including
    mission operations, data processing, and mission extension costs.
    """

    def __init__(self) -> None:
        """Initialize operational cost model."""
        # Operational cost components ($ per month)
        self.monthly_costs = {
            "mission_operations": {
                "flight_operations": 500000,  # Flight ops team
                "mission_planning": 200000,  # Mission planning
                "spacecraft_health": 150000,  # Spacecraft monitoring
                "data_processing": 300000,  # Science data processing
                "ground_communications": 100000,  # Ground station costs
            },
            "science_operations": {
                "science_team": 200000,  # Science team support
                "data_analysis": 150000,  # Data analysis
                "instrument_operations": 100000,  # Instrument operations
            },
            "administrative": {
                "program_management": 150000,  # Program management
                "systems_engineering": 100000,  # Systems engineering
                "quality_assurance": 50000,  # QA/Safety
            },
        }

        logger.info("Initialized OperationalCostModel")

    def estimate_operational_costs(
        self,
        mission_duration_months: int,
        mission_phase: str = "nominal",
        science_level: str = "standard",
    ) -> dict[str, Any]:
        """Estimate operational costs for mission duration.

        Args:
            mission_duration_months: Mission duration in months
            mission_phase: Mission phase ('commissioning', 'nominal', 'extended')
            science_level: Science operations level ('minimal', 'standard', 'intensive')

        Returns
        -------
            Operational cost breakdown
        """
        logger.info(
            f"Estimating operational costs for {mission_duration_months} month mission"
        )

        # Phase-dependent cost multipliers
        phase_multipliers = {
            "commissioning": 1.5,  # Higher staffing during commissioning
            "nominal": 1.0,  # Baseline operations
            "extended": 0.7,  # Reduced staffing for mission extensions
        }

        # Science level multipliers
        science_multipliers = {
            "minimal": 0.5,  # Minimal science operations
            "standard": 1.0,  # Standard science operations
            "intensive": 1.5,  # Intensive science operations
        }

        phase_factor = phase_multipliers.get(mission_phase, 1.0)
        science_factor = science_multipliers.get(science_level, 1.0)

        # Calculate costs by category
        total_costs = {}
        monthly_total = 0

        for category, subcosts in self.monthly_costs.items():
            category_total = 0
            category_breakdown = {}

            for subcat, monthly_cost in subcosts.items():
                # Apply appropriate multipliers
                if category == "science_operations":
                    adjusted_cost = monthly_cost * science_factor * phase_factor
                else:
                    adjusted_cost = monthly_cost * phase_factor

                total_cost = adjusted_cost * mission_duration_months
                category_breakdown[subcat] = {
                    "monthly_cost": adjusted_cost,
                    "total_cost": total_cost,
                }
                category_total += total_cost

            total_costs[category] = {
                "breakdown": category_breakdown,
                "total": category_total,
            }
            monthly_total += category_total / mission_duration_months

        grand_total = sum(
            float(cat["total"])
            for cat in total_costs.values()
            if isinstance(cat, dict)
            and "total" in cat
            and isinstance(cat["total"], (int, float))
        )

        result = {
            "cost_breakdown": total_costs,
            "monthly_average": monthly_total,
            "total_cost": grand_total,
            "mission_duration_months": mission_duration_months,
            "mission_phase": mission_phase,
            "science_level": science_level,
            "cost_per_month_actual": grand_total / mission_duration_months,
        }

        logger.info(
            f"Total operational cost: ${grand_total/1e6:.1f}M over {mission_duration_months} months"
        )

        return result

    def cost_reduction_analysis(self, base_costs: dict[str, Any]) -> dict[str, Any]:
        """Analyze potential cost reduction strategies.

        Args:
            base_costs: Base operational cost structure

        Returns
        -------
            Cost reduction analysis
        """
        logger.info("Analyzing operational cost reduction strategies")

        reduction_strategies = {
            "automation": {
                "description": "Increased automation of routine operations",
                "affected_categories": ["mission_operations", "science_operations"],
                "reduction_factor": 0.8,  # 20% reduction
                "implementation_cost": 2000000,  # $2M implementation cost
            },
            "remote_operations": {
                "description": "Remote operations to reduce facility costs",
                "affected_categories": ["administrative"],
                "reduction_factor": 0.85,  # 15% reduction
                "implementation_cost": 500000,  # $0.5M implementation cost
            },
            "data_processing_optimization": {
                "description": "Optimized data processing workflows",
                "affected_categories": ["mission_operations"],
                "affected_subcategories": ["data_processing"],
                "reduction_factor": 0.7,  # 30% reduction
                "implementation_cost": 1000000,  # $1M implementation cost
            },
            "extended_mission_efficiency": {
                "description": "Efficiency improvements for extended mission",
                "applicable_phases": ["extended"],
                "reduction_factor": 0.6,  # 40% reduction vs nominal
                "implementation_cost": 0,  # No additional cost
            },
        }

        analysis_results = {
            "base_total_cost": base_costs["total_cost"],
            "strategies": {},
        }

        for strategy_name, strategy in reduction_strategies.items():
            # Calculate potential savings
            savings = 0
            affected_cost = 0

            for category_name, category_data in base_costs["cost_breakdown"].items():
                if category_name in strategy.get("affected_categories", []):
                    if "affected_subcategories" in strategy:
                        # Only specific subcategories
                        for subcat in strategy["affected_subcategories"]:
                            if subcat in category_data["breakdown"]:
                                subcost = category_data["breakdown"][subcat][
                                    "total_cost"
                                ]
                                affected_cost += subcost
                                savings += subcost * (1 - strategy["reduction_factor"])
                    else:
                        # Entire category
                        affected_cost += category_data["total"]
                        savings += category_data["total"] * (
                            1 - strategy["reduction_factor"]
                        )

            net_savings = savings - strategy["implementation_cost"]
            roi = (
                net_savings / strategy["implementation_cost"]
                if strategy["implementation_cost"] > 0
                else float("inf")
            )

            analysis_results["strategies"][strategy_name] = {
                "description": strategy["description"],
                "affected_cost": affected_cost,
                "gross_savings": savings,
                "implementation_cost": strategy["implementation_cost"],
                "net_savings": net_savings,
                "roi": roi,
                "payback_months": (
                    strategy["implementation_cost"]
                    / (savings / base_costs["mission_duration_months"])
                    if savings > 0
                    else float("inf")
                ),
            }

        # Rank strategies by ROI
        ranked_strategies = sorted(
            analysis_results["strategies"].items(),
            key=lambda x: x[1]["roi"],
            reverse=True,
        )

        analysis_results["recommended_order"] = [name for name, _ in ranked_strategies]

        return analysis_results


def create_cost_model_suite() -> dict[str, Any]:
    """Create a complete suite of cost models.

    Returns
    -------
        Dictionary containing all cost model instances
    """
    return {
        "mission_cost_model": MissionCostModel(),
        "launch_cost_model": LaunchCostModel(),
        "operational_cost_model": OperationalCostModel(),
    }
