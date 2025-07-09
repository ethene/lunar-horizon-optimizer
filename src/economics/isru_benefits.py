"""ISRU (In-Situ Resource Utilization) benefits analysis for Task 5.

This module provides economic analysis of lunar ISRU operations including
resource extraction, processing, and economic value calculations.
"""

from typing import Any
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ResourceProperty:
    """Properties of a lunar resource."""

    name: str
    abundance: float        # Resource abundance (ppm or percentage)
    extraction_difficulty: float  # Difficulty factor (1-5 scale)
    processing_complexity: float  # Processing complexity (1-5 scale)
    earth_value: float     # Value on Earth ($/kg)
    space_value: float     # Value in space ($/kg)
    transportation_cost: float  # Cost to transport from Moon ($/kg)


class ResourceValueModel:
    """Economic value model for lunar resources.

    This class provides valuation of lunar resources considering extraction,
    processing, and transportation costs versus Earth and space market values.
    """

    def __init__(self) -> None:
        """Initialize resource value model."""
        # Lunar resource database
        self.resources = {
            "water_ice": ResourceProperty(
                name="Water Ice",
                abundance=1000,  # ppm (estimated polar deposits)
                extraction_difficulty=2.0,
                processing_complexity=2.0,
                earth_value=1.0,  # $/kg (essentially free on Earth)
                space_value=20000,  # $/kg (extremely valuable in space)
                transportation_cost=5000  # $/kg from Moon to LEO
            ),
            "oxygen": ResourceProperty(
                name="Oxygen",
                abundance=500,   # ppm (from water or regolith)
                extraction_difficulty=3.0,
                processing_complexity=3.5,
                earth_value=0.5,  # $/kg
                space_value=15000,  # $/kg (life support, propellant)
                transportation_cost=5000  # $/kg
            ),
            "hydrogen": ResourceProperty(
                name="Hydrogen",
                abundance=100,   # ppm (from water)
                extraction_difficulty=3.0,
                processing_complexity=4.0,
                earth_value=2.0,  # $/kg
                space_value=25000,  # $/kg (propellant)
                transportation_cost=5000  # $/kg
            ),
            "helium3": ResourceProperty(
                name="Helium-3",
                abundance=0.01,  # ppm (very rare)
                extraction_difficulty=5.0,
                processing_complexity=5.0,
                earth_value=1000000,  # $/kg (fusion fuel)
                space_value=1000000,  # $/kg
                transportation_cost=10000  # $/kg (special handling)
            ),
            "rare_earth_elements": ResourceProperty(
                name="Rare Earth Elements",
                abundance=10,    # ppm (estimated)
                extraction_difficulty=4.5,
                processing_complexity=4.5,
                earth_value=50000,  # $/kg (average)
                space_value=100000,  # $/kg (manufacturing)
                transportation_cost=8000   # $/kg
            ),
            "titanium": ResourceProperty(
                name="Titanium",
                abundance=5000,  # ppm (from ilmenite)
                extraction_difficulty=4.0,
                processing_complexity=4.0,
                earth_value=30,   # $/kg
                space_value=500,  # $/kg (structural materials)
                transportation_cost=5000  # $/kg
            ),
            "aluminum": ResourceProperty(
                name="Aluminum",
                abundance=8000,  # ppm (abundant in regolith)
                extraction_difficulty=3.5,
                processing_complexity=3.5,
                earth_value=2,    # $/kg
                space_value=50,   # $/kg (structural materials)
                transportation_cost=5000  # $/kg
            )
        }

        logger.info(f"Initialized ResourceValueModel with {len(self.resources)} resource types")

    def calculate_resource_value(self,
                                resource_name: str,
                                extracted_mass: float,
                                market: str = "space") -> dict[str, Any]:
        """Calculate economic value of extracted resource.

        Args:
            resource_name: Name of the resource
            extracted_mass: Mass of extracted resource [kg]
            market: Target market ('earth', 'space', 'both')

        Returns
        -------
            Dictionary with value breakdown
        """
        if resource_name not in self.resources:
            msg = f"Unknown resource: {resource_name}"
            raise ValueError(msg)

        resource = self.resources[resource_name]

        # Calculate base values
        earth_value = extracted_mass * resource.earth_value
        space_value = extracted_mass * resource.space_value

        # Calculate transportation costs
        transport_cost = extracted_mass * resource.transportation_cost

        # Net values after transportation
        earth_net_value = earth_value - transport_cost
        space_net_value = space_value  # No transport needed for space market

        result = {
            "extracted_mass": extracted_mass,
            "resource_name": resource_name,
            "earth_gross_value": earth_value,
            "space_gross_value": space_value,
            "transportation_cost": transport_cost,
            "earth_net_value": earth_net_value,
            "space_net_value": space_net_value,
            "optimal_market": "space" if space_net_value > earth_net_value else "earth",
            "optimal_value": max(space_net_value, earth_net_value)
        }

        logger.info(f"Resource value calculated: {extracted_mass:.0f} kg {resource_name} "
                   f"= ${result['optimal_value']:,.0f} optimal value")

        return result

    def estimate_extraction_costs(self,
                                 resource_name: str,
                                 target_mass: float,
                                 facility_scale: str = "pilot") -> dict[str, Any]:
        """Estimate costs for resource extraction.

        Args:
            resource_name: Name of the resource
            target_mass: Target extraction mass [kg]
            facility_scale: Scale of operation ('pilot', 'commercial', 'industrial')

        Returns
        -------
            Extraction cost breakdown
        """
        if resource_name not in self.resources:
            msg = f"Unknown resource: {resource_name}"
            raise ValueError(msg)

        resource = self.resources[resource_name]

        # Scale factors for different facility sizes
        scale_factors = {
            "pilot": {"capital": 1.0, "operations": 1.0, "efficiency": 0.5},
            "commercial": {"capital": 0.7, "operations": 0.8, "efficiency": 1.0},
            "industrial": {"capital": 0.5, "operations": 0.6, "efficiency": 1.5}
        }

        scale = scale_factors[facility_scale]

        # Base extraction costs (parametric model)
        base_capital_cost = 1e8  # $100M base facility
        base_extraction_cost = 1000  # $/kg base extraction cost

        # Adjust for resource difficulty
        difficulty_factor = (resource.extraction_difficulty + resource.processing_complexity) / 2

        # Calculate costs
        capital_cost = base_capital_cost * difficulty_factor * scale["capital"]

        # Mass-dependent costs (includes regolith processing)
        regolith_needed = target_mass / (resource.abundance / 1e6)  # kg regolith
        processing_cost = regolith_needed * 10  # $10/kg regolith processing

        extraction_cost = target_mass * base_extraction_cost * difficulty_factor * scale["operations"] / scale["efficiency"]

        total_cost = capital_cost + processing_cost + extraction_cost
        cost_per_kg = total_cost / target_mass

        result = {
            "capital_cost": capital_cost,
            "processing_cost": processing_cost,
            "extraction_cost": extraction_cost,
            "total_cost": total_cost,
            "cost_per_kg": cost_per_kg,
            "regolith_mass_needed": regolith_needed,
            "facility_scale": facility_scale,
            "efficiency_factor": scale["efficiency"]
        }

        logger.info(f"Extraction cost estimate: ${total_cost/1e6:.1f}M total "
                   f"(${cost_per_kg:.0f}/kg) for {target_mass:.0f} kg {resource_name}")

        return result


class ISRUBenefitAnalyzer:
    """Comprehensive ISRU benefit analysis for lunar missions.

    This class provides complete economic analysis of ISRU operations
    including break-even analysis, ROI calculations, and scenario modeling.
    """

    def __init__(self) -> None:
        """Initialize ISRU benefit analyzer."""
        self.resource_model = ResourceValueModel()

        # ISRU facility parameters
        self.facility_params = {
            "water_extraction": {
                "peak_production_rate": 1000,  # kg/month
                "ramp_up_months": 12,          # months to reach peak
                "operational_lifetime": 60,    # months
                "reliability": 0.8,            # operational reliability
                "maintenance_cost_rate": 0.05  # % of capital cost per month
            },
            "oxygen_production": {
                "peak_production_rate": 500,   # kg/month
                "ramp_up_months": 18,
                "operational_lifetime": 60,
                "reliability": 0.75,
                "maintenance_cost_rate": 0.06
            },
            "metal_extraction": {
                "peak_production_rate": 100,   # kg/month
                "ramp_up_months": 24,
                "operational_lifetime": 120,
                "reliability": 0.7,
                "maintenance_cost_rate": 0.08
            }
        }

        logger.info("Initialized ISRUBenefitAnalyzer")

    def analyze_isru_economics(self,
                              resource_name: str,
                              facility_scale: str = "commercial",
                              operation_duration_months: int = 60,
                              discount_rate: float = 0.08) -> dict[str, Any]:
        """Perform comprehensive ISRU economic analysis.

        Args:
            resource_name: Primary resource to extract
            facility_scale: Scale of ISRU facility
            operation_duration_months: Duration of operations
            discount_rate: Discount rate for NPV calculation

        Returns
        -------
            Complete ISRU economic analysis
        """
        logger.info(f"Analyzing ISRU economics for {resource_name} over {operation_duration_months} months")

        # Get facility parameters
        facility_type = self._get_facility_type(resource_name)
        facility_params = self.facility_params.get(facility_type, self.facility_params["water_extraction"])

        # Calculate production profile
        production_profile = self._calculate_production_profile(
            facility_params, operation_duration_months
        )

        total_production = sum(production_profile)

        # Calculate costs
        extraction_costs = self.resource_model.estimate_extraction_costs(
            resource_name, total_production, facility_scale
        )

        # Calculate operational costs over time
        monthly_maintenance = extraction_costs["capital_cost"] * facility_params["maintenance_cost_rate"] / 12
        total_operational_cost = monthly_maintenance * operation_duration_months

        # Calculate revenues
        resource_value = self.resource_model.calculate_resource_value(
            resource_name, total_production, "space"
        )

        # NPV analysis
        npv_analysis = self._calculate_isru_npv(
            extraction_costs, total_operational_cost, resource_value,
            production_profile, discount_rate
        )

        # Break-even analysis
        break_even = self._calculate_break_even(
            extraction_costs, monthly_maintenance, resource_value, facility_params
        )

        # Risk analysis
        risk_analysis = self._perform_risk_analysis(
            npv_analysis["npv"], facility_params, resource_value
        )

        result = {
            "resource_analysis": {
                "resource_name": resource_name,
                "total_production_kg": total_production,
                "facility_scale": facility_scale,
                "operation_duration_months": operation_duration_months
            },
            "cost_analysis": {
                "capital_cost": extraction_costs["capital_cost"],
                "operational_cost": total_operational_cost,
                "total_cost": extraction_costs["capital_cost"] + total_operational_cost,
                "cost_per_kg": (extraction_costs["capital_cost"] + total_operational_cost) / total_production
            },
            "revenue_analysis": resource_value,
            "financial_metrics": npv_analysis,
            "break_even_analysis": break_even,
            "risk_analysis": risk_analysis,
            "production_profile": {
                "monthly_production": production_profile,
                "peak_production_rate": facility_params["peak_production_rate"],
                "ramp_up_months": facility_params["ramp_up_months"]
            }
        }

        logger.info(f"ISRU analysis complete: NPV = ${npv_analysis['npv']/1e6:.1f}M, "
                   f"ROI = {npv_analysis['roi']:.1%}")

        return result

    def compare_isru_vs_earth_supply(self,
                                   resource_name: str,
                                   annual_demand: float,
                                   years: int = 10) -> dict[str, Any]:
        """Compare ISRU production vs Earth supply for space operations.

        Args:
            resource_name: Resource to analyze
            annual_demand: Annual demand in space [kg/year]
            years: Analysis period [years]

        Returns
        -------
            Comparative analysis
        """
        logger.info(f"Comparing ISRU vs Earth supply for {annual_demand:.0f} kg/year {resource_name}")

        total_demand = annual_demand * years

        # Earth supply option
        resource = self.resource_model.resources[resource_name]
        earth_supply_cost = total_demand * resource.transportation_cost

        # ISRU option
        isru_analysis = self.analyze_isru_economics(
            resource_name, "commercial", years * 12
        )

        isru_cost = isru_analysis["cost_analysis"]["total_cost"]
        isru_production = isru_analysis["resource_analysis"]["total_production_kg"]
        isru_ramp_time = isru_analysis["production_profile"]["ramp_up_months"]

        # Calculate coverage
        isru_coverage = min(1.0, isru_production / total_demand)
        shortfall = max(0, total_demand - isru_production)
        shortfall_cost = shortfall * resource.transportation_cost

        total_isru_cost = isru_cost + shortfall_cost

        # Savings and metrics
        cost_savings = earth_supply_cost - total_isru_cost
        savings_percentage = cost_savings / earth_supply_cost if earth_supply_cost > 0 else 0

        # Risk factors
        risk_factors = {
            "technology_risk": 0.3,    # 30% chance of technical issues
            "schedule_risk": 0.2,      # 20% chance of delays
            "performance_risk": 0.25,  # 25% chance of reduced performance
            "cost_overrun_risk": 0.4   # 40% chance of cost overruns
        }

        comparison: dict[str, Any] = {
            "demand_analysis": {
                "annual_demand": annual_demand,
                "total_demand": total_demand,
                "analysis_years": years
            },
            "earth_supply": {
                "total_cost": earth_supply_cost,
                "cost_per_kg": resource.transportation_cost,
                "availability": "immediate",
                "risk": "low"
            },
            "isru_option": {
                "capital_cost": isru_analysis["cost_analysis"]["capital_cost"],
                "operational_cost": isru_analysis["cost_analysis"]["operational_cost"],
                "total_cost": total_isru_cost,
                "production_capacity": isru_production,
                "coverage_percentage": isru_coverage,
                "ramp_up_time_months": isru_ramp_time,
                "risk_factors": risk_factors
            },
            "comparison": {
                "cost_savings": cost_savings,
                "savings_percentage": savings_percentage,
                "break_even_demand": isru_cost / resource.transportation_cost,
                "payback_period_years": isru_cost / (annual_demand * resource.transportation_cost),
                "recommendation": "ISRU" if cost_savings > 0 else "Earth Supply"
            }
        }

        logger.info(f"Comparison complete: {comparison['comparison']['recommendation']} preferred "
                   f"with {savings_percentage:.1%} cost {'savings' if cost_savings > 0 else 'premium'}")

        return comparison

    def _get_facility_type(self, resource_name: str) -> str:
        """Map resource name to facility type."""
        mapping = {
            "water_ice": "water_extraction",
            "oxygen": "oxygen_production",
            "hydrogen": "water_extraction",
            "titanium": "metal_extraction",
            "aluminum": "metal_extraction",
            "rare_earth_elements": "metal_extraction"
        }
        return mapping.get(resource_name, "water_extraction")

    def _calculate_production_profile(self, facility_params: dict[str, Any], duration_months: int) -> list[float]:
        """Calculate monthly production profile with ramp-up."""
        production = []
        peak_rate = facility_params["peak_production_rate"]
        ramp_months = facility_params["ramp_up_months"]
        reliability = facility_params["reliability"]

        for month in range(duration_months):
            if month < ramp_months:
                # Ramp-up phase
                ramp_factor = (month + 1) / ramp_months
                monthly_production = peak_rate * ramp_factor * reliability
            else:
                # Full production phase
                monthly_production = peak_rate * reliability

            production.append(monthly_production)

        return production

    def _calculate_isru_npv(self, extraction_costs: dict[str, Any], operational_cost: float,
                           resource_value: dict[str, Any], production_profile: list[float],
                           discount_rate: float) -> dict[str, Any]:
        """Calculate NPV for ISRU operation."""
        # Initial capital investment (negative cash flow)
        cash_flows = [-extraction_costs["capital_cost"]]

        # Monthly operational cash flows
        monthly_operational_cost = operational_cost / len(production_profile)
        value_per_kg = resource_value["optimal_value"] / resource_value["extracted_mass"]

        for monthly_production in production_profile:
            monthly_revenue = monthly_production * value_per_kg
            monthly_cash_flow = monthly_revenue - monthly_operational_cost
            cash_flows.append(monthly_cash_flow)

        # Calculate NPV
        monthly_discount_rate = discount_rate / 12
        npv = 0
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / (1 + monthly_discount_rate) ** i

        # Calculate other metrics
        total_investment = extraction_costs["capital_cost"]
        total_revenue = sum(production_profile) * value_per_kg
        total_profit = total_revenue - total_investment - operational_cost
        roi = total_profit / total_investment if total_investment > 0 else 0

        return {
            "npv": npv,
            "total_investment": total_investment,
            "total_revenue": total_revenue,
            "total_profit": total_profit,
            "roi": roi,
            "cash_flows": cash_flows
        }

    def _calculate_break_even(self, extraction_costs: dict[str, Any], monthly_maintenance: float,
                             resource_value: dict[str, Any], facility_params: dict[str, Any]) -> dict[str, float]:
        """Calculate break-even metrics."""
        value_per_kg = resource_value["optimal_value"] / resource_value["extracted_mass"]
        monthly_production = facility_params["peak_production_rate"] * facility_params["reliability"]
        monthly_revenue = monthly_production * value_per_kg
        monthly_profit = monthly_revenue - monthly_maintenance

        if monthly_profit > 0:
            payback_months = extraction_costs["capital_cost"] / monthly_profit
            break_even_production = extraction_costs["capital_cost"] / value_per_kg
        else:
            payback_months = float("inf")
            break_even_production = float("inf")

        return {
            "payback_period_months": payback_months,
            "break_even_production_kg": break_even_production,
            "monthly_break_even_rate": break_even_production / payback_months if payback_months < float("inf") else 0
        }

    def _perform_risk_analysis(self, base_npv: float, facility_params: dict[str, Any],
                              resource_value: dict[str, Any]) -> dict[str, Any]:
        """Perform Monte Carlo risk analysis."""
        # Simplified risk analysis (in practice, would use Monte Carlo simulation)
        risks = {
            "technology_failure": {
                "probability": 0.2,
                "impact": -0.5 * base_npv  # 50% NPV loss
            },
            "reduced_performance": {
                "probability": 0.3,
                "impact": -0.3 * base_npv  # 30% NPV loss
            },
            "cost_overrun": {
                "probability": 0.4,
                "impact": -0.2 * base_npv  # 20% NPV loss
            },
            "market_changes": {
                "probability": 0.25,
                "impact": -0.4 * base_npv  # 40% NPV loss
            }
        }

        # Calculate expected value
        expected_loss = sum(risk["probability"] * risk["impact"] for risk in risks.values())
        risk_adjusted_npv = base_npv + expected_loss

        # Value at Risk (simplified)
        worst_case_npv = base_npv + sum(risk["impact"] for risk in risks.values())

        return {
            "base_npv": base_npv,
            "expected_loss": expected_loss,
            "risk_adjusted_npv": risk_adjusted_npv,
            "worst_case_npv": worst_case_npv,
            "risk_factors": risks,
            "confidence_level": 0.7  # 70% confidence in base case
        }


def analyze_lunar_resource_portfolio(resources: list[str],
                                   facility_scale: str = "commercial",
                                   operation_years: int = 5) -> dict[str, Any]:
    """Analyze a portfolio of lunar resources for ISRU development.

    Args:
        resources: List of resource names to analyze
        facility_scale: Scale of ISRU facilities
        operation_years: Duration of operations

    Returns
    -------
        Portfolio analysis results
    """
    analyzer = ISRUBenefitAnalyzer()
    portfolio_results = {}

    total_investment = 0
    total_npv = 0

    for resource in resources:
        analysis = analyzer.analyze_isru_economics(
            resource, facility_scale, operation_years * 12
        )
        portfolio_results[resource] = analysis

        total_investment += analysis["cost_analysis"]["capital_cost"]
        total_npv += analysis["financial_metrics"]["npv"]

    portfolio_roi = (total_npv / total_investment) if total_investment > 0 else 0

    return {
        "resources": portfolio_results,
        "portfolio_metrics": {
            "total_investment": total_investment,
            "total_npv": total_npv,
            "portfolio_roi": portfolio_roi,
            "num_resources": len(resources),
            "diversification_benefit": 0.1 * portfolio_roi  # Simplified diversification benefit
        }
    }

