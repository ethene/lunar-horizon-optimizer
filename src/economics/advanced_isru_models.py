"""Advanced ISRU models with time dependencies for Task 9.

This module extends the basic ISRU benefits analysis with time-dependent
production models, including ramp-up periods, maintenance downtime, and
technology learning curves.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.economics.isru_benefits import ISRUBenefitAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProductionProfile:
    """Time-dependent production profile for ISRU operations."""

    resource_type: str
    initial_capacity: float  # kg/day at startup
    peak_capacity: float  # kg/day at full production
    ramp_up_months: int  # Time to reach peak capacity
    maintenance_frequency_days: int  # Days between maintenance
    maintenance_duration_days: int  # Duration of maintenance
    efficiency_degradation_rate: float  # Annual efficiency loss (%)
    learning_curve_factor: float  # Production improvement rate

    def __post_init__(self):
        """Validate production profile parameters."""
        if self.peak_capacity < self.initial_capacity:
            raise ValueError("Peak capacity must be >= initial capacity")
        if self.efficiency_degradation_rate < 0 or self.efficiency_degradation_rate > 1:
            raise ValueError("Efficiency degradation rate must be between 0 and 1")


@dataclass
class ISRUFacility:
    """Represents an ISRU production facility with time-dependent characteristics."""

    name: str
    resources_produced: List[str]
    capital_cost: float  # Initial investment
    operational_cost_per_day: float
    lifetime_years: int
    production_profiles: Dict[str, ProductionProfile]
    startup_date: datetime

    def __post_init__(self):
        """Validate facility configuration."""
        if not self.resources_produced:
            raise ValueError("Facility must produce at least one resource")
        if self.lifetime_years <= 0:
            raise ValueError("Facility lifetime must be positive")


class TimeBasedISRUModel:
    """Advanced ISRU model with time-dependent production and economics.

    This class models ISRU operations over time, including production ramp-up,
    maintenance cycles, efficiency degradation, and learning curve effects.
    """

    def __init__(self, base_isru_analyzer: Optional[ISRUBenefitAnalyzer] = None):
        """Initialize time-based ISRU model.

        Args:
            base_isru_analyzer: Base ISRU analyzer for resource properties
        """
        self.base_analyzer = base_isru_analyzer or ISRUBenefitAnalyzer()
        self.facilities: List[ISRUFacility] = []
        self.production_history: Dict[str, List[Tuple[datetime, float]]] = {}

        logger.info("Initialized TimeBasedISRUModel")

    def add_facility(self, facility: ISRUFacility) -> None:
        """Add an ISRU facility to the model.

        Args:
            facility: ISRU facility to add
        """
        self.facilities.append(facility)
        logger.info(f"Added ISRU facility: {facility.name}")

    def calculate_production_rate(
        self, facility: ISRUFacility, resource: str, date: datetime
    ) -> float:
        """Calculate production rate for a resource at a specific date.

        Args:
            facility: ISRU facility
            resource: Resource type
            date: Date to calculate production rate

        Returns:
            Production rate in kg/day
        """
        if resource not in facility.production_profiles:
            return 0.0

        profile = facility.production_profiles[resource]

        # Calculate time since startup
        days_operational = (date - facility.startup_date).days
        if days_operational < 0:
            return 0.0  # Facility not yet operational

        # Check if facility has exceeded lifetime
        if days_operational > facility.lifetime_years * 365.25:
            return 0.0  # Facility at end of life

        # Calculate base production considering ramp-up
        months_operational = days_operational / 30.44  # Average month length
        if months_operational < profile.ramp_up_months:
            # Linear ramp-up
            ramp_factor = months_operational / profile.ramp_up_months
            base_production = (
                profile.initial_capacity
                + (profile.peak_capacity - profile.initial_capacity) * ramp_factor
            )
        else:
            base_production = profile.peak_capacity

        # Apply learning curve improvements
        learning_factor = 1.0 + profile.learning_curve_factor * np.log1p(
            months_operational / 12
        )

        # Apply efficiency degradation
        years_operational = days_operational / 365.25
        degradation_factor = (
            1 - profile.efficiency_degradation_rate
        ) ** years_operational

        # Check for maintenance periods
        if profile.maintenance_frequency_days > 0:
            # Calculate which maintenance cycle we're in
            days_in_cycle = days_operational % profile.maintenance_frequency_days

            # Maintenance happens at the end of each cycle
            maintenance_start = (
                profile.maintenance_frequency_days - profile.maintenance_duration_days
            )
            if days_in_cycle >= maintenance_start:
                return 0.0  # Facility in maintenance

        # Calculate final production rate
        production_rate = base_production * learning_factor * degradation_factor

        return production_rate

    def calculate_cumulative_production(
        self,
        resource: str,
        start_date: datetime,
        end_date: datetime,
        time_step_days: int = 1,
    ) -> Dict[str, float]:
        """Calculate cumulative production over a time period.

        Args:
            resource: Resource type
            start_date: Start of calculation period
            end_date: End of calculation period
            time_step_days: Time step for integration

        Returns:
            Dictionary with cumulative production and statistics
        """
        current_date = start_date
        cumulative_production = 0.0
        production_timeline = []

        while current_date <= end_date:
            daily_production = 0.0

            # Sum production from all facilities
            for facility in self.facilities:
                if resource in facility.resources_produced:
                    daily_production += self.calculate_production_rate(
                        facility, resource, current_date
                    )

            cumulative_production += daily_production * time_step_days
            production_timeline.append((current_date, daily_production))

            current_date += timedelta(days=time_step_days)

        # Store production history
        if resource not in self.production_history:
            self.production_history[resource] = []
        self.production_history[resource].extend(production_timeline)

        # Calculate statistics
        production_rates = [rate for _, rate in production_timeline]

        return {
            "cumulative_production": cumulative_production,
            "average_daily_production": np.mean(production_rates),
            "peak_production": np.max(production_rates),
            "production_days": len([r for r in production_rates if r > 0]),
            "downtime_days": len([r for r in production_rates if r == 0]),
            "timeline": production_timeline,
        }

    def calculate_time_dependent_economics(
        self,
        resource: str,
        start_date: datetime,
        end_date: datetime,
        discount_rate: float = 0.08,
    ) -> Dict[str, float]:
        """Calculate economics with time-dependent production.

        Args:
            resource: Resource type
            start_date: Start of analysis period
            end_date: End of analysis period
            discount_rate: Annual discount rate

        Returns:
            Economic analysis results
        """
        # Get resource properties
        resource_props = self.base_analyzer.resource_model.resources.get(resource)
        if not resource_props:
            raise ValueError(f"Unknown resource: {resource}")

        # Calculate production
        production_data = self.calculate_cumulative_production(
            resource, start_date, end_date
        )

        # Calculate revenues over time
        cash_flows = []
        for date, daily_production in production_data["timeline"]:
            if daily_production > 0:
                # Revenue from space market (assuming all production sold in space)
                daily_revenue = daily_production * resource_props.space_value

                # Discount to present value
                years_from_start = (date - start_date).days / 365.25
                discount_factor = (1 + discount_rate) ** (-years_from_start)

                cash_flows.append(
                    {
                        "date": date,
                        "revenue": daily_revenue,
                        "discounted_revenue": daily_revenue * discount_factor,
                    }
                )

        # Calculate costs
        total_capital_cost = sum(
            facility.capital_cost
            for facility in self.facilities
            if resource in facility.resources_produced
        )

        total_operational_cost = 0.0
        for facility in self.facilities:
            if resource in facility.resources_produced:
                days_operational = min(
                    (end_date - facility.startup_date).days,
                    facility.lifetime_years * 365.25,
                )
                if days_operational > 0:
                    total_operational_cost += (
                        facility.operational_cost_per_day * days_operational
                    )

        # Calculate financial metrics
        total_revenue = sum(cf["revenue"] for cf in cash_flows)
        total_discounted_revenue = sum(cf["discounted_revenue"] for cf in cash_flows)

        net_revenue = total_revenue - total_capital_cost - total_operational_cost
        discounted_net_revenue = (
            total_discounted_revenue - total_capital_cost - total_operational_cost
        )

        return {
            "cumulative_production": production_data["cumulative_production"],
            "total_revenue": total_revenue,
            "total_discounted_revenue": total_discounted_revenue,
            "total_capital_cost": total_capital_cost,
            "total_operational_cost": total_operational_cost,
            "net_revenue": net_revenue,
            "discounted_net_revenue": discounted_net_revenue,
            "roi": net_revenue / total_capital_cost if total_capital_cost > 0 else 0,
            "average_daily_production": production_data["average_daily_production"],
            "production_efficiency": production_data["production_days"]
            / (production_data["production_days"] + production_data["downtime_days"]),
        }

    def optimize_facility_deployment(
        self,
        resources: List[str],
        budget: float,
        start_date: datetime,
        analysis_period_years: int = 10,
        candidate_facilities: Optional[List[ISRUFacility]] = None,
    ) -> Dict[str, Any]:
        """Optimize ISRU facility deployment schedule.

        Args:
            resources: Target resources to produce
            budget: Total available budget
            start_date: Start date for deployment
            analysis_period_years: Analysis period in years
            candidate_facilities: List of candidate facilities to consider

        Returns:
            Optimization results including recommended deployment schedule
        """
        if candidate_facilities is None:
            candidate_facilities = self._generate_default_facilities(
                resources, start_date
            )

        # Simple greedy optimization (can be enhanced with more sophisticated methods)
        selected_facilities = []
        remaining_budget = budget
        deployment_schedule = []

        # Sort facilities by ROI potential
        facility_scores = []
        for facility in candidate_facilities:
            # Estimate ROI for each facility
            temp_model = TimeBasedISRUModel(self.base_analyzer)
            temp_model.add_facility(facility)

            total_score = 0.0
            for resource in facility.resources_produced:
                economics = temp_model.calculate_time_dependent_economics(
                    resource,
                    facility.startup_date,
                    facility.startup_date
                    + timedelta(days=analysis_period_years * 365.25),
                )
                total_score += economics["roi"]

            facility_scores.append((total_score, facility))

        # Sort by score (descending)
        facility_scores.sort(key=lambda x: x[0], reverse=True)

        # Select facilities within budget
        for score, facility in facility_scores:
            if facility.capital_cost <= remaining_budget:
                selected_facilities.append(facility)
                remaining_budget -= facility.capital_cost
                deployment_schedule.append(
                    {
                        "facility": facility.name,
                        "deployment_date": facility.startup_date,
                        "capital_cost": facility.capital_cost,
                        "expected_roi": score,
                    }
                )

        # Calculate combined economics
        optimized_model = TimeBasedISRUModel(self.base_analyzer)
        for facility in selected_facilities:
            optimized_model.add_facility(facility)

        combined_economics = {}
        for resource in resources:
            end_date = start_date + timedelta(days=analysis_period_years * 365.25)
            combined_economics[resource] = (
                optimized_model.calculate_time_dependent_economics(
                    resource, start_date, end_date
                )
            )

        return {
            "selected_facilities": [f.name for f in selected_facilities],
            "deployment_schedule": deployment_schedule,
            "total_capital_cost": budget - remaining_budget,
            "remaining_budget": remaining_budget,
            "combined_economics": combined_economics,
        }

    def _generate_default_facilities(
        self, resources: List[str], start_date: datetime
    ) -> List[ISRUFacility]:
        """Generate default facility configurations.

        Args:
            resources: Resources to produce
            start_date: Base start date

        Returns:
            List of candidate facilities
        """
        facilities = []

        # Water extraction facility
        if "water_ice" in resources or "oxygen" in resources or "hydrogen" in resources:
            water_facility = ISRUFacility(
                name="Polar Water Extraction Plant",
                resources_produced=["water_ice", "oxygen", "hydrogen"],
                capital_cost=50e6,  # $50M
                operational_cost_per_day=10000,  # $10k/day
                lifetime_years=15,
                production_profiles={
                    "water_ice": ProductionProfile(
                        resource_type="water_ice",
                        initial_capacity=100,  # 100 kg/day
                        peak_capacity=1000,  # 1000 kg/day
                        ramp_up_months=12,
                        maintenance_frequency_days=90,
                        maintenance_duration_days=3,
                        efficiency_degradation_rate=0.02,  # 2% per year
                        learning_curve_factor=0.1,
                    ),
                    "oxygen": ProductionProfile(
                        resource_type="oxygen",
                        initial_capacity=80,
                        peak_capacity=800,
                        ramp_up_months=12,
                        maintenance_frequency_days=90,
                        maintenance_duration_days=3,
                        efficiency_degradation_rate=0.02,
                        learning_curve_factor=0.1,
                    ),
                    "hydrogen": ProductionProfile(
                        resource_type="hydrogen",
                        initial_capacity=10,
                        peak_capacity=100,
                        ramp_up_months=12,
                        maintenance_frequency_days=90,
                        maintenance_duration_days=3,
                        efficiency_degradation_rate=0.02,
                        learning_curve_factor=0.1,
                    ),
                },
                startup_date=start_date,
            )
            facilities.append(water_facility)

        # Regolith processing facility
        if "titanium" in resources or "aluminum" in resources:
            regolith_facility = ISRUFacility(
                name="Regolith Processing Facility",
                resources_produced=["titanium", "aluminum", "oxygen"],
                capital_cost=100e6,  # $100M
                operational_cost_per_day=20000,  # $20k/day
                lifetime_years=20,
                production_profiles={
                    "titanium": ProductionProfile(
                        resource_type="titanium",
                        initial_capacity=50,
                        peak_capacity=500,
                        ramp_up_months=18,
                        maintenance_frequency_days=60,
                        maintenance_duration_days=5,
                        efficiency_degradation_rate=0.03,
                        learning_curve_factor=0.15,
                    ),
                    "aluminum": ProductionProfile(
                        resource_type="aluminum",
                        initial_capacity=100,
                        peak_capacity=1000,
                        ramp_up_months=18,
                        maintenance_frequency_days=60,
                        maintenance_duration_days=5,
                        efficiency_degradation_rate=0.03,
                        learning_curve_factor=0.15,
                    ),
                },
                startup_date=start_date + timedelta(days=365),  # Start 1 year later
            )
            facilities.append(regolith_facility)

        # Helium-3 extraction (if requested)
        if "helium3" in resources:
            he3_facility = ISRUFacility(
                name="Helium-3 Extraction Facility",
                resources_produced=["helium3"],
                capital_cost=500e6,  # $500M (very expensive)
                operational_cost_per_day=100000,  # $100k/day
                lifetime_years=25,
                production_profiles={
                    "helium3": ProductionProfile(
                        resource_type="helium3",
                        initial_capacity=0.01,  # 10g/day
                        peak_capacity=0.1,  # 100g/day
                        ramp_up_months=36,  # 3 years to peak
                        maintenance_frequency_days=30,
                        maintenance_duration_days=7,
                        efficiency_degradation_rate=0.05,
                        learning_curve_factor=0.2,
                    )
                },
                startup_date=start_date + timedelta(days=730),  # Start 2 years later
            )
            facilities.append(he3_facility)

        return facilities


def create_isru_production_forecast(
    resources: List[str],
    start_date: datetime,
    forecast_years: int = 20,
    facilities: Optional[List[ISRUFacility]] = None,
) -> Dict[str, Any]:
    """Create ISRU production forecast with visualization data.

    Args:
        resources: Resources to forecast
        start_date: Start date of forecast
        forecast_years: Number of years to forecast
        facilities: ISRU facilities (if None, uses defaults)

    Returns:
        Forecast data suitable for visualization
    """
    model = TimeBasedISRUModel()

    # Add facilities
    if facilities is None:
        default_facilities = model._generate_default_facilities(resources, start_date)
        for facility in default_facilities:
            model.add_facility(facility)
    else:
        for facility in facilities:
            model.add_facility(facility)

    # Generate forecast
    end_date = start_date + timedelta(days=forecast_years * 365.25)
    forecast_data = {}

    for resource in resources:
        production_data = model.calculate_cumulative_production(
            resource, start_date, end_date, time_step_days=30  # Monthly resolution
        )

        economics = model.calculate_time_dependent_economics(
            resource, start_date, end_date
        )

        # Extract timeline for visualization
        dates = [t[0] for t in production_data["timeline"]]
        production_rates = [t[1] for t in production_data["timeline"]]

        # Calculate cumulative production over time
        cumulative = []
        total = 0.0
        for rate in production_rates:
            total += rate * 30  # Monthly production
            cumulative.append(total)

        forecast_data[resource] = {
            "dates": dates,
            "daily_production": production_rates,
            "cumulative_production": cumulative,
            "economics": economics,
            "peak_production_date": (
                dates[np.argmax(production_rates)] if production_rates else None
            ),
            "breakeven_date": _calculate_breakeven_date(
                dates, cumulative, economics, resource
            ),
        }

    return forecast_data


def _calculate_breakeven_date(
    dates: List[datetime],
    cumulative_production: List[float],
    economics: Dict[str, float],
    resource: str,
) -> Optional[datetime]:
    """Calculate breakeven date for ISRU operations.

    Args:
        dates: Timeline dates
        cumulative_production: Cumulative production over time
        economics: Economic analysis results
        resource: Resource type

    Returns:
        Breakeven date or None if not achieved
    """
    # Simplified breakeven calculation
    # In reality, would need detailed cash flow analysis

    if economics["total_capital_cost"] == 0:
        return dates[0] if dates else None

    # Estimate when cumulative revenue exceeds total costs
    resource_value = 20000  # Simplified space value $/kg

    for i, (date, cum_prod) in enumerate(
        zip(dates, cumulative_production, strict=False)
    ):
        cum_revenue = cum_prod * resource_value
        total_costs = economics["total_capital_cost"] + (
            economics["total_operational_cost"] * (i + 1) / len(dates)
        )

        if cum_revenue >= total_costs:
            return date

    return None  # Breakeven not achieved in forecast period
