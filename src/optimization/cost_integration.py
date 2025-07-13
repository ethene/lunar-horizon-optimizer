"""Cost integration module for economic objectives in global optimization.

This module provides cost calculation capabilities for the multi-objective
optimization, integrating mission cost factors with trajectory parameters,
including Wright's law learning curves and environmental costs.
"""

import logging
import math

import numpy as np

from src.config.costs import CostFactors

# Configure logging
logger = logging.getLogger(__name__)


def launch_price(
    year: int,
    base_price: float,
    learning_rate: float = 0.90,
    base_year: int = 2024,
    cumulative_units_base: int = 10,
) -> float:
    """Calculate launch price using Wright's law learning curve.

    Wright's law states that for every doubling of cumulative production,
    costs decrease by a constant percentage (learning rate).

    Formula: Unit_Cost = First_Unit_Cost × (Cumulative_Units^-b)
    Where: b = ln(Learning_Rate) / ln(2)

    Args:
        year: Target year for price calculation
        base_price: Launch price at base year [USD/kg]
        learning_rate: Wright's law learning rate (0.90 = 10% reduction per doubling)
        base_year: Reference year for base production level
        cumulative_units_base: Cumulative production units at base year

    Returns:
        Adjusted launch price accounting for learning curve [USD/kg]
    """
    if year <= base_year:
        return base_price

    # Estimate production growth (conservative aerospace estimate: ~20% annual growth)
    years_elapsed = year - base_year
    annual_growth_rate = 0.20
    cumulative_units_current = cumulative_units_base * (
        (1 + annual_growth_rate) ** years_elapsed
    )

    # Calculate Wright's law learning exponent
    # b = ln(learning_rate) / ln(2)
    learning_exponent = math.log(learning_rate) / math.log(2)

    # Apply Wright's law
    # Cost_ratio = (Current_Units / Base_Units)^learning_exponent
    cost_ratio = (cumulative_units_current / cumulative_units_base) ** learning_exponent

    adjusted_price = base_price * cost_ratio

    logger.debug(
        f"Learning curve calculation - Year: {year}, "
        f"Cumulative units: {cumulative_units_current:.1f}, "
        f"Cost ratio: {cost_ratio:.3f}, "
        f"Adjusted price: ${adjusted_price:.0f}/kg"
    )

    return adjusted_price


def co2_cost(payload_mass_kg: float, co2_per_kg: float, price_per_ton: float) -> float:
    """Calculate CO₂ environmental cost.

    Args:
        payload_mass_kg: Payload mass delivered [kg]
        co2_per_kg: CO₂ emissions per kg payload [tCO₂/kg]
        price_per_ton: Carbon price [USD/tCO₂]

    Returns:
        Total CO₂ cost [USD]
    """
    total_co2_tons = payload_mass_kg * co2_per_kg
    co2_cost_usd = total_co2_tons * price_per_ton

    logger.debug(
        f"CO₂ cost calculation - Payload: {payload_mass_kg}kg, "
        f"Emissions: {total_co2_tons:.2f}tCO₂, "
        f"Cost: ${co2_cost_usd:.0f}"
    )

    return co2_cost_usd


class CostCalculator:
    """Economic cost calculator for lunar mission optimization.

    This class provides methods to calculate mission costs based on
    trajectory parameters and economic factors, supporting the cost
    objective in multi-objective optimization.
    """

    def __init__(
        self, cost_factors: CostFactors | None = None, mission_year: int = 2025
    ) -> None:
        """Initialize cost calculator with learning curves and environmental costs.

        Args:
            cost_factors: Economic cost parameters
            mission_year: Target mission year for learning curve calculations
        """
        self.cost_factors = cost_factors or CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=50000.0,
            development_cost=500000000.0,
        )
        self.mission_year = mission_year

        # Mission parameters for cost calculations
        self.spacecraft_mass = 5000.0  # kg (typical lunar mission)
        self.fuel_efficiency = 0.85  # Propellant efficiency factor
        self.operational_overhead = 1.2  # 20% overhead on operations

        logger.info(
            f"Initialized CostCalculator with launch cost: "
            f"${self.cost_factors.launch_cost_per_kg}/kg, "
            f"learning rate: {self.cost_factors.learning_rate:.2f}, "
            f"carbon price: ${self.cost_factors.carbon_price_per_ton_co2}/tCO₂"
        )

    def calculate_mission_cost(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
    ) -> float:
        """Calculate total mission cost based on trajectory parameters.

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Transfer time [days]
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]

        Returns
        -------
            Total mission cost [cost units]
        """
        # Calculate propellant requirements
        propellant_cost = self._calculate_propellant_cost(total_dv)

        # Calculate launch costs
        launch_cost = self._calculate_launch_cost(earth_orbit_alt)

        # Calculate operational costs
        operations_cost = self._calculate_operations_cost(transfer_time)

        # Calculate development costs (amortized)
        development_cost = self._calculate_development_cost()

        # Calculate altitude-dependent costs
        altitude_cost = self._calculate_altitude_cost(earth_orbit_alt, moon_orbit_alt)

        # Total cost with contingency
        base_cost = (
            propellant_cost
            + launch_cost
            + operations_cost
            + development_cost
            + altitude_cost
        )

        total_cost = base_cost * (1 + self.cost_factors.contingency_percentage / 100)

        logger.debug(
            f"Cost breakdown - Propellant: ${propellant_cost:.0f}, "
            f"Launch: ${launch_cost:.0f}, Operations: ${operations_cost:.0f}, "
            f"Development: ${development_cost:.0f}, Total: ${total_cost:.0f}"
        )

        return total_cost

    def calculate_trajectory_cost(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
    ) -> float:
        """Calculate trajectory cost (alias for calculate_mission_cost for backward compatibility).

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Total transfer time [days]
            earth_orbit_alt: Earth parking orbit altitude [km]
            moon_orbit_alt: Lunar orbit altitude [km]

        Returns
        -------
            Total trajectory cost [USD]
        """
        return self.calculate_mission_cost(
            total_dv, transfer_time, earth_orbit_alt, moon_orbit_alt
        )

    def _calculate_propellant_cost(self, total_dv: float) -> float:
        """Calculate propellant-related costs.

        Args:
            total_dv: Total delta-v requirement [m/s]

        Returns
        -------
            Propellant cost component
        """
        # Use rocket equation to estimate propellant mass
        # ΔV = Isp * g * ln(m_initial / m_final)
        # Assuming typical Isp = 450s for chemical propulsion

        isp = 450.0  # seconds
        g = 9.81  # m/s²

        # Calculate mass ratio
        mass_ratio = np.exp(total_dv / (isp * g))

        # Calculate propellant mass
        propellant_mass = self.spacecraft_mass * (mass_ratio - 1) / self.fuel_efficiency

        # Cost based on propellant mass ($/kg)
        propellant_cost_per_kg = 10.0  # Typical propellant cost

        return propellant_mass * propellant_cost_per_kg

    def _calculate_launch_cost(self, earth_orbit_alt: float) -> float:
        """Calculate launch costs with learning curve adjustments.

        Args:
            earth_orbit_alt: Earth orbit altitude [km]

        Returns
        -------
            Launch cost component with learning curve and environmental costs
        """
        # Apply learning curve to base launch cost
        adjusted_launch_cost_per_kg = launch_price(
            year=self.mission_year,
            base_price=self.cost_factors.launch_cost_per_kg,
            learning_rate=self.cost_factors.learning_rate,
            base_year=self.cost_factors.base_production_year,
            cumulative_units_base=self.cost_factors.cumulative_production_units,
        )

        # Higher orbits require more energy, thus higher cost
        altitude_factor = 1 + (earth_orbit_alt - 200) / 1000  # Scale factor

        launch_mass = self.spacecraft_mass + 2000  # Include service module
        base_launch_cost = launch_mass * adjusted_launch_cost_per_kg

        # Calculate environmental cost
        payload_mass = 1000.0  # Typical payload mass [kg]
        environmental_cost = co2_cost(
            payload_mass_kg=payload_mass,
            co2_per_kg=self.cost_factors.co2_emissions_per_kg_payload,
            price_per_ton=self.cost_factors.carbon_price_per_ton_co2,
        )

        # Apply environmental compliance factor
        total_environmental_cost = (
            environmental_cost * self.cost_factors.environmental_compliance_factor
        )

        total_cost = (base_launch_cost * altitude_factor) + total_environmental_cost

        logger.debug(
            f"Launch cost breakdown - Base: ${base_launch_cost * altitude_factor:.0f}, "
            f"Environmental: ${total_environmental_cost:.0f}, "
            f"Learning curve adjustment: {adjusted_launch_cost_per_kg/self.cost_factors.launch_cost_per_kg:.3f}"
        )

        return total_cost

    def _calculate_operations_cost(self, transfer_time: float) -> float:
        """Calculate operational costs based on mission duration.

        Args:
            transfer_time: Transfer time [days]

        Returns
        -------
            Operations cost component
        """
        base_ops_cost = transfer_time * self.cost_factors.operations_cost_per_day
        return base_ops_cost * self.operational_overhead

    def _calculate_development_cost(self) -> float:
        """Calculate amortized development costs.

        Returns
        -------
            Development cost component
        """
        # Amortized over multiple missions
        amortization_factor = 0.1  # 10% of development cost per mission
        return self.cost_factors.development_cost * amortization_factor

    def _calculate_altitude_cost(
        self, earth_orbit_alt: float, moon_orbit_alt: float
    ) -> float:
        """Calculate altitude-dependent cost factors.

        Args:
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]

        Returns
        -------
            Altitude-dependent cost component
        """
        # Lower lunar orbits require more precise navigation and operations
        lunar_precision_factor = max(1.0, (200 - moon_orbit_alt) / 100)

        # Higher Earth orbits may require additional maneuvers
        earth_complexity_factor = 1 + max(0, (earth_orbit_alt - 500) / 1000)

        base_altitude_cost = 50000.0  # Base altitude cost

        return base_altitude_cost * lunar_precision_factor * earth_complexity_factor

    def calculate_cost_breakdown(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
    ) -> dict[str, float]:
        """Calculate detailed cost breakdown with learning curves and environmental costs.

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Transfer time [days]
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]

        Returns
        -------
            Dictionary with detailed cost breakdown including environmental costs
        """
        propellant_cost = self._calculate_propellant_cost(total_dv)
        launch_cost = self._calculate_launch_cost(earth_orbit_alt)
        operations_cost = self._calculate_operations_cost(transfer_time)
        development_cost = self._calculate_development_cost()
        altitude_cost = self._calculate_altitude_cost(earth_orbit_alt, moon_orbit_alt)

        # Calculate learning curve impact
        base_launch_price = self.cost_factors.launch_cost_per_kg
        adjusted_launch_price = launch_price(
            year=self.mission_year,
            base_price=base_launch_price,
            learning_rate=self.cost_factors.learning_rate,
            base_year=self.cost_factors.base_production_year,
            cumulative_units_base=self.cost_factors.cumulative_production_units,
        )
        learning_curve_savings = (base_launch_price - adjusted_launch_price) * (
            self.spacecraft_mass + 2000
        )

        # Calculate environmental costs separately
        payload_mass = 1000.0  # Typical payload mass [kg]
        environmental_cost = (
            co2_cost(
                payload_mass_kg=payload_mass,
                co2_per_kg=self.cost_factors.co2_emissions_per_kg_payload,
                price_per_ton=self.cost_factors.carbon_price_per_ton_co2,
            )
            * self.cost_factors.environmental_compliance_factor
        )

        base_cost = (
            propellant_cost
            + launch_cost
            + operations_cost
            + development_cost
            + altitude_cost
        )

        contingency_cost = base_cost * (self.cost_factors.contingency_percentage / 100)
        total_cost = base_cost + contingency_cost

        return {
            "propellant_cost": propellant_cost,
            "launch_cost": launch_cost,
            "operations_cost": operations_cost,
            "development_cost": development_cost,
            "altitude_cost": altitude_cost,
            "environmental_cost": environmental_cost,
            "learning_curve_savings": learning_curve_savings,
            "contingency_cost": contingency_cost,
            "total_cost": total_cost,
            "propellant_fraction": propellant_cost / total_cost,
            "launch_fraction": launch_cost / total_cost,
            "operations_fraction": operations_cost / total_cost,
            "environmental_fraction": environmental_cost / total_cost,
            "learning_curve_adjustment": adjusted_launch_price / base_launch_price,
        }


class EconomicObjectives:
    """Economic objective functions for multi-objective optimization.

    This class provides various economic objective functions that can be
    used in the global optimization framework.
    """

    def __init__(self, cost_calculator: CostCalculator = None) -> None:
        """Initialize economic objectives.

        Args:
            cost_calculator: Cost calculator instance
        """
        self.cost_calculator = cost_calculator or CostCalculator()

    def minimize_total_cost(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
    ) -> float:
        """Objective function to minimize total mission cost.

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Transfer time [days]
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]

        Returns
        -------
            Total mission cost [cost units]
        """
        return self.cost_calculator.calculate_mission_cost(
            total_dv,
            transfer_time,
            earth_orbit_alt,
            moon_orbit_alt,
        )

    def minimize_cost_per_kg(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
        payload_mass: float = 1000.0,
    ) -> float:
        """Objective function to minimize cost per kg of payload.

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Transfer time [days]
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]
            payload_mass: Payload mass [kg]

        Returns
        -------
            Cost per kg of payload [cost units/kg]
        """
        total_cost = self.cost_calculator.calculate_mission_cost(
            total_dv,
            transfer_time,
            earth_orbit_alt,
            moon_orbit_alt,
        )
        return total_cost / payload_mass

    def maximize_cost_efficiency(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
    ) -> float:
        """Objective function to maximize cost efficiency (minimize negative efficiency).

        Cost efficiency is defined as payload capability per unit cost.

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Transfer time [days]
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]

        Returns
        -------
            Negative cost efficiency (for minimization)
        """
        total_cost = self.cost_calculator.calculate_mission_cost(
            total_dv,
            transfer_time,
            earth_orbit_alt,
            moon_orbit_alt,
        )

        # Simple efficiency metric: inverse of normalized cost and delta-v
        efficiency = 1.0 / (total_cost / 1e6 + total_dv / 1e4)

        # Return negative for minimization
        return -efficiency

    def calculate_roi_objective(
        self,
        total_dv: float,
        transfer_time: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
        mission_revenue: float = 5e6,
    ) -> float:
        """Calculate ROI-based objective (minimize negative ROI).

        Args:
            total_dv: Total delta-v requirement [m/s]
            transfer_time: Transfer time [days]
            earth_orbit_alt: Earth orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]
            mission_revenue: Expected mission revenue [cost units]

        Returns
        -------
            Negative ROI (for minimization)
        """
        total_cost = self.cost_calculator.calculate_mission_cost(
            total_dv,
            transfer_time,
            earth_orbit_alt,
            moon_orbit_alt,
        )

        roi = (mission_revenue - total_cost) / total_cost if total_cost > 0 else 0.0

        # Return negative ROI for minimization
        return -roi


def create_cost_calculator(
    launch_cost_per_kg: float = 10000.0,
    operations_cost_per_day: float = 100000.0,
    development_cost: float = 1e9,
    contingency_percentage: float = 20.0,
) -> CostCalculator:
    """Create cost calculator with specified parameters.

    Args:
        launch_cost_per_kg: Launch cost per kg [$/kg]
        operations_cost_per_day: Operations cost per day [$/day]
        development_cost: Development cost [$ total]
        contingency_percentage: Contingency percentage [%]

    Returns
    -------
        Configured cost calculator
    """
    cost_factors = CostFactors(
        launch_cost_per_kg=launch_cost_per_kg,
        operations_cost_per_day=operations_cost_per_day,
        development_cost=development_cost,
        contingency_percentage=contingency_percentage,
    )

    return CostCalculator(cost_factors)
