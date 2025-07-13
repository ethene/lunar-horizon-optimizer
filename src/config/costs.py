"""Cost-related configuration models.

This module defines models for managing mission cost factors and economic parameters,
including learning curve adjustments and environmental costs.
"""

from pydantic import BaseModel, Field


class CostFactors(BaseModel):
    """Economic cost factors for mission planning with learning curves and environmental costs."""

    launch_cost_per_kg: float = Field(
        ...,  # Required field
        gt=0,  # Must be greater than 0
        description="Cost per kilogram for launch to LEO (USD/kg)",
    )

    operations_cost_per_day: float = Field(
        ...,
        gt=0,
        description="Daily mission operations cost (USD/day)",
    )

    development_cost: float = Field(
        ...,
        gt=0,
        description="Total mission development and preparation cost (USD)",
    )

    contingency_percentage: float | None = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Contingency percentage for cost calculations (%)",
    )

    # Learning curve parameters
    learning_rate: float = Field(
        default=0.90,
        gt=0.0,
        le=1.0,
        description="Wright's law learning rate (0.90 = 10% cost reduction per doubling)",
    )

    base_production_year: int = Field(
        default=2024,
        ge=2000,
        le=2100,
        description="Reference year for base launch vehicle production",
    )

    cumulative_production_units: int = Field(
        default=10,
        ge=1,
        description="Cumulative launch vehicles produced at base year",
    )

    # Environmental cost parameters
    carbon_price_per_ton_co2: float = Field(
        default=50.0,
        ge=0.0,
        description="Carbon price per ton of CO₂ equivalent (USD/tCO₂)",
    )

    co2_emissions_per_kg_payload: float = Field(
        default=2.5,
        ge=0.0,
        description="CO₂ emissions per kg of payload delivered (tCO₂/kg)",
    )

    environmental_compliance_factor: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Environmental compliance cost multiplier (1.1 = 10% overhead)",
    )
