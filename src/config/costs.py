"""Cost-related configuration models.

This module defines models for managing mission cost factors and economic parameters.
"""

from pydantic import BaseModel, Field

class CostFactors(BaseModel):
    """Economic cost factors for mission planning."""

    launch_cost_per_kg: float = Field(
        ...,  # Required field
        gt=0,  # Must be greater than 0
        description="Cost per kilogram for launch to LEO (USD/kg)"
    )

    operations_cost_per_day: float = Field(
        ...,
        gt=0,
        description="Daily mission operations cost (USD/day)"
    )

    development_cost: float = Field(
        ...,
        gt=0,
        description="Total mission development and preparation cost (USD)"
    )

    contingency_percentage: float | None = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Contingency percentage for cost calculations (%)"
    )
