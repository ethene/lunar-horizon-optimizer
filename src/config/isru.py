"""Configuration models for In-Situ Resource Utilization (ISRU) capabilities.

This module defines the configuration models for ISRU systems, including:
- Resource extraction rates and efficiencies
- Power consumption and operational parameters
- Processing capabilities and maintenance requirements

All models use consistent units:
- Mass: kilograms (kg)
- Power: kilowatts (kW)
- Time: days
- Rates: kilograms per day (kg/day)
- Efficiencies: percentage (0-100)
"""

from pydantic import BaseModel, Field, validator

from .enums import IsruResourceType


class ResourceExtractionRate(BaseModel):
    """Defines the extraction rate and efficiency for a specific resource."""

    resource_type: IsruResourceType = Field(
        ...,
        description="Type of resource being extracted",
    )
    max_rate: float = Field(
        ...,
        gt=0,
        description="Maximum extraction rate in kg/day",
    )
    efficiency: float = Field(
        ...,
        ge=0,
        le=100,
        description="Extraction efficiency percentage (0-100)",
    )
    power_per_kg: float = Field(
        ...,
        gt=0,
        description="Power required per kg of resource extracted (kW/kg)",
    )


class IsruCapabilities(BaseModel):
    """Defines the capabilities and parameters of an ISRU system."""

    mass: float = Field(
        ...,
        gt=0,
        description="Total mass of the ISRU system in kg",
    )
    base_power: float = Field(
        ...,
        ge=0,
        description="Base power consumption in kW (when idle)",
    )
    extraction_rates: dict[IsruResourceType, ResourceExtractionRate] = Field(
        ...,
        description="Extraction rates and efficiencies for each resource type",
    )
    processing_efficiency: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall processing efficiency percentage (0-100)",
    )
    startup_time_days: float = Field(
        ...,
        ge=0,
        description="Time required to start up the system in days",
    )
    maintenance_downtime: float = Field(
        ...,
        ge=0,
        description="Expected maintenance downtime in days per month",
    )
    max_storage_capacity: dict[IsruResourceType, float] = Field(
        ...,
        description="Maximum storage capacity in kg for each resource type",
    )

    @validator("extraction_rates")
    @classmethod
    def validate_extraction_rates(
        cls, v: dict[IsruResourceType, ResourceExtractionRate]
    ) -> dict[IsruResourceType, ResourceExtractionRate]:
        """Validate that extraction rates are provided for supported resource types."""
        if not v:
            msg = "At least one resource extraction rate must be defined"
            raise ValueError(msg)
        return v

    def calculate_power_consumption(
        self, active_resources: dict[IsruResourceType, float]
    ) -> float:
        """Calculate total power consumption based on active resource extraction rates.

        Args:
            active_resources: Dict mapping resource types to their current extraction rates (kg/day)

        Returns
        -------
            Total power consumption in kW
        """
        power = self.base_power

        for resource_type, rate in active_resources.items():
            if resource_type not in self.extraction_rates:
                msg = f"No extraction rate defined for resource type: {resource_type}"
                raise ValueError(msg)

            if rate > self.extraction_rates[resource_type].max_rate:
                msg = f"Requested rate exceeds maximum for {resource_type}"
                raise ValueError(msg)

            power += rate * self.extraction_rates[resource_type].power_per_kg

        return power
