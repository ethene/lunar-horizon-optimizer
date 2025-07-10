"""Spacecraft configuration models.

This module defines models for spacecraft specifications, including:
- Mass properties
- Propulsion characteristics
- Payload configurations
"""

import numpy as np
from pydantic import BaseModel, Field, model_validator


class PayloadSpecification(BaseModel):
    """Spacecraft payload specifications."""

    dry_mass: float = Field(
        ...,
        gt=0,
        description="Dry mass of the spacecraft without propellant (kg)",
    )

    payload_mass: float = Field(
        ...,
        gt=0,
        description="Mass of the mission payload (kg)",
    )

    max_propellant_mass: float = Field(
        ...,
        gt=0,
        description="Maximum propellant capacity (kg)",
    )

    specific_impulse: float = Field(
        ...,
        gt=0,
        description="Specific impulse of the propulsion system (seconds)",
    )

    @model_validator(mode="after")
    def validate_masses(self) -> "PayloadSpecification":
        """Validate mass relationships."""
        if self.payload_mass >= self.dry_mass:
            msg = (
                f"Payload mass ({self.payload_mass} kg) must be less than "
                f"dry mass ({self.dry_mass} kg)"
            )
            raise ValueError(
                msg,
            )
        return self

    def calculate_delta_v(self, propellant_mass: float) -> float:
        """Calculate available delta-v using the rocket equation.

        Args:
            propellant_mass: Mass of propellant to use (kg)

        Returns
        -------
            Available delta-v in m/s

        Raises
        ------
            ValueError: If propellant mass exceeds capacity
        """
        if propellant_mass > self.max_propellant_mass:
            msg = f"Propellant mass exceeds capacity: {self.max_propellant_mass} kg"
            raise ValueError(msg)

        g0 = 9.80665  # Standard gravity in m/s^2
        mass_ratio = (self.dry_mass + propellant_mass) / self.dry_mass
        return self.specific_impulse * g0 * np.log(mass_ratio)


class SpacecraftConfig(BaseModel):
    """Complete spacecraft configuration including payload, propulsion, and subsystems."""
    
    name: str = Field(
        default="Default Spacecraft",
        description="Name of the spacecraft configuration"
    )
    
    dry_mass: float = Field(
        ...,
        gt=0,
        description="Dry mass of the spacecraft without propellant (kg)"
    )
    
    propellant_mass: float = Field(
        ...,
        gt=0,
        description="Propellant mass capacity (kg)"
    )
    
    payload_mass: float = Field(
        ...,
        gt=0,
        description="Mass of the mission payload (kg)"
    )
    
    power_system_mass: float = Field(
        ...,
        gt=0,
        description="Mass of the power system (kg)"
    )
    
    propulsion_isp: float = Field(
        ...,
        gt=0,
        description="Specific impulse of the propulsion system (seconds)"
    )

    @model_validator(mode="after") 
    def validate_spacecraft_masses(self) -> "SpacecraftConfig":
        """Validate spacecraft mass relationships."""
        total_dry_mass = self.payload_mass + self.power_system_mass
        if total_dry_mass > self.dry_mass:
            msg = (
                f"Sum of payload mass ({self.payload_mass} kg) and power system mass "
                f"({self.power_system_mass} kg) exceeds dry mass ({self.dry_mass} kg)"
            )
            raise ValueError(msg)
        return self

    @property
    def total_mass(self) -> float:
        """Total spacecraft mass including propellant."""
        return self.dry_mass + self.propellant_mass

    @property 
    def mass_ratio(self) -> float:
        """Mass ratio for rocket equation calculations."""
        return self.total_mass / self.dry_mass

    def calculate_max_delta_v(self) -> float:
        """Calculate maximum delta-v with full propellant."""
        g0 = 9.80665  # Standard gravity in m/s^2
        return self.propulsion_isp * g0 * np.log(self.mass_ratio)
