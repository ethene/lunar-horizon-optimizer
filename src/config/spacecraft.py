"""Spacecraft configuration models.

This module defines models for spacecraft specifications, including:
- Mass properties
- Propulsion characteristics
- Payload configurations
"""

from pydantic import BaseModel, Field, model_validator
import numpy as np

class PayloadSpecification(BaseModel):
    """Spacecraft payload specifications."""

    dry_mass: float = Field(
        ...,
        gt=0,
        description="Dry mass of the spacecraft without propellant (kg)"
    )

    payload_mass: float = Field(
        ...,
        gt=0,
        description="Mass of the mission payload (kg)"
    )

    max_propellant_mass: float = Field(
        ...,
        gt=0,
        description="Maximum propellant capacity (kg)"
    )

    specific_impulse: float = Field(
        ...,
        gt=0,
        description="Specific impulse of the propulsion system (seconds)"
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
                msg
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
