"""Orbital parameters and configuration models.

This module defines models for orbital mechanics calculations and mission planning.
All parameters use consistent units:
- Distances: kilometers (km)
- Angles: degrees
- Time: seconds (s)
"""

from pydantic import BaseModel, Field, model_validator
import numpy as np

class OrbitParameters(BaseModel):
    """Orbital parameters specification."""

    semi_major_axis: float = Field(
        ...,
        gt=0,
        description="Semi-major axis (km)"
    )

    eccentricity: float = Field(
        ...,
        ge=0,
        lt=1,
        description="Orbit eccentricity"
    )

    inclination: float = Field(
        ...,
        ge=0,
        le=180,
        description="Orbit inclination (degrees)"
    )

    raan: float = Field(
        default=0.0,
        ge=0,
        lt=360,
        description="Right ascension of ascending node (degrees)"
    )

    argument_of_periapsis: float = Field(
        default=0.0,
        ge=0,
        lt=360,
        description="Argument of periapsis (degrees)"
    )

    true_anomaly: float = Field(
        default=0.0,
        ge=0,
        lt=360,
        description="True anomaly (degrees)"
    )

    @model_validator(mode="after")
    def validate_orbit(self) -> "OrbitParameters":
        """Validate orbit parameters."""
        if self.semi_major_axis < 6378.0:  # Earth radius
            raise ValueError(
                f"Semi-major axis ({self.semi_major_axis} km) must be greater "
                "than Earth's radius (6378 km)"
            )
        return self

    def calculate_period(self, mu: float = 398600.4418) -> float:
        """Calculate orbital period using Kepler's Third Law.
        
        Args:
            mu: Gravitational parameter (km³/s²), defaults to Earth's value
            
        Returns
        -------
            Orbital period in seconds
        """
        return 2 * np.pi * np.sqrt(self.semi_major_axis**3 / mu)

    def calculate_velocities(self, mu: float = 398600.4418) -> tuple[float, float]:
        """Calculate periapsis and apoapsis velocities.
        
        Args:
            mu: Gravitational parameter (km³/s²), defaults to Earth's value
            
        Returns
        -------
            Tuple of (periapsis velocity, apoapsis velocity) in km/s
        """
        rp = self.semi_major_axis * (1 - self.eccentricity)
        ra = self.semi_major_axis * (1 + self.eccentricity)

        vp = np.sqrt(mu * (2/rp - 1/self.semi_major_axis))
        va = np.sqrt(mu * (2/ra - 1/self.semi_major_axis))

        return vp, va
