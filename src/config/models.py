"""Central configuration models module.

This module provides the main configuration classes for mission planning,
consolidating the various configuration components into a unified interface.

The module imports and re-exports classes from specialized modules:
- spacecraft.py: Spacecraft and payload specifications
- costs.py: Cost factors and economic parameters
- isru.py: In-Situ Resource Utilization configurations
- orbit.py: Orbital parameters and constraints
- enums.py: Configuration enumerations
"""

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .costs import CostFactors

# Import specialized configuration components
from .enums import IsruResourceType
from .isru import IsruCapabilities, ResourceExtractionRate
from .orbit import OrbitParameters
from .spacecraft import PayloadSpecification


class MissionConfig(BaseModel):
    """Complete mission configuration.

    This class combines all mission configuration parameters including:
    - Mission identification and description
    - Spacecraft and payload specifications
    - Cost factors and economic parameters
    - ISRU targets and capabilities
    - Orbital parameters and constraints
    - Mission timeline and duration

    Attributes
    ----------
        name: Unique mission identifier
        description: Detailed mission description
        payload: Spacecraft and payload specifications
        cost_factors: Mission cost factors
        isru_targets: List of ISRU production targets
        mission_duration_days: Total planned mission duration (days)
        target_orbit: Target lunar orbit parameters
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Unique mission identifier",
    )

    description: str | None = Field(
        None,
        description="Detailed mission description",
    )

    payload: PayloadSpecification = Field(
        ...,
        description="Spacecraft and payload specifications",
    )

    cost_factors: CostFactors = Field(
        ...,
        description="Mission cost factors",
    )

    isru_targets: list[IsruCapabilities] = Field(
        default_factory=list,
        description="List of ISRU production targets",
    )

    mission_duration_days: float = Field(
        ...,
        gt=0,
        description="Total planned mission duration (days)",
    )

    target_orbit: OrbitParameters = Field(
        ...,
        description="Target lunar orbit parameters (km, degrees)",
    )

    @model_validator(mode="after")
    def validate_mission_parameters(self) -> "MissionConfig":
        """Validate overall mission parameters."""
        if self.mission_duration_days > 1095:  # 3 years
            msg = (
                f"Mission duration ({self.mission_duration_days} days) exceeds "
                "3 years - please verify this is intended"
            )
            raise ValueError(
                msg,
            )
        return self

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        json_encoders = {
            np.float64: float,
            np.int64: int,
        }


# Create alias for backward compatibility
IsruTarget = IsruCapabilities


# Re-export all configuration classes for easy access
__all__ = [
    "CostFactors",
    "IsruCapabilities",
    "IsruResourceType",
    "IsruTarget",
    "MissionConfig",
    "OrbitParameters",
    "PayloadSpecification",
    "ResourceExtractionRate",
]
