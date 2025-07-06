"""Mission configuration data models.

DEPRECATED: This module is maintained for backward compatibility.
Please import directly from the specific modules:
- src.config.costs
- src.config.isru
- src.config.spacecraft
- src.config.orbit
- src.config.mission
"""

import warnings
from typing import Optional, List, Dict
import numpy as np
from pydantic import BaseModel, Field, model_validator

# Re-export enums
from .enums import IsruResourceType

# Re-export models
from .costs import CostFactors
from .isru import IsruCapabilities, ResourceExtractionRate

def _warn_deprecated(old_name: str, new_module: str):
    warnings.warn(
        f"{old_name} is deprecated. Import from src.config.{new_module} instead.",
        DeprecationWarning,
        stacklevel=2
    )

class PayloadSpecification(BaseModel):
    """Spacecraft payload specifications.
    
    DEPRECATED: Use spacecraft.PayloadSpecification instead.
    """
    
    def __init__(self, *args, **kwargs):
        _warn_deprecated("PayloadSpecification", "spacecraft")
        super().__init__(*args, **kwargs)
    
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
    
    @model_validator(mode='after')
    def validate_masses(self) -> 'PayloadSpecification':
        """Validate mass relationships."""
        if self.payload_mass >= self.dry_mass:
            raise ValueError(
                f"Payload mass ({self.payload_mass} kg) must be less than "
                f"dry mass ({self.dry_mass} kg)"
            )
        return self

class OrbitParameters(BaseModel):
    """Orbital parameters specification.
    
    DEPRECATED: Use orbit.OrbitParameters instead.
    """
    
    def __init__(self, *args, **kwargs):
        _warn_deprecated("OrbitParameters", "orbit")
        super().__init__(*args, **kwargs)
    
    semi_major_axis: float = Field(gt=0, description="Semi-major axis (km)")
    eccentricity: float = Field(ge=0, lt=1, description="Orbit eccentricity")
    inclination: float = Field(ge=0, le=180, description="Orbit inclination (degrees)")
    
    @model_validator(mode='after')
    def validate_orbit(self) -> 'OrbitParameters':
        """Validate orbit parameters."""
        if self.semi_major_axis < 6378.0:  # Earth radius
            raise ValueError(
                f"Semi-major axis ({self.semi_major_axis} km) must be greater "
                "than Earth's radius (6378 km)"
            )
        return self

class MissionConfig(BaseModel):
    """Complete mission configuration.
    
    DEPRECATED: Use mission.MissionConfig instead.
    """
    
    def __init__(self, *args, **kwargs):
        _warn_deprecated("MissionConfig", "mission")
        super().__init__(*args, **kwargs)
    
    name: str = Field(
        ...,
        min_length=1,
        description="Unique mission identifier"
    )
    
    description: Optional[str] = Field(
        None,
        description="Detailed mission description"
    )
    
    payload: PayloadSpecification = Field(
        ...,
        description="Spacecraft and payload specifications"
    )
    
    cost_factors: CostFactors = Field(
        ...,
        description="Mission cost factors"
    )
    
    isru_targets: List[IsruCapabilities] = Field(
        default_factory=list,
        description="List of ISRU production targets"
    )
    
    mission_duration_days: float = Field(
        ...,
        gt=0,
        description="Total planned mission duration (days)"
    )
    
    target_orbit: OrbitParameters = Field(
        ...,
        description="Target lunar orbit parameters (km, degrees)"
    )
    
    @model_validator(mode='after')
    def validate_mission_parameters(self) -> 'MissionConfig':
        """Validate overall mission parameters."""
        if self.mission_duration_days > 1095:  # 3 years
            raise ValueError(
                f"Mission duration ({self.mission_duration_days} days) exceeds "
                "3 years - please verify this is intended"
            )
        return self
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        json_encoders = {
            np.float64: float,
            np.int64: int
        } 