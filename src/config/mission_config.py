"""Mission configuration data models - Backward Compatibility Module.

DEPRECATED: This module is maintained for backward compatibility only.
Please import directly from the specific modules:
- src.config.models for MissionConfig
- src.config.spacecraft for PayloadSpecification
- src.config.orbit for OrbitParameters
- src.config.costs for CostFactors
- src.config.isru for IsruCapabilities

This module will be removed in a future version.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "mission_config.py is deprecated. Import from src.config.models instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from the new models module for backward compatibility
from .costs import CostFactors
from .enums import IsruResourceType
from .isru import IsruCapabilities
from .models import MissionConfig
from .orbit import OrbitParameters
from .spacecraft import PayloadSpecification

# Backward compatibility aliases
IsruTarget = IsruCapabilities

__all__ = [
    "CostFactors",
    "IsruCapabilities",
    "IsruResourceType",
    "IsruTarget",
    "MissionConfig",
    "OrbitParameters",
    "PayloadSpecification",
]
