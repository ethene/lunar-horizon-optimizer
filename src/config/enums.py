"""Enumeration types for mission configuration.

This module contains all enum definitions used across the configuration models.
"""

from enum import Enum


class IsruResourceType(str, Enum):
    """Types of resources that can be extracted via ISRU."""

    # Primary resources
    WATER = "water"  # Water ice or bound water
    REGOLITH = "regolith"  # Raw lunar regolith

    # Derived resources
    OXYGEN = "oxygen"  # Oxygen from regolith/minerals
    HYDROGEN = "hydrogen"  # Hydrogen from water or other sources
    METHANE = "methane"  # Methane from CO2 and H2

    # Advanced resources
    METALS = "metals"  # Various metallic resources
    HELIUM3 = "helium3"  # Helium-3 from regolith
