"""Transfer parameters for trajectory generation.

This module defines the TransferParameters class which encapsulates
all parameters needed to generate a transfer trajectory.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .constants import PhysicalConstants as PC
from .defaults import TransferDefaults as TD
from .orbit_state import OrbitState


@dataclass
class TransferParameters:
    """Parameters for generating transfer trajectories.

    Attributes
    ----------
        departure_time: UTC departure time
        tof_days: Time of flight in days
        initial_orbit: Initial parking orbit altitude in km
        final_orbit: Target orbit radius in km
        max_revs: Maximum number of revolutions (default: 1)
        max_tli_dv: Maximum allowed TLI delta-v in km/s
        min_tli_dv: Minimum allowed TLI delta-v in km/s
    """

    departure_time: datetime
    tof_days: float
    initial_orbit: float
    final_orbit: float
    max_revs: int = TD.MAX_REVOLUTIONS
    max_tli_dv: float = TD.MAX_TLI_DV
    min_tli_dv: float = TD.MIN_TLI_DV

    def __post_init__(self):
        """Validate parameters after initialization."""
        if not self.departure_time.tzinfo:
            msg = "departure_time must be timezone-aware"
            raise ValueError(msg)
        self.validate()

    def validate(self) -> None:
        """Validate all parameters."""
        # Time of flight validation
        if not TD.MIN_TOF <= self.tof_days <= TD.MAX_TOF:
            msg = f"Time of flight must be between {TD.MIN_TOF} and {TD.MAX_TOF} days"
            raise ValueError(
                msg,
            )

        # Initial orbit validation
        if self.initial_orbit < PC.MIN_PERIGEE:
            msg = f"Initial orbit altitude must be at least {PC.MIN_PERIGEE} km"
            raise ValueError(
                msg,
            )

        # Final orbit validation
        if self.final_orbit > PC.MAX_APOGEE:
            msg = f"Final orbit radius cannot exceed {PC.MAX_APOGEE} km"
            raise ValueError(
                msg,
            )

        # Delta-v constraints validation
        if not 0 < self.min_tli_dv <= self.max_tli_dv:
            msg = "min_tli_dv must be positive and not greater than max_tli_dv"
            raise ValueError(
                msg,
            )

        # Revolutions validation
        if self.max_revs < 0:
            msg = "max_revs cannot be negative"
            raise ValueError(msg)

    def get_initial_state(self) -> OrbitState:
        """Convert initial orbit specification to OrbitState.

        Returns
        -------
            OrbitState object representing the initial parking orbit
        """
        radius = PC.EARTH_RADIUS + self.initial_orbit
        velocity = np.sqrt(PC.EARTH_MU / radius)  # Circular orbit velocity

        return OrbitState(
            radius=radius,
            velocity=velocity,
            inclination=TD.DEFAULT_INCLINATION,
        )
