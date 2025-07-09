"""Orbital maneuver representation and calculations.

This module provides the Maneuver class for representing impulsive maneuvers
and utility functions for maneuver calculations and validation.
"""

from dataclasses import dataclass
import numpy as np
from datetime import datetime
import logging

from .trajectory_physics import validate_delta_v
from src.utils.unit_conversions import kmps_to_mps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class Maneuver:
    """Represents an impulsive orbital maneuver.

    Attributes
    ----------
        delta_v: Velocity change vector in km/s [dvx, dvy, dvz]
        epoch: Time of maneuver execution
        description: Optional description of the maneuver's purpose
    """

    delta_v: tuple[float, float, float]
    epoch: datetime
    description: str | None = None

    def __post_init__(self):
        """Validate maneuver parameters after initialization."""
        if not isinstance(self.delta_v, tuple | list | np.ndarray) or len(self.delta_v) != 3:
            msg = "delta_v must be a 3D vector"
            raise ValueError(msg)

        # Convert to numpy array if needed
        if not isinstance(self.delta_v, np.ndarray):
            self.delta_v = np.array(self.delta_v)

        # Validate delta-v magnitude using trajectory_physics
        dv_ms = kmps_to_mps(self.delta_v)
        if not validate_delta_v(dv_ms):
            msg = "Invalid delta-v magnitude"
            raise ValueError(msg)

        if not isinstance(self.epoch, datetime):
            msg = "epoch must be a datetime object"
            raise ValueError(msg)
        if self.epoch.tzinfo is None:
            msg = "epoch must be timezone-aware"
            raise ValueError(msg)

    @property
    def magnitude(self) -> float:
        """Get the magnitude of the delta-v vector in km/s."""
        return float(np.linalg.norm(self.delta_v))

    def get_delta_v_si(self) -> np.ndarray:
        """Get the delta-v vector in SI units (m/s)."""
        return kmps_to_mps(self.delta_v)

    def get_delta_v_ms(self) -> np.ndarray:
        """Get the delta-v vector in m/s (alias for get_delta_v_si)."""
        return self.get_delta_v_si()

    def apply_to_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Apply the maneuver to a velocity vector.

        Args:
            velocity: Initial velocity vector in km/s

        Returns
        -------
            New velocity vector in km/s after applying the maneuver
        """
        if not isinstance(velocity, np.ndarray) or velocity.shape != (3,):
            msg = "velocity must be a 3D numpy array"
            raise ValueError(msg)

        return velocity + self.delta_v

    def scale(self, factor: float) -> "Maneuver":
        """Create a new maneuver with delta-v scaled by a factor.

        Args:
            factor: Scale factor to apply to delta-v

        Returns
        -------
            New Maneuver object with scaled delta-v
        """
        if factor < 0:
            msg = "Scale factor must be non-negative"
            raise ValueError(msg)

        return Maneuver(
            delta_v=tuple(self.delta_v * factor),
            epoch=self.epoch,
            description=f"{self.description} (scaled by {factor})" if self.description else None
        )

    def reverse(self) -> "Maneuver":
        """Create a new maneuver with reversed delta-v direction.

        Returns
        -------
            New Maneuver object with reversed delta-v
        """
        return Maneuver(
            delta_v=tuple(-self.delta_v),
            epoch=self.epoch,
            description=f"{self.description} (reversed)" if self.description else None
        )

    def __str__(self) -> str:
        """String representation of the maneuver."""
        desc = f" ({self.description})" if self.description else ""
        return f"Maneuver at {self.epoch.isoformat()}: dv = {self.magnitude:.2f} km/s{desc}"
