"""Base trajectory class and core trajectory functionality.

This module provides the base Trajectory class that handles common trajectory operations
like propagation, maneuver application, and state validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
import numpy as np
import logging
from abc import ABC, abstractmethod

from .orbit_state import OrbitState
from .maneuver import Maneuver
from .trajectory_physics import validate_state_vector, propagate_orbit
from src.utils.unit_conversions import km_to_m, m_to_km, mps_to_kmps, kmps_to_mps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class Trajectory(ABC):
    """Base class for orbital trajectories.
    
    Attributes:
        initial_state: Initial orbital state
        maneuvers: List of maneuvers to apply
        start_epoch: Start time of the trajectory
        end_epoch: End time of the trajectory
        propagated_states: Dictionary of propagated states at different epochs
    """
    initial_state: OrbitState
    maneuvers: List[Maneuver] = field(default_factory=list)
    start_epoch: datetime = field(init=False)
    end_epoch: Optional[datetime] = None
    propagated_states: Dict[datetime, OrbitState] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize derived attributes and validate inputs."""
        self.start_epoch = self.initial_state.epoch
        
        # Validate maneuver epochs
        for maneuver in self.maneuvers:
            if maneuver.epoch < self.start_epoch:
                raise ValueError("Maneuver epoch cannot be before trajectory start")
            if self.end_epoch and maneuver.epoch > self.end_epoch:
                raise ValueError("Maneuver epoch cannot be after trajectory end")
                
        # Initialize propagated states with initial state
        self.propagated_states = {self.start_epoch: self.initial_state}
        
    @abstractmethod
    def validate_trajectory(self) -> bool:
        """Validate the complete trajectory including maneuvers.
        
        Returns:
            bool: True if trajectory is valid
        
        This method should be implemented by subclasses to add specific
        validation logic for different types of trajectories.
        """
        pass
        
    def propagate_to(self, target_epoch: datetime) -> OrbitState:
        """Propagate the trajectory to a specific epoch.
        
        Args:
            target_epoch: Time to propagate to
            
        Returns:
            Orbital state at target epoch
            
        Raises:
            ValueError: If target_epoch is invalid or propagation fails
        """
        if target_epoch < self.start_epoch:
            raise ValueError("Cannot propagate backwards from start epoch")
        if self.end_epoch and target_epoch > self.end_epoch:
            raise ValueError("Target epoch exceeds trajectory end time")
            
        # Return cached state if available
        if target_epoch in self.propagated_states:
            return self.propagated_states[target_epoch]
            
        # Find last known state and maneuvers to apply
        last_epoch = max(epoch for epoch in self.propagated_states.keys() if epoch <= target_epoch)
        last_state = self.propagated_states[last_epoch]
        
        # Apply maneuvers and propagate segments
        current_state = last_state
        for maneuver in sorted(self.maneuvers, key=lambda m: m.epoch):
            if last_epoch < maneuver.epoch <= target_epoch:
                # Propagate to maneuver
                pre_maneuver_state = self._propagate_segment(current_state, maneuver.epoch)
                self.propagated_states[maneuver.epoch] = pre_maneuver_state
                
                # Apply maneuver
                new_velocity = maneuver.apply_to_velocity(pre_maneuver_state.velocity)
                current_state = OrbitState(
                    position=pre_maneuver_state.position,
                    velocity=new_velocity,
                    epoch=maneuver.epoch
                )
                
        # Final propagation to target epoch
        final_state = self._propagate_segment(current_state, target_epoch)
        self.propagated_states[target_epoch] = final_state
        return final_state
        
    def _propagate_segment(self, state: OrbitState, target_epoch: datetime) -> OrbitState:
        """Propagate a single trajectory segment without maneuvers.
        
        Args:
            state: Initial state to propagate from
            target_epoch: Time to propagate to
            
        Returns:
            Final state after propagation
        """
        dt = (target_epoch - state.epoch).total_seconds()
        if dt < 0:
            raise ValueError("Cannot propagate backwards")
            
        # Convert to SI units for propagation
        pos_m = km_to_m(state.position)
        vel_ms = kmps_to_mps(state.velocity)
        
        # Propagate using trajectory_physics
        try:
            final_pos_m, final_vel_ms = propagate_orbit(pos_m, vel_ms, dt)
        except Exception as e:
            logger.error(f"Propagation failed: {str(e)}")
            raise ValueError(f"Propagation failed: {str(e)}")
            
        # Convert back to km, km/s
        final_pos = m_to_km(final_pos_m)
        final_vel = mps_to_kmps(final_vel_ms)
        
        return OrbitState(
            position=final_pos,
            velocity=final_vel,
            epoch=target_epoch
        )
        
    def get_state_at(self, epoch: datetime) -> OrbitState:
        """Get the orbital state at a specific epoch.
        
        Args:
            epoch: Time to get state for
            
        Returns:
            Orbital state at specified epoch
        """
        return self.propagate_to(epoch)
        
    def add_maneuver(self, maneuver: Maneuver) -> None:
        """Add a maneuver to the trajectory.
        
        Args:
            maneuver: Maneuver to add
            
        Raises:
            ValueError: If maneuver epoch is invalid
        """
        if maneuver.epoch < self.start_epoch:
            raise ValueError("Maneuver epoch cannot be before trajectory start")
        if self.end_epoch and maneuver.epoch > self.end_epoch:
            raise ValueError("Maneuver epoch cannot be after trajectory end")
            
        self.maneuvers.append(maneuver)
        # Clear cached states after maneuver epoch
        self.propagated_states = {
            epoch: state for epoch, state in self.propagated_states.items()
            if epoch <= maneuver.epoch
        }
        
    def get_total_delta_v(self) -> float:
        """Calculate total delta-v cost of all maneuvers.
        
        Returns:
            Total delta-v in km/s
        """
        return sum(maneuver.magnitude for maneuver in self.maneuvers)
        
    def __str__(self) -> str:
        """String representation of the trajectory."""
        duration = (self.end_epoch - self.start_epoch).total_seconds() if self.end_epoch else "undefined"
        return (
            f"Trajectory from {self.start_epoch.isoformat()} "
            f"(duration: {duration}s, "
            f"maneuvers: {len(self.maneuvers)}, "
            f"total dv: {self.get_total_delta_v():.2f} km/s)"
        ) 