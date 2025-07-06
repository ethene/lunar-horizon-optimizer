"""Lunar transfer trajectory generation module.

This module implements lunar transfer trajectory generation using PyKEP's Lambert solver
and propagation tools. It accounts for both Earth and lunar gravitational effects and
generates practical transfer orbits with reasonable delta-v requirements.

The module is structured into several components:
- TrajectoryValidator: Input validation and constraints
- TrajectoryPropagator: Trajectory propagation with gravity effects
- LunarTransfer: Main trajectory generation logic

Example:
    ```python
    # Initialize lunar transfer generator
    transfer = LunarTransfer(
        min_earth_alt=200,  # km
        max_earth_alt=1000,  # km
        min_moon_alt=50,    # km
        max_moon_alt=500    # km
    )
    
    # Generate transfer trajectory
    trajectory, total_dv = transfer.generate_transfer(
        epoch=10000.0,           # days since J2000
        earth_orbit_alt=300.0,   # km
        moon_orbit_alt=100.0,    # km
        transfer_time=4.0,       # days
        max_revolutions=0
    )
    ```
"""

import numpy as np
from typing import Tuple, Optional, Dict
import pykep as pk
import logging

from .constants import PhysicalConstants as PC
from .celestial_bodies import CelestialBody
from .models import Trajectory, Maneuver, OrbitState
from .target_state import calculate_target_state
from .phase_optimization import find_optimal_phase
from .trajectory_validator import TrajectoryValidator  # Import from renamed module
from .propagator import TrajectoryPropagator

class LunarTransfer:
    """Generates lunar transfer trajectories using PyKEP.
    
    This class handles the computation of transfer trajectories from Earth parking orbits
    to lunar orbits, accounting for:
    - Earth and Moon gravitational parameters
    - Realistic delta-v constraints
    - Proper phasing for efficient transfers
    - Conservation of energy and angular momentum
    
    The implementation is split into several modules for maintainability:
    - validator.py: Input validation and constraints
    - propagator.py: Trajectory propagation with gravity effects
    - target_state.py: Target state calculation
    - phase_optimization.py: Phase angle optimization
    
    Attributes:
        moon_soi (float): Moon's sphere of influence radius [m]
        moon_radius (float): Moon's radius [m]
        celestial (CelestialBody): Celestial body calculator instance
        validator (TrajectoryValidator): Trajectory parameter validator
        propagator (TrajectoryPropagator): Trajectory propagator
    """
    
    def __init__(self,
                moon_soi: float = PC.MOON_SOI,
                moon_radius: float = PC.MOON_RADIUS,
                min_earth_alt: float = 200,
                max_earth_alt: float = 1000,
                min_moon_alt: float = 50,
                max_moon_alt: float = 500):
        """Initialize lunar transfer trajectory generator.

        Args:
            moon_soi: Moon's sphere of influence radius [m]
            moon_radius: Moon's radius [m]
            min_earth_alt: Minimum Earth parking orbit altitude [km]
            max_earth_alt: Maximum Earth parking orbit altitude [km]
            min_moon_alt: Minimum lunar orbit altitude [km]
            max_moon_alt: Maximum lunar orbit altitude [km]
            
        Note:
            All altitude inputs should be in kilometers, but moon_soi and moon_radius
            should be in meters.
        """
        self.moon_soi = moon_soi
        self.moon_radius = moon_radius
        self.celestial = CelestialBody()
        
        # Initialize components
        self.validator = TrajectoryValidator(
            min_earth_alt=min_earth_alt,
            max_earth_alt=max_earth_alt,
            min_moon_alt=min_moon_alt,
            max_moon_alt=max_moon_alt
        )
        self.propagator = TrajectoryPropagator(self.celestial)
    
    def generate_transfer(self,
                         epoch: float,
                         earth_orbit_alt: float,
                         moon_orbit_alt: float,
                         transfer_time: float,
                         max_revolutions: int = 0) -> Tuple[Trajectory, float]:
        """Generate lunar transfer trajectory.
        
        This method orchestrates the complete lunar transfer trajectory generation
        by delegating to specialized methods for each step.
        
        Args:
            epoch: Start epoch in days since J2000
            earth_orbit_alt: Initial parking orbit altitude [km]
            moon_orbit_alt: Final lunar orbit altitude [km]
            transfer_time: Transfer time [days]
            max_revolutions: Maximum number of revolutions for Lambert solver
            
        Returns:
            Tuple[Trajectory, float]: Trajectory object and total delta-v [m/s]
            
        Raises:
            ValueError: If input parameters are invalid or no trajectory is found
        """
        # Step 1: Validate and prepare inputs
        transfer_params = self._validate_and_prepare_inputs(
            epoch, earth_orbit_alt, moon_orbit_alt, transfer_time
        )
        
        # Step 2: Calculate celestial body states
        moon_states = self._calculate_moon_states(
            transfer_params['mjd2000_epoch'], transfer_time, transfer_params['r_moon_orbit']
        )
        
        # Step 3: Find optimal departure point
        departure_state = self._find_optimal_departure(
            transfer_params, moon_states, max_revolutions
        )
        
        # Step 4: Calculate trajectory and maneuvers
        trajectory, total_dv = self._build_trajectory(
            epoch, transfer_time, departure_state, moon_states, transfer_params
        )
        
        return trajectory, total_dv
    
    def _validate_and_prepare_inputs(self, epoch: float, earth_orbit_alt: float, 
                                   moon_orbit_alt: float, transfer_time: float) -> Dict[str, float]:
        """Validate inputs and prepare transfer parameters.
        
        Args:
            epoch: Start epoch in days since J2000
            earth_orbit_alt: Initial parking orbit altitude [km]
            moon_orbit_alt: Final lunar orbit altitude [km]
            transfer_time: Transfer time [days]
            
        Returns:
            Dictionary containing prepared transfer parameters
            
        Raises:
            ValueError: If input validation fails
        """
        # Validate inputs
        self.validator.validate_inputs(earth_orbit_alt, moon_orbit_alt, transfer_time)
        
        # Convert units and prepare parameters
        return {
            'r_park': PC.EARTH_RADIUS + earth_orbit_alt * 1000,  # [m]
            'r_moon_orbit': self.moon_radius + moon_orbit_alt * 1000,  # [m]
            'tof': transfer_time * 86400,  # [s]
            'mjd2000_epoch': epoch + 0.5  # Convert J2000 to MJD2000
        }
    
    def _calculate_moon_states(self, mjd2000_epoch: float, transfer_time: float, 
                             r_moon_orbit: float = None) -> Dict[str, np.ndarray]:
        """Calculate Moon states at departure and arrival.
        
        Args:
            mjd2000_epoch: Epoch in MJD2000 format
            transfer_time: Transfer time [days]
            
        Returns:
            Dictionary containing Moon position and velocity states
        """
        # Get Moon states at departure and arrival
        moon_pos_i, moon_vel_i = self.celestial.get_moon_state_earth_centered(mjd2000_epoch)
        moon_pos_f, moon_vel_f = self.celestial.get_moon_state_earth_centered(
            mjd2000_epoch + transfer_time
        )
        
        # Calculate target state in lunar orbit
        target_pos, target_vel = calculate_target_state(
            moon_pos_f, moon_vel_f, r_moon_orbit or (self.moon_radius + 100000)
        )
        
        return {
            'moon_pos_initial': moon_pos_i,
            'moon_vel_initial': moon_vel_i,
            'moon_pos_final': moon_pos_f,
            'moon_vel_final': moon_vel_f,
            'target_pos': target_pos,
            'target_vel': target_vel
        }
    
    def _find_optimal_departure(self, transfer_params: Dict[str, float], 
                              moon_states: Dict[str, np.ndarray], 
                              max_revolutions: int) -> Dict[str, np.ndarray]:
        """Find optimal departure point and initial orbit conditions.
        
        Args:
            transfer_params: Transfer parameters from validation step
            moon_states: Moon states from calculation step
            max_revolutions: Maximum revolutions for Lambert solver
            
        Returns:
            Dictionary containing departure state information
            
        Raises:
            ValueError: If optimal departure phase cannot be found
        """
        # Calculate moon orbital plane unit vector
        moon_h = np.cross(moon_states['moon_pos_initial'], moon_states['moon_vel_initial'])
        moon_h_unit = moon_h / np.linalg.norm(moon_h)
        
        # Find optimal departure phase
        try:
            phase, r1 = find_optimal_phase(
                r_park=transfer_params['r_park'],
                moon_pos=moon_states['moon_pos_final'],
                moon_vel=moon_states['moon_vel_final'],
                transfer_time=transfer_params['tof'],
                orbit_radius=transfer_params['r_moon_orbit'],
                max_revs=max_revolutions
            )
        except ValueError as e:
            raise ValueError(f"Failed to find optimal departure phase: {str(e)}")
        
        # Calculate initial orbital velocity (circular)
        v_init = np.sqrt(PC.EARTH_MU / transfer_params['r_park'])
        init_vel = v_init * np.cross(moon_h_unit, r1 / np.linalg.norm(r1))
        
        return {
            'position': r1,
            'velocity': init_vel,
            'phase': phase,
            'moon_h_unit': moon_h_unit
        }
    
    def _build_trajectory(self, epoch: float, transfer_time: float, 
                        departure_state: Dict[str, np.ndarray],
                        moon_states: Dict[str, np.ndarray],
                        transfer_params: Dict[str, float]) -> Tuple[Trajectory, float]:
        """Build complete trajectory with maneuvers.
        
        Args:
            epoch: Start epoch in days since J2000
            transfer_time: Transfer time [days]
            departure_state: Departure state information
            moon_states: Moon states information
            transfer_params: Transfer parameters
            
        Returns:
            Tuple of trajectory object and total delta-v [m/s]
        """
        r1 = departure_state['position']
        init_vel = departure_state['velocity']
        target_pos = moon_states['target_pos']
        target_vel = moon_states['target_vel']
        tof = transfer_params['tof']
        
        # Calculate maneuver delta-v values
        tli_dv, loi_dv, arrival_pos, arrival_vel = self._calculate_maneuvers(
            r1, init_vel, target_vel, tof
        )
        
        # Create trajectory object
        trajectory = Trajectory(
            departure_epoch=epoch,
            arrival_epoch=epoch + transfer_time,
            departure_pos=tuple(r1 / 1000.0),  # Convert to km
            departure_vel=tuple(init_vel / 1000.0),  # Convert to km/s
            arrival_pos=tuple(arrival_pos / 1000.0),  # Convert to km
            arrival_vel=tuple(arrival_vel / 1000.0)  # Convert to km/s
        )
        
        # Add maneuver objects to trajectory
        self._add_maneuvers_to_trajectory(trajectory, epoch, transfer_time, tli_dv, loi_dv)
        
        # Calculate total delta-v
        total_dv = np.linalg.norm(tli_dv) + np.linalg.norm(loi_dv)
        return trajectory, total_dv
    
    def _calculate_maneuvers(self, r1: np.ndarray, init_vel: np.ndarray, 
                           target_vel: np.ndarray, tof: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate TLI and LOI maneuver delta-v values.
        
        Args:
            r1: Initial position vector [m]
            init_vel: Initial velocity vector [m/s]
            target_vel: Target velocity vector [m/s]
            tof: Time of flight [s]
            
        Returns:
            Tuple of (TLI delta-v, LOI delta-v, arrival position, arrival velocity)
        """
        # Calculate TLI delta-v
        tli_dv = init_vel - target_vel
        
        # Propagate trajectory to calculate arrival state
        arrival_pos, arrival_vel = self.propagator.propagate_to_target(
            r1, init_vel + tli_dv, tof
        )
        
        # Calculate LOI delta-v
        loi_dv = target_vel - arrival_vel
        
        # Validate delta-v magnitudes
        tli_dv_mag = np.linalg.norm(tli_dv)
        loi_dv_mag = np.linalg.norm(loi_dv)
        self.validator.validate_delta_v(tli_dv_mag, loi_dv_mag)
        
        return tli_dv, loi_dv, arrival_pos, arrival_vel
    
    def _add_maneuvers_to_trajectory(self, trajectory: Trajectory, epoch: float, 
                                   transfer_time: float, tli_dv: np.ndarray, 
                                   loi_dv: np.ndarray) -> None:
        """Add TLI and LOI maneuvers to the trajectory.
        
        Args:
            trajectory: Trajectory object to add maneuvers to
            epoch: Start epoch in days since J2000
            transfer_time: Transfer time [days]
            tli_dv: Trans-Lunar Injection delta-v [m/s]
            loi_dv: Lunar Orbit Insertion delta-v [m/s]
        """
        # Create maneuver objects (convert delta-v from m/s to km/s)
        tli = Maneuver(
            epoch=epoch,
            delta_v=tuple(tli_dv / 1000.0),  # Convert to km/s
            name="Trans-Lunar Injection"
        )
        
        loi = Maneuver(
            epoch=epoch + transfer_time,
            delta_v=tuple(loi_dv / 1000.0),  # Convert to km/s
            name="Lunar Orbit Insertion"
        )
        
        # Add maneuvers to trajectory
        trajectory.add_maneuver(tli)
        trajectory.add_maneuver(loi)