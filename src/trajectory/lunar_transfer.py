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
from typing import Tuple, Optional
import pykep as pk
import logging

from .constants import PhysicalConstants as PC
from .celestial_bodies import CelestialBody
from .models import Trajectory, Maneuver, OrbitState
from .target_state import calculate_target_state
from .phase_optimization import find_optimal_phase
from .validation import TrajectoryValidator
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
        
        This method generates a complete lunar transfer trajectory by:
        1. Validating input parameters
        2. Computing Moon states at departure and arrival
        3. Finding optimal departure phase angle
        4. Calculating transfer orbit parameters
        5. Propagating trajectory with gravity effects
        6. Computing maneuver delta-v values
        
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
            
        Note:
            The returned trajectory includes both the Trans-Lunar Injection (TLI)
            and Lunar Orbit Insertion (LOI) maneuvers.
        """
        # Validate inputs
        self.validator.validate_inputs(earth_orbit_alt, moon_orbit_alt, transfer_time)
        
        # Convert units
        r_park = PC.EARTH_RADIUS + earth_orbit_alt * 1000
        r_moon_orbit = self.moon_radius + moon_orbit_alt * 1000
        tof = transfer_time * 86400
        
        # Convert J2000 epoch to MJD2000
        mjd2000_epoch = epoch + 0.5
        
        # Get Moon states
        moon_pos_i, moon_vel_i = self.celestial.get_moon_state_earth_centered(mjd2000_epoch)
        moon_pos_f, moon_vel_f = self.celestial.get_moon_state_earth_centered(mjd2000_epoch + transfer_time)
        
        # Calculate target state
        target_pos, target_vel = calculate_target_state(
            moon_pos_f, moon_vel_f, r_moon_orbit)
        
        # Find optimal departure point
        moon_h = np.cross(moon_pos_i, moon_vel_i)
        moon_h_unit = moon_h / np.linalg.norm(moon_h)
        
        try:
            phase, r1 = find_optimal_phase(
                r_park=r_park,
                moon_pos=moon_pos_f,
                moon_vel=moon_vel_f,
                transfer_time=tof,
                orbit_radius=r_moon_orbit,
                max_revs=max_revolutions
            )
        except ValueError as e:
            raise ValueError(f"Failed to find optimal departure phase: {str(e)}")
        
        # Calculate initial orbit velocity (circular)
        v_init = np.sqrt(PC.EARTH_MU / r_park)
        init_vel = v_init * np.cross(moon_h_unit, r1/np.linalg.norm(r1))
        
        # Create initial and final states
        initial_state = OrbitState.from_state_vectors(
            position=tuple(r1/1000.0),  # Convert to km
            velocity=tuple(init_vel/1000.0),  # Convert to km/s
            epoch=epoch,
            mu=PC.EARTH_MU
        )
        
        final_state = OrbitState.from_state_vectors(
            position=tuple(target_pos/1000.0),  # Convert to km
            velocity=tuple(target_vel/1000.0),  # Convert to km/s
            epoch=epoch + transfer_time,
            mu=PC.EARTH_MU
        )
        
        # Calculate TLI delta-v
        tli_dv = init_vel - target_vel
        
        # Calculate arrival velocity by propagating the trajectory
        arrival_pos, arrival_vel = self.propagator.propagate_to_target(r1, init_vel + tli_dv, tof)
        
        # Calculate LOI delta-v as difference between arrival and target velocities
        loi_dv = target_vel - arrival_vel
        
        # Validate delta-v values
        tli_dv_mag = np.linalg.norm(tli_dv)
        loi_dv_mag = np.linalg.norm(loi_dv)
        self.validator.validate_delta_v(tli_dv_mag, loi_dv_mag)
        
        # Create trajectory and maneuver objects
        trajectory = Trajectory(
            departure_epoch=epoch,
            arrival_epoch=epoch + transfer_time,
            departure_pos=tuple(r1/1000.0),  # Convert to km
            departure_vel=tuple(init_vel/1000.0),  # Convert to km/s
            arrival_pos=tuple(arrival_pos/1000.0),  # Convert to km
            arrival_vel=tuple(arrival_vel/1000.0)  # Convert to km/s
        )
        
        # Convert delta-v from m/s to km/s for maneuvers
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
        
        trajectory.add_maneuver(tli)
        trajectory.add_maneuver(loi)
        
        total_dv = tli_dv_mag + loi_dv_mag
        return trajectory, total_dv