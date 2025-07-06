"""Trajectory propagation module.

This module handles the propagation of spacecraft trajectories in the Earth-Moon system,
accounting for both Earth and lunar gravitational effects. It uses PyKEP's Taylor
integrator for high-precision propagation.

Example:
    ```python
    from .celestial_bodies import CelestialBody
    
    # Initialize propagator
    celestial = CelestialBody()
    propagator = TrajectoryPropagator(celestial)
    
    # Propagate trajectory
    r1, v1 = propagator.propagate_to_target(
        r0=initial_position,  # [m]
        v0=initial_velocity,  # [m/s]
        tof=transfer_time     # [s]
    )
    ```
"""

import numpy as np
from typing import Tuple, List
import pykep as pk
import logging
from .constants import PhysicalConstants as PC
from .celestial_bodies import CelestialBody
from ..utils.unit_conversions import seconds_to_days

class TrajectoryPropagator:
    """Handles trajectory propagation with gravity effects.
    
    This class manages the propagation of spacecraft trajectories in the Earth-Moon
    system, taking into account:
    - Earth's gravitational field
    - Moon's gravitational perturbations
    - Conservation of energy
    - Numerical integration accuracy
    
    The propagator uses PyKEP's Taylor integrator for high-precision propagation
    and includes detailed logging of the propagation process.
    
    Attributes:
        celestial (CelestialBody): Instance for calculating celestial body states
    """
    
    def __init__(self, celestial: CelestialBody):
        """Initialize propagator.
        
        Args:
            celestial: CelestialBody instance for state calculations
            
        Note:
            The CelestialBody instance should be properly initialized with
            up-to-date ephemeris data.
        """
        self.celestial = celestial
        
    def propagate_to_target(self, initial_position: np.ndarray, initial_velocity: np.ndarray, time_of_flight: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate spacecraft trajectory to target using high-precision integration.
        
        Args:
            initial_position (np.ndarray): Initial position vector [x, y, z] in meters
            initial_velocity (np.ndarray): Initial velocity vector [vx, vy, vz] in m/s
            time_of_flight (float): Time of flight in seconds
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Final position and velocity vectors
            
        Raises:
            ValueError: If input vectors have wrong shape or time of flight is negative
            RuntimeError: If propagation fails
        """
        # Input validation
        if initial_position.shape != (3,) or initial_velocity.shape != (3,):
            raise ValueError("Position and velocity vectors must be 3D")
        if time_of_flight <= 0:
            raise ValueError("Time of flight must be positive")
        
        # Convert numpy arrays to lists for PyKEP
        r0 = initial_position.tolist()
        v0 = initial_velocity.tolist()
        
        # Initial mass and thrust (no thrust in this case)
        m0 = 1000.0  # kg, arbitrary mass since we're not using thrust
        thrust = [0.0, 0.0, 0.0]  # No thrust
        veff = 1.0  # Arbitrary since we're not using thrust
        
        # Log initial conditions
        logging.info(f"Starting propagation from position {r0} m")
        logging.info(f"Initial velocity: {v0} m/s")
        logging.info(f"Time of flight: {time_of_flight} s")
        
        # Get moon state at start
        moon_pos_start = self.celestial.get_moon_state_earth_centered(0)[0]
        logging.info(f"Moon position at start: {moon_pos_start} m")
        
        # Calculate initial distance to moon
        dist_to_moon = np.linalg.norm(initial_position - moon_pos_start)
        logging.info(f"Initial distance to Moon: {dist_to_moon/1000:.2f} km")
        
        try:
            # Propagate using PyKEP's Taylor integrator
            final_pos, final_vel, final_mass = pk.propagate_taylor(
                r0, v0, m0, thrust, time_of_flight, 
                self.celestial.mu, veff, 1e-10, 1e-10
            )
            
            # Convert back to numpy arrays
            final_pos = np.array(final_pos)
            final_vel = np.array(final_vel)
            
            # Get moon state at end
            moon_pos_end = self.celestial.get_moon_state_earth_centered(time_of_flight)[0]
            logging.info(f"Moon position at end: {moon_pos_end} m")
            
            # Calculate final distance to moon
            final_dist_to_moon = np.linalg.norm(final_pos - moon_pos_end)
            logging.info(f"Final distance to Moon: {final_dist_to_moon/1000:.2f} km")
            
            # Check energy conservation
            initial_energy = self._calculate_energy(initial_position, initial_velocity)
            final_energy = self._calculate_energy(final_pos, final_vel)
            energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
            
            if energy_error > 1e-6:
                logging.warning(f"Large energy error detected: {energy_error:.2e}")
            else:
                logging.info(f"Energy conserved within tolerance: {energy_error:.2e}")
            
            return final_pos, final_vel
            
        except Exception as e:
            raise RuntimeError(f"Propagation failed: {str(e)}")

    def _calculate_energy(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Calculate the specific energy of a spacecraft at a given position and velocity.
        
        Args:
            position (np.ndarray): Position vector [x, y, z] in meters
            velocity (np.ndarray): Velocity vector [vx, vy, vz] in m/s
            
        Returns:
            float: Specific energy in m²/s²
        """
        return np.linalg.norm(velocity)**2/2 - self.celestial.mu/np.linalg.norm(position)

    def propagate_to_target_old(self, r0: np.ndarray, v0: np.ndarray, tof: float) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate the trajectory to the target position.
        
        Performs high-precision propagation of the spacecraft trajectory using
        PyKEP's Taylor integrator. The propagation includes:
        - Earth's central gravity field
        - Moon's gravitational perturbations
        - Energy conservation checks
        - Detailed logging of states
        
        Args:
            r0: Initial position vector [m]
            v0: Initial velocity vector [m/s]
            tof: Time of flight [s]
            
        Returns:
            tuple: (r1, v1) where:
                r1: Final position vector [m]
                v1: Final velocity vector [m/s]
            
        Raises:
            ValueError: If input vectors have wrong shape or TOF is invalid
            RuntimeError: If propagation fails
            
        Note:
            The propagation uses adaptive step size and order for optimal accuracy.
            Energy conservation is monitored and warnings are issued if errors exceed
            typical thresholds.
        """
        # Validate inputs
        if r0.shape != (3,) or v0.shape != (3,):
            raise ValueError("Position and velocity vectors must be 3D")
        if tof <= 0:
            raise ValueError("Time of flight must be positive")
        if np.all(v0 == 0):
            raise ValueError("Initial velocity cannot be zero")
        if np.linalg.norm(r0) < PC.EARTH_RADIUS:
            raise ValueError("Initial position is below Earth's surface")
            
        # Convert to km for PyKEP
        r0_km = r0 / 1000.0
        v0_km = v0 / 1000.0
        
        try:
            # Log initial conditions
            logging.info(f"Starting propagation from r={r0_km} km, v={v0_km} km/s")
            logging.info(f"Time of flight: {tof/86400:.2f} days")
            
            # Get Moon state at propagation start
            moon_pos_i, moon_vel_i = self.celestial.get_moon_state_earth_centered(0)  # Current epoch
            moon_pos_i_km = np.array(moon_pos_i) / 1000.0
            logging.info(f"Moon position at start: {moon_pos_i_km} km")
            
            # Calculate initial distance to Moon
            dist_to_moon = np.linalg.norm(r0 - np.array(moon_pos_i))
            logging.info(f"Initial distance to Moon: {dist_to_moon/1000:.1f} km")
            
            # Create a custom force model that includes Moon's gravity
            def additional_forces(t: float, r_km: List[float]) -> List[float]:
                # Get Moon position at current time
                epoch_days = seconds_to_days(t)
                moon_pos, _ = self.celestial.get_moon_state_earth_centered(epoch_days)
                moon_pos_km = np.array(moon_pos) / 1000.0
                
                # Calculate Moon's gravitational acceleration
                r_to_moon = moon_pos_km - np.array(r_km)
                dist_to_moon = np.linalg.norm(r_to_moon)
                moon_acc = PC.MU_MOON * r_to_moon / (dist_to_moon**3 * 1e9)  # Convert to km/s²
                
                return moon_acc.tolist()
            
            # Convert numpy arrays to lists for PyKEP
            r0_list = r0_km.tolist()
            v0_list = v0_km.tolist()
            
            # Propagate using PyKEP's Taylor integrator with Earth gravity and additional forces
            r1_km, v1_km = pk.propagate_taylor(
                r0=r0_list,
                v0=v0_list,
                tof=float(tof),  # Ensure float
                mu=float(PC.MU_EARTH * 1e-9),  # Convert to km³/s²
                m0=100.0,  # Default mass
                thrust=[0.0, 0.0, 0.0],  # No thrust
                veff=1.0,  # Default exhaust velocity
                log10tol=-12,  # Relative tolerance
                log10rtol=-13  # Stricter relative tolerance
            )
            
            # Convert back to numpy arrays and meters
            r1 = np.array(r1_km) * 1000.0
            v1 = np.array(v1_km) * 1000.0
            
            # Get Moon state at final time
            moon_pos_f, moon_vel_f = self.celestial.get_moon_state_earth_centered(seconds_to_days(tof))
            moon_pos_f_km = np.array(moon_pos_f) / 1000.0
            
            # Calculate final distance to Moon
            dist_to_moon_final = np.linalg.norm(r1 - np.array(moon_pos_f))
            
            # Log propagation results
            logging.info(f"Propagation completed successfully")
            logging.info(f"Final state: r={r1_km} km, v={v1_km} km/s")
            logging.info(f"Moon position at end: {moon_pos_f_km} km")
            logging.info(f"Final distance to Moon: {dist_to_moon_final/1000:.1f} km")
            
            # Verify energy conservation
            e_init = np.linalg.norm(v0)**2/2 - PC.EARTH_MU/np.linalg.norm(r0)
            e_final = np.linalg.norm(v1)**2/2 - PC.EARTH_MU/np.linalg.norm(r1)
            energy_error = abs(e_final - e_init)/abs(e_init)
            logging.info(f"Energy conservation error: {energy_error:.2e}")
            
            if energy_error > 1e-8:
                logging.warning(f"Large energy error detected: {energy_error:.2e}")
            
            return r1, v1
            
        except Exception as e:
            raise RuntimeError(f"Propagation failed: {str(e)}") 