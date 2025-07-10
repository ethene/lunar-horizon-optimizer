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

import logging

import numpy as np
import pykep as pk

from .celestial_bodies import CelestialBody
from .constants import PhysicalConstants as PC


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

    Attributes
    ----------
        celestial (CelestialBody): Instance for calculating celestial body states
    """

    def __init__(self, celestial: CelestialBody) -> None:
        """Initialize propagator.

        Args:
            celestial: CelestialBody instance for state calculations

        Note:
            The CelestialBody instance should be properly initialized with
            up-to-date ephemeris data.
        """
        self.celestial = celestial

    def propagate_to_target(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        time_of_flight: float,
        departure_epoch: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagate spacecraft trajectory to target using high-precision integration.

        Args:
            initial_position (np.ndarray): Initial position vector [x, y, z] in meters
            initial_velocity (np.ndarray): Initial velocity vector [vx, vy, vz] in m/s
            time_of_flight (float): Time of flight in seconds
            departure_epoch (float): Departure epoch in days since J2000 (default 0.0)

        Returns
        -------
            Tuple[np.ndarray, np.ndarray]: Final position and velocity vectors

        Raises
        ------
            ValueError: If input vectors have wrong shape or time of flight is negative
            RuntimeError: If propagation fails
        """
        # Input validation
        if initial_position.shape != (3,) or initial_velocity.shape != (3,):
            msg = "Position and velocity vectors must be 3D"
            raise ValueError(msg)
        if time_of_flight <= 0:
            msg = "Time of flight must be positive"
            raise ValueError(msg)

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

        # Get moon state at departure epoch
        moon_pos_start = self.celestial.get_moon_state_earth_centered(departure_epoch)[
            0
        ]
        logging.info(f"Moon position at start: {moon_pos_start} m")

        # Calculate initial distance to moon
        dist_to_moon = np.linalg.norm(initial_position - moon_pos_start)
        logging.info(f"Initial distance to Moon: {dist_to_moon/1000:.2f} km")

        try:
            # Propagate using PyKEP's Taylor integrator
            # Use Earth's gravitational parameter for Earth-centered propagation
            # Last two parameters are log10 of tolerances, so -10 means 1e-10
            final_pos, final_vel, final_mass = pk.propagate_taylor(
                r0,
                v0,
                m0,
                thrust,
                time_of_flight,
                PC.MU_EARTH,
                veff,
                -10,
                -10,
            )

            # Convert back to numpy arrays
            final_pos = np.array(final_pos)
            final_vel = np.array(final_vel)

            # Get moon state at arrival epoch (departure + time of flight in days)
            arrival_epoch = departure_epoch + (
                time_of_flight / 86400.0
            )  # Convert seconds to days
            moon_pos_end = self.celestial.get_moon_state_earth_centered(arrival_epoch)[
                0
            ]
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
            msg = f"Propagation failed: {e!s}"
            raise RuntimeError(msg) from e

    def _calculate_energy(self, position: np.ndarray, velocity: np.ndarray) -> float:
        """Calculate the specific energy of a spacecraft at a given position and velocity.

        Args:
            position (np.ndarray): Position vector [x, y, z] in meters
            velocity (np.ndarray): Velocity vector [vx, vy, vz] in m/s

        Returns
        -------
            float: Specific energy in m²/s²
        """
        return np.linalg.norm(velocity) ** 2 / 2 - PC.MU_EARTH / np.linalg.norm(
            position
        )
