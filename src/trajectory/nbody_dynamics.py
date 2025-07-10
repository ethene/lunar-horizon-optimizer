"""N-body dynamics module for enhanced trajectory propagation.

This module implements n-body gravitational dynamics for more accurate
Earth-Moon trajectory calculations, completing Task 3 requirements.
"""

import logging
from typing import Any

import numpy as np
import pykep as pk
from scipy.integrate import solve_ivp

from .celestial_bodies import CelestialBody
from .constants import PhysicalConstants as PC

# Configure logging
logger = logging.getLogger(__name__)


class NBodyPropagator:
    """N-body gravitational dynamics propagator for accurate trajectory calculation."""

    def __init__(
        self,
        bodies: list[str] | None = None,
        include_sun: bool = True,
        include_moon: bool = True,
    ) -> None:
        """Initialize the N-body propagator.

        Args:
            bodies: List of celestial bodies to include ['earth', 'moon', 'sun']
            include_sun: Include solar gravitational effects
            include_moon: Include lunar gravitational effects
        """
        if bodies is None:
            bodies = ["earth"]
            if include_moon:
                bodies.append("moon")
            if include_sun:
                bodies.append("sun")

        self.bodies = bodies
        self.celestial = CelestialBody()

        # Gravitational parameters [m³/s²]
        self.mu = {
            "earth": PC.EARTH_MU,
            "moon": PC.MOON_MU,
            "sun": PC.SUN_MU,
        }

        logger.info(f"Initialized N-body propagator with bodies: {self.bodies}")

    def propagate_trajectory(
        self,
        initial_state: np.ndarray,
        time_span: tuple[float, float],
        num_points: int = 1000,
        method: str = "DOP853",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate trajectory using n-body dynamics.

        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz] [m, m/s]
            time_span: (t_start, t_end) in seconds
            num_points: Number of output points
            method: Integration method ('RK45', 'DOP853', 'Radau')

        Returns
        -------
            Tuple of (time_array, state_history) where state_history is [6, n] array

        Raises
        ------
            ValueError: If propagation fails
        """
        logger.debug(
            f"Propagating n-body trajectory from t={time_span[0]} to t={time_span[1]}"
        )

        # Time evaluation points
        t_eval = np.linspace(time_span[0], time_span[1], num_points)

        try:
            # Solve the n-body problem
            solution = solve_ivp(
                fun=self._nbody_dynamics,
                t_span=time_span,
                y0=initial_state,
                method=method,
                t_eval=t_eval,
                rtol=1e-12,
                atol=1e-15,
                args=(time_span[0],),  # Reference epoch for celestial body positions
            )

            if not solution.success:
                msg = f"Integration failed: {solution.message}"
                raise ValueError(msg)

            logger.debug(
                f"N-body propagation completed successfully with {len(solution.t)} points"
            )
            return solution.t, solution.y

        except Exception as e:
            msg = f"N-body propagation failed: {e!s}"
            raise ValueError(msg) from e

    def _nbody_dynamics(self, t: float, state: np.ndarray, epoch: float) -> np.ndarray:
        """Compute derivatives for n-body gravitational dynamics.

        Args:
            t: Current time [s]
            state: Current state vector [x, y, z, vx, vy, vz]
            epoch: Reference epoch for celestial body positions

        Returns
        -------
            State derivatives [vx, vy, vz, ax, ay, az]
        """
        # Extract position and velocity
        r = state[:3]  # Position [m]
        v = state[3:]  # Velocity [m/s]

        # Initialize acceleration
        acceleration = np.zeros(3)

        # Time since epoch
        dt_days = t / 86400.0

        # Add gravitational effects from each body
        for body in self.bodies:
            if body == "earth":
                # Earth's gravity (always at origin in Earth-centered frame)
                r_earth = np.array([0.0, 0.0, 0.0])
                r_rel = r - r_earth
                r_mag = np.linalg.norm(r_rel)

                if r_mag > 0:
                    acceleration -= self.mu["earth"] * r_rel / r_mag**3

            elif body == "moon":
                # Moon's gravity
                try:
                    # Get moon position at current time
                    moon_pos, _ = self.celestial.get_moon_state_earth_centered(
                        epoch + dt_days
                    )
                    r_rel = r - moon_pos
                    r_mag = np.linalg.norm(r_rel)

                    if r_mag > 0:
                        acceleration -= self.mu["moon"] * r_rel / r_mag**3

                except Exception as e:
                    logger.warning(f"Could not get moon state at t={t}: {e}")

            elif body == "sun":
                # Sun's gravity (simplified - assumes fixed direction)
                # In reality, would need full solar ephemeris
                sun_distance = 1.496e11  # 1 AU in meters
                sun_pos = np.array([sun_distance, 0.0, 0.0])  # Simplified position

                # Acceleration due to sun on spacecraft
                r_rel_sc = r - sun_pos
                r_mag_sc = np.linalg.norm(r_rel_sc)
                if r_mag_sc > 0:
                    acceleration -= self.mu["sun"] * r_rel_sc / r_mag_sc**3

                # Acceleration due to sun on Earth (indirect effect)
                r_rel_earth = np.array([0.0, 0.0, 0.0]) - sun_pos
                r_mag_earth = np.linalg.norm(r_rel_earth)
                if r_mag_earth > 0:
                    acceleration += self.mu["sun"] * r_rel_earth / r_mag_earth**3

        # Return state derivatives
        return np.concatenate([v, acceleration])

    def calculate_trajectory_accuracy(
        self, nbody_trajectory: np.ndarray, twobody_trajectory: np.ndarray
    ) -> dict[str, float]:
        """Calculate accuracy improvement of n-body vs two-body propagation.

        Args:
            nbody_trajectory: N-body propagated trajectory [6, n]
            twobody_trajectory: Two-body propagated trajectory [6, n]

        Returns
        -------
            Dictionary with accuracy metrics
        """
        # Position differences
        pos_diff = nbody_trajectory[:3, :] - twobody_trajectory[:3, :]
        pos_error = np.linalg.norm(pos_diff, axis=0)

        # Velocity differences
        vel_diff = nbody_trajectory[3:, :] - twobody_trajectory[3:, :]
        vel_error = np.linalg.norm(vel_diff, axis=0)

        return {
            "max_position_error_m": np.max(pos_error),
            "final_position_error_m": pos_error[-1],
            "rms_position_error_m": np.sqrt(np.mean(pos_error**2)),
            "max_velocity_error_ms": np.max(vel_error),
            "final_velocity_error_ms": vel_error[-1],
            "rms_velocity_error_ms": np.sqrt(np.mean(vel_error**2)),
        }


class HighFidelityPropagator:
    """High-fidelity propagator combining PyKEP and n-body dynamics."""

    def __init__(
        self, use_nbody: bool = True, nbody_threshold: float = 1e8
    ) -> None:  # 100,000 km
        """Initialize high-fidelity propagator.

        Args:
            use_nbody: Use n-body dynamics when appropriate
            nbody_threshold: Distance threshold for switching to n-body [m]
        """
        self.use_nbody = use_nbody
        self.nbody_threshold = nbody_threshold
        self.nbody_propagator = NBodyPropagator()
        self.celestial = CelestialBody()

        logger.info(f"Initialized high-fidelity propagator (n-body: {use_nbody})")

    def propagate_adaptive(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        time_of_flight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Adaptive propagation switching between two-body and n-body.

        Args:
            initial_position: Initial position [m]
            initial_velocity: Initial velocity [m/s]
            time_of_flight: Time of flight [s]

        Returns
        -------
            Tuple of (final_position, final_velocity) [m, m/s]
        """
        r_initial = np.linalg.norm(initial_position)

        # Use PyKEP for close Earth orbits, n-body for lunar transfers
        if r_initial < self.nbody_threshold and not self.use_nbody:
            # Use PyKEP two-body propagation
            logger.debug("Using PyKEP two-body propagation")
            final_pos, final_vel = pk.propagate_lagrangian(
                initial_position,
                initial_velocity,
                time_of_flight,
                PC.EARTH_MU,
            )
            return np.array(final_pos), np.array(final_vel)

        # Use n-body propagation
        logger.debug("Using n-body propagation")
        initial_state = np.concatenate([initial_position, initial_velocity])
        time_span = (0.0, time_of_flight)

        _, trajectory = self.nbody_propagator.propagate_trajectory(
            initial_state,
            time_span,
            num_points=100,
        )

        final_position = trajectory[:3, -1]
        final_velocity = trajectory[3:, -1]

        return final_position, final_velocity

    def compare_propagation_methods(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        time_of_flight: float,
    ) -> dict[str, Any]:
        """Compare different propagation methods.

        Args:
            initial_position: Initial position [m]
            initial_velocity: Initial velocity [m/s]
            time_of_flight: Time of flight [s]

        Returns
        -------
            Dictionary with comparison results
        """
        results = {}

        # PyKEP two-body
        try:
            pos_2body, vel_2body = pk.propagate_lagrangian(
                initial_position,
                initial_velocity,
                time_of_flight,
                PC.EARTH_MU,
            )
            results["twobody"] = {
                "position": np.array(pos_2body),
                "velocity": np.array(vel_2body),
                "success": True,
            }
        except Exception as e:
            results["twobody"] = {"success": False, "error": str(e)}

        # N-body
        try:
            pos_nbody, vel_nbody = self.propagate_adaptive(
                initial_position,
                initial_velocity,
                time_of_flight,
            )
            results["nbody"] = {
                "position": pos_nbody,
                "velocity": vel_nbody,
                "success": True,
            }
        except Exception as e:
            results["nbody"] = {"success": False, "error": str(e)}

        # Calculate differences if both succeeded
        if results.get("twobody", {}).get("success") and results.get("nbody", {}).get(
            "success"
        ):
            pos_diff = np.linalg.norm(
                results["nbody"]["position"] - results["twobody"]["position"]
            )
            vel_diff = np.linalg.norm(
                results["nbody"]["velocity"] - results["twobody"]["velocity"]
            )

            results["comparison"] = {
                "position_difference_m": pos_diff,
                "velocity_difference_ms": vel_diff,
                "position_difference_km": pos_diff / 1000,
                "velocity_difference_kms": vel_diff / 1000,
            }

        return results


def enhanced_trajectory_propagation(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    time_of_flight: float,
    high_fidelity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced trajectory propagation for Task 3 completion.

    Args:
        initial_position: Initial position vector [m]
        initial_velocity: Initial velocity vector [m/s]
        time_of_flight: Time of flight [s]
        high_fidelity: Use high-fidelity n-body propagation

    Returns
    -------
        Tuple of (final_position, final_velocity)
    """
    if high_fidelity:
        propagator = HighFidelityPropagator(use_nbody=True)
        return propagator.propagate_adaptive(
            initial_position,
            initial_velocity,
            time_of_flight,
        )
    # Standard PyKEP propagation
    final_pos, final_vel = pk.propagate_lagrangian(
        initial_position,
        initial_velocity,
        time_of_flight,
        PC.EARTH_MU,
    )
    return np.array(final_pos), np.array(final_vel)
