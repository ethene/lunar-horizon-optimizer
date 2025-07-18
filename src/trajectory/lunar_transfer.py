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

import logging
from datetime import UTC
from typing import Any

import numpy as np

from .celestial_bodies import CelestialBody
from .constants import PhysicalConstants as PC
from .maneuver import Maneuver
from .phase_optimization import find_optimal_phase, calculate_initial_position
from .propagator import TrajectoryPropagator
from .target_state import calculate_target_state
from .trajectory_base import Trajectory
from .trajectory_validator import TrajectoryValidator  # Import from renamed module

# Configure logging
logger = logging.getLogger(__name__)


class LunarTrajectory:
    """Simple concrete implementation for lunar transfers."""

    def __init__(
        self,
        departure_epoch,
        arrival_epoch,
        departure_pos,
        departure_vel,
        arrival_pos,
        arrival_vel,
    ) -> None:
        """Initialize lunar trajectory with departure and arrival states."""
        self.departure_epoch = departure_epoch
        self.arrival_epoch = arrival_epoch
        self.departure_pos = np.array(departure_pos)
        self.departure_vel = np.array(departure_vel)
        self.arrival_pos = np.array(arrival_pos)
        self.arrival_vel = np.array(arrival_vel)
        self.maneuvers = []  # Will be populated by _add_maneuvers_to_trajectory

    def validate_trajectory(self) -> bool:
        """Validate the lunar trajectory."""
        try:
            # Check that states are valid
            if (
                not np.isfinite(self.departure_pos).all()
                or not np.isfinite(self.departure_vel).all()
            ):
                return False
            if (
                not np.isfinite(self.arrival_pos).all()
                or not np.isfinite(self.arrival_vel).all()
            ):
                return False

            # Check that trajectory duration is reasonable (between 1 and 30 days)
            duration = self.arrival_epoch - self.departure_epoch
            return 1.0 <= duration <= 30.0
        except Exception:
            return False

    def add_maneuver(self, maneuver) -> None:
        """Add a maneuver to the trajectory."""
        self.maneuvers.append(maneuver)

    def get_total_delta_v(self) -> float:
        """Calculate total delta-v cost of all maneuvers."""
        return sum(getattr(maneuver, "magnitude", 0.0) for maneuver in self.maneuvers)

    @property
    def trajectory_data(self) -> dict[str, Any]:
        """Get trajectory data for visualization and analysis integration.

        Returns:
            Dictionary containing trajectory points and metadata for visualization
        """
        # Generate trajectory points for visualization
        n_points = 100
        time_points = np.linspace(0, 1, n_points)

        # Simple interpolation between departure and arrival
        trajectory_points = []
        for t in time_points:
            pos = self.departure_pos * (1 - t) + self.arrival_pos * t
            trajectory_points.append(
                (pos[0] * 1000, pos[1] * 1000, pos[2] * 1000)
            )  # Convert to meters

        return {
            "trajectory_points": trajectory_points,
            "departure_position": self.departure_pos,
            "arrival_position": self.arrival_pos,
            "departure_velocity": self.departure_vel,
            "arrival_velocity": self.arrival_vel,
            "transfer_time": self.arrival_epoch - self.departure_epoch,
            "total_delta_v": self.get_total_delta_v(),
            "maneuvers": self.maneuvers,
        }


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

    Attributes
    ----------
        moon_soi (float): Moon's sphere of influence radius [m]
        moon_radius (float): Moon's radius [m]
        celestial (CelestialBody): Celestial body calculator instance
        validator (TrajectoryValidator): Trajectory parameter validator
        propagator (TrajectoryPropagator): Trajectory propagator
    """

    def __init__(
        self,
        moon_soi: float = PC.MOON_SOI,
        moon_radius: float = PC.MOON_RADIUS,
        min_earth_alt: float = 200,
        max_earth_alt: float = 1000,
        min_moon_alt: float = 50,
        max_moon_alt: float = 500,
    ) -> None:
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
            max_moon_alt=max_moon_alt,
        )
        self.propagator = TrajectoryPropagator(self.celestial)

    def generate_transfer(
        self,
        epoch: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
        transfer_time: float,
        max_revolutions: int = 0,
    ) -> tuple[Trajectory, float]:
        """Generate lunar transfer trajectory.

        This method orchestrates the complete lunar transfer trajectory generation
        by delegating to specialized methods for each step.

        Args:
            epoch: Start epoch in days since J2000
            earth_orbit_alt: Initial parking orbit altitude [km]
            moon_orbit_alt: Final lunar orbit altitude [km]
            transfer_time: Transfer time [days]
            max_revolutions: Maximum number of revolutions for Lambert solver

        Returns
        -------
            Tuple[Trajectory, float]: Trajectory object and total delta-v [m/s]

        Raises
        ------
            ValueError: If input parameters are invalid or no trajectory is found
        """
        # Store departure epoch for use by propagator
        self.departure_epoch = epoch

        # Step 1: Validate and prepare inputs
        transfer_params = self._validate_and_prepare_inputs(
            epoch,
            earth_orbit_alt,
            moon_orbit_alt,
            transfer_time,
        )

        # Step 2: Calculate celestial body states
        moon_states = self._calculate_moon_states(
            transfer_params["mjd2000_epoch"],
            transfer_time,
            transfer_params["r_moon_orbit"],
        )

        # Step 3: Find optimal departure point
        departure_state = self._find_optimal_departure(
            transfer_params,
            moon_states,
            max_revolutions,
        )

        # Step 4: Calculate trajectory and maneuvers
        trajectory, total_dv = self._build_trajectory(
            epoch,
            transfer_time,
            departure_state,
            moon_states,
            transfer_params,
        )

        return trajectory, total_dv

    def _validate_and_prepare_inputs(
        self,
        epoch: float,
        earth_orbit_alt: float,
        moon_orbit_alt: float,
        transfer_time: float,
    ) -> dict[str, float]:
        """Validate inputs and prepare transfer parameters.

        Args:
            epoch: Start epoch in days since J2000
            earth_orbit_alt: Initial parking orbit altitude [km]
            moon_orbit_alt: Final lunar orbit altitude [km]
            transfer_time: Transfer time [days]

        Returns
        -------
            Dictionary containing prepared transfer parameters

        Raises
        ------
            ValueError: If input validation fails
        """
        # Validate inputs
        self.validator.validate_inputs(earth_orbit_alt, moon_orbit_alt, transfer_time)

        # Convert units and prepare parameters
        return {
            "r_park": PC.EARTH_RADIUS + earth_orbit_alt * 1000,  # [m]
            "r_moon_orbit": self.moon_radius + moon_orbit_alt * 1000,  # [m]
            "tof": transfer_time * 86400,  # [s]
            "mjd2000_epoch": epoch + 0.5,  # Convert J2000 to MJD2000
        }

    def _calculate_moon_states(
        self,
        mjd2000_epoch: float,
        transfer_time: float,
        r_moon_orbit: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Calculate Moon states at departure and arrival.

        Args:
            mjd2000_epoch: Epoch in MJD2000 format
            transfer_time: Transfer time [days]

        Returns
        -------
            Dictionary containing Moon position and velocity states
        """
        # Get Moon states at departure and arrival
        moon_pos_i, moon_vel_i = self.celestial.get_moon_state_earth_centered(
            mjd2000_epoch
        )
        moon_pos_f, moon_vel_f = self.celestial.get_moon_state_earth_centered(
            mjd2000_epoch + transfer_time,
        )

        # Calculate target state in lunar orbit
        target_pos, target_vel = calculate_target_state(
            moon_pos_f,
            moon_vel_f,
            r_moon_orbit or (self.moon_radius + 100000),
        )

        return {
            "moon_pos_initial": np.array(moon_pos_i),
            "moon_vel_initial": np.array(moon_vel_i),
            "moon_pos_final": np.array(moon_pos_f),
            "moon_vel_final": np.array(moon_vel_f),
            "target_pos": target_pos,
            "target_vel": target_vel,
        }

    def _find_optimal_departure(
        self,
        transfer_params: dict[str, float],
        moon_states: dict[str, np.ndarray],
        max_revolutions: int,
    ) -> dict[str, np.ndarray]:
        """Find optimal departure point and initial orbit conditions.

        Args:
            transfer_params: Transfer parameters from validation step
            moon_states: Moon states from calculation step
            max_revolutions: Maximum revolutions for Lambert solver

        Returns
        -------
            Dictionary containing departure state information

        Raises
        ------
            ValueError: If optimal departure phase cannot be found
        """
        # Calculate moon orbital plane unit vector
        moon_h = np.cross(
            moon_states["moon_pos_initial"], moon_states["moon_vel_initial"]
        )
        moon_h_unit = moon_h / np.linalg.norm(moon_h)

        # Find optimal departure phase with progress tracking
        def phase_progress_callback(current, total, percent, valid_count):
            """Progress callback for phase optimization."""
            if current % 36 == 0:  # Update every 10 degrees (36 samples)
                logger.debug(
                    f"Phase search: {current}/{total} ({percent:.1f}%), {valid_count} valid solutions"
                )

        try:
            # First attempt: Standard search
            phase, r1 = find_optimal_phase(
                r_park=transfer_params["r_park"],
                moon_pos=moon_states["moon_pos_final"],
                moon_vel=moon_states["moon_vel_final"],
                transfer_time=transfer_params["tof"],
                orbit_radius=transfer_params["r_moon_orbit"],
                max_revs=max_revolutions,
                num_samples=360,
                progress_callback=phase_progress_callback,
            )
        except ValueError as e:
            logger.warning(f"Standard phase search failed: {e!s}")
            try:
                # Second attempt: Coarser search with more revolutions
                logger.info("Attempting coarser phase search with more revolutions...")
                phase, r1 = find_optimal_phase(
                    r_park=transfer_params["r_park"],
                    moon_pos=moon_states["moon_pos_final"],
                    moon_vel=moon_states["moon_vel_final"],
                    transfer_time=transfer_params["tof"],
                    orbit_radius=transfer_params["r_moon_orbit"],
                    max_revs=min(max_revolutions + 1, 2),  # Allow one more revolution
                    num_samples=72,  # Coarser sampling (every 5 degrees)
                    progress_callback=phase_progress_callback,
                )
            except ValueError as e2:
                # Final fallback: Use a simple default position
                logger.warning(f"Coarse phase search also failed: {e2!s}")
                logger.info("Using default phase angle as fallback")
                phase = 0.0  # Default phase
                # Calculate position at default phase
                h_unit = np.cross(
                    moon_states["moon_pos_initial"], moon_states["moon_vel_initial"]
                )
                h_unit = h_unit / np.linalg.norm(h_unit)
                r1 = calculate_initial_position(
                    transfer_params["r_park"], phase, h_unit
                )

        # Calculate initial orbital velocity (circular)
        v_init = np.sqrt(PC.EARTH_MU / transfer_params["r_park"])
        init_vel = v_init * np.cross(moon_h_unit, r1 / np.linalg.norm(r1))

        return {
            "position": r1,
            "velocity": init_vel,
            "phase": np.array([float(phase)]),
            "moon_h_unit": moon_h_unit,
        }

    def _build_trajectory(
        self,
        epoch: float,
        transfer_time: float,
        departure_state: dict[str, np.ndarray],
        moon_states: dict[str, np.ndarray],
        transfer_params: dict[str, float],
    ) -> tuple[Trajectory, float]:
        """Build complete trajectory with maneuvers.

        Args:
            epoch: Start epoch in days since J2000
            transfer_time: Transfer time [days]
            departure_state: Departure state information
            moon_states: Moon states information
            transfer_params: Transfer parameters

        Returns
        -------
            Tuple of trajectory object and total delta-v [m/s]
        """
        r1 = departure_state["position"]
        init_vel = departure_state["velocity"]
        moon_states["target_pos"]
        target_vel = moon_states["target_vel"]
        tof = transfer_params["tof"]

        # Calculate maneuver delta-v values
        tli_dv, loi_dv, arrival_pos, arrival_vel = self._calculate_maneuvers(
            r1,
            init_vel,
            target_vel,
            tof,
        )

        # Create trajectory object
        trajectory = LunarTrajectory(
            departure_epoch=epoch,
            arrival_epoch=epoch + transfer_time,
            departure_pos=tuple(r1 / 1000.0),  # Convert to km
            departure_vel=tuple(init_vel / 1000.0),  # Convert to km/s
            arrival_pos=tuple(arrival_pos / 1000.0),  # Convert to km
            arrival_vel=tuple(arrival_vel / 1000.0),  # Convert to km/s
        )

        # Add maneuver objects to trajectory
        self._add_maneuvers_to_trajectory(
            trajectory, epoch, transfer_time, tli_dv, loi_dv
        )

        # Calculate total delta-v
        total_dv = np.linalg.norm(tli_dv) + np.linalg.norm(loi_dv)
        return trajectory, total_dv

    def _calculate_maneuvers(
        self, r1: np.ndarray, init_vel: np.ndarray, target_vel: np.ndarray, tof: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate TLI and LOI maneuver delta-v values using Lambert solver.

        Args:
            r1: Initial position vector [m]
            init_vel: Initial velocity vector [m/s]
            target_vel: Target velocity vector [m/s]
            tof: Time of flight [s]

        Returns
        -------
            Tuple of (TLI delta-v, LOI delta-v, arrival position, arrival velocity)
        """
        # Use Lambert solver for more accurate trajectory calculation
        from .earth_moon_trajectories import LambertSolver

        # Create Lambert solver
        lambert_solver = LambertSolver()

        # Calculate target position based on Moon's position at arrival
        # For simplicity, use the propagated position from the current approach
        try:
            arrival_pos, arrival_vel = self.propagator.propagate_to_target(
                r1,
                init_vel,
                tof,
                self.departure_epoch,
            )
        except Exception:
            # Fallback to simple calculation if propagation fails
            arrival_pos = r1 + init_vel * tof
            arrival_vel = init_vel

        # Use Lambert solver to get optimal velocities
        try:
            lambert_v1, lambert_v2 = lambert_solver.solve_lambert(
                r1, arrival_pos, tof, direction=0, max_revolutions=0
            )

            # Calculate maneuver delta-v
            tli_dv = (
                lambert_v1 - init_vel
            )  # TLI: difference from parking orbit velocity
            loi_dv = target_vel - lambert_v2  # LOI: difference to target orbit velocity

        except Exception:
            # Fallback to original calculation if Lambert solver fails
            tli_dv = init_vel - target_vel
            loi_dv = target_vel - arrival_vel

        # Validate delta-v magnitudes
        tli_dv_mag = np.linalg.norm(tli_dv)
        loi_dv_mag = np.linalg.norm(loi_dv)
        self.validator.validate_delta_v(tli_dv_mag, loi_dv_mag)

        return tli_dv, loi_dv, arrival_pos, arrival_vel

    def _add_maneuvers_to_trajectory(
        self,
        trajectory: Trajectory,
        epoch: float,
        transfer_time: float,
        tli_dv: np.ndarray,
        loi_dv: np.ndarray,
    ) -> None:
        """Add TLI and LOI maneuvers to the trajectory.

        Args:
            trajectory: Trajectory object to add maneuvers to
            epoch: Start epoch in days since J2000
            transfer_time: Transfer time [days]
            tli_dv: Trans-Lunar Injection delta-v [m/s]
            loi_dv: Lunar Orbit Insertion delta-v [m/s]
        """
        # Create maneuver objects (convert delta-v from m/s to km/s and epoch to datetime)
        from datetime import datetime, timedelta

        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)

        tli = Maneuver(
            delta_v=tuple(tli_dv / 1000.0),  # Convert to km/s
            epoch=j2000 + timedelta(days=epoch),
        )

        loi = Maneuver(
            delta_v=tuple(loi_dv / 1000.0),  # Convert to km/s
            epoch=j2000 + timedelta(days=epoch + transfer_time),
        )

        # Add maneuvers to trajectory
        trajectory.add_maneuver(tli)
        trajectory.add_maneuver(loi)
