"""Earth-Moon trajectory generation functions for Task 3.2 completion.

This module implements comprehensive Earth-Moon trajectory generation including
Lambert solvers, patched conics approximation, and optimal timing calculations.
"""

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pykep as pk

from .celestial_bodies import CelestialBody
from .constants import PhysicalConstants as PC
from .lunar_transfer import LunarTransfer
from .models import OrbitState, Trajectory

# Configure logging
logger = logging.getLogger(__name__)


class LambertSolver:
    """Lambert problem solver for Earth-Moon trajectory generation.

    This class provides Lambert problem solutions for two-body trajectories
    between Earth and Moon, supporting the core trajectory generation needs.
    """

    def __init__(self, central_body_mu: float = PC.EARTH_MU) -> None:
        """Initialize Lambert solver.

        Args:
            central_body_mu: Gravitational parameter of central body [m³/s²]
        """
        self.mu = central_body_mu
        self.max_iterations = 100
        self.tolerance = 1e-12

        logger.info(f"Initialized Lambert solver with μ = {self.mu:.3e} m³/s²")

    def solve_lambert(self,
                     r1: np.ndarray,
                     r2: np.ndarray,
                     time_of_flight: float,
                     direction: int = 0,
                     max_revolutions: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Solve Lambert problem for given position vectors and time.

        Args:
            r1: Initial position vector [m]
            r2: Final position vector [m]
            time_of_flight: Time of flight [s]
            direction: Transfer direction (0=auto, 1=prograde, -1=retrograde)
            max_revolutions: Maximum number of revolutions

        Returns
        -------
            Tuple of (initial_velocity, final_velocity) [m/s]

        Raises
        ------
            ValueError: If Lambert solution fails to converge
        """
        try:
            # Use PyKEP's Lambert solver
            lambert = pk.lambert_problem(
                r1.tolist(), r2.tolist(), time_of_flight, self.mu,
                cw=direction > 0, max_revs=max_revolutions,
            )

            if not lambert.get_v1():
                msg = "Lambert solver failed to find solution"
                raise ValueError(msg)

            v1 = np.array(lambert.get_v1()[0])  # Initial velocity
            v2 = np.array(lambert.get_v2()[0])  # Final velocity

            logger.debug(f"Lambert solution: v1 = {np.linalg.norm(v1):.1f} m/s, "
                        f"v2 = {np.linalg.norm(v2):.1f} m/s")

            return v1, v2

        except Exception as e:
            msg = f"Lambert problem solution failed: {e!s}"
            raise ValueError(msg) from e

    def solve_multiple_revolution(self,
                                r1: np.ndarray,
                                r2: np.ndarray,
                                time_of_flight: float,
                                max_revs: int = 2) -> list[tuple[np.ndarray, np.ndarray]]:
        """Solve Lambert problem for multiple revolution cases.

        Args:
            r1: Initial position vector [m]
            r2: Final position vector [m]
            time_of_flight: Time of flight [s]
            max_revs: Maximum revolutions to consider

        Returns
        -------
            List of (v1, v2) solutions for different revolution numbers
        """
        solutions = []

        for revs in range(max_revs + 1):
            try:
                v1, v2 = self.solve_lambert(r1, r2, time_of_flight,
                                          max_revolutions=revs)
                solutions.append((v1, v2))
                logger.debug(f"Found solution for {revs} revolutions")
            except ValueError:
                logger.debug(f"No solution found for {revs} revolutions")
                continue

        return solutions

    def calculate_transfer_deltav(self,
                                r1: np.ndarray,
                                v1_current: np.ndarray,
                                r2: np.ndarray,
                                v2_target: np.ndarray,
                                time_of_flight: float) -> tuple[float, np.ndarray, np.ndarray]:
        """Calculate delta-v required for Lambert transfer.

        Args:
            r1: Initial position [m]
            v1_current: Current velocity at r1 [m/s]
            r2: Final position [m]
            v2_target: Target velocity at r2 [m/s]
            time_of_flight: Time of flight [s]

        Returns
        -------
            Tuple of (total_deltav, deltav1, deltav2) [m/s]
        """
        # Solve Lambert problem
        v1_transfer, v2_transfer = self.solve_lambert(r1, r2, time_of_flight)

        # Calculate required delta-v maneuvers
        deltav1 = v1_transfer - v1_current  # Departure maneuver
        deltav2 = v2_target - v2_transfer   # Arrival maneuver

        total_deltav = np.linalg.norm(deltav1) + np.linalg.norm(deltav2)

        return total_deltav, deltav1, deltav2


class PatchedConicsApproximation:
    """Patched conics approximation for Earth-Moon trajectories.

    This class implements the patched conics method for approximating
    trajectories in the Earth-Moon system, providing faster calculations
    than full n-body integration.
    """

    def __init__(self) -> None:
        """Initialize patched conics approximation."""
        self.earth_soi = 9.24e8  # Earth sphere of influence [m]
        self.moon_soi = PC.MOON_SOI  # Moon sphere of influence [m]
        self.celestial = CelestialBody()

        logger.info("Initialized patched conics approximation")

    def calculate_trajectory(self,
                           earth_departure: OrbitState,
                           moon_arrival: OrbitState,
                           transfer_time: float) -> dict[str, any]:
        """Calculate trajectory using patched conics approximation.

        Args:
            earth_departure: Departure orbit state around Earth
            moon_arrival: Arrival orbit state around Moon
            transfer_time: Transfer time [s]

        Returns
        -------
            Dictionary with trajectory components and characteristics
        """
        logger.info(f"Calculating patched conics trajectory for {transfer_time/86400:.1f} days")

        # Phase 1: Earth escape trajectory
        earth_escape = self._calculate_earth_escape(earth_departure)

        # Phase 2: Heliocentric transfer (simplified to Earth-Moon system)
        transfer_trajectory = self._calculate_earth_moon_transfer(
            earth_escape, moon_arrival, transfer_time,
        )

        # Phase 3: Moon capture trajectory
        moon_capture = self._calculate_moon_capture(moon_arrival, transfer_trajectory)

        # Combine phases
        trajectory_data = {
            "earth_escape": earth_escape,
            "transfer": transfer_trajectory,
            "moon_capture": moon_capture,
            "total_deltav": (earth_escape["deltav"] +
                           transfer_trajectory["deltav"] +
                           moon_capture["deltav"]),
            "transfer_time": transfer_time,
            "method": "patched_conics",
        }

        logger.info(f"Patched conics trajectory complete: "
                   f"Total ΔV = {trajectory_data['total_deltav']:.1f} m/s")

        return trajectory_data

    def _calculate_earth_escape(self, departure_state: OrbitState) -> dict[str, any]:
        """Calculate Earth escape phase.

        Args:
            departure_state: Departure orbit around Earth

        Returns
        -------
            Earth escape trajectory data
        """
        # Calculate escape velocity from parking orbit
        r_park = np.linalg.norm(departure_state.position) * 1000  # Convert km to m
        v_park = np.sqrt(PC.EARTH_MU / r_park)  # Circular velocity
        v_escape = np.sqrt(2 * PC.EARTH_MU / r_park)  # Escape velocity

        # Delta-v for escape
        deltav_escape = v_escape - v_park

        # Hyperbolic excess velocity (simplified)
        v_infinity = 1000.0  # Typical lunar transfer velocity [m/s]

        return {
            "parking_radius": r_park,
            "parking_velocity": v_park,
            "escape_velocity": v_escape,
            "deltav": deltav_escape,
            "v_infinity": v_infinity,
            "phase": "earth_escape",
        }

    def _calculate_earth_moon_transfer(self,
                                     earth_escape: dict[str, any],
                                     moon_arrival: OrbitState,
                                     transfer_time: float) -> dict[str, any]:
        """Calculate Earth-Moon transfer phase.

        Args:
            earth_escape: Earth escape trajectory data
            moon_arrival: Moon arrival orbit state
            transfer_time: Transfer time [s]

        Returns
        -------
            Transfer trajectory data
        """
        # Simplified transfer calculation
        # In reality, this would use more sophisticated orbital mechanics

        # Average Earth-Moon distance
        earth_moon_distance = 3.844e8  # m

        # Estimate transfer velocity
        transfer_velocity = earth_moon_distance / transfer_time

        # Simplified delta-v for course corrections
        deltav_transfer = 100.0  # Typical mid-course correction [m/s]

        return {
            "transfer_distance": earth_moon_distance,
            "transfer_velocity": transfer_velocity,
            "transfer_time": transfer_time,
            "deltav": deltav_transfer,
            "phase": "earth_moon_transfer",
        }

    def _calculate_moon_capture(self,
                              arrival_state: OrbitState,
                              transfer_data: dict[str, any]) -> dict[str, any]:
        """Calculate Moon capture phase.

        Args:
            arrival_state: Arrival orbit around Moon
            transfer_data: Transfer trajectory data

        Returns
        -------
            Moon capture trajectory data
        """
        # Calculate capture requirements
        r_moon = np.linalg.norm(arrival_state.position) * 1000  # Convert km to m
        v_moon_orbit = np.sqrt(PC.MOON_MU / r_moon)  # Circular velocity around Moon

        # Incoming velocity (simplified)
        v_approach = transfer_data["transfer_velocity"]

        # Delta-v for capture (simplified)
        deltav_capture = abs(v_approach - v_moon_orbit)

        return {
            "lunar_radius": r_moon,
            "lunar_velocity": v_moon_orbit,
            "approach_velocity": v_approach,
            "deltav": deltav_capture,
            "phase": "moon_capture",
        }


class OptimalTimingCalculator:
    """Calculator for optimal departure and arrival timing.

    This class provides methods to calculate optimal launch windows
    and arrival times for Earth-Moon trajectories.
    """

    def __init__(self) -> None:
        """Initialize optimal timing calculator."""
        self.celestial = CelestialBody()
        self.lunar_transfer = LunarTransfer()

        logger.info("Initialized optimal timing calculator")

    def find_optimal_departure_time(self,
                                   start_epoch: float,
                                   search_days: int = 30,
                                   earth_orbit_alt: float = 300.0,
                                   moon_orbit_alt: float = 100.0) -> dict[str, any]:
        """Find optimal departure time within search period.

        Args:
            start_epoch: Search start epoch [days since J2000]
            search_days: Number of days to search
            earth_orbit_alt: Earth parking orbit altitude [km]
            moon_orbit_alt: Moon orbit altitude [km]

        Returns
        -------
            Dictionary with optimal timing information
        """
        logger.info(f"Searching for optimal departure time over {search_days} days")

        best_deltav = float("inf")
        best_epoch = start_epoch
        best_transfer_time = 4.0  # Default transfer time [days]

        # Search over time windows
        for day_offset in range(0, search_days, 2):  # Every 2 days
            current_epoch = start_epoch + day_offset

            # Try different transfer times
            for transfer_time in [3.0, 4.0, 5.0, 6.0, 7.0]:
                try:
                    trajectory, total_dv = self.lunar_transfer.generate_transfer(
                        epoch=current_epoch,
                        earth_orbit_alt=earth_orbit_alt,
                        moon_orbit_alt=moon_orbit_alt,
                        transfer_time=transfer_time,
                        max_revolutions=0,
                    )

                    if total_dv < best_deltav:
                        best_deltav = total_dv
                        best_epoch = current_epoch
                        best_transfer_time = transfer_time

                except Exception as e:
                    logger.debug(f"Failed trajectory at epoch {current_epoch}: {e}")
                    continue

        # Convert epoch to date
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        optimal_date = j2000 + timedelta(days=best_epoch)

        result = {
            "optimal_epoch": best_epoch,
            "optimal_date": optimal_date,
            "optimal_transfer_time": best_transfer_time,
            "optimal_deltav": best_deltav,
            "search_period_days": search_days,
            "earth_orbit_alt": earth_orbit_alt,
            "moon_orbit_alt": moon_orbit_alt,
        }

        logger.info(f"Optimal departure: {optimal_date.strftime('%Y-%m-%d')}, "
                   f"Transfer time: {best_transfer_time:.1f} days, "
                   f"ΔV: {best_deltav:.0f} m/s")

        return result

    def calculate_launch_windows(self,
                               year: int = 2025,
                               month: int = 6,
                               num_windows: int = 5) -> list[dict[str, any]]:
        """Calculate multiple launch windows for a given month.

        Args:
            year: Launch year
            month: Launch month
            num_windows: Number of windows to find

        Returns
        -------
            List of launch window opportunities
        """
        # Calculate search period
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        # Convert to PyKEP epochs
        j2000 = datetime(2000, 1, 1, 12, 0, 0)
        start_epoch = (start_date - j2000).total_seconds() / 86400.0
        search_days = (end_date - start_date).days

        logger.info(f"Calculating launch windows for {start_date.strftime('%B %Y')}")

        windows = []
        search_step = max(1, search_days // (num_windows * 2))  # Distribute search

        for i in range(num_windows):
            window_start = start_epoch + i * search_step
            window_result = self.find_optimal_departure_time(
                start_epoch=window_start,
                search_days=min(search_step + 5, search_days),
                earth_orbit_alt=300.0,
                moon_orbit_alt=100.0,
            )
            windows.append(window_result)

        # Sort by delta-v (best first)
        windows.sort(key=lambda w: w["optimal_deltav"])

        logger.info(f"Found {len(windows)} launch windows")
        return windows

    def analyze_timing_sensitivity(self,
                                 optimal_epoch: float,
                                 time_variations: list[float] | None = None) -> dict[str, any]:
        """Analyze sensitivity to timing variations.

        Args:
            optimal_epoch: Optimal departure epoch [days since J2000]
            time_variations: Time variations to test [days]

        Returns
        -------
            Timing sensitivity analysis
        """
        if time_variations is None:
            time_variations = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

        logger.info(f"Analyzing timing sensitivity around epoch {optimal_epoch}")

        sensitivity_data = {
            "reference_epoch": optimal_epoch,
            "time_variations": time_variations,
            "deltav_variations": [],
            "sensitivity_slope": 0.0,
        }

        reference_deltav = None

        for time_delta in time_variations:
            test_epoch = optimal_epoch + time_delta

            try:
                trajectory, total_dv = self.lunar_transfer.generate_transfer(
                    epoch=test_epoch,
                    earth_orbit_alt=300.0,
                    moon_orbit_alt=100.0,
                    transfer_time=4.0,
                    max_revolutions=0,
                )

                if time_delta == 0.0:
                    reference_deltav = total_dv

                deltav_change = total_dv - (reference_deltav or total_dv)
                sensitivity_data["deltav_variations"].append(deltav_change)

            except Exception:
                sensitivity_data["deltav_variations"].append(float("inf"))

        # Calculate sensitivity slope (deltav change per day)
        valid_points = [(t, dv) for t, dv in zip(time_variations, sensitivity_data["deltav_variations"], strict=False)
                       if dv != float("inf")]

        if len(valid_points) > 1:
            times, dv_changes = zip(*valid_points, strict=False)
            sensitivity_data["sensitivity_slope"] = np.polyfit(times, dv_changes, 1)[0]

        logger.info(f"Timing sensitivity: {sensitivity_data['sensitivity_slope']:.1f} m/s per day")
        return sensitivity_data


def generate_earth_moon_trajectory(departure_epoch: float,
                                 earth_orbit_alt: float = 300.0,
                                 moon_orbit_alt: float = 100.0,
                                 transfer_time: float = 4.0,
                                 method: str = "lambert") -> tuple[Trajectory, float]:
    """Convenience function for Earth-Moon trajectory generation.

    Args:
        departure_epoch: Departure epoch [days since J2000]
        earth_orbit_alt: Earth orbit altitude [km]
        moon_orbit_alt: Moon orbit altitude [km]
        transfer_time: Transfer time [days]
        method: Generation method ('lambert', 'patched_conics')

    Returns
    -------
        Tuple of (trajectory, total_deltav)
    """
    if method == "lambert":
        lunar_transfer = LunarTransfer()
        return lunar_transfer.generate_transfer(
            epoch=departure_epoch,
            earth_orbit_alt=earth_orbit_alt,
            moon_orbit_alt=moon_orbit_alt,
            transfer_time=transfer_time,
            max_revolutions=0,
        )
    if method == "patched_conics":
        # Create orbit states from position and velocity vectors
        from datetime import datetime, timedelta
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
        departure_time = j2000 + timedelta(days=departure_epoch)
        arrival_time = j2000 + timedelta(days=departure_epoch + transfer_time)

        earth_state = OrbitState.from_state_vectors(
            position=((PC.EARTH_RADIUS + earth_orbit_alt * 1000) / 1000, 0, 0),  # Convert to km
            velocity=(0, np.sqrt(PC.EARTH_MU / (PC.EARTH_RADIUS + earth_orbit_alt * 1000)) / 1000, 0),  # Convert to km/s
            epoch=departure_time,
            mu=PC.EARTH_MU,
        )

        moon_state = OrbitState.from_state_vectors(
            position=((PC.MOON_RADIUS + moon_orbit_alt * 1000) / 1000, 0, 0),  # Convert to km
            velocity=(0, np.sqrt(PC.MOON_MU / (PC.MOON_RADIUS + moon_orbit_alt * 1000)) / 1000, 0),  # Convert to km/s
            epoch=arrival_time,
            mu=PC.MOON_MU,
        )

        # Use patched conics
        patched_conics = PatchedConicsApproximation()
        trajectory_data = patched_conics.calculate_trajectory(
            earth_state, moon_state, transfer_time * 86400,
        )

        # Create trajectory object (simplified)
        from .lunar_transfer import LunarTrajectory
        trajectory = LunarTrajectory(
            departure_epoch=departure_epoch,
            arrival_epoch=departure_epoch + transfer_time,
            departure_pos=earth_state.position,
            departure_vel=earth_state.velocity(mu=PC.EARTH_MU),
            arrival_pos=moon_state.position,
            arrival_vel=moon_state.velocity(mu=PC.MOON_MU),
        )

        return trajectory, trajectory_data["total_deltav"]
    msg = f"Unknown trajectory generation method: {method}"
    raise ValueError(msg)


def find_optimal_launch_window(target_date: datetime,
                             window_days: int = 30,
                             earth_orbit_alt: float = 300.0,
                             moon_orbit_alt: float = 100.0) -> dict[str, any]:
    """Find optimal launch window around target date.

    Args:
        target_date: Target launch date
        window_days: Search window size [days]
        earth_orbit_alt: Earth orbit altitude [km]
        moon_orbit_alt: Moon orbit altitude [km]

    Returns
    -------
        Optimal launch window information
    """
    timing_calc = OptimalTimingCalculator()

    # Convert to epoch
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    target_epoch = (target_date - j2000).total_seconds() / 86400.0

    return timing_calc.find_optimal_departure_time(
        start_epoch=target_epoch - window_days // 2,
        search_days=window_days,
        earth_orbit_alt=earth_orbit_alt,
        moon_orbit_alt=moon_orbit_alt,
    )
