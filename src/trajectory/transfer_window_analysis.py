"""Transfer Window Analysis Module for Task 3 completion.

This module provides comprehensive Earth-Moon trajectory window analysis,
including multiple transfer opportunities, launch window optimization,
and trajectory performance metrics.
"""

import logging
from datetime import datetime, timedelta

import numpy as np

from .celestial_bodies import CelestialBody
from .constants import PhysicalConstants as PC
from .lunar_transfer import LunarTransfer
from .models import Trajectory

# Configure logging
logger = logging.getLogger(__name__)


class TransferWindow:
    """Represents a transfer window opportunity."""

    def __init__(self,
                 departure_date: datetime,
                 arrival_date: datetime,
                 total_dv: float,
                 c3_energy: float,
                 trajectory: Trajectory) -> None:
        self.departure_date = departure_date
        self.arrival_date = arrival_date
        self.total_dv = total_dv  # m/s
        self.c3_energy = c3_energy  # m²/s²
        self.trajectory = trajectory
        self.transfer_time = (arrival_date - departure_date).total_seconds() / 86400  # days

    def __str__(self) -> str:
        return (f"TransferWindow(departure={self.departure_date.strftime('%Y-%m-%d')}, "
                f"transfer_time={self.transfer_time:.1f}d, dv={self.total_dv:.0f}m/s)")


class TrajectoryWindowAnalyzer:
    """Analyzes Earth-Moon transfer windows for Task 3 implementation."""

    def __init__(self,
                 min_earth_alt: float = 200,  # km
                 max_earth_alt: float = 1000,  # km
                 min_moon_alt: float = 50,    # km
                 max_moon_alt: float = 500) -> None:  # km
        """Initialize the trajectory window analyzer.

        Args:
            min_earth_alt: Minimum Earth parking orbit altitude [km]
            max_earth_alt: Maximum Earth parking orbit altitude [km]
            min_moon_alt: Minimum lunar orbit altitude [km]
            max_moon_alt: Maximum lunar orbit altitude [km]
        """
        self.min_earth_alt = min_earth_alt
        self.max_earth_alt = max_earth_alt
        self.min_moon_alt = min_moon_alt
        self.max_moon_alt = max_moon_alt

        self.celestial = CelestialBody()
        self.lunar_transfer = LunarTransfer(
            min_earth_alt, max_earth_alt, min_moon_alt, max_moon_alt,
        )

        logger.info(f"Initialized TrajectoryWindowAnalyzer with altitude ranges: "
                   f"Earth [{min_earth_alt}-{max_earth_alt}] km, "
                   f"Moon [{min_moon_alt}-{max_moon_alt}] km")

    def find_transfer_windows(self,
                            start_date: datetime,
                            end_date: datetime,
                            earth_orbit_alt: float = 300.0,  # km
                            moon_orbit_alt: float = 100.0,   # km
                            min_transfer_time: float = 3.0,  # days
                            max_transfer_time: float = 7.0,  # days
                            time_step: float = 1.0) -> list[TransferWindow]:
        """Find optimal transfer windows in a given time period.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            earth_orbit_alt: Earth parking orbit altitude [km]
            moon_orbit_alt: Target lunar orbit altitude [km]
            min_transfer_time: Minimum transfer time [days]
            max_transfer_time: Maximum transfer time [days]
            time_step: Time step for analysis [days]

        Returns
        -------
            List of viable transfer windows sorted by total delta-v
        """
        logger.info(f"Analyzing transfer windows from {start_date} to {end_date}")

        windows = []
        current_date = start_date

        while current_date <= end_date:
            # Convert to PyKEP epoch (days since J2000)
            epoch = self._datetime_to_pykep_epoch(current_date)

            # Analyze different transfer times for this departure date
            for transfer_time in np.arange(min_transfer_time, max_transfer_time + 0.5, 0.5):
                try:
                    # Generate trajectory for this window
                    trajectory, total_dv = self.lunar_transfer.generate_transfer(
                        epoch=epoch,
                        earth_orbit_alt=earth_orbit_alt,
                        moon_orbit_alt=moon_orbit_alt,
                        transfer_time=transfer_time,
                        max_revolutions=0,
                    )

                    # Calculate C3 energy (characteristic energy)
                    c3_energy = self._calculate_c3_energy(earth_orbit_alt, total_dv)

                    # Create transfer window
                    arrival_date = current_date + timedelta(days=transfer_time)
                    window = TransferWindow(
                        departure_date=current_date,
                        arrival_date=arrival_date,
                        total_dv=total_dv,
                        c3_energy=c3_energy,
                        trajectory=trajectory,
                    )

                    windows.append(window)
                    logger.debug(f"Found window: {window}")

                except Exception as e:
                    logger.debug(f"Failed to generate trajectory for {current_date} + {transfer_time}d: {e}")
                    continue

            # Move to next analysis date
            current_date += timedelta(days=time_step)

        # Sort by total delta-v (most efficient first)
        windows.sort(key=lambda w: w.total_dv)

        logger.info(f"Found {len(windows)} viable transfer windows")
        return windows

    def optimize_launch_window(self,
                              target_date: datetime,
                              window_days: int = 30,
                              earth_orbit_alt: float = 300.0,
                              moon_orbit_alt: float = 100.0) -> TransferWindow:
        """Optimize launch window around a target date.

        Args:
            target_date: Preferred launch date
            window_days: Days before/after target to analyze
            earth_orbit_alt: Earth parking orbit altitude [km]
            moon_orbit_alt: Target lunar orbit altitude [km]

        Returns
        -------
            Best transfer window within the specified period

        Raises
        ------
            ValueError: If no viable windows found
        """
        start_date = target_date - timedelta(days=window_days // 2)
        end_date = target_date + timedelta(days=window_days // 2)

        windows = self.find_transfer_windows(
            start_date=start_date,
            end_date=end_date,
            earth_orbit_alt=earth_orbit_alt,
            moon_orbit_alt=moon_orbit_alt,
            time_step=0.5,  # Higher resolution for optimization
        )

        if not windows:
            msg = f"No viable transfer windows found around {target_date}"
            raise ValueError(msg)

        best_window = windows[0]  # Already sorted by delta-v
        logger.info(f"Optimized launch window: {best_window}")

        return best_window

    def analyze_trajectory_sensitivity(self,
                                     window: TransferWindow,
                                     altitude_variations: list[float] | None = None,
                                     time_variations: list[float] | None = None) -> dict[str, list[float]]:
        """Analyze trajectory sensitivity to parameter variations.

        Args:
            window: Reference transfer window
            altitude_variations: Earth orbit altitude variations [km]
            time_variations: Transfer time variations [days]

        Returns
        -------
            Dictionary with sensitivity analysis results
        """
        if altitude_variations is None:
            altitude_variations = [-50, -25, 0, 25, 50]  # km
        if time_variations is None:
            time_variations = [-1.0, -0.5, 0, 0.5, 1.0]  # days

        base_epoch = self._datetime_to_pykep_epoch(window.departure_date)
        results = {
            "altitude_variations": altitude_variations,
            "altitude_dv_changes": [],
            "time_variations": time_variations,
            "time_dv_changes": [],
        }

        # Analyze altitude sensitivity
        for alt_delta in altitude_variations:
            try:
                _, dv = self.lunar_transfer.generate_transfer(
                    epoch=base_epoch,
                    earth_orbit_alt=300.0 + alt_delta,  # Base + variation
                    moon_orbit_alt=100.0,
                    transfer_time=window.transfer_time,
                    max_revolutions=0,
                )
                dv_change = dv - window.total_dv
                results["altitude_dv_changes"].append(dv_change)
            except:
                results["altitude_dv_changes"].append(float("inf"))

        # Analyze time sensitivity
        for time_delta in time_variations:
            try:
                _, dv = self.lunar_transfer.generate_transfer(
                    epoch=base_epoch,
                    earth_orbit_alt=300.0,
                    moon_orbit_alt=100.0,
                    transfer_time=window.transfer_time + time_delta,
                    max_revolutions=0,
                )
                dv_change = dv - window.total_dv
                results["time_dv_changes"].append(dv_change)
            except:
                results["time_dv_changes"].append(float("inf"))

        return results

    def _datetime_to_pykep_epoch(self, dt: datetime) -> float:
        """Convert datetime to PyKEP epoch (days since J2000)."""
        j2000 = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
        delta = dt - j2000
        return delta.total_seconds() / 86400.0

    def _calculate_c3_energy(self, earth_orbit_alt: float, total_dv: float) -> float:
        """Calculate characteristic energy (C3) for the transfer.

        Args:
            earth_orbit_alt: Earth parking orbit altitude [km]
            total_dv: Total delta-v for transfer [m/s]

        Returns
        -------
            C3 characteristic energy [m²/s²]
        """
        r_park = (PC.EARTH_RADIUS + earth_orbit_alt * 1000)  # [m]
        np.sqrt(PC.EARTH_MU / r_park)  # Circular velocity [m/s]
        v_infinity = total_dv  # Approximation for C3 calculation

        # C3 = v_infinity^2
        return v_infinity**2


def generate_multiple_transfer_options(start_date: datetime,
                                     end_date: datetime,
                                     max_options: int = 10) -> list[TransferWindow]:
    """Generate multiple transfer options for Task 3 completion.

    Args:
        start_date: Start of analysis period
        end_date: End of analysis period
        max_options: Maximum number of options to return

    Returns
    -------
        List of best transfer options
    """
    analyzer = TrajectoryWindowAnalyzer()

    windows = analyzer.find_transfer_windows(
        start_date=start_date,
        end_date=end_date,
        time_step=2.0,  # Faster analysis for multiple options
    )

    # Return top options
    return windows[:max_options]


def analyze_launch_opportunities(target_year: int = 2025) -> dict[str, list[TransferWindow]]:
    """Analyze launch opportunities for a given year.

    Args:
        target_year: Year to analyze

    Returns
    -------
        Dictionary with monthly launch opportunities
    """
    analyzer = TrajectoryWindowAnalyzer()
    monthly_opportunities = {}

    for month in range(1, 13):
        start_date = datetime(target_year, month, 1)
        # Get last day of month
        if month == 12:
            end_date = datetime(target_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(target_year, month + 1, 1) - timedelta(days=1)

        opportunities = analyzer.find_transfer_windows(
            start_date=start_date,
            end_date=end_date,
            time_step=5.0,  # Weekly analysis
        )

        month_name = start_date.strftime("%B")
        monthly_opportunities[month_name] = opportunities[:5]  # Top 5 per month

    return monthly_opportunities
