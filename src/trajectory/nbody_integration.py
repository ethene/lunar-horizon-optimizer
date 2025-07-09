"""N-body dynamics integration and trajectory I/O for Task 3.3 completion.

This module implements numerical integrators, n-body propagation, and
trajectory serialization/comparison utilities for the Earth-Moon system.
"""

import numpy as np
from typing import Any
from collections.abc import Callable
from scipy.integrate import solve_ivp
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

from .constants import PhysicalConstants as PC
from .celestial_bodies import CelestialBody
from .models import Trajectory

# Configure logging
logger = logging.getLogger(__name__)


class NumericalIntegrator:
    """Numerical integration methods for trajectory propagation.
    
    This class provides various integration schemes for orbital mechanics
    calculations, including Runge-Kutta and specialized orbital integrators.
    """

    def __init__(self,
                 method: str = "DOP853",
                 rtol: float = 1e-12,
                 atol: float = 1e-15):
        """Initialize numerical integrator.
        
        Args:
            method: Integration method ('RK45', 'DOP853', 'Radau', 'RK4')
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol

        # Available integration methods
        self.scipy_methods = ["RK45", "DOP853", "Radau", "RK23", "BDF", "LSODA"]
        self.custom_methods = ["RK4", "Verlet", "LeapFrog"]

        logger.info(f"Initialized numerical integrator: {method}, "
                   f"rtol={rtol:.1e}, atol={atol:.1e}")

    def integrate_trajectory(self,
                           dynamics_function: Callable,
                           initial_state: np.ndarray,
                           time_span: tuple[float, float],
                           num_points: int = 1000,
                           **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Integrate trajectory using specified method.
        
        Args:
            dynamics_function: Function computing state derivatives
            initial_state: Initial state vector [position, velocity]
            time_span: (t_start, t_end) integration interval [s]
            num_points: Number of output points
            **kwargs: Additional arguments for dynamics function
            
        Returns
        -------
            Tuple of (time_array, state_history)
        """
        logger.debug(f"Integrating trajectory: {time_span[0]:.1f}s to {time_span[1]:.1f}s")

        # Time evaluation points
        t_eval = np.linspace(time_span[0], time_span[1], num_points)

        if self.method in self.scipy_methods:
            return self._integrate_scipy(
                dynamics_function, initial_state, time_span, t_eval, **kwargs
            )
        if self.method in self.custom_methods:
            return self._integrate_custom(
                dynamics_function, initial_state, time_span, num_points, **kwargs
            )
        raise ValueError(f"Unknown integration method: {self.method}")

    def _integrate_scipy(self,
                        dynamics_function: Callable,
                        initial_state: np.ndarray,
                        time_span: tuple[float, float],
                        t_eval: np.ndarray,
                        **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Integrate using SciPy solve_ivp."""
        solution = solve_ivp(
            fun=lambda t, y: dynamics_function(t, y, **kwargs),
            t_span=time_span,
            y0=initial_state,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True
        )

        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")

        return solution.t, solution.y

    def _integrate_custom(self,
                         dynamics_function: Callable,
                         initial_state: np.ndarray,
                         time_span: tuple[float, float],
                         num_points: int,
                         **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Integrate using custom methods."""
        if self.method == "RK4":
            return self._integrate_rk4(
                dynamics_function, initial_state, time_span, num_points, **kwargs
            )
        if self.method == "Verlet":
            return self._integrate_verlet(
                dynamics_function, initial_state, time_span, num_points, **kwargs
            )
        raise NotImplementedError(f"Custom method {self.method} not implemented")

    def _integrate_rk4(self,
                      dynamics_function: Callable,
                      initial_state: np.ndarray,
                      time_span: tuple[float, float],
                      num_points: int,
                      **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Fourth-order Runge-Kutta integration."""
        t_start, t_end = time_span
        dt = (t_end - t_start) / (num_points - 1)

        # Initialize arrays
        times = np.linspace(t_start, t_end, num_points)
        states = np.zeros((len(initial_state), num_points))
        states[:, 0] = initial_state

        # RK4 integration loop
        for i in range(num_points - 1):
            t = times[i]
            y = states[:, i]

            k1 = dt * dynamics_function(t, y, **kwargs)
            k2 = dt * dynamics_function(t + dt/2, y + k1/2, **kwargs)
            k3 = dt * dynamics_function(t + dt/2, y + k2/2, **kwargs)
            k4 = dt * dynamics_function(t + dt, y + k3, **kwargs)

            states[:, i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

        return times, states

    def _integrate_verlet(self,
                         dynamics_function: Callable,
                         initial_state: np.ndarray,
                         time_span: tuple[float, float],
                         num_points: int,
                         **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Verlet integration (for position/velocity problems)."""
        # Note: This is a simplified Verlet implementation
        # For orbital mechanics, consider specialized symplectic integrators

        t_start, t_end = time_span
        dt = (t_end - t_start) / (num_points - 1)

        # Split state into position and velocity
        n_dim = len(initial_state) // 2
        pos = initial_state[:n_dim]
        vel = initial_state[n_dim:]

        # Initialize arrays
        times = np.linspace(t_start, t_end, num_points)
        positions = np.zeros((n_dim, num_points))
        velocities = np.zeros((n_dim, num_points))

        positions[:, 0] = pos
        velocities[:, 0] = vel

        # Get initial acceleration
        state = np.concatenate([pos, vel])
        derivatives = dynamics_function(t_start, state, **kwargs)
        acc = derivatives[n_dim:]

        # Verlet integration loop
        for i in range(num_points - 1):
            t = times[i]

            # Update position
            positions[:, i+1] = positions[:, i] + velocities[:, i] * dt + 0.5 * acc * dt**2

            # Calculate new acceleration
            state = np.concatenate([positions[:, i+1], velocities[:, i]])
            derivatives = dynamics_function(t + dt, state, **kwargs)
            new_acc = derivatives[n_dim:]

            # Update velocity
            velocities[:, i+1] = velocities[:, i] + 0.5 * (acc + new_acc) * dt

            acc = new_acc

        # Combine position and velocity
        states = np.vstack([positions, velocities])

        return times, states


class EarthMoonNBodyPropagator:
    """Complete Earth-Moon n-body propagator with multiple body effects.
    
    This class extends the basic n-body propagator with specific Earth-Moon
    system dynamics, including perturbations from Sun and other effects.
    """

    def __init__(self,
                 include_sun: bool = True,
                 include_perturbations: bool = False,
                 integrator_method: str = "DOP853"):
        """Initialize Earth-Moon n-body propagator.
        
        Args:
            include_sun: Include solar gravitational effects
            include_perturbations: Include additional perturbations
            integrator_method: Numerical integration method
        """
        self.include_sun = include_sun
        self.include_perturbations = include_perturbations
        self.integrator = NumericalIntegrator(method=integrator_method)
        self.celestial = CelestialBody()

        # Gravitational parameters
        self.mu = {
            "earth": PC.EARTH_MU,
            "moon": PC.MOON_MU,
            "sun": PC.SUN_MU
        }

        logger.info(f"Initialized Earth-Moon n-body propagator "
                   f"(Sun: {include_sun}, Perturbations: {include_perturbations})")

    def propagate_spacecraft(self,
                           initial_position: np.ndarray,
                           initial_velocity: np.ndarray,
                           reference_epoch: float,
                           propagation_time: float,
                           num_points: int = 1000) -> dict[str, np.ndarray]:
        """Propagate spacecraft trajectory in Earth-Moon system.
        
        Args:
            initial_position: Initial position in Earth-centered frame [m]
            initial_velocity: Initial velocity in Earth-centered frame [m/s]
            reference_epoch: Reference epoch [days since J2000]
            propagation_time: Propagation time [s]
            num_points: Number of output points
            
        Returns
        -------
            Dictionary with propagation results
        """
        logger.info(f"Propagating spacecraft for {propagation_time/86400:.1f} days")

        # Initial state vector
        initial_state = np.concatenate([initial_position, initial_velocity])

        # Time span
        time_span = (0.0, propagation_time)

        # Propagate trajectory
        times, states = self.integrator.integrate_trajectory(
            dynamics_function=self._nbody_dynamics,
            initial_state=initial_state,
            time_span=time_span,
            num_points=num_points,
            reference_epoch=reference_epoch
        )

        # Extract position and velocity
        positions = states[:3, :]
        velocities = states[3:, :]

        # Calculate additional quantities
        distances_earth = np.linalg.norm(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=0)

        # Calculate energies
        kinetic_energy = 0.5 * speeds**2
        potential_energy = -self.mu["earth"] / distances_earth
        total_energy = kinetic_energy + potential_energy

        result = {
            "times": times,
            "positions": positions,
            "velocities": velocities,
            "distances_earth": distances_earth,
            "speeds": speeds,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
            "total_energy": total_energy,
            "propagation_time": propagation_time,
            "num_points": num_points
        }

        logger.info(f"Propagation completed: {len(times)} points generated")
        return result

    def _nbody_dynamics(self, t: float, state: np.ndarray, reference_epoch: float) -> np.ndarray:
        """Compute n-body dynamics for Earth-Moon-Sun system.
        
        Args:
            t: Current time [s]
            state: Current state [position, velocity] [m, m/s]
            reference_epoch: Reference epoch [days since J2000]
            
        Returns
        -------
            State derivatives [velocity, acceleration]
        """
        # Extract position and velocity
        r = state[:3]  # Position [m]
        v = state[3:]  # Velocity [m/s]

        # Initialize acceleration
        acceleration = np.zeros(3)

        # Current epoch
        current_epoch = reference_epoch + t / 86400.0

        # Earth gravity (always present in Earth-centered frame)
        r_earth = np.linalg.norm(r)
        if r_earth > 0:
            acceleration -= self.mu["earth"] * r / r_earth**3

        # Moon gravity
        try:
            moon_pos, moon_vel = self.celestial.get_moon_state_earth_centered(current_epoch)
            r_moon_sc = r - moon_pos  # Spacecraft position relative to Moon
            r_moon_sc_mag = np.linalg.norm(r_moon_sc)

            if r_moon_sc_mag > 0:
                # Direct moon gravity on spacecraft
                acceleration -= self.mu["moon"] * r_moon_sc / r_moon_sc_mag**3

                # Indirect effect (moon's effect on Earth)
                moon_pos_mag = np.linalg.norm(moon_pos)
                if moon_pos_mag > 0:
                    acceleration += self.mu["moon"] * moon_pos / moon_pos_mag**3

        except Exception as e:
            logger.warning(f"Failed to get Moon state at epoch {current_epoch}: {e}")

        # Sun gravity (if enabled)
        if self.include_sun:
            # Simplified sun position (would need full ephemeris for accuracy)
            sun_distance = 1.496e11  # 1 AU [m]
            sun_pos = np.array([sun_distance, 0.0, 0.0])  # Simplified position

            # Direct sun gravity on spacecraft
            r_sun_sc = r - sun_pos
            r_sun_sc_mag = np.linalg.norm(r_sun_sc)
            if r_sun_sc_mag > 0:
                acceleration -= self.mu["sun"] * r_sun_sc / r_sun_sc_mag**3

            # Indirect effect (sun's effect on Earth)
            sun_pos_mag = np.linalg.norm(sun_pos)
            if sun_pos_mag > 0:
                acceleration += self.mu["sun"] * sun_pos / sun_pos_mag**3

        # Additional perturbations (if enabled)
        if self.include_perturbations:
            acceleration += self._calculate_perturbations(r, v, current_epoch)

        # Return state derivatives
        return np.concatenate([v, acceleration])

    def _calculate_perturbations(self, position: np.ndarray, velocity: np.ndarray, epoch: float) -> np.ndarray:
        """Calculate additional perturbations (simplified).
        
        Args:
            position: Spacecraft position [m]
            velocity: Spacecraft velocity [m/s]
            epoch: Current epoch [days since J2000]
            
        Returns
        -------
            Perturbation acceleration [m/s²]
        """
        # Simplified perturbations
        perturbation = np.zeros(3)

        # Solar radiation pressure (very simplified)
        r_mag = np.linalg.norm(position)
        if r_mag > 6.6e6:  # Above Earth's shadow
            # Simplified radiation pressure
            radiation_accel = 1e-7  # Very small acceleration [m/s²]
            perturbation += radiation_accel * position / r_mag

        return perturbation

    def compare_with_twobody(self,
                           initial_position: np.ndarray,
                           initial_velocity: np.ndarray,
                           propagation_time: float) -> dict[str, Any]:
        """Compare n-body propagation with two-body propagation.
        
        Args:
            initial_position: Initial position [m]
            initial_velocity: Initial velocity [m/s]
            propagation_time: Propagation time [s]
            
        Returns
        -------
            Comparison results
        """
        logger.info("Comparing n-body vs two-body propagation")

        # N-body propagation
        nbody_result = self.propagate_spacecraft(
            initial_position, initial_velocity, 10000.0, propagation_time
        )

        # Two-body propagation using Kepler's laws
        twobody_result = self._propagate_twobody(
            initial_position, initial_velocity, propagation_time
        )

        # Calculate differences
        pos_diff = nbody_result["positions"] - twobody_result["positions"]
        vel_diff = nbody_result["velocities"] - twobody_result["velocities"]

        pos_error = np.linalg.norm(pos_diff, axis=0)
        vel_error = np.linalg.norm(vel_diff, axis=0)

        comparison = {
            "nbody_result": nbody_result,
            "twobody_result": twobody_result,
            "position_errors": pos_error,
            "velocity_errors": vel_error,
            "max_position_error": np.max(pos_error),
            "final_position_error": pos_error[-1],
            "max_velocity_error": np.max(vel_error),
            "final_velocity_error": vel_error[-1],
            "propagation_time": propagation_time
        }

        logger.info(f"Comparison complete: max position error = {np.max(pos_error)/1000:.1f} km")
        return comparison

    def _propagate_twobody(self,
                          initial_position: np.ndarray,
                          initial_velocity: np.ndarray,
                          propagation_time: float) -> dict[str, np.ndarray]:
        """Propagate using two-body dynamics for comparison."""

        def twobody_dynamics(t: float, state: np.ndarray) -> np.ndarray:
            r = state[:3]
            v = state[3:]
            r_mag = np.linalg.norm(r)
            acceleration = -self.mu["earth"] * r / r_mag**3
            return np.concatenate([v, acceleration])

        initial_state = np.concatenate([initial_position, initial_velocity])
        time_span = (0.0, propagation_time)

        times, states = self.integrator.integrate_trajectory(
            dynamics_function=twobody_dynamics,
            initial_state=initial_state,
            time_span=time_span
        )

        return {
            "times": times,
            "positions": states[:3, :],
            "velocities": states[3:, :]
        }


class TrajectoryIO:
    """Trajectory serialization and I/O utilities.
    
    This class provides methods to save, load, and manage trajectory data
    in various formats including JSON, pickle, and custom binary formats.
    """

    def __init__(self, base_directory: str = "trajectories"):
        """Initialize trajectory I/O manager.
        
        Args:
            base_directory: Base directory for trajectory storage
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)

        # Supported file formats
        self.formats = ["json", "pickle", "npz", "csv"]

        logger.info(f"Initialized TrajectoryIO with base directory: {self.base_directory}")

    def save_trajectory(self,
                       trajectory: Trajectory,
                       filename: str,
                       format: str = "json",
                       metadata: dict[str, Any] = None) -> Path:
        """Save trajectory to file.
        
        Args:
            trajectory: Trajectory object to save
            filename: Output filename (without extension)
            format: File format ('json', 'pickle', 'npz')
            metadata: Additional metadata to save
            
        Returns
        -------
            Path to saved file
        """
        if format not in self.formats:
            raise ValueError(f"Unsupported format: {format}")

        # Create full filepath
        filepath = self.base_directory / f"{filename}.{format}"

        logger.info(f"Saving trajectory to {filepath}")

        # Prepare data for saving
        data = self._trajectory_to_dict(trajectory)
        if metadata:
            data["metadata"] = metadata
        data["save_timestamp"] = datetime.now().isoformat()

        # Save in specified format
        if format == "json":
            self._save_json(data, filepath)
        elif format == "pickle":
            self._save_pickle(data, filepath)
        elif format == "npz":
            self._save_npz(data, filepath)

        logger.info(f"Trajectory saved successfully to {filepath}")
        return filepath

    def load_trajectory(self, filepath: Path) -> tuple[Trajectory, dict[str, Any]]:
        """Load trajectory from file.
        
        Args:
            filepath: Path to trajectory file
            
        Returns
        -------
            Tuple of (trajectory, metadata)
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")

        logger.info(f"Loading trajectory from {filepath}")

        # Determine format from extension
        format = filepath.suffix[1:]  # Remove dot

        # Load data
        if format == "json":
            data = self._load_json(filepath)
        elif format == "pickle":
            data = self._load_pickle(filepath)
        elif format == "npz":
            data = self._load_npz(filepath)
        else:
            raise ValueError(f"Unsupported file format: {format}")

        # Convert back to trajectory object
        trajectory = self._dict_to_trajectory(data)
        metadata = data.get("metadata", {})

        logger.info(f"Trajectory loaded successfully from {filepath}")
        return trajectory, metadata

    def save_propagation_result(self,
                              result: dict[str, np.ndarray],
                              filename: str,
                              format: str = "npz") -> Path:
        """Save propagation result to file.
        
        Args:
            result: Propagation result dictionary
            filename: Output filename
            format: File format ('npz', 'pickle')
            
        Returns
        -------
            Path to saved file
        """
        filepath = self.base_directory / f"{filename}.{format}"

        logger.info(f"Saving propagation result to {filepath}")

        if format == "npz":
            # Save numpy arrays
            np.savez_compressed(filepath, **result)
        elif format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(result, f)
        else:
            raise ValueError(f"Unsupported format for propagation result: {format}")

        return filepath

    def load_propagation_result(self, filepath: Path) -> dict[str, np.ndarray]:
        """Load propagation result from file.
        
        Args:
            filepath: Path to result file
            
        Returns
        -------
            Propagation result dictionary
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Result file not found: {filepath}")

        format = filepath.suffix[1:]

        if format == "npz":
            with np.load(filepath) as data:
                return {key: data[key] for key in data.keys()}
        elif format == "pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def list_trajectories(self) -> list[dict[str, Any]]:
        """List all saved trajectories.
        
        Returns
        -------
            List of trajectory information
        """
        trajectories = []

        for filepath in self.base_directory.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                info = {
                    "filename": filepath.name,
                    "filepath": filepath,
                    "departure_epoch": data.get("departure_epoch"),
                    "arrival_epoch": data.get("arrival_epoch"),
                    "save_timestamp": data.get("save_timestamp"),
                    "metadata": data.get("metadata", {})
                }
                trajectories.append(info)

            except Exception as e:
                logger.warning(f"Failed to read trajectory info from {filepath}: {e}")

        return sorted(trajectories, key=lambda x: x.get("save_timestamp", ""))

    def _trajectory_to_dict(self, trajectory: Trajectory) -> dict[str, Any]:
        """Convert trajectory object to dictionary."""
        return {
            "departure_epoch": trajectory.departure_epoch,
            "arrival_epoch": trajectory.arrival_epoch,
            "departure_pos": trajectory.departure_pos,
            "departure_vel": trajectory.departure_vel,
            "arrival_pos": trajectory.arrival_pos,
            "arrival_vel": trajectory.arrival_vel,
            "maneuvers": [
                {
                    "epoch": m.epoch,
                    "delta_v": m.delta_v,
                    "name": m.name
                } for m in trajectory.maneuvers
            ]
        }

    def _dict_to_trajectory(self, data: dict[str, Any]) -> Trajectory:
        """Convert dictionary to trajectory object."""
        trajectory = Trajectory(
            departure_epoch=data["departure_epoch"],
            arrival_epoch=data["arrival_epoch"],
            departure_pos=tuple(data["departure_pos"]),
            departure_vel=tuple(data["departure_vel"]),
            arrival_pos=tuple(data["arrival_pos"]),
            arrival_vel=tuple(data["arrival_vel"])
        )

        # Add maneuvers if present
        for maneuver_data in data.get("maneuvers", []):
            from .models import Maneuver
            maneuver = Maneuver(
                epoch=maneuver_data["epoch"],
                delta_v=tuple(maneuver_data["delta_v"]),
                name=maneuver_data["name"]
            )
            trajectory.add_maneuver(maneuver)

        return trajectory

    def _save_json(self, data: dict[str, Any], filepath: Path) -> None:
        """Save data to JSON file."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_json(self, filepath: Path) -> dict[str, Any]:
        """Load data from JSON file."""
        with open(filepath) as f:
            return json.load(f)

    def _save_pickle(self, data: dict[str, Any], filepath: Path) -> None:
        """Save data to pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def _load_pickle(self, filepath: Path) -> dict[str, Any]:
        """Load data from pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def _save_npz(self, data: dict[str, Any], filepath: Path) -> None:
        """Save numerical data to NPZ file."""
        # Convert to numpy arrays where possible
        numpy_data = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                numpy_data[key] = np.array(value)
            elif isinstance(value, (int, float)):
                numpy_data[key] = np.array([value])
            else:
                numpy_data[key] = str(value)  # Convert to string

        np.savez_compressed(filepath, **numpy_data)

    def _load_npz(self, filepath: Path) -> dict[str, Any]:
        """Load data from NPZ file."""
        with np.load(filepath, allow_pickle=True) as npz_data:
            data = {}
            for key in npz_data.keys():
                value = npz_data[key]
                if value.ndim == 0:  # Scalar
                    data[key] = value.item()
                else:
                    data[key] = value.tolist()
            return data


class TrajectoryComparison:
    """Utility class for comparing and analyzing trajectories.
    
    This class provides methods to compare different trajectory solutions,
    analyze accuracy, and generate comparative reports.
    """

    def __init__(self):
        """Initialize trajectory comparison utility."""
        logger.info("Initialized TrajectoryComparison")

    def compare_trajectories(self,
                           trajectory1: Trajectory,
                           trajectory2: Trajectory,
                           label1: str = "Trajectory 1",
                           label2: str = "Trajectory 2") -> dict[str, Any]:
        """Compare two trajectory solutions.
        
        Args:
            trajectory1: First trajectory
            trajectory2: Second trajectory
            label1: Label for first trajectory
            label2: Label for second trajectory
            
        Returns
        -------
            Comparison analysis
        """
        logger.info(f"Comparing trajectories: {label1} vs {label2}")

        # Calculate delta-v for each trajectory
        dv1 = sum(np.linalg.norm(m.delta_v) for m in trajectory1.maneuvers)
        dv2 = sum(np.linalg.norm(m.delta_v) for m in trajectory2.maneuvers)

        # Calculate transfer times
        time1 = trajectory1.arrival_epoch - trajectory1.departure_epoch
        time2 = trajectory2.arrival_epoch - trajectory2.departure_epoch

        # Position and velocity differences
        dep_pos_diff = np.linalg.norm(np.array(trajectory1.departure_pos) - np.array(trajectory2.departure_pos))
        arr_pos_diff = np.linalg.norm(np.array(trajectory1.arrival_pos) - np.array(trajectory2.arrival_pos))

        comparison = {
            "trajectories": {
                label1: {
                    "total_deltav": dv1,
                    "transfer_time": time1,
                    "num_maneuvers": len(trajectory1.maneuvers),
                    "departure_epoch": trajectory1.departure_epoch,
                    "arrival_epoch": trajectory1.arrival_epoch
                },
                label2: {
                    "total_deltav": dv2,
                    "transfer_time": time2,
                    "num_maneuvers": len(trajectory2.maneuvers),
                    "departure_epoch": trajectory2.departure_epoch,
                    "arrival_epoch": trajectory2.arrival_epoch
                }
            },
            "differences": {
                "deltav_difference": abs(dv1 - dv2),
                "time_difference": abs(time1 - time2),
                "departure_position_difference": dep_pos_diff,
                "arrival_position_difference": arr_pos_diff
            },
            "best_deltav": label1 if dv1 < dv2 else label2,
            "best_time": label1 if time1 < time2 else label2
        }

        logger.info(f"Comparison complete: ΔV difference = {comparison['differences']['deltav_difference']:.1f} m/s")
        return comparison

    def analyze_accuracy(self,
                        computed_trajectory: dict[str, np.ndarray],
                        reference_trajectory: dict[str, np.ndarray]) -> dict[str, float]:
        """Analyze accuracy of computed trajectory vs reference.
        
        Args:
            computed_trajectory: Computed trajectory data
            reference_trajectory: Reference trajectory data
            
        Returns
        -------
            Accuracy metrics
        """
        # Position accuracy
        pos_errors = np.linalg.norm(
            computed_trajectory["positions"] - reference_trajectory["positions"],
            axis=0
        )

        # Velocity accuracy
        vel_errors = np.linalg.norm(
            computed_trajectory["velocities"] - reference_trajectory["velocities"],
            axis=0
        )

        metrics = {
            "max_position_error": np.max(pos_errors),
            "rms_position_error": np.sqrt(np.mean(pos_errors**2)),
            "final_position_error": pos_errors[-1],
            "max_velocity_error": np.max(vel_errors),
            "rms_velocity_error": np.sqrt(np.mean(vel_errors**2)),
            "final_velocity_error": vel_errors[-1],
            "mean_position_error": np.mean(pos_errors),
            "mean_velocity_error": np.mean(vel_errors)
        }

        return metrics


# Convenience functions
def create_nbody_propagator(include_sun: bool = True,
                          include_perturbations: bool = False) -> EarthMoonNBodyPropagator:
    """Create Earth-Moon n-body propagator.
    
    Args:
        include_sun: Include solar effects
        include_perturbations: Include additional perturbations
        
    Returns
    -------
        Configured n-body propagator
    """
    return EarthMoonNBodyPropagator(include_sun, include_perturbations)


def create_trajectory_io(base_directory: str = "trajectories") -> TrajectoryIO:
    """Create trajectory I/O manager.
    
    Args:
        base_directory: Base directory for storage
        
    Returns
    -------
        Configured trajectory I/O manager
    """
    return TrajectoryIO(base_directory)
