"""Example lunar descent flight stage extension.

This module demonstrates how to implement a new flight stage extension
for lunar descent operations, showing the complete implementation pattern
for adding new mission phases to the system.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_extension import FlightStageExtension, ExtensionMetadata, ExtensionType

logger = logging.getLogger(__name__)


class LunarDescentExtension(FlightStageExtension):
    """Example extension for lunar descent trajectory planning.

    This extension demonstrates how to implement a new flight stage
    for powered descent from lunar orbit to the surface.
    """

    # Class-level extension type for registry
    EXTENSION_TYPE = ExtensionType.FLIGHT_STAGE

    def __init__(
        self,
        metadata: Optional[ExtensionMetadata] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the lunar descent extension.

        Args:
            metadata: Extension metadata (auto-generated if None)
            config: Extension configuration
        """
        if metadata is None:
            metadata = ExtensionMetadata(
                name="lunar_descent",
                version="1.0.0",
                description="Lunar descent trajectory planning and analysis",
                author="Lunar Horizon Optimizer Team",
                extension_type=ExtensionType.FLIGHT_STAGE,
                required_dependencies=[],
                optional_dependencies=["matplotlib"],
                api_version="1.0",
                enabled=True,
                configuration_schema={
                    "type": "object",
                    "properties": {
                        "max_descent_rate": {"type": "number", "default": 3.0},
                        "throttle_range": {"type": "array", "default": [0.1, 1.0]},
                        "landing_site_radius": {"type": "number", "default": 100.0},
                        "safety_margin": {"type": "number", "default": 1.2},
                    },
                },
            )

        super().__init__(metadata, config)

        # Extension-specific attributes
        self.stage_name = "lunar_descent"
        self._lunar_gravity = 1.62  # m/s^2
        self._min_altitude = 0.0  # Surface altitude

        # Configuration parameters
        self.max_descent_rate = self.config.get("max_descent_rate", 3.0)  # m/s
        self.throttle_range = self.config.get("throttle_range", [0.1, 1.0])
        self.landing_site_radius = self.config.get("landing_site_radius", 100.0)  # m
        self.safety_margin = self.config.get("safety_margin", 1.2)

    def initialize(self) -> bool:
        """Initialize the lunar descent extension."""
        try:
            # Validate configuration
            if not self.validate_configuration():
                return False

            # Initialize descent planning algorithms
            self._initialize_descent_algorithms()

            self._initialized = True
            self.logger.info("Lunar descent extension initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize lunar descent extension: {e}")
            return False

    def validate_configuration(self) -> bool:
        """Validate the extension configuration."""
        try:
            # Validate descent rate
            if self.max_descent_rate <= 0 or self.max_descent_rate > 10:
                self.logger.error("Max descent rate must be between 0 and 10 m/s")
                return False

            # Validate throttle range
            if (
                len(self.throttle_range) != 2
                or self.throttle_range[0] >= self.throttle_range[1]
                or self.throttle_range[0] < 0
                or self.throttle_range[1] > 1
            ):
                self.logger.error(
                    "Throttle range must be [min, max] with 0 <= min < max <= 1"
                )
                return False

            # Validate landing site radius
            if self.landing_site_radius <= 0:
                self.logger.error("Landing site radius must be positive")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def plan_trajectory(
        self,
        initial_state: Dict[str, Any],
        target_state: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Plan a lunar descent trajectory.

        Args:
            initial_state: Initial state in lunar orbit
            target_state: Target state on lunar surface
            constraints: Optional trajectory constraints

        Returns:
            Dictionary containing trajectory data and performance metrics
        """
        try:
            # Extract states
            r_initial = np.array(initial_state["position"])  # m
            v_initial = np.array(initial_state["velocity"])  # m/s
            m_initial = initial_state.get("mass", 5000.0)  # kg

            r_target = np.array(target_state["position"])  # m
            v_target = np.array(target_state.get("velocity", [0, 0, 0]))  # m/s

            # Extract constraints
            constraints = constraints or {}
            max_thrust = constraints.get("max_thrust", 45000.0)  # N
            specific_impulse = constraints.get("specific_impulse", 320.0)  # s

            # Plan descent trajectory
            trajectory = self._plan_powered_descent(
                r_initial,
                v_initial,
                m_initial,
                r_target,
                v_target,
                max_thrust,
                specific_impulse,
            )

            # Calculate performance metrics
            delta_v = self.calculate_delta_v(trajectory)
            fuel_consumption = self._calculate_fuel_consumption(trajectory)
            flight_time = trajectory["time_points"][-1] - trajectory["time_points"][0]

            result = {
                "trajectory": trajectory,
                "performance_metrics": {
                    "delta_v": delta_v,
                    "fuel_consumption": fuel_consumption,
                    "flight_time": flight_time,
                    "landing_accuracy": self._calculate_landing_accuracy(
                        trajectory, r_target
                    ),
                },
                "maneuver_sequence": self._generate_maneuver_sequence(trajectory),
                "stage_name": self.stage_name,
                "success": True,
            }

            self.logger.info(
                f"Lunar descent trajectory planned successfully (ΔV: {delta_v:.1f} m/s)"
            )
            return result

        except Exception as e:
            self.logger.error(f"Trajectory planning failed: {e}")
            return {
                "trajectory": {},
                "performance_metrics": {},
                "maneuver_sequence": [],
                "stage_name": self.stage_name,
                "success": False,
                "error": str(e),
            }

    def calculate_delta_v(self, trajectory: Dict[str, Any]) -> float:
        """Calculate total delta-v for the descent trajectory.

        Args:
            trajectory: Trajectory data from plan_trajectory

        Returns:
            Total delta-v requirement in m/s
        """
        try:
            if "thrust_profile" not in trajectory:
                return 0.0

            thrust_profile = trajectory["thrust_profile"]
            mass_profile = trajectory["mass_profile"]
            time_points = trajectory["time_points"]

            total_delta_v = 0.0

            for i in range(len(time_points) - 1):
                dt = time_points[i + 1] - time_points[i]
                thrust = thrust_profile[i]
                mass = mass_profile[i]

                if mass > 0 and thrust > 0:
                    # Delta-v for this time step
                    dv = (thrust / mass) * dt
                    total_delta_v += dv

            return total_delta_v

        except Exception as e:
            self.logger.error(f"Delta-v calculation failed: {e}")
            return 0.0

    def estimate_cost(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """Estimate costs for the lunar descent stage.

        Args:
            trajectory: Trajectory data from plan_trajectory

        Returns:
            Dictionary of cost estimates by category
        """
        try:
            # Extract trajectory metrics
            performance = trajectory.get("performance_metrics", {})
            fuel_consumption = performance.get("fuel_consumption", 0.0)
            flight_time = performance.get("flight_time", 0.0)

            # Cost parameters (in millions USD)
            fuel_cost_per_kg = 0.001  # $1000 per kg
            operations_cost_per_hour = 0.1  # $100K per hour
            hardware_base_cost = 50.0  # $50M for descent stage hardware

            # Calculate costs
            fuel_cost = fuel_consumption * fuel_cost_per_kg
            operations_cost = (flight_time / 3600) * operations_cost_per_hour
            hardware_cost = hardware_base_cost

            # Risk and complexity factors
            complexity_factor = 1.3  # 30% complexity overhead
            risk_factor = 1.2  # 20% risk margin

            total_factor = complexity_factor * risk_factor

            costs = {
                "fuel": fuel_cost * total_factor,
                "operations": operations_cost * total_factor,
                "hardware": hardware_cost * total_factor,
                "development": hardware_cost
                * 0.4
                * total_factor,  # 40% of hardware cost
                "testing": hardware_cost * 0.2 * total_factor,  # 20% of hardware cost
            }

            costs["total"] = sum(costs.values())

            self.logger.debug(f"Lunar descent cost estimate: ${costs['total']:.1f}M")
            return costs

        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            return {"total": 0.0, "error": str(e)}

    def get_capabilities(self) -> Dict[str, Any]:
        """Get lunar descent extension capabilities."""
        return {
            "type": "flight_stage",
            "stage_name": "lunar_descent",
            "provides_trajectory_planning": True,
            "provides_delta_v_calculation": True,
            "provides_cost_estimation": True,
            "supported_maneuvers": [
                "powered_descent",
                "terminal_guidance",
                "landing_burn",
            ],
            "trajectory_types": [
                "apollo_style_descent",
                "powered_descent_explicit_guidance",
                "gravity_turn_descent",
            ],
            "constraints_supported": [
                "max_thrust",
                "specific_impulse",
                "landing_site_constraints",
                "descent_rate_limits",
                "fuel_constraints",
            ],
            "outputs": [
                "trajectory_state_history",
                "thrust_profile",
                "mass_profile",
                "performance_metrics",
                "cost_breakdown",
                "maneuver_sequence",
            ],
        }

    def _initialize_descent_algorithms(self) -> None:
        """Initialize descent planning algorithms."""
        # Initialize guidance algorithms
        self._guidance_algorithms = {
            "explicit_guidance": self._explicit_guidance,
            "gravity_turn": self._gravity_turn_guidance,
            "apollo_style": self._apollo_style_guidance,
        }

        # Set default algorithm
        self._current_algorithm = "explicit_guidance"

    def _plan_powered_descent(
        self,
        r_initial: np.ndarray,
        v_initial: np.ndarray,
        m_initial: float,
        r_target: np.ndarray,
        v_target: np.ndarray,
        max_thrust: float,
        specific_impulse: float,
    ) -> Dict[str, Any]:
        """Plan the powered descent trajectory."""
        # Calculate descent parameters
        altitude_initial = np.linalg.norm(r_initial) - 1737.4e3  # Lunar radius
        altitude_target = np.linalg.norm(r_target) - 1737.4e3

        descent_distance = altitude_initial - altitude_target

        # Estimate descent time based on average descent rate
        avg_descent_rate = min(
            self.max_descent_rate, descent_distance / 120.0
        )  # At least 2 minutes
        descent_time = descent_distance / avg_descent_rate

        # Generate time points
        n_points = max(
            50, int(descent_time / 10)
        )  # At least 50 points, one every 10 seconds
        time_points = np.linspace(0, descent_time, n_points)

        # Initialize trajectory arrays
        positions = np.zeros((n_points, 3))
        velocities = np.zeros((n_points, 3))
        masses = np.zeros(n_points)
        thrusts = np.zeros(n_points)

        # Initial conditions
        positions[0] = r_initial
        velocities[0] = v_initial
        masses[0] = m_initial

        # Integrate trajectory using simple explicit guidance
        g_accel = np.array([0, 0, -self._lunar_gravity])  # Lunar gravity
        exhaust_velocity = specific_impulse * 9.81  # m/s

        for i in range(n_points - 1):
            dt = time_points[i + 1] - time_points[i]

            # Current state
            r = positions[i]
            v = velocities[i]
            m = masses[i]

            # Calculate required thrust for guidance
            thrust_required = self._calculate_guidance_thrust(
                r, v, r_target, v_target, m, time_points[i], descent_time
            )

            # Limit thrust
            thrust_magnitude = min(thrust_required, max_thrust)
            thrusts[i] = thrust_magnitude

            # Calculate thrust direction (toward target with velocity correction)
            if thrust_magnitude > 0:
                thrust_direction = self._calculate_thrust_direction(
                    r, v, r_target, v_target
                )
                thrust_vector = thrust_magnitude * thrust_direction
            else:
                thrust_vector = np.zeros(3)

            # Integrate dynamics
            acceleration = g_accel + thrust_vector / m

            # Update state using simple Euler integration
            positions[i + 1] = r + v * dt + 0.5 * acceleration * dt**2
            velocities[i + 1] = v + acceleration * dt

            # Update mass (rocket equation)
            if thrust_magnitude > 0:
                mass_flow_rate = thrust_magnitude / exhaust_velocity
                masses[i + 1] = m - mass_flow_rate * dt
            else:
                masses[i + 1] = m

        # Final thrust value
        thrusts[-1] = 0.0

        return {
            "time_points": time_points,
            "positions": positions,
            "velocities": velocities,
            "mass_profile": masses,
            "thrust_profile": thrusts,
            "algorithm": self._current_algorithm,
        }

    def _calculate_guidance_thrust(
        self,
        r: np.ndarray,
        v: np.ndarray,
        r_target: np.ndarray,
        v_target: np.ndarray,
        mass: float,
        current_time: float,
        total_time: float,
    ) -> float:
        """Calculate required thrust for explicit guidance."""
        # Time remaining
        time_to_go = total_time - current_time

        if time_to_go <= 0:
            return 0.0

        # Position and velocity errors
        r_error = r_target - r
        v_error = v_target - v

        # Desired acceleration (proportional navigation)
        kp = 2.0  # Position gain
        kd = 1.5  # Velocity gain

        desired_accel = (kp * r_error + kd * v_error) / time_to_go

        # Account for gravity
        gravity_accel = np.array([0, 0, -self._lunar_gravity])
        thrust_accel_required = desired_accel - gravity_accel

        # Required thrust magnitude
        thrust_required = mass * np.linalg.norm(thrust_accel_required)

        return max(0.0, thrust_required)

    def _calculate_thrust_direction(
        self, r: np.ndarray, v: np.ndarray, r_target: np.ndarray, v_target: np.ndarray
    ) -> np.ndarray:
        """Calculate thrust direction vector."""
        # Direction toward target with velocity correction
        r_error = r_target - r
        v_error = v_target - v

        # Combine position and velocity guidance
        guidance_vector = r_error + 0.5 * v_error

        # Normalize
        magnitude = np.linalg.norm(guidance_vector)
        if magnitude > 0:
            return guidance_vector / magnitude
        else:
            return np.array([0, 0, -1])  # Default downward

    def _calculate_fuel_consumption(self, trajectory: Dict[str, Any]) -> float:
        """Calculate total fuel consumption."""
        try:
            mass_profile = trajectory["mass_profile"]
            return mass_profile[0] - mass_profile[-1]  # Initial mass - final mass
        except Exception:
            return 0.0

    def _calculate_landing_accuracy(
        self, trajectory: Dict[str, Any], target_position: np.ndarray
    ) -> float:
        """Calculate landing accuracy (distance from target)."""
        try:
            final_position = trajectory["positions"][-1]
            return np.linalg.norm(final_position - target_position)
        except Exception:
            return float("inf")

    def _generate_maneuver_sequence(
        self, trajectory: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate sequence of maneuvers for the descent."""
        try:
            time_points = trajectory["time_points"]
            thrust_profile = trajectory["thrust_profile"]

            maneuvers = []

            # Find thrust phases
            in_burn = False
            burn_start_time = 0
            burn_start_idx = 0

            for i, thrust in enumerate(thrust_profile):
                if thrust > 1000 and not in_burn:  # Start of burn (>1000 N)
                    in_burn = True
                    burn_start_time = time_points[i]
                    burn_start_idx = i
                elif thrust <= 1000 and in_burn:  # End of burn
                    in_burn = False
                    burn_duration = time_points[i] - burn_start_time
                    avg_thrust = np.mean(thrust_profile[burn_start_idx:i])

                    maneuvers.append(
                        {
                            "type": "powered_descent_burn",
                            "start_time": burn_start_time,
                            "duration": burn_duration,
                            "average_thrust": avg_thrust,
                            "description": f"Descent burn {len(maneuvers) + 1}",
                        }
                    )

            # Handle final burn if trajectory ends during burn
            if in_burn:
                burn_duration = time_points[-1] - burn_start_time
                avg_thrust = np.mean(thrust_profile[burn_start_idx:])

                maneuvers.append(
                    {
                        "type": "final_landing_burn",
                        "start_time": burn_start_time,
                        "duration": burn_duration,
                        "average_thrust": avg_thrust,
                        "description": "Final landing burn",
                    }
                )

            return maneuvers

        except Exception as e:
            self.logger.error(f"Maneuver sequence generation failed: {e}")
            return []

    def _explicit_guidance(self, *args, **kwargs) -> Dict[str, Any]:
        """Explicit guidance algorithm implementation."""
        # Placeholder for future implementation
        return {"algorithm": "explicit_guidance"}

    def _gravity_turn_guidance(self, *args, **kwargs) -> Dict[str, Any]:
        """Gravity turn guidance algorithm implementation."""
        # Placeholder for future implementation
        return {"algorithm": "gravity_turn"}

    def _apollo_style_guidance(self, *args, **kwargs) -> Dict[str, Any]:
        """Apollo-style guidance algorithm implementation."""
        # Placeholder for future implementation
        return {"algorithm": "apollo_style"}
