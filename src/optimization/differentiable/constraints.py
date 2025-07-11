"""
Differentiable Constraint Handling Module

This module implements sophisticated constraint handling for trajectory and economic
optimization using JAX automatic differentiation, providing penalty methods,
barrier functions, and augmented Lagrangian techniques.

Features:
- Differentiable constraint functions with automatic gradients
- Multiple constraint handling strategies (penalty, barrier, augmented Lagrangian)
- Physics-based constraints for trajectory optimization
- Economic feasibility constraints
- Adaptive penalty parameter updates
- Constraint violation monitoring and analysis

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Local imports
from .differentiable_models import TrajectoryModel, EconomicModel


class ConstraintType(Enum):
    """Enumeration of constraint types."""

    EQUALITY = "equality"  # g(x) = 0
    INEQUALITY = "inequality"  # g(x) <= 0
    BOX = "box"  # lb <= x <= ub
    PHYSICS = "physics"  # Physics-based constraints
    ECONOMIC = "economic"  # Economic feasibility constraints


class ConstraintHandlingMethod(Enum):
    """Enumeration of constraint handling methods."""

    PENALTY = "penalty"  # Penalty method
    BARRIER = "barrier"  # Barrier method
    AUGMENTED_LAGRANGIAN = "augmented_lagrangian"  # Augmented Lagrangian
    EXACT_PENALTY = "exact_penalty"  # Exact penalty method
    ADAPTIVE_PENALTY = "adaptive_penalty"  # Adaptive penalty parameters


@dataclass
class ConstraintViolation:
    """Container for constraint violation information."""

    name: str
    constraint_type: ConstraintType
    violation_value: float
    tolerance: float
    is_satisfied: bool
    penalty_contribution: float
    gradient_norm: float


@dataclass
class ConstraintConfig:
    """Configuration for constraint handling."""

    # Method configuration
    handling_method: ConstraintHandlingMethod = ConstraintHandlingMethod.PENALTY

    # Penalty parameters
    penalty_factor: float = 1e6
    penalty_growth_rate: float = 2.0
    max_penalty_factor: float = 1e12

    # Barrier parameters
    barrier_parameter: float = 0.1
    barrier_reduction_rate: float = 0.5
    min_barrier_parameter: float = 1e-8

    # Tolerance parameters
    equality_tolerance: float = 1e-6
    inequality_tolerance: float = 1e-3
    feasibility_tolerance: float = 1e-4

    # Adaptive parameters
    adaptation_rate: float = 0.1
    violation_threshold: float = 1e-2
    improvement_threshold: float = 0.1

    # Constraint weights
    constraint_weights: Dict[str, float] = field(
        default_factory=lambda: {"bounds": 1.0, "physics": 2.0, "economics": 1.5}
    )


class ConstraintHandler:
    """
    Differentiable constraint handler for trajectory optimization.

    This class provides sophisticated constraint handling using various methods
    including penalty functions, barrier methods, and augmented Lagrangian.
    """

    def __init__(
        self,
        trajectory_model: TrajectoryModel,
        economic_model: EconomicModel,
        config: Optional[ConstraintConfig] = None,
        use_jit: bool = True,
    ):
        """
        Initialize constraint handler.

        Args:
            trajectory_model: JAX trajectory model
            economic_model: JAX economic model
            config: Constraint handling configuration
            use_jit: Whether to use JIT compilation
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for differentiable constraint handling")

        self.trajectory_model = trajectory_model
        self.economic_model = economic_model
        self.config = config or ConstraintConfig()
        self.use_jit = use_jit

        # Constraint violation tracking
        self.violation_history: List[Dict[str, ConstraintViolation]] = []
        self.current_violations: Dict[str, ConstraintViolation] = {}

        # Adaptive parameters
        self.current_penalty_factor = self.config.penalty_factor
        self.current_barrier_parameter = self.config.barrier_parameter
        self.lagrange_multipliers: Dict[str, float] = {}

        # Setup compiled functions
        self._setup_compiled_functions()

    def _setup_compiled_functions(self):
        """Setup JIT-compiled constraint functions."""
        if self.use_jit:
            self.compute_constraint_values = jit(self._compute_constraint_values)
            self.compute_penalty_terms = jit(self._compute_penalty_terms)
            self.compute_barrier_terms = jit(self._compute_barrier_terms)
            self.compute_augmented_lagrangian = jit(self._compute_augmented_lagrangian)
        else:
            self.compute_constraint_values = self._compute_constraint_values
            self.compute_penalty_terms = self._compute_penalty_terms
            self.compute_barrier_terms = self._compute_barrier_terms
            self.compute_augmented_lagrangian = self._compute_augmented_lagrangian

    def _compute_constraint_values(
        self, parameters: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute all constraint values for given parameters.

        Args:
            parameters: Optimization parameters [r1, r2, tof]

        Returns:
            Dictionary of constraint values
        """
        constraints = {}

        # Extract parameters
        r1, r2, tof = parameters[0], parameters[1], parameters[2]

        # Box constraints (parameter bounds)
        constraints.update(self._compute_box_constraints(r1, r2, tof))

        # Physics constraints
        constraints.update(self._compute_physics_constraints(parameters))

        # Economic constraints
        constraints.update(self._compute_economic_constraints(parameters))

        return constraints

    def _compute_box_constraints(
        self, r1: jnp.ndarray, r2: jnp.ndarray, tof: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute box constraints for optimization parameters.

        Args:
            r1: Earth departure radius
            r2: Lunar orbit radius
            tof: Time of flight

        Returns:
            Dictionary of box constraint values
        """
        constraints = {}

        # Earth departure altitude constraints (200-600 km)
        earth_radius = 6.378e6
        min_r1, max_r1 = earth_radius + 200e3, earth_radius + 600e3
        constraints["r1_lower"] = min_r1 - r1  # <= 0
        constraints["r1_upper"] = r1 - max_r1  # <= 0

        # Lunar orbit altitude constraints (100-400 km)
        moon_radius = 1.737e6
        min_r2, max_r2 = moon_radius + 100e3, moon_radius + 400e3
        constraints["r2_lower"] = min_r2 - r2  # <= 0
        constraints["r2_upper"] = r2 - max_r2  # <= 0

        # Time of flight constraints (3-10 days)
        min_tof, max_tof = 3.0 * 24 * 3600, 10.0 * 24 * 3600
        constraints["tof_lower"] = min_tof - tof  # <= 0
        constraints["tof_upper"] = tof - max_tof  # <= 0

        return constraints

    def _compute_physics_constraints(
        self, parameters: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute physics-based constraints.

        Args:
            parameters: Optimization parameters

        Returns:
            Dictionary of physics constraint values
        """
        constraints = {}

        # Compute trajectory results
        traj_result = self.trajectory_model._trajectory_cost(parameters)

        # Delta-v feasibility constraints
        min_delta_v, max_delta_v = 2500.0, 15000.0  # Realistic bounds
        constraints["delta_v_lower"] = min_delta_v - traj_result["delta_v"]  # <= 0
        constraints["delta_v_upper"] = traj_result["delta_v"] - max_delta_v  # <= 0

        # Energy constraints (bound orbits)
        constraints["energy_bound"] = traj_result["energy"]  # <= 0 for bound orbits

        # Velocity constraints (reasonable magnitudes)
        v_final = jnp.linalg.norm(traj_result["final_velocity"])
        max_velocity = 5000.0  # 5 km/s maximum final velocity
        constraints["velocity_limit"] = v_final - max_velocity  # <= 0

        # Orbital mechanics constraints
        r1, r2 = parameters[0], parameters[1]

        # Ensure departure orbit is below arrival orbit for typical transfers
        # (This is a simplified constraint - real missions may vary)
        constraints["orbit_ordering"] = r1 - r2  # r1 <= r2 for Earth-Moon transfer

        # Circular orbit velocity constraints
        earth_mu = 3.986004418e14
        moon_mu = 4.9048695e12

        v1_circular = jnp.sqrt(earth_mu / r1)
        v2_circular = jnp.sqrt(moon_mu / r2)

        # Velocities should be within reasonable bounds of circular velocity
        max_velocity_factor = 2.0
        constraints["v1_feasibility"] = (
            v1_circular - max_velocity_factor * v1_circular
        )  # Complex constraint
        constraints["v2_feasibility"] = (
            v2_circular - max_velocity_factor * v2_circular
        )  # Complex constraint

        return constraints

    def _compute_economic_constraints(
        self, parameters: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute economic feasibility constraints.

        Args:
            parameters: Optimization parameters

        Returns:
            Dictionary of economic constraint values
        """
        constraints = {}

        # Compute trajectory and economic results
        traj_result = self.trajectory_model._trajectory_cost(parameters)
        econ_params = jnp.array([traj_result["delta_v"], traj_result["time_of_flight"]])
        econ_result = self.economic_model._economic_cost(econ_params)

        # Cost feasibility constraints
        max_total_cost = 100e9  # $100B maximum mission cost
        constraints["cost_feasibility"] = (
            econ_result["total_cost"] - max_total_cost
        )  # <= 0

        # Minimum mission viability (positive NPV)
        min_npv = 0.0
        constraints["npv_viability"] = min_npv - econ_result["npv"]  # <= 0

        # ROI constraints (minimum return)
        min_roi = 0.1  # 10% minimum ROI
        constraints["roi_minimum"] = min_roi - econ_result["roi"]  # <= 0

        # Launch cost reasonableness
        max_launch_cost = 50e9  # $50B maximum launch cost
        constraints["launch_cost_limit"] = (
            econ_result["launch_cost"] - max_launch_cost
        )  # <= 0

        # Operations cost constraints
        max_ops_cost_per_day = 1e6  # $1M per day maximum
        daily_ops_cost = econ_result["operations_cost"] / (
            traj_result["time_of_flight"] / (24 * 3600)
        )
        constraints["ops_cost_daily"] = daily_ops_cost - max_ops_cost_per_day  # <= 0

        return constraints

    def _compute_penalty_terms(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Compute penalty method terms for constraint violations.

        Args:
            parameters: Optimization parameters

        Returns:
            Total penalty value
        """
        constraints = self.compute_constraint_values(parameters)
        total_penalty = 0.0

        for name, value in constraints.items():
            # Get constraint weight
            constraint_type = self._get_constraint_type(name)
            weight = self.config.constraint_weights.get(constraint_type, 1.0)

            # Apply penalty based on constraint type
            if "equality" in name.lower():
                # Equality constraint: g(x) = 0
                penalty = value**2
            else:
                # Inequality constraint: g(x) <= 0
                penalty = jnp.maximum(0.0, value) ** 2

            total_penalty += weight * self.current_penalty_factor * penalty

        return total_penalty

    def _compute_barrier_terms(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Compute barrier method terms for inequality constraints.

        Args:
            parameters: Optimization parameters

        Returns:
            Total barrier value
        """
        constraints = self.compute_constraint_values(parameters)
        total_barrier = 0.0

        for name, value in constraints.items():
            # Only apply barrier to inequality constraints
            if "equality" not in name.lower():
                # Logarithmic barrier: -μ * log(-g(x)) for g(x) < 0
                # Add small epsilon to avoid numerical issues
                epsilon = 1e-8
                barrier_arg = -value - epsilon

                # Only add barrier if constraint is not violated
                barrier_term = jnp.where(
                    barrier_arg > 0,
                    -self.current_barrier_parameter * jnp.log(barrier_arg),
                    1e12,  # Large penalty for violated constraints
                )

                constraint_type = self._get_constraint_type(name)
                weight = self.config.constraint_weights.get(constraint_type, 1.0)
                total_barrier += weight * barrier_term

        return total_barrier

    def _compute_augmented_lagrangian(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Compute augmented Lagrangian terms.

        Args:
            parameters: Optimization parameters

        Returns:
            Total augmented Lagrangian value
        """
        constraints = self.compute_constraint_values(parameters)
        total_augmented = 0.0

        for name, value in constraints.items():
            # Get Lagrange multiplier (initialize if not exists)
            if name not in self.lagrange_multipliers:
                self.lagrange_multipliers[name] = 0.0

            lambda_val = self.lagrange_multipliers[name]

            if "equality" in name.lower():
                # Equality constraint: λ*g(x) + (ρ/2)*g(x)²
                augmented_term = (
                    lambda_val * value + 0.5 * self.current_penalty_factor * value**2
                )
            else:
                # Inequality constraint: max(λ + ρ*g(x), 0)*g(x) - λ²/(2ρ)
                lambda_update = jnp.maximum(
                    0.0, lambda_val + self.current_penalty_factor * value
                )
                augmented_term = lambda_update * value - lambda_val**2 / (
                    2.0 * self.current_penalty_factor
                )

            constraint_type = self._get_constraint_type(name)
            weight = self.config.constraint_weights.get(constraint_type, 1.0)
            total_augmented += weight * augmented_term

        return total_augmented

    def _get_constraint_type(self, constraint_name: str) -> str:
        """
        Determine constraint type from constraint name.

        Args:
            constraint_name: Name of the constraint

        Returns:
            Constraint type string
        """
        if any(prefix in constraint_name for prefix in ["r1_", "r2_", "tof_"]):
            return "bounds"
        elif any(
            prefix in constraint_name
            for prefix in ["delta_v", "energy", "velocity", "orbit"]
        ):
            return "physics"
        elif any(
            prefix in constraint_name
            for prefix in ["cost", "npv", "roi", "launch", "ops"]
        ):
            return "economics"
        else:
            return "other"

    def compute_constraint_function(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the constraint function value based on configured method.

        Args:
            parameters: Optimization parameters

        Returns:
            Constraint function value to add to objective
        """
        if self.config.handling_method == ConstraintHandlingMethod.PENALTY:
            return self._compute_penalty_terms(parameters)
        elif self.config.handling_method == ConstraintHandlingMethod.BARRIER:
            return self._compute_barrier_terms(parameters)
        elif (
            self.config.handling_method == ConstraintHandlingMethod.AUGMENTED_LAGRANGIAN
        ):
            return self._compute_augmented_lagrangian(parameters)
        elif self.config.handling_method == ConstraintHandlingMethod.ADAPTIVE_PENALTY:
            # Use penalty method with adaptive parameters
            return self._compute_penalty_terms(parameters)
        else:
            # Default to penalty method
            return self._compute_penalty_terms(parameters)

    def analyze_constraint_violations(
        self, parameters: jnp.ndarray
    ) -> Dict[str, ConstraintViolation]:
        """
        Analyze constraint violations for given parameters.

        Args:
            parameters: Optimization parameters

        Returns:
            Dictionary of constraint violation analysis
        """
        constraints = self.compute_constraint_values(parameters)
        violations = {}

        for name, value in constraints.items():
            # Determine constraint type
            if "equality" in name.lower():
                constraint_type = ConstraintType.EQUALITY
                tolerance = self.config.equality_tolerance
                is_satisfied = abs(float(value)) <= tolerance
                violation_value = abs(float(value))
            else:
                constraint_type = ConstraintType.INEQUALITY
                tolerance = self.config.inequality_tolerance
                is_satisfied = float(value) <= tolerance
                violation_value = max(0.0, float(value))

            # Compute penalty contribution
            if constraint_type == ConstraintType.EQUALITY:
                penalty = float(value) ** 2
            else:
                penalty = max(0.0, float(value)) ** 2

            penalty_contribution = (
                self.current_penalty_factor
                * self.config.constraint_weights.get(
                    self._get_constraint_type(name), 1.0
                )
                * penalty
            )

            # Compute gradient norm (approximate)
            try:
                grad_fn = grad(lambda x: self.compute_constraint_values(x)[name])
                gradient = grad_fn(parameters)
                gradient_norm = float(jnp.linalg.norm(gradient))
            except:
                gradient_norm = 0.0

            violations[name] = ConstraintViolation(
                name=name,
                constraint_type=constraint_type,
                violation_value=violation_value,
                tolerance=tolerance,
                is_satisfied=is_satisfied,
                penalty_contribution=penalty_contribution,
                gradient_norm=gradient_norm,
            )

        self.current_violations = violations
        self.violation_history.append(violations.copy())

        return violations

    def update_adaptive_parameters(self, parameters: jnp.ndarray):
        """
        Update adaptive constraint handling parameters.

        Args:
            parameters: Current optimization parameters
        """
        violations = self.analyze_constraint_violations(parameters)

        if self.config.handling_method == ConstraintHandlingMethod.ADAPTIVE_PENALTY:
            # Adaptive penalty parameter update
            max_violation = (
                max(
                    v.violation_value for v in violations.values() if not v.is_satisfied
                )
                if any(not v.is_satisfied for v in violations.values())
                else 0.0
            )

            if max_violation > self.config.violation_threshold:
                # Increase penalty factor if violations are too large
                self.current_penalty_factor = min(
                    self.current_penalty_factor * self.config.penalty_growth_rate,
                    self.config.max_penalty_factor,
                )
            elif max_violation < self.config.violation_threshold * 0.1:
                # Decrease penalty factor if violations are very small
                self.current_penalty_factor = max(
                    self.current_penalty_factor / self.config.penalty_growth_rate,
                    self.config.penalty_factor,
                )

        elif self.config.handling_method == ConstraintHandlingMethod.BARRIER:
            # Adaptive barrier parameter update
            if all(v.is_satisfied for v in violations.values()):
                # Reduce barrier parameter if all constraints are satisfied
                self.current_barrier_parameter = max(
                    self.current_barrier_parameter * self.config.barrier_reduction_rate,
                    self.config.min_barrier_parameter,
                )

        elif (
            self.config.handling_method == ConstraintHandlingMethod.AUGMENTED_LAGRANGIAN
        ):
            # Update Lagrange multipliers
            for name, violation in violations.items():
                if name in self.lagrange_multipliers:
                    if violation.constraint_type == ConstraintType.EQUALITY:
                        # Update for equality constraints
                        self.lagrange_multipliers[name] += (
                            self.current_penalty_factor * violation.violation_value
                        )
                    else:
                        # Update for inequality constraints
                        self.lagrange_multipliers[name] = max(
                            0.0,
                            self.lagrange_multipliers[name]
                            + self.current_penalty_factor * violation.violation_value,
                        )

    def get_constraint_summary(self, parameters: jnp.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive constraint violation summary.

        Args:
            parameters: Optimization parameters

        Returns:
            Dictionary with constraint summary
        """
        violations = self.analyze_constraint_violations(parameters)

        summary = {
            "total_violations": len(
                [v for v in violations.values() if not v.is_satisfied]
            ),
            "total_constraints": len(violations),
            "satisfaction_rate": len([v for v in violations.values() if v.is_satisfied])
            / len(violations),
            "max_violation": max(v.violation_value for v in violations.values()),
            "total_penalty": sum(v.penalty_contribution for v in violations.values()),
            "constraint_function_value": float(
                self.compute_constraint_function(parameters)
            ),
            "handling_method": self.config.handling_method.value,
            "current_penalty_factor": self.current_penalty_factor,
            "current_barrier_parameter": self.current_barrier_parameter,
            "violations_by_type": {},
            "violations_by_name": {
                name: v.violation_value for name, v in violations.items()
            },
        }

        # Group violations by type
        for constraint_type in ["bounds", "physics", "economics"]:
            type_violations = [
                v
                for name, v in violations.items()
                if self._get_constraint_type(name) == constraint_type
                and not v.is_satisfied
            ]
            summary["violations_by_type"][constraint_type] = len(type_violations)

        return summary


# Utility functions for creating constraint handlers
def create_penalty_constraint_handler(
    trajectory_model: TrajectoryModel,
    economic_model: EconomicModel,
    penalty_factor: float = 1e6,
    **kwargs,
) -> ConstraintHandler:
    """
    Create a penalty method constraint handler.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        penalty_factor: Penalty parameter
        **kwargs: Additional configuration parameters

    Returns:
        Configured ConstraintHandler instance
    """
    config = ConstraintConfig(
        handling_method=ConstraintHandlingMethod.PENALTY,
        penalty_factor=penalty_factor,
        **kwargs,
    )

    return ConstraintHandler(trajectory_model, economic_model, config)


def create_barrier_constraint_handler(
    trajectory_model: TrajectoryModel,
    economic_model: EconomicModel,
    barrier_parameter: float = 0.1,
    **kwargs,
) -> ConstraintHandler:
    """
    Create a barrier method constraint handler.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        barrier_parameter: Barrier parameter
        **kwargs: Additional configuration parameters

    Returns:
        Configured ConstraintHandler instance
    """
    config = ConstraintConfig(
        handling_method=ConstraintHandlingMethod.BARRIER,
        barrier_parameter=barrier_parameter,
        **kwargs,
    )

    return ConstraintHandler(trajectory_model, economic_model, config)


def create_adaptive_constraint_handler(
    trajectory_model: TrajectoryModel, economic_model: EconomicModel, **kwargs
) -> ConstraintHandler:
    """
    Create an adaptive constraint handler with automatic parameter updates.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        **kwargs: Additional configuration parameters

    Returns:
        Configured ConstraintHandler instance
    """
    config = ConstraintConfig(
        handling_method=ConstraintHandlingMethod.ADAPTIVE_PENALTY,
        penalty_factor=1e4,  # Start with lower penalty
        penalty_growth_rate=2.0,
        adaptation_rate=0.1,
        **kwargs,
    )

    return ConstraintHandler(trajectory_model, economic_model, config)
