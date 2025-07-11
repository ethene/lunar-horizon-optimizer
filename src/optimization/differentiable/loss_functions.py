"""
Multi-Objective Loss Functions Module

This module implements sophisticated multi-objective loss functions for trajectory
and economic optimization, providing flexible weighting schemes, normalization
methods, and penalty functions for constraint handling.

Features:
- Multi-objective loss function combinations
- Adaptive weighting schemes (Pareto, lexicographic, etc.)
- Normalization strategies for different objective scales
- Penalty functions for constraint violations
- Loss function scheduling and adaptation
- Performance-aware objective balancing

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

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


class WeightingStrategy(Enum):
    """Enumeration of weighting strategies for multi-objective optimization."""

    FIXED = "fixed"  # Fixed weights throughout optimization
    ADAPTIVE = "adaptive"  # Weights adapt based on objective progress
    PARETO = "pareto"  # Pareto-based weighting
    LEXICOGRAPHIC = "lexicographic"  # Prioritized objective ordering
    SCALARIZED = "scalarized"  # Traditional scalarization
    ACHIEVEMENT = "achievement"  # Achievement scalarization
    WEIGHTED_SUM = "weighted_sum"  # Simple weighted sum


class NormalizationMethod(Enum):
    """Enumeration of normalization methods for objectives."""

    NONE = "none"  # No normalization
    MIN_MAX = "min_max"  # Min-max scaling [0, 1]
    Z_SCORE = "z_score"  # Z-score standardization
    ROBUST = "robust"  # Robust scaling using median/IQR
    UNIT_VECTOR = "unit_vector"  # Unit vector normalization
    REFERENCE_POINT = "reference_point"  # Reference point normalization


@dataclass
class ObjectiveMetrics:
    """Container for objective function metrics and statistics."""

    # Current values
    current_values: Dict[str, float] = field(default_factory=dict)
    normalized_values: Dict[str, float] = field(default_factory=dict)
    weighted_values: Dict[str, float] = field(default_factory=dict)

    # Statistics
    min_values: Dict[str, float] = field(default_factory=dict)
    max_values: Dict[str, float] = field(default_factory=dict)
    mean_values: Dict[str, float] = field(default_factory=dict)
    std_values: Dict[str, float] = field(default_factory=dict)

    # Progress tracking
    improvement_rates: Dict[str, float] = field(default_factory=dict)
    convergence_flags: Dict[str, bool] = field(default_factory=dict)
    iteration_count: int = 0


@dataclass
class LossFunctionConfig:
    """Configuration for multi-objective loss functions."""

    # Weighting configuration
    weighting_strategy: WeightingStrategy = WeightingStrategy.FIXED
    initial_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "delta_v": 1.0,
            "cost": 1.0,
            "time": 0.5,
            "safety": 0.3,
        }
    )

    # Normalization configuration
    normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX
    reference_values: Dict[str, float] = field(default_factory=dict)

    # Adaptive weighting parameters
    adaptation_rate: float = 0.1
    convergence_threshold: float = 1e-6
    stagnation_threshold: int = 10

    # Penalty parameters
    penalty_factor: float = 1e6
    constraint_tolerance: float = 1e-3
    penalty_growth_rate: float = 2.0

    # Performance parameters
    target_improvement_rate: float = 0.01
    objective_priorities: Dict[str, int] = field(default_factory=dict)


class MultiObjectiveLoss:
    """
    Multi-objective loss function for trajectory and economic optimization.

    This class provides flexible multi-objective optimization with various
    weighting strategies, normalization methods, and constraint handling.
    """

    def __init__(
        self,
        trajectory_model: TrajectoryModel,
        economic_model: EconomicModel,
        config: Optional[LossFunctionConfig] = None,
        use_jit: bool = True,
    ):
        """
        Initialize multi-objective loss function.

        Args:
            trajectory_model: JAX trajectory model
            economic_model: JAX economic model
            config: Loss function configuration
            use_jit: Whether to use JIT compilation
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for multi-objective loss functions")

        self.trajectory_model = trajectory_model
        self.economic_model = economic_model
        self.config = config or LossFunctionConfig()
        self.use_jit = use_jit

        # Initialize metrics tracking
        self.metrics = ObjectiveMetrics()
        self.history: List[ObjectiveMetrics] = []

        # Setup compiled functions
        self._setup_compiled_functions()

        # Initialize normalization parameters
        self._normalization_params = {}
        self._constraint_penalties = {}

    def _setup_compiled_functions(self):
        """Setup JIT-compiled loss functions."""
        if self.use_jit:
            self.compute_raw_objectives = jit(self._compute_raw_objectives)
            self.compute_normalized_objectives = jit(
                self._compute_normalized_objectives
            )
            self.compute_penalty_terms = jit(self._compute_penalty_terms)
        else:
            self.compute_raw_objectives = self._compute_raw_objectives
            self.compute_normalized_objectives = self._compute_normalized_objectives
            self.compute_penalty_terms = self._compute_penalty_terms

    def _compute_raw_objectives(
        self, parameters: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute raw objective values from trajectory and economic models.

        Args:
            parameters: Optimization parameters [r1, r2, tof]

        Returns:
            Dictionary of raw objective values
        """
        # Compute trajectory objectives
        traj_result = self.trajectory_model._trajectory_cost(parameters)

        # Compute economic objectives
        econ_params = jnp.array([traj_result["delta_v"], traj_result["time_of_flight"]])
        econ_result = self.economic_model._economic_cost(econ_params)

        # Extract key objectives
        objectives = {
            "delta_v": traj_result["delta_v"],
            "time": traj_result["time_of_flight"],
            "cost": econ_result["total_cost"],
            "energy": jnp.abs(traj_result["energy"]),
            "launch_cost": econ_result["launch_cost"],
            "operations_cost": econ_result["operations_cost"],
            "npv": -econ_result["npv"],  # Negative for minimization
            "roi": -econ_result["roi"],  # Negative for minimization
        }

        return objectives

    def _compute_normalized_objectives(
        self, raw_objectives: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Normalize objectives according to configured method.

        Args:
            raw_objectives: Raw objective values

        Returns:
            Normalized objective values
        """
        normalized = {}

        for key, value in raw_objectives.items():
            if self.config.normalization_method == NormalizationMethod.NONE:
                normalized[key] = value

            elif self.config.normalization_method == NormalizationMethod.MIN_MAX:
                if key in self._normalization_params:
                    min_val, max_val = self._normalization_params[key]
                    normalized[key] = (value - min_val) / (max_val - min_val + 1e-8)
                else:
                    normalized[key] = value

            elif self.config.normalization_method == NormalizationMethod.Z_SCORE:
                if key in self._normalization_params:
                    mean_val, std_val = self._normalization_params[key]
                    normalized[key] = (value - mean_val) / (std_val + 1e-8)
                else:
                    normalized[key] = value

            elif (
                self.config.normalization_method == NormalizationMethod.REFERENCE_POINT
            ):
                if key in self.config.reference_values:
                    ref_val = self.config.reference_values[key]
                    normalized[key] = value / (ref_val + 1e-8)
                else:
                    normalized[key] = value

            elif self.config.normalization_method == NormalizationMethod.UNIT_VECTOR:
                # Will be normalized after all objectives are computed
                normalized[key] = value

            else:
                normalized[key] = value

        # Apply unit vector normalization if specified
        if self.config.normalization_method == NormalizationMethod.UNIT_VECTOR:
            total_norm = jnp.sqrt(sum(v**2 for v in normalized.values()))
            normalized = {k: v / (total_norm + 1e-8) for k, v in normalized.items()}

        return normalized

    def _compute_penalty_terms(self, parameters: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Compute penalty terms for constraint violations.

        Args:
            parameters: Optimization parameters

        Returns:
            Dictionary of penalty terms
        """
        penalties = {}

        # Parameter bounds penalties
        r1, r2, tof = parameters[0], parameters[1], parameters[2]

        # Earth departure altitude constraints (200-600 km)
        earth_radius = 6.378e6
        min_r1, max_r1 = earth_radius + 200e3, earth_radius + 600e3
        penalties["r1_bounds"] = (
            jnp.maximum(0, min_r1 - r1) ** 2 + jnp.maximum(0, r1 - max_r1) ** 2
        )

        # Lunar orbit altitude constraints (100-400 km)
        moon_radius = 1.737e6
        min_r2, max_r2 = moon_radius + 100e3, moon_radius + 400e3
        penalties["r2_bounds"] = (
            jnp.maximum(0, min_r2 - r2) ** 2 + jnp.maximum(0, r2 - max_r2) ** 2
        )

        # Time of flight constraints (3-10 days)
        min_tof, max_tof = 3.0 * 24 * 3600, 10.0 * 24 * 3600
        penalties["tof_bounds"] = (
            jnp.maximum(0, min_tof - tof) ** 2 + jnp.maximum(0, tof - max_tof) ** 2
        )

        # Physics-based constraints
        # Minimum delta-v constraint (must be realistic)
        traj_result = self.trajectory_model._trajectory_cost(parameters)
        min_delta_v = 3000.0  # Minimum realistic delta-v for lunar mission
        penalties["min_delta_v"] = (
            jnp.maximum(0, min_delta_v - traj_result["delta_v"]) ** 2
        )

        # Maximum delta-v constraint (must be achievable)
        max_delta_v = 15000.0  # Maximum reasonable delta-v
        penalties["max_delta_v"] = (
            jnp.maximum(0, traj_result["delta_v"] - max_delta_v) ** 2
        )

        return penalties

    def _apply_weighting_strategy(
        self, normalized_objectives: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply weighting strategy to normalized objectives.

        Args:
            normalized_objectives: Normalized objective values

        Returns:
            Weighted objective values
        """
        weighted = {}

        if self.config.weighting_strategy == WeightingStrategy.FIXED:
            # Use fixed weights from configuration
            for key, value in normalized_objectives.items():
                weight = self.config.initial_weights.get(key, 1.0)
                weighted[key] = weight * value

        elif self.config.weighting_strategy == WeightingStrategy.ADAPTIVE:
            # Adaptive weights based on improvement rates
            weights = self._compute_adaptive_weights(normalized_objectives)
            for key, value in normalized_objectives.items():
                weight = weights.get(key, 1.0)
                weighted[key] = weight * value

        elif self.config.weighting_strategy == WeightingStrategy.PARETO:
            # Pareto-based weighting
            weights = self._compute_pareto_weights(normalized_objectives)
            for key, value in normalized_objectives.items():
                weight = weights.get(key, 1.0)
                weighted[key] = weight * value

        elif self.config.weighting_strategy == WeightingStrategy.LEXICOGRAPHIC:
            # Lexicographic ordering based on priorities
            weighted = self._apply_lexicographic_weighting(normalized_objectives)

        elif self.config.weighting_strategy == WeightingStrategy.ACHIEVEMENT:
            # Achievement scalarization
            weighted = self._apply_achievement_scalarization(normalized_objectives)

        else:
            # Default to weighted sum
            for key, value in normalized_objectives.items():
                weight = self.config.initial_weights.get(key, 1.0)
                weighted[key] = weight * value

        return weighted

    def _compute_adaptive_weights(
        self, objectives: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """
        Compute adaptive weights based on objective improvement rates.

        Args:
            objectives: Current objective values

        Returns:
            Dictionary of adaptive weights
        """
        weights = {}

        for key in objectives.keys():
            base_weight = self.config.initial_weights.get(key, 1.0)

            if key in self.metrics.improvement_rates:
                improvement_rate = self.metrics.improvement_rates[key]
                target_rate = self.config.target_improvement_rate

                # Increase weight for objectives with poor improvement
                if improvement_rate < target_rate * 0.5:
                    adaptation_factor = 1.5
                elif improvement_rate < target_rate:
                    adaptation_factor = 1.2
                else:
                    adaptation_factor = 1.0

                weights[key] = base_weight * adaptation_factor
            else:
                weights[key] = base_weight

        return weights

    def _compute_pareto_weights(
        self, objectives: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """
        Compute Pareto-based weights using dominance relationships.

        Args:
            objectives: Current objective values

        Returns:
            Dictionary of Pareto weights
        """
        # For this implementation, use a simplified Pareto weighting
        # In practice, this would involve more sophisticated Pareto analysis
        weights = {}

        total_objectives = len(objectives)
        equal_weight = 1.0 / total_objectives

        for key in objectives.keys():
            # Start with equal weights and adjust based on objective priority
            priority = self.config.objective_priorities.get(key, 1)
            weights[key] = equal_weight * priority

        return weights

    def _apply_lexicographic_weighting(
        self, objectives: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply lexicographic weighting based on objective priorities.

        Args:
            objectives: Normalized objective values

        Returns:
            Lexicographically weighted objectives
        """
        weighted = {}

        # Sort objectives by priority
        sorted_objectives = sorted(
            objectives.items(),
            key=lambda x: self.config.objective_priorities.get(x[0], 1),
            reverse=True,
        )

        # Apply exponentially decreasing weights
        for i, (key, value) in enumerate(sorted_objectives):
            weight = 10.0 ** (-i)  # Exponentially decreasing importance
            weighted[key] = weight * value

        return weighted

    def _apply_achievement_scalarization(
        self, objectives: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """
        Apply achievement scalarization method.

        Args:
            objectives: Normalized objective values

        Returns:
            Achievement-scalarized objectives
        """
        weighted = {}

        for key, value in objectives.items():
            reference_value = self.config.reference_values.get(key, 1.0)
            weight = self.config.initial_weights.get(key, 1.0)

            # Achievement function: minimize worst relative deviation
            achievement = weight * jnp.maximum(value / reference_value, 0.0)
            weighted[key] = achievement

        return weighted

    def compute_loss(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the complete multi-objective loss function.

        Args:
            parameters: Optimization parameters [r1, r2, tof]

        Returns:
            Scalar loss value
        """
        # Compute raw objectives
        raw_objectives = self.compute_raw_objectives(parameters)

        # Normalize objectives
        normalized_objectives = self.compute_normalized_objectives(raw_objectives)

        # Apply weighting strategy
        weighted_objectives = self._apply_weighting_strategy(normalized_objectives)

        # Compute constraint penalties
        penalties = self.compute_penalty_terms(parameters)

        # Combine objectives
        objective_sum = sum(weighted_objectives.values())

        # Add penalties
        penalty_sum = self.config.penalty_factor * sum(penalties.values())

        # Total loss
        total_loss = objective_sum + penalty_sum

        return total_loss

    def update_metrics(self, parameters: jnp.ndarray) -> ObjectiveMetrics:
        """
        Update metrics tracking for adaptive strategies.

        Args:
            parameters: Current optimization parameters

        Returns:
            Updated objective metrics
        """
        # Compute current objectives
        raw_objectives = self.compute_raw_objectives(parameters)
        normalized_objectives = self.compute_normalized_objectives(raw_objectives)
        weighted_objectives = self._apply_weighting_strategy(normalized_objectives)

        # Update current values
        self.metrics.current_values = {k: float(v) for k, v in raw_objectives.items()}
        self.metrics.normalized_values = {
            k: float(v) for k, v in normalized_objectives.items()
        }
        self.metrics.weighted_values = {
            k: float(v) for k, v in weighted_objectives.items()
        }

        # Update statistics
        for key, value in self.metrics.current_values.items():
            if key not in self.metrics.min_values:
                self.metrics.min_values[key] = value
                self.metrics.max_values[key] = value
                self.metrics.mean_values[key] = value
                self.metrics.std_values[key] = 0.0
            else:
                self.metrics.min_values[key] = min(self.metrics.min_values[key], value)
                self.metrics.max_values[key] = max(self.metrics.max_values[key], value)

        # Update normalization parameters
        self._update_normalization_parameters()

        # Increment iteration count
        self.metrics.iteration_count += 1

        # Store history
        self.history.append(
            ObjectiveMetrics(
                current_values=self.metrics.current_values.copy(),
                normalized_values=self.metrics.normalized_values.copy(),
                weighted_values=self.metrics.weighted_values.copy(),
                iteration_count=self.metrics.iteration_count,
            )
        )

        return self.metrics

    def _update_normalization_parameters(self):
        """Update normalization parameters based on observed data."""
        if self.config.normalization_method == NormalizationMethod.MIN_MAX:
            for key in self.metrics.current_values.keys():
                min_val = self.metrics.min_values[key]
                max_val = self.metrics.max_values[key]
                self._normalization_params[key] = (min_val, max_val)

        elif self.config.normalization_method == NormalizationMethod.Z_SCORE:
            # Update running mean and std (simplified)
            for key in self.metrics.current_values.keys():
                if len(self.history) > 1:
                    values = [
                        h.current_values[key]
                        for h in self.history
                        if key in h.current_values
                    ]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    self._normalization_params[key] = (mean_val, std_val)

    def get_objective_breakdown(self, parameters: jnp.ndarray) -> Dict[str, Any]:
        """
        Get detailed breakdown of objective function components.

        Args:
            parameters: Optimization parameters

        Returns:
            Dictionary with detailed objective breakdown
        """
        raw_objectives = self.compute_raw_objectives(parameters)
        normalized_objectives = self.compute_normalized_objectives(raw_objectives)
        weighted_objectives = self._apply_weighting_strategy(normalized_objectives)
        penalties = self.compute_penalty_terms(parameters)

        return {
            "raw_objectives": {k: float(v) for k, v in raw_objectives.items()},
            "normalized_objectives": {
                k: float(v) for k, v in normalized_objectives.items()
            },
            "weighted_objectives": {
                k: float(v) for k, v in weighted_objectives.items()
            },
            "penalties": {k: float(v) for k, v in penalties.items()},
            "total_objective": float(sum(weighted_objectives.values())),
            "total_penalty": float(
                self.config.penalty_factor * sum(penalties.values())
            ),
            "total_loss": float(self.compute_loss(parameters)),
            "weighting_strategy": self.config.weighting_strategy.value,
            "normalization_method": self.config.normalization_method.value,
        }


# Utility functions for creating common loss function configurations
def create_balanced_loss_function(
    trajectory_model: TrajectoryModel, economic_model: EconomicModel, **kwargs
) -> MultiObjectiveLoss:
    """
    Create a balanced multi-objective loss function with equal weighting.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        **kwargs: Additional configuration parameters

    Returns:
        Configured MultiObjectiveLoss instance
    """
    config = LossFunctionConfig(
        weighting_strategy=WeightingStrategy.FIXED,
        initial_weights={"delta_v": 1.0, "cost": 1.0, "time": 0.5},
        normalization_method=NormalizationMethod.MIN_MAX,
        **kwargs,
    )

    return MultiObjectiveLoss(trajectory_model, economic_model, config)


def create_performance_focused_loss_function(
    trajectory_model: TrajectoryModel, economic_model: EconomicModel, **kwargs
) -> MultiObjectiveLoss:
    """
    Create a performance-focused loss function emphasizing delta-v and time.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        **kwargs: Additional configuration parameters

    Returns:
        Configured MultiObjectiveLoss instance
    """
    config = LossFunctionConfig(
        weighting_strategy=WeightingStrategy.FIXED,
        initial_weights={"delta_v": 2.0, "time": 2.0, "cost": 0.5},
        normalization_method=NormalizationMethod.REFERENCE_POINT,
        reference_values={"delta_v": 3500.0, "time": 4.5 * 24 * 3600, "cost": 30e6},
        **kwargs,
    )

    return MultiObjectiveLoss(trajectory_model, economic_model, config)


def create_economic_focused_loss_function(
    trajectory_model: TrajectoryModel, economic_model: EconomicModel, **kwargs
) -> MultiObjectiveLoss:
    """
    Create an economics-focused loss function emphasizing cost and ROI.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        **kwargs: Additional configuration parameters

    Returns:
        Configured MultiObjectiveLoss instance
    """
    config = LossFunctionConfig(
        weighting_strategy=WeightingStrategy.FIXED,
        initial_weights={
            "cost": 2.0,
            "npv": 1.5,
            "roi": 1.5,
            "delta_v": 0.5,
            "time": 0.3,
        },
        normalization_method=NormalizationMethod.Z_SCORE,
        **kwargs,
    )

    return MultiObjectiveLoss(trajectory_model, economic_model, config)


def create_adaptive_loss_function(
    trajectory_model: TrajectoryModel, economic_model: EconomicModel, **kwargs
) -> MultiObjectiveLoss:
    """
    Create an adaptive loss function that adjusts weights during optimization.

    Args:
        trajectory_model: JAX trajectory model
        economic_model: JAX economic model
        **kwargs: Additional configuration parameters

    Returns:
        Configured MultiObjectiveLoss instance
    """
    config = LossFunctionConfig(
        weighting_strategy=WeightingStrategy.ADAPTIVE,
        initial_weights={"delta_v": 1.0, "cost": 1.0, "time": 0.5},
        normalization_method=NormalizationMethod.MIN_MAX,
        adaptation_rate=0.1,
        target_improvement_rate=0.01,
        **kwargs,
    )

    return MultiObjectiveLoss(trajectory_model, economic_model, config)
