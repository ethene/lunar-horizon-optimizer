"""Multi-mission genome design for constellation optimization.

This module implements the multi-mission genome architecture for optimizing
K simultaneous lunar transfers, such as deploying lunar communication satellite
constellations. Each chromosome encodes multiple missions with shared and
individual parameters.

Architecture:
- MultiMissionGenome: Dataclass encoding K transfers
- MultiMissionProblem: PyGMO problem for constellation optimization
- Migration utilities for backward compatibility
"""

from dataclasses import dataclass, field
import logging
from typing import List, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.optimization.global_optimizer import LunarMissionProblem

from src.config.costs import CostFactors
from src.optimization.cost_integration import CostCalculator
from src.trajectory.lunar_transfer import LunarTransfer

logger = logging.getLogger(__name__)


@dataclass
class MultiMissionGenome:
    """Multi-mission genome encoding K simultaneous lunar transfers.

    For a constellation of K missions, this genome encodes:
    - K launch epochs (timing optimization)
    - K parking orbit altitudes (Earth orbit optimization)
    - K orbital planes (RAAN values for constellation geometry)
    - K payload masses (mission-specific requirements)

    Design Philosophy:
    - Shared parameters reduce dimensionality where possible
    - Individual parameters allow mission-specific optimization
    - Constellation-aware objectives (coverage, redundancy, cost)
    """

    # Number of missions in constellation
    num_missions: int

    # Timing parameters - K values
    epochs: List[float] = field(default_factory=list)  # days since J2000

    # Orbital parameters - K values
    parking_altitudes: List[float] = field(default_factory=list)  # km, Earth orbit
    plane_raan: List[float] = field(
        default_factory=list
    )  # deg, orbital plane orientation

    # Mission parameters - K values
    payload_masses: List[float] = field(default_factory=list)  # kg

    # Shared parameters (apply to all missions)
    lunar_altitude: float = 100.0  # km, shared lunar orbit
    transfer_time: float = 5.0  # days, shared transfer duration

    def __post_init__(self):
        """Validate genome structure and initialize missing values."""
        if self.num_missions <= 0:
            raise ValueError("num_missions must be positive")

        # Initialize empty lists with default values
        if not self.epochs:
            self.epochs = [10000.0] * self.num_missions
        if not self.parking_altitudes:
            self.parking_altitudes = [400.0] * self.num_missions
        if not self.plane_raan:
            self.plane_raan = [
                i * (360.0 / self.num_missions) for i in range(self.num_missions)
            ]
        if not self.payload_masses:
            self.payload_masses = [1000.0] * self.num_missions

        # Validate lengths
        expected_len = self.num_missions
        if len(self.epochs) != expected_len:
            raise ValueError(
                f"epochs length {len(self.epochs)} != num_missions {expected_len}"
            )
        if len(self.parking_altitudes) != expected_len:
            raise ValueError(
                f"parking_altitudes length {len(self.parking_altitudes)} != num_missions {expected_len}"
            )
        if len(self.plane_raan) != expected_len:
            raise ValueError(
                f"plane_raan length {len(self.plane_raan)} != num_missions {expected_len}"
            )
        if len(self.payload_masses) != expected_len:
            raise ValueError(
                f"payload_masses length {len(self.payload_masses)} != num_missions {expected_len}"
            )

    @classmethod
    def from_decision_vector(
        cls, x: List[float], num_missions: int
    ) -> "MultiMissionGenome":
        """Create genome from PyGMO decision vector.

        Decision vector structure (length = 4*K + 2):
        - epochs[0..K-1]
        - parking_altitudes[0..K-1]
        - plane_raan[0..K-1]
        - payload_masses[0..K-1]
        - lunar_altitude (shared)
        - transfer_time (shared)

        Args:
            x: Decision vector from PyGMO
            num_missions: Number of missions K

        Returns:
            MultiMissionGenome instance
        """
        expected_length = 4 * num_missions + 2
        if len(x) != expected_length:
            raise ValueError(
                f"Decision vector length {len(x)} != expected {expected_length}"
            )

        # Extract mission-specific parameters (first 4*K elements)
        epochs = x[0:num_missions]
        parking_altitudes = x[num_missions : 2 * num_missions]
        plane_raan = x[2 * num_missions : 3 * num_missions]
        payload_masses = x[3 * num_missions : 4 * num_missions]

        # Extract shared parameters (last 2 elements)
        lunar_altitude = x[4 * num_missions]
        transfer_time = x[4 * num_missions + 1]

        return cls(
            num_missions=num_missions,
            epochs=epochs,
            parking_altitudes=parking_altitudes,
            plane_raan=plane_raan,
            payload_masses=payload_masses,
            lunar_altitude=lunar_altitude,
            transfer_time=transfer_time,
        )

    def to_decision_vector(self) -> List[float]:
        """Convert genome to PyGMO decision vector.

        Returns:
            Flattened decision vector for PyGMO
        """
        return (
            self.epochs
            + self.parking_altitudes
            + self.plane_raan
            + self.payload_masses
            + [self.lunar_altitude, self.transfer_time]
        )

    def get_mission_parameters(self, mission_idx: int) -> dict[str, float]:
        """Get parameters for specific mission.

        Args:
            mission_idx: Mission index (0 to K-1)

        Returns:
            Dictionary with mission parameters
        """
        if not 0 <= mission_idx < self.num_missions:
            raise ValueError(
                f"Mission index {mission_idx} out of range [0, {self.num_missions})"
            )

        return {
            "epoch": self.epochs[mission_idx],
            "earth_orbit_alt": self.parking_altitudes[mission_idx],
            "moon_orbit_alt": self.lunar_altitude,  # shared
            "transfer_time": self.transfer_time,  # shared
            "plane_raan": self.plane_raan[mission_idx],
            "payload_mass": self.payload_masses[mission_idx],
        }

    def validate_constellation_geometry(self) -> bool:
        """Validate constellation geometry constraints.

        Returns:
            True if constellation geometry is valid
        """
        # Check RAAN separation for coverage
        if self.num_missions > 1:
            raan_values = sorted(self.plane_raan)
            min_separation = min(
                raan_values[i + 1] - raan_values[i] for i in range(len(raan_values) - 1)
            )
            # Add wrap-around separation
            wrap_separation = 360.0 - raan_values[-1] + raan_values[0]
            min_separation = min(min_separation, wrap_separation)

            # Require minimum 10° separation for constellation
            if min_separation < 10.0:
                return False

        return True


class MultiMissionProblem:
    """PyGMO problem for multi-mission constellation optimization.

    This extends the single-mission optimization to handle K simultaneous
    lunar transfers with constellation-specific objectives and constraints.
    """

    def __init__(
        self,
        num_missions: int = 3,
        cost_factors: CostFactors = None,
        # Parameter bounds
        min_epoch: float = 9000.0,  # days since J2000
        max_epoch: float = 11000.0,  # days since J2000
        min_earth_alt: float = 200.0,  # km
        max_earth_alt: float = 1000.0,  # km
        min_moon_alt: float = 50.0,  # km
        max_moon_alt: float = 500.0,  # km
        min_transfer_time: float = 3.0,  # days
        max_transfer_time: float = 10.0,  # days
        min_payload: float = 500.0,  # kg
        max_payload: float = 2000.0,  # kg
        # Constellation objectives
        constellation_mode: bool = True,
        coverage_weight: float = 1.0,
        redundancy_weight: float = 0.5,
    ):
        """Initialize multi-mission optimization problem.

        Args:
            num_missions: Number of missions K in constellation
            cost_factors: Economic cost parameters
            Parameter bounds for decision variables
            constellation_mode: Enable constellation-specific objectives
            coverage_weight: Weight for coverage objective
            redundancy_weight: Weight for redundancy objective
        """
        self.num_missions = num_missions
        self.cost_factors = cost_factors or CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
        )

        # Parameter bounds
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.min_earth_alt = min_earth_alt
        self.max_earth_alt = max_earth_alt
        self.min_moon_alt = min_moon_alt
        self.max_moon_alt = max_moon_alt
        self.min_transfer_time = min_transfer_time
        self.max_transfer_time = max_transfer_time
        self.min_payload = min_payload
        self.max_payload = max_payload

        # Constellation configuration
        self.constellation_mode = constellation_mode
        self.coverage_weight = coverage_weight
        self.redundancy_weight = redundancy_weight

        # Initialize single-mission components
        self.lunar_transfer = LunarTransfer(
            min_earth_alt=min_earth_alt,
            max_earth_alt=max_earth_alt,
            min_moon_alt=min_moon_alt,
            max_moon_alt=max_moon_alt,
        )
        self.cost_calculator = CostCalculator(self.cost_factors)

        # Cache for expensive calculations
        self._trajectory_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(f"Initialized MultiMissionProblem for {num_missions} missions")

    def fitness(self, x: List[float]) -> List[float]:
        """Evaluate fitness for multi-mission optimization.

        Args:
            x: Decision vector encoding K missions

        Returns:
            List of objective values:
            - If constellation_mode: [total_delta_v, total_time, total_cost, coverage_metric, redundancy_metric]
            - Else: [total_delta_v, total_time, total_cost]
        """
        try:
            # Decode decision vector
            genome = MultiMissionGenome.from_decision_vector(x, self.num_missions)

            # Validate bounds
            if not self._validate_bounds(genome):
                return self._get_penalty_values()

            # Validate constellation geometry
            if self.constellation_mode and not genome.validate_constellation_geometry():
                return self._get_penalty_values()

            # Evaluate each mission
            mission_results = []
            total_delta_v = 0.0
            total_time = 0.0
            total_cost = 0.0

            for i in range(self.num_missions):
                mission_params = genome.get_mission_parameters(i)

                # Generate trajectory for this mission
                try:
                    trajectory, dv = self.lunar_transfer.generate_transfer(
                        epoch=mission_params["epoch"],
                        earth_orbit_alt=mission_params["earth_orbit_alt"],
                        moon_orbit_alt=mission_params["moon_orbit_alt"],
                        transfer_time=mission_params["transfer_time"],
                        max_revolutions=0,
                    )

                    # Calculate cost for this mission
                    mission_cost = self.cost_calculator.calculate_mission_cost(
                        total_dv=dv,
                        transfer_time=mission_params["transfer_time"],
                        earth_orbit_alt=mission_params["earth_orbit_alt"],
                        moon_orbit_alt=mission_params["moon_orbit_alt"],
                    )

                    mission_results.append(
                        {
                            "delta_v": dv,
                            "time": mission_params["transfer_time"],
                            "cost": mission_cost,
                            "trajectory": trajectory,
                        }
                    )

                    total_delta_v += dv
                    total_time += (
                        mission_params["transfer_time"] * 86400
                    )  # Convert to seconds
                    total_cost += mission_cost

                except Exception as e:
                    logger.debug(f"Mission {i} failed: {e}")
                    return self._get_penalty_values()

            # Base objectives (sum of individual missions)
            objectives = [total_delta_v, total_time, total_cost]

            # Add constellation-specific objectives
            if self.constellation_mode:
                coverage_metric = self._calculate_coverage_metric(genome)
                redundancy_metric = self._calculate_redundancy_metric(
                    genome, mission_results
                )

                objectives.extend([coverage_metric, redundancy_metric])

            return objectives

        except Exception as e:
            logger.debug(f"Fitness evaluation failed: {e}")
            return self._get_penalty_values()

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get optimization bounds for decision variables.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = []
        upper = []

        # Mission-specific bounds (4*K parameters)
        for _ in range(self.num_missions):
            # Epochs
            lower.append(self.min_epoch)
            upper.append(self.max_epoch)

        for _ in range(self.num_missions):
            # Parking altitudes
            lower.append(self.min_earth_alt)
            upper.append(self.max_earth_alt)

        for _ in range(self.num_missions):
            # Plane RAAN (0-360 degrees)
            lower.append(0.0)
            upper.append(360.0)

        for _ in range(self.num_missions):
            # Payload masses
            lower.append(self.min_payload)
            upper.append(self.max_payload)

        # Shared parameters (2 parameters)
        lower.extend([self.min_moon_alt, self.min_transfer_time])
        upper.extend([self.max_moon_alt, self.max_transfer_time])

        return lower, upper

    def get_nobj(self) -> int:
        """Get number of objectives."""
        return 5 if self.constellation_mode else 3

    def get_nec(self) -> int:
        """Get number of equality constraints."""
        return 0

    def get_nic(self) -> int:
        """Get number of inequality constraints."""
        return 0

    def get_name(self) -> str:
        """Get problem name."""
        return f"Multi-Mission Lunar Constellation ({self.num_missions} missions)"

    def _validate_bounds(self, genome: MultiMissionGenome) -> bool:
        """Validate genome parameters are within bounds."""
        # Check epoch bounds
        if not all(
            self.min_epoch <= epoch <= self.max_epoch for epoch in genome.epochs
        ):
            return False

        # Check altitude bounds
        if not all(
            self.min_earth_alt <= alt <= self.max_earth_alt
            for alt in genome.parking_altitudes
        ):
            return False

        # Check payload bounds
        if not all(
            self.min_payload <= mass <= self.max_payload
            for mass in genome.payload_masses
        ):
            return False

        # Check shared parameter bounds
        if not (self.min_moon_alt <= genome.lunar_altitude <= self.max_moon_alt):
            return False
        if not (
            self.min_transfer_time <= genome.transfer_time <= self.max_transfer_time
        ):
            return False

        return True

    def _get_penalty_values(self) -> List[float]:
        """Get penalty values for invalid solutions."""
        base_penalties = [1e8, 1e8, 1e12]  # [delta_v, time, cost]

        if self.constellation_mode:
            base_penalties.extend([1e6, 1e6])  # [coverage, redundancy]

        return base_penalties

    def _calculate_coverage_metric(self, genome: MultiMissionGenome) -> float:
        """Calculate constellation coverage metric.

        For lunar communication satellites, coverage depends on:
        - RAAN distribution (orbital plane spacing)
        - Number of satellites
        - Orbital altitude (affects coverage footprint)

        Args:
            genome: Multi-mission genome

        Returns:
            Coverage metric (lower is better for minimization)
        """
        # Simple coverage metric based on RAAN distribution
        raan_values = np.array(genome.plane_raan)

        # Calculate uniformity of RAAN distribution
        if self.num_missions == 1:
            uniformity = 1.0  # Single satellite
        else:
            # Calculate gaps between adjacent planes
            sorted_raan = np.sort(raan_values)
            gaps = np.diff(sorted_raan)
            # Add wrap-around gap
            wrap_gap = 360.0 - sorted_raan[-1] + sorted_raan[0]
            all_gaps = np.append(gaps, wrap_gap)

            # Ideal gap would be 360/K degrees
            ideal_gap = 360.0 / self.num_missions

            # Calculate uniformity (0 = perfect, higher = worse)
            uniformity = np.std(all_gaps - ideal_gap)

        # Coverage score (to minimize) - higher uniformity is better coverage
        coverage_score = uniformity * self.coverage_weight

        return coverage_score

    def _calculate_redundancy_metric(
        self, genome: MultiMissionGenome, mission_results: List[dict]
    ) -> float:
        """Calculate constellation redundancy metric.

        Redundancy is important for robust communications. Factors:
        - Number of satellites
        - Performance similarity (similar delta-v, cost)
        - Temporal distribution (launch timing)

        Args:
            genome: Multi-mission genome
            mission_results: Results from trajectory evaluation

        Returns:
            Redundancy metric (lower is better for minimization)
        """
        if self.num_missions == 1:
            return 1000.0  # Single satellite has no redundancy

        # Performance similarity - low variance in delta-v and cost is good
        delta_vs = [result["delta_v"] for result in mission_results]
        costs = [result["cost"] for result in mission_results]

        dv_variance = np.var(delta_vs) / (np.mean(delta_vs) + 1e-6)
        cost_variance = np.var(costs) / (np.mean(costs) + 1e-6)

        performance_diversity = (dv_variance + cost_variance) / 2.0

        # Temporal distribution - epochs should be reasonably spread
        epoch_variance = np.var(genome.epochs) / (np.mean(genome.epochs) + 1e-6)
        temporal_diversity = max(
            0.0, 0.1 - epoch_variance
        )  # Penalize too much clustering

        # Redundancy score (to minimize)
        redundancy_score = (
            performance_diversity + temporal_diversity
        ) * self.redundancy_weight

        return redundancy_score


def create_backward_compatible_problem(
    enable_multi: bool = False, num_missions: int = 1, **kwargs
) -> Union["LunarMissionProblem", MultiMissionProblem]:
    """Create optimization problem with backward compatibility.

    This function provides a migration path by allowing selection
    between single-mission (original) and multi-mission modes.

    Args:
        enable_multi: Enable multi-mission mode
        num_missions: Number of missions (only used if enable_multi=True)
        **kwargs: Additional arguments for problem initialization

    Returns:
        Either LunarMissionProblem (original) or MultiMissionProblem
    """
    if enable_multi and num_missions > 1:
        logger.info(f"Creating MultiMissionProblem with {num_missions} missions")
        return MultiMissionProblem(num_missions=num_missions, **kwargs)
    else:
        logger.info("Creating single-mission LunarMissionProblem (backward compatible)")
        # Filter kwargs for single-mission problem and provide defaults
        single_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "cost_factors",
                "min_earth_alt",
                "max_earth_alt",
                "min_moon_alt",
                "max_moon_alt",
                "min_transfer_time",
                "max_transfer_time",
                "reference_epoch",
            ]
        }

        # Ensure cost_factors is provided with reasonable defaults
        if "cost_factors" not in single_kwargs or single_kwargs["cost_factors"] is None:
            single_kwargs["cost_factors"] = CostFactors(
                launch_cost_per_kg=10000.0,
                operations_cost_per_day=100000.0,
                development_cost=1e9,
            )

        # Import here to avoid circular imports
        from src.optimization.global_optimizer import LunarMissionProblem

        return LunarMissionProblem(**single_kwargs)
