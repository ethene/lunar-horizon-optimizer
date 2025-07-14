"""Decision vector definition with descent parameters for lunar mission optimization.

This module extends the existing multi-mission genome architecture to include
powered descent parameters, enabling end-to-end optimization from Earth orbit
to lunar surface landing.

Decision Vector Structure:
- Base parameters: 4*K + 2 (existing MultiMissionGenome)
- Descent parameters: +3 (burn_time, thrust, isp)
- Total length: 4*K + 5

Example for K=3 missions:
[epoch1, epoch2, epoch3,                    # Mission timing
 alt1, alt2, alt3,                          # Earth parking altitudes
 raan1, raan2, raan3,                       # Orbital plane orientations
 mass1, mass2, mass3,                       # Payload masses
 lunar_altitude, transfer_time,              # Shared orbital parameters
 descent_burn_time, descent_thrust, descent_isp]  # Shared descent parameters
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import logging

from src.optimization.multi_mission_genome import MultiMissionGenome

# from src.trajectory.continuous_thrust import powered_descent  # Not currently used
# import jax.numpy as jnp  # Not currently used

logger = logging.getLogger(__name__)


@dataclass
class DescentParameters:
    """Powered descent parameters for lunar landing optimization.

    These parameters define the powered descent phase from lunar orbit
    to surface landing using continuous thrust propulsion.
    """

    burn_time: float = 300.0  # Duration of powered descent [s] (5 minutes default)
    thrust: float = 15000.0  # Thrust magnitude [N] (15 kN default)
    isp: float = 300.0  # Specific impulse [s] (chemical propulsion default)

    def __post_init__(self):
        """Validate descent parameters are within reasonable bounds."""
        if self.burn_time <= 0:
            raise ValueError(f"burn_time must be positive, got {self.burn_time}")
        if self.thrust <= 0:
            raise ValueError(f"thrust must be positive, got {self.thrust}")
        if self.isp <= 0:
            raise ValueError(f"isp must be positive, got {self.isp}")


@dataclass
class MissionGenome:
    """Extended mission genome including descent parameters.

    Extends the MultiMissionGenome to include powered descent parameters
    for complete Earth-to-surface mission optimization.
    """

    # Base multi-mission parameters (existing architecture)
    base_genome: MultiMissionGenome = field(
        default_factory=lambda: MultiMissionGenome(num_missions=1)
    )

    # Powered descent parameters (shared across all missions)
    descent: DescentParameters = field(default_factory=DescentParameters)

    @property
    def num_missions(self) -> int:
        """Number of missions in constellation."""
        return self.base_genome.num_missions

    @classmethod
    def from_decision_vector(cls, x: List[float], num_missions: int) -> "MissionGenome":
        """Create genome from PyGMO decision vector with descent parameters.

        Decision vector structure (length = 4*K + 5):
        - Base parameters: 4*K + 2 (epochs, altitudes, raan, masses, lunar_alt, transfer_time)
        - Descent parameters: +3 (burn_time, thrust, isp)

        Args:
            x: Decision vector from PyGMO
            num_missions: Number of missions K

        Returns:
            MissionGenome instance with base and descent parameters
        """
        expected_length = 4 * num_missions + 5  # Base (4*K + 2) + Descent (3)
        if len(x) != expected_length:
            raise ValueError(
                f"Decision vector length {len(x)} != expected {expected_length} "
                f"for {num_missions} missions"
            )

        # Extract base parameters (first 4*K + 2 elements)
        base_length = 4 * num_missions + 2
        base_vector = x[:base_length]

        # Create base genome using existing MultiMissionGenome
        base_genome = MultiMissionGenome.from_decision_vector(base_vector, num_missions)

        # Extract descent parameters (last 3 elements)
        descent_start_idx = base_length
        descent_params = DescentParameters(
            burn_time=x[descent_start_idx],
            thrust=x[descent_start_idx + 1],
            isp=x[descent_start_idx + 2],
        )

        return cls(base_genome=base_genome, descent=descent_params)

    def to_decision_vector(self) -> List[float]:
        """Convert genome to PyGMO decision vector.

        Returns:
            Flattened decision vector: base_params + descent_params
        """
        base_vector = self.base_genome.to_decision_vector()
        descent_vector = [self.descent.burn_time, self.descent.thrust, self.descent.isp]

        return base_vector + descent_vector

    def get_mission_parameters(self, mission_idx: int) -> dict[str, float]:
        """Get complete parameters for specific mission including descent.

        Args:
            mission_idx: Mission index (0 to K-1)

        Returns:
            Dictionary with mission and descent parameters
        """
        # Get base mission parameters
        mission_params = self.base_genome.get_mission_parameters(mission_idx)

        # Add descent parameters
        mission_params.update(
            {
                "descent_burn_time": self.descent.burn_time,
                "descent_thrust": self.descent.thrust,
                "descent_isp": self.descent.isp,
            }
        )

        return mission_params


class LunarMissionProblem:
    """Extended PyGMO problem with powered descent optimization.

    This problem extends the existing lunar mission optimization to include
    powered descent from lunar orbit to surface landing, with objectives
    for total delta-v, mission time, and cost including descent operations.
    """

    def __init__(
        self,
        num_missions: int = 1,
        # Base parameter bounds (existing)
        min_epoch: float = 9000.0,
        max_epoch: float = 11000.0,
        min_earth_alt: float = 200.0,
        max_earth_alt: float = 1000.0,
        min_moon_alt: float = 50.0,
        max_moon_alt: float = 500.0,
        min_transfer_time: float = 3.0,
        max_transfer_time: float = 10.0,
        min_payload: float = 500.0,
        max_payload: float = 2000.0,
        # Descent parameter bounds (new)
        min_burn_time: float = 100.0,  # s (1.7 minutes minimum)
        max_burn_time: float = 1000.0,  # s (16.7 minutes maximum)
        min_thrust: float = 1000.0,  # N (1 kN minimum)
        max_thrust: float = 50000.0,  # N (50 kN maximum)
        min_isp: float = 200.0,  # s (cold gas thrusters)
        max_isp: float = 450.0,  # s (high-performance chemical)
        **kwargs,
    ):
        """Initialize extended lunar mission problem with descent parameters.

        Args:
            num_missions: Number of missions in constellation
            Base parameter bounds for orbital transfer optimization
            Descent parameter bounds for powered landing optimization
            **kwargs: Additional arguments for base problem initialization
        """
        self.num_missions = num_missions

        # Store base parameter bounds
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

        # Store descent parameter bounds
        self.min_burn_time = min_burn_time
        self.max_burn_time = max_burn_time
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.min_isp = min_isp
        self.max_isp = max_isp

        logger.info(
            f"Initialized extended LunarMissionProblem for {num_missions} missions "
            f"with descent parameters: "
            f"burn_time [{min_burn_time}-{max_burn_time}] s, "
            f"thrust [{min_thrust/1000:.0f}-{max_thrust/1000:.0f}] kN, "
            f"isp [{min_isp}-{max_isp}] s"
        )

    def decode(self, x: List[float]) -> MissionGenome:
        """Decode decision vector into mission genome including descent values.

        This method extracts all mission parameters including descent parameters
        and prepares them for trajectory generation and powered_descent() calls.

        Args:
            x: Decision vector from PyGMO optimizer

        Returns:
            MissionGenome with complete mission and descent parameters
        """
        # Decode decision vector using MissionGenome
        genome = MissionGenome.from_decision_vector(x, self.num_missions)

        # Validate all parameters are within bounds
        if not self._validate_bounds(genome):
            raise ValueError("Decoded parameters exceed defined bounds")

        return genome

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get optimization bounds for all decision variables including descent.

        Returns:
            Tuple of (lower_bounds, upper_bounds) for complete decision vector
        """
        lower = []
        upper = []

        # Mission-specific bounds (4*K parameters)
        for _ in range(self.num_missions):
            # Epochs
            lower.append(self.min_epoch)
            upper.append(self.max_epoch)

        for _ in range(self.num_missions):
            # Earth parking altitudes
            lower.append(self.min_earth_alt)
            upper.append(self.max_earth_alt)

        for _ in range(self.num_missions):
            # Orbital plane RAAN (0-360 degrees)
            lower.append(0.0)
            upper.append(360.0)

        for _ in range(self.num_missions):
            # Payload masses
            lower.append(self.min_payload)
            upper.append(self.max_payload)

        # Shared orbital parameters (2 parameters)
        lower.extend([self.min_moon_alt, self.min_transfer_time])
        upper.extend([self.max_moon_alt, self.max_transfer_time])

        # Descent parameters (3 parameters)
        lower.extend([self.min_burn_time, self.min_thrust, self.min_isp])
        upper.extend([self.max_burn_time, self.max_thrust, self.max_isp])

        return lower, upper

    def _validate_bounds(self, genome: MissionGenome) -> bool:
        """Validate genome parameters are within defined bounds.

        Args:
            genome: Mission genome to validate

        Returns:
            True if all parameters are within bounds
        """
        # Validate base parameters using existing logic
        base_valid = (
            all(
                self.min_epoch <= epoch <= self.max_epoch
                for epoch in genome.base_genome.epochs
            )
            and all(
                self.min_earth_alt <= alt <= self.max_earth_alt
                for alt in genome.base_genome.parking_altitudes
            )
            and all(
                self.min_payload <= mass <= self.max_payload
                for mass in genome.base_genome.payload_masses
            )
            and self.min_moon_alt
            <= genome.base_genome.lunar_altitude
            <= self.max_moon_alt
            and self.min_transfer_time
            <= genome.base_genome.transfer_time
            <= self.max_transfer_time
        )

        # Validate descent parameters
        descent_valid = (
            self.min_burn_time <= genome.descent.burn_time <= self.max_burn_time
            and self.min_thrust <= genome.descent.thrust <= self.max_thrust
            and self.min_isp <= genome.descent.isp <= self.max_isp
        )

        return base_valid and descent_valid


# Example usage and integration comment:
"""
Integration with powered_descent() in fitness evaluation:

def fitness(self, x: List[float]) -> List[float]:
    # 1. Decode decision vector including descent parameters
    genome = self.decode(x)
    
    total_objectives = []
    
    # 2. Evaluate each mission in constellation
    for i in range(genome.num_missions):
        mission_params = genome.get_mission_parameters(i)
        
        # 3. Generate orbital transfer (existing functionality)
        transfer_trajectory, transfer_dv = generate_orbital_transfer(
            epoch=mission_params["epoch"],
            earth_orbit_alt=mission_params["earth_orbit_alt"],
            moon_orbit_alt=mission_params["moon_orbit_alt"],
            transfer_time=mission_params["transfer_time"]
        )
        
        # 4. Generate powered descent trajectory (new functionality)
        # Extract final orbital state and convert to Moon-centered inertial frame
        final_orbital_state = convert_to_moon_centered_inertial(transfer_trajectory.final_state)
        
        # Call powered_descent() with decoded parameters
        descent_states, descent_times, descent_dv = powered_descent(
            start_state=final_orbital_state,
            thrust=mission_params["descent_thrust"],
            isp=mission_params["descent_isp"], 
            burn_time=mission_params["descent_burn_time"],
            steps=50  # Integration steps
        )
        
        # 5. Calculate total mission performance
        total_dv = transfer_dv + descent_dv
        total_time = mission_params["transfer_time"] * 86400 + mission_params["descent_burn_time"]
        total_cost = calculate_mission_cost(transfer_cost + descent_cost)
        
        total_objectives.append([total_dv, total_time, total_cost])
    
    # 6. Return aggregated objectives for multi-objective optimization
    return aggregate_constellation_objectives(total_objectives)

This architecture seamlessly integrates powered descent into the existing
multi-objective optimization framework while maintaining backward compatibility.
"""
