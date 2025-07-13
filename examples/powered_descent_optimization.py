#!/usr/bin/env python3
"""Example demonstrating decision vector integration with powered descent optimization.

This example shows how to use the extended decision vector framework to optimize
complete Earth-to-surface lunar missions including powered descent parameters.
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimization.decision_vector import (
    DescentParameters,
    MissionGenome,
    LunarMissionProblem
)
from src.trajectory.continuous_thrust import powered_descent
from src.config.costs import CostFactors


class IntegratedLunarMissionOptimizer:
    """Complete mission optimizer including orbital transfer and powered descent."""
    
    def __init__(self, num_missions: int = 1):
        """Initialize integrated optimizer.
        
        Args:
            num_missions: Number of missions in constellation
        """
        self.num_missions = num_missions
        
        # Initialize decision vector problem
        self.problem = LunarMissionProblem(
            num_missions=num_missions,
            # Realistic descent parameter bounds
            min_burn_time=120.0,    # 2 minutes minimum
            max_burn_time=900.0,    # 15 minutes maximum  
            min_thrust=5000.0,      # 5 kN minimum for landing
            max_thrust=35000.0,     # 35 kN maximum
            min_isp=250.0,          # Cold gas backup
            max_isp=420.0           # High-performance chemical
        )
        
        # Moon parameters for powered descent
        self.moon_radius = 1.737e6  # m
        self.mu_moon = 4.9048695e12  # m³/s²
        
    def evaluate_complete_mission(self, x: List[float]) -> dict:
        """Evaluate complete mission including orbital transfer and descent.
        
        Args:
            x: Decision vector including descent parameters
            
        Returns:
            Dictionary with mission performance metrics
        """
        try:
            # Decode decision vector
            genome = self.problem.decode(x)
            
            # Initialize mission results
            total_delta_v = 0.0
            total_time = 0.0
            total_cost = 0.0
            mission_results = []
            
            # Evaluate each mission in constellation
            for i in range(genome.num_missions):
                mission_params = genome.get_mission_parameters(i)
                
                # 1. ORBITAL TRANSFER PHASE (simplified for demonstration)
                # In real implementation, this would use LunarTransfer.generate_transfer()
                transfer_dv = self._estimate_transfer_delta_v(
                    earth_alt=mission_params["earth_orbit_alt"],
                    moon_alt=mission_params["moon_orbit_alt"],
                    transfer_time=mission_params["transfer_time"]
                )
                transfer_time = mission_params["transfer_time"] * 86400  # Convert to seconds
                
                # 2. POWERED DESCENT PHASE
                descent_result = self._evaluate_powered_descent(mission_params)
                
                # 3. MISSION TOTALS
                mission_total_dv = transfer_dv + descent_result["delta_v"]
                mission_total_time = transfer_time + mission_params["descent_burn_time"]
                mission_cost = self._estimate_mission_cost(
                    total_dv=mission_total_dv,
                    total_time=mission_total_time,
                    payload_mass=mission_params["payload_mass"]
                )
                
                # Store mission result
                mission_result = {
                    "mission_id": i,
                    "transfer_dv": transfer_dv,
                    "descent_dv": descent_result["delta_v"],
                    "total_dv": mission_total_dv,
                    "total_time": mission_total_time,
                    "cost": mission_cost,
                    "final_altitude": descent_result["final_altitude"],
                    "landing_speed": descent_result["landing_speed"],
                    "fuel_consumed": descent_result["fuel_consumed"]
                }
                
                mission_results.append(mission_result)
                
                # Accumulate totals
                total_delta_v += mission_total_dv
                total_time += mission_total_time
                total_cost += mission_cost
            
            # Return comprehensive results
            return {
                "success": True,
                "constellation_metrics": {
                    "total_delta_v": total_delta_v,
                    "total_time": total_time,
                    "total_cost": total_cost,
                    "num_missions": genome.num_missions
                },
                "mission_results": mission_results,
                "descent_parameters": {
                    "burn_time": genome.descent.burn_time,
                    "thrust": genome.descent.thrust,
                    "isp": genome.descent.isp
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "constellation_metrics": {
                    "total_delta_v": 1e8,
                    "total_time": 1e8,
                    "total_cost": 1e12
                }
            }
    
    def _evaluate_powered_descent(self, mission_params: dict) -> dict:
        """Evaluate powered descent phase using JAX integration.
        
        Args:
            mission_params: Mission parameters including descent params
            
        Returns:
            Dictionary with descent performance metrics
        """
        # Create initial state for 100 km circular lunar orbit
        orbit_altitude = mission_params["moon_orbit_alt"] * 1000  # Convert km to m
        orbit_radius = self.moon_radius + orbit_altitude
        orbital_velocity = np.sqrt(self.mu_moon / orbit_radius)
        
        # Initial state: [x, y, z, vx, vy, vz, m] in Moon-centered inertial frame
        initial_mass = mission_params["payload_mass"] + 2000.0  # Add fuel mass estimate
        start_state = jnp.array([
            orbit_radius, 0.0, 0.0,         # Position [m]
            0.0, orbital_velocity, 0.0,     # Velocity [m/s]
            initial_mass                    # Mass [kg]
        ])
        
        # Run powered descent simulation
        states, times, total_delta_v = powered_descent(
            start_state=start_state,
            thrust=mission_params["descent_thrust"],
            isp=mission_params["descent_isp"],
            burn_time=mission_params["descent_burn_time"],
            steps=50  # Integration steps
        )
        
        # Extract final state
        final_state = states[-1]
        final_position = final_state[:3]
        final_velocity = final_state[3:6]
        final_mass = final_state[6]
        
        # Calculate performance metrics
        final_radius = jnp.linalg.norm(final_position)
        final_altitude = final_radius - self.moon_radius
        final_speed = jnp.linalg.norm(final_velocity)
        fuel_consumed = initial_mass - final_mass
        
        return {
            "delta_v": float(total_delta_v),
            "final_altitude": float(final_altitude),
            "landing_speed": float(final_speed),
            "fuel_consumed": float(fuel_consumed),
            "mass_ratio": float(fuel_consumed / initial_mass)
        }
    
    def _estimate_transfer_delta_v(self, earth_alt: float, moon_alt: float, transfer_time: float) -> float:
        """Estimate orbital transfer delta-v (simplified).
        
        Args:
            earth_alt: Earth orbit altitude [km]
            moon_alt: Moon orbit altitude [km] 
            transfer_time: Transfer time [days]
            
        Returns:
            Estimated delta-v [m/s]
        """
        # Simplified Hohmann-like transfer estimate
        # In real implementation, use LunarTransfer.generate_transfer()
        
        # Base delta-v for Earth escape and lunar insertion
        base_dv = 3200.0  # m/s (typical lunar transfer)
        
        # Altitude penalties (higher orbits need more delta-v)
        earth_penalty = max(0, (earth_alt - 400.0) * 2.0)  # 2 m/s per km above 400 km
        moon_penalty = max(0, (moon_alt - 100.0) * 1.5)    # 1.5 m/s per km above 100 km
        
        # Time penalty (faster transfers cost more)
        time_penalty = max(0, (5.0 - transfer_time) * 200.0)  # 200 m/s per day under 5 days
        
        return base_dv + earth_penalty + moon_penalty + time_penalty
    
    def _estimate_mission_cost(self, total_dv: float, total_time: float, payload_mass: float) -> float:
        """Estimate total mission cost including descent.
        
        Args:
            total_dv: Total mission delta-v [m/s]
            total_time: Total mission time [s]
            payload_mass: Payload mass [kg]
            
        Returns:
            Estimated cost [$]
        """
        # Simplified cost model
        # In real implementation, use CostCalculator
        
        # Base mission cost
        base_cost = 50e6  # $50M base
        
        # Delta-v cost (fuel and propulsion)
        dv_cost = total_dv * 5000.0  # $5000 per m/s
        
        # Time cost (operations)
        time_cost = (total_time / 86400.0) * 100000.0  # $100k per day
        
        # Payload cost
        payload_cost = payload_mass * 20000.0  # $20k per kg
        
        return base_cost + dv_cost + time_cost + payload_cost


def demonstrate_single_mission_optimization():
    """Demonstrate single mission optimization with descent parameters."""
    print("=== Single Mission Powered Descent Optimization ===\n")
    
    optimizer = IntegratedLunarMissionOptimizer(num_missions=1)
    
    # Create sample decision vector for single mission
    # Structure: [epoch, earth_alt, raan, payload, moon_alt, transfer_time, burn_time, thrust, isp]
    decision_vector = [
        10000.0,  # epoch (days since J2000)
        400.0,    # Earth orbit altitude (km)
        0.0,      # RAAN (degrees)
        1200.0,   # payload mass (kg)
        100.0,    # lunar orbit altitude (km)
        5.0,      # transfer time (days)
        300.0,    # descent burn time (s)
        18000.0,  # descent thrust (N)
        320.0     # descent specific impulse (s)
    ]
    
    print(f"Decision Vector: {decision_vector}")
    print(f"Vector Length: {len(decision_vector)} (expected: 4*1 + 5 = 9)\n")
    
    # Evaluate mission
    result = optimizer.evaluate_complete_mission(decision_vector)
    
    if result["success"]:
        print("Mission Evaluation Results:")
        print("-" * 40)
        
        constellation = result["constellation_metrics"]
        print(f"Total Delta-V: {constellation['total_delta_v']:.0f} m/s")
        print(f"Total Time: {constellation['total_time']/86400:.1f} days")
        print(f"Total Cost: ${constellation['total_cost']/1e6:.1f}M")
        
        mission = result["mission_results"][0]
        print(f"\nMission Breakdown:")
        print(f"  Transfer Delta-V: {mission['transfer_dv']:.0f} m/s")
        print(f"  Descent Delta-V: {mission['descent_dv']:.0f} m/s")
        print(f"  Final Altitude: {mission['final_altitude']/1000:.1f} km")
        print(f"  Landing Speed: {mission['landing_speed']:.1f} m/s")
        print(f"  Fuel Consumed: {mission['fuel_consumed']:.0f} kg")
        
        descent = result["descent_parameters"]
        print(f"\nDescent Parameters:")
        print(f"  Burn Time: {descent['burn_time']:.0f} s")
        print(f"  Thrust: {descent['thrust']/1000:.0f} kN")
        print(f"  Specific Impulse: {descent['isp']:.0f} s")
        
    else:
        print(f"Mission evaluation failed: {result['error']}")


def demonstrate_multi_mission_optimization():
    """Demonstrate multi-mission constellation optimization."""
    print("\n\n=== Multi-Mission Constellation Optimization ===\n")
    
    optimizer = IntegratedLunarMissionOptimizer(num_missions=3)
    
    # Create decision vector for 3-mission constellation
    # Structure: 4*3 + 5 = 17 parameters
    decision_vector = [
        # Mission epochs (staggered launches)
        10000.0, 10030.0, 10060.0,
        # Earth orbit altitudes (different parking orbits)
        400.0, 500.0, 600.0,
        # RAAN values (120° separation for coverage)
        0.0, 120.0, 240.0,
        # Payload masses (mission-specific)
        1000.0, 1200.0, 800.0,
        # Shared lunar orbit
        100.0,    # lunar altitude (km)
        5.0,      # transfer time (days)
        # Shared descent parameters
        420.0,    # burn time (s) - 7 minutes
        22000.0,  # thrust (N) - 22 kN
        340.0     # isp (s)
    ]
    
    print(f"Decision Vector: {decision_vector}")
    print(f"Vector Length: {len(decision_vector)} (expected: 4*3 + 5 = 17)\n")
    
    # Evaluate constellation
    result = optimizer.evaluate_complete_mission(decision_vector)
    
    if result["success"]:
        print("Constellation Evaluation Results:")
        print("-" * 45)
        
        constellation = result["constellation_metrics"]
        print(f"Total Delta-V: {constellation['total_delta_v']:.0f} m/s")
        print(f"Total Time: {constellation['total_time']/86400:.1f} days")
        print(f"Total Cost: ${constellation['total_cost']/1e6:.1f}M")
        print(f"Missions: {constellation['num_missions']}")
        
        print(f"\nPer-Mission Breakdown:")
        for i, mission in enumerate(result["mission_results"]):
            print(f"  Mission {i+1}:")
            print(f"    Total Delta-V: {mission['total_dv']:.0f} m/s")
            print(f"    Landing Speed: {mission['landing_speed']:.1f} m/s")
            print(f"    Cost: ${mission['cost']/1e6:.1f}M")
        
        descent = result["descent_parameters"]
        print(f"\nShared Descent Parameters:")
        print(f"  Burn Time: {descent['burn_time']:.0f} s ({descent['burn_time']/60:.1f} min)")
        print(f"  Thrust: {descent['thrust']/1000:.0f} kN")
        print(f"  Specific Impulse: {descent['isp']:.0f} s")
        
    else:
        print(f"Constellation evaluation failed: {result['error']}")


def demonstrate_parameter_sensitivity():
    """Demonstrate sensitivity to descent parameters."""
    print("\n\n=== Descent Parameter Sensitivity Analysis ===\n")
    
    optimizer = IntegratedLunarMissionOptimizer(num_missions=1)
    
    # Base decision vector
    base_vector = [10000.0, 400.0, 0.0, 1200.0, 100.0, 5.0, 300.0, 15000.0, 300.0]
    
    # Test different thrust levels
    thrust_levels = [10000.0, 15000.0, 20000.0, 25000.0]
    print("Thrust Sensitivity:")
    print("Thrust (kN) | Delta-V (m/s) | Landing Speed (m/s) | Fuel (kg)")
    print("-" * 65)
    
    for thrust in thrust_levels:
        test_vector = base_vector.copy()
        test_vector[7] = thrust  # Update thrust parameter
        
        result = optimizer.evaluate_complete_mission(test_vector)
        if result["success"]:
            mission = result["mission_results"][0]
            print(f"{thrust/1000:8.0f}   | {mission['descent_dv']:8.0f}    | "
                  f"{mission['landing_speed']:12.1f}    | {mission['fuel_consumed']:6.0f}")
    
    # Test different specific impulse values
    isp_levels = [250.0, 300.0, 350.0, 400.0]
    print(f"\n\nSpecific Impulse Sensitivity:")
    print("ISP (s) | Delta-V (m/s) | Fuel Consumed (kg) | Efficiency")
    print("-" * 55)
    
    for isp in isp_levels:
        test_vector = base_vector.copy()
        test_vector[8] = isp  # Update ISP parameter
        
        result = optimizer.evaluate_complete_mission(test_vector)
        if result["success"]:
            mission = result["mission_results"][0]
            efficiency = mission['descent_dv'] / mission['fuel_consumed']  # m/s per kg
            print(f"{isp:5.0f}   | {mission['descent_dv']:8.0f}    | "
                  f"{mission['fuel_consumed']:12.0f}     | {efficiency:6.2f}")


if __name__ == "__main__":
    print("Powered Descent Decision Vector Integration Demo")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_single_mission_optimization()
    demonstrate_multi_mission_optimization()
    demonstrate_parameter_sensitivity()
    
    print("\n" + "=" * 50)
    print("Demo complete. Decision vector successfully integrates powered descent!")