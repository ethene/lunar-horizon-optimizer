"""
JAX Differentiable Optimization Demonstration

This module demonstrates the complete JAX optimization pipeline by integrating
trajectory and economic models with the DifferentiableOptimizer.

Features:
- End-to-end optimization workflow
- Integration of trajectory and economic models
- Gradient-based improvement of initial solutions
- Performance comparison with baseline results
- Real-world parameter optimization

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0
"""

import time
import numpy as np
from typing import Dict

# JAX imports
try:
    import jax.numpy as jnp
    from jax import grad, jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Local imports
from .differentiable_models import TrajectoryModel, EconomicModel, create_combined_model
from .jax_optimizer import DifferentiableOptimizer


class OptimizationDemonstration:
    """
    Demonstration of JAX-based differentiable optimization for lunar missions.

    This class showcases the complete optimization pipeline from initial
    parameter guess through gradient-based refinement to final solution.
    """

    def __init__(self, use_jit: bool = True, verbose: bool = True):
        """
        Initialize the optimization demonstration.

        Args:
            use_jit: Whether to use JIT compilation
            verbose: Whether to print detailed progress
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for optimization demonstration")

        self.use_jit = use_jit
        self.verbose = verbose

        # Initialize models
        self.trajectory_model = TrajectoryModel(use_jit=use_jit)
        self.economic_model = EconomicModel(use_jit=use_jit)

        # Create combined model for optimization
        self.combined_model = create_combined_model(
            self.trajectory_model,
            self.economic_model,
            weights={"delta_v": 0.4, "cost": 0.4, "time": 0.2},
        )

        # Setup optimizer
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup the differentiable optimizer with appropriate configuration."""
        # Parameter bounds for trajectory optimization
        # [r1 (Earth departure altitude), r2 (Lunar orbit altitude), tof (time of flight)]
        self.bounds = [
            (6.578e6, 6.978e6),  # r1: 200-600 km above Earth (6378 + 200-600)
            (1.837e6, 2.137e6),  # r2: 100-400 km above Moon (1737 + 100-400)
            (3.0 * 24 * 3600, 10.0 * 24 * 3600),  # tof: 3-10 days
        ]

        self.optimizer = DifferentiableOptimizer(
            objective_function=self.combined_model,
            bounds=self.bounds,
            method="L-BFGS-B",
            use_jit=self.use_jit,
            tolerance=1e-6,
            max_iterations=100,
            verbose=self.verbose,
        )

    def generate_initial_guess(self, suboptimal: bool = True) -> jnp.ndarray:
        """
        Generate a reasonable initial guess for optimization.

        Args:
            suboptimal: If True, generate a deliberately suboptimal guess to show optimization

        Returns:
            Initial parameter vector [r1, r2, tof]
        """
        if suboptimal:
            # Deliberately suboptimal mission parameters to demonstrate optimization
            r1_earth = 6.878e6  # 500 km altitude (higher than optimal)
            r2_moon = 2.037e6  # 300 km altitude lunar orbit (higher than optimal)
            tof_days = 6.0  # 6 day transfer (longer than optimal)
        else:
            # Default mission parameters
            r1_earth = 6.778e6  # 400 km altitude Earth orbit
            r2_moon = 1.937e6  # 200 km altitude lunar orbit
            tof_days = 4.5  # 4.5 day transfer

        initial_params = jnp.array([r1_earth, r2_moon, tof_days * 24 * 3600])

        if self.verbose:
            print(f"Initial guess ({'suboptimal' if suboptimal else 'default'}):")
            print(f"  Earth departure altitude: {(r1_earth - 6.378e6) / 1000:.1f} km")
            print(f"  Lunar orbit altitude: {(r2_moon - 1.737e6) / 1000:.1f} km")
            print(f"  Time of flight: {tof_days:.1f} days")

        return initial_params

    def evaluate_initial_solution(self, params: jnp.ndarray) -> Dict[str, float]:
        """
        Evaluate the initial solution before optimization.

        Args:
            params: Initial parameter vector

        Returns:
            Dictionary with initial solution metrics
        """
        # Evaluate trajectory
        traj_result = self.trajectory_model.evaluate_trajectory(params)

        # Evaluate economics
        econ_params = jnp.array([traj_result["delta_v"], traj_result["time"]])
        econ_result = self.economic_model.evaluate_economics(econ_params)

        # Calculate combined objective
        combined_obj = float(self.combined_model(params))

        results = {
            "combined_objective": combined_obj,
            "delta_v": traj_result["delta_v"],
            "time_of_flight_days": traj_result["time"] / (24 * 3600),
            "total_cost_millions": econ_result["total_cost"] / 1e6,
            "npv_millions": econ_result["npv"] / 1e6,
            "roi": econ_result["roi"],
        }

        if self.verbose:
            print("\nInitial solution evaluation:")
            print(f"  Combined objective: {combined_obj:.6f}")
            print(f"  Delta-v: {traj_result['delta_v']:.1f} m/s")
            print(f"  Time of flight: {results['time_of_flight_days']:.2f} days")
            print(f"  Total cost: ${results['total_cost_millions']:.1f}M")
            print(f"  NPV: ${results['npv_millions']:.1f}M")
            print(f"  ROI: {results['roi']:.3f}")

        return results

    def run_optimization(self, initial_params: jnp.ndarray) -> Dict[str, any]:
        """
        Run the complete optimization process.

        Args:
            initial_params: Starting parameter values

        Returns:
            Dictionary containing optimization results and analysis
        """
        if self.verbose:
            print("\n🚀 Starting JAX differentiable optimization...")

        start_time = time.time()

        # Run optimization
        opt_result = self.optimizer.optimize(initial_params)

        optimization_time = time.time() - start_time

        if self.verbose:
            print(f"\n✅ Optimization completed in {optimization_time:.2f}s")
            print(f"  Success: {opt_result.success}")
            print(f"  Iterations: {opt_result.nit}")
            print(f"  Function evaluations: {opt_result.nfev}")
            if opt_result.improvement_percentage is not None:
                print(f"  Improvement: {opt_result.improvement_percentage:.2f}%")

        return {
            "optimization_result": opt_result,
            "optimization_time": optimization_time,
            "optimized_params": opt_result.x,
        }

    def evaluate_optimized_solution(
        self, optimized_params: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the optimized solution.

        Args:
            optimized_params: Optimized parameter vector

        Returns:
            Dictionary with optimized solution metrics
        """
        # Convert to JAX array
        params = jnp.array(optimized_params)

        # Evaluate trajectory
        traj_result = self.trajectory_model.evaluate_trajectory(params)

        # Evaluate economics
        econ_params = jnp.array([traj_result["delta_v"], traj_result["time"]])
        econ_result = self.economic_model.evaluate_economics(econ_params)

        # Calculate combined objective
        combined_obj = float(self.combined_model(params))

        results = {
            "combined_objective": combined_obj,
            "delta_v": traj_result["delta_v"],
            "time_of_flight_days": traj_result["time"] / (24 * 3600),
            "total_cost_millions": econ_result["total_cost"] / 1e6,
            "npv_millions": econ_result["npv"] / 1e6,
            "roi": econ_result["roi"],
            # Physical parameters
            "earth_altitude_km": (optimized_params[0] - 6.378e6) / 1000,
            "lunar_altitude_km": (optimized_params[1] - 1.737e6) / 1000,
        }

        if self.verbose:
            print("\nOptimized solution evaluation:")
            print(f"  Combined objective: {combined_obj:.6f}")
            print(f"  Delta-v: {traj_result['delta_v']:.1f} m/s")
            print(f"  Time of flight: {results['time_of_flight_days']:.2f} days")
            print(f"  Total cost: ${results['total_cost_millions']:.1f}M")
            print(f"  NPV: ${results['npv_millions']:.1f}M")
            print(f"  ROI: {results['roi']:.3f}")
            print(f"  Earth departure altitude: {results['earth_altitude_km']:.1f} km")
            print(f"  Lunar orbit altitude: {results['lunar_altitude_km']:.1f} km")

        return results

    def compare_solutions(
        self, initial_results: Dict[str, float], optimized_results: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare initial and optimized solutions.

        Args:
            initial_results: Results from initial solution
            optimized_results: Results from optimized solution

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}

        for key in initial_results:
            if key in optimized_results:
                initial_val = initial_results[key]
                optimized_val = optimized_results[key]

                if initial_val != 0:
                    improvement_pct = (
                        100 * (initial_val - optimized_val) / abs(initial_val)
                    )
                else:
                    improvement_pct = 0.0

                comparison[f"{key}_improvement_pct"] = improvement_pct
                comparison[f"{key}_absolute_change"] = optimized_val - initial_val

        if self.verbose:
            print("\n📊 Solution Comparison:")
            print(
                f"  Objective improvement: {comparison.get('combined_objective_improvement_pct', 0):.2f}%"
            )
            print(
                f"  Delta-v change: {comparison.get('delta_v_absolute_change', 0):.1f} m/s"
            )
            print(
                f"  Cost change: ${comparison.get('total_cost_millions_absolute_change', 0):.1f}M"
            )
            print(
                f"  NPV change: ${comparison.get('npv_millions_absolute_change', 0):.1f}M"
            )
            print(f"  ROI change: {comparison.get('roi_absolute_change', 0):.3f}")

        return comparison

    def run_complete_demonstration(self) -> Dict[str, any]:
        """
        Run the complete optimization demonstration.

        Returns:
            Complete results dictionary with all analysis
        """
        if self.verbose:
            print("=" * 60)
            print("🌙 JAX Differentiable Optimization Demonstration")
            print("=" * 60)

        # Step 1: Generate initial guess (suboptimal to show optimization)
        initial_params = self.generate_initial_guess(suboptimal=True)

        # Step 2: Evaluate initial solution
        initial_results = self.evaluate_initial_solution(initial_params)

        # Step 3: Run optimization
        optimization_results = self.run_optimization(initial_params)

        # Step 4: Evaluate optimized solution
        optimized_results = self.evaluate_optimized_solution(
            optimization_results["optimized_params"]
        )

        # Step 5: Compare solutions
        comparison = self.compare_solutions(initial_results, optimized_results)

        # Compile complete results
        complete_results = {
            "initial_params": initial_params,
            "initial_results": initial_results,
            "optimization_results": optimization_results,
            "optimized_results": optimized_results,
            "comparison": comparison,
            "demonstration_success": optimization_results[
                "optimization_result"
            ].success,
        }

        if self.verbose:
            success_icon = "✅" if complete_results["demonstration_success"] else "❌"
            print(f"\n{success_icon} Demonstration completed!")
            print(f"  Overall success: {complete_results['demonstration_success']}")

        return complete_results


def run_quick_demo() -> bool:
    """
    Run a quick demonstration of the JAX optimization pipeline.

    Returns:
        True if demonstration completed successfully
    """
    try:
        # Create demonstration instance
        demo = OptimizationDemonstration(use_jit=True, verbose=True)

        # Run complete demonstration
        results = demo.run_complete_demonstration()

        return results["demonstration_success"]

    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the demonstration when script is executed directly
    print("Running JAX Differentiable Optimization Demonstration...")
    success = run_quick_demo()
    print(f"\nDemo result: {'SUCCESS' if success else 'FAILED'}")
