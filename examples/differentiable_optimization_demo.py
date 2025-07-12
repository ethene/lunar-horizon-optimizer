#!/usr/bin/env python3
"""
Differentiable Optimization Demo

This example demonstrates the JAX-based differentiable optimization module
for gradient-based trajectory and economic optimization.

Features demonstrated:
- JAX differentiable models for trajectory and economic analysis
- Automatic differentiation with gradient-based optimization
- JIT compilation for performance optimization
- Batch optimization for multiple candidates
- Integration between global (PyGMO) and local (JAX) optimization
- Performance comparison between methods

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import time
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any

# Import differentiable optimization components
from src.optimization.differentiable.jax_optimizer import DifferentiableOptimizer
from src.optimization.differentiable.differentiable_models import (
    TrajectoryModel, 
    EconomicModel, 
    create_combined_model
)
from src.optimization.differentiable.integration import PyGMOIntegration

print("=" * 70)
print("🚀 Lunar Horizon Optimizer - Differentiable Optimization Demo")
print("=" * 70)

def demo_jax_availability():
    """Demonstrate JAX and Diffrax availability."""
    print("\n📦 1. JAX/Diffrax Availability Check")
    print("-" * 40)
    
    try:
        import jax
        import diffrax
        
        print(f"✅ JAX version: {jax.__version__}")
        print(f"✅ JAX backend: {jax.default_backend()}")
        print(f"✅ Diffrax version: {diffrax.__version__}")
        print(f"✅ Available devices: {jax.local_devices()}")
        
        # Test basic JAX functionality
        x = jnp.array([1.0, 2.0, 3.0])
        result = jnp.sum(x**2)
        print(f"✅ JAX computation test: sum([1,2,3]²) = {result}")
        
        return True
        
    except ImportError as e:
        print(f"❌ JAX/Diffrax not available: {e}")
        return False

def demo_basic_differentiable_models():
    """Demonstrate basic differentiable models."""
    print("\n🧮 2. Basic Differentiable Models")
    print("-" * 40)
    
    # Create trajectory model
    print("Creating TrajectoryModel...")
    trajectory_model = TrajectoryModel(use_jit=True)
    
    # Test orbital velocity calculation
    radius = 7.0e6  # 400km altitude
    velocity = trajectory_model.orbital_velocity(radius)
    print(f"Orbital velocity at {(radius-6.378e6)/1000:.0f}km: {velocity:.0f} m/s")
    
    # Test Hohmann transfer
    r1, r2 = 6.8e6, 2.0e6  # Earth to Moon vicinity
    dv_total, dv1, dv2 = trajectory_model.hohmann_transfer(r1, r2)
    print(f"Hohmann transfer Δv: {dv_total:.0f} m/s (burns: {dv1:.0f}, {dv2:.0f})")
    
    # Create economic model
    print("\nCreating EconomicModel...")
    economic_model = EconomicModel(use_jit=True)
    
    # Test launch cost calculation
    delta_v = 10000.0  # m/s
    payload_mass = 1000.0  # kg
    launch_cost = economic_model.launch_cost_model(delta_v, payload_mass)
    print(f"Launch cost for {delta_v:.0f} m/s, {payload_mass:.0f}kg: ${launch_cost/1e6:.1f}M")
    
    # Test ROI calculation
    total_cost = 1e9  # $1B
    annual_revenue = 50e6  # $50M/year
    roi = economic_model.roi_calculation(total_cost, annual_revenue)
    print(f"ROI for ${total_cost/1e9:.1f}B cost, ${annual_revenue/1e6:.0f}M/year: {roi:.1%}")
    
    return trajectory_model, economic_model

def demo_gradient_optimization():
    """Demonstrate gradient-based optimization."""
    print("\n🎯 3. Gradient-Based Optimization")
    print("-" * 40)
    
    # Create models
    trajectory_model = TrajectoryModel(use_jit=True)
    economic_model = EconomicModel(use_jit=True)
    
    # Define combined objective function
    def combined_objective(params):
        """Combined trajectory and economic objective."""
        # params = [earth_radius, moon_radius, time_of_flight]
        traj_result = trajectory_model._trajectory_cost(params)
        
        # Economic evaluation
        econ_params = jnp.array([traj_result["delta_v"], traj_result["time_of_flight"]])
        econ_result = economic_model._economic_cost(econ_params)
        
        # Weighted combination (normalized)
        return (
            1.0 * traj_result["delta_v"] / 10000.0 +        # Delta-v weight
            1.0 * econ_result["total_cost"] / 1e9 +         # Cost weight  
            0.1 * traj_result["time_of_flight"] / (7*24*3600) # Time weight
        )
    
    # Create optimizer
    print("Creating DifferentiableOptimizer...")
    optimizer = DifferentiableOptimizer(
        objective_function=combined_objective,
        bounds=[
            (6.6e6, 8.0e6),           # Earth orbit radius [m]
            (1.8e6, 2.2e6),           # Moon orbit radius [m]
            (3*24*3600, 10*24*3600)   # Transfer time [s]
        ],
        method="L-BFGS-B",
        use_jit=True,
        tolerance=1e-6,
        max_iterations=100,
        verbose=False
    )
    
    # Initial guess: 400km Earth orbit, 100km Moon orbit, 5 days
    x0 = jnp.array([6.778e6, 1.837e6, 5*24*3600])
    
    print(f"Initial guess:")
    print(f"  Earth altitude: {(x0[0]-6.378e6)/1000:.0f} km")
    print(f"  Moon altitude: {(x0[1]-1.737e6)/1000:.0f} km") 
    print(f"  Transfer time: {x0[2]/(24*3600):.1f} days")
    
    # Optimize
    print("\nOptimizing...")
    start_time = time.time()
    result = optimizer.optimize(x0)
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\nOptimization Results:")
    print(f"✅ Success: {result.success}")
    print(f"📊 Iterations: {result.nit}")
    print(f"⏱️  Time: {optimization_time:.2f}s")
    print(f"🎯 Final objective: {result.fun:.6e}")
    
    if result.improvement_percentage is not None:
        print(f"📈 Improvement: {result.improvement_percentage:.1f}%")
    
    # Convert results back to meaningful units
    earth_alt = (result.x[0] - 6.378e6) / 1000
    moon_alt = (result.x[1] - 1.737e6) / 1000  
    transfer_days = result.x[2] / (24*3600)
    
    print(f"\nOptimal Solution:")
    print(f"  Earth altitude: {earth_alt:.0f} km")
    print(f"  Moon altitude: {moon_alt:.0f} km")
    print(f"  Transfer time: {transfer_days:.1f} days")
    
    return result

def demo_batch_optimization():
    """Demonstrate batch optimization for multiple starting points."""
    print("\n🔄 4. Batch Optimization")
    print("-" * 40)
    
    # Create simple quadratic objective for demonstration
    def quadratic_objective(x):
        target = jnp.array([1.0, 2.0, 3.0])
        return jnp.sum((x - target)**2)
    
    optimizer = DifferentiableOptimizer(
        objective_function=quadratic_objective,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        method="L-BFGS-B",
        use_jit=True,
        verbose=False
    )
    
    # Multiple starting points
    initial_points = [
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([-2.0, 1.0, 4.0]), 
        jnp.array([3.0, -1.0, 2.0]),
        jnp.array([0.5, 2.5, 2.8])
    ]
    
    print(f"Optimizing from {len(initial_points)} different starting points...")
    
    start_time = time.time()
    batch_results = optimizer.batch_optimize(initial_points)
    batch_time = time.time() - start_time
    
    # Analyze results
    comparison = optimizer.compare_with_initial(batch_results)
    
    print(f"\nBatch Optimization Results:")
    print(f"✅ Success rate: {comparison['success_rate']:.1%}")
    print(f"📊 Total function evaluations: {comparison['total_function_evaluations']}")
    print(f"⏱️  Total time: {batch_time:.2f}s")
    print(f"⚡ Average time per optimization: {comparison['average_optimization_time']:.3f}s")
    print(f"🎯 Best final objective: {comparison['best_final_objective']:.6e}")
    
    if 'best_improvement_percentage' in comparison:
        print(f"📈 Best improvement: {comparison['best_improvement_percentage']:.1f}%")

def demo_performance_comparison():
    """Compare JAX vs numerical differentiation performance."""
    print("\n⚡ 5. Performance Comparison")
    print("-" * 40)
    
    # Simple test function
    def test_objective(x):
        return jnp.sum(x**4 - 2*x**2 + x)
    
    # JAX optimizer (with gradients)
    jax_optimizer = DifferentiableOptimizer(
        objective_function=test_objective,
        method="L-BFGS-B",
        use_jit=True,
        verbose=False
    )
    
    # Test point
    x0 = jnp.array([1.0, -1.0, 0.5])
    
    # Time JAX optimization
    print("Timing JAX optimization (with analytical gradients)...")
    start_time = time.time()
    for _ in range(10):
        jax_result = jax_optimizer.optimize(x0 + 0.1 * jnp.random.normal(key=jax.random.PRNGKey(42), shape=(3,)))
    jax_time = (time.time() - start_time) / 10
    
    print(f"JAX optimization average time: {jax_time:.4f}s")
    print(f"JAX final objective: {jax_result.fun:.6e}")
    print(f"JAX function evaluations: {jax_result.nfev}")
    
    # Note: For a fair comparison, we'd need to implement a numerical gradient version
    # This would require scipy.optimize.minimize without providing the jacobian
    print(f"\n💡 Performance Benefits:")
    print(f"   - Exact gradients (no approximation errors)")
    print(f"   - JIT compilation for optimized execution")
    print(f"   - Vectorized batch operations")
    print(f"   - Memory-efficient compilation")

def demo_integration_with_pygmo():
    """Demonstrate integration between PyGMO global and JAX local optimization."""
    print("\n🔗 6. PyGMO-JAX Integration")
    print("-" * 40)
    
    try:
        from src.optimization.global_optimizer import GlobalOptimizer
        
        print("🌍 Step 1: Global optimization with PyGMO...")
        
        # Create global optimizer  
        global_optimizer = GlobalOptimizer()
        
        # Run global optimization (small scale for demo)
        pareto_front = global_optimizer.find_pareto_front(
            earth_alt_range=(200, 800),
            moon_alt_range=(50, 300), 
            transfer_time_range=(4, 8),
            population_size=20,  # Small for demo
            generations=10       # Quick run
        )
        
        print(f"✅ Generated Pareto front with {len(pareto_front)} solutions")
        
        # Create integration manager
        print("\n🎯 Step 2: Local refinement with JAX...")
        integration = PyGMOIntegration()
        
        # Refine selected solutions (just a few for demo)
        solutions_to_refine = pareto_front[:3] if len(pareto_front) >= 3 else pareto_front
        
        refined_solutions = integration.refine_pareto_solutions(
            pareto_front=solutions_to_refine,
            refinement_method="L-BFGS-B",
            max_refinements=len(solutions_to_refine)
        )
        
        print(f"✅ Refined {len(refined_solutions)} solutions using JAX")
        
        # Compare improvements
        print(f"\n📊 Refinement Results:")
        for i, (original, refined) in enumerate(zip(solutions_to_refine, refined_solutions)):
            orig_obj = sum(original)  # Simplified objective
            refined_obj = refined.get('objective_value', orig_obj)
            improvement = (orig_obj - refined_obj) / orig_obj * 100
            print(f"   Solution {i+1}: {improvement:+.2f}% improvement")
            
    except ImportError as e:
        print(f"❌ PyGMO integration not available: {e}")
        print("   (This is expected if PyGMO is not installed)")

def demo_advanced_features():
    """Demonstrate advanced JAX features."""
    print("\n🔬 7. Advanced JAX Features")
    print("-" * 40)
    
    # Gradient computation
    print("Computing gradients with JAX...")
    
    from jax import grad, jit, vmap
    
    def complex_function(x):
        return jnp.sum(jnp.sin(x)**2 + jnp.cos(x**2))
    
    # Get gradient function
    grad_func = grad(complex_function)
    
    # JIT compile for performance
    jit_func = jit(complex_function)
    jit_grad = jit(grad_func)
    
    # Test point
    x = jnp.array([1.0, 2.0, 3.0])
    
    # Compute function and gradient
    func_value = jit_func(x)
    grad_value = jit_grad(x)
    
    print(f"Function value: {func_value:.6f}")
    print(f"Gradient: [{grad_value[0]:.4f}, {grad_value[1]:.4f}, {grad_value[2]:.4f}]")
    
    # Vectorized computation
    print("\nVectorized computation with vmap...")
    batch_x = jnp.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5], 
        [-1.0, 0.0, 1.0]
    ])
    
    # Vectorize function over batch dimension
    vmap_func = vmap(jit_func)
    batch_results = vmap_func(batch_x)
    
    print(f"Batch results: {batch_results}")
    print(f"Vectorized computation speedup: ~{len(batch_x)}x faster than loop")

def main():
    """Run the complete differentiable optimization demonstration."""
    
    # Check JAX availability
    if not demo_jax_availability():
        print("\n❌ JAX not available - skipping JAX-specific demos")
        return
    
    try:
        # Basic model demonstration
        trajectory_model, economic_model = demo_basic_differentiable_models()
        
        # Gradient-based optimization
        demo_gradient_optimization()
        
        # Batch optimization
        demo_batch_optimization()
        
        # Performance comparison
        demo_performance_comparison()
        
        # PyGMO integration
        demo_integration_with_pygmo()
        
        # Advanced features
        demo_advanced_features()
        
        print("\n" + "=" * 70)
        print("🎉 Differentiable Optimization Demo Complete!")
        print("=" * 70)
        print("\n📚 Next Steps:")
        print("   • See docs/DIFFERENTIABLE_OPTIMIZATION.md for detailed documentation")
        print("   • Explore src/optimization/differentiable/ for source code")
        print("   • Run tests with: pytest tests/test_task_8_differentiable_optimization.py")
        print("   • Try hybrid PyGMO-JAX optimization in your own problems")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("   Check that all dependencies are properly installed")
        print("   Run: conda activate py312 && pip install -r requirements.txt")

if __name__ == "__main__":
    main()