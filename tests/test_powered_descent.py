#!/opt/anaconda3/envs/py312/bin/python
"""Unit tests for powered descent trajectory function."""

import jax.numpy as jnp
import numpy as np
import pytest
from src.trajectory.continuous_thrust import (
    powered_descent,
    create_jitted_powered_descent,
)


class TestPoweredDescent:
    """Test powered descent trajectory function."""

    def test_powered_descent_basic_functionality(self):
        """Test basic powered descent functionality."""
        # Initial state: 100 km circular lunar orbit
        moon_radius = 1.737e6  # m
        orbit_altitude = 100e3  # m
        orbit_radius = moon_radius + orbit_altitude

        # Circular orbital velocity
        mu_moon = 4.9048695e12  # m³/s²
        orbital_velocity = np.sqrt(mu_moon / orbit_radius)

        # Initial state
        start_state = jnp.array(
            [
                orbit_radius,
                0.0,
                0.0,  # Position [m]
                0.0,
                orbital_velocity,
                0.0,  # Velocity [m/s]
                5000.0,  # Mass [kg]
            ]
        )

        # Spacecraft parameters
        thrust = 5000.0  # N
        isp = 300.0  # s
        burn_time = 60.0  # s
        steps = 20  # Integration steps

        # Run powered descent
        states, times, total_delta_v = powered_descent(
            start_state, thrust, isp, burn_time, steps
        )

        # Verify results
        assert states.shape == (steps + 1, 7)
        assert times.shape == (steps + 1,)
        assert total_delta_v > 0
        assert np.isfinite(total_delta_v)

        # Check physical constraints
        final_mass = states[-1, 6]
        initial_mass = start_state[6]
        assert final_mass < initial_mass  # Mass should decrease
        assert final_mass > 0  # Mass should remain positive

        # Check velocity reduction (braking effect)
        final_velocity = np.linalg.norm(states[-1, 3:6])
        initial_velocity = np.linalg.norm(start_state[3:6])
        assert final_velocity < initial_velocity

    def test_powered_descent_gradient_compatibility(self):
        """Test that powered descent is compatible with JAX gradients."""
        import jax

        # Simple test state
        start_state = jnp.array([1.837e6, 0.0, 0.0, 0.0, 1633.0, 0.0, 2000.0])

        def objective(thrust):
            """Test objective function for gradient computation."""
            states, times, delta_v = powered_descent(
                start_state, thrust, 300.0, 30.0, 10
            )
            return jnp.linalg.norm(states[-1, :3])  # Final position magnitude

        # Test gradient computation
        thrust_test = 2000.0
        grad_func = jax.grad(objective)
        gradient = grad_func(thrust_test)

        # Verify gradient is finite and reasonable
        assert np.isfinite(gradient)
        assert abs(gradient) > 1e-10  # Should have non-zero gradient

    def test_powered_descent_jit_compilation(self):
        """Test JIT compilation functionality."""
        # Create JIT-compiled version
        steps = 15
        jitted_descent = create_jitted_powered_descent(steps)

        # Test state
        start_state = jnp.array([1.837e6, 0.0, 0.0, 0.0, 1500.0, 0.0, 3000.0])

        # Run JIT-compiled version
        states, times, delta_v = jitted_descent(start_state, 3000.0, 300.0, 45.0)

        # Verify results
        assert states.shape == (steps + 1, 7)
        assert times.shape == (steps + 1,)
        assert delta_v > 0

    def test_powered_descent_physical_limits(self):
        """Test powered descent with physical edge cases."""
        # Test with very low thrust
        start_state = jnp.array([1.837e6, 0.0, 0.0, 0.0, 1600.0, 0.0, 1000.0])

        states, times, delta_v = powered_descent(
            start_state, 100.0, 300.0, 30.0, 10  # Very low thrust
        )

        # Should still produce valid results
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(times))
        assert np.isfinite(delta_v)

        # Mass should decrease monotonically
        masses = states[:, 6]
        assert np.all(np.diff(masses) <= 0)  # Non-increasing mass

    def test_powered_descent_zero_thrust(self):
        """Test powered descent with zero thrust (free fall)."""
        start_state = jnp.array([1.837e6, 0.0, 0.0, 0.0, 1600.0, 0.0, 1000.0])

        states, times, delta_v = powered_descent(
            start_state, 0.0, 300.0, 30.0, 10  # Zero thrust
        )

        # Should produce valid results
        assert np.all(np.isfinite(states))
        assert delta_v == 0.0  # No delta-v with zero thrust

        # Mass should remain constant
        masses = states[:, 6]
        assert np.allclose(masses, start_state[6])  # Constant mass


if __name__ == "__main__":
    pytest.main([__file__])
