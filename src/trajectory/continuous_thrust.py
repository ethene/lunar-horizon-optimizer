"""Minimal continuous-thrust propagator using JAX/Diffrax.

Implements Edelbaum-like planar model for low-thrust trajectory optimization.
State: [r, θ, v, m] - radius, angle, velocity, mass
Control: α - thrust angle relative to velocity vector

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import jax
import jax.numpy as jnp
import diffrax
from typing import Tuple

# Physical constants
G0 = 9.81  # Standard gravity [m/s²]
MU_EARTH = 3.986004418e14  # Earth gravitational parameter [m³/s²]


def continuous_dynamics(t: float, state: jnp.ndarray, args: Tuple) -> jnp.ndarray:
    """Edelbaum planar continuous-thrust dynamics.

    Args:
        t: Time [s]
        state: [r, θ, v, m] - radius [m], angle [rad], velocity [m/s], mass [kg]
        args: (alpha, T, Isp, mu) - thrust angle [rad], thrust [N], Isp [s], μ [m³/s²]

    Returns:
        State derivatives [ṙ, θ̇, v̇, ṁ]
    """
    r, theta, v, m = state
    alpha, T, Isp, mu = args

    # Ensure reasonable bounds to avoid numerical issues
    r = jnp.maximum(r, 1e6)  # Minimum 1000 km radius
    m = jnp.maximum(m, 1.0)  # Minimum 1 kg mass
    v = jnp.maximum(v, 100.0)  # Minimum 100 m/s velocity

    # Edelbaum equations in polar coordinates with gravitational acceleration
    gravity_accel = mu / (r * r)

    rdot = v * jnp.sin(alpha)
    thetadot = v * jnp.cos(alpha) / r
    vdot = -gravity_accel * jnp.sin(alpha) + T / m
    mdot = -T / (Isp * G0)

    return jnp.array([rdot, thetadot, vdot, mdot])


def low_thrust_transfer(
    start_state: jnp.ndarray,
    target_state: jnp.ndarray,
    T: float,
    Isp: float,
    tf_guess: float,
    alpha_constant: float = 0.0,
    mu: float = MU_EARTH,
) -> Tuple[float, jnp.ndarray]:
    """Continuous-thrust transfer with constant thrust angle.

    Args:
        start_state: [r0, θ0, v0, m0] initial state
        target_state: [rf, θf, vf, mf] target state (mf ignored)
        T: Thrust magnitude [N]
        Isp: Specific impulse [s]
        tf_guess: Transfer time estimate [s]
        alpha_constant: Constant thrust angle [rad]
        mu: Gravitational parameter [m³/s²]

    Returns:
        (delta_v_equivalent, trajectory_profile)
    """
    # Integration setup
    args = (alpha_constant, T, Isp, mu)
    term = diffrax.ODETerm(continuous_dynamics)
    solver = diffrax.Tsit5()

    # Integrate trajectory with error handling
    try:
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=tf_guess,
            dt0=tf_guess / 1000,
            y0=start_state,
            args=args,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, tf_guess, 100)),
            rtol=1e-6,
            atol=1e-8,
            max_steps=10000,
        )

        # Check for successful integration
        if solution.ys.shape[0] == 0:
            # Integration failed, return large penalty
            return 1e6, jnp.zeros((100, 4))

        # Calculate equivalent ΔV from propellant consumption
        m0, mf = start_state[3], solution.ys[-1, 3]

        # Avoid log of negative or zero mass
        if mf <= 0 or m0 <= 0 or mf >= m0:
            return 1e6, solution.ys

        delta_v_equivalent = Isp * G0 * jnp.log(m0 / mf)

        return delta_v_equivalent, solution.ys

    except Exception:
        # Return penalty values for failed integration
        return 1e6, jnp.zeros((100, 4))


# Integration with differentiable optimization framework
def optimize_thrust_angle(
    start_state: jnp.ndarray, target_radius: float, T: float, Isp: float, tf: float
) -> float:
    """Find optimal constant thrust angle to reach target radius using simple gradient descent."""

    def objective(alpha):
        _, trajectory = low_thrust_transfer(start_state, None, T, Isp, tf, alpha)
        final_radius = trajectory[-1, 0]
        return (final_radius - target_radius) ** 2

    # Simple gradient descent optimization
    alpha = 0.1  # Initial guess [rad]
    learning_rate = 0.01

    for _ in range(10):  # Simple iteration
        grad = jax.grad(objective)(alpha)
        alpha = alpha - learning_rate * grad
        alpha = jnp.clip(alpha, -jnp.pi / 4, jnp.pi / 4)  # Keep reasonable bounds

    return alpha
