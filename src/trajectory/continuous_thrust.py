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
MU_MOON = 4.9048695e12  # Moon gravitational parameter [m³/s²]


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


def powered_descent_dynamics(t: float, state: jnp.ndarray, args: Tuple) -> jnp.ndarray:
    """Powered descent dynamics in Moon-centered inertial frame.

    Args:
        t: Time [s]
        state: [x, y, z, vx, vy, vz, m] - position [m], velocity [m/s], mass [kg]
        args: (thrust, isp) - thrust magnitude [N], specific impulse [s]

    Returns:
        State derivatives [ẋ, ẏ, ż, v̇x, v̇y, v̇z, ṁ]
    """
    x, y, z, vx, vy, vz, m = state
    thrust, isp = args

    # Position vector and distance from Moon center
    r_vec = jnp.array([x, y, z])
    r = jnp.linalg.norm(r_vec)

    # Ensure minimum mass to avoid division by zero
    m = jnp.maximum(m, 1.0)

    # Gravitational acceleration (Moon-centered)
    r_safe = jnp.maximum(r, 1e3)  # Minimum 1 km radius to avoid singularity
    gravity_accel = -MU_MOON * r_vec / (r_safe**3)

    # Thrust acceleration (directed opposite to velocity for descent)
    v_vec = jnp.array([vx, vy, vz])
    v_mag = jnp.linalg.norm(v_vec)

    # Thrust direction: opposite to velocity for braking
    thrust_direction = jnp.where(
        v_mag > 1e-6,  # If velocity is significant
        -v_vec / v_mag,  # Thrust opposite to velocity
        jnp.array([0.0, 0.0, 1.0]),  # Default upward thrust if no velocity
    )

    thrust_accel = (thrust / m) * thrust_direction

    # Total acceleration
    total_accel = gravity_accel + thrust_accel

    # Mass flow rate (negative for fuel consumption)
    mdot = -thrust / (isp * G0)

    return jnp.array([vx, vy, vz, total_accel[0], total_accel[1], total_accel[2], mdot])


def powered_descent(
    start_state: jnp.ndarray, thrust: float, isp: float, burn_time: float, steps: int
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Powered descent trajectory from lunar orbit to surface.

    Models continuous-thrust landing from lunar orbit using JAX and Diffrax.
    The trajectory assumes Moon-centered inertial frame with thrust directed
    opposite to velocity vector for optimal braking.

    Args:
        start_state: JAX array [x, y, z, vx, vy, vz, m] in Moon-centered inertial frame
                    Position [m], velocity [m/s], mass [kg]
        thrust: Thrust magnitude [N]
        isp: Specific impulse [s]
        burn_time: Duration of powered descent [s]
        steps: Number of integration steps

    Returns:
        Tuple of (states, times, total_delta_v) as JAX arrays
        - states: Array of shape (steps+1, 7) with trajectory states
        - times: Array of shape (steps+1,) with time points
        - total_delta_v: Scalar representing total velocity change [m/s]
    """
    # Integration setup
    args = (thrust, isp)
    term = diffrax.ODETerm(powered_descent_dynamics)
    solver = diffrax.Tsit5()  # 5th order Runge-Kutta solver

    # Time points for solution
    times = jnp.linspace(0.0, burn_time, steps + 1)

    # Create stepsize controller for error control
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

    # Integrate the trajectory
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=burn_time,
        dt0=burn_time / 1000,  # More conservative initial step
        y0=start_state,
        args=args,
        saveat=diffrax.SaveAt(ts=times),
        stepsize_controller=stepsize_controller,
        max_steps=100000,  # More generous step limit
        throw=False,  # Don't throw errors, let us handle them
    )

    # Check if solution was successful (use simple heuristic)
    # Check if final time reached and states are finite
    success = jnp.isfinite(solution.ys[-1]).all()

    # Calculate total delta-v from mass consumption
    m0 = start_state[6]  # Initial mass
    mf = jnp.where(success, solution.ys[-1, 6], m0)  # Use initial mass if failed

    # Use rocket equation: Δv = Isp * g0 * ln(m0/mf)
    # Ensure positive masses for log calculation
    m0_safe = jnp.maximum(m0, 1.0)
    mf_safe = jnp.maximum(mf, 0.1)
    mf_safe = jnp.minimum(mf_safe, m0_safe)  # Final mass can't exceed initial

    total_delta_v = jnp.where(
        success,
        isp * G0 * jnp.log(m0_safe / mf_safe),
        0.0,  # Return 0 delta-v if integration failed
    )

    return solution.ys, times, total_delta_v


def create_jitted_powered_descent(steps: int):
    """Create a JIT-compiled version of powered_descent with fixed steps.

    Args:
        steps: Number of integration steps (must be known at compile time)

    Returns:
        JIT-compiled powered descent function
    """

    @jax.jit
    def jitted_powered_descent(
        start_state: jnp.ndarray, thrust: float, isp: float, burn_time: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """JIT-compiled powered descent with fixed number of steps."""
        return powered_descent(start_state, thrust, isp, burn_time, steps)

    return jitted_powered_descent
