"""Basic smoke tests to verify the Python environment and dependencies."""

import pytest
import scipy
import pykep
import pygmo
import jax
import jax.numpy as jnp
import diffrax
import plotly.graph_objects as go
from poliastro.bodies import Earth, Moon

def test_scipy_version():
    """Verify SciPy version is compatible with PyKEP."""
    assert scipy.__version__ == "1.13.1", "SciPy version must be 1.13.1 for PyKEP compatibility"

def test_jax_configuration():
    """Verify JAX is properly configured."""
    # Print JAX device info for debugging
    print(f"\nJAX is using device: {jax.devices()}")

    # Simple JAX computation test
    x = jnp.array([1.0, 2.0, 3.0])
    y = jax.grad(lambda x: jnp.sum(x**2))(x)
    assert y.shape == (3,), "JAX gradient computation failed"

def test_pykep_basic():
    """Verify PyKEP basic functionality."""
    # Create a simple planet object
    earth = pykep.planet.jpl_lp("earth")
    assert earth.name == "earth", "PyKEP planet creation failed"

def test_pygmo_basic():
    """Verify PyGMO basic functionality."""
    # Simple optimization problem
    prob = pygmo.problem(pygmo.rosenbrock())
    algo = pygmo.algorithm(pygmo.de(gen=1))  # Just 1 generation for testing
    pop = pygmo.population(prob, size=10)
    pop = algo.evolve(pop)
    assert len(pop) == 10, "PyGMO optimization test failed"

def test_diffrax_basic():
    """Verify Diffrax basic functionality."""
    # Simple ODE: dy/dt = -y
    def f(t, y, args):
        return -y

    term = diffrax.ODETerm(f)
    solver = diffrax.Euler()
    t0, t1 = 0.0, 1.0
    y0 = jnp.array([1.0])

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=y0
    )
    assert solution.ts.shape[0] > 0, "Diffrax ODE solution failed"

def test_plotly_basic():
    """Verify Plotly basic functionality."""
    fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    assert isinstance(fig, go.Figure), "Plotly figure creation failed"

def test_poliastro_basic():
    """Verify Poliastro basic functionality."""
    earth = Earth
    moon = Moon
    assert earth.name == "Earth", "Poliastro Earth object creation failed"
    assert moon.name == "Moon", "Poliastro Moon object creation failed"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
