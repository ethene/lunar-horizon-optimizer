"""Unit tests for the TrajectoryPropagator class.

Tests the propagation of spacecraft trajectories in the Earth-Moon system,
including Earth and Moon gravity effects, energy conservation, and error handling.
"""

import pytest
import numpy as np
from src.trajectory.propagator import TrajectoryPropagator
from src.trajectory.celestial_bodies import CelestialBody
from src.trajectory.constants import PhysicalConstants as PC
from src.utils.unit_conversions import seconds_to_days

@pytest.fixture
def propagator():
    """Create a TrajectoryPropagator instance with Earth as the central body."""
    celestial = CelestialBody('EARTH', PC.MU_EARTH)  # Initialize with Earth's parameters
    return TrajectoryPropagator(celestial)

def test_moon_gravity(propagator):
    """Test that Moon's gravity affects the trajectory."""
    # Initial state in Earth orbit (circular, 200km altitude)
    r0 = np.array([PC.EARTH_RADIUS + 200e3, 0, 0])  # 200km circular orbit
    v0 = np.array([0, np.sqrt(PC.MU_EARTH/np.linalg.norm(r0)), 0])  # Circular orbit velocity
    tof = 24 * 3600  # 1 day propagation
    
    # Propagate trajectory
    r1, v1 = propagator.propagate_to_target(r0, v0, tof)
    
    # Verify that final position is different from what it would be in pure Kepler orbit
    # due to Moon's gravity (approximate check)
    r1_norm = np.linalg.norm(r1)
    assert abs(r1_norm - np.linalg.norm(r0)) > 100  # At least 100m deviation

def test_energy_conservation(propagator):
    """Test that energy is approximately conserved during propagation."""
    # Initial state (elliptical orbit)
    r0 = np.array([PC.EARTH_RADIUS + 1000e3, 0, 0])
    v0 = np.array([0, 9000, 1000])  # Non-circular orbit
    tof = 12 * 3600  # 12 hours
    
    # Propagate
    r1, v1 = propagator.propagate_to_target(r0, v0, tof)
    
    # Calculate specific energy at start and end
    e0 = np.linalg.norm(v0)**2/2 - PC.MU_EARTH/np.linalg.norm(r0)
    e1 = np.linalg.norm(v1)**2/2 - PC.MU_EARTH/np.linalg.norm(r1)
    
    # Energy should be conserved to within numerical precision
    relative_error = abs(e1 - e0)/abs(e0)
    assert relative_error < 1e-8, f"Energy error {relative_error} exceeds tolerance"

def test_invalid_inputs(propagator):
    """Test that invalid inputs raise appropriate exceptions."""
    r0 = np.array([PC.EARTH_RADIUS + 1000e3, 0, 0])
    v0 = np.array([0, 7800, 0])
    
    # Test negative time
    with pytest.raises(ValueError, match="Time of flight must be positive"):
        propagator.propagate_to_target(r0, v0, -1)
    
    # Test zero velocity
    with pytest.raises(ValueError, match="Initial velocity cannot be zero"):
        propagator.propagate_to_target(r0, np.zeros(3), 3600)
    
    # Test position below Earth surface
    r0_low = np.array([PC.EARTH_RADIUS - 1000, 0, 0])
    with pytest.raises(ValueError, match="Initial position is below Earth's surface"):
        propagator.propagate_to_target(r0_low, v0, 3600)
    
    # Test wrong vector shape
    with pytest.raises(ValueError, match="Position and velocity vectors must be 3D"):
        propagator.propagate_to_target(np.array([1, 1]), v0, 3600) 