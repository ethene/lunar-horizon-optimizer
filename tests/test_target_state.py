import numpy as np
import pytest
from src.trajectory.target_state import calculate_target_state
from src.trajectory.constants import PhysicalConstants as PC


def test_target_state_basic():
    """Test basic target state calculation."""
    moon_pos = np.array([384400e3, 0, 0])  # Moon at +x axis
    moon_vel = np.array([0, 1023, 0])  # Moving in +y direction
    orbit_radius = 100e3  # 100 km orbit

    target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)

    # Verify target position is at correct distance from Moon
    moon_dist = np.linalg.norm(target_pos - moon_pos)
    assert np.isclose(moon_dist, orbit_radius, rtol=1e-6)

    # Verify target velocity has correct circular orbit component
    v_circ = np.sqrt(PC.MOON_MU / orbit_radius)
    vel_diff = target_vel - moon_vel
    vel_diff_mag = np.linalg.norm(vel_diff)
    assert np.isclose(
        vel_diff_mag, v_circ, rtol=0.1
    )  # Allow 10% tolerance for capture component


def test_target_state_velocity_matching():
    """Test that target state properly matches Moon's velocity."""
    moon_pos = np.array([384400e3, 0, 0])
    moon_vel = np.array([0, 1023, 0])  # Realistic Moon velocity
    orbit_radius = 100e3

    target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)

    # Calculate velocity components
    target_radial = (target_pos - moon_pos) / orbit_radius
    orbit_normal = np.cross(moon_pos, moon_vel)
    orbit_normal = orbit_normal / np.linalg.norm(orbit_normal)

    vel_diff = target_vel - moon_vel
    vel_diff_radial = np.dot(vel_diff, target_radial)
    vel_diff_normal = np.dot(vel_diff, orbit_normal)

    # Verify radial component is -2 m/s (for capture)
    assert np.isclose(vel_diff_radial, -2.0, atol=0.1)

    # Verify negligible out-of-plane component
    assert abs(vel_diff_normal) < 0.1

    # Verify total velocity difference is close to circular orbit velocity
    v_circ = np.sqrt(PC.MOON_MU / orbit_radius)
    vel_diff_mag = np.linalg.norm(vel_diff)
    assert np.isclose(vel_diff_mag, v_circ, rtol=0.1)


def test_target_state_invalid_inputs():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        calculate_target_state([1, 2], [1, 2, 3], 100e3)  # Wrong position shape

    with pytest.raises(ValueError):
        calculate_target_state([1, 2, 3], [1, 2], 100e3)  # Wrong velocity shape


def test_physical_constants():
    """Verify physical constants are in correct ranges and relationships."""
    # Moon's orbital velocity should match Kepler's laws
    expected_moon_velocity = np.sqrt(PC.MU_EARTH / PC.MOON_ORBIT_RADIUS)
    assert np.isclose(PC.MOON_ORBITAL_VELOCITY, expected_moon_velocity, rtol=1e-6)

    # Moon's SOI should be about 1/6 of orbital radius (rough approximation)
    assert 0.15 < PC.MOON_SOI / PC.MOON_ORBIT_RADIUS < 0.20

    # Escape velocity should be sqrt(2) times circular orbit velocity at surface
    moon_surface_v_circ = np.sqrt(PC.MOON_MU / PC.MOON_RADIUS)
    assert np.isclose(
        PC.MOON_ESCAPE_VELOCITY, moon_surface_v_circ * np.sqrt(2), rtol=1e-6
    )


def test_circular_orbit_velocities():
    """Test circular orbit velocity calculations at different altitudes."""
    test_radii = [50e3, 100e3, 200e3]  # Different orbit radii
    for radius in test_radii:
        v_circ = np.sqrt(PC.MOON_MU / radius)

        # Verify orbital period matches Kepler's third law
        period = 2 * np.pi * radius / v_circ
        expected_period = 2 * np.pi * np.sqrt(radius**3 / PC.MOON_MU)
        assert np.isclose(period, expected_period, rtol=1e-6)

        # Energy should be -μ/2a for circular orbit
        specific_energy = v_circ**2 / 2 - PC.MOON_MU / radius
        expected_energy = -PC.MOON_MU / (2 * radius)
        assert np.isclose(specific_energy, expected_energy, rtol=1e-6)


def test_target_state_edge_cases():
    """Test target state calculation with edge cases."""
    moon_pos = np.array([PC.MOON_ORBIT_RADIUS, 0, 0])
    moon_vel = np.array([0, PC.MOON_ORBITAL_VELOCITY, 0])

    # Test very low orbit (20 km)
    low_orbit = 20e3
    target_pos_low, target_vel_low = calculate_target_state(
        moon_pos, moon_vel, low_orbit
    )
    assert np.isclose(np.linalg.norm(target_pos_low - moon_pos), low_orbit, rtol=1e-6)

    # Test high orbit (1000 km)
    high_orbit = 1000e3
    target_pos_high, target_vel_high = calculate_target_state(
        moon_pos, moon_vel, high_orbit
    )
    assert np.isclose(np.linalg.norm(target_pos_high - moon_pos), high_orbit, rtol=1e-6)

    # Verify higher orbit has lower velocity (v ∝ 1/√r)
    v_diff_low = np.linalg.norm(target_vel_low - moon_vel)
    v_diff_high = np.linalg.norm(target_vel_high - moon_vel)
    assert v_diff_low > v_diff_high
    assert np.isclose(
        v_diff_low / v_diff_high, np.sqrt(high_orbit / low_orbit), rtol=0.1
    )


def test_target_state_energy():
    """Test energy of the resulting orbit relative to the Moon."""
    moon_pos = np.array([PC.MOON_ORBIT_RADIUS, 0, 0])
    moon_vel = np.array([0, PC.MOON_ORBITAL_VELOCITY, 0])
    orbit_radius = 100e3

    target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)

    # Calculate velocity relative to Moon
    rel_vel = target_vel - moon_vel
    rel_pos = target_pos - moon_pos

    # Calculate specific orbital energy
    v_mag_squared = np.dot(rel_vel, rel_vel)
    r_mag = np.linalg.norm(rel_pos)
    energy = v_mag_squared / 2 - PC.MOON_MU / r_mag

    # Energy should be negative (bound orbit) and close to circular orbit energy
    expected_energy = -PC.MOON_MU / (2 * orbit_radius)
    assert energy < 0, "Orbit should be bound (negative energy)"
    assert np.isclose(energy, expected_energy, rtol=0.1)


def test_target_state_angular_momentum():
    """Test conservation of angular momentum in target orbit."""
    moon_pos = np.array([PC.MOON_ORBIT_RADIUS, 0, 0])
    moon_vel = np.array([0, PC.MOON_ORBITAL_VELOCITY, 0])
    orbit_radius = 100e3

    target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)

    # Calculate angular momentum relative to Moon
    rel_pos = target_pos - moon_pos
    rel_vel = target_vel - moon_vel
    h_vec = np.cross(rel_pos, rel_vel)
    h_mag = np.linalg.norm(h_vec)

    # For circular orbit, h = r * v_circ
    expected_h = orbit_radius * np.sqrt(PC.MOON_MU / orbit_radius)
    assert np.isclose(h_mag, expected_h, rtol=0.1)
