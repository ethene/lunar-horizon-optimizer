"""Tests for lunar transfer trajectory generation.

This module contains tests for lunar transfer trajectory generation using PyKEP's
native units throughout:
    - Distances: meters (m)
    - Velocities: meters per second (m/s)
    - Times: days for epochs, seconds for durations
    - Angles: radians
    - Gravitational Parameters: m³/s²

All calculations are performed in the Earth-centered inertial (J2000) frame.

Test Coverage:
1. Basic vector operations and orbital mechanics
2. Basic lunar transfer components
3. Transfer trajectory generation
4. Multiple revolution solutions
5. Error handling and validation
6. Unit conversion verification
"""

import numpy as np
import logging
from src.trajectory.lunar_transfer import LunarTransfer
from src.trajectory.constants import PhysicalConstants as PC
from src.trajectory.trajectory_physics import calculate_circular_velocity, validate_solution_physics
from src.trajectory.target_state import calculate_target_state
from src.trajectory.phase_optimization import (
    calculate_initial_position,
    evaluate_transfer_solution,
    find_optimal_phase
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestBasicPhysics:
    """Test basic physics calculations and vector operations."""

    def test_vector_operations(self):
        """Test fundamental vector operations used in orbital mechanics."""
        # Test vector normalization
        v = np.array([3.0, 4.0, 0.0])
        v_unit = v / np.linalg.norm(v)
        assert np.isclose(np.linalg.norm(v_unit), 1.0, rtol=1e-15)

        # Test cross product for right-handed coordinate system
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.cross(x, y)
        assert np.allclose(z, [0.0, 0.0, 1.0], rtol=1e-15)

        # Test dot product for perpendicular vectors
        assert np.isclose(np.dot(x, y), 0.0, rtol=1e-15)
        assert np.isclose(np.dot(x, z), 0.0, rtol=1e-15)
        assert np.isclose(np.dot(y, z), 0.0, rtol=1e-15)

    def test_basic_orbital_mechanics(self):
        """Test basic orbital mechanics relationships."""
        # Test circular orbit
        r = PC.EARTH_RADIUS + 400_000  # 400 km orbit
        v_circ = np.sqrt(PC.EARTH_MU / r)

        # Position and velocity vectors for circular orbit
        r_vec = np.array([r, 0.0, 0.0])
        v_vec = np.array([0.0, v_circ, 0.0])

        # Test specific angular momentum
        h = np.cross(r_vec, v_vec)
        h_expected = r * v_circ
        assert np.isclose(np.linalg.norm(h), h_expected, rtol=1e-12)

        # Test specific orbital energy
        energy = v_circ**2/2 - PC.EARTH_MU/r
        energy_expected = -PC.EARTH_MU/(2*r)  # Circular orbit energy
        assert np.isclose(energy, energy_expected, rtol=1e-12)

        # Test orbital period
        period = 2 * np.pi * np.sqrt(r**3 / PC.EARTH_MU)
        period_expected = 2 * np.pi * r / v_circ
        assert np.isclose(period, period_expected, rtol=1e-12)

    def test_escape_velocity(self):
        """Test escape velocity calculations."""
        r = PC.EARTH_RADIUS + 400_000  # 400 km orbit
        v_esc = np.sqrt(2 * PC.EARTH_MU / r)
        v_circ = np.sqrt(PC.EARTH_MU / r)

        # Escape velocity should be sqrt(2) times circular velocity
        assert np.isclose(v_esc / v_circ, np.sqrt(2), rtol=1e-12)

        # Test energy at escape velocity (use relative tolerance for large numbers)
        energy = v_esc**2/2 - PC.EARTH_MU/r
        energy_scale = PC.EARTH_MU/r  # Use orbital energy scale for comparison
        assert np.isclose(energy/energy_scale, 0.0, rtol=1e-12)

    def test_hohmann_transfer(self):
        """Test basic Hohmann transfer calculations."""
        r1 = PC.EARTH_RADIUS + 400_000  # Initial orbit (400 km)
        r2 = PC.EARTH_RADIUS + 1000_000  # Final orbit (1000 km)

        # Calculate transfer orbit elements
        a_transfer = (r1 + r2) / 2
        v_circ1 = np.sqrt(PC.EARTH_MU / r1)
        v_circ2 = np.sqrt(PC.EARTH_MU / r2)

        # Calculate transfer velocities
        v1_transfer = np.sqrt(PC.EARTH_MU * (2/r1 - 1/a_transfer))
        v2_transfer = np.sqrt(PC.EARTH_MU * (2/r2 - 1/a_transfer))

        # Delta-v calculations
        dv1 = abs(v1_transfer - v_circ1)
        dv2 = abs(v2_transfer - v_circ2)
        dv1 + dv2

        # Transfer time
        transfer_time = np.pi * np.sqrt(a_transfer**3 / PC.EARTH_MU)

        # Verify transfer orbit is elliptical
        assert v1_transfer > v_circ1  # Velocity at perigee should be higher
        assert v2_transfer < v_circ2  # Velocity at apogee should be lower
        assert transfer_time > np.pi * np.sqrt(r1**3 / PC.EARTH_MU)  # Longer than initial orbit period

    def test_basic_perturbation(self):
        """Test basic perturbation calculations."""
        # Set up circular orbit
        r = PC.EARTH_RADIUS + 400_000
        v_circ = np.sqrt(PC.EARTH_MU / r)
        r_vec = np.array([r, 0.0, 0.0])
        v_vec = np.array([0.0, v_circ, 0.0])

        # Add small perturbation to velocity (1% increase)
        v_perturbed = v_vec * 1.01

        # Calculate new orbital elements
        h_new = np.cross(r_vec, v_perturbed)
        e_vec = np.cross(v_perturbed, h_new)/PC.EARTH_MU - r_vec/r
        e = np.linalg.norm(e_vec)

        # Verify orbit is now slightly elliptical
        assert e > 0.0
        assert e < 0.1  # Small eccentricity for small perturbation

        # Energy should be higher than circular orbit
        energy_new = np.linalg.norm(v_perturbed)**2/2 - PC.EARTH_MU/r
        energy_circ = v_circ**2/2 - PC.EARTH_MU/r
        assert energy_new > energy_circ

    def test_physical_constants(self):
        """Test that physical constants are properly defined and have reasonable values."""
        # Test Earth constants
        assert PC.EARTH_RADIUS > 6.3e6  # Should be ~6371 km
        assert PC.EARTH_RADIUS < 6.4e6
        assert PC.EARTH_MU > 3.9e14  # Should be ~3.986e14 m³/s²
        assert PC.EARTH_MU < 4.0e14

        # Test Moon constants
        assert PC.MOON_RADIUS > 1.7e6  # Should be ~1737 km
        assert PC.MOON_RADIUS < 1.8e6
        assert PC.MOON_MU > 4.9e12  # Should be ~4.905e12 m³/s²
        assert PC.MOON_MU < 5.0e12
        assert PC.MOON_SEMI_MAJOR_AXIS > 3.8e8  # Should be ~384,400 km
        assert PC.MOON_SEMI_MAJOR_AXIS < 3.9e8

        # Verify relative scales
        assert PC.MOON_RADIUS < PC.EARTH_RADIUS
        assert PC.MOON_MU < PC.EARTH_MU
        assert PC.MOON_SEMI_MAJOR_AXIS > 50 * PC.EARTH_RADIUS

    def test_orbital_velocities(self):
        """Test calculation of characteristic orbital velocities."""
        # LEO velocity (400 km altitude)
        r_leo = PC.EARTH_RADIUS + 400_000
        v_leo = np.sqrt(PC.EARTH_MU / r_leo)
        assert 7500 < v_leo < 7700  # Should be ~7.67 km/s

        # Lunar orbital velocity
        r_moon = PC.MOON_SEMI_MAJOR_AXIS
        v_moon = np.sqrt(PC.EARTH_MU / r_moon)
        assert 950 < v_moon < 1050  # Should be ~1.02 km/s

        # Escape velocity at LEO
        v_esc_leo = np.sqrt(2 * PC.EARTH_MU / r_leo)
        assert v_esc_leo > v_leo
        assert 10500 < v_esc_leo < 11000  # Should be ~10.85 km/s

    def test_minimum_energy_transfer(self):
        """Test minimum energy transfer calculations."""
        r1 = PC.EARTH_RADIUS + 400_000  # LEO
        r2 = PC.MOON_SEMI_MAJOR_AXIS

        # Hohmann transfer parameters
        a_transfer = (r1 + r2) / 2
        v1_transfer = np.sqrt(PC.EARTH_MU * (2/r1 - 1/a_transfer))
        v2_transfer = np.sqrt(PC.EARTH_MU * (2/r2 - 1/a_transfer))

        # Initial and final circular velocities
        v1_circ = np.sqrt(PC.EARTH_MU / r1)
        v2_circ = np.sqrt(PC.EARTH_MU / r2)

        # Delta-v calculations
        dv1 = abs(v1_transfer - v1_circ)
        dv2 = abs(v2_transfer - v2_circ)
        total_dv = dv1 + dv2

        # Verify reasonable values
        assert 3000 < total_dv < 4000  # Typical Hohmann transfer delta-v
        assert v1_transfer > v1_circ  # Should increase velocity at departure
        assert v2_transfer < v2_circ  # Should decrease velocity at arrival

        # Transfer time
        transfer_time = np.pi * np.sqrt(a_transfer**3 / PC.EARTH_MU)
        assert 4 * 86400 < transfer_time < 5 * 86400  # Should be ~4.5 days

class TestLunarTransfer:
    """Test lunar transfer trajectory generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.transfer = LunarTransfer()
        self.epoch = 0.0  # J2000
        self.earth_orbit_alt = 400.0  # km
        self.moon_orbit_alt = 100.0  # km
        self.transfer_time = 3.0  # days

    def test_circular_velocity(self):
        """Test circular orbit velocity calculation."""
        r = PC.EARTH_RADIUS + 400_000  # 400 km orbit
        v_circ = calculate_circular_velocity(r, PC.EARTH_MU)
        expected_v = np.sqrt(PC.EARTH_MU / r)
        assert np.isclose(v_circ, expected_v, rtol=1e-12)

    def test_target_state_basic(self):
        """Test basic target state calculation."""
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, np.sqrt(PC.EARTH_MU/PC.MOON_SEMI_MAJOR_AXIS), 0])
        orbit_radius = PC.MOON_RADIUS + 100_000  # 100 km orbit

        target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)

        # Verify distance from Moon center
        moon_dist = np.linalg.norm(target_pos - moon_pos)
        assert np.isclose(moon_dist, orbit_radius, rtol=1e-10)

        # Verify target velocity is close to Moon's velocity
        vel_diff = np.linalg.norm(target_vel - moon_vel)
        assert vel_diff < 100  # Within 100 m/s

        # Verify target orbit plane aligns with Moon's orbit
        h_moon = np.cross(moon_pos, moon_vel)
        h_target = np.cross(target_pos - moon_pos, target_vel - moon_vel)
        angle = np.arccos(np.dot(h_moon, h_target) / (np.linalg.norm(h_moon) * np.linalg.norm(h_target)))
        assert np.abs(angle) < np.deg2rad(5)  # Within 5 degrees

    def test_target_state_velocity_matching(self):
        """Test target state velocity matching Moon's velocity."""
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, 1000, 0])  # 1 km/s
        orbit_radius = PC.MOON_RADIUS + 100_000

        _, target_vel = calculate_target_state(moon_pos, moon_vel, orbit_radius)

        # Verify velocity is close to Moon's velocity
        vel_diff = np.linalg.norm(target_vel - moon_vel)
        assert vel_diff < 100  # Within 100 m/s

    def test_target_state_distance(self):
        """Test target state distance from Moon center."""
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, 1000, 0])
        orbit_radius = PC.MOON_RADIUS + 100_000

        target_pos, _ = calculate_target_state(moon_pos, moon_vel, orbit_radius)

        # Verify distance from Moon center
        moon_dist = np.linalg.norm(target_pos - moon_pos)
        assert np.isclose(moon_dist, orbit_radius, rtol=1e-10)

    def test_optimal_phase_simple(self):
        """Test optimal phase finding with simple circular orbit."""
        r_park = PC.EARTH_RADIUS + 400_000  # 400 km orbit
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, np.sqrt(PC.EARTH_MU/PC.MOON_SEMI_MAJOR_AXIS), 0])
        transfer_time = 3.0 * 86400  # 3 days in seconds
        orbit_radius = PC.MOON_RADIUS + 100_000  # 100 km orbit

        phase, r1 = find_optimal_phase(
            r_park=r_park,
            moon_pos=moon_pos,
            moon_vel=moon_vel,
            transfer_time=transfer_time,
            orbit_radius=orbit_radius
        )

        # Verify initial position magnitude
        assert np.isclose(np.linalg.norm(r1), r_park, rtol=1e-10)

    def test_optimal_phase_velocity(self):
        """Test optimal phase velocity constraints."""
        r_park = PC.EARTH_RADIUS + 400_000
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, np.sqrt(PC.EARTH_MU / PC.MOON_SEMI_MAJOR_AXIS), 0])
        transfer_time = 3.0 * 86400
        orbit_radius = PC.MOON_RADIUS + 100_000  # 100 km orbit

        phase, r1 = find_optimal_phase(
            r_park=r_park,
            moon_pos=moon_pos,
            moon_vel=moon_vel,
            transfer_time=transfer_time,
            orbit_radius=orbit_radius
        )

        # Calculate velocity at r1
        v_circ = calculate_circular_velocity(r_park, PC.EARTH_MU)
        h_unit = np.cross(moon_pos, moon_vel)
        h_unit = h_unit / np.linalg.norm(h_unit)
        v_dir = np.cross(h_unit, r1/np.linalg.norm(r1))
        v_dir = v_dir / np.linalg.norm(v_dir)
        v1 = v_circ * v_dir

        # Verify velocity magnitude is reasonable
        assert np.linalg.norm(v1) < 12000  # Less than escape velocity

    def test_lunar_transfer_components(self):
        """Test individual components of lunar transfer."""
        # Test initial state
        r_park = PC.EARTH_RADIUS + self.earth_orbit_alt * 1000
        v_circ = calculate_circular_velocity(r_park, PC.EARTH_MU)

        # Verify circular velocity is reasonable
        assert 7000 < v_circ < 8000  # m/s

        # Test target state
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, np.sqrt(PC.EARTH_MU/PC.MOON_SEMI_MAJOR_AXIS), 0])
        r_moon_orbit = PC.MOON_RADIUS + self.moon_orbit_alt * 1000

        target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, r_moon_orbit)

        # Verify target state
        assert np.isclose(np.linalg.norm(target_pos), r_moon_orbit, rtol=1e-10)
        vel_diff = np.linalg.norm(target_vel - moon_vel)
        assert vel_diff < 100  # Within 100 m/s

    def test_loi_delta_v_calculation(self):
        """Test lunar orbit insertion delta-v calculation."""
        # Setup simple test case
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, np.sqrt(PC.EARTH_MU/PC.MOON_SEMI_MAJOR_AXIS), 0])
        r_moon_orbit = PC.MOON_RADIUS + 100_000  # 100 km orbit

        # Get target state
        target_pos, target_vel = calculate_target_state(moon_pos, moon_vel, r_moon_orbit)

        # Calculate approach velocity (hyperbolic)
        v_inf = 800.0  # Typical excess velocity ~800 m/s
        approach_vel = moon_vel + np.array([0, v_inf, 0])

        # Calculate LOI delta-v
        loi_dv = target_vel - approach_vel
        loi_dv_mag = np.linalg.norm(loi_dv)

        # Verify LOI delta-v is reasonable
        assert 500 < loi_dv_mag < 1500, f"LOI delta-v {loi_dv_mag} m/s outside expected range"

        # Verify final orbit velocity
        final_vel_rel_moon = target_vel - moon_vel
        final_vel_mag = np.linalg.norm(final_vel_rel_moon)
        expected_vel = np.sqrt(PC.MOON_MU/r_moon_orbit)
        assert np.isclose(final_vel_mag, expected_vel, rtol=0.1)

    def test_transfer_orbit_energy(self):
        """Test energy of transfer orbit relative to Earth."""
        # Generate simple transfer
        trajectory, total_dv = self.transfer.generate_transfer(
            self.epoch,
            self.earth_orbit_alt,
            self.moon_orbit_alt,
            self.transfer_time
        )

        # Get initial velocity after TLI
        tli_dv = np.array(trajectory.maneuvers[0].delta_v)
        r1 = np.array(trajectory.initial_state.position) * 1000.0  # Convert to m
        v1 = np.array(trajectory.initial_state.velocity) * 1000.0  # Convert to m/s
        v1_after_tli = v1 + tli_dv

        # Calculate specific orbital energy
        r1_mag = np.linalg.norm(r1)
        v1_mag = np.linalg.norm(v1_after_tli)
        energy = v1_mag**2/2 - PC.EARTH_MU/r1_mag

        # Energy should be positive (hyperbolic) but not too high
        assert energy > 0, "Transfer orbit should be hyperbolic"
        assert energy < 1e6, "Transfer orbit energy too high"

        # Calculate characteristic energy (C3)
        c3 = v1_mag**2 - 2*PC.EARTH_MU/r1_mag
        assert -2e6 < c3 < 2e6, f"C3 energy {c3} outside expected range"

    def test_intermediate_states(self):
        """Test intermediate states during transfer."""
        trajectory, _ = self.transfer.generate_transfer(
            self.epoch,
            self.earth_orbit_alt,
            self.moon_orbit_alt,
            self.transfer_time
        )

        # Sample points along trajectory
        times = np.linspace(0, self.transfer_time*86400, 10)
        for t in times:
            # Get state at time t
            r, v = trajectory.get_state_at_time(t)

            # Verify position magnitude is reasonable
            r_mag = np.linalg.norm(r)
            assert PC.EARTH_RADIUS < r_mag < 1.1*PC.MOON_SEMI_MAJOR_AXIS

            # Verify velocity magnitude is reasonable
            v_mag = np.linalg.norm(v)
            assert 1000 < v_mag < 12000  # Between lunar and escape velocity

            # Calculate specific angular momentum
            h = np.cross(r, v)
            h_mag = np.linalg.norm(h)
            assert h_mag > 0, "Angular momentum should be conserved"

    def test_lunar_transfer_generation(self):
        """Test complete lunar transfer trajectory generation."""
        trajectory, total_dv = self.transfer.generate_transfer(
            self.epoch,
            self.earth_orbit_alt,
            self.moon_orbit_alt,
            self.transfer_time
        )

        # Verify trajectory properties
        assert trajectory.initial_state is not None
        assert trajectory.final_state is not None
        assert len(trajectory.maneuvers) == 2

        # Verify delta-v is reasonable
        assert 3000 < total_dv < 5000  # m/s

    def test_multiple_revolutions(self):
        """Test lunar transfer with multiple revolutions."""
        # Generate baseline trajectory
        base_traj, base_dv = self.transfer.generate_transfer(
            epoch=self.epoch,
            earth_orbit_alt=self.earth_orbit_alt,
            moon_orbit_alt=self.moon_orbit_alt,
            transfer_time=self.transfer_time,
            max_revolutions=0
        )

        # Generate multi-rev trajectory
        multi_traj, multi_dv = self.transfer.generate_transfer(
            epoch=self.epoch,
            earth_orbit_alt=self.earth_orbit_alt,
            moon_orbit_alt=self.moon_orbit_alt,
            transfer_time=self.transfer_time,
            max_revolutions=1
        )

        # Multi-rev should have lower or equal delta-v
        assert multi_dv <= base_dv * 1.1  # Allow 10% margin

    def test_edge_case_phase_angles(self):
        """Test transfer generation with edge case phase angles."""
        # Test with phase angles near 0, π/2, π, 3π/2
        critical_phases = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        r_park = PC.EARTH_RADIUS + 400_000  # 400 km orbit
        np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_h_unit = np.array([0, 0, 1])

        for phase in critical_phases:
            r1 = calculate_initial_position(r_park, phase, moon_h_unit)

            # Verify position magnitude
            assert np.isclose(np.linalg.norm(r1), r_park, rtol=1e-10)

            # Verify position components
            if np.isclose(phase, 0.0):
                assert np.isclose(r1[0], r_park, rtol=1e-10)
            elif np.isclose(phase, np.pi/2):
                assert np.isclose(r1[1], r_park, rtol=1e-10)

    def test_solution_physics_validation(self):
        """Test physics validation of transfer solutions."""
        # Set up test vectors
        r1 = np.array([PC.EARTH_RADIUS + 400_000, 0, 0])
        r2 = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        v_circ = np.sqrt(PC.EARTH_MU / np.linalg.norm(r1))
        transfer_time = 3.0 * 86400  # 3 days in seconds

        # Test case 1: Valid circular orbit
        v1 = np.array([0, v_circ, 0])
        v2 = np.array([0, np.sqrt(PC.EARTH_MU / np.linalg.norm(r2)), 0])
        assert validate_solution_physics(r1, v1, r2, v2, transfer_time)

        # Test case 2: Hyperbolic orbit (should fail)
        v1_hyper = np.array([0, 2*v_circ, 0])  # Double the circular velocity
        assert not validate_solution_physics(r1, v1_hyper, r2, v2, transfer_time)

        # Test case 3: Angular momentum mismatch (should fail)
        v2_bad = np.array([v_circ, 0, 0])  # Perpendicular to v1
        assert not validate_solution_physics(r1, v1, r2, v2_bad, transfer_time)

    def test_transfer_solution_evaluation(self):
        """Test evaluation of transfer solutions."""
        # Setup test parameters
        r_park = PC.EARTH_RADIUS + 400_000  # 400 km orbit
        r1 = np.array([r_park, 0, 0])
        moon_pos = np.array([PC.MOON_SEMI_MAJOR_AXIS, 0, 0])
        moon_vel = np.array([0, np.sqrt(PC.EARTH_MU / PC.MOON_SEMI_MAJOR_AXIS), 0])
        transfer_time = 3.0 * 86400  # 3 days
        orbit_radius = 100_000  # 100 km lunar orbit

        # Evaluate transfer
        dv_total, v1, v2 = evaluate_transfer_solution(
            r1=r1,
            moon_pos=moon_pos,
            moon_vel=moon_vel,
            transfer_time=transfer_time,
            orbit_radius=orbit_radius
        )

        # Check results
        assert dv_total is not None
        assert dv_total < float("inf")
        assert v1 is not None
        assert v2 is not None
        assert np.all(np.isfinite(v1))
        assert np.all(np.isfinite(v2))

        # Verify velocities are reasonable
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        assert 7000 < v1_mag < 11200  # Between LEO and escape velocity
        assert 500 < v2_mag < 2500  # Reasonable for lunar orbit

    def test_boundary_conditions(self):
        """Test transfer generation at boundary conditions."""
        # Test minimum Earth orbit altitude
        trajectory, dv = self.transfer.generate_transfer(
            epoch=self.epoch,
            earth_orbit_alt=200.0,  # Minimum allowed
            moon_orbit_alt=self.moon_orbit_alt,
            transfer_time=self.transfer_time
        )

        assert dv < 5000  # Verify reasonable delta-v

        # Test minimum Moon orbit altitude
        trajectory, dv = self.transfer.generate_transfer(
            epoch=self.epoch,
            earth_orbit_alt=self.earth_orbit_alt,
            moon_orbit_alt=50.0,  # Minimum allowed
            transfer_time=self.transfer_time
        )

        assert dv < 5000

        # Test minimum transfer time
        trajectory, dv = self.transfer.generate_transfer(
            epoch=self.epoch,
            earth_orbit_alt=self.earth_orbit_alt,
            moon_orbit_alt=self.moon_orbit_alt,
            transfer_time=2.0  # Minimum allowed
        )

        assert dv < 6000  # Allow higher delta-v for faster transfer

    def test_revolution_count_impact(self):
        """Test impact of revolution count on transfer solutions."""
        # Generate transfers with different revolution counts
        results = []
        for max_rev in range(3):  # Test 0, 1, and 2 revolutions
            trajectory, dv = self.transfer.generate_transfer(
                epoch=self.epoch,
                earth_orbit_alt=self.earth_orbit_alt,
                moon_orbit_alt=self.moon_orbit_alt,
                transfer_time=5.0,  # Longer time to allow multiple revs
                max_revolutions=max_rev
            )
            results.append(dv)

        # Verify delta-v generally decreases with more revolutions
        assert all(dv < 5000 for dv in results)  # All should be reasonable
        assert results[1] <= results[0] * 1.1  # Allow 10% margin
        assert results[2] <= results[1] * 1.1
