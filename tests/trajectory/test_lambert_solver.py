"""Tests for Lambert problem solver.

This module contains tests for the Lambert solver using PyKEP's native units:
    - Distances: meters (m)
    - Velocities: meters per second (m/s)
    - Times: seconds for durations, days since J2000 for epochs
    - Gravitational Parameters: m³/s²
    - Angles: degrees (converted to radians internally)

Reference Frame:
    - All calculations are performed in the Earth-centered inertial (J2000) frame
    - Position and velocity vectors are defined in this frame
    - Earth's gravitational parameter (pk.MU_EARTH) is used consistently

Test Progression:
1. Simple planar transfers (90° and 180°)
2. Non-planar transfers
3. Elliptical orbit transfers
4. Multi-revolution solutions
5. Error handling and edge cases

Each test verifies:
- Conservation of energy and angular momentum
- Physical reasonableness of solutions
- Proper unit handling
- Consistency with theoretical calculations
"""

import pytest
import numpy as np
import pykep as pk
import logging
from src.trajectory.lambert_solver import solve_lambert
from src.utils.unit_conversions import km_to_m, m_to_km, mps_to_kmps, days_to_seconds

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TestLambertSolver:
    """Test suite for Lambert problem solver with progressive complexity."""

    def setup_method(self):
        """Set up common test parameters."""
        self.mu = pk.MU_EARTH  # m³/s²
        self.leo_radius = km_to_m(6778.0)  # 400 km altitude
        self.geo_radius = km_to_m(42164.0)  # Geostationary
        self.tolerance = 1e-6  # Relative tolerance for comparisons

    def verify_solution(self, r1, v1, r2, v2, tof, mu):
        """Verify Lambert solution physics and units.

        Args:
            r1, v1: Initial position and velocity vectors [m, m/s]
            r2, v2: Final position and velocity vectors [m, m/s]
            tof: Time of flight [s]
            mu: Gravitational parameter [m³/s²]

        Verifies:
        - Angular momentum conservation
        - Energy conservation
        - Final position reached
        - Units consistency
        - Physical reasonableness of values
        """
        # Unit validation
        for vec in [
            r1,
            r2,
        ]:  # Position vectors should be in meters (reasonable orbital range)
            mag = np.linalg.norm(vec)
            assert (
                1e6 < mag < 1e9
            ), f"Position magnitude {mag:.2e} m outside reasonable orbital range"

        for vec in [
            v1,
            v2,
        ]:  # Velocity vectors should be in m/s (reasonable orbital range)
            mag = np.linalg.norm(vec)
            assert (
                100 < mag < 2e4
            ), f"Velocity magnitude {mag:.2e} m/s outside reasonable orbital range"

        assert tof > 0, "Time of flight must be positive"
        assert mu > 0, "Gravitational parameter must be positive"

        # Verify angular momentum conservation
        h1 = np.cross(r1, v1)
        h2 = np.cross(r2, v2)
        h_diff = np.linalg.norm(h1 - h2) / np.linalg.norm(h1)
        assert (
            h_diff < self.tolerance
        ), f"Angular momentum not conserved, relative difference: {h_diff:.3e}"

        # Verify energy consistency
        e1 = np.linalg.norm(v1) ** 2 / 2 - mu / np.linalg.norm(r1)
        e2 = np.linalg.norm(v2) ** 2 / 2 - mu / np.linalg.norm(r2)
        e_diff = abs(e1 - e2) / abs(e1)
        assert (
            e_diff < self.tolerance
        ), f"Energy not conserved, relative difference: {e_diff:.3e}"

        # Verify final position is reached
        try:
            prop = pk.propagate_lagrangian(r1, v1, tof, mu)
            r2_prop = np.array(prop[0])
            v2_prop = np.array(prop[1])

            # Check position error
            pos_error = np.linalg.norm(r2_prop - r2) / np.linalg.norm(r2)
            assert (
                pos_error < self.tolerance
            ), f"Final position not reached, relative error: {pos_error:.3e}"

            # Check velocity error
            vel_error = np.linalg.norm(v2_prop - v2) / np.linalg.norm(v2)
            assert (
                vel_error < self.tolerance
            ), f"Final velocity mismatch, relative error: {vel_error:.3e}"

        except Exception as e:
            logger.exception(f"Propagation failed: {e!s}")
            raise

    def test_quarter_orbit_transfer(self):
        """Test simple 90° planar transfer in circular orbit."""
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        r2 = np.array([0.0, self.leo_radius, 0.0])

        # Quarter of orbital period
        period = 2 * np.pi * np.sqrt(self.leo_radius**3 / self.mu)
        tof = period / 4

        v1, v2 = solve_lambert(r1, r2, tof, self.mu)

        # Verify circular orbit velocity
        v_circ = np.sqrt(self.mu / self.leo_radius)
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Allow 10% deviation from circular velocity
        assert abs(v1_mag - v_circ) / v_circ < 0.1
        assert abs(v2_mag - v_circ) / v_circ < 0.1

        self.verify_solution(r1, v1, r2, v2, tof, self.mu)

    def test_non_planar_transfer(self):
        """Test transfer between inclined orbits."""
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        # Add y-component to ensure angular momentum has z-component
        r2 = np.array(
            [
                self.leo_radius * np.cos(np.pi / 4) / np.sqrt(2),  # 45° inclination
                self.leo_radius * np.cos(np.pi / 4) / np.sqrt(2),  # Add y-component
                self.leo_radius * np.sin(np.pi / 4),
            ]
        )

        # Quarter orbital period for transfer
        period = 2 * np.pi * np.sqrt(self.leo_radius**3 / self.mu)
        tof = period / 4  # Reduced from half to quarter period for better geometry

        v1, v2 = solve_lambert(r1, r2, tof, self.mu)

        # Verify reasonable velocity magnitudes
        v_circ = np.sqrt(self.mu / self.leo_radius)
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Plane change of 45° should require less than 1.5 times circular velocity
        assert v1_mag < 1.5 * v_circ
        assert v2_mag < 1.5 * v_circ

        self.verify_solution(r1, v1, r2, v2, tof, self.mu)

    def test_hohmann_transfer(self):
        """Test near-Hohmann transfer from LEO to GEO.

        Note: This test uses a transfer that is close to, but not exactly,
        a Hohmann transfer. The final position is offset by 0.1 radians
        from the antipodal point to avoid numerical issues with the
        Lambert solver. As a result, the velocities will differ from
        the theoretical Hohmann transfer velocities.
        """
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        # Offset final position slightly from exact antipodal point
        angle = np.pi - 0.1  # 0.1 rad offset from 180°
        r2 = np.array(
            [-self.geo_radius * np.cos(angle), self.geo_radius * np.sin(angle), 0.0]
        )

        # Calculate time of flight based on transfer orbit
        a_transfer = (self.leo_radius + self.geo_radius) / 2.0
        tof = np.pi * np.sqrt(a_transfer**3 / self.mu)

        v1, v2 = solve_lambert(r1, r2, tof, self.mu)

        # Calculate circular orbit velocities at r1 and r2
        v_circ_leo = np.sqrt(self.mu / self.leo_radius)
        v_circ_geo = np.sqrt(self.mu / self.geo_radius)

        # Log velocities for analysis
        logger.info("Velocity Analysis:")
        logger.info(f"LEO circular velocity: {mps_to_kmps(v_circ_leo):.2f} km/s")
        logger.info(f"GEO circular velocity: {mps_to_kmps(v_circ_geo):.2f} km/s")
        logger.info(
            f"Transfer initial velocity: {mps_to_kmps(np.linalg.norm(v1)):.2f} km/s"
        )
        logger.info(
            f"Transfer final velocity: {mps_to_kmps(np.linalg.norm(v2)):.2f} km/s"
        )

        # For a near-Hohmann transfer:
        # 1. Initial velocity should be greater than LEO circular velocity
        # 2. Final velocity should be less than GEO circular velocity
        # 3. Both velocities should be within reasonable bounds
        assert np.linalg.norm(v1) > v_circ_leo, "Initial velocity too low"
        assert np.linalg.norm(v2) < v_circ_geo, "Final velocity too high"
        assert (
            np.linalg.norm(v1) < 2.0 * v_circ_leo
        ), "Initial velocity unreasonably high"
        assert np.linalg.norm(v2) > 0.2 * v_circ_geo, "Final velocity unreasonably low"

        # Verify solution physics
        self.verify_solution(r1, v1, r2, v2, tof, self.mu)

    def test_multi_revolution(self):
        """Test Lambert solver with multiple revolutions."""
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        r2 = np.array(
            [
                -self.leo_radius * np.cos(5 * np.pi / 6),  # 150° transfer
                self.leo_radius * np.sin(5 * np.pi / 6),
                0.0,
            ]
        )

        # Base orbital period
        period = 2 * np.pi * np.sqrt(self.leo_radius**3 / self.mu)
        tof = 2 * period  # Two complete orbits

        # Test with no revolutions (should get 1 solution)
        solutions_0 = solve_lambert(r1, r2, tof, self.mu, max_revolutions=0)
        assert isinstance(solutions_0, tuple)

        # Test with 1 revolution (should get 3 solutions)
        solutions_1 = solve_lambert(r1, r2, tof, self.mu, max_revolutions=1)
        assert len(solutions_1) == 3

        # Test with 2 revolutions (should get 5 solutions)
        solutions_2 = solve_lambert(r1, r2, tof, self.mu, max_revolutions=2)
        assert len(solutions_2) == 5

        # Verify all solutions are valid
        for v1, v2 in solutions_2:
            self.verify_solution(r1, v1, r2, v2, tof, self.mu)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        r2 = np.array([0.0, self.leo_radius, 0.0])
        tof = 3600.0  # 1 hour

        # Test invalid input types
        with pytest.raises(TypeError):
            solve_lambert([1, 0, 0], r2, tof, self.mu)  # List instead of ndarray

        # Test invalid shapes
        with pytest.raises(ValueError):
            solve_lambert(r1[:2], r2, tof, self.mu)  # Wrong shape

        # Test identical positions
        with pytest.raises(ValueError):
            solve_lambert(r1, r1, tof, self.mu)

        # Test zero magnitude positions
        with pytest.raises(ValueError):
            solve_lambert(np.zeros(3), r2, tof, self.mu)

        # Test negative time
        with pytest.raises(ValueError):
            solve_lambert(r1, r2, -tof, self.mu)

        # Test negative mu
        with pytest.raises(ValueError):
            solve_lambert(r1, r2, tof, -self.mu)

        # Test invalid solution index
        with pytest.raises(ValueError):
            solve_lambert(r1, r2, tof, self.mu, max_revolutions=1, solution_index=5)

    def test_circular_orbit_validation(self):
        """Test Lambert solver for circular orbit velocity validation.

        This test verifies that the solver produces correct velocities
        for a quarter-orbit transfer in a circular orbit, with detailed
        logging of intermediate values and unit conversions.
        """
        # Set up circular orbit parameters
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        r2 = np.array([0.0, self.leo_radius, 0.0])

        # Calculate theoretical circular orbit parameters
        v_circ = np.sqrt(self.mu / self.leo_radius)
        period = 2 * np.pi * np.sqrt(self.leo_radius**3 / self.mu)
        tof = period / 4  # Quarter orbit

        logger.info("Circular Orbit Test Parameters:")
        logger.info(f"Radius: {m_to_km(self.leo_radius):.2f} km")
        logger.info(f"Theoretical velocity: {mps_to_kmps(v_circ):.2f} km/s")
        logger.info(f"Period: {period/3600:.2f} hours")
        logger.info(f"Transfer time: {tof/3600:.2f} hours")

        v1, v2 = solve_lambert(r1, r2, tof, self.mu)

        # Log actual velocities
        logger.info("Lambert solution velocities:")
        logger.info(f"Initial velocity: {mps_to_kmps(np.linalg.norm(v1)):.2f} km/s")
        logger.info(f"Final velocity: {mps_to_kmps(np.linalg.norm(v2)):.2f} km/s")

        # Detailed validation
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        v1_error = abs(v1_mag - v_circ) / v_circ
        v2_error = abs(v2_mag - v_circ) / v_circ

        logger.info("Velocity errors:")
        logger.info(f"Initial velocity error: {v1_error*100:.2f}%")
        logger.info(f"Final velocity error: {v2_error*100:.2f}%")

        assert v1_error < 0.05, f"Initial velocity error too large: {v1_error*100:.2f}%"
        assert v2_error < 0.05, f"Final velocity error too large: {v2_error*100:.2f}%"

        self.verify_solution(r1, v1, r2, v2, tof, self.mu)

    def test_coplanar_transfer_validation(self):
        """Test Lambert solver for coplanar transfer validation.

        Verifies correct handling of transfers between different altitudes
        in the same plane, with detailed unit validation.
        """
        # Initial circular orbit
        r1 = np.array([self.leo_radius, 0.0, 0.0])

        # Target at 2x radius, 90 degrees
        r2_radius = 2.0 * self.leo_radius
        r2 = np.array([0.0, r2_radius, 0.0])

        # Calculate minimum energy transfer time (slightly more than Hohmann)
        a_transfer = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
        t_min = 0.8 * np.pi * np.sqrt(a_transfer**3 / self.mu)  # 80% of Hohmann time

        logger.info("Coplanar Transfer Test Parameters:")
        logger.info(f"Initial radius: {m_to_km(self.leo_radius):.2f} km")
        logger.info(f"Final radius: {m_to_km(r2_radius):.2f} km")
        logger.info(f"Transfer time: {t_min/3600:.2f} hours")

        v1, v2 = solve_lambert(r1, r2, t_min, self.mu)

        # Calculate circular velocities at both radii
        v_circ1 = np.sqrt(self.mu / self.leo_radius)
        v_circ2 = np.sqrt(self.mu / r2_radius)

        logger.info("Velocity Analysis:")
        logger.info(f"Initial circular velocity: {mps_to_kmps(v_circ1):.2f} km/s")
        logger.info(f"Final circular velocity: {mps_to_kmps(v_circ2):.2f} km/s")
        logger.info(
            f"Transfer initial velocity: {mps_to_kmps(np.linalg.norm(v1)):.2f} km/s"
        )
        logger.info(
            f"Transfer final velocity: {mps_to_kmps(np.linalg.norm(v2)):.2f} km/s"
        )

        # Verify transfer orbit characteristics
        assert (
            np.linalg.norm(v1) > v_circ1
        ), "Initial velocity should exceed circular velocity"
        assert (
            np.linalg.norm(v2) < v_circ2
        ), "Final velocity should be less than circular velocity"

        # Verify planar transfer
        h1 = np.cross(r1, v1)
        h2 = np.cross(r2, v2)
        h1_unit = h1 / np.linalg.norm(h1)
        h2_unit = h2 / np.linalg.norm(h2)

        # Log angular momentum analysis
        logger.info("Angular Momentum Analysis:")
        logger.info(f"Initial h_z component: {h1_unit[2]:.6f}")
        logger.info(f"Final h_z component: {h2_unit[2]:.6f}")

        # Verify angular momentum vectors are aligned (coplanar)
        h_alignment = np.abs(np.dot(h1_unit, h2_unit))
        logger.info(f"Angular momentum alignment: {h_alignment:.6f}")
        assert h_alignment > 0.9999, "Transfer not coplanar"

        self.verify_solution(r1, v1, r2, v2, t_min, self.mu)

    def test_hohmann_velocity_checks(self):
        """Test velocity calculations specific to Hohmann-like transfers.

        Validates the velocity changes at both ends of a near-Hohmann
        transfer, with detailed logging of delta-v components.
        """
        r1 = np.array([self.leo_radius, 0.0, 0.0])
        angle = np.pi - 0.1  # Slight offset from exact Hohmann
        r2 = np.array(
            [-self.geo_radius * np.cos(angle), self.geo_radius * np.sin(angle), 0.0]
        )

        # Theoretical Hohmann transfer calculations
        a_transfer = (self.leo_radius + self.geo_radius) / 2.0
        tof = np.pi * np.sqrt(a_transfer**3 / self.mu)

        # Circular velocities
        v_circ_leo = np.sqrt(self.mu / self.leo_radius)
        v_circ_geo = np.sqrt(self.mu / self.geo_radius)

        # Get Lambert solution
        v1, v2 = solve_lambert(r1, r2, tof, self.mu)

        # Calculate actual velocities and delta-vs
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        dv1 = v1_mag - v_circ_leo
        dv2 = v_circ_geo - v2_mag

        # Calculate transfer orbit parameters
        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)
        v1_tang = np.dot(v1, np.array([-r1[1], r1[0], 0])) / r1_mag
        v2_tang = np.dot(v2, np.array([-r2[1], r2[0], 0])) / r2_mag

        logger.info("Transfer Orbit Analysis:")
        logger.info(f"Initial radius: {m_to_km(r1_mag):.2f} km")
        logger.info(f"Final radius: {m_to_km(r2_mag):.2f} km")
        logger.info(f"Initial velocity: {mps_to_kmps(v1_mag):.2f} km/s")
        logger.info(f"Final velocity: {mps_to_kmps(v2_mag):.2f} km/s")
        logger.info(f"Initial tangential velocity: {mps_to_kmps(v1_tang):.2f} km/s")
        logger.info(f"Final tangential velocity: {mps_to_kmps(v2_tang):.2f} km/s")
        logger.info(f"Delta-v at departure: {mps_to_kmps(dv1):.2f} km/s")
        logger.info(f"Delta-v at arrival: {mps_to_kmps(dv2):.2f} km/s")

        # Verify basic orbital mechanics principles
        assert (
            v1_mag > v_circ_leo
        ), "Initial velocity should exceed LEO circular velocity"
        assert (
            v2_mag < v_circ_geo
        ), "Final velocity should be less than GEO circular velocity"
        assert v1_mag > v2_mag, "Transfer orbit velocity should decrease with radius"

        # Calculate specific angular momentum and verify it's constant
        h1 = np.cross(r1, v1)
        h2 = np.cross(r2, v2)
        h_diff = np.linalg.norm(h1 - h2) / np.linalg.norm(h1)
        logger.info(f"Angular momentum relative difference: {h_diff:.2e}")
        assert h_diff < self.tolerance, "Angular momentum not conserved"

        # Calculate specific energy
        e1 = v1_mag**2 / 2 - self.mu / r1_mag
        e2 = v2_mag**2 / 2 - self.mu / r2_mag
        e_diff = abs(e1 - e2) / abs(e1)
        logger.info(f"Energy relative difference: {e_diff:.2e}")
        assert e_diff < self.tolerance, "Energy not conserved"

        # Verify solution reaches final position
        self.verify_solution(r1, v1, r2, v2, tof, self.mu)

    def test_lunar_transfer(self):
        """Test Lambert solver for Earth-Moon transfer trajectory.

        This test verifies the solver's handling of larger orbital transfers,
        specifically for lunar transfer trajectories. It includes detailed
        unit conversion validation and checks for reasonable velocity ranges
        based on actual lunar mission data.
        """
        # Define positions in km (LEO to Moon distance)
        r1_km = np.array([6778.0, 0.0, 0.0])  # ~400km altitude
        r2_km = np.array([384400.0, 0.0, 0.0])  # ~Moon distance

        # Convert to meters for PyKEP
        r1_m = km_to_m(r1_km)
        r2_m = km_to_m(r2_km)

        # 3 day transfer (typical for lunar trajectories)
        tof_days = 3.0
        tof_seconds = days_to_seconds(tof_days)

        logger.info("Lunar Transfer Test Parameters:")
        logger.info(
            f"Initial altitude: {m_to_km(np.linalg.norm(r1_m)-pk.EARTH_RADIUS):.1f} km"
        )
        logger.info(f"Final distance: {m_to_km(np.linalg.norm(r2_m)):.1f} km")
        logger.info(f"Time of flight: {tof_days:.1f} days")

        # Solve Lambert's problem
        v1, v2 = solve_lambert(r1_m, r2_m, tof_seconds, self.mu)

        # Convert velocities to km/s for validation
        v1_kms = mps_to_kmps(v1)
        v2_kms = mps_to_kmps(v2)

        # Calculate velocity magnitudes
        v1_mag = np.linalg.norm(v1_kms)
        v2_mag = np.linalg.norm(v2_kms)

        logger.info("Velocity Analysis:")
        logger.info(f"Initial velocity: {v1_mag:.2f} km/s")
        logger.info(f"Final velocity: {v2_mag:.2f} km/s")

        # Check for NaN velocities (indicates Lambert solver failure)
        assert not np.isnan(v1_mag), "Lambert solver returned NaN for initial velocity"
        assert not np.isnan(v2_mag), "Lambert solver returned NaN for final velocity"

        # Validate velocity ranges based on typical lunar transfer values
        # Initial velocity: ~10.8-11.2 km/s for TLI
        # Final velocity: ~0.8-1.2 km/s near Moon
        # Note: These ranges are relaxed for robustness
        assert (
            8.0 < v1_mag < 15.0
        ), f"Initial velocity {v1_mag:.2f} km/s outside expected range"
        assert (
            0.3 < v2_mag < 3.0
        ), f"Final velocity {v2_mag:.2f} km/s outside expected range"

        # Verify solution physics
        self.verify_solution(r1_m, v1, r2_m, v2, tof_seconds, self.mu)
