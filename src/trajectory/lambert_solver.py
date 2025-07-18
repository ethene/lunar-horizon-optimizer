"""Lambert problem solver for trajectory calculations.

This module provides functions to solve Lambert's problem - finding a transfer orbit
between two position vectors with a specified time of flight. All calculations use
PyKEP's native units.

Unit Conventions (PyKEP Native):
    - Positions: meters (m)
    - Velocities: meters per second (m/s)
    - Time: seconds
    - Gravitational Parameter: m³/s²
"""

from typing import Union

import numpy as np
import pykep as pk
from numpy.typing import NDArray

# Type aliases for cleaner annotations
Float64Array = NDArray[np.float64]
VelocityPair = tuple[Float64Array, Float64Array]


def get_num_solutions(max_revolutions: int) -> int:
    """Calculate number of solutions for given maximum revolutions.

    Args:
        max_revolutions: Maximum number of revolutions

    Returns
    -------
        Number of possible solutions
    """
    if max_revolutions == 0:
        return 1
    return 2 * max_revolutions + 1


def solve_lambert(
    r1: Float64Array,
    r2: Float64Array,
    tof: float,
    mu: float,
    max_revolutions: int = 0,
    prograde: bool = True,
    solution_index: int | None = None,
) -> Union[VelocityPair, list[VelocityPair]]:
    """Solve Lambert's problem using PyKEP.

    Args:
        r1: Initial position vector [x, y, z] in meters
        r2: Final position vector [x, y, z] in meters
        tof: Time of flight in seconds
        mu: Gravitational parameter in m³/s²
        max_revolutions: Maximum number of revolutions (default: 0)
        prograde: If True, seek prograde solution (default: True)
        solution_index: Index of solution to return (default: None, returns minimum energy solution)

    Returns
    -------
        If max_revolutions == 0 or solution_index is specified:
            Tuple containing:
                - Initial velocity vector [vx, vy, vz] in m/s
                - Final velocity vector [vx, vy, vz] in m/s
        If max_revolutions > 0 and solution_index is None:
            List of tuples, each containing:
                - Initial velocity vector [vx, vy, vz] in m/s
                - Final velocity vector [vx, vy, vz] in m/s

    Raises
    ------
        ValueError: If no solution found, input parameters invalid, or solution_index out of range
        TypeError: If input vectors are not numpy arrays or have wrong shape
    """
    # Validate all inputs
    _validate_lambert_inputs(r1, r2, tof, mu)

    # Prepare inputs for PyKEP
    r1_prep, r2_prep = _prepare_position_vectors(r1, r2)

    # Solve Lambert problem
    lambert = _solve_lambert_problem(
        r1_prep, r2_prep, tof, mu, max_revolutions, prograde
    )

    # Extract and return solutions
    return _extract_lambert_solutions(lambert, max_revolutions, solution_index)


def get_all_solutions(
    r1: Float64Array,
    r2: Float64Array,
    tof: float,
    mu: float,
    max_revolutions: int = 0,
    prograde: bool = True,
) -> list[VelocityPair]:
    """Get all possible solutions for the Lambert problem.

    Args:
        r1: Initial position vector [x, y, z] in meters
        r2: Final position vector [x, y, z] in meters
        tof: Time of flight in seconds
        mu: Gravitational parameter in m³/s²
        max_revolutions: Maximum number of revolutions (default: 0)
        prograde: If True, seek prograde solution (default: True)

    Returns
    -------
        List of tuples, each containing:
            - Initial velocity vector [vx, vy, vz] in m/s
            - Final velocity vector [vx, vy, vz] in m/s
    """
    solutions = []
    num_solutions = get_num_solutions(max_revolutions)

    for i in range(num_solutions):
        v1, v2 = solve_lambert(r1, r2, tof, mu, max_revolutions, prograde, i)
        solutions.append((v1, v2))

    return solutions


def _validate_lambert_inputs(
    r1: Float64Array, r2: Float64Array, tof: float, mu: float
) -> None:
    """Validate inputs for Lambert problem solver."""
    # Type and shape validation
    if not isinstance(r1, np.ndarray) or not isinstance(r2, np.ndarray):
        msg = "Position vectors must be numpy arrays"
        raise TypeError(msg)
    if r1.shape != (3,) or r2.shape != (3,):
        msg = "Position vectors must have shape (3,)"
        raise ValueError(msg)

    # Parameter validation
    if np.allclose(r1, r2):
        msg = "Initial and final positions are the same"
        raise ValueError(msg)
    if tof <= 0:
        msg = "Time of flight must be positive"
        raise ValueError(msg)
    if mu <= 0:
        msg = "Gravitational parameter must be positive"
        raise ValueError(msg)

    # Check for zero magnitude position vectors
    if np.linalg.norm(r1) < 1e-10 or np.linalg.norm(r2) < 1e-10:
        msg = "Position vectors cannot have zero magnitude"
        raise ValueError(msg)


def _prepare_position_vectors(
    r1: Float64Array, r2: Float64Array
) -> tuple[Float64Array, Float64Array]:
    """Prepare position vectors for PyKEP format."""
    return np.array(r1, dtype=float), np.array(r2, dtype=float)


def _solve_lambert_problem(
    r1: Float64Array,
    r2: Float64Array,
    tof: float,
    mu: float,
    max_revolutions: int,
    prograde: bool,
) -> pk.lambert_problem:
    """Create and solve Lambert problem using PyKEP."""
    try:
        return pk.lambert_problem(
            r1=tuple(r1),
            r2=tuple(r2),
            tof=float(tof),
            mu=float(mu),
            max_revs=max_revolutions,
            cw=(not prograde),
        )
    except RuntimeError as e:
        msg = f"Failed to solve Lambert's problem: {e!s}"
        raise ValueError(msg) from e
    except Exception as e:
        msg = f"Unexpected error in PyKEP Lambert solver: {e!s}"
        raise ValueError(msg) from e


def _extract_lambert_solutions(
    lambert: pk.lambert_problem, max_revolutions: int, solution_index: int | None
) -> Union[VelocityPair, list[VelocityPair]]:
    """Extract velocity solutions from Lambert problem."""
    # Get velocity lists
    try:
        v1_list = lambert.get_v1()
        v2_list = lambert.get_v2()
    except Exception as e:
        msg = f"Failed to extract velocity vectors: {e!s}"
        raise ValueError(msg) from e

    if not v1_list or not v2_list:
        msg = "No valid solutions found"
        raise ValueError(msg)

    # Return specific solution if requested
    if solution_index is not None:
        if solution_index >= len(v1_list):
            msg = f"Solution index {solution_index} out of range (0-{len(v1_list)-1})"
            raise ValueError(msg)
        return np.array(v1_list[solution_index]), np.array(v2_list[solution_index])

    # Return single solution for zero revolutions
    if max_revolutions == 0:
        return np.array(v1_list[0]), np.array(v2_list[0])

    # Return all solutions for multiple revolutions
    return [
        (np.array(v1), np.array(v2)) for v1, v2 in zip(v1_list, v2_list, strict=False)
    ]
