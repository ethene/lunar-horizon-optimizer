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

import numpy as np
import pykep as pk
from typing import Tuple, Optional, List, Union

def get_num_solutions(max_revolutions: int) -> int:
    """Calculate number of solutions for given maximum revolutions.
    
    Args:
        max_revolutions: Maximum number of revolutions
        
    Returns:
        Number of possible solutions
    """
    if max_revolutions == 0:
        return 1
    return 2 * max_revolutions + 1

def solve_lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    max_revolutions: int = 0,
    prograde: bool = True,
    solution_index: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    """Solve Lambert's problem using PyKEP.
    
    Args:
        r1: Initial position vector [x, y, z] in meters
        r2: Final position vector [x, y, z] in meters
        tof: Time of flight in seconds
        mu: Gravitational parameter in m³/s²
        max_revolutions: Maximum number of revolutions (default: 0)
        prograde: If True, seek prograde solution (default: True)
        solution_index: Index of solution to return (default: None, returns minimum energy solution)
        
    Returns:
        If max_revolutions == 0 or solution_index is specified:
            Tuple containing:
                - Initial velocity vector [vx, vy, vz] in m/s
                - Final velocity vector [vx, vy, vz] in m/s
        If max_revolutions > 0 and solution_index is None:
            List of tuples, each containing:
                - Initial velocity vector [vx, vy, vz] in m/s
                - Final velocity vector [vx, vy, vz] in m/s
            
    Raises:
        ValueError: If no solution found, input parameters invalid, or solution_index out of range
        TypeError: If input vectors are not numpy arrays or have wrong shape
    """
    # Validate input types and shapes
    if not isinstance(r1, np.ndarray) or not isinstance(r2, np.ndarray):
        raise TypeError("Position vectors must be numpy arrays")
    if r1.shape != (3,) or r2.shape != (3,):
        raise ValueError("Position vectors must have shape (3,)")
    
    # Convert inputs to PyKEP format and ensure float type
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    
    # Validate inputs
    if np.allclose(r1, r2):
        raise ValueError("Initial and final positions are the same")
    if tof <= 0:
        raise ValueError("Time of flight must be positive")
    if mu <= 0:
        raise ValueError("Gravitational parameter must be positive")
    
    # Check for zero magnitude position vectors with explicit tolerance
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    if r1_norm < 1e-10:  # 0.1 mm tolerance
        raise ValueError("Initial position vector has zero magnitude")
    if r2_norm < 1e-10:  # 0.1 mm tolerance
        raise ValueError("Final position vector has zero magnitude")
    
    # Create Lambert problem
    try:
        lambert = pk.lambert_problem(
            r1=tuple(r1),
            r2=tuple(r2),
            tof=float(tof),  # Ensure float type
            mu=float(mu),    # Ensure float type
            max_revs=max_revolutions,
            cw=(not prograde)
        )
    except RuntimeError as e:
        raise ValueError(f"Failed to solve Lambert's problem: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error in PyKEP Lambert solver: {str(e)}")
    
    # Get all solutions
    try:
        v1_list = lambert.get_v1()
        v2_list = lambert.get_v2()
    except Exception as e:
        raise ValueError(f"Failed to extract velocity vectors: {str(e)}")
    
    if not v1_list or not v2_list:
        raise ValueError("No valid solutions found")
    
    # Return single solution if requested
    if solution_index is not None:
        if solution_index >= len(v1_list):
            raise ValueError(f"Solution index {solution_index} out of range (0-{len(v1_list)-1})")
        return np.array(v1_list[solution_index]), np.array(v2_list[solution_index])
    
    # Return single solution for zero revolutions
    if max_revolutions == 0:
        return np.array(v1_list[0]), np.array(v2_list[0])
    
    # Return list of solutions for multiple revolutions
    return [(np.array(v1), np.array(v2)) for v1, v2 in zip(v1_list, v2_list)]

def get_all_solutions(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float,
    max_revolutions: int = 0,
    prograde: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Get all possible solutions for the Lambert problem.
    
    Args:
        r1: Initial position vector [x, y, z] in meters
        r2: Final position vector [x, y, z] in meters
        tof: Time of flight in seconds
        mu: Gravitational parameter in m³/s²
        max_revolutions: Maximum number of revolutions (default: 0)
        prograde: If True, seek prograde solution (default: True)
        
    Returns:
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