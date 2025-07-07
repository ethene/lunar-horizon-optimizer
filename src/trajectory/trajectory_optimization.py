"""Trajectory optimization module for Task 3 completion.

This module provides advanced trajectory optimization algorithms
for Earth-Moon transfers, including multi-objective optimization
and constraint handling.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
import logging

from .constants import PhysicalConstants as PC
from .lunar_transfer import LunarTransfer
from .transfer_window_analysis import TransferWindow, TrajectoryWindowAnalyzer
from .nbody_dynamics import enhanced_trajectory_propagation

# Configure logging
logger = logging.getLogger(__name__)


class TrajectoryOptimizer:
    """Advanced trajectory optimization for Earth-Moon transfers."""
    
    def __init__(self,
                 min_earth_alt: float = 200,
                 max_earth_alt: float = 1000,
                 min_moon_alt: float = 50,
                 max_moon_alt: float = 500):
        """Initialize trajectory optimizer.
        
        Args:
            min_earth_alt: Minimum Earth orbit altitude [km]
            max_earth_alt: Maximum Earth orbit altitude [km] 
            min_moon_alt: Minimum Moon orbit altitude [km]
            max_moon_alt: Maximum Moon orbit altitude [km]
        """
        self.min_earth_alt = min_earth_alt
        self.max_earth_alt = max_earth_alt
        self.min_moon_alt = min_moon_alt
        self.max_moon_alt = max_moon_alt
        
        self.lunar_transfer = LunarTransfer(
            min_earth_alt, max_earth_alt, min_moon_alt, max_moon_alt
        )
        
        logger.info("Initialized TrajectoryOptimizer for multi-objective optimization")
    
    def optimize_single_objective(self,
                                 epoch: float,
                                 objective: str = 'delta_v',
                                 bounds: Dict[str, Tuple[float, float]] = None,
                                 method: str = 'differential_evolution') -> Dict[str, any]:
        """Optimize trajectory for a single objective.
        
        Args:
            epoch: Launch epoch [days since J2000]
            objective: Optimization objective ('delta_v', 'time', 'c3_energy')
            bounds: Parameter bounds {'earth_alt': (min, max), 'moon_alt': (min, max), 'time': (min, max)}
            method: Optimization method ('differential_evolution', 'minimize')
            
        Returns:
            Dictionary with optimization results
        """
        if bounds is None:
            bounds = {
                'earth_alt': (self.min_earth_alt, self.max_earth_alt),
                'moon_alt': (self.min_moon_alt, self.max_moon_alt),
                'transfer_time': (3.0, 7.0)
            }
        
        logger.info(f"Optimizing trajectory for {objective} objective using {method}")
        
        def objective_function(params):
            """Objective function for optimization."""
            earth_alt, moon_alt, transfer_time = params
            
            try:
                trajectory, total_dv = self.lunar_transfer.generate_transfer(
                    epoch=epoch,
                    earth_orbit_alt=earth_alt,
                    moon_orbit_alt=moon_alt,
                    transfer_time=transfer_time,
                    max_revolutions=0
                )
                
                if objective == 'delta_v':
                    return total_dv
                elif objective == 'time':
                    return transfer_time * 86400  # Convert to seconds
                elif objective == 'c3_energy':
                    # Approximate C3 calculation
                    r_park = PC.EARTH_RADIUS + earth_alt * 1000
                    v_park = np.sqrt(PC.EARTH_MU / r_park)
                    return (total_dv / 1000)**2  # Simplified C3
                else:
                    return total_dv  # Default to delta-v
                    
            except Exception as e:
                logger.debug(f"Objective evaluation failed for params {params}: {e}")
                return 1e6  # Penalty for infeasible solutions
        
        # Set up optimization bounds
        opt_bounds = [
            bounds['earth_alt'],
            bounds['moon_alt'], 
            bounds['transfer_time']
        ]
        
        # Perform optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds=opt_bounds,
                seed=42,
                maxiter=100,
                popsize=15
            )
        else:
            # Use initial guess at center of bounds
            x0 = [np.mean(b) for b in opt_bounds]
            result = minimize(
                objective_function,
                x0=x0,
                bounds=opt_bounds,
                method='L-BFGS-B'
            )
        
        # Extract optimal parameters
        optimal_earth_alt, optimal_moon_alt, optimal_transfer_time = result.x
        
        # Generate optimal trajectory
        optimal_trajectory, optimal_dv = self.lunar_transfer.generate_transfer(
            epoch=epoch,
            earth_orbit_alt=optimal_earth_alt,
            moon_orbit_alt=optimal_moon_alt,
            transfer_time=optimal_transfer_time,
            max_revolutions=0
        )
        
        optimization_result = {
            'success': result.success,
            'objective_value': result.fun,
            'optimal_parameters': {
                'earth_orbit_alt': optimal_earth_alt,
                'moon_orbit_alt': optimal_moon_alt,
                'transfer_time': optimal_transfer_time
            },
            'optimal_trajectory': optimal_trajectory,
            'total_delta_v': optimal_dv,
            'optimization_info': {
                'method': method,
                'iterations': getattr(result, 'nit', None),
                'function_evaluations': getattr(result, 'nfev', None)
            }
        }
        
        logger.info(f"Optimization completed: {objective} = {result.fun:.2f}")
        return optimization_result
    
    def pareto_front_analysis(self,
                            epoch: float,
                            objectives: List[str] = None,
                            num_solutions: int = 50) -> List[Dict[str, any]]:
        """Generate Pareto front for multi-objective optimization.
        
        Args:
            epoch: Launch epoch [days since J2000]
            objectives: List of objectives ['delta_v', 'time', 'c3_energy']
            num_solutions: Number of Pareto solutions to generate
            
        Returns:
            List of Pareto-optimal solutions
        """
        if objectives is None:
            objectives = ['delta_v', 'time']
        
        logger.info(f"Generating Pareto front for objectives: {objectives}")
        
        # Generate random parameter combinations
        np.random.seed(42)
        n_samples = num_solutions * 10  # Oversample to ensure diversity
        
        earth_alts = np.random.uniform(self.min_earth_alt, self.max_earth_alt, n_samples)
        moon_alts = np.random.uniform(self.min_moon_alt, self.max_moon_alt, n_samples)
        transfer_times = np.random.uniform(3.0, 7.0, n_samples)
        
        solutions = []
        
        for i in range(n_samples):
            try:
                trajectory, total_dv = self.lunar_transfer.generate_transfer(
                    epoch=epoch,
                    earth_orbit_alt=earth_alts[i],
                    moon_orbit_alt=moon_alts[i],
                    transfer_time=transfer_times[i],
                    max_revolutions=0
                )
                
                # Calculate objective values
                obj_values = {}
                if 'delta_v' in objectives:
                    obj_values['delta_v'] = total_dv
                if 'time' in objectives:
                    obj_values['time'] = transfer_times[i] * 86400
                if 'c3_energy' in objectives:
                    obj_values['c3_energy'] = (total_dv / 1000)**2
                
                solution = {
                    'parameters': {
                        'earth_orbit_alt': earth_alts[i],
                        'moon_orbit_alt': moon_alts[i],
                        'transfer_time': transfer_times[i]
                    },
                    'objectives': obj_values,
                    'trajectory': trajectory,
                    'total_delta_v': total_dv
                }
                
                solutions.append(solution)
                
            except Exception as e:
                logger.debug(f"Failed to generate solution {i}: {e}")
                continue
        
        # Filter for Pareto-optimal solutions
        pareto_solutions = self._find_pareto_front(solutions, objectives)
        
        logger.info(f"Generated {len(pareto_solutions)} Pareto-optimal solutions")
        return pareto_solutions[:num_solutions]
    
    def optimize_with_constraints(self,
                                epoch: float,
                                constraints: Dict[str, float] = None,
                                objective: str = 'delta_v') -> Dict[str, any]:
        """Optimize trajectory with constraints.
        
        Args:
            epoch: Launch epoch [days since J2000]
            constraints: Constraint dictionary {'max_delta_v': 5000, 'max_time': 5.0, 'min_c3': None}
            objective: Primary objective to minimize
            
        Returns:
            Constrained optimization result
        """
        if constraints is None:
            constraints = {'max_delta_v': 10000}  # Default: max 10 km/s delta-v
        
        logger.info(f"Optimizing {objective} with constraints: {constraints}")
        
        def constrained_objective(params):
            """Objective function with constraint penalties."""
            earth_alt, moon_alt, transfer_time = params
            
            try:
                trajectory, total_dv = self.lunar_transfer.generate_transfer(
                    epoch=epoch,
                    earth_orbit_alt=earth_alt,
                    moon_orbit_alt=moon_alt,
                    transfer_time=transfer_time,
                    max_revolutions=0
                )
                
                # Base objective value
                if objective == 'delta_v':
                    obj_value = total_dv
                elif objective == 'time':
                    obj_value = transfer_time * 86400
                else:
                    obj_value = total_dv
                
                # Apply constraint penalties
                penalty = 0
                
                if 'max_delta_v' in constraints and total_dv > constraints['max_delta_v']:
                    penalty += (total_dv - constraints['max_delta_v']) * 10
                
                if 'max_time' in constraints and transfer_time > constraints['max_time']:
                    penalty += (transfer_time - constraints['max_time']) * 86400 * 100
                
                if 'min_c3' in constraints:
                    c3 = (total_dv / 1000)**2
                    if c3 < constraints['min_c3']:
                        penalty += (constraints['min_c3'] - c3) * 1000
                
                return obj_value + penalty
                
            except Exception as e:
                return 1e6  # Large penalty for infeasible solutions
        
        # Optimization bounds
        bounds = [
            (self.min_earth_alt, self.max_earth_alt),
            (self.min_moon_alt, self.max_moon_alt),
            (3.0, 7.0)
        ]
        
        # Perform constrained optimization
        result = differential_evolution(
            constrained_objective,
            bounds=bounds,
            seed=42,
            maxiter=150,
            popsize=20
        )
        
        # Generate final trajectory
        optimal_earth_alt, optimal_moon_alt, optimal_transfer_time = result.x
        optimal_trajectory, optimal_dv = self.lunar_transfer.generate_transfer(
            epoch=epoch,
            earth_orbit_alt=optimal_earth_alt,
            moon_orbit_alt=optimal_moon_alt,
            transfer_time=optimal_transfer_time,
            max_revolutions=0
        )
        
        return {
            'success': result.success,
            'objective_value': result.fun,
            'constraints': constraints,
            'optimal_parameters': {
                'earth_orbit_alt': optimal_earth_alt,
                'moon_orbit_alt': optimal_moon_alt,
                'transfer_time': optimal_transfer_time
            },
            'optimal_trajectory': optimal_trajectory,
            'total_delta_v': optimal_dv,
            'constraint_satisfaction': self._check_constraints(optimal_dv, optimal_transfer_time, constraints)
        }
    
    def _find_pareto_front(self, solutions: List[Dict], objectives: List[str]) -> List[Dict]:
        """Find Pareto-optimal solutions from a set of solutions."""
        pareto_solutions = []
        
        for i, sol1 in enumerate(solutions):
            is_pareto = True
            
            for j, sol2 in enumerate(solutions):
                if i == j:
                    continue
                
                # Check if sol2 dominates sol1
                dominates = True
                for obj in objectives:
                    if sol2['objectives'][obj] >= sol1['objectives'][obj]:
                        dominates = False
                        break
                
                if dominates:
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_solutions.append(sol1)
        
        return pareto_solutions
    
    def _check_constraints(self, delta_v: float, transfer_time: float, constraints: Dict[str, float]) -> Dict[str, bool]:
        """Check if solution satisfies constraints."""
        satisfaction = {}
        
        if 'max_delta_v' in constraints:
            satisfaction['max_delta_v'] = delta_v <= constraints['max_delta_v']
        
        if 'max_time' in constraints:
            satisfaction['max_time'] = transfer_time <= constraints['max_time']
        
        if 'min_c3' in constraints:
            c3 = (delta_v / 1000)**2
            satisfaction['min_c3'] = c3 >= constraints['min_c3']
        
        return satisfaction


def optimize_trajectory_parameters(epoch: float,
                                 optimization_type: str = 'single_objective',
                                 **kwargs) -> Dict[str, any]:
    """Convenience function for trajectory optimization.
    
    Args:
        epoch: Launch epoch [days since J2000]
        optimization_type: 'single_objective', 'pareto_front', or 'constrained'
        **kwargs: Additional parameters for specific optimization type
        
    Returns:
        Optimization results
    """
    optimizer = TrajectoryOptimizer()
    
    if optimization_type == 'single_objective':
        return optimizer.optimize_single_objective(epoch, **kwargs)
    elif optimization_type == 'pareto_front':
        return optimizer.pareto_front_analysis(epoch, **kwargs)
    elif optimization_type == 'constrained':
        return optimizer.optimize_with_constraints(epoch, **kwargs)
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")


def batch_trajectory_optimization(epochs: List[float],
                                objective: str = 'delta_v',
                                max_workers: int = 4) -> List[Dict[str, any]]:
    """Perform batch optimization for multiple epochs.
    
    Args:
        epochs: List of launch epochs [days since J2000]
        objective: Optimization objective
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of optimization results for each epoch
    """
    optimizer = TrajectoryOptimizer()
    results = []
    
    logger.info(f"Performing batch optimization for {len(epochs)} epochs")
    
    for epoch in epochs:
        try:
            result = optimizer.optimize_single_objective(epoch, objective=objective)
            result['epoch'] = epoch
            results.append(result)
        except Exception as e:
            logger.error(f"Optimization failed for epoch {epoch}: {e}")
            results.append({'epoch': epoch, 'success': False, 'error': str(e)})
    
    return results