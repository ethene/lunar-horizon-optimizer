"""Global Optimization Module using PyGMO for Task 4 completion.

This module implements multi-objective trajectory optimization using PyGMO's
NSGA-II algorithm to generate Pareto fronts balancing delta-v, time, and cost.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pygmo as pg
import logging
from datetime import datetime
import pickle
import hashlib

from trajectory.lunar_transfer import LunarTransfer
from config.costs import CostFactors
from config.models import MissionConfig
from optimization.cost_integration import CostCalculator

# Configure logging
logger = logging.getLogger(__name__)


class LunarMissionProblem:
    """PyGMO problem implementation for lunar mission optimization.
    
    This class defines the multi-objective optimization problem for lunar
    transfers, implementing the PyGMO problem interface with objectives
    for delta-v, time, and cost.
    """
    
    def __init__(self,
                 cost_factors: CostFactors = None,
                 min_earth_alt: float = 200,    # km
                 max_earth_alt: float = 1000,   # km  
                 min_moon_alt: float = 50,      # km
                 max_moon_alt: float = 500,     # km
                 min_transfer_time: float = 3.0,  # days
                 max_transfer_time: float = 10.0, # days
                 reference_epoch: float = 10000.0):  # days since J2000
        """Initialize the lunar mission optimization problem.
        
        Args:
            cost_factors: Economic cost parameters
            min_earth_alt: Minimum Earth orbit altitude [km]
            max_earth_alt: Maximum Earth orbit altitude [km]
            min_moon_alt: Minimum lunar orbit altitude [km] 
            max_moon_alt: Maximum lunar orbit altitude [km]
            min_transfer_time: Minimum transfer time [days]
            max_transfer_time: Maximum transfer time [days]
            reference_epoch: Reference epoch for calculations [days since J2000]
        """
        self.cost_factors = cost_factors or CostFactors()
        self.min_earth_alt = min_earth_alt
        self.max_earth_alt = max_earth_alt
        self.min_moon_alt = min_moon_alt
        self.max_moon_alt = max_moon_alt
        self.min_transfer_time = min_transfer_time
        self.max_transfer_time = max_transfer_time
        self.reference_epoch = reference_epoch
        
        # Initialize trajectory generator
        self.lunar_transfer = LunarTransfer(
            min_earth_alt=min_earth_alt,
            max_earth_alt=max_earth_alt,
            min_moon_alt=min_moon_alt,
            max_moon_alt=max_moon_alt
        )
        
        # Initialize cost calculator
        self.cost_calculator = CostCalculator(self.cost_factors)
        
        # Cache for expensive trajectory calculations
        self._trajectory_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized LunarMissionProblem with bounds: "
                   f"Earth [{min_earth_alt}-{max_earth_alt}] km, "
                   f"Moon [{min_moon_alt}-{max_moon_alt}] km, "
                   f"Time [{min_transfer_time}-{max_transfer_time}] days")
    
    def fitness(self, x: List[float]) -> List[float]:
        """Evaluate fitness for multi-objective optimization.
        
        Args:
            x: Decision vector [earth_alt, moon_alt, transfer_time] 
            
        Returns:
            List of objective values [delta_v, time, cost]
        """
        earth_alt, moon_alt, transfer_time = x
        
        # Create cache key for this parameter combination
        cache_key = self._create_cache_key(earth_alt, moon_alt, transfer_time)
        
        # Check cache first
        if cache_key in self._trajectory_cache:
            self._cache_hits += 1
            return self._trajectory_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            # Generate trajectory
            trajectory, total_dv = self.lunar_transfer.generate_transfer(
                epoch=self.reference_epoch,
                earth_orbit_alt=earth_alt,
                moon_orbit_alt=moon_alt,
                transfer_time=transfer_time,
                max_revolutions=0
            )
            
            # Calculate objectives
            obj1_delta_v = total_dv  # m/s
            obj2_time = transfer_time * 86400  # Convert days to seconds
            obj3_cost = self.cost_calculator.calculate_mission_cost(
                total_dv=total_dv,
                transfer_time=transfer_time,
                earth_orbit_alt=earth_alt,
                moon_orbit_alt=moon_alt
            )
            
            objectives = [obj1_delta_v, obj2_time, obj3_cost]
            
            # Cache the result
            self._trajectory_cache[cache_key] = objectives
            
            return objectives
            
        except Exception as e:
            logger.debug(f"Failed to evaluate fitness for {x}: {e}")
            # Return penalty values for infeasible solutions
            return [1e6, 1e6, 1e6]
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get optimization bounds for decision variables.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower = [self.min_earth_alt, self.min_moon_alt, self.min_transfer_time]
        upper = [self.max_earth_alt, self.max_moon_alt, self.max_transfer_time]
        return lower, upper
    
    def get_nobj(self) -> int:
        """Get number of objectives.
        
        Returns:
            Number of objectives (3: delta_v, time, cost)
        """
        return 3
    
    def get_name(self) -> str:
        """Get problem name.
        
        Returns:
            Problem identifier string
        """
        return "Lunar Mission Multi-Objective Optimization"
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get trajectory cache statistics.
        
        Returns:
            Dictionary with cache hit/miss statistics
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_evaluations': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._trajectory_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the trajectory cache."""
        self._trajectory_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Trajectory cache cleared")
    
    def _create_cache_key(self, earth_alt: float, moon_alt: float, transfer_time: float) -> str:
        """Create cache key for parameter combination.
        
        Args:
            earth_alt: Earth orbit altitude [km]
            moon_alt: Moon orbit altitude [km] 
            transfer_time: Transfer time [days]
            
        Returns:
            Cache key string
        """
        # Round to reasonable precision to improve cache hits
        key_data = f"{earth_alt:.1f}_{moon_alt:.1f}_{transfer_time:.2f}_{self.reference_epoch:.1f}"
        return hashlib.md5(key_data.encode()).hexdigest()


class GlobalOptimizer:
    """PyGMO-based global optimizer for lunar mission design.
    
    This class implements multi-objective optimization using NSGA-II to
    generate Pareto fronts for lunar transfer trajectories.
    """
    
    def __init__(self,
                 problem: LunarMissionProblem = None,
                 population_size: int = 100,
                 num_generations: int = 100,
                 seed: int = 42):
        """Initialize the global optimizer.
        
        Args:
            problem: Optimization problem instance
            population_size: NSGA-II population size
            num_generations: Number of generations to evolve
            seed: Random seed for reproducibility
        """
        self.problem = problem or LunarMissionProblem()
        self.population_size = population_size
        self.num_generations = num_generations
        self.seed = seed
        
        # Create PyGMO problem
        self.pg_problem = pg.problem(self.problem)
        
        # Initialize NSGA-II algorithm
        self.algorithm = pg.algorithm(pg.nsga2(gen=1, seed=seed))
        self.algorithm.set_verbosity(1)
        
        # Initialize population
        self.population = None
        self.optimization_history = []
        
        logger.info(f"Initialized GlobalOptimizer with NSGA-II: "
                   f"pop_size={population_size}, generations={num_generations}")
    
    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """Run multi-objective optimization.
        
        Args:
            verbose: Enable detailed logging
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting NSGA-II optimization: {self.num_generations} generations")
        
        # Initialize population
        self.population = pg.population(self.pg_problem, self.population_size, seed=self.seed)
        
        if verbose:
            logger.info(f"Initial population: {len(self.population)} individuals")
            self._log_population_stats(self.population, "Initial")
        
        # Evolution loop
        for generation in range(self.num_generations):
            # Evolve population for one generation
            self.population = self.algorithm.evolve(self.population)
            
            # Log progress
            if verbose and (generation + 1) % 10 == 0:
                logger.info(f"Generation {generation + 1}/{self.num_generations}")
                self._log_population_stats(self.population, f"Gen {generation + 1}")
            
            # Store history
            pareto_front = self._extract_pareto_front(self.population)
            history_entry = {
                'generation': generation + 1,
                'pareto_size': len(pareto_front),
            }
            
            if pareto_front:
                if len(pareto_front[0]) >= 1:
                    history_entry['best_objective_1'] = min(f[0] for f in pareto_front)
                if len(pareto_front[0]) >= 2:
                    history_entry['best_objective_2'] = min(f[1] for f in pareto_front)
                if len(pareto_front[0]) >= 3:
                    history_entry['best_objective_3'] = min(f[2] for f in pareto_front)
            
            self.optimization_history.append(history_entry)
        
        # Extract final results
        final_pareto = self._extract_pareto_front(self.population)
        
        results = {
            'success': len(final_pareto) > 0,
            'pareto_front': final_pareto,
            'pareto_solutions': self._extract_pareto_solutions(self.population),
            'population_size': len(self.population),
            'generations': self.num_generations,
            'optimization_history': self.optimization_history,
            'algorithm_info': {
                'name': 'NSGA-II',
                'population_size': self.population_size,
                'generations': self.num_generations,
                'seed': self.seed
            }
        }
        
        # Add cache stats if available
        if hasattr(self.problem, 'get_cache_stats'):
            results['cache_stats'] = self.problem.get_cache_stats()
        else:
            results['cache_stats'] = {}
        
        if verbose:
            logger.info(f"Optimization completed: {len(final_pareto)} Pareto solutions found")
            if hasattr(self.problem, 'get_cache_stats'):
                cache_stats = self.problem.get_cache_stats()
                logger.info(f"Cache efficiency: {cache_stats.get('hit_rate', 0):.1%} "
                           f"({cache_stats.get('cache_hits', 0)}/{cache_stats.get('total_evaluations', 0)} hits)")
        
        return results
    
    def get_best_solutions(self, 
                          num_solutions: int = 10,
                          preference_weights: List[float] = None) -> List[Dict[str, Any]]:
        """Get best solutions from Pareto front based on preferences.
        
        Args:
            num_solutions: Number of solutions to return
            preference_weights: Weights for [delta_v, time, cost] objectives
            
        Returns:
            List of best solutions with parameters and objectives
        """
        if self.population is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        if preference_weights is None:
            preference_weights = [1.0, 1.0, 1.0]  # Equal weighting
        
        pareto_solutions = self._extract_pareto_solutions(self.population)
        
        if not pareto_solutions:
            return []
        
        # Calculate weighted scores for ranking
        weighted_solutions = []
        for sol in pareto_solutions:
            objectives = sol['objectives']
            
            # Convert objectives to list if it's a dictionary
            if isinstance(objectives, dict):
                obj_list = [objectives['delta_v'], objectives['time'], objectives['cost']]
            else:
                obj_list = objectives
            
            # Normalize objectives (lower is better for all)
            norm_objectives = self._normalize_objectives([obj_list], pareto_solutions)[0]
            # Calculate weighted score
            weighted_score = sum(w * obj for w, obj in zip(preference_weights, norm_objectives))
            
            # Add weighted score to solution for testing
            sol_with_score = sol.copy()
            sol_with_score['weighted_score'] = weighted_score
            weighted_solutions.append((weighted_score, sol_with_score))
        
        # Sort by weighted score and return top solutions
        weighted_solutions.sort(key=lambda x: x[0])
        return [sol for _, sol in weighted_solutions[:num_solutions]]
    
    def _extract_pareto_front(self, population: pg.population) -> List[List[float]]:
        """Extract Pareto front from population.
        
        Args:
            population: PyGMO population
            
        Returns:
            List of Pareto-optimal objective vectors
        """
        # Use general fast non-dominated sort instead of 2D-specific function
        fitness_values = population.get_f()
        if len(fitness_values) == 0:
            return []
        
        # Use fast non-dominated sort which works for any number of objectives
        ndf, _, _, _ = pg.fast_non_dominated_sorting(fitness_values)
        
        # Return the first (best) front
        if len(ndf) > 0:
            pareto_indices = ndf[0]
            return [population.get_f()[i].tolist() for i in pareto_indices]
        else:
            return []
    
    def _extract_pareto_solutions(self, population: pg.population) -> List[Dict[str, Any]]:
        """Extract complete Pareto solutions with parameters and objectives.
        
        Args:
            population: PyGMO population
            
        Returns:
            List of Pareto solutions with parameters and objectives
        """
        # Use general fast non-dominated sort
        fitness_values = population.get_f()
        if len(fitness_values) == 0:
            return []
        
        ndf, _, _, _ = pg.fast_non_dominated_sorting(fitness_values)
        
        if len(ndf) == 0:
            return []
        
        pareto_indices = ndf[0]  # First (best) front
        solutions = []
        
        for i in pareto_indices:
            params = population.get_x()[i]
            objectives = population.get_f()[i]
            
            # Handle different problem structures flexibly
            solution = {
                'parameters': params,
                'objectives': objectives,
                'objective_vector': objectives,
                'parameter_vector': params
            }
            
            # If this is a lunar mission problem (3 parameters, 3 objectives), add named structure
            if len(params) >= 3 and len(objectives) >= 3:
                solution['parameters'] = {
                    'earth_orbit_alt': params[0],
                    'moon_orbit_alt': params[1], 
                    'transfer_time': params[2]
                }
                solution['objectives'] = {
                    'delta_v': objectives[0],      # m/s
                    'time': objectives[1],         # seconds
                    'cost': objectives[2]          # cost units
                }
            
            solutions.append(solution)
        
        return solutions
    
    def _normalize_objectives(self, 
                            objective_vectors: List[List[float]], 
                            all_solutions: List[Dict[str, Any]]) -> List[List[float]]:
        """Normalize objectives for weighted ranking.
        
        Args:
            objective_vectors: Objective vectors to normalize
            all_solutions: All solutions for finding min/max ranges
            
        Returns:
            Normalized objective vectors
        """
        # Find min/max for each objective
        all_objectives = [sol['objectives'] for sol in all_solutions]
        
        # Handle both dictionary and list formats
        if all_objectives and isinstance(all_objectives[0], dict):
            delta_v_values = [obj['delta_v'] for obj in all_objectives]
            time_values = [obj['time'] for obj in all_objectives]
            cost_values = [obj['cost'] for obj in all_objectives]
        else:
            # Assume list format [delta_v, time, cost]
            delta_v_values = [obj[0] for obj in all_objectives]
            time_values = [obj[1] for obj in all_objectives]
            cost_values = [obj[2] for obj in all_objectives]
        
        min_vals = [min(delta_v_values), min(time_values), min(cost_values)]
        max_vals = [max(delta_v_values), max(time_values), max(cost_values)]
        
        # Normalize each objective vector
        normalized = []
        for objectives in objective_vectors:
            norm_obj = []
            for i, (obj, min_val, max_val) in enumerate(zip(objectives, min_vals, max_vals)):
                if max_val > min_val:
                    norm_obj.append((obj - min_val) / (max_val - min_val))
                else:
                    norm_obj.append(0.0)
            normalized.append(norm_obj)
        
        return normalized
    
    def _log_population_stats(self, population: pg.population, label: str) -> None:
        """Log population statistics.
        
        Args:
            population: PyGMO population
            label: Label for logging context
        """
        fitness_values = population.get_f()
        if len(fitness_values) > 0:
            delta_v_values = [f[0] for f in fitness_values]
            time_values = [f[1] for f in fitness_values]
            cost_values = [f[2] for f in fitness_values]
            
            logger.info(f"{label} - Best ΔV: {min(delta_v_values):.0f} m/s, "
                       f"Best Time: {min(time_values)/86400:.1f} days, "
                       f"Best Cost: {min(cost_values):.0f}")


def optimize_lunar_mission(cost_factors: CostFactors = None,
                          optimization_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for lunar mission optimization.
    
    Args:
        cost_factors: Economic cost parameters
        optimization_config: Configuration for optimization parameters
        
    Returns:
        Optimization results
    """
    config = optimization_config or {}
    
    # Create problem
    problem = LunarMissionProblem(
        cost_factors=cost_factors,
        **config.get('problem_params', {})
    )
    
    # Create optimizer
    optimizer = GlobalOptimizer(
        problem=problem,
        **config.get('optimizer_params', {})
    )
    
    # Run optimization
    return optimizer.optimize(verbose=config.get('verbose', True))