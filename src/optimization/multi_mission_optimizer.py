"""Multi-mission global optimizer extending the original GlobalOptimizer.

This module provides enhanced optimization capabilities for constellation missions
while maintaining full backward compatibility with single-mission optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pygmo as pg

from src.config.costs import CostFactors
from src.optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
from src.optimization.multi_mission_genome import (
    MultiMissionProblem, 
    MultiMissionGenome,
    create_backward_compatible_problem
)

logger = logging.getLogger(__name__)


class MultiMissionOptimizer(GlobalOptimizer):
    """Enhanced optimizer for both single and multi-mission problems.
    
    This optimizer extends GlobalOptimizer to handle constellation optimization
    while maintaining full backward compatibility with existing single-mission code.
    
    Key Features:
    - Automatic problem type detection
    - Enhanced Pareto analysis for constellation objectives  
    - Constellation-specific solution ranking
    - Migration utilities for existing code
    """
    
    def __init__(
        self,
        problem: Optional[Union[LunarMissionProblem, MultiMissionProblem]] = None,
        population_size: int = 100,
        num_generations: int = 100,
        seed: int = 42,
        # Multi-mission specific parameters
        multi_mission_mode: bool = False,
        num_missions: int = 1,
        constellation_preferences: Optional[Dict[str, float]] = None
    ):
        """Initialize the multi-mission optimizer.
        
        Args:
            problem: Optimization problem (single or multi-mission)
            population_size: NSGA-II population size
            num_generations: Number of generations to evolve
            seed: Random seed for reproducibility
            multi_mission_mode: Enable multi-mission optimization
            num_missions: Number of missions for constellation
            constellation_preferences: Weights for constellation objectives
        """
        self.multi_mission_mode = multi_mission_mode
        self.num_missions = num_missions
        self.constellation_preferences = constellation_preferences or {
            'delta_v': 1.0,
            'time': 1.0, 
            'cost': 1.0,
            'coverage': 1.0,
            'redundancy': 0.5
        }
        
        # Create problem if not provided
        if problem is None:
            problem = create_backward_compatible_problem(
                enable_multi=multi_mission_mode,
                num_missions=num_missions
            )
        
        # Detect problem type
        self.is_multi_mission = isinstance(problem, MultiMissionProblem)
        
        # Initialize parent class
        super().__init__(
            problem=problem,
            population_size=population_size,
            num_generations=num_generations,
            seed=seed
        )
        
        logger.info(
            f"Initialized MultiMissionOptimizer: "
            f"{'multi-mission' if self.is_multi_mission else 'single-mission'} mode, "
            f"{num_missions} missions"
        )
    
    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """Run optimization with enhanced multi-mission support.
        
        Args:
            verbose: Enable detailed logging
            
        Returns:
            Enhanced results dictionary with constellation analysis
        """
        # Run base optimization
        results = super().optimize(verbose=verbose)
        
        # Add multi-mission specific analysis
        if self.is_multi_mission and results.get('success', False):
            results = self._enhance_multi_mission_results(results)
        
        return results
    
    def get_best_constellation_solutions(
        self, 
        num_solutions: int = 5,
        preference_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Get best constellation solutions with enhanced analysis.
        
        Args:
            num_solutions: Number of solutions to return
            preference_weights: Custom preference weights
            
        Returns:
            List of best constellation solutions with detailed analysis
        """
        if not self.is_multi_mission:
            # Fallback to parent method for single missions
            return super().get_best_solutions(num_solutions)
        
        if self.population is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Use provided weights or defaults
        weights = preference_weights or self.constellation_preferences
        
        # Extract Pareto solutions
        pareto_solutions = self._extract_pareto_solutions(self.population)
        
        if not pareto_solutions:
            return []
        
        # Enhanced ranking for constellation problems
        ranked_solutions = []
        for sol in pareto_solutions:
            objectives = sol["objectives"] 
            
            # Convert to list if needed
            if isinstance(objectives, dict):
                obj_list = list(objectives.values())
            else:
                obj_list = objectives
            
            # Calculate weighted score for constellation
            if len(obj_list) >= 5:  # Multi-mission problem
                weighted_score = (
                    weights.get('delta_v', 1.0) * obj_list[0] / 1e6 +      # Normalize delta-v
                    weights.get('time', 1.0) * obj_list[1] / 1e6 +        # Normalize time  
                    weights.get('cost', 1.0) * obj_list[2] / 1e12 +       # Normalize cost
                    weights.get('coverage', 1.0) * obj_list[3] / 1e3 +    # Normalize coverage
                    weights.get('redundancy', 0.5) * obj_list[4] / 1e3     # Normalize redundancy
                )
            else:  # Single mission fallback
                weighted_score = (
                    weights.get('delta_v', 1.0) * obj_list[0] / 1e6 +
                    weights.get('time', 1.0) * obj_list[1] / 1e6 +
                    weights.get('cost', 1.0) * obj_list[2] / 1e12
                )
            
            # Add constellation-specific analysis
            enhanced_sol = sol.copy()
            enhanced_sol['weighted_score'] = weighted_score
            enhanced_sol['constellation_analysis'] = self._analyze_constellation_solution(sol)
            
            ranked_solutions.append((weighted_score, enhanced_sol))
        
        # Sort by weighted score and return top solutions
        ranked_solutions.sort(key=lambda x: x[0])
        return [sol for _, sol in ranked_solutions[:num_solutions]]
    
    def get_constellation_metrics(self) -> Dict[str, Any]:
        """Get constellation-specific performance metrics.
        
        Returns:
            Dictionary with constellation analysis
        """
        if not self.is_multi_mission or self.population is None:
            return {}
        
        pareto_solutions = self._extract_pareto_solutions(self.population)
        
        if not pareto_solutions:
            return {}
        
        # Analyze constellation performance
        metrics = {
            'num_pareto_solutions': len(pareto_solutions),
            'num_missions': self.num_missions,
            'constellation_coverage': [],
            'constellation_redundancy': [],
            'mission_efficiency': []
        }
        
        for sol in pareto_solutions:
            objectives = sol.get("objectives", [])
            if len(objectives) >= 5:
                metrics['constellation_coverage'].append(objectives[3])
                metrics['constellation_redundancy'].append(objectives[4])
                
                # Calculate efficiency (payload capacity per unit cost)
                total_cost = objectives[2]
                total_payload = self.num_missions * 1000.0  # Estimate
                efficiency = total_payload / (total_cost / 1e6) if total_cost > 0 else 0
                metrics['mission_efficiency'].append(efficiency)
        
        # Summary statistics
        if metrics['constellation_coverage']:
            metrics['coverage_stats'] = {
                'mean': np.mean(metrics['constellation_coverage']),
                'std': np.std(metrics['constellation_coverage']),
                'min': np.min(metrics['constellation_coverage']),
                'max': np.max(metrics['constellation_coverage'])
            }
        
        if metrics['constellation_redundancy']:
            metrics['redundancy_stats'] = {
                'mean': np.mean(metrics['constellation_redundancy']),
                'std': np.std(metrics['constellation_redundancy']),
                'min': np.min(metrics['constellation_redundancy']),
                'max': np.max(metrics['constellation_redundancy'])
            }
        
        return metrics
    
    def _enhance_multi_mission_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with multi-mission analysis.
        
        Args:
            results: Base optimization results
            
        Returns:
            Enhanced results with constellation metrics
        """
        enhanced_results = results.copy()
        
        # Add constellation metrics
        enhanced_results['constellation_metrics'] = self.get_constellation_metrics()
        
        # Add best constellation solutions
        best_constellations = self.get_best_constellation_solutions(num_solutions=3)
        enhanced_results['best_constellations'] = best_constellations
        
        # Add problem-specific information
        enhanced_results['problem_info'] = {
            'problem_type': 'multi_mission',
            'num_missions': self.num_missions,
            'objectives': ['delta_v', 'time', 'cost', 'coverage', 'redundancy'],
            'decision_variables': 4 * self.num_missions + 2,
            'constellation_mode': True
        }
        
        return enhanced_results
    
    def _analyze_constellation_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual constellation solution.
        
        Args:
            solution: Pareto solution to analyze
            
        Returns:
            Dictionary with constellation analysis
        """
        analysis = {}
        
        # Extract parameters if available
        params = solution.get("parameters", [])
        objectives = solution.get("objectives", [])
        
        if len(params) >= 4 * self.num_missions + 2:
            # Decode as multi-mission genome
            try:
                genome = MultiMissionGenome.from_decision_vector(params, self.num_missions)
                
                analysis['mission_details'] = []
                for i in range(self.num_missions):
                    mission_params = genome.get_mission_parameters(i)
                    analysis['mission_details'].append({
                        'mission_id': i,
                        'epoch': mission_params['epoch'],
                        'earth_altitude': mission_params['earth_orbit_alt'],
                        'lunar_altitude': mission_params['moon_orbit_alt'], 
                        'plane_raan': mission_params['plane_raan'],
                        'payload_mass': mission_params['payload_mass']
                    })
                
                # Constellation geometry analysis
                analysis['geometry'] = {
                    'raan_distribution': genome.plane_raan,
                    'altitude_uniformity': np.std(genome.parking_altitudes),
                    'timing_spread': np.std(genome.epochs),
                    'payload_distribution': genome.payload_masses
                }
                
                # Performance per mission
                if len(objectives) >= 3:
                    analysis['performance'] = {
                        'avg_delta_v_per_mission': objectives[0] / self.num_missions,
                        'avg_time_per_mission': objectives[1] / self.num_missions,
                        'avg_cost_per_mission': objectives[2] / self.num_missions
                    }
                
            except Exception as e:
                logger.debug(f"Failed to analyze constellation solution: {e}")
                analysis['error'] = str(e)
        
        return analysis


def optimize_constellation(
    num_missions: int,
    cost_factors: Optional[CostFactors] = None,
    optimization_config: Optional[Dict[str, Any]] = None,
    constellation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function for constellation optimization.
    
    Args:
        num_missions: Number of missions in constellation
        cost_factors: Economic cost parameters
        optimization_config: Optimization algorithm configuration
        constellation_config: Constellation-specific configuration
        
    Returns:
        Optimization results with constellation analysis
    """
    opt_config = optimization_config or {}
    const_config = constellation_config or {}
    
    # Create multi-mission problem
    problem_params = const_config.get('problem_params', {})
    # Ensure constellation_mode is True (override any conflicting setting)
    problem_params['constellation_mode'] = True
    
    problem = MultiMissionProblem(
        num_missions=num_missions,
        cost_factors=cost_factors,
        **problem_params
    )
    
    # Create optimizer
    optimizer = MultiMissionOptimizer(
        problem=problem,
        multi_mission_mode=True,
        num_missions=num_missions,
        **opt_config.get('optimizer_params', {})
    )
    
    # Run optimization
    results = optimizer.optimize(verbose=opt_config.get('verbose', True))
    
    return results


# Migration utilities for backward compatibility
def migrate_single_to_multi(
    single_config: Dict[str, Any],
    num_missions: int = 3
) -> Dict[str, Any]:
    """Migrate single-mission configuration to multi-mission.
    
    Args:
        single_config: Single-mission optimization configuration
        num_missions: Number of missions for constellation
        
    Returns:
        Multi-mission configuration
    """
    multi_config = single_config.copy()
    
    # Add multi-mission parameters
    multi_config['num_missions'] = num_missions
    multi_config['constellation_mode'] = True
    
    # Scale population size for increased dimensionality
    if 'population_size' in multi_config:
        multi_config['population_size'] = max(
            multi_config['population_size'],
            50 * num_missions  # Minimum population for constellation
        )
    
    # Scale generations for convergence
    if 'num_generations' in multi_config:
        multi_config['num_generations'] = max(
            multi_config['num_generations'],
            100 + 20 * num_missions  # Additional generations
        )
    
    logger.info(f"Migrated single-mission config to {num_missions}-mission constellation")
    
    return multi_config