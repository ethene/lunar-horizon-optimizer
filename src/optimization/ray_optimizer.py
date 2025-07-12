"""
Ray-based parallel optimization for GlobalOptimizer.

This module provides Ray actor-based parallelization of fitness evaluation
in PyGMO global optimization, enabling efficient multi-core utilization.

Features:
- Ray actor-based parallel fitness evaluation
- Batch processing for reduced overhead
- Resource pre-loading (SPICE kernels, etc.)
- Graceful fallback when Ray is unavailable
- Performance monitoring and benchmarking

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import logging
import time
from typing import Any, List, Optional, Union
import os
import sys

# Ray imports with graceful fallback
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

import numpy as np
import pygmo as pg

from src.config.costs import CostFactors
from src.optimization.global_optimizer import LunarMissionProblem, GlobalOptimizer
from src.trajectory.lunar_transfer import LunarTransfer
from src.optimization.cost_integration import CostCalculator

# Configure logging
logger = logging.getLogger(__name__)


if RAY_AVAILABLE:
    @ray.remote
    class FitnessWorker:
        """Ray actor for parallel fitness evaluation.
        
        This actor pre-loads heavy resources (SPICE kernels, trajectory models)
        and provides batch fitness evaluation to minimize inter-process overhead.
        """
        
        def __init__(self, 
                     cost_factors: Optional[dict] = None,
                     min_earth_alt: float = 200,
                     max_earth_alt: float = 1000,
                     min_moon_alt: float = 50,
                     max_moon_alt: float = 500,
                     min_transfer_time: float = 3.0,
                     max_transfer_time: float = 10.0,
                     reference_epoch: float = 10000.0,
                     worker_id: int = 0):
            """Initialize fitness worker with pre-loaded resources.
            
            Args:
                cost_factors: Cost factor parameters as dict
                min_earth_alt: Minimum Earth orbit altitude [km]
                max_earth_alt: Maximum Earth orbit altitude [km]
                min_moon_alt: Minimum lunar orbit altitude [km]
                max_moon_alt: Maximum lunar orbit altitude [km]
                min_transfer_time: Minimum transfer time [days]
                max_transfer_time: Maximum transfer time [days]
                reference_epoch: Reference epoch [days since J2000]
                worker_id: Unique worker identifier
            """
            self.worker_id = worker_id
            self.evaluation_count = 0
            self.total_time = 0.0
            
            # Initialize cost factors
            if cost_factors:
                self.cost_factors = CostFactors(**cost_factors)
            else:
                self.cost_factors = CostFactors()
            
            # Store bounds
            self.bounds = {
                'min_earth_alt': min_earth_alt,
                'max_earth_alt': max_earth_alt,
                'min_moon_alt': min_moon_alt,
                'max_moon_alt': max_moon_alt,
                'min_transfer_time': min_transfer_time,
                'max_transfer_time': max_transfer_time,
                'reference_epoch': reference_epoch
            }
            
            # Pre-load heavy resources
            self._initialize_resources()
            
            logger.info(f"FitnessWorker {worker_id} initialized with pre-loaded resources")
        
        def _initialize_resources(self):
            """Pre-load heavy computational resources."""
            try:
                # Initialize trajectory generator (pre-loads SPICE kernels if available)
                self.lunar_transfer = LunarTransfer(
                    min_earth_alt=self.bounds['min_earth_alt'],
                    max_earth_alt=self.bounds['max_earth_alt'],
                    min_moon_alt=self.bounds['min_moon_alt'],
                    max_moon_alt=self.bounds['max_moon_alt'],
                )
                
                # Initialize cost calculator
                self.cost_calculator = CostCalculator(self.cost_factors)
                
                # Initialize local cache for this worker
                self._local_cache = {}
                self._cache_hits = 0
                self._cache_misses = 0
                
                logger.debug(f"Worker {self.worker_id}: Resources pre-loaded successfully")
                
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: Resource initialization failed: {e}")
                # Create minimal fallback resources
                self.lunar_transfer = None
                self.cost_calculator = CostCalculator(self.cost_factors)
                self._local_cache = {}
        
        def evaluate_batch(self, population_chunk: List[List[float]]) -> List[List[float]]:
            """Evaluate fitness for a batch of individuals.
            
            Args:
                population_chunk: List of decision vectors to evaluate
                
            Returns:
                List of fitness vectors [delta_v, time, cost]
            """
            start_time = time.time()
            fitness_results = []
            
            for individual in population_chunk:
                fitness = self._evaluate_single(individual)
                fitness_results.append(fitness)
                self.evaluation_count += 1
            
            batch_time = time.time() - start_time
            self.total_time += batch_time
            
            logger.debug(f"Worker {self.worker_id}: Evaluated {len(population_chunk)} individuals in {batch_time:.3f}s")
            
            return fitness_results
        
        def _evaluate_single(self, x: List[float]) -> List[float]:
            """Evaluate fitness for a single individual.
            
            Args:
                x: Decision vector [earth_alt, moon_alt, transfer_time]
                
            Returns:
                Fitness vector [delta_v, time, cost]
            """
            earth_alt, moon_alt, transfer_time = x
            
            # Validate bounds - return penalty if out of bounds
            if (earth_alt < self.bounds['min_earth_alt'] or 
                earth_alt > self.bounds['max_earth_alt'] or
                moon_alt < self.bounds['min_moon_alt'] or 
                moon_alt > self.bounds['max_moon_alt'] or
                transfer_time < self.bounds['min_transfer_time'] or 
                transfer_time > self.bounds['max_transfer_time']):
                return [1e12, 1e12, 1e12]
            
            # Create cache key
            cache_key = f"{earth_alt:.1f}_{moon_alt:.1f}_{transfer_time:.2f}_{self.bounds['reference_epoch']:.1f}"
            
            # Check local cache
            if cache_key in self._local_cache:
                self._cache_hits += 1
                return self._local_cache[cache_key]
            
            self._cache_misses += 1
            
            try:
                # Generate trajectory
                if self.lunar_transfer:
                    trajectory, total_dv = self.lunar_transfer.generate_transfer(
                        epoch=self.bounds['reference_epoch'],
                        earth_orbit_alt=earth_alt,
                        moon_orbit_alt=moon_alt,
                        transfer_time=transfer_time,
                        max_revolutions=0,
                    )
                else:
                    # Fallback simple calculation if trajectory generator failed
                    total_dv = 3000 + (earth_alt - 300) * 2 + (moon_alt - 100) * 1
                
                # Calculate objectives
                obj1_delta_v = total_dv  # m/s
                obj2_time = transfer_time * 86400  # Convert days to seconds
                obj3_cost = self.cost_calculator.calculate_mission_cost(
                    total_dv=total_dv,
                    transfer_time=transfer_time,
                    earth_orbit_alt=earth_alt,
                    moon_orbit_alt=moon_alt,
                )
                
                objectives = [obj1_delta_v, obj2_time, obj3_cost]
                
                # Cache the result
                self._local_cache[cache_key] = objectives
                
                return objectives
                
            except Exception as e:
                logger.debug(f"Worker {self.worker_id}: Fitness evaluation failed for {x}: {e}")
                return [1e6, 1e6, 1e6]  # Penalty for infeasible solutions
        
        def get_stats(self) -> dict:
            """Get worker performance statistics.
            
            Returns:
                Dictionary with worker performance metrics
            """
            total_evaluations = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_evaluations if total_evaluations > 0 else 0
            avg_time = self.total_time / self.evaluation_count if self.evaluation_count > 0 else 0
            
            return {
                'worker_id': self.worker_id,
                'evaluations': self.evaluation_count,
                'total_time': self.total_time,
                'avg_time_per_eval': avg_time,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_rate': hit_rate,
                'cache_size': len(self._local_cache)
            }


class RayParallelOptimizer(GlobalOptimizer):
    """Ray-parallelized version of GlobalOptimizer.
    
    This class extends GlobalOptimizer to use Ray actors for parallel
    fitness evaluation, significantly improving performance on multi-core systems.
    """
    
    def __init__(self,
                 problem: Optional[LunarMissionProblem] = None,
                 population_size: int = 100,
                 num_generations: int = 100,
                 seed: int = 42,
                 num_workers: Optional[int] = None,
                 chunk_size: Optional[int] = None,
                 ray_config: Optional[dict] = None):
        """Initialize Ray-parallel optimizer.
        
        Args:
            problem: Optimization problem instance
            population_size: NSGA-II population size
            num_generations: Number of generations to evolve
            seed: Random seed for reproducibility
            num_workers: Number of Ray workers (default: CPU count)
            chunk_size: Individuals per worker batch (default: auto)
            ray_config: Ray initialization configuration
        """
        super().__init__(problem, population_size, num_generations, seed)
        
        self.use_ray = RAY_AVAILABLE
        self.num_workers = num_workers or os.cpu_count()
        self.chunk_size = chunk_size or max(1, population_size // (self.num_workers * 2))
        self.ray_config = ray_config or {}
        
        # Ray worker management
        self.workers = []
        self.worker_stats = []
        
        if not self.use_ray:
            logger.warning("Ray not available, falling back to sequential evaluation")
        else:
            logger.info(f"RayParallelOptimizer: {self.num_workers} workers, chunk_size={self.chunk_size}")
    
    def _initialize_ray_workers(self):
        """Initialize Ray workers for parallel fitness evaluation."""
        if not self.use_ray:
            return
        
        try:
            # Initialize Ray if not already running
            if not ray.is_initialized():
                ray.init(**self.ray_config)
                logger.info("Ray initialized for parallel optimization")
            
            # Create worker configuration from problem
            worker_config = {
                'min_earth_alt': self.problem.min_earth_alt,
                'max_earth_alt': self.problem.max_earth_alt,
                'min_moon_alt': self.problem.min_moon_alt,
                'max_moon_alt': self.problem.max_moon_alt,
                'min_transfer_time': self.problem.min_transfer_time,
                'max_transfer_time': self.problem.max_transfer_time,
                'reference_epoch': self.problem.reference_epoch,
            }
            
            # Convert cost factors to dict for serialization
            if hasattr(self.problem, 'cost_factors'):
                worker_config['cost_factors'] = {
                    'launch_cost_per_kg': self.problem.cost_factors.launch_cost_per_kg,
                    'operations_cost_per_day': self.problem.cost_factors.operations_cost_per_day,
                    'development_cost': self.problem.cost_factors.development_cost,
                }
            
            # Create Ray workers
            self.workers = []
            for i in range(self.num_workers):
                worker_config['worker_id'] = i
                worker = FitnessWorker.remote(**worker_config)
                self.workers.append(worker)
            
            logger.info(f"Created {len(self.workers)} Ray workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray workers: {e}")
            self.use_ray = False
    
    def _shutdown_ray_workers(self):
        """Clean up Ray workers and collect final statistics."""
        if not self.use_ray or not self.workers:
            return
        
        try:
            # Collect worker statistics
            stats_futures = [worker.get_stats.remote() for worker in self.workers]
            self.worker_stats = ray.get(stats_futures)
            
            # Log combined statistics
            total_evaluations = sum(stats['evaluations'] for stats in self.worker_stats)
            total_time = sum(stats['total_time'] for stats in self.worker_stats)
            total_cache_hits = sum(stats['cache_hits'] for stats in self.worker_stats)
            total_cache_misses = sum(stats['cache_misses'] for stats in self.worker_stats)
            
            if total_evaluations > 0:
                avg_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)
                logger.info(f"Ray workers completed: {total_evaluations} evaluations, "
                           f"{total_time:.2f}s total, {avg_hit_rate:.1%} cache hit rate")
            
            self.workers = []
            
        except Exception as e:
            logger.warning(f"Error during Ray worker shutdown: {e}")
    
    def optimize(self, verbose: bool = True) -> dict[str, Any]:
        """Run multi-objective optimization with Ray parallelization.
        
        Args:
            verbose: Enable detailed logging
            
        Returns:
            Dictionary with optimization results including Ray statistics
        """
        # Initialize Ray workers
        start_time = time.time()
        self._initialize_ray_workers()
        
        try:
            # Use custom population evaluation if Ray is available
            if self.use_ray:
                results = self._optimize_with_ray(verbose)
            else:
                results = super().optimize(verbose)
            
            # Add Ray performance statistics
            setup_time = time.time() - start_time
            results['ray_stats'] = {
                'ray_available': RAY_AVAILABLE,
                'ray_used': self.use_ray,
                'num_workers': self.num_workers if self.use_ray else 0,
                'chunk_size': self.chunk_size if self.use_ray else 0,
                'setup_time': setup_time,
                'worker_stats': self.worker_stats
            }
            
            return results
            
        finally:
            # Always clean up Ray workers
            self._shutdown_ray_workers()
    
    def _optimize_with_ray(self, verbose: bool = True) -> dict[str, Any]:
        """Run optimization using Ray parallel evaluation.
        
        This method implements a custom evolution loop that intercepts
        population evaluation and distributes it across Ray workers.
        
        Args:
            verbose: Enable detailed logging
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Ray-parallel NSGA-II optimization: {self.num_generations} generations")
        
        # Initialize population normally
        self.population = pg.population(self.pg_problem, self.population_size, seed=self.seed)
        
        if verbose:
            logger.info(f"Initial population: {len(self.population)} individuals")
            self._log_population_stats(self.population, "Initial")
        
        # Custom evolution loop with Ray parallelization
        for generation in range(self.num_generations):
            generation_start = time.time()
            
            # Get current population decision vectors
            population_x = self.population.get_x()
            
            # Evaluate population using Ray workers
            population_f = self._evaluate_population_parallel(population_x)
            
            # Create new population with updated fitness
            new_population = pg.population(self.pg_problem, 0)  # Empty population
            for i, (x, f) in enumerate(zip(population_x, population_f)):
                new_population.push_back(x, f)
            
            # Apply NSGA-II selection and reproduction
            self.population = self.algorithm.evolve(new_population)
            
            generation_time = time.time() - generation_start
            
            # Log progress
            if verbose and (generation + 1) % 10 == 0:
                logger.info(f"Generation {generation + 1}/{self.num_generations} "
                           f"({generation_time:.2f}s)")
                self._log_population_stats(self.population, f"Gen {generation + 1}")
            
            # Store history
            pareto_front = self._extract_pareto_front(self.population)
            history_entry = {
                "generation": generation + 1,
                "pareto_size": len(pareto_front),
                "generation_time": generation_time,
            }
            
            if pareto_front:
                if len(pareto_front[0]) >= 1:
                    history_entry["best_objective_1"] = min(f[0] for f in pareto_front)
                if len(pareto_front[0]) >= 2:
                    history_entry["best_objective_2"] = min(f[1] for f in pareto_front)
                if len(pareto_front[0]) >= 3:
                    history_entry["best_objective_3"] = min(f[2] for f in pareto_front)
            
            self.optimization_history.append(history_entry)
        
        # Extract final results (same as parent class)
        final_pareto = self._extract_pareto_front(self.population)
        
        results = {
            "success": len(final_pareto) > 0,
            "pareto_front": final_pareto,
            "pareto_solutions": self._extract_pareto_solutions(self.population),
            "population_size": len(self.population),
            "generations": self.num_generations,
            "optimization_history": self.optimization_history,
            "algorithm_info": {
                "name": "NSGA-II (Ray Parallel)",
                "population_size": self.population_size,
                "generations": self.num_generations,
                "seed": self.seed,
                "num_workers": self.num_workers,
                "chunk_size": self.chunk_size,
            },
        }
        
        if verbose:
            logger.info(f"Ray-parallel optimization completed: {len(final_pareto)} Pareto solutions found")
        
        return results
    
    def _evaluate_population_parallel(self, population_x: np.ndarray) -> List[List[float]]:
        """Evaluate population fitness using Ray workers.
        
        Args:
            population_x: Array of decision vectors to evaluate
            
        Returns:
            List of fitness vectors
        """
        if not self.use_ray or not self.workers:
            # Fallback to sequential evaluation
            return [self.problem.fitness(x.tolist()) for x in population_x]
        
        eval_start = time.time()
        
        # Split population into chunks for workers
        chunks = []
        for i in range(0, len(population_x), self.chunk_size):
            chunk = population_x[i:i + self.chunk_size].tolist()
            chunks.append(chunk)
        
        # Distribute chunks to workers (round-robin)
        worker_futures = []
        for i, chunk in enumerate(chunks):
            worker_idx = i % len(self.workers)
            future = self.workers[worker_idx].evaluate_batch.remote(chunk)
            worker_futures.append(future)
        
        # Gather results
        chunk_results = ray.get(worker_futures)
        
        # Flatten results back to population order
        population_f = []
        for chunk_result in chunk_results:
            population_f.extend(chunk_result)
        
        eval_time = time.time() - eval_start
        logger.debug(f"Parallel evaluation: {len(population_x)} individuals in {eval_time:.3f}s "
                    f"({len(chunks)} chunks, {len(self.workers)} workers)")
        
        return population_f


def create_ray_optimizer(problem_config: Optional[dict] = None,
                        optimizer_config: Optional[dict] = None,
                        ray_config: Optional[dict] = None) -> Union[RayParallelOptimizer, GlobalOptimizer]:
    """Create Ray-parallel optimizer with fallback to sequential.
    
    Args:
        problem_config: Configuration for LunarMissionProblem
        optimizer_config: Configuration for optimizer parameters
        ray_config: Configuration for Ray initialization
        
    Returns:
        RayParallelOptimizer if Ray available, otherwise GlobalOptimizer
    """
    problem_config = problem_config or {}
    optimizer_config = optimizer_config or {}
    
    # Ensure cost_factors are provided with defaults if not specified
    if 'cost_factors' not in problem_config:
        problem_config['cost_factors'] = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9
        )
    
    # Create problem
    problem = LunarMissionProblem(**problem_config)
    
    # Create appropriate optimizer
    if RAY_AVAILABLE:
        optimizer = RayParallelOptimizer(
            problem=problem,
            ray_config=ray_config,
            **optimizer_config
        )
        logger.info("Created RayParallelOptimizer")
    else:
        optimizer = GlobalOptimizer(
            problem=problem,
            **optimizer_config
        )
        logger.info("Created sequential GlobalOptimizer (Ray not available)")
    
    return optimizer