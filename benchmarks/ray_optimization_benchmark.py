#!/usr/bin/env python3
"""
Ray Optimization Benchmark

This script benchmarks Ray-parallel optimization against sequential
optimization to measure performance improvements.

Usage:
    python benchmarks/ray_optimization_benchmark.py --individuals 100 --workers 8
    python benchmarks/ray_optimization_benchmark.py --profile --generations 10
    python benchmarks/ray_optimization_benchmark.py --compare-all

Author: Lunar Horizon Optimizer Development Team  
Date: July 2025
"""

import argparse
import time
import os
import sys
import logging
from typing import Dict, Any, List
import json
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
from config.costs import CostFactors

# Ray imports with graceful fallback
try:
    import ray
    from optimization.ray_optimizer import RayParallelOptimizer, RAY_AVAILABLE
except ImportError:
    RAY_AVAILABLE = False
    RayParallelOptimizer = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizationBenchmark:
    """Benchmark suite for optimization performance comparison."""
    
    def __init__(self, individuals: int = 100, generations: int = 20, workers: int = None):
        """Initialize benchmark parameters.
        
        Args:
            individuals: Population size for optimization
            generations: Number of generations to run
            workers: Number of Ray workers (default: CPU count)
        """
        self.individuals = individuals
        self.generations = generations
        self.workers = workers or os.cpu_count()
        self.results = {}
        
        # Create test problem
        self.cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9
        )
        
        self.problem_config = {
            'cost_factors': self.cost_factors,
            'min_earth_alt': 200,
            'max_earth_alt': 800,
            'min_moon_alt': 50,
            'max_moon_alt': 300,
            'min_transfer_time': 3.0,
            'max_transfer_time': 8.0,
            'reference_epoch': 10000.0
        }
        
        logger.info(f"Benchmark setup: {individuals} individuals, {generations} generations, {self.workers} workers")
    
    def benchmark_sequential(self, seed: int = 42) -> Dict[str, Any]:
        """Benchmark sequential GlobalOptimizer.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Benchmark results dictionary
        """
        logger.info("Running sequential optimization benchmark...")
        
        start_time = time.time()
        
        # Create sequential optimizer
        problem = LunarMissionProblem(**self.problem_config)
        optimizer = GlobalOptimizer(
            problem=problem,
            population_size=self.individuals,
            num_generations=self.generations,
            seed=seed
        )
        
        setup_time = time.time() - start_time
        
        # Run optimization
        optimization_start = time.time()
        results = optimizer.optimize(verbose=False)
        optimization_time = time.time() - optimization_start
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        cache_stats = results.get('cache_stats', {})
        total_evaluations = cache_stats.get('total_evaluations', 0)
        
        benchmark_results = {
            'method': 'sequential',
            'total_time': total_time,
            'setup_time': setup_time,
            'optimization_time': optimization_time,
            'individuals': self.individuals,
            'generations': self.generations,
            'total_evaluations': total_evaluations,
            'evaluations_per_second': total_evaluations / optimization_time if optimization_time > 0 else 0,
            'pareto_solutions': len(results.get('pareto_front', [])),
            'cache_hit_rate': cache_stats.get('hit_rate', 0),
            'success': results.get('success', False)
        }
        
        logger.info(f"Sequential optimization completed in {total_time:.2f}s "
                   f"({total_evaluations} evaluations, {benchmark_results['evaluations_per_second']:.1f} eval/s)")
        
        return benchmark_results
    
    def benchmark_ray_parallel(self, seed: int = 42) -> Dict[str, Any]:
        """Benchmark Ray-parallel optimization.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Benchmark results dictionary
        """
        if not RAY_AVAILABLE:
            logger.error("Ray not available for parallel benchmark")
            return {'method': 'ray_parallel', 'error': 'Ray not available'}
        
        logger.info("Running Ray parallel optimization benchmark...")
        
        start_time = time.time()
        
        # Create Ray parallel optimizer
        problem = LunarMissionProblem(**self.problem_config)
        optimizer = RayParallelOptimizer(
            problem=problem,
            population_size=self.individuals,
            num_generations=self.generations,
            seed=seed,
            num_workers=self.workers,
            ray_config={'ignore_reinit_error': True}
        )
        
        setup_time = time.time() - start_time
        
        # Run optimization
        optimization_start = time.time()
        results = optimizer.optimize(verbose=False)
        optimization_time = time.time() - optimization_start
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        ray_stats = results.get('ray_stats', {})
        worker_stats = ray_stats.get('worker_stats', [])
        
        total_evaluations = sum(stats.get('evaluations', 0) for stats in worker_stats)
        total_worker_time = sum(stats.get('total_time', 0) for stats in worker_stats)
        avg_cache_hit_rate = np.mean([stats.get('cache_hit_rate', 0) for stats in worker_stats]) if worker_stats else 0
        
        benchmark_results = {
            'method': 'ray_parallel',
            'total_time': total_time,
            'setup_time': setup_time,
            'optimization_time': optimization_time,
            'individuals': self.individuals,
            'generations': self.generations,
            'workers': self.workers,
            'total_evaluations': total_evaluations,
            'total_worker_time': total_worker_time,
            'evaluations_per_second': total_evaluations / optimization_time if optimization_time > 0 else 0,
            'parallel_efficiency': (total_worker_time / optimization_time) / self.workers if optimization_time > 0 else 0,
            'pareto_solutions': len(results.get('pareto_front', [])),
            'cache_hit_rate': avg_cache_hit_rate,
            'success': results.get('success', False),
            'ray_setup_time': ray_stats.get('setup_time', 0)
        }
        
        logger.info(f"Ray parallel optimization completed in {total_time:.2f}s "
                   f"({total_evaluations} evaluations, {benchmark_results['evaluations_per_second']:.1f} eval/s, "
                   f"{benchmark_results['parallel_efficiency']:.1%} efficiency)")
        
        return benchmark_results
    
    def run_comparison(self, runs: int = 3) -> Dict[str, Any]:
        """Run comparison benchmark between sequential and parallel methods.
        
        Args:
            runs: Number of benchmark runs to average
            
        Returns:
            Comparison results
        """
        logger.info(f"Running comparison benchmark with {runs} runs...")
        
        sequential_results = []
        ray_results = []
        
        for run in range(runs):
            logger.info(f"Run {run + 1}/{runs}")
            
            # Sequential benchmark
            seq_result = self.benchmark_sequential(seed=42 + run)
            sequential_results.append(seq_result)
            
            # Ray benchmark (if available)
            if RAY_AVAILABLE:
                ray_result = self.benchmark_ray_parallel(seed=42 + run)
                ray_results.append(ray_result)
                
                # Shutdown Ray between runs to ensure clean state
                if ray.is_initialized():
                    ray.shutdown()
            
            logger.info(f"Run {run + 1} completed")
        
        # Calculate averages
        def average_results(results_list):
            if not results_list:
                return {}
            
            avg_results = {'method': results_list[0]['method']}
            numeric_keys = ['total_time', 'setup_time', 'optimization_time', 
                           'evaluations_per_second', 'cache_hit_rate']
            
            for key in numeric_keys:
                values = [r.get(key, 0) for r in results_list if key in r]
                if values:
                    avg_results[key] = np.mean(values)
                    avg_results[f'{key}_std'] = np.std(values)
            
            return avg_results
        
        comparison = {
            'benchmark_config': {
                'individuals': self.individuals,
                'generations': self.generations,
                'workers': self.workers,
                'runs': runs
            },
            'sequential': {
                'average': average_results(sequential_results),
                'runs': sequential_results
            }
        }
        
        if ray_results:
            comparison['ray_parallel'] = {
                'average': average_results(ray_results),
                'runs': ray_results
            }
            
            # Calculate speedup
            seq_avg_time = comparison['sequential']['average'].get('optimization_time', 0)
            ray_avg_time = comparison['ray_parallel']['average'].get('optimization_time', 0)
            
            if ray_avg_time > 0:
                speedup = seq_avg_time / ray_avg_time
                comparison['speedup'] = speedup
                comparison['efficiency'] = speedup / self.workers
                
                logger.info(f"Ray speedup: {speedup:.2f}x ({comparison['efficiency']:.1%} efficiency)")
            else:
                comparison['speedup'] = 0
                comparison['efficiency'] = 0
        
        return comparison
    
    def profile_fitness_function(self, num_evaluations: int = 1000) -> Dict[str, Any]:
        """Profile individual fitness function performance.
        
        Args:
            num_evaluations: Number of fitness evaluations to profile
            
        Returns:
            Profiling results
        """
        logger.info(f"Profiling fitness function with {num_evaluations} evaluations...")
        
        # Create problem
        problem = LunarMissionProblem(**self.problem_config)
        
        # Generate random test points
        bounds = problem.get_bounds()
        lower, upper = bounds
        
        test_points = []
        for _ in range(num_evaluations):
            point = [
                np.random.uniform(lower[i], upper[i]) 
                for i in range(len(lower))
            ]
            test_points.append(point)
        
        # Profile sequential evaluation
        start_time = time.time()
        fitness_results = []
        
        for point in test_points:
            fitness = problem.fitness(point)
            fitness_results.append(fitness)
        
        total_time = time.time() - start_time
        avg_time_per_eval = total_time / num_evaluations
        
        # Get cache statistics
        cache_stats = problem.get_cache_stats()
        
        profile_results = {
            'num_evaluations': num_evaluations,
            'total_time': total_time,
            'avg_time_per_eval': avg_time_per_eval,
            'evaluations_per_second': num_evaluations / total_time,
            'cache_stats': cache_stats,
            'sample_fitness_values': fitness_results[:5]  # First 5 for verification
        }
        
        logger.info(f"Fitness profiling completed: {avg_time_per_eval*1000:.2f}ms per evaluation, "
                   f"{profile_results['evaluations_per_second']:.1f} eval/s")
        
        return profile_results


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Ray optimization benchmark")
    parser.add_argument('--individuals', type=int, default=100, 
                       help='Population size (default: 100)')
    parser.add_argument('--generations', type=int, default=20,
                       help='Number of generations (default: 20)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of Ray workers (default: CPU count)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of benchmark runs (default: 3)')
    parser.add_argument('--profile', action='store_true',
                       help='Run fitness function profiling')
    parser.add_argument('--profile-evals', type=int, default=1000,
                       help='Number of evaluations for profiling (default: 1000)')
    parser.add_argument('--sequential-only', action='store_true',
                       help='Run only sequential benchmark')
    parser.add_argument('--ray-only', action='store_true',
                       help='Run only Ray benchmark')
    parser.add_argument('--compare-all', action='store_true',
                       help='Run comprehensive comparison')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (JSON)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create benchmark instance
    benchmark = OptimizationBenchmark(
        individuals=args.individuals,
        generations=args.generations,
        workers=args.workers
    )
    
    results = {}
    
    # Run profiling if requested
    if args.profile:
        results['profiling'] = benchmark.profile_fitness_function(args.profile_evals)
    
    # Run benchmarks
    if args.compare_all or (not args.sequential_only and not args.ray_only):
        results['comparison'] = benchmark.run_comparison(args.runs)
    elif args.sequential_only:
        results['sequential'] = benchmark.benchmark_sequential()
    elif args.ray_only:
        if RAY_AVAILABLE:
            results['ray_parallel'] = benchmark.benchmark_ray_parallel()
        else:
            logger.error("Ray not available for Ray-only benchmark")
            return
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    if 'profiling' in results:
        prof = results['profiling']
        print(f"Fitness Function Profiling:")
        print(f"  Average time per evaluation: {prof['avg_time_per_eval']*1000:.2f}ms")
        print(f"  Evaluations per second: {prof['evaluations_per_second']:.1f}")
        print(f"  Cache hit rate: {prof['cache_stats'].get('hit_rate', 0):.1%}")
        print()
    
    if 'comparison' in results:
        comp = results['comparison']
        seq = comp['sequential']['average']
        print(f"Sequential Optimization:")
        print(f"  Average time: {seq.get('optimization_time', 0):.2f}s")
        print(f"  Evaluations/sec: {seq.get('evaluations_per_second', 0):.1f}")
        
        if 'ray_parallel' in comp:
            ray_res = comp['ray_parallel']['average']
            print(f"Ray Parallel Optimization:")
            print(f"  Average time: {ray_res.get('optimization_time', 0):.2f}s") 
            print(f"  Evaluations/sec: {ray_res.get('evaluations_per_second', 0):.1f}")
            print(f"  Speedup: {comp.get('speedup', 0):.2f}x")
            print(f"  Efficiency: {comp.get('efficiency', 0):.1%}")
        else:
            print("Ray parallel optimization not available")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()