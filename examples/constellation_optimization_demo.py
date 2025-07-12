"""Constellation Optimization Demonstration

This script demonstrates the multi-mission optimization capability
for lunar communication satellite constellations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import logging

from src.config.costs import CostFactors
from src.optimization.multi_mission_optimizer import (
    MultiMissionOptimizer,
    optimize_constellation
)
from src.optimization.multi_mission_genome import MultiMissionGenome

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_constellation_optimization():
    """Demonstrate constellation optimization with different sizes."""
    
    print("="*60)
    print("LUNAR CONSTELLATION OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Cost configuration
    cost_factors = CostFactors(
        launch_cost_per_kg=12000.0,      # Slightly higher launch cost
        operations_cost_per_day=75000.0,  # Operations cost
        development_cost=2e9,             # 2B development cost
        contingency_percentage=25.0       # 25% contingency
    )
    
    # Test different constellation sizes
    constellation_sizes = [1, 3, 6]
    results = {}
    
    for K in constellation_sizes:
        print(f"\n{'-'*40}")
        print(f"OPTIMIZING {K}-SATELLITE CONSTELLATION")
        print(f"{'-'*40}")
        
        # Configuration for this constellation size
        optimization_config = {
            'optimizer_params': {
                'population_size': max(50, 30 * K),
                'num_generations': max(20, 15 + 5 * K),
                'seed': 42
            },
            'verbose': True
        }
        
        constellation_config = {
            'problem_params': {
                'coverage_weight': 1.5,      # Emphasize coverage
                'redundancy_weight': 0.8,    # Some redundancy importance
                'constellation_mode': True
            }
        }
        
        try:
            # Run optimization
            result = optimize_constellation(
                num_missions=K,
                cost_factors=cost_factors,
                optimization_config=optimization_config,
                constellation_config=constellation_config
            )
            
            results[K] = result
            
            # Print summary
            print_constellation_summary(result, K)
            
        except Exception as e:
            logger.error(f"Optimization failed for K={K}: {e}")
            results[K] = {'success': False, 'error': str(e)}
    
    # Compare results
    print(f"\n{'='*60}")
    print("CONSTELLATION COMPARISON")
    print(f"{'='*60}")
    
    compare_constellations(results)
    
    return results


def print_constellation_summary(result: Dict[str, Any], K: int):
    """Print summary for a constellation optimization result."""
    
    if not result.get('success', False):
        print(f"❌ Optimization failed for {K}-satellite constellation")
        return
    
    print(f"✅ Successfully optimized {K}-satellite constellation")
    
    # Basic optimization info
    pareto_front = result.get('pareto_front', [])
    print(f"   • Pareto solutions found: {len(pareto_front)}")
    
    # Best constellation solutions
    best_constellations = result.get('best_constellations', [])
    if best_constellations:
        best = best_constellations[0]
        objectives = best.get('objectives', [])
        
        if len(objectives) >= 3:
            total_dv = objectives[0]
            total_time = objectives[1] / 86400  # Convert to days
            total_cost = objectives[2] / 1e6    # Convert to millions
            
            print(f"   • Best Solution:")
            print(f"     - Total ΔV: {total_dv:.0f} m/s ({total_dv/K:.0f} m/s per satellite)")
            print(f"     - Total Time: {total_time:.1f} days ({total_time/K:.1f} days per satellite)")  
            print(f"     - Total Cost: ${total_cost:.1f}M (${total_cost/K:.1f}M per satellite)")
            
            if len(objectives) >= 5:
                coverage = objectives[3]
                redundancy = objectives[4]
                print(f"     - Coverage Score: {coverage:.2f}")
                print(f"     - Redundancy Score: {redundancy:.2f}")
    
    # Constellation metrics
    const_metrics = result.get('constellation_metrics', {})
    if const_metrics:
        coverage_stats = const_metrics.get('coverage_stats', {})
        if coverage_stats:
            mean_coverage = coverage_stats.get('mean', 0)
            std_coverage = coverage_stats.get('std', 0)
            print(f"   • Coverage Analysis: {mean_coverage:.2f} ± {std_coverage:.2f}")
    
    # Cache efficiency
    cache_stats = result.get('cache_stats', {})
    if cache_stats:
        hit_rate = cache_stats.get('hit_rate', 0)
        total_evals = cache_stats.get('total_evaluations', 0)
        print(f"   • Cache Efficiency: {hit_rate:.1%} ({total_evals:,} evaluations)")


def compare_constellations(results: Dict[int, Dict[str, Any]]):
    """Compare optimization results across different constellation sizes."""
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        print("❌ No successful optimizations to compare")
        return
    
    print(f"{'Satellites':<12} {'Best ΔV':<12} {'Best Time':<12} {'Best Cost':<12} {'Per Sat Cost':<12}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for K in sorted(successful_results.keys()):
        result = successful_results[K]
        best_constellations = result.get('best_constellations', [])
        
        if best_constellations:
            best = best_constellations[0]
            objectives = best.get('objectives', [])
            
            if len(objectives) >= 3:
                total_dv = objectives[0]
                total_time = objectives[1] / 86400
                total_cost = objectives[2] / 1e6
                per_sat_cost = total_cost / K
                
                print(f"{K:<12} {total_dv:<12.0f} {total_time:<12.1f} {total_cost:<12.1f} {per_sat_cost:<12.1f}")
    
    # Analysis
    print(f"\nInsights:")
    
    if len(successful_results) >= 2:
        constellation_sizes = sorted(successful_results.keys())
        
        # Calculate scaling efficiency
        single_cost = None
        for K in constellation_sizes:
            result = successful_results[K]
            best = result.get('best_constellations', [{}])[0]
            objectives = best.get('objectives', [])
            
            if len(objectives) >= 3:
                total_cost = objectives[2] / 1e6
                per_sat_cost = total_cost / K
                
                if K == 1:
                    single_cost = per_sat_cost
                elif single_cost:
                    efficiency = single_cost / per_sat_cost
                    print(f"• {K}-satellite constellation: {efficiency:.2f}x cost efficiency vs single satellite")
        
        # Coverage analysis
        max_coverage = 0
        best_coverage_K = 0
        for K in constellation_sizes:
            result = successful_results[K]
            const_metrics = result.get('constellation_metrics', {})
            coverage_stats = const_metrics.get('coverage_stats', {})
            mean_coverage = coverage_stats.get('mean', float('inf'))
            
            # Lower coverage score is better (it's minimized)
            if mean_coverage < max_coverage or max_coverage == 0:
                max_coverage = mean_coverage
                best_coverage_K = K
        
        if best_coverage_K > 0:
            print(f"• Best coverage achieved with {best_coverage_K}-satellite constellation")


def demonstrate_constellation_geometry():
    """Demonstrate constellation geometry analysis."""
    
    print(f"\n{'='*60}")
    print("CONSTELLATION GEOMETRY DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create example constellations with different geometries
    geometries = {
        "Uniform 6-sat": MultiMissionGenome(
            num_missions=6,
            plane_raan=[0, 60, 120, 180, 240, 300],  # Uniform 60° spacing
            epochs=[10000.0] * 6,
            parking_altitudes=[400.0] * 6,
            payload_masses=[1000.0] * 6
        ),
        "Clustered 6-sat": MultiMissionGenome(
            num_missions=6,  
            plane_raan=[0, 10, 20, 180, 190, 200],   # Two clusters
            epochs=[10000.0] * 6,
            parking_altitudes=[400.0] * 6,
            payload_masses=[1000.0] * 6
        ),
        "Random 6-sat": MultiMissionGenome(
            num_missions=6,
            plane_raan=[15, 87, 134, 198, 256, 321], # Random spacing
            epochs=[10000.0] * 6,
            parking_altitudes=[400.0] * 6,
            payload_masses=[1000.0] * 6
        )
    }
    
    for name, genome in geometries.items():
        print(f"\n{name}:")
        print(f"  RAAN values: {genome.plane_raan}")
        
        # Calculate spacing uniformity
        raan_array = np.array(sorted(genome.plane_raan))
        gaps = np.diff(raan_array)
        wrap_gap = 360.0 - raan_array[-1] + raan_array[0]
        all_gaps = np.append(gaps, wrap_gap)
        
        ideal_gap = 360.0 / genome.num_missions
        uniformity = np.std(all_gaps - ideal_gap)
        
        print(f"  Gap uniformity: {uniformity:.2f}° (lower is better)")
        print(f"  Geometry valid: {genome.validate_constellation_geometry()}")


def demonstrate_mission_parameters():
    """Demonstrate individual mission parameter extraction."""
    
    print(f"\n{'='*60}")
    print("MISSION PARAMETER DEMONSTRATION")
    print(f"{'='*60}")
    
    # Create a heterogeneous constellation
    genome = MultiMissionGenome(
        num_missions=4,
        epochs=[10000.0, 10002.0, 10004.0, 10006.0],      # Staggered launches
        parking_altitudes=[300.0, 400.0, 500.0, 600.0],   # Different altitudes
        plane_raan=[0.0, 90.0, 180.0, 270.0],             # Orthogonal planes
        payload_masses=[800.0, 1000.0, 1200.0, 1500.0],   # Variable payloads
        lunar_altitude=120.0,
        transfer_time=5.5
    )
    
    print(f"4-Satellite Heterogeneous Constellation:")
    print(f"{'Mission':<8} {'Epoch':<12} {'Earth Alt':<12} {'RAAN':<8} {'Payload':<10}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
    
    for i in range(genome.num_missions):
        params = genome.get_mission_parameters(i)
        print(f"{i+1:<8} {params['epoch']:<12.1f} {params['earth_orbit_alt']:<12.1f} "
              f"{params['plane_raan']:<8.1f} {params['payload_mass']:<10.1f}")
    
    print(f"\nShared Parameters:")
    print(f"  Lunar Altitude: {genome.lunar_altitude} km")
    print(f"  Transfer Time: {genome.transfer_time} days")
    
    # Demonstrate decision vector encoding
    decision_vector = genome.to_decision_vector()
    print(f"\nDecision Vector (length {len(decision_vector)}):")
    print(f"  {decision_vector}")
    
    # Verify round-trip encoding
    genome_decoded = MultiMissionGenome.from_decision_vector(decision_vector, 4)
    decision_vector_2 = genome_decoded.to_decision_vector()
    
    encoding_error = np.max(np.abs(np.array(decision_vector) - np.array(decision_vector_2)))
    print(f"  Round-trip encoding error: {encoding_error:.2e}")


def main():
    """Main demonstration function."""
    
    try:
        # Run constellation optimization demo
        results = demonstrate_constellation_optimization()
        
        # Demonstrate geometry analysis
        demonstrate_constellation_geometry()
        
        # Demonstrate mission parameters
        demonstrate_mission_parameters()
        
        print(f"\n{'='*60}")
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        logger.exception("Full traceback:")
        return None


if __name__ == '__main__':
    main()