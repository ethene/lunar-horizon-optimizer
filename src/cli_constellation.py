"""Command-line interface for constellation optimization.

This module provides CLI support for multi-mission optimization with
the --multi K flag while maintaining full backward compatibility.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.config.costs import CostFactors
from src.optimization.multi_mission_optimizer import (
    MultiMissionOptimizer,
    optimize_constellation,
    migrate_single_to_multi
)
from src.optimization.global_optimizer import optimize_lunar_mission
from src.optimization.multi_mission_genome import create_backward_compatible_problem

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(level=level, format=format_str)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    
    return config


def parse_constellation_weights(weights_str: str) -> Dict[str, float]:
    """Parse constellation weights from command line string.
    
    Args:
        weights_str: String like "coverage=2.0,redundancy=1.0,cost=1.5"
        
    Returns:
        Dictionary of weights
    """
    weights = {}
    
    if not weights_str:
        return weights
    
    for item in weights_str.split(','):
        if '=' in item:
            key, value = item.strip().split('=', 1)
            try:
                weights[key.strip()] = float(value.strip())
            except ValueError:
                logger.warning(f"Invalid weight value: {item}")
    
    return weights


def create_cost_factors(config: Dict[str, Any]) -> CostFactors:
    """Create cost factors from configuration."""
    cost_config = config.get('costs', {})
    
    return CostFactors(
        launch_cost_per_kg=cost_config.get('launch_cost_per_kg', 10000.0),
        operations_cost_per_day=cost_config.get('operations_cost_per_day', 100000.0),
        development_cost=cost_config.get('development_cost', 1e9),
        contingency_percentage=cost_config.get('contingency_percentage', 20.0)
    )


def run_single_mission_optimization(config: Dict[str, Any], 
                                  output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run single-mission optimization (original behavior)."""
    logger.info("Running single-mission optimization (backward compatible)")
    
    # Extract configuration
    cost_factors = create_cost_factors(config)
    optimization_config = config.get('optimization', {})
    
    # Run optimization
    results = optimize_lunar_mission(
        cost_factors=cost_factors,
        optimization_config=optimization_config
    )
    
    # Save results if requested
    if output_file:
        save_results(results, output_file)
    
    return results


def run_constellation_optimization(config: Dict[str, Any], 
                                 num_missions: int,
                                 constellation_weights: Optional[Dict[str, float]] = None,
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run multi-mission constellation optimization."""
    logger.info(f"Running {num_missions}-mission constellation optimization")
    
    # Extract configuration
    cost_factors = create_cost_factors(config)
    optimization_config = config.get('optimization', {})
    constellation_config = config.get('constellation', {})
    
    # Migrate single-mission config if needed
    if 'constellation' not in config:
        logger.info("Migrating single-mission config to multi-mission")
        optimization_config = migrate_single_to_multi(optimization_config, num_missions)
    
    # Add constellation weights if provided
    if constellation_weights:
        constellation_config.setdefault('problem_params', {}).update({
            'coverage_weight': constellation_weights.get('coverage', 1.0),
            'redundancy_weight': constellation_weights.get('redundancy', 0.5)
        })
        
        # Update optimizer preferences
        optimization_config.setdefault('optimizer_params', {}).update({
            'constellation_preferences': constellation_weights
        })
    
    # Run constellation optimization
    results = optimize_constellation(
        num_missions=num_missions,
        cost_factors=cost_factors,
        optimization_config=optimization_config,
        constellation_config=constellation_config
    )
    
    # Save results if requested  
    if output_file:
        save_results(results, output_file)
    
    return results


def save_results(results: Dict[str, Any], output_file: str):
    """Save optimization results to file."""
    output_path = Path(output_file)
    
    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for serialization
    serializable_results = prepare_for_serialization(results)
    
    # Save based on file extension
    if output_file.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    elif output_file.endswith(('.yaml', '.yml')):
        with open(output_path, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
    else:
        # Default to JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def prepare_for_serialization(results: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare results for JSON/YAML serialization."""
    import numpy as np
    
    def convert_value(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_value(item) for item in obj]
        else:
            return obj
    
    return convert_value(results)


def print_results_summary(results: Dict[str, Any], num_missions: int = 1):
    """Print a summary of optimization results."""
    print("\n" + "="*60)
    print(f"OPTIMIZATION RESULTS SUMMARY ({num_missions} missions)")
    print("="*60)
    
    # Basic results
    success = results.get('success', False)
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    
    if not success:
        return
    
    # Pareto front info
    pareto_front = results.get('pareto_front', [])
    print(f"Pareto Solutions Found: {len(pareto_front)}")
    
    # Best solutions
    if num_missions == 1:
        best_solutions = results.get('pareto_solutions', [])[:3]
    else:
        best_solutions = results.get('best_constellations', [])[:3]
    
    if best_solutions:
        print(f"\nTop {len(best_solutions)} Solutions:")
        for i, sol in enumerate(best_solutions):
            objectives = sol.get('objectives', {})
            if isinstance(objectives, dict):
                delta_v = objectives.get('delta_v', 0)
                time = objectives.get('time', 0) / 86400  # Convert to days
                cost = objectives.get('cost', 0) / 1e6    # Convert to millions
            else:
                delta_v = objectives[0] if len(objectives) > 0 else 0
                time = objectives[1] / 86400 if len(objectives) > 1 else 0
                cost = objectives[2] / 1e6 if len(objectives) > 2 else 0
            
            print(f"  {i+1}. ΔV: {delta_v:.0f} m/s, Time: {time:.1f} days, Cost: ${cost:.1f}M")
    
    # Constellation metrics
    if num_missions > 1:
        const_metrics = results.get('constellation_metrics', {})
        if const_metrics:
            print(f"\nConstellation Metrics:")
            coverage_stats = const_metrics.get('coverage_stats', {})
            if coverage_stats:
                print(f"  Coverage: {coverage_stats.get('mean', 0):.2f} ± {coverage_stats.get('std', 0):.2f}")
            
            redundancy_stats = const_metrics.get('redundancy_stats', {})
            if redundancy_stats:
                print(f"  Redundancy: {redundancy_stats.get('mean', 0):.2f} ± {redundancy_stats.get('std', 0):.2f}")
    
    # Cache efficiency
    cache_stats = results.get('cache_stats', {})
    if cache_stats:
        hit_rate = cache_stats.get('hit_rate', 0)
        total_evals = cache_stats.get('total_evaluations', 0)
        print(f"\nCache Efficiency: {hit_rate:.1%} ({total_evals:,} evaluations)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lunar Horizon Optimizer - Single and Multi-Mission Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single mission (original)
  python cli_constellation.py config/lunar_mission.yaml
  
  # 3-satellite constellation
  python cli_constellation.py config/lunar_mission.yaml --multi 3
  
  # 24-satellite constellation with custom weights
  python cli_constellation.py config/constellation.yaml --multi 24 \\
      --constellation-weights "coverage=2.0,redundancy=1.0,cost=1.5"
  
  # Save results to file
  python cli_constellation.py config/lunar_mission.yaml --multi 8 \\
      --output results/constellation_8.json
        """
    )
    
    # Required arguments
    parser.add_argument('config', 
                       help='Configuration file (YAML or JSON)')
    
    # Multi-mission flag
    parser.add_argument('--multi', '-m', type=int, metavar='K',
                       help='Enable multi-mission mode with K missions')
    
    # Constellation options
    parser.add_argument('--constellation-weights', 
                       help='Constellation objective weights (e.g., "coverage=2.0,redundancy=1.0")')
    
    # Output options
    parser.add_argument('--output', '-o', 
                       help='Output file for results (JSON or YAML)')
    
    # Optimization overrides
    parser.add_argument('--population', type=int,
                       help='Override population size')
    parser.add_argument('--generations', type=int,
                       help='Override number of generations')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', 
                       help='Save logs to file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output (except errors)')
    
    args = parser.parse_args()
    
    # Setup logging
    if not args.quiet:
        setup_logging(verbose=args.verbose, log_file=args.log_file)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override optimization parameters if provided
        if args.population or args.generations:
            opt_config = config.setdefault('optimization', {})
            if args.population:
                opt_config['population_size'] = args.population
            if args.generations:
                opt_config['num_generations'] = args.generations
        
        # Parse constellation weights
        constellation_weights = None
        if args.constellation_weights:
            constellation_weights = parse_constellation_weights(args.constellation_weights)
        
        # Run optimization
        if args.multi and args.multi > 1:
            # Multi-mission constellation optimization
            results = run_constellation_optimization(
                config=config,
                num_missions=args.multi,
                constellation_weights=constellation_weights,
                output_file=args.output
            )
            num_missions = args.multi
        else:
            # Single mission optimization (backward compatible)
            results = run_single_mission_optimization(
                config=config,
                output_file=args.output
            )
            num_missions = 1
        
        # Print summary unless quiet
        if not args.quiet:
            print_results_summary(results, num_missions)
            
        # Exit with appropriate code
        sys.exit(0 if results.get('success', False) else 1)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()