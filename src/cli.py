#!/usr/bin/env python3
"""Lunar Horizon Optimizer - Command Line Interface.

===============================================

Command-line interface for the Lunar Horizon Optimizer providing easy access
to integrated mission analysis capabilities.

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0-rc1
"""

import argparse
import json
import logging
import sys
import os
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.costs import CostFactors
from config.models import MissionConfig
from config.orbit import OrbitParameters
from config.spacecraft import PayloadSpecification
from lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig
import contextlib


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config_from_file(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(1)
    except json.JSONDecodeError:
        sys.exit(1)


def create_mission_config_from_dict(
    config_dict: dict[str, Any], cli_args=None
) -> MissionConfig:
    """Create MissionConfig from dictionary with CLI overrides."""
    mission_params = config_dict.get("mission", {})

    # Create payload specification
    payload = create_spacecraft_config_from_dict(config_dict)

    # Create cost factors with CLI overrides
    cost_factors = create_cost_factors_from_dict(config_dict, cli_args)

    # Create orbit parameters
    target_orbit = OrbitParameters(
        altitude=mission_params.get("moon_orbit_alt", 100.0),
        inclination=mission_params.get("inclination", 0.0),
        eccentricity=mission_params.get("eccentricity", 0.0),
    )

    return MissionConfig(
        name=mission_params.get("name", "CLI Mission"),
        description=mission_params.get("description", "Mission created via CLI"),
        payload=payload,
        cost_factors=cost_factors,
        mission_duration_days=mission_params.get("transfer_time", 4.5),
        target_orbit=target_orbit,
    )


def create_cost_factors_from_dict(
    config_dict: dict[str, Any], cli_args=None
) -> CostFactors:
    """Create CostFactors from dictionary with CLI overrides."""
    cost_params = config_dict.get("costs", {})

    # Use CLI arguments as overrides if provided
    learning_rate = getattr(cli_args, "learning_rate", None) or cost_params.get(
        "learning_rate", 0.90
    )
    carbon_price = getattr(cli_args, "carbon_price", None) or cost_params.get(
        "carbon_price_per_ton_co2", 50.0
    )

    return CostFactors(
        launch_cost_per_kg=cost_params.get("launch_cost_per_kg", 10000.0),
        operations_cost_per_day=cost_params.get("operations_cost_per_day", 100000.0),
        development_cost=cost_params.get("development_cost", 1e9),
        contingency_percentage=cost_params.get("contingency_percentage", 20.0),
        learning_rate=learning_rate,
        carbon_price_per_ton_co2=carbon_price,
        base_production_year=cost_params.get("base_production_year", 2024),
        cumulative_production_units=cost_params.get("cumulative_production_units", 10),
        co2_emissions_per_kg_payload=cost_params.get(
            "co2_emissions_per_kg_payload", 2.5
        ),
        environmental_compliance_factor=cost_params.get(
            "environmental_compliance_factor", 1.1
        ),
    )


def create_spacecraft_config_from_dict(
    config_dict: dict[str, Any],
) -> PayloadSpecification:
    """Create SpacecraftConfig from dictionary."""
    spacecraft_params = config_dict.get("spacecraft", {})
    return PayloadSpecification(
        # name=spacecraft_params.get("name", "CLI Spacecraft"),  # Not supported
        dry_mass=spacecraft_params.get("dry_mass", 5000.0),
        max_propellant_mass=spacecraft_params.get("propellant_mass", 3000.0),
        payload_mass=spacecraft_params.get("payload_mass", 1000.0),
        # power_system_mass=spacecraft_params.get("power_system_mass", 500.0),
        specific_impulse=spacecraft_params.get("propulsion_isp", 320.0),
    )


def create_optimization_config_from_dict(
    config_dict: dict[str, Any],
) -> OptimizationConfig:
    """Create OptimizationConfig from dictionary."""
    opt_params = config_dict.get("optimization", {})
    return OptimizationConfig(
        population_size=opt_params.get("population_size", 100),
        num_generations=opt_params.get("num_generations", 100),
        seed=opt_params.get("seed", 42),
        min_earth_alt=opt_params.get("min_earth_alt", 200.0),
        max_earth_alt=opt_params.get("max_earth_alt", 1000.0),
        min_moon_alt=opt_params.get("min_moon_alt", 50.0),
        max_moon_alt=opt_params.get("max_moon_alt", 500.0),
        min_transfer_time=opt_params.get("min_transfer_time", 3.0),
        max_transfer_time=opt_params.get("max_transfer_time", 10.0),
    )


def analyze_command(args):
    """Handle the analyze command."""
    # Load configuration if provided
    config_dict = {}
    if args.config:
        config_dict = load_config_from_file(args.config)

    # Create configuration objects with CLI overrides
    mission_config = create_mission_config_from_dict(config_dict, args)
    cost_factors = create_cost_factors_from_dict(config_dict, args)
    spacecraft_config = create_spacecraft_config_from_dict(config_dict)
    optimization_config = create_optimization_config_from_dict(config_dict)

    # Override with command line arguments
    if args.mission_name:
        mission_config.name = args.mission_name

    if args.population_size:
        optimization_config.population_size = args.population_size

    if args.generations:
        optimization_config.num_generations = args.generations

    # Initialize optimizer
    optimizer = LunarHorizonOptimizer(
        mission_config=mission_config,
        cost_factors=cost_factors,
        spacecraft_config=spacecraft_config,
    )

    # Run analysis
    results = optimizer.analyze_mission(
        mission_name=mission_config.name,
        optimization_config=optimization_config,
        include_sensitivity=not args.no_sensitivity,
        include_isru=not args.no_isru,
        verbose=args.verbose,
    )

    # Export results
    output_dir = (
        args.output or f"analysis_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    optimizer.export_results(results, output_dir)

    # Show visualizations if requested
    if args.show_plots:
        for _name, fig in results.visualization_assets.items():
            if fig is not None:
                with contextlib.suppress(Exception):
                    fig.show()

    return results


def config_command(args) -> None:
    """Handle the config command to generate sample configuration."""
    sample_config = {
        "mission": {
            "name": "Sample Lunar Mission",
            "earth_orbit_alt": 400.0,
            "moon_orbit_alt": 100.0,
            "transfer_time": 4.5,
            "departure_epoch": 10000.0,
        },
        "spacecraft": {
            "name": "Sample Spacecraft",
            "dry_mass": 5000.0,
            "propellant_mass": 3000.0,
            "payload_mass": 1000.0,
            "power_system_mass": 500.0,
            "propulsion_isp": 320.0,
        },
        "costs": {
            "launch_cost_per_kg": 10000.0,
            "operations_cost_per_day": 100000.0,
            "development_cost": 1000000000.0,
            "contingency_percentage": 20.0,
        },
        "optimization": {
            "population_size": 100,
            "num_generations": 100,
            "seed": 42,
            "min_earth_alt": 200.0,
            "max_earth_alt": 1000.0,
            "min_moon_alt": 50.0,
            "max_moon_alt": 500.0,
            "min_transfer_time": 3.0,
            "max_transfer_time": 10.0,
        },
    }

    output_file = args.output or "sample_config.json"

    with open(output_file, "w") as f:
        json.dump(sample_config, f, indent=2)


def validate_command(args) -> None:
    """Handle the validate command to check environment and dependencies."""
    validation_passed = True

    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 10:
        validation_passed = False
    else:
        pass

    # Check required packages
    required_packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("plotly", "Plotly"),
        ("pandas", "Pandas"),
        ("pykep", "PyKEP"),
        ("pygmo", "PyGMO"),
    ]

    for package, _display_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            validation_passed = False

    # Try to initialize optimizer
    try:
        LunarHorizonOptimizer()
    except Exception:
        validation_passed = False

    # Summary
    if validation_passed:
        pass
    else:
        sys.exit(1)


def create_sample_command(args):
    """Handle the sample command to run a quick demo analysis."""
    # Quick configuration for demo
    optimization_config = OptimizationConfig(
        population_size=20,
        num_generations=10,
        seed=42,
    )

    optimizer = LunarHorizonOptimizer()

    return optimizer.analyze_mission(
        mission_name="Sample Demo Mission",
        optimization_config=optimization_config,
        include_sensitivity=False,
        include_isru=False,
        verbose=True,
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lunar Horizon Optimizer - Integrated Mission Analysis",
        prog="lunar-optimizer",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run mission analysis")
    analyze_parser.add_argument(
        "--config", "-c", type=str, help="Configuration file (JSON)"
    )
    analyze_parser.add_argument("--mission-name", "-n", type=str, help="Mission name")
    analyze_parser.add_argument("--output", "-o", type=str, help="Output directory")
    analyze_parser.add_argument(
        "--population-size", "-p", type=int, help="Optimization population size"
    )
    analyze_parser.add_argument(
        "--generations", "-g", type=int, help="Optimization generations"
    )
    analyze_parser.add_argument(
        "--no-sensitivity", action="store_true", help="Skip sensitivity analysis"
    )
    analyze_parser.add_argument(
        "--no-isru", action="store_true", help="Skip ISRU analysis"
    )
    analyze_parser.add_argument(
        "--show-plots", action="store_true", help="Show plots after analysis"
    )
    analyze_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.90,
        help="Wright's law learning rate for launch cost reduction (default: 0.90)",
    )
    analyze_parser.add_argument(
        "--carbon-price",
        type=float,
        default=50.0,
        help="Carbon price per ton CO₂ for environmental cost calculation (default: $50/tCO₂)",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Generate sample configuration"
    )
    config_parser.add_argument(
        "--output", "-o", type=str, help="Output configuration file"
    )

    # Validate command
    subparsers.add_parser("validate", help="Validate environment")

    # Sample command
    subparsers.add_parser("sample", help="Run quick sample analysis")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Route to appropriate command
    if args.command == "analyze":
        return analyze_command(args)
    if args.command == "config":
        return config_command(args)
    if args.command == "validate":
        return validate_command(args)
    if args.command == "sample":
        return create_sample_command(args)
    parser.print_help()
    return None


if __name__ == "__main__":
    main()
