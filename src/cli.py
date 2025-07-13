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
import time
import threading
from datetime import UTC, datetime
from typing import Any

# Add the src directory to Python path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.config.costs import CostFactors
from src.config.models import MissionConfig
from src.config.orbit import OrbitParameters
from src.config.spacecraft import PayloadSpecification
from src.lunar_horizon_optimizer import (
    LunarHorizonOptimizer,
    OptimizationConfig,
)


class ProgressTracker:
    """Progress tracking for long-running analysis with phase-based tracking."""

    def __init__(self, population_size: int, num_generations: int):
        self.population_size = population_size
        self.num_generations = num_generations
        self.start_time = time.time()
        self.current_phase = "Initializing"
        self.phase_start_time = time.time()
        self.progress_pct = 0
        self.running = True
        self.update_thread = None
        self.original_stdout = None  # Store original stdout for progress updates

        # Define analysis phases with expected completion percentages
        self.phases = {
            "Trajectory Analysis": {"start": 5, "end": 20, "weight": 15},
            "Multi-objective Optimization": {"start": 20, "end": 65, "weight": 45},
            "Economic Analysis": {"start": 65, "end": 85, "weight": 20},
            "Visualization Generation": {"start": 85, "end": 95, "weight": 10},
            "Results Compilation": {"start": 95, "end": 100, "weight": 5},
        }

        self.current_phase_info = None
        self.subphase_progress = 0.0  # Progress within current phase (0-1)

        # Estimate time based on population and generations
        base_time_per_eval = 0.8  # seconds per individual per generation
        estimated_evaluations = population_size * num_generations
        self.estimated_total_time = max(30, estimated_evaluations * base_time_per_eval)

    def set_original_stdout(self, stdout_fd):
        """Store original stdout for progress updates during output suppression."""
        self.original_stdout = stdout_fd

    def start_continuous_updates(self):
        """Start continuous progress updates in background thread."""

        def update_loop():
            while self.running:
                time.sleep(2)  # Update every 2 seconds
                if self.running:  # Check if still running
                    # Only auto-advance if no explicit phase updates are happening
                    if self.current_phase_info:
                        # Within a known phase, slowly advance subphase progress if stalled
                        phase_end = self.current_phase_info["end"]
                        if (
                            self.progress_pct < phase_end - 1
                        ):  # Leave room for real updates
                            self.subphase_progress = min(
                                0.9, self.subphase_progress + 0.05
                            )  # Slow advance
                            self.update_phase(
                                self.current_phase, self.subphase_progress
                            )
                    else:
                        # Fallback behavior for unknown phases
                        if self.progress_pct < 90:  # Conservative cap
                            self.progress_pct += 1
                            self._update_display()

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def stop_continuous_updates(self):
        """Stop continuous updates."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)

    def update_phase(self, phase_name: str, subphase_progress: float = 0.0):
        """Update current phase and subphase progress.

        Args:
            phase_name: Name of the current phase
            subphase_progress: Progress within the phase (0.0 to 1.0)
        """
        # Check if phase changed
        if self.current_phase != phase_name:
            if self.current_phase != "Initializing":
                print()  # New line for phase change
            self.current_phase = phase_name
            self.phase_start_time = time.time()
            self.current_phase_info = self.phases.get(phase_name)

        # Update subphase progress
        self.subphase_progress = max(0.0, min(1.0, subphase_progress))

        # Calculate overall progress
        if self.current_phase_info:
            phase_start = self.current_phase_info["start"]
            phase_end = self.current_phase_info["end"]
            phase_progress = (
                phase_start + (phase_end - phase_start) * self.subphase_progress
            )
            self.progress_pct = phase_progress
        else:
            # Fallback for phases not in the defined list
            self.progress_pct = min(95, self.progress_pct + 1)

        self._update_display()

    def set_phase_progress(self, phase_name: str, current: int, total: int):
        """Set progress for a phase with current/total counts.

        Args:
            phase_name: Name of the current phase
            current: Current item being processed
            total: Total items to process
        """
        progress = current / total if total > 0 else 0.0
        self.update_phase(phase_name, progress)

    def _update_display(self):
        """Internal method to update the display."""
        elapsed = time.time() - self.start_time

        # Estimate remaining time based on progress
        if self.progress_pct > 5:
            estimated_remaining = (elapsed / (self.progress_pct / 100)) - elapsed
        else:
            estimated_remaining = self.estimated_total_time - elapsed

        estimated_remaining = max(0, estimated_remaining)

        # Format time
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds/60:.1f}m"
            else:
                return f"{seconds/3600:.1f}h"

        # Create progress message
        progress_msg = f"\r🔄 {self.current_phase} | Elapsed: {format_time(elapsed)} | ETA: {format_time(estimated_remaining)} | {self.progress_pct:.1f}%"

        # Write directly to original stdout if available (bypasses suppression)
        if self.original_stdout is not None:
            try:
                import os

                os.write(self.original_stdout, (progress_msg).encode())
                os.fsync(self.original_stdout)
            except:
                # Fallback to regular print if direct write fails
                print(progress_msg, end="", flush=True)
        else:
            # Normal print when no stdout suppression
            print(progress_msg, end="", flush=True)

    def finish(self):
        """Mark analysis as complete."""
        self.stop_continuous_updates()

        # Show 100% completion
        self.progress_pct = 100
        self._update_display()

        total_time = time.time() - self.start_time

        if total_time < 60:
            print(
                f"\n✅ Analysis completed in {total_time:.1f} seconds (faster than progress tracking!)"
            )
        else:
            print(f"\n✅ Analysis completed in {total_time/60:.1f} minutes")


def estimate_analysis_time(population_size: int, num_generations: int) -> str:
    """Estimate analysis time based on population and generations."""
    # Base estimates in minutes
    base_time = 0.5  # minutes per population per generation
    estimated_minutes = (
        population_size * num_generations * base_time
    ) / 52  # Normalize to 52 pop

    if estimated_minutes < 2:
        return "1-2 minutes"
    elif estimated_minutes < 5:
        return "2-5 minutes"
    elif estimated_minutes < 15:
        return "5-15 minutes"
    elif estimated_minutes < 30:
        return "15-30 minutes"
    elif estimated_minutes < 60:
        return "30-60 minutes"
    else:
        return f"{estimated_minutes/60:.1f}+ hours"


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    if verbose:
        level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        # Suppress most logging except critical errors
        level = logging.ERROR
        format_str = "%(levelname)s: %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="%H:%M:%S",
        force=True,  # Override any existing logging config
    )

    # Specifically suppress debug logs from trajectory modules unless verbose
    if not verbose:
        logging.getLogger("src.trajectory").setLevel(logging.ERROR)
        logging.getLogger("src.optimization").setLevel(logging.ERROR)
        logging.getLogger("src.economics").setLevel(logging.ERROR)


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
    orbit_params = config_dict.get("orbit", {})
    target_orbit = OrbitParameters(
        semi_major_axis=orbit_params.get("semi_major_axis", 6778.0),  # Default LEO
        inclination=orbit_params.get("inclination", 0.0),
        eccentricity=orbit_params.get("eccentricity", 0.0),
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
        dry_mass=spacecraft_params.get("dry_mass", 5000.0),
        max_propellant_mass=spacecraft_params.get("max_propellant_mass", 3000.0),
        payload_mass=spacecraft_params.get("payload_mass", 1000.0),
        specific_impulse=spacecraft_params.get("specific_impulse", 450.0),
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
    # Set up logging first
    setup_logging(args.verbose)

    print("🚀 Starting Lunar Horizon Optimizer Analysis...")

    try:
        # Load configuration if provided
        if args.config:
            print(f"📁 Loading configuration from {args.config}")
            config_dict = load_config_from_file(args.config)
            mission_config = create_mission_config_from_dict(config_dict, args)
        else:
            print("📝 Using default configuration")
            mission_config = LunarHorizonOptimizer._create_default_mission_config()

        # Override mission name if provided
        if args.mission_name:
            mission_config.name = args.mission_name

        # Create optimization config
        # Note: PyGMO NSGA-II requires population size to be multiple of 4 and at least 5
        pop_size = args.population_size or 52  # Default to 52 (multiple of 4)
        if pop_size % 4 != 0:
            pop_size = ((pop_size // 4) + 1) * 4  # Round up to nearest multiple of 4
        if pop_size < 8:
            pop_size = 8  # Minimum viable size

        optimization_config = OptimizationConfig(
            population_size=pop_size, num_generations=args.generations or 30, seed=42
        )

        print(f"🎯 Mission: {mission_config.name}")
        print(
            f"⚙️  Optimization: {optimization_config.population_size} pop, {optimization_config.num_generations} gen"
        )

        # Estimate and display expected time
        estimated_time = estimate_analysis_time(
            optimization_config.population_size, optimization_config.num_generations
        )
        print(f"⏱️  Estimated time: {estimated_time}")
        print("💡 Use --verbose for debug output, otherwise only progress is shown")
        print()

        # Initialize progress tracker
        progress = ProgressTracker(
            optimization_config.population_size, optimization_config.num_generations
        )

        # Initialize optimizer with real integration
        progress.update_phase("Initializing", 0.0)
        optimizer = LunarHorizonOptimizer(mission_config=mission_config)
        progress.update_phase("Initializing", 1.0)

        # For quick analyses, show different message
        if (
            optimization_config.population_size * optimization_config.num_generations
            < 200
        ):
            print(
                "⚡ Quick analysis mode - progress may complete before tracking updates"
            )
        else:
            progress.start_continuous_updates()

        # Temporarily suppress output for the analysis to avoid spam
        original_stdout_fd = None
        original_stderr_fd = None
        if not args.verbose:
            # Use comprehensive output suppression for PyGMO C++ output
            import os

            # Save original file descriptors for progress tracking
            original_stdout_fd = os.dup(1)
            original_stderr_fd = os.dup(2)

            # Give progress tracker access to original stdout
            progress.set_original_stdout(original_stdout_fd)

            # Redirect both stdout and stderr to devnull
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)

        try:
            # Run analysis with proper phase-based progress tracking
            results = optimizer.analyze_mission(
                mission_name=mission_config.name,
                optimization_config=optimization_config,
                include_sensitivity=not args.no_sensitivity,
                include_isru=not args.no_isru,
                verbose=False,  # Always suppress internal verbose for clean output
                progress_tracker=progress,
            )
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
            progress.stop_continuous_updates()
            return None
        finally:
            if (
                not args.verbose
                and original_stdout_fd is not None
                and original_stderr_fd is not None
            ):
                # Restore original stdout and stderr
                os.dup2(original_stdout_fd, 1)
                os.dup2(original_stderr_fd, 2)
                os.close(original_stdout_fd)
                os.close(original_stderr_fd)

        progress.finish()

        # Export results
        output_dir = (
            args.output or f"analysis_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"💾 Exporting results to {output_dir}/")
        optimizer.export_results(results, output_dir)

        # Show results location and contents
        print(f"\n📁 Results Location: {os.path.abspath(output_dir)}/")
        print("📄 Generated Files:")
        print("   • analysis_metadata.json - Configuration and performance metrics")
        print("   • financial_summary.json - Economic analysis results")
        print("   • *.html files - Interactive visualization dashboards")
        if os.path.exists(output_dir):
            html_files = [f for f in os.listdir(output_dir) if f.endswith(".html")]
            if html_files:
                print(
                    f"   • Found {len(html_files)} visualization(s): {', '.join(html_files)}"
                )
        print(f"\n🌐 Open visualizations: open {output_dir}/*.html")

        # Print summary
        print("\n📊 Analysis Complete!")

        # Extract economic data from solution analyses
        econ_analyses = results.economic_analysis.get("solution_analyses", [])
        if econ_analyses:
            financial_summary = econ_analyses[0].get("financial_summary")
            if financial_summary:
                total_cost = financial_summary.total_investment
                roi = (
                    financial_summary.return_on_investment * 100
                )  # Convert to percentage
                npv = financial_summary.net_present_value
                print(f"   Total Cost: ${total_cost:,.0f}")
                print(f"   NPV: ${npv:,.0f}")
                print(f"   ROI: {roi:.1f}%")

        # Extract trajectory data from baseline
        baseline = results.trajectory_results.get("baseline", {})
        if baseline:
            delta_v = baseline.get("total_dv", 0)
            transfer_time = baseline.get("transfer_time", 0)
            print(f"   Delta-V: {delta_v:.0f} m/s")
            print(f"   Transfer Time: {transfer_time:.1f} days")

        return results

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return None


def config_command(args) -> None:
    """Handle the config command to generate sample configuration."""
    print("📝 Generating sample configuration file...")

    sample_config = {
        "mission": {
            "name": "Sample Lunar Mission",
            "description": "Basic lunar cargo delivery mission",
            "transfer_time": 4.5,
        },
        "spacecraft": {
            "dry_mass": 5000.0,
            "max_propellant_mass": 3000.0,
            "payload_mass": 1000.0,
            "specific_impulse": 450.0,
        },
        "costs": {
            "launch_cost_per_kg": 10000.0,
            "operations_cost_per_day": 100000.0,
            "development_cost": 1000000000.0,
            "contingency_percentage": 20.0,
            "discount_rate": 0.08,
            "learning_rate": 0.90,
            "carbon_price_per_ton_co2": 50.0,
            "co2_emissions_per_kg_payload": 2.5,
        },
        "orbit": {"semi_major_axis": 6778.0, "inclination": 0.0, "eccentricity": 0.0},
        "optimization": {"population_size": 50, "num_generations": 30, "seed": 42},
    }

    output_file = args.output or "sample_mission_config.json"

    with open(output_file, "w") as f:
        json.dump(sample_config, f, indent=2)

    print(f"✅ Sample configuration saved to {output_file}")
    print(
        "   Edit this file and use with: python src/cli.py analyze --config "
        + output_file
    )


def validate_command(args) -> None:
    """Handle the validate command to check environment and dependencies."""
    print("🔍 Validating Lunar Horizon Optimizer Environment...")
    validation_passed = True

    # Check Python version
    python_version = sys.version_info
    print(
        f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )
    if python_version.major != 3 or python_version.minor < 10:
        print("   ❌ Python 3.10+ required")
        validation_passed = False
    else:
        print("   ✅ Python version OK")

    # Check required packages
    required_packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("plotly", "Plotly"),
        ("pykep", "PyKEP"),
        ("pygmo", "PyGMO"),
        ("jax", "JAX"),
        ("pydantic", "Pydantic"),
    ]

    print("\n📦 Checking Dependencies:")
    for package, display_name in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"   ✅ {display_name}: {version}")
        except ImportError:
            print(f"   ❌ {display_name}: Not installed")
            validation_passed = False

    # Test optimizer initialization
    print("\n🚀 Testing Optimizer:")
    try:
        LunarHorizonOptimizer()
        print("   ✅ LunarHorizonOptimizer: OK")
    except Exception as e:
        print(f"   ❌ LunarHorizonOptimizer: {e}")
        validation_passed = False

    # Test configuration loading
    print("\n⚙️  Testing Configuration:")
    try:
        LunarHorizonOptimizer._create_default_mission_config()
        print("   ✅ Sample configuration: OK")
    except Exception as e:
        print(f"   ❌ Sample configuration: {e}")
        validation_passed = False

    # Test performance optimizations
    print("\n🚀 Testing Performance Optimizations:")
    try:
        from src.utils.performance import (
            get_optimization_status,
            enable_performance_optimizations,
        )

        enable_performance_optimizations()
        opt_status = get_optimization_status()

        for package, info in opt_status.items():
            status_icon = "✅" if info["available"] else "⚠️"
            version = info.get("version", "N/A")
            print(f"   {status_icon} {package.capitalize()}: {version}")

        if not any(info["available"] for info in opt_status.values()):
            print("   💡 Install speed-up packages: python install_speedup_packages.py")

    except Exception as e:
        print(f"   ⚠️ Performance optimizations: {e}")

    # Summary
    print(f"\n{'='*50}")
    if validation_passed:
        print("🎉 Environment validation PASSED!")
        print("   Ready to run lunar mission analysis")
        print("\n💡 Try: python src/cli.py sample")
    else:
        print("❌ Environment validation FAILED!")
        print("   Please install missing dependencies")
        print("\n💡 Try: conda activate py312 && pip install -r requirements.txt")
        sys.exit(1)


def create_sample_command(args):
    """Handle the sample command to run a quick demo analysis."""
    print("🚀 Running Quick Sample Analysis...")
    print("   This demonstrates basic lunar mission optimization")

    try:
        # Quick configuration for demo
        mission_config = LunarHorizonOptimizer._create_default_mission_config()
        mission_config.name = "Quick Demo Mission"

        optimization_config = OptimizationConfig(
            population_size=20, num_generations=10, seed=42
        )

        # Initialize progress tracker for sample
        estimated_time = estimate_analysis_time(20, 10)
        print(f"⏱️  Estimated time: {estimated_time}")
        print()

        progress = ProgressTracker(20, 10)

        progress.update_phase("Initializing demo", 5)
        optimizer = LunarHorizonOptimizer(mission_config=mission_config)

        progress.update_phase("Running quick optimization", 10)
        progress.start_continuous_updates()

        # Suppress output for clean progress display
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            results = optimizer.analyze_mission(
                mission_name=mission_config.name,
                optimization_config=optimization_config,
                include_sensitivity=False,
                include_isru=True,
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout

        progress.finish()

        # Export to quick_demo directory
        output_dir = "quick_demo_results"
        optimizer.export_results(results, output_dir)

        # Show results location and contents
        print(f"\n📁 Results Location: {os.path.abspath(output_dir)}/")
        print("📄 Generated Files:")
        print("   • analysis_metadata.json - Configuration and performance metrics")
        print("   • financial_summary.json - Economic analysis results")
        print("   • *.html files - Interactive visualization dashboards")
        print(f"🌐 Open visualizations: open {output_dir}/*.html")

        print("\n🎉 Quick Demo Complete!")
        print(f"   Mission: {results.mission_name}")

        # Extract economic data properly
        econ_analyses = results.economic_analysis.get("solution_analyses", [])
        if econ_analyses:
            financial_summary = econ_analyses[0].get("financial_summary")
            if financial_summary:
                total_cost = financial_summary.total_investment
                roi = financial_summary.return_on_investment * 100
                print(f"   Total Cost: ${total_cost:,.0f}")
                print(f"   ROI: {roi:.1f}%")

        # Extract trajectory data properly
        baseline = results.trajectory_results.get("baseline", {})
        if baseline:
            delta_v = baseline.get("total_dv", 0)
            print(f"   Delta-V: {delta_v:.0f} m/s")

        print(f"   Results saved to {output_dir}/")

        return results

    except Exception as e:
        print(f"❌ Sample analysis failed: {e}")
        return None


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
    analyze_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
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
