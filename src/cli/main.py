#!/usr/bin/env python3
"""Lunar Horizon Optimizer - Modern CLI Interface.

A comprehensive command-line interface for lunar mission analysis
providing scenario-based workflows with rich progress tracking
and automated report generation.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add project root to path
current_file = Path(__file__)
project_root = current_file.parents[2]
sys.path.insert(0, str(project_root))

# Initialize console first
console = Console()

try:
    from src.cli.scenario_manager import ScenarioManager
    from src.cli.progress_tracker import EnhancedProgressTracker, OptimizationCallback
    from src.cli.output_manager import OutputManager
    from src.lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig
    from src.config.models import MissionConfig
except ImportError as e:
    console.print(f"[red]❌ Import error: {e}[/red]")
    console.print("[yellow]💡 Try: pip install click rich pydantic[/yellow]")
    sys.exit(1)


def print_banner():
    """Print the application banner."""
    banner_text = Text(
        "🌙 Lunar Horizon Optimizer", style="bold blue", justify="center"
    )
    subtitle = Text(
        "Advanced Mission Analysis & Optimization Platform",
        style="dim",
        justify="center",
    )

    console.print(Panel.fit(f"{banner_text}\n{subtitle}", border_style="blue"))


@click.group()
@click.version_option(version="1.0.0-rc1", prog_name="lunar-opt")
@click.option(
    "--verbose", "-v", is_flag=True, help="Enable verbose output with detailed logging"
)
@click.pass_context
def cli(ctx, verbose):
    """🌙 Lunar Horizon Optimizer - Advanced mission analysis for Earth-Moon systems.

    The Lunar Horizon Optimizer provides comprehensive analysis capabilities for
    planning and optimizing lunar missions including:

    \b
    🚀 CORE CAPABILITIES:
    • Multi-objective trajectory optimization using PyGMO
    • Economic analysis with ISRU (In-Situ Resource Utilization) modeling
    • Monte Carlo risk assessment and sensitivity analysis
    • Interactive visualization dashboards and reporting
    • Automated scenario discovery and management

    \b
    📋 COMMON USAGE PATTERNS:

    # List all available scenarios
    ./lunar_opt.py run list

    # Get detailed information about a scenario
    ./lunar_opt.py run info 01_basic_transfer

    # Run analysis with default parameters
    ./lunar_opt.py run scenario 01_basic_transfer

    # Run quick analysis for testing
    ./lunar_opt.py run scenario 01_basic_transfer --gens 5 --population 8 --no-sensitivity

    # Run comprehensive analysis with all features
    ./lunar_opt.py run scenario 06_isru_economics --gens 50 --risk --refine --export-pdf

    \b
    📚 DOCUMENTATION:
    • User Guide: docs/guides/NEW_CLI_USER_GUIDE.md
    • CLI Reference: CLI_README.md
    • Scenarios: docs/USE_CASES.md
    • Technical Docs: docs/technical/

    \b
    🔧 ENVIRONMENT:
    Run './lunar_opt.py validate' to check your installation.
    Requires conda py312 environment with PyKEP, PyGMO, and dependencies.

    Use './lunar_opt.py COMMAND --help' for detailed command information.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Configure logging based on verbose flag
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s:%(levelname)s:%(message)s"
        )
        print_banner()
    else:
        # Suppress all logging below ERROR unless explicitly verbose
        logging.basicConfig(level=logging.ERROR, format="%(message)s")
        # Specifically silence noisy loggers
        logging.getLogger("src.trajectory").setLevel(logging.ERROR)
        logging.getLogger("src.optimization").setLevel(logging.ERROR)
        logging.getLogger("src.economics").setLevel(logging.ERROR)
        logging.getLogger("jax").setLevel(logging.ERROR)
        logging.getLogger("root").setLevel(logging.ERROR)
        # Also silence matplotlib, plotly, and other third-party loggers
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("plotly").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)


@cli.group()
def run():
    """🚀 Run lunar mission analysis scenarios.

    Execute predefined scenarios through the complete analysis pipeline with
    automatic progress tracking, result generation, and organized output.

    The analysis pipeline includes:
    1. Trajectory Generation - Compute Earth-Moon transfer trajectories
    2. Global Optimization - Multi-objective optimization with PyGMO NSGA-II
    3. Economic Analysis - Cost modeling, ROI, NPV, and ISRU benefits
    4. Risk Assessment - Monte Carlo uncertainty analysis (optional)
    5. Visualization - Interactive dashboards and plots
    6. Export - Organized results with reports and data files

    \b
    SCENARIO CATEGORIES:
    • Cargo Delivery - Basic Earth-Moon transfer missions
    • Resource Extraction - ISRU and lunar mining operations
    • Trade Studies - Multi-objective optimization and comparisons
    • Risk Analysis - Uncertainty quantification and sensitivity
    • Advanced Missions - Complex multi-phase operations

    \b
    QUICK START:
    ./lunar_opt.py run list                    # See all scenarios
    ./lunar_opt.py run scenario 01_basic_transfer  # Run basic analysis

    For detailed scenario information, use 'run info SCENARIO_NAME'
    """
    pass


@run.command("list")
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed scenario information including modules and expected results",
)
@click.option(
    "--type", "-t", help='Filter by mission type (e.g., "cargo", "resource", "trade")'
)
@click.option(
    "--complexity",
    "-c",
    help='Filter by complexity level ("beginner", "intermediate", "advanced")',
)
def list_scenarios(detailed, type, complexity):
    """📋 List available analysis scenarios.

    Displays all discovered scenarios with their descriptions, complexity levels,
    estimated runtimes, and mission types. Use filters to narrow down results.

    \b
    SCENARIO OVERVIEW:
    • 01_basic_transfer - Apollo-class cargo delivery (Beginner)
    • 02_launch_windows - Artemis crew transport optimization (Intermediate)
    • 03_propulsion_comparison - Chemical vs electric trade study (Intermediate)
    • 04_pareto_optimization - Multi-objective Gateway resupply (Advanced)
    • 05_constellation_optimization - Multi-satellite deployment (Intermediate)
    • 06_isru_economics - Lunar water mining business case (Beginner)
    • 07_environmental_economics - Carbon impact and learning curves (Intermediate)
    • 08_risk_analysis - High-risk mining with uncertainty (Advanced)
    • 09_complete_mission - Comprehensive lunar base (Advanced)
    • 10_multi_mission_campaign - Multi-mission development (Advanced)

    \b
    FILTERING EXAMPLES:
    ./lunar_opt.py run list --type="cargo"          # Cargo delivery missions
    ./lunar_opt.py run list --complexity="beginner" # Beginner-friendly scenarios
    ./lunar_opt.py run list --detailed              # Full information display
    """
    manager = ScenarioManager()

    if not manager.scenarios:
        console.print("[red]❌ No scenarios found! Check scenarios/ directory.[/red]")
        return

    # Apply filters
    scenarios = list(manager.scenarios.values())

    if type:
        scenarios = [s for s in scenarios if type.lower() in s.mission_type.lower()]

    if complexity:
        scenarios = [s for s in scenarios if complexity.lower() in s.complexity.lower()]

    if not scenarios:
        console.print("[yellow]⚠️  No scenarios match the specified filters.[/yellow]")
        return

    # Display scenarios
    if detailed:
        manager.list_scenarios(detailed=True)
    else:
        manager.list_scenarios(detailed=False)

    # Show filter summary
    if type or complexity:
        filters = []
        if type:
            filters.append(f"type='{type}'")
        if complexity:
            filters.append(f"complexity='{complexity}'")
        console.print(f"\n[dim]Filtered by: {', '.join(filters)}[/dim]")

    console.print(f"\n[green]📊 Found {len(scenarios)} scenario(s)[/green]")


@run.command("scenario")
@click.argument("scenario_name")
@click.option(
    "--output", "-o", help="Output directory (default: auto-generated with timestamp)"
)
@click.option(
    "--gens", type=int, help="Override number of optimization generations (default: 25)"
)
@click.option(
    "--population",
    "-p",
    type=int,
    help="Override population size (default: 40, must be multiple of 4)",
)
@click.option(
    "--refine", is_flag=True, help="Enable JAX gradient-based local refinement"
)
@click.option(
    "--risk", is_flag=True, help="Enable Monte Carlo risk analysis and sensitivity"
)
@click.option(
    "--no-sensitivity", is_flag=True, help="Skip economic sensitivity analysis"
)
@click.option(
    "--no-isru", is_flag=True, help="Skip ISRU (In-Situ Resource Utilization) analysis"
)
@click.option(
    "--export-pdf", is_flag=True, help="Export figures to PDF format (requires Kaleido)"
)
@click.option(
    "--open-dashboard",
    is_flag=True,
    help="Open interactive dashboard in browser after completion",
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="Use parallel/sequential optimization (default: parallel)",
)
@click.option(
    "--gpu",
    is_flag=True,
    help="Enable GPU acceleration for JAX operations (if available)",
)
@click.option(
    "--seed", type=int, help="Random seed for reproducible optimization results"
)
@click.option(
    "--include-descent",
    is_flag=True,
    help="Enable powered-descent optimization and cost modeling",
)
@click.option(
    "--3d-viz",
    "three_d_viz",
    is_flag=True,
    help="Generate 3D landing trajectory visualization (requires --include-descent)",
)
@click.pass_context
def run_scenario(
    ctx,
    scenario_name,
    output,
    gens,
    population,
    refine,
    risk,
    no_sensitivity,
    no_isru,
    export_pdf,
    open_dashboard,
    parallel,
    gpu,
    seed,
    include_descent,
    three_d_viz,
):
    """🚀 Run a specific lunar mission analysis scenario.

    SCENARIO_NAME: ID of the scenario to run (e.g., '01_basic_transfer')

    \b
    ANALYSIS PIPELINE:
    This command executes the complete 6-phase analysis pipeline:

    1. 🔧 Initialization - Load configuration and setup analysis framework
    2. 🛰️  Trajectory Generation - Compute Earth-Moon transfer trajectories using PyKEP
    3. 🎯 Global Optimization - Multi-objective optimization with PyGMO NSGA-II
    4. 🔬 Local Refinement - JAX gradient-based refinement (optional)
    5. 💰 Economic Analysis - Cost modeling, ROI, NPV, and ISRU benefits
    6. 📊 Risk Assessment - Monte Carlo uncertainty analysis (optional)
    7. 📈 Visualization - Generate interactive dashboards and plots
    8. 💾 Export - Save organized results with reports and data files

    \b
    PERFORMANCE TUNING:

    # Quick analysis for testing (fast)
    ./lunar_opt.py run scenario 01_basic_transfer --gens 5 --population 8 --no-sensitivity --no-isru

    # Standard analysis (recommended)
    ./lunar_opt.py run scenario 01_basic_transfer

    # High-fidelity analysis (comprehensive)
    ./lunar_opt.py run scenario 06_isru_economics --gens 100 --population 80 --risk --refine

    # Powered descent analysis with landing cost modeling
    ./lunar_opt.py run scenario 11_powered_descent_mission --include-descent

    \b
    OUTPUT STRUCTURE:
    results/TIMESTAMP_scenario_name/
    ├── dashboard.html              # Interactive analysis dashboard
    ├── reports/summary_report.txt  # Executive summary
    ├── data/analysis_results.json  # Complete structured results
    ├── data/scenario_config.json   # Configuration used
    └── figures/*.pdf               # Exported plots (if --export-pdf)

    \b
    POWERED DESCENT ANALYSIS:
    --include-descent flag enables lunar landing cost optimization:
    • Reads descent_parameters from scenario configuration
    • Calculates propellant costs using rocket equation
    • Includes lander hardware costs in economic analysis
    • Requires scenario with thrust, isp, and burn_time parameters

    \b
    PARAMETER CONSTRAINTS:
    • Population size must be multiple of 4 (NSGA-II requirement)
    • Minimum population size is 8 for NSGA-II
    • Higher generations/population = better accuracy but longer runtime
    • GPU acceleration requires JAX with CUDA/Metal support

    Use './lunar_opt.py run info SCENARIO_NAME' to see scenario details.
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Initialize managers
        scenario_manager = ScenarioManager()
        output_manager = OutputManager()

        # Validate scenario
        scenario = scenario_manager.get_scenario(scenario_name)
        if not scenario:
            console.print(f"[red]❌ Scenario '{scenario_name}' not found![/red]")
            console.print("\n[blue]Available scenarios:[/blue]")
            scenario_manager.list_scenarios(detailed=False)
            sys.exit(1)

        if not scenario_manager.validate_scenario(scenario_name):
            console.print(
                f"[red]❌ Scenario '{scenario_name}' validation failed![/red]"
            )
            sys.exit(1)

        # Load configuration
        config = scenario_manager.get_scenario_config(scenario_name)
        if not config:
            console.print(
                f"[red]❌ Failed to load configuration for '{scenario_name}'[/red]"
            )
            sys.exit(1)

        # Create output directory
        output_dir = output_manager.create_output_directory(scenario.name)
        console.print(f"[blue]📁 Output directory: {output_dir}[/blue]")

        # Apply parameter overrides
        opt_config = _create_optimization_config(config, gens, population, seed)

        # Extract descent parameters if flag is set
        descent_params = None
        if include_descent:
            descent_params = config.get("descent_parameters")
            if descent_params:
                console.print(
                    f"[blue]🛬 Powered descent enabled: {descent_params.get('engine_type', 'Generic')} engine[/blue]"
                )
            else:
                console.print(
                    "[yellow]⚠️  --include-descent flag set but no descent_parameters found in scenario[/yellow]"
                )

        # Configure analysis options
        skip_phases = []
        if not refine:
            skip_phases.append("Local Refinement")
        if not risk:
            skip_phases.append("Risk Assessment")

        # Initialize progress tracker
        progress_tracker = EnhancedProgressTracker(scenario.name)

        # Run analysis with live progress
        console.print(f"\n[green]🚀 Starting analysis: {scenario.name}[/green]")

        with progress_tracker.live_progress() as tracker:
            try:
                # Start tracking
                tracker.start(skip_phases=skip_phases)

                # Initialize the optimizer
                tracker.update_phase_progress(0.1, "Initializing optimizer...")
                optimizer = LunarHorizonOptimizer()

                tracker.update_phase_progress(0.5, "Loading mission configuration...")
                _convert_config_to_mission_config(config)

                tracker.update_phase_progress(1.0, "Initialization complete")
                tracker.complete_current_phase()

                # Set up optimization callback
                OptimizationCallback(tracker)

                # Run the analysis
                results = optimizer.analyze_mission(
                    mission_name=scenario.name,
                    optimization_config=opt_config,
                    include_sensitivity=not no_sensitivity,
                    include_isru=not no_isru,
                    verbose=verbose,
                    progress_tracker=tracker,
                    descent_params=descent_params,
                )

                # Complete remaining phases
                while tracker.current_phase_index < len(tracker.phases):
                    tracker.complete_current_phase()

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
                return
            except Exception as e:
                tracker.fail_current_phase(str(e))
                console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
                if verbose:
                    import traceback

                    console.print(traceback.format_exc())
                sys.exit(1)

        # Save results and generate outputs
        console.print("\n[blue]📊 Generating outputs...[/blue]")

        # Save analysis results
        output_manager.save_analysis_results(output_dir, results)

        # Save configuration
        scenario_metadata = {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
            "complexity": scenario.complexity,
            "mission_type": scenario.mission_type,
            "modules_used": scenario.modules_used,
            "analysis_timestamp": time.time(),
        }
        output_manager.save_configuration(output_dir, config, scenario_metadata)

        # Generate HTML dashboard
        dashboard_file = output_manager.generate_html_dashboard(
            output_dir, results, scenario_metadata
        )

        # Generate 3D landing visualization if requested
        if three_d_viz and include_descent:
            try:
                from src.visualization.enhanced_landing_3d_visualization import (
                    generate_3d_landing_visualization,
                )

                console.print(
                    "[blue]🌙 Generating 3D landing trajectory visualization...[/blue]"
                )
                viz_file = generate_3d_landing_visualization(
                    output_dir, results, scenario_metadata, config
                )
                if viz_file:
                    console.print(
                        f"[green]✅ 3D visualization saved: {viz_file}[/green]"
                    )
            except ImportError as e:
                console.print(f"[yellow]⚠️  3D visualization unavailable: {e}[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ 3D visualization failed: {e}[/red]")
        elif three_d_viz and not include_descent:
            console.print(
                "[yellow]⚠️  --3d-viz requires --include-descent flag[/yellow]"
            )

        # Export PDFs if requested
        if export_pdf:
            pdf_files = output_manager.export_figures_to_pdf(output_dir, results)
            if pdf_files:
                console.print(
                    f"[green]📄 Exported {len(pdf_files)} figures to PDF[/green]"
                )

        # Generate summary report
        analysis_summary = progress_tracker.get_summary()
        output_manager.generate_summary_report(
            output_dir, results, scenario_metadata, analysis_summary
        )

        # Create output summary
        output_manager.create_output_summary(output_dir)

        # Print final summary
        progress_tracker.print_summary()

        # Open dashboard if requested
        if open_dashboard and dashboard_file:
            output_manager.open_dashboard(dashboard_file)

        console.print("\n[green]✅ Analysis completed successfully![/green]")
        console.print(f"[blue]📁 Results saved to: {output_dir}[/blue]")

        # Clean up old results
        output_manager.cleanup_old_results(keep_recent=10)

    except Exception as e:
        console.print(f"[red]💥 Unexpected error: {e}[/red]")
        if ctx.obj.get("verbose", False):
            import traceback

            console.print(traceback.format_exc())
        sys.exit(1)


@run.command("info")
@click.argument("scenario_name")
def scenario_info(scenario_name):
    """ℹ️  Show detailed information about a specific scenario.

    SCENARIO_NAME: ID of the scenario to examine
    """
    manager = ScenarioManager()
    scenario = manager.get_scenario(scenario_name)

    if not scenario:
        console.print(f"[red]❌ Scenario '{scenario_name}' not found![/red]")
        return

    # Create detailed info panel
    info_lines = [
        f"[bold]Name:[/bold] {scenario.name}",
        f"[bold]ID:[/bold] {scenario.id}",
        f"[bold]Mission Type:[/bold] {scenario.mission_type}",
        f"[bold]Complexity:[/bold] {scenario.complexity}",
        f"[bold]Estimated Runtime:[/bold] {scenario.runtime_estimate}",
        "",
        "[bold]Description:[/bold]",
        scenario.description,
        "",
        "[bold]Modules Used:[/bold]",
    ]

    for module in scenario.modules_used:
        info_lines.append(f"  • {module}")

    if scenario.expected_results:
        info_lines.extend(
            [
                "",
                "[bold]Expected Results:[/bold]",
            ]
        )
        for key, value in scenario.expected_results.items():
            info_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")

    info_lines.extend(
        [
            "",
            f"[bold]Configuration File:[/bold] {scenario.file_path}",
        ]
    )

    console.print(
        Panel(
            "\n".join(info_lines), title="📊 Scenario Information", border_style="blue"
        )
    )


@cli.group()
def analyze():
    """🔬 Legacy analysis commands (backward compatibility).

    These commands provide backward compatibility with the original CLI.
    For new workflows, use 'lunar-opt run' instead.
    """
    pass


@analyze.command("config")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output directory")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def analyze_config(config_file, output, verbose):
    """🔬 Analyze a mission using a configuration file (legacy).

    This command provides backward compatibility with the original CLI.
    Consider using 'lunar-opt run scenario' for new workflows.
    """
    console.print("[yellow]⚠️  Using legacy analysis mode[/yellow]")
    console.print(
        "[blue]💡 Consider using 'lunar-opt run scenario' for improved experience[/blue]"
    )

    # Import and use the original CLI logic
    try:
        from src.cli_legacy import analyze_command

        # Convert Click arguments to argparse-like namespace
        class Args:
            def __init__(self):
                self.config = config_file
                self.output = output or "legacy_analysis"
                self.verbose = verbose
                self.mission_name = None
                self.population_size = None
                self.generations = None
                self.no_sensitivity = False
                self.no_isru = False
                self.show_plots = False
                self.learning_rate = 0.90
                self.carbon_price = 50.0

        args = Args()
        analyze_command(args)

    except ImportError:
        console.print("[red]❌ Legacy CLI not available[/red]")
        sys.exit(1)


@cli.command("validate")
def validate():
    """✅ Validate the installation and environment.

    Checks that all required dependencies are installed and functioning
    correctly, including PyKEP, PyGMO, JAX, and visualization libraries.
    """
    from src.cli.error_handling import ErrorHandler

    error_handler = ErrorHandler(console)
    issues = error_handler.validate_environment()

    if error_handler.print_validation_results(issues):
        console.print("[green]✅ All systems ready for lunar mission analysis![/green]")
    else:
        console.print(
            "[red]❌ Please resolve the issues above before running analyses.[/red]"
        )
        sys.exit(1)


@cli.command("sample")
def sample():
    """🎯 Run a quick sample analysis.

    Executes a fast demonstration analysis to verify the system
    is working correctly. Uses minimal parameters for quick execution.
    """
    console.print("[blue]🎯 Running sample analysis...[/blue]")

    try:
        # Run scenario 01_basic_transfer with minimal parameters
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            run_scenario,
            [
                "01_basic_transfer",
                "--gens",
                "5",
                "--population",
                "8",
                "--no-sensitivity",
                "--no-isru",
            ],
        )

        if result.exit_code == 0:
            console.print("[green]✅ Sample analysis completed successfully![/green]")
        else:
            console.print(f"[red]❌ Sample analysis failed: {result.output}[/red]")

    except Exception as e:
        console.print(f"[red]❌ Sample analysis error: {e}[/red]")


def _create_optimization_config(
    config: dict, gens: Optional[int], population: Optional[int], seed: Optional[int]
) -> OptimizationConfig:
    """Create optimization configuration with overrides."""
    opt_section = config.get("optimization", {})

    return OptimizationConfig(
        population_size=population or opt_section.get("population_size", 40),
        num_generations=gens or opt_section.get("num_generations", 25),
        seed=seed or opt_section.get("seed", 12345),
    )


def _convert_config_to_mission_config(config: dict) -> MissionConfig:
    """Convert scenario config to MissionConfig object."""
    from src.config.spacecraft import PayloadSpecification
    from src.config.costs import CostFactors
    from src.config.orbit import OrbitParameters

    mission_section = config.get("mission", {})
    spacecraft_section = config.get("spacecraft", {})
    costs_section = config.get("costs", {})
    orbit_section = config.get("orbit", {})

    # Create payload specification
    payload = PayloadSpecification(
        dry_mass=spacecraft_section.get("dry_mass", 8000.0),
        payload_mass=spacecraft_section.get("payload_mass", 2000.0),
        max_propellant_mass=spacecraft_section.get("max_propellant_mass", 6000.0),
        specific_impulse=spacecraft_section.get("specific_impulse", 440.0),
    )

    # Create cost factors
    cost_factors = CostFactors(
        launch_cost_per_kg=costs_section.get("launch_cost_per_kg", 8000.0),
        operations_cost_per_day=costs_section.get("operations_cost_per_day", 150000.0),
        development_cost=costs_section.get("development_cost", 500000000.0),
        contingency_percentage=costs_section.get("contingency_percentage", 25.0),
        discount_rate=costs_section.get("discount_rate", 0.07),
        learning_rate=costs_section.get("learning_rate", 0.92),
        carbon_price_per_ton_co2=costs_section.get("carbon_price_per_ton_co2", 45.0),
        co2_emissions_per_kg_payload=costs_section.get(
            "co2_emissions_per_kg_payload", 2.8
        ),
    )

    # Create orbit parameters
    target_orbit = OrbitParameters(
        semi_major_axis=orbit_section.get("semi_major_axis", 6778.0),
        eccentricity=orbit_section.get("eccentricity", 0.0),
        inclination=orbit_section.get("inclination", 28.5),
    )

    return MissionConfig(
        name=mission_section.get("name", "Lunar Mission"),
        description=mission_section.get("description", ""),
        payload=payload,
        cost_factors=cost_factors,
        mission_duration_days=mission_section.get("transfer_time", 4.5)
        * 1.0,  # Convert to days
        target_orbit=target_orbit,
    )


if __name__ == "__main__":
    cli()
