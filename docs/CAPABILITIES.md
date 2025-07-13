# Lunar Horizon Optimizer - Complete Code Capabilities

This document provides a comprehensive overview of ALL classes, functions, and capabilities
across the entire Lunar Horizon Optimizer codebase.

**Generated**: 2025-07-13 03:31:21

## 📊 Codebase Statistics

- **Total Python files**: 150
- **Total classes**: 326
- **Total functions**: 340

## 📁 Module Breakdown

- ****: 3 files, 3 classes, 22 functions
- **config**: 10 files, 11 classes, 0 functions
- **config/management**: 3 files, 3 classes, 0 functions
- **economics**: 7 files, 21 classes, 9 functions
- **examples**: 10 files, 0 classes, 63 functions
- **extensibility**: 5 files, 15 classes, 4 functions
- **extensibility/examples**: 2 files, 2 classes, 0 functions
- **optimization**: 7 files, 14 classes, 11 functions
- **optimization/differentiable**: 12 files, 37 classes, 27 functions
- **scripts**: 3 files, 0 classes, 21 functions
- **tests**: 53 files, 178 classes, 113 functions
- **trajectory**: 25 files, 29 classes, 37 functions
- **trajectory/validation**: 3 files, 0 classes, 9 functions
- **utils**: 1 files, 0 classes, 16 functions
- **visualization**: 6 files, 13 classes, 8 functions

---

# Module: 

## /cli.py

**Description**: Lunar Horizon Optimizer - Command Line Interface.

- **Lines of code**: 387
- **Classes**: 0
- **Functions**: 11

### Functions

#### `setup_logging(verbose)`
Set up logging configuration.

#### `load_config_from_file(config_path)`
Load configuration from JSON file.

#### `create_mission_config_from_dict(config_dict, cli_args)`
Create MissionConfig from dictionary with CLI overrides.

#### `create_cost_factors_from_dict(config_dict, cli_args)`
Create CostFactors from dictionary with CLI overrides.

#### `create_spacecraft_config_from_dict(config_dict)`
Create SpacecraftConfig from dictionary.

#### `create_optimization_config_from_dict(config_dict)`
Create OptimizationConfig from dictionary.

#### `analyze_command(args)`
Handle the analyze command.

#### `config_command(args)`
Handle the config command to generate sample configuration.

#### `validate_command(args)`
Handle the validate command to check environment and dependencies.

#### `create_sample_command(args)`
Handle the sample command to run a quick demo analysis.

#### `main()`
Main CLI entry point.

---

## /cli_constellation.py

**Description**: Command-line interface for constellation optimization.

- **Lines of code**: 395
- **Classes**: 0
- **Functions**: 10

### Functions

#### `setup_logging(verbose, log_file)`
Setup logging configuration.

#### `load_config(config_path)`
Load configuration from YAML file.

#### `parse_constellation_weights(weights_str)`
Parse constellation weights from command line string.

#### `create_cost_factors(config)`
Create cost factors from configuration.

#### `run_single_mission_optimization(config, output_file)`
Run single-mission optimization (original behavior).

#### `run_constellation_optimization(config, num_missions, constellation_weights, output_file)`
Run multi-mission constellation optimization.

#### `save_results(results, output_file)`
Save optimization results to file.

#### `prepare_for_serialization(results)`
Prepare results for JSON/YAML serialization.

#### `print_results_summary(results, num_missions)`
Print a summary of optimization results.

#### `main()`
Main CLI entry point.

---

## /lunar_horizon_optimizer.py

**Description**: Lunar Horizon Optimizer - Main Integration Module.

- **Lines of code**: 763
- **Classes**: 3
- **Functions**: 1

### Classes

#### `OptimizationConfig`
Configuration for optimization parameters.

#### `AnalysisResults`
Container for complete mission analysis results.

#### `LunarHorizonOptimizer`
Main integration class for the Lunar Horizon Optimizer system.

**Methods:**
- `__init__()`: Initialize the Lunar Horizon Optimizer.
- `_initialize_components()`: Initialize all core analysis components.
- `analyze_mission()`: Perform comprehensive lunar mission analysis.
- `_analyze_trajectories()`: Analyze trajectory options and transfer windows.
- `_perform_optimization()`: Perform multi-objective optimization.
- `_analyze_economics()`: Perform comprehensive economic analysis.
- `_analyze_solution_economics()`: Analyze economics for a single optimization solution.
- `_create_visualizations()`: Create comprehensive visualizations.
- `_economic_model()`: Economic model for sensitivity analysis.
- `_calculate_performance_metrics()`: Calculate system performance metrics.
- `_print_analysis_summary()`: Print a summary of analysis results.
- `export_results()`: Export analysis results to files.
- `_create_default_mission_config()`: Create default mission configuration.
- `_create_default_cost_factors()`: Create default cost factors.
- `_create_default_spacecraft_config()`: Create default spacecraft configuration.

### Functions

#### `main()`
Main function demonstrating the integrated Lunar Horizon Optimizer.

---

# Module: config

## config/costs.py

**Description**: Cost-related configuration models.

- **Lines of code**: 78
- **Classes**: 1
- **Functions**: 0

### Classes

#### `CostFactors`
Economic cost factors for mission planning with learning curves and environmental costs.

---

## config/enums.py

**Description**: Enumeration types for mission configuration.

- **Lines of code**: 24
- **Classes**: 1
- **Functions**: 0

### Classes

#### `IsruResourceType`
Types of resources that can be extracted via ISRU.

---

## config/isru.py

**Description**: Configuration models for In-Situ Resource Utilization (ISRU) capabilities.

- **Lines of code**: 121
- **Classes**: 2
- **Functions**: 0

### Classes

#### `ResourceExtractionRate`
Defines the extraction rate and efficiency for a specific resource.

#### `IsruCapabilities`
Defines the capabilities and parameters of an ISRU system.

**Methods:**
- `validate_extraction_rates()`: Validate that extraction rates are provided for supported resource types.
- `calculate_power_consumption()`: Calculate total power consumption based on active resource extraction rates.

---

## config/loader.py

**Description**: Configuration loader module.

- **Lines of code**: 199
- **Classes**: 2
- **Functions**: 0

### Classes

#### `ConfigurationError`
Exception raised for configuration-related errors.

#### `ConfigLoader`
Configuration loader with validation support.

**Methods:**
- `__init__()`: Initialize the configuration loader.
- `load_file()`: Load and validate a configuration file.
- `_read_config_file()`: Read configuration from a file.
- `_merge_with_defaults()`: Merge configuration with default values.
- `_validate_config()`: Validate configuration data using Pydantic model.
- `save_config()`: Save configuration to a file.
- `load_default_config()`: Create a ConfigLoader with default lunar mission configuration.

---

## config/manager.py

**Description**: Legacy configuration manager module - DEPRECATED.

- **Lines of code**: 31
- **Classes**: 0
- **Functions**: 0

---

## config/mission_config.py

**Description**: Mission configuration data models - Backward Compatibility Module.

- **Lines of code**: 43
- **Classes**: 0
- **Functions**: 0

---

## config/models.py

**Description**: Central configuration models module.

- **Lines of code**: 123
- **Classes**: 1
- **Functions**: 0

### Classes

#### `MissionConfig`
Complete mission configuration.

**Methods:**
- `validate_mission_parameters()`: Validate overall mission parameters.

---

## config/orbit.py

**Description**: Orbital parameters and configuration models.

- **Lines of code**: 100
- **Classes**: 1
- **Functions**: 0

### Classes

#### `OrbitParameters`
Orbital parameters specification.

**Methods:**
- `validate_orbit()`: Validate orbit parameters.
- `calculate_period()`: Calculate orbital period using Kepler's Third Law.
- `calculate_velocities()`: Calculate periapsis and apoapsis velocities.

---

## config/registry.py

**Description**: Configuration registry module.

- **Lines of code**: 225
- **Classes**: 1
- **Functions**: 0

### Classes

#### `ConfigRegistry`
Registry for managing configuration templates and defaults.

**Methods:**
- `__init__()`: Initialize the configuration registry.
- `_load_default_templates()`: Load built-in default configuration templates.
- `register_template()`: Register a new configuration template.
- `get_template()`: Get a configuration template by name.
- `list_templates()`: Get list of available template names.
- `load_template_file()`: Load a template configuration from a file.
- `load_templates_dir()`: Load all template configurations from a directory.
- `save_template()`: Save a template configuration to a file.

---

## config/spacecraft.py

**Description**: Spacecraft configuration models.

- **Lines of code**: 129
- **Classes**: 2
- **Functions**: 0

### Classes

#### `PayloadSpecification`
Spacecraft payload specifications.

**Methods:**
- `validate_masses()`: Validate mass relationships.
- `calculate_delta_v()`: Calculate available delta-v using the rocket equation.

#### `SpacecraftConfig`
Complete spacecraft configuration including payload, propulsion, and subsystems.

**Methods:**
- `validate_spacecraft_masses()`: Validate spacecraft mass relationships.
- `total_mass()`: Total spacecraft mass including propellant.
- `mass_ratio()`: Mass ratio for rocket equation calculations.
- `calculate_max_delta_v()`: Calculate maximum delta-v with full propellant.

---

# Module: config/management

## config/management/config_manager.py

**Description**: Core configuration manager using composition pattern.

- **Lines of code**: 220
- **Classes**: 1
- **Functions**: 0

### Classes

#### `ConfigManager`
Manager for handling mission configuration using composition.

**Methods:**
- `__init__()`: Initialize the configuration manager.
- `active_config()`: Get the currently active configuration.
- `registry()`: Get the configuration registry from the template manager.
- `load_config()`: Load a configuration from file and set as active.
- `save_config()`: Save the active configuration to a file.
- `create_from_template()`: Create a new configuration from a template.
- `validate_config()`: Validate a configuration dictionary against the mission config model.
- `update_config()`: Update the active configuration with new values.
- `get_available_templates()`: Get list of available template names.
- `get_supported_file_formats()`: Get list of supported file formats.
- `_deep_update()`: Perform deep update of nested dictionaries.

---

## config/management/file_operations.py

**Description**: File operations for configuration management.

- **Lines of code**: 83
- **Classes**: 1
- **Functions**: 0

### Classes

#### `FileOperations`
Handles file operations for configuration management.

**Methods:**
- `__init__()`: Initialize file operations handler.
- `load_from_file()`: Load a configuration from file.
- `save_to_file()`: Save a configuration to file.
- `validate_file_path()`: Validate if a file path is suitable for configuration operations.
- `get_supported_formats()`: Get list of supported file formats.

---

## config/management/template_manager.py

**Description**: Template management for configuration creation.

- **Lines of code**: 137
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TemplateManager`
Manages configuration templates and template-based creation.

**Methods:**
- `__init__()`: Initialize template manager.
- `create_from_template()`: Create a new configuration from a template.
- `get_available_templates()`: Get list of available template names.
- `get_template()`: Get a specific template by name.
- `validate_template_overrides()`: Validate that template overrides will produce a valid configuration.
- `_apply_overrides()`: Apply override values to a configuration dictionary.

---

# Module: economics

## economics/advanced_isru_models.py

**Description**: Advanced ISRU models with time dependencies for Task 9.

- **Lines of code**: 612
- **Classes**: 3
- **Functions**: 2

### Classes

#### `ProductionProfile`
Time-dependent production profile for ISRU operations.

**Methods:**
- `__post_init__()`: Validate production profile parameters.

#### `ISRUFacility`
Represents an ISRU production facility with time-dependent characteristics.

**Methods:**
- `__post_init__()`: Validate facility configuration.

#### `TimeBasedISRUModel`
Advanced ISRU model with time-dependent production and economics.

**Methods:**
- `__init__()`: Initialize time-based ISRU model.
- `add_facility()`: Add an ISRU facility to the model.
- `calculate_production_rate()`: Calculate production rate for a resource at a specific date.
- `calculate_cumulative_production()`: Calculate cumulative production over a time period.
- `calculate_time_dependent_economics()`: Calculate economics with time-dependent production.
- `optimize_facility_deployment()`: Optimize ISRU facility deployment schedule.
- `_generate_default_facilities()`: Generate default facility configurations.

### Functions

#### `create_isru_production_forecast(resources, start_date, forecast_years, facilities)`
Create ISRU production forecast with visualization data.

#### `_calculate_breakeven_date(dates, cumulative_production, economics, resource)`
Calculate breakeven date for ISRU operations.

---

## economics/cost_models.py

**Description**: Cost models for lunar mission economic analysis - Task 5 implementation.

- **Lines of code**: 702
- **Classes**: 4
- **Functions**: 1

### Classes

#### `CostBreakdown`
Detailed cost breakdown for mission components.

**Methods:**
- `__post_init__()`: Calculate total if not provided.

#### `MissionCostModel`
Comprehensive mission cost model for lunar missions.

**Methods:**
- `__init__()`: Initialize mission cost model.
- `estimate_total_mission_cost()`: Estimate total mission cost with detailed breakdown.
- `_calculate_base_costs()`: Calculate base costs using parametric relationships.
- `_estimate_launch_cost()`: Estimate launch cost based on spacecraft mass.
- `cost_sensitivity_analysis()`: Perform cost sensitivity analysis.

#### `LaunchCostModel`
Specialized launch cost model with vehicle-specific calculations.

**Methods:**
- `__init__()`: Initialize launch cost model.
- `find_optimal_launch_vehicle()`: Find optimal launch vehicle for given payload.
- `calculate_multi_launch_strategy()`: Calculate optimal multi-launch strategy for large payloads.

#### `OperationalCostModel`
Operational cost model for lunar mission operations.

**Methods:**
- `__init__()`: Initialize operational cost model.
- `estimate_operational_costs()`: Estimate operational costs for mission duration.
- `cost_reduction_analysis()`: Analyze potential cost reduction strategies.

### Functions

#### `create_cost_model_suite()`
Create a complete suite of cost models.

---

## economics/financial_models.py

**Description**: Financial models for economic analysis - Task 5 core implementation.

- **Lines of code**: 680
- **Classes**: 5
- **Functions**: 1

### Classes

#### `CashFlow`
Represents a cash flow event in mission economics.

**Methods:**
- `__post_init__()`: Validate cash flow data.

#### `FinancialParameters`
Financial parameters for economic analysis with environmental costs.

**Methods:**
- `__post_init__()`: Validate financial parameters.
- `total_cost()`: Calculate total cost including environmental costs.

#### `CashFlowModel`
Cash flow modeling for lunar mission economics.

**Methods:**
- `__init__()`: Initialize cash flow model.
- `add_cash_flow()`: Add a cash flow event.
- `add_development_costs()`: Add development costs spread over development period.
- `add_launch_costs()`: Add launch costs for multiple launches.
- `add_operational_costs()`: Add operational costs over mission duration.
- `add_revenue_stream()`: Add revenue stream over mission duration.
- `get_cash_flows_by_category()`: Get cash flows by category.
- `get_total_by_category()`: Get total cash flows by category.
- `get_annual_cash_flows()`: Get annual cash flow totals.

#### `NPVAnalyzer`
Net Present Value (NPV) analysis for lunar missions.

**Methods:**
- `__init__()`: Initialize NPV analyzer.
- `calculate_npv()`: Calculate Net Present Value of cash flows.
- `calculate_irr()`: Calculate Internal Rate of Return (IRR).
- `_calculate_irr_newton_raphson()`: Calculate IRR using Newton-Raphson method.
- `calculate_payback_period()`: Calculate payback period in years.
- `sensitivity_analysis()`: Perform sensitivity analysis on NPV.
- `_apply_sensitivity_change()`: Apply sensitivity change to cash flow model.

#### `ROICalculator`
Return on Investment (ROI) calculator for lunar missions.

**Methods:**
- `__init__()`: Initialize ROI calculator.
- `calculate_simple_roi()`: Calculate simple ROI.
- `calculate_annualized_roi()`: Calculate annualized ROI.
- `calculate_risk_adjusted_roi()`: Calculate risk-adjusted ROI using CAPM.
- `compare_investments()`: Compare multiple investment options.

### Functions

#### `create_mission_cash_flow_model(mission_config)`
Create a complete cash flow model for a lunar mission.

---

## economics/isru_benefits.py

**Description**: ISRU (In-Situ Resource Utilization) benefits analysis for Task 5.

- **Lines of code**: 684
- **Classes**: 3
- **Functions**: 1

### Classes

#### `ResourceProperty`
Properties of a lunar resource.

#### `ResourceValueModel`
Economic value model for lunar resources.

**Methods:**
- `__init__()`: Initialize resource value model.
- `calculate_resource_value()`: Calculate economic value of extracted resource.
- `estimate_extraction_costs()`: Estimate costs for resource extraction.

#### `ISRUBenefitAnalyzer`
Comprehensive ISRU benefit analysis for lunar missions.

**Methods:**
- `__init__()`: Initialize ISRU benefit analyzer.
- `analyze_isru_economics()`: Perform comprehensive ISRU economic analysis.
- `compare_isru_vs_earth_supply()`: Compare ISRU production vs Earth supply for space operations.
- `_get_facility_type()`: Map resource name to facility type.
- `_calculate_production_profile()`: Calculate monthly production profile with ramp-up.
- `_calculate_isru_npv()`: Calculate NPV for ISRU operation.
- `_calculate_break_even()`: Calculate break-even metrics.
- `_perform_risk_analysis()`: Perform Monte Carlo risk analysis.

### Functions

#### `analyze_lunar_resource_portfolio(resources, facility_scale, operation_years)`
Analyze a portfolio of lunar resources for ISRU development.

---

## economics/reporting.py

**Description**: Economic reporting and summary generation for Task 5 completion.

- **Lines of code**: 746
- **Classes**: 2
- **Functions**: 1

### Classes

#### `FinancialSummary`
Financial summary data structure for lunar mission economics.

**Methods:**
- `__post_init__()`: Set default analysis date if not provided.

#### `EconomicReporter`
Comprehensive economic reporting for lunar mission analysis.

**Methods:**
- `__init__()`: Initialize economic reporter.
- `generate_executive_summary()`: Generate executive summary report.
- `generate_detailed_financial_report()`: Generate detailed financial analysis report.
- `generate_comparison_report()`: Generate comparative analysis report for multiple alternatives.
- `export_to_csv()`: Export data to CSV format.
- `export_to_json()`: Export data to JSON format.
- `generate_dashboard_data()`: Generate data structure for economic dashboard visualization.
- `_assess_project_viability()`: Assess overall project viability.
- `_generate_key_insights()`: Generate key insights section.
- `_format_cash_flow_section()`: Format cash flow analysis section.
- `_format_npv_section()`: Format NPV analysis section.
- `_format_cost_section()`: Format cost analysis section.
- `_format_isru_section()`: Format ISRU analysis section.
- `_format_sensitivity_section()`: Format sensitivity analysis section.
- `_format_scenario_section()`: Format scenario analysis section.
- `_create_comparison_table()`: Create comparison table for alternatives.
- `_rank_alternatives()`: Rank alternatives using weighted scoring.
- `_justify_recommendation()`: Justify the recommended alternative.

### Functions

#### `create_financial_summary_from_analysis(analysis_results)`
Create FinancialSummary from analysis results.

---

## economics/scenario_comparison.py

**Description**: Advanced scenario comparison tools for Task 9.

- **Lines of code**: 688
- **Classes**: 3
- **Functions**: 1

### Classes

#### `ScenarioDefinition`
Definition of a mission scenario for comparison.

#### `ScenarioResults`
Results from scenario analysis.

#### `AdvancedScenarioComparator`
Advanced tools for comparing multiple mission scenarios.

**Methods:**
- `__init__()`: Initialize scenario comparator.
- `add_scenario()`: Add a scenario for comparison.
- `analyze_scenario()`: Analyze a single scenario.
- `compare_all_scenarios()`: Compare all scenarios and return summary DataFrame.
- `rank_scenarios()`: Rank scenarios using multi-criteria decision analysis.
- `generate_decision_matrix()`: Generate decision matrix for scenario comparison.
- `_calculate_scenario_costs()`: Calculate costs for a scenario.
- `_generate_cash_flows()`: Generate cash flows for scenario.
- `_calculate_scenario_revenue()`: Calculate total revenue for scenario.
- `_calculate_isru_revenue()`: Calculate annual ISRU revenue.
- `_calculate_risk_adjusted_npv()`: Calculate risk-adjusted NPV.
- `_calculate_success_probability()`: Calculate probability of mission success.
- `_run_sensitivity_analysis()`: Run sensitivity analysis for scenario.
- `_run_monte_carlo_simulation()`: Run Monte Carlo simulation for scenario.
- `_calculate_break_even_year()`: Calculate break-even year from cash flow profile.
- `_normalize_metric()`: Normalize metric to 0-1 scale.
- `_calculate_strategic_score()`: Calculate strategic alignment score.

### Functions

#### `create_scenario_comparison_report(comparator, output_format)`
Create comprehensive scenario comparison report.

---

## economics/sensitivity_analysis.py

**Description**: Economic sensitivity analysis module for Task 5 completion.

- **Lines of code**: 702
- **Classes**: 1
- **Functions**: 2

### Classes

#### `EconomicSensitivityAnalyzer`
Economic sensitivity and scenario analysis for lunar missions.

**Methods:**
- `__init__()`: Initialize economic sensitivity analyzer.
- `one_way_sensitivity()`: Perform one-way sensitivity analysis.
- `tornado_diagram_data()`: Generate data for tornado diagram visualization.
- `scenario_analysis()`: Perform scenario analysis with multiple defined scenarios.
- `monte_carlo_simulation()`: Perform Monte Carlo simulation for risk analysis.
- `_calculate_elasticity()`: Calculate elasticity at base point.
- `_generate_monte_carlo_samples()`: Generate Monte Carlo samples from specified distributions.
- `_calculate_skewness()`: Calculate skewness of data.
- `_calculate_kurtosis()`: Calculate kurtosis of data.
- `_analyze_parameter_correlations()`: Analyze correlations between parameters and results.
- `comprehensive_sensitivity_report()`: Generate comprehensive sensitivity analysis report.
- `_generate_sensitivity_summary()`: Generate summary of sensitivity analysis results.

### Functions

#### `create_lunar_mission_scenarios()`
Create predefined scenarios for lunar mission analysis.

#### `create_parameter_distributions()`
Create parameter distributions for Monte Carlo analysis.

---

# Module: examples

## examples/advanced_trajectory_test.py

**Description**: Advanced Trajectory Generation Integration Test

- **Lines of code**: 320
- **Classes**: 0
- **Functions**: 7

### Functions

#### `test_lambert_solver_integration()`
Test Lambert solver integration with trajectory generation.

#### `test_trajectory_generation_with_lambert()`
Test trajectory generation using Lambert solver.

#### `test_patched_conics_integration()`
Test patched conics approximation integration.

#### `test_optimal_timing_integration()`
Test optimal timing calculator integration.

#### `test_main_optimizer_integration()`
Test integration with main optimizer.

#### `test_visualization_integration()`
Test visualization integration with trajectory data.

#### `main()`
Run all advanced trajectory integration tests.

---

## examples/constellation_optimization_demo.py

**Description**: Constellation Optimization Demonstration

- **Lines of code**: 334
- **Classes**: 0
- **Functions**: 6

### Functions

#### `demonstrate_constellation_optimization()`
Demonstrate constellation optimization with different sizes.

#### `print_constellation_summary(result, K)`
Print summary for a constellation optimization result.

#### `compare_constellations(results)`
Compare optimization results across different constellation sizes.

#### `demonstrate_constellation_geometry()`
Demonstrate constellation geometry analysis.

#### `demonstrate_mission_parameters()`
Demonstrate individual mission parameter extraction.

#### `main()`
Main demonstration function.

---

## examples/continuous_thrust_demo.py

**Description**: Continuous-Thrust Trajectory Optimization Demo.

- **Lines of code**: 365
- **Classes**: 0
- **Functions**: 6

### Functions

#### `demo_basic_dynamics()`
Demonstrate basic continuous-thrust dynamics.

#### `demo_orbit_raising()`
Demonstrate realistic orbit raising maneuver.

#### `demo_thrust_angle_optimization()`
Demonstrate thrust angle optimization.

#### `demo_comparison_with_chemical()`
Compare electric vs chemical propulsion.

#### `plot_trajectory_example(trajectory)`
Plot trajectory if available and matplotlib works.

#### `main()`
Run all continuous-thrust demos.

---

## examples/cost_comparison_demo.py

**Description**: Cost Comparison Demo: Before vs After Learning Curves and Environmental Costs.

- **Lines of code**: 271
- **Classes**: 0
- **Functions**: 5

### Functions

#### `demo_learning_curve_impact()`
Demonstrate the impact of Wright's law learning curves.

#### `demo_environmental_cost_impact()`
Demonstrate the impact of CO₂ environmental costs.

#### `demo_cost_calculator_comparison()`
Compare old vs new cost calculation methodology.

#### `demo_cli_flag_examples()`
Demonstrate CLI flag usage examples.

#### `main()`
Run all cost comparison demonstrations.

---

## examples/differentiable_optimization_demo.py

**Description**: Differentiable Optimization Demo

- **Lines of code**: 400
- **Classes**: 0
- **Functions**: 8

### Functions

#### `demo_jax_availability()`
Demonstrate JAX and Diffrax availability.

#### `demo_basic_differentiable_models()`
Demonstrate basic differentiable models.

#### `demo_gradient_optimization()`
Demonstrate gradient-based optimization.

#### `demo_batch_optimization()`
Demonstrate batch optimization for multiple starting points.

#### `demo_performance_comparison()`
Compare JAX vs numerical differentiation performance.

#### `demo_integration_with_pygmo()`
Demonstrate integration between PyGMO global and JAX local optimization.

#### `demo_advanced_features()`
Demonstrate advanced JAX features.

#### `main()`
Run the complete differentiable optimization demonstration.

---

## examples/final_integration_test.py

**Description**: Final Integration Test - Comprehensive PRD Compliance Check

- **Lines of code**: 345
- **Classes**: 0
- **Functions**: 8

### Functions

#### `test_global_optimization_api()`
Test global optimization API with find_pareto_front.

#### `test_economic_dashboard_integration()`
Test economic dashboard with scenario comparison.

#### `test_trajectory_generation_integration()`
Test advanced trajectory generation with Lambert solvers.

#### `test_integrated_dashboard()`
Test integrated dashboard functionality.

#### `test_workflow_automation()`
Test cross-module workflow automation.

#### `test_configuration_system()`
Test configuration system compatibility.

#### `calculate_prd_compliance_improvement()`
Calculate the improvement in PRD compliance.

#### `main()`
Run final integration test and PRD compliance check.

---

## examples/integration_test.py

**Description**: Integration Test - Demonstrating Fixed Components

- **Lines of code**: 289
- **Classes**: 0
- **Functions**: 6

### Functions

#### `test_global_optimization_fix()`
Test the fixed global optimization API.

#### `test_economic_dashboard_fix()`
Test the fixed economic dashboard.

#### `test_integrated_dashboard_fix()`
Test the new integrated dashboard.

#### `test_workflow_automation()`
Test cross-module workflow automation.

#### `test_configuration_compatibility()`
Test configuration system compatibility.

#### `main()`
Run all integration tests.

---

## examples/quick_start.py

**Description**: Quick Start Example for Lunar Horizon Optimizer

- **Lines of code**: 217
- **Classes**: 0
- **Functions**: 5

### Functions

#### `example_1_basic_configuration()`
Example 1: Create and validate a basic mission configuration.

#### `example_2_economic_analysis()`
Example 2: Perform basic economic analysis.

#### `example_3_trajectory_calculations()`
Example 3: Basic trajectory calculations.

#### `example_4_isru_benefits()`
Example 4: Calculate ISRU benefits.

#### `main()`
Run all examples.

---

## examples/simple_trajectory_test.py

**Description**: Simple Trajectory Integration Test

- **Lines of code**: 301
- **Classes**: 0
- **Functions**: 6

### Functions

#### `test_basic_trajectory_integration()`
Test basic trajectory generation integration.

#### `test_trajectory_data_structure()`
Test trajectory data structure for integration.

#### `test_lambert_solver_direct()`
Test Lambert solver directly.

#### `test_visualization_compatibility()`
Test visualization compatibility.

#### `test_integration_with_existing_code()`
Test integration with existing code paths.

#### `main()`
Run all simple trajectory integration tests.

---

## examples/working_example.py

**Description**: Working Example for Lunar Horizon Optimizer

- **Lines of code**: 285
- **Classes**: 0
- **Functions**: 6

### Functions

#### `example_1_configuration()`
Working configuration example.

#### `example_2_jax_optimization()`
Working JAX optimization example.

#### `example_3_economics()`
Working economics example.

#### `example_4_isru_analysis()`
Working ISRU analysis example.

#### `example_5_visualization()`
Working visualization example.

#### `main()`
Run all working examples.

---

# Module: extensibility

## extensibility/base_extension.py

**Description**: Base extension interface for Task 10 extensibility framework.

- **Lines of code**: 285
- **Classes**: 4
- **Functions**: 1

### Classes

#### `ExtensionType`
Types of extensions supported by the system.

#### `ExtensionMetadata`
Metadata describing an extension.

**Methods:**
- `__post_init__()`: Initialize default values.

#### `BaseExtension`
Abstract base class for all extensions.

**Methods:**
- `__init__()`: Initialize the extension.
- `initialize()`: Initialize the extension.
- `validate_configuration()`: Validate the extension configuration.
- `get_capabilities()`: Get the capabilities provided by this extension.
- `enable()`: Enable the extension.
- `disable()`: Disable the extension.
- `is_enabled()`: Check if the extension is enabled.
- `is_initialized()`: Check if the extension is initialized.
- `shutdown()`: Shutdown the extension.
- `update_configuration()`: Update the extension configuration.
- `get_status()`: Get the current status of the extension.
- `__str__()`: String representation of the extension.
- `__repr__()`: Detailed string representation of the extension.

#### `FlightStageExtension`
Specialized extension for flight stages.

**Methods:**
- `plan_trajectory()`: Plan a trajectory for this flight stage.
- `calculate_delta_v()`: Calculate delta-v requirements for the trajectory.
- `estimate_cost()`: Estimate costs for this flight stage.
- `get_capabilities()`: Get flight stage capabilities.

### Functions

#### `create_extension_from_config(config)`
Factory function to create extensions from configuration.

---

## extensibility/data_transform.py

**Description**: Data transformation layer for the Task 10 extensibility framework.

- **Lines of code**: 546
- **Classes**: 3
- **Functions**: 0

### Classes

#### `DataFormat`
Supported data formats for transformations.

#### `TransformationRule`
Definition of a data transformation rule.

**Methods:**
- `__post_init__()`: Initialize default values.

#### `DataTransformLayer`
Data transformation and validation layer for extensibility.

**Methods:**
- `__init__()`: Initialize the data transformation layer.
- `register_transformation_rule()`: Register a new transformation rule.
- `transform_data()`: Transform data from one format to another.
- `normalize_trajectory_state()`: Normalize trajectory state to standard format.
- `normalize_optimization_result()`: Normalize optimization result to standard format.
- `normalize_cost_breakdown()`: Normalize cost breakdown to standard format.
- `convert_units()`: Convert between different units.
- `validate_extension_data()`: Validate data for a specific extension type.
- `_register_default_transformations()`: Register default transformation rules.
- `_register_default_schemas()`: Register default data format schemas.
- `_register_unit_conversions()`: Register unit conversion factors.
- `_validate_transformation_rule()`: Validate a transformation rule.
- `_find_transformation_rule()`: Find transformation rule for format conversion.
- `_validate_data_format()`: Validate data against format schema.
- `_apply_transformation()`: Apply transformation rule to data.
- `_normalize_vector()`: Normalize vector to standard list format.
- `_normalize_objectives()`: Normalize objectives to standard list format.
- `_get_unit_conversion_factor()`: Get conversion factor between units.
- `_get_extension_schema()`: Get validation schema for extension type.
- `_validate_field_type()`: Validate field type.
- `_set_nested_field()`: Set value in nested dictionary using dot notation.

---

## extensibility/extension_manager.py

**Description**: Extension manager for the Task 10 extensibility framework.

- **Lines of code**: 385
- **Classes**: 2
- **Functions**: 0

### Classes

#### `ExtensionLoadError`
Exception raised when extension loading fails.

#### `ExtensionManager`
Centralized manager for all system extensions.

**Methods:**
- `__init__()`: Initialize the extension manager.
- `load_extensions_from_config()`: Load extensions from configuration file.
- `load_extension_from_config()`: Load a single extension from configuration.
- `register_extension()`: Register an extension with the manager.
- `unregister_extension()`: Unregister an extension.
- `get_extension()`: Get an extension by name.
- `get_extensions_by_type()`: Get all extensions of a specific type.
- `list_extensions()`: List all registered extensions.
- `enable_extension()`: Enable an extension.
- `disable_extension()`: Disable an extension.
- `shutdown_all_extensions()`: Shutdown all registered extensions.
- `get_system_status()`: Get comprehensive system status.
- `_check_dependencies()`: Check if extension dependencies are satisfied.
- `_create_generic_extension_class()`: Create a generic extension class for unknown extensions.
- `execute_extension_method()`: Execute a method on a specific extension.

---

## extensibility/plugin_interface.py

**Description**: Plugin interface for the Task 10 extensibility framework.

- **Lines of code**: 513
- **Classes**: 5
- **Functions**: 0

### Classes

#### `PluginResult`
Standard result format for plugin operations.

**Methods:**
- `__post_init__()`: Initialize default values.

#### `TrajectoryPlannerInterface`
Interface for trajectory planning plugins.

**Methods:**
- `plan_trajectory()`: Plan a trajectory between two states.
- `calculate_delta_v()`: Calculate total delta-v for trajectory.
- `get_maneuver_sequence()`: Get sequence of maneuvers for trajectory.

#### `CostAnalyzerInterface`
Interface for cost analysis plugins.

**Methods:**
- `estimate_mission_cost()`: Estimate total mission cost.
- `breakdown_costs()`: Break down costs by category.
- `analyze_cost_drivers()`: Identify primary cost drivers.

#### `OptimizerInterface`
Interface for optimization plugins.

**Methods:**
- `optimize()`: Perform optimization.
- `get_optimization_history()`: Get optimization iteration history.
- `set_algorithm_parameters()`: Configure optimization algorithm parameters.

#### `PluginInterface`
Main interface for plugin interactions with the system.

**Methods:**
- `__init__()`: Initialize the plugin interface.
- `register_plugin()`: Register a plugin with the interface.
- `unregister_plugin()`: Unregister a plugin.
- `get_trajectory_planners()`: Get list of available trajectory planning plugins.
- `get_cost_analyzers()`: Get list of available cost analysis plugins.
- `get_optimizers()`: Get list of available optimization plugins.
- `plan_trajectory()`: Plan trajectory using specified plugin.
- `estimate_cost()`: Estimate cost using specified plugin.
- `optimize()`: Perform optimization using specified plugin.
- `get_plugin_capabilities()`: Get capabilities of a specific plugin.
- `list_all_plugins()`: List all registered plugins with their capabilities.
- `_get_plugin()`: Get plugin by name.
- `_validate_plugin_interfaces()`: Validate that plugin implements required interfaces.
- `_implements_interface()`: Check if plugin implements a specific interface.

---

## extensibility/registry.py

**Description**: Extension registry for the Task 10 extensibility framework.

- **Lines of code**: 317
- **Classes**: 1
- **Functions**: 3

### Classes

#### `ExtensionRegistry`
Registry for managing extension class definitions and discovery.

**Methods:**
- `__init__()`: Initialize the extension registry.
- `register_extension_class()`: Register an extension class with the registry.
- `unregister_extension_class()`: Unregister an extension class.
- `get_extension_class()`: Get an extension class by name.
- `get_extension_classes_by_type()`: Get all extension classes of a specific type.
- `list_registered_extensions()`: List all registered extension classes.
- `get_extension_count()`: Get total number of registered extensions.
- `get_extension_count_by_type()`: Get extension counts by type.
- `validate_extension_class()`: Validate that an extension class meets requirements.
- `find_extensions_by_capability()`: Find extensions that provide a specific capability.
- `get_registry_status()`: Get comprehensive registry status.
- `clear_registry()`: Clear all registered extensions.
- `import_extension_module()`: Import an extension module and auto-register discovered extensions.

### Functions

#### `get_global_registry()`
Get the global extension registry instance.

#### `register_extension(name, extension_class, extension_type)`
Convenience function to register an extension with the global registry.

#### `get_extension_class(name)`
Convenience function to get an extension class from the global registry.

---

# Module: extensibility/examples

## extensibility/examples/custom_cost_model.py

**Description**: Example custom cost model extension.

- **Lines of code**: 627
- **Classes**: 1
- **Functions**: 0

### Classes

#### `CustomCostModel`
Example custom cost model extension.

**Methods:**
- `__init__()`: Initialize the custom cost model extension.
- `initialize()`: Initialize the custom cost model extension.
- `validate_configuration()`: Validate the extension configuration.
- `get_capabilities()`: Get custom cost model capabilities.
- `estimate_mission_cost()`: Estimate total mission cost using custom model.
- `breakdown_costs()`: Break down costs by detailed categories.
- `analyze_cost_drivers()`: Identify and analyze primary cost drivers.
- `_initialize_cost_relationships()`: Initialize cost estimating relationships (CERs).
- `_initialize_learning_curves()`: Initialize learning curve parameters.
- `_initialize_risk_factors()`: Initialize risk factors for different cost categories.
- `_calculate_parametric_costs()`: Calculate base costs using parametric relationships.
- `_apply_trl_adjustments()`: Apply technology readiness level adjustments.
- `_apply_learning_curve()`: Apply learning curve effects.
- `_apply_risk_adjustments()`: Apply risk adjustments to costs.
- `_generate_cost_breakdown()`: Generate comprehensive cost breakdown.
- `_calculate_confidence_intervals()`: Calculate confidence intervals for cost estimates.
- `_analyze_cost_drivers()`: Analyze primary cost drivers.
- `_get_cost_assumptions()`: Get list of cost model assumptions.
- `_get_sensitivity_parameters()`: Get list of parameters for sensitivity analysis.

---

## extensibility/examples/lunar_descent_extension.py

**Description**: Example lunar descent flight stage extension.

- **Lines of code**: 570
- **Classes**: 1
- **Functions**: 0

### Classes

#### `LunarDescentExtension`
Example extension for lunar descent trajectory planning.

**Methods:**
- `__init__()`: Initialize the lunar descent extension.
- `initialize()`: Initialize the lunar descent extension.
- `validate_configuration()`: Validate the extension configuration.
- `plan_trajectory()`: Plan a lunar descent trajectory.
- `calculate_delta_v()`: Calculate total delta-v for the descent trajectory.
- `estimate_cost()`: Estimate costs for the lunar descent stage.
- `get_capabilities()`: Get lunar descent extension capabilities.
- `_initialize_descent_algorithms()`: Initialize descent planning algorithms.
- `_plan_powered_descent()`: Plan the powered descent trajectory.
- `_calculate_guidance_thrust()`: Calculate required thrust for explicit guidance.
- `_calculate_thrust_direction()`: Calculate thrust direction vector.
- `_calculate_fuel_consumption()`: Calculate total fuel consumption.
- `_calculate_landing_accuracy()`: Calculate landing accuracy (distance from target).
- `_generate_maneuver_sequence()`: Generate sequence of maneuvers for the descent.
- `_explicit_guidance()`: Explicit guidance algorithm implementation.
- `_gravity_turn_guidance()`: Gravity turn guidance algorithm implementation.
- `_apollo_style_guidance()`: Apollo-style guidance algorithm implementation.

---

# Module: optimization

## optimization/cost_integration.py

**Description**: Cost integration module for economic objectives in global optimization.

- **Lines of code**: 574
- **Classes**: 2
- **Functions**: 3

### Classes

#### `CostCalculator`
Economic cost calculator for lunar mission optimization.

**Methods:**
- `__init__()`: Initialize cost calculator with learning curves and environmental costs.
- `calculate_mission_cost()`: Calculate total mission cost based on trajectory parameters.
- `calculate_trajectory_cost()`: Calculate trajectory cost (alias for calculate_mission_cost for backward compatibility).
- `_calculate_propellant_cost()`: Calculate propellant-related costs.
- `_calculate_launch_cost()`: Calculate launch costs with learning curve adjustments.
- `_calculate_operations_cost()`: Calculate operational costs based on mission duration.
- `_calculate_development_cost()`: Calculate amortized development costs.
- `_calculate_altitude_cost()`: Calculate altitude-dependent cost factors.
- `calculate_cost_breakdown()`: Calculate detailed cost breakdown with learning curves and environmental costs.

#### `EconomicObjectives`
Economic objective functions for multi-objective optimization.

**Methods:**
- `__init__()`: Initialize economic objectives.
- `minimize_total_cost()`: Objective function to minimize total mission cost.
- `minimize_cost_per_kg()`: Objective function to minimize cost per kg of payload.
- `maximize_cost_efficiency()`: Objective function to maximize cost efficiency (minimize negative efficiency).
- `calculate_roi_objective()`: Calculate ROI-based objective (minimize negative ROI).

### Functions

#### `launch_price(year, base_price, learning_rate, base_year, cumulative_units_base)`
Calculate launch price using Wright's law learning curve.

#### `co2_cost(payload_mass_kg, co2_per_kg, price_per_ton)`
Calculate CO₂ environmental cost.

#### `create_cost_calculator(launch_cost_per_kg, operations_cost_per_day, development_cost, contingency_percentage)`
Create cost calculator with specified parameters.

---

## optimization/global_optimizer.py

**Description**: Global Optimization Module using PyGMO for Task 4 completion.

- **Lines of code**: 602
- **Classes**: 2
- **Functions**: 1

### Classes

#### `LunarMissionProblem`
PyGMO problem implementation for lunar mission optimization.

**Methods:**
- `__init__()`: Initialize the lunar mission optimization problem.
- `fitness()`: Evaluate fitness for multi-objective optimization.
- `get_bounds()`: Get optimization bounds for decision variables.
- `get_nobj()`: Get number of objectives.
- `get_nec()`: Get number of equality constraints.
- `get_nic()`: Get number of inequality constraints.
- `get_name()`: Get problem name.
- `get_cache_stats()`: Get trajectory cache statistics.
- `clear_cache()`: Clear the trajectory cache.
- `_create_cache_key()`: Create cache key for parameter combination.

#### `GlobalOptimizer`
PyGMO-based global optimizer for lunar mission design.

**Methods:**
- `__init__()`: Initialize the global optimizer.
- `optimize()`: Run multi-objective optimization.
- `get_best_solutions()`: Get best solutions from Pareto front based on preferences.
- `_extract_pareto_front()`: Extract Pareto front from population.
- `_extract_pareto_solutions()`: Extract complete Pareto solutions with parameters and objectives.
- `_normalize_objectives()`: Normalize objectives for weighted ranking.
- `_log_population_stats()`: Log population statistics.

### Functions

#### `optimize_lunar_mission(cost_factors, optimization_config)`
Convenience function for lunar mission optimization.

---

## optimization/multi_mission_genome.py

**Description**: Multi-mission genome design for constellation optimization.

- **Lines of code**: 596
- **Classes**: 2
- **Functions**: 1

### Classes

#### `MultiMissionGenome`
Multi-mission genome encoding K simultaneous lunar transfers.

**Methods:**
- `__post_init__()`: Validate genome structure and initialize missing values.
- `from_decision_vector()`: Create genome from PyGMO decision vector.
- `to_decision_vector()`: Convert genome to PyGMO decision vector.
- `get_mission_parameters()`: Get parameters for specific mission.
- `validate_constellation_geometry()`: Validate constellation geometry constraints.

#### `MultiMissionProblem`
PyGMO problem for multi-mission constellation optimization.

**Methods:**
- `__init__()`: Initialize multi-mission optimization problem.
- `fitness()`: Evaluate fitness for multi-mission optimization.
- `get_bounds()`: Get optimization bounds for decision variables.
- `get_nobj()`: Get number of objectives.
- `get_nec()`: Get number of equality constraints.
- `get_nic()`: Get number of inequality constraints.
- `get_name()`: Get problem name.
- `_validate_bounds()`: Validate genome parameters are within bounds.
- `_get_penalty_values()`: Get penalty values for invalid solutions.
- `_calculate_coverage_metric()`: Calculate constellation coverage metric.
- `_calculate_redundancy_metric()`: Calculate constellation redundancy metric.

### Functions

#### `create_backward_compatible_problem(enable_multi, num_missions)`
Create optimization problem with backward compatibility.

---

## optimization/multi_mission_optimizer.py

**Description**: Multi-mission global optimizer extending the original GlobalOptimizer.

- **Lines of code**: 407
- **Classes**: 1
- **Functions**: 2

### Classes

#### `MultiMissionOptimizer`
Enhanced optimizer for both single and multi-mission problems.

**Methods:**
- `__init__()`: Initialize the multi-mission optimizer.
- `optimize()`: Run optimization with enhanced multi-mission support.
- `get_best_constellation_solutions()`: Get best constellation solutions with enhanced analysis.
- `get_constellation_metrics()`: Get constellation-specific performance metrics.
- `_enhance_multi_mission_results()`: Enhance results with multi-mission analysis.
- `_analyze_constellation_solution()`: Analyze individual constellation solution.

### Functions

#### `optimize_constellation(num_missions, cost_factors, optimization_config, constellation_config)`
Convenience function for constellation optimization.

#### `migrate_single_to_multi(single_config, num_missions)`
Migrate single-mission configuration to multi-mission.

---

## optimization/parallel_hooks.py

**Description**: Parallel evaluation hooks for GlobalOptimizer.

- **Lines of code**: 157
- **Classes**: 3
- **Functions**: 2

### Classes

#### `PopulationEvaluator`
Protocol for population evaluation strategies.

**Methods:**
- `evaluate_population()`: Evaluate population fitness.

#### `SequentialEvaluator`
Sequential population evaluation (default behavior).

**Methods:**
- `evaluate_population()`: Evaluate population sequentially.

#### `RayEvaluator`
Ray-based parallel population evaluation.

**Methods:**
- `__init__()`: Initialize Ray evaluator.
- `evaluate_population()`: Evaluate population using Ray workers.

### Functions

#### `add_parallel_evaluation_to_optimizer(optimizer, evaluator)`
Add parallel evaluation capability to existing GlobalOptimizer.

#### `get_recommended_evaluator(problem_config)`
Get the recommended evaluator based on system capabilities.

---

## optimization/pareto_analysis.py

**Description**: Pareto analysis and result processing for global optimization.

- **Lines of code**: 571
- **Classes**: 2
- **Functions**: 1

### Classes

#### `OptimizationResult`
Container for optimization results and analysis.

**Methods:**
- `__post_init__()`: Post-initialization processing.
- `num_pareto_solutions()`: Number of Pareto-optimal solutions.
- `objective_ranges()`: Get min/max ranges for each objective.
- `get_best_solutions()`: Get best solutions for a specific objective.
- `to_dict()`: Convert to dictionary representation.
- `from_dict()`: Create from dictionary representation.

#### `ParetoAnalyzer`
Analysis tools for Pareto fronts and multi-objective optimization results.

**Methods:**
- `__init__()`: Initialize Pareto analyzer.
- `find_pareto_front()`: Find Pareto-optimal solutions from a set of candidate solutions.
- `_dominates()`: Check if obj1 dominates obj2 (for minimization problems).
- `analyze_pareto_front()`: Analyze optimization results and create structured result object.
- `rank_solutions_by_preference()`: Rank solutions by user preferences using weighted objectives.
- `find_knee_solutions()`: Find knee points in the Pareto front (best trade-off solutions).
- `compare_optimization_runs()`: Compare multiple optimization runs.
- `calculate_hypervolume()`: Calculate hypervolume indicator for Pareto front quality.
- `export_results()`: Export optimization results to JSON file.
- `_calculate_optimization_stats()`: Calculate optimization statistics.
- `_extract_problem_config()`: Extract problem configuration from optimization result.
- `_minmax_normalize()`: Min-max normalization of objectives.
- `_zscore_normalize()`: Z-score normalization of objectives.
- `_find_knee_points()`: Find knee points using perpendicular distance method.
- `_calculate_hypervolume_3d()`: Simple hypervolume calculation for 3D objectives.

### Functions

#### `create_pareto_analyzer()`
Create a Pareto analyzer instance.

---

## optimization/ray_optimizer.py

**Description**: Ray-based parallel optimization for GlobalOptimizer.

- **Lines of code**: 609
- **Classes**: 2
- **Functions**: 1

### Classes

#### `RayParallelOptimizer`
Ray-parallelized version of GlobalOptimizer.

**Methods:**
- `__init__()`: Initialize Ray-parallel optimizer.
- `_initialize_ray_workers()`: Initialize Ray workers for parallel fitness evaluation.
- `_shutdown_ray_workers()`: Clean up Ray workers and collect final statistics.
- `optimize()`: Run multi-objective optimization with Ray parallelization.
- `_optimize_with_ray()`: Run optimization using Ray parallel evaluation.
- `_evaluate_population_parallel()`: Evaluate population fitness using Ray workers.

#### `FitnessWorker`
Ray actor for parallel fitness evaluation.

**Methods:**
- `__init__()`: Initialize fitness worker with pre-loaded resources.
- `_initialize_resources()`: Pre-load heavy computational resources.
- `evaluate_batch()`: Evaluate fitness for a batch of individuals.
- `_evaluate_single()`: Evaluate fitness for a single individual.
- `get_stats()`: Get worker performance statistics.

### Functions

#### `create_ray_optimizer(problem_config, optimizer_config, ray_config)`
Create Ray-parallel optimizer with fallback to sequential.

---

# Module: optimization/differentiable

## optimization/differentiable/advanced_demo.py

**Description**: Advanced JAX Differentiable Optimization Demonstration

- **Lines of code**: 545
- **Classes**: 1
- **Functions**: 1

### Classes

#### `AdvancedOptimizationDemo`
Advanced demonstration of JAX differentiable optimization capabilities.

**Methods:**
- `__init__()`: Initialize advanced optimization demonstration.
- `demonstrate_loss_functions()`: Demonstrate different multi-objective loss function configurations.
- `demonstrate_constraint_handling()`: Demonstrate different constraint handling methods.
- `demonstrate_hybrid_optimization()`: Demonstrate PyGMO-JAX hybrid optimization.
- `benchmark_optimization_methods()`: Benchmark different optimization methods and configurations.
- `_benchmark_basic_jax()`: Benchmark basic JAX optimization performance.
- `_benchmark_loss_functions()`: Benchmark different loss function configurations.
- `_benchmark_constraint_methods()`: Benchmark different constraint handling methods.
- `run_complete_demonstration()`: Run the complete advanced optimization demonstration.

### Functions

#### `run_advanced_demo()`
Run the advanced JAX optimization demonstration.

---

## optimization/differentiable/comparison_demo.py

**Description**: Result Comparison Demonstration Module

- **Lines of code**: 625
- **Classes**: 1
- **Functions**: 1

### Classes

#### `ComparisonDemonstration`
Comprehensive demonstration of result comparison capabilities.

**Methods:**
- `__init__()`: Initialize comparison demonstration.
- `demonstrate_single_comparison()`: Demonstrate comparison of single global vs local optimization run.
- `demonstrate_convergence_analysis()`: Demonstrate detailed convergence analysis.
- `demonstrate_solution_ranking()`: Demonstrate solution ranking across multiple runs.
- `demonstrate_pareto_analysis()`: Demonstrate Pareto front analysis for multi-objective optimization.
- `demonstrate_method_benchmark()`: Demonstrate benchmarking of different optimization methods.
- `demonstrate_comprehensive_analysis()`: Run comprehensive analysis demonstration covering all features.
- `_simulate_global_optimization()`: Simulate global optimization result (PyGMO-style).
- `_print_comparison_results()`: Print detailed comparison results.
- `_print_convergence_analysis()`: Print detailed convergence analysis.
- `_run_method_benchmark()`: Run actual method benchmarking.
- `_print_benchmark_results()`: Print benchmark results.
- `_generate_comprehensive_summary()`: Generate comprehensive summary of demonstration.

### Functions

#### `run_comparison_demo()`
Run the complete result comparison demonstration.

---

## optimization/differentiable/constraints.py

**Description**: Differentiable Constraint Handling Module

- **Lines of code**: 731
- **Classes**: 5
- **Functions**: 3

### Classes

#### `ConstraintType`
Enumeration of constraint types.

#### `ConstraintHandlingMethod`
Enumeration of constraint handling methods.

#### `ConstraintViolation`
Container for constraint violation information.

#### `ConstraintConfig`
Configuration for constraint handling.

#### `ConstraintHandler`
Differentiable constraint handler for trajectory optimization.

**Methods:**
- `__init__()`: Initialize constraint handler.
- `_setup_compiled_functions()`: Setup JIT-compiled constraint functions.
- `_compute_constraint_values()`: Compute all constraint values for given parameters.
- `_compute_box_constraints()`: Compute box constraints for optimization parameters.
- `_compute_physics_constraints()`: Compute physics-based constraints.
- `_compute_economic_constraints()`: Compute economic feasibility constraints.
- `_compute_penalty_terms()`: Compute penalty method terms for constraint violations.
- `_compute_barrier_terms()`: Compute barrier method terms for inequality constraints.
- `_compute_augmented_lagrangian()`: Compute augmented Lagrangian terms.
- `_get_constraint_type()`: Determine constraint type from constraint name.
- `compute_constraint_function()`: Compute the constraint function value based on configured method.
- `analyze_constraint_violations()`: Analyze constraint violations for given parameters.
- `update_adaptive_parameters()`: Update adaptive constraint handling parameters.
- `get_constraint_summary()`: Get comprehensive constraint violation summary.

### Functions

#### `create_penalty_constraint_handler(trajectory_model, economic_model, penalty_factor)`
Create a penalty method constraint handler.

#### `create_barrier_constraint_handler(trajectory_model, economic_model, barrier_parameter)`
Create a barrier method constraint handler.

#### `create_adaptive_constraint_handler(trajectory_model, economic_model)`
Create an adaptive constraint handler with automatic parameter updates.

---

## optimization/differentiable/continuous_thrust_integration.py

**Description**: Integration of continuous-thrust propagator with differentiable optimization.

- **Lines of code**: 231
- **Classes**: 2
- **Functions**: 2

### Classes

#### `ContinuousThrustModel`
JAX-compatible continuous-thrust trajectory model.

**Methods:**
- `__init__()`: Initialize continuous-thrust model.
- `compute_trajectory()`: Compute continuous-thrust trajectory.

#### `ContinuousThrustLoss`
Loss function for continuous-thrust optimization.

**Methods:**
- `__init__()`: Initialize loss function.
- `compute_loss()`: Compute multi-objective loss for continuous-thrust transfer.

### Functions

#### `optimize_continuous_thrust_transfer(r0, v0, T, Isp, target_radius, max_transfer_time)`
Optimize continuous-thrust Earth-Moon transfer.

#### `demonstrate_continuous_thrust_optimization()`
Demonstrate continuous-thrust optimization integration.

---

## optimization/differentiable/demo_optimization.py

**Description**: JAX Differentiable Optimization Demonstration

- **Lines of code**: 371
- **Classes**: 1
- **Functions**: 1

### Classes

#### `OptimizationDemonstration`
Demonstration of JAX-based differentiable optimization for lunar missions.

**Methods:**
- `__init__()`: Initialize the optimization demonstration.
- `_setup_optimizer()`: Setup the differentiable optimizer with appropriate configuration.
- `generate_initial_guess()`: Generate a reasonable initial guess for optimization.
- `evaluate_initial_solution()`: Evaluate the initial solution before optimization.
- `run_optimization()`: Run the complete optimization process.
- `evaluate_optimized_solution()`: Evaluate the optimized solution.
- `compare_solutions()`: Compare initial and optimized solutions.
- `run_complete_demonstration()`: Run the complete optimization demonstration.

### Functions

#### `run_quick_demo()`
Run a quick demonstration of the JAX optimization pipeline.

---

## optimization/differentiable/differentiable_models.py

**Description**: Differentiable Models Module

- **Lines of code**: 549
- **Classes**: 4
- **Functions**: 2

### Classes

#### `TrajectoryResult`
Result of trajectory calculation.

#### `EconomicResult`
Result of economic calculation.

#### `TrajectoryModel`
JAX-based differentiable trajectory model.

**Methods:**
- `__init__()`: Initialize trajectory model.
- `_orbital_velocity()`: Calculate circular orbital velocity.
- `_orbital_energy()`: Calculate specific orbital energy.
- `_hohmann_transfer()`: Calculate Hohmann transfer parameters.
- `_lambert_solver_simple()`: Simplified Lambert problem solver using JAX.
- `_trajectory_cost()`: Calculate trajectory cost from optimization parameters.
- `evaluate_trajectory()`: Evaluate trajectory for optimization.

#### `EconomicModel`
JAX-based differentiable economic model.

**Methods:**
- `__init__()`: Initialize economic model.
- `_launch_cost_model()`: Calculate launch cost based on delta-v and payload.
- `_operations_cost_model()`: Calculate operations cost based on mission duration.
- `_npv_calculation()`: Calculate Net Present Value.
- `_roi_calculation()`: Calculate Return on Investment.
- `_economic_cost()`: Calculate economic cost from optimization parameters.
- `evaluate_economics()`: Evaluate economics for optimization.

### Functions

#### `create_combined_model(trajectory_model, economic_model, weights)`
Create a combined trajectory-economic model for optimization.

#### `validate_against_pykep(parameters, trajectory_model)`
Validate JAX trajectory model against PyKEP calculations.

---

## optimization/differentiable/integration.py

**Description**: PyGMO Integration Module

- **Lines of code**: 836
- **Classes**: 4
- **Functions**: 3

### Classes

#### `HybridOptimizationConfig`
Configuration for hybrid PyGMO-JAX optimization.

#### `SolutionComparison`
Container for comparing optimization solutions.

#### `PyGMOProblem`
PyGMO-compatible problem wrapper for JAX differentiable models.

**Methods:**
- `__init__()`: Initialize PyGMO problem wrapper.
- `fitness()`: Compute fitness for PyGMO optimization.
- `get_bounds()`: Get parameter bounds for PyGMO.
- `get_nobj()`: Get number of objectives.
- `get_nec()`: Get number of equality constraints.
- `get_nic()`: Get number of inequality constraints.

#### `PyGMOIntegration`
Integration interface between PyGMO global optimization and JAX local optimization.

**Methods:**
- `__init__()`: Initialize PyGMO-JAX integration.
- `_create_constraint_functions()`: Create constraint functions for JAX optimizer.
- `_get_jax_bounds()`: Get bounds for JAX optimizer.
- `run_global_optimization()`: Run PyGMO global optimization.
- `select_local_start_points()`: Select starting points for local optimization from global results.
- `_select_diverse_solutions()`: Select diverse solutions using simple distance-based selection.
- `run_local_optimization()`: Run JAX local optimization from multiple starting points.
- `run_hybrid_optimization()`: Run complete hybrid optimization workflow.
- `analyze_hybrid_results()`: Analyze and compare global vs local optimization results.
- `get_optimization_summary()`: Get comprehensive summary of all optimization runs.

### Functions

#### `create_standard_hybrid_optimizer(trajectory_model, economic_model, loss_function, constraint_handler)`
Create a standard hybrid optimizer with balanced configuration.

#### `create_fast_hybrid_optimizer(trajectory_model, economic_model, loss_function, constraint_handler)`
Create a fast hybrid optimizer with reduced computational requirements.

#### `create_thorough_hybrid_optimizer(trajectory_model, economic_model, loss_function, constraint_handler)`
Create a thorough hybrid optimizer for high-quality solutions.

---

## optimization/differentiable/jax_optimizer.py

**Description**: JAX-based Differentiable Optimizer

- **Lines of code**: 548
- **Classes**: 2
- **Functions**: 2

### Classes

#### `OptimizationResult`
Result of differentiable optimization.

#### `DifferentiableOptimizer`
JAX-based differentiable optimizer for trajectory and economic optimization.

**Methods:**
- `__init__()`: Initialize the differentiable optimizer.
- `_setup_compiled_functions()`: Setup JIT-compiled objective and gradient functions with optimization.
- `_callback()`: Callback function for optimization progress tracking.
- `optimize()`: Perform gradient-based optimization starting from initial point x0.
- `_analyze_objective_components()`: Analyze individual components of the objective function.
- `_analyze_constraint_violations()`: Analyze constraint violations at the solution.
- `batch_optimize()`: Optimize multiple initial points in batch with performance optimizations.
- `evaluate_batch_objectives()`: Evaluate objective function for a batch of parameters efficiently.
- `evaluate_batch_gradients()`: Evaluate gradients for a batch of parameters efficiently.
- `compare_with_initial()`: Compare optimization results with initial solutions.

### Functions

#### `create_trajectory_optimizer(trajectory_model, economic_model, weights)`
Create a differentiable optimizer for trajectory optimization.

#### `create_economic_optimizer(economic_model, objective)`
Create a differentiable optimizer focused on economic objectives.

---

## optimization/differentiable/loss_functions.py

**Description**: Multi-Objective Loss Functions Module

- **Lines of code**: 721
- **Classes**: 5
- **Functions**: 4

### Classes

#### `WeightingStrategy`
Enumeration of weighting strategies for multi-objective optimization.

#### `NormalizationMethod`
Enumeration of normalization methods for objectives.

#### `ObjectiveMetrics`
Container for objective function metrics and statistics.

#### `LossFunctionConfig`
Configuration for multi-objective loss functions.

#### `MultiObjectiveLoss`
Multi-objective loss function for trajectory and economic optimization.

**Methods:**
- `__init__()`: Initialize multi-objective loss function.
- `_setup_compiled_functions()`: Setup JIT-compiled loss functions.
- `_compute_raw_objectives()`: Compute raw objective values from trajectory and economic models.
- `_compute_normalized_objectives()`: Normalize objectives according to configured method.
- `_compute_penalty_terms()`: Compute penalty terms for constraint violations.
- `_apply_weighting_strategy()`: Apply weighting strategy to normalized objectives.
- `_compute_adaptive_weights()`: Compute adaptive weights based on objective improvement rates.
- `_compute_pareto_weights()`: Compute Pareto-based weights using dominance relationships.
- `_apply_lexicographic_weighting()`: Apply lexicographic weighting based on objective priorities.
- `_apply_achievement_scalarization()`: Apply achievement scalarization method.
- `compute_loss()`: Compute the complete multi-objective loss function.
- `update_metrics()`: Update metrics tracking for adaptive strategies.
- `_update_normalization_parameters()`: Update normalization parameters based on observed data.
- `get_objective_breakdown()`: Get detailed breakdown of objective function components.

### Functions

#### `create_balanced_loss_function(trajectory_model, economic_model)`
Create a balanced multi-objective loss function with equal weighting.

#### `create_performance_focused_loss_function(trajectory_model, economic_model)`
Create a performance-focused loss function emphasizing delta-v and time.

#### `create_economic_focused_loss_function(trajectory_model, economic_model)`
Create an economics-focused loss function emphasizing cost and ROI.

#### `create_adaptive_loss_function(trajectory_model, economic_model)`
Create an adaptive loss function that adjusts weights during optimization.

---

## optimization/differentiable/performance_demo.py

**Description**: Performance Optimization Demonstration

- **Lines of code**: 497
- **Classes**: 1
- **Functions**: 1

### Classes

#### `PerformanceDemo`
Comprehensive performance demonstration for JAX optimization.

**Methods:**
- `__init__()`: Initialize performance demonstration.
- `demonstrate_jit_compilation_benefits()`: Demonstrate the performance benefits of JIT compilation.
- `demonstrate_vectorization_benefits()`: Demonstrate the performance benefits of vectorization.
- `demonstrate_optimizer_performance()`: Demonstrate performance improvements in optimization.
- `demonstrate_compilation_cache_benefits()`: Demonstrate benefits of compilation caching.
- `demonstrate_comprehensive_benchmark()`: Run comprehensive performance benchmark covering all optimization techniques.
- `_generate_performance_summary()`: Generate summary of performance improvements.

### Functions

#### `run_performance_demo()`
Run the complete performance optimization demonstration.

---

## optimization/differentiable/performance_optimization.py

**Description**: Performance Optimization Module for JAX Differentiable Optimization

- **Lines of code**: 771
- **Classes**: 6
- **Functions**: 3

### Classes

#### `PerformanceConfig`
Configuration for performance optimization.

#### `PerformanceMetrics`
Container for performance metrics.

#### `JITOptimizer`
Advanced JIT compilation optimizer for differentiable optimization functions.

**Methods:**
- `__init__()`: Initialize JIT optimizer.
- `_setup_compilation_cache()`: Setup JAX compilation cache for faster recompilation.
- `compile_objective_function()`: Compile objective function with advanced JIT optimizations.
- `compile_gradient_function()`: Compile gradient function with optimization.
- `compile_hessian_function()`: Compile Hessian function with optimization.
- `create_vectorized_function()`: Create vectorized version of function for batch processing.
- `create_parallel_function()`: Create parallel version of function for multi-device execution.

#### `BatchOptimizer`
Batch optimization utilities for efficient parameter space exploration.

**Methods:**
- `__init__()`: Initialize batch optimizer.
- `_setup_batch_functions()`: Setup vectorized batch processing functions.
- `evaluate_batch()`: Evaluate loss function for a batch of parameters.
- `optimize_batch()`: Perform batch gradient descent optimization.

#### `MemoryOptimizer`
Memory optimization utilities for large-scale differentiable optimization.

**Methods:**
- `__init__()`: Initialize memory optimizer.
- `create_memory_efficient_gradient()`: Create memory-efficient gradient function using chunking.
- `preallocate_workspace()`: Preallocate memory workspace for optimization.
- `get_workspace_array()`: Get preallocated array from workspace.

#### `PerformanceBenchmark`
Performance benchmarking and profiling utilities.

**Methods:**
- `__init__()`: Initialize performance benchmark.
- `benchmark_function()`: Benchmark function performance.
- `compare_implementations()`: Compare performance of different implementations.
- `profile_memory_usage()`: Profile memory usage of function.

### Functions

#### `optimize_differentiable_optimizer(optimizer, config)`
Apply performance optimizations to a DifferentiableOptimizer instance.

#### `create_performance_optimized_loss_function(loss_function, config)`
Create performance-optimized version of loss function.

#### `benchmark_optimization_performance(trajectory_model, economic_model, test_parameters, config)`
Comprehensive performance benchmark for optimization components.

---

## optimization/differentiable/result_comparison.py

**Description**: Result Comparison and Evaluation Module for Differentiable Optimization

- **Lines of code**: 950
- **Classes**: 5
- **Functions**: 4

### Classes

#### `ComparisonMetric`
Enumeration of comparison metrics for optimization results.

#### `SolutionQuality`
Enumeration of solution quality categories.

#### `ComparisonResult`
Container for optimization comparison results.

#### `ConvergenceAnalysis`
Container for convergence analysis results.

#### `ResultComparator`
Advanced result comparison and evaluation system.

**Methods:**
- `__init__()`: Initialize result comparator.
- `compare_optimization_results()`: Compare global and local optimization results comprehensively.
- `analyze_convergence()`: Perform detailed convergence analysis for optimization result.
- `rank_solutions()`: Rank optimization solutions based on multiple criteria.
- `compute_pareto_front()`: Compute Pareto front for multi-objective optimization results.
- `benchmark_methods()`: Benchmark different optimization methods on test problems.
- `_compute_objective_improvement()`: Compute objective improvement percentage.
- `_analyze_convergence_improvement()`: Analyze convergence improvement.
- `_assess_solution_quality()`: Assess solution quality based on multiple criteria.
- `_compute_speedup_factor()`: Compute speedup factor.
- `_compute_efficiency_ratio()`: Compute efficiency ratio.
- `_perform_component_analysis()`: Perform component-wise analysis of solutions.
- `_perform_sensitivity_analysis()`: Perform sensitivity analysis around solution.
- `_generate_recommendations()`: Generate optimization recommendations.
- `_compute_convergence_rate()`: Compute convergence rate from objective history.
- `_find_convergence_point()`: Find iteration where convergence occurred.
- `_compute_solution_stability()`: Compute solution stability metric.
- `_assess_convergence_quality()`: Assess overall convergence quality.
- `_find_stagnation_periods()`: Find periods of stagnation in optimization.
- `_estimate_step_sizes()`: Estimate step sizes from objective history.
- `_get_criterion_weight()`: Get weight for comparison criterion.
- `_evaluate_criterion()`: Evaluate specific criterion for result.
- `_dominates()`: Check if solution1 dominates solution2 (for minimization).
- `_run_benchmark_iteration()`: Run single benchmark iteration (placeholder).
- `_create_benchmark_summary()`: Create summary of benchmark results.

### Functions

#### `compare_single_run(global_result, local_result)`
Compare single optimization run results.

#### `analyze_optimization_convergence(result)`
Analyze convergence of optimization result.

#### `rank_optimization_results(results)`
Rank optimization results by quality.

#### `evaluate_solution_quality(result)`
Evaluate solution quality of optimization result.

---

# Module: scripts

## scripts/benchmark_performance.py

**Description**: Performance Benchmark Script for PRD Compliance Validation

- **Lines of code**: 123
- **Classes**: 0
- **Functions**: 5

### Functions

#### `benchmark_config_loading()`
Benchmark configuration loading performance.

#### `benchmark_economic_analysis()`
Benchmark economic analysis calculations.

#### `benchmark_trajectory_validation()`
Benchmark trajectory validation performance.

#### `benchmark_optimization_setup()`
Benchmark optimization module setup.

#### `main()`
Run all benchmarks.

---

## scripts/test_prd_user_flows.py

**Description**: PRD User Flow Validation Test

- **Lines of code**: 485
- **Classes**: 0
- **Functions**: 8

### Functions

#### `test_user_flow_1_mission_configuration()`
Test PRD User Flow 1: Load mission configuration.

#### `test_user_flow_2_global_optimization()`
Test PRD User Flow 2: Global optimization with Pareto front.

#### `test_user_flow_3_local_optimization()`
Test PRD User Flow 3: Local optimization refinement.

#### `test_user_flow_4_economic_analysis()`
Test PRD User Flow 4: Economic analysis (ROI, NPV, IRR).

#### `test_user_flow_5_visualization()`
Test PRD User Flow 5: Interactive 3D visualizations.

#### `test_user_personas_support()`
Test support for all PRD user personas.

#### `test_technical_architecture()`
Test PRD technical architecture components.

#### `main()`
Run all PRD validation tests.

---

## scripts/verify_dependencies.py

**Description**: Dependency verification script for Lunar Horizon Optimizer.

- **Lines of code**: 202
- **Classes**: 0
- **Functions**: 8

### Functions

#### `check_import(module_name, min_version)`
Check if a module can be imported and optionally verify its version.

#### `verify_jax_gpu()`
Verify JAX GPU support.

#### `check_scipy_version()`
Check SciPy version and compatibility.

#### `test_trajectory_components()`
Test trajectory calculation components (PyKEP, PyGMO).

#### `test_optimization_components()`
Test optimization components (JAX, Diffrax).

#### `test_visualization_components()`
Test visualization components (Plotly, Poliastro).

#### `print_section_results(title, results)`
Print results for a section with proper formatting.

#### `main()`
Main verification routine.

---

# Module: tests

## tests/conftest.py

- **Lines of code**: 92
- **Classes**: 0
- **Functions**: 5

### Functions

#### `config_manager()`
Fixture that provides a ConfigurationManager instance.

#### `sample_config()`
Fixture that provides a sample valid configuration.

#### `invalid_config()`
Fixture that provides an invalid configuration missing required fields.

#### `different_config()`
Fixture that provides a valid but different configuration for comparison.

#### `missing_fields_config()`
Fixture that provides a configuration with some fields missing for comparison.

---

## tests/run_comprehensive_test_analysis.py

**Description**: Comprehensive Test Analysis and Coverage Report

- **Lines of code**: 603
- **Classes**: 1
- **Functions**: 1

### Classes

#### `TestAnalyzer`
Comprehensive test analysis and execution framework.

**Methods:**
- `__init__()`: No docstring
- `run_comprehensive_analysis()`: Run comprehensive test analysis and coverage report.
- `_get_environment_info()`: Get environment information.
- `_analyze_module_coverage()`: Analyze test coverage for all modules.
- `_find_source_modules()`: Find all source modules.
- `_find_test_files()`: Find all test files.
- `_test_file_exists()`: Check if test file exists.
- `_extract_tested_modules()`: Extract modules that have dedicated tests.
- `_execute_all_tests()`: Execute all test suites and collect results.
- `_run_pytest()`: Run pytest on a specific test file.
- `_parse_pytest_output()`: Parse pytest output to extract test statistics.
- `_perform_sanity_checks()`: Perform sanity checks on test results and calculations.
- `_check_realistic_ranges()`: Check that test values are within realistic ranges.
- `_check_physical_constants()`: Check that physical constants are reasonable.
- `_check_calculation_consistency()`: Check that calculations are consistent and sensible.
- `_check_result_validation()`: Check that test results are being properly validated.
- `_print_coverage_summary()`: Print module coverage summary.
- `_generate_comprehensive_report()`: Generate comprehensive test report.
- `_generate_recommendations()`: Generate recommendations based on analysis.
- `_save_report()`: Save comprehensive report to file.

### Functions

#### `main()`
Main function to run comprehensive test analysis.

---

## tests/run_comprehensive_tests.py

**Description**: Comprehensive test runner and validation script for Tasks 3, 4, and 5

- **Lines of code**: 383
- **Classes**: 1
- **Functions**: 1

### Classes

#### `ComprehensiveTestRunner`
Comprehensive test runner with validation and reporting.

**Methods:**
- `__init__()`: No docstring
- `run_test_suite()`: Run a specific test suite and capture results.
- `validate_test_sanity()`: Validate that test results are sane and correct.
- `_check_trajectory_tests()`: Check if trajectory generation tests cover key functionality.
- `_check_optimization_tests()`: Check if optimization tests cover key functionality.
- `_check_economic_tests()`: Check if economic analysis tests cover key functionality.
- `_check_integration_tests()`: Check if integration tests cover key functionality.
- `generate_report()`: Generate comprehensive test report.
- `run_all_tests()`: Run all test suites and generate comprehensive report.

### Functions

#### `main()`
Main function to run comprehensive tests.

---

## tests/run_working_tests.py

**Description**: Working Test Runner for Tasks 3, 4, and 5

- **Lines of code**: 258
- **Classes**: 1
- **Functions**: 1

### Classes

#### `WorkingTestRunner`
Test runner for validated working tests.

**Methods:**
- `__init__()`: No docstring
- `run_test_suite()`: Run a specific test suite and capture results.
- `generate_summary_report()`: Generate summary report of test results.
- `run_all_tests()`: Run all working test suites.

### Functions

#### `main()`
Main function to run working tests.

---

## tests/test_config_loader.py

**Description**: Tests for configuration loader functionality.

- **Lines of code**: 190
- **Classes**: 0
- **Functions**: 13

### Functions

#### `valid_config_dict()`
Fixture providing a valid configuration dictionary.

#### `temp_json_config(tmp_path, valid_config_dict)`
Fixture creating a temporary JSON configuration file.

#### `temp_yaml_config(tmp_path, valid_config_dict)`
Fixture creating a temporary YAML configuration file.

#### `test_load_json_config(temp_json_config)`
Test loading a valid JSON configuration file.

#### `test_load_yaml_config(temp_yaml_config)`
Test loading a valid YAML configuration file.

#### `test_load_nonexistent_file()`
Test loading a non-existent file.

#### `test_load_invalid_format(tmp_path)`
Test loading a file with unsupported format.

#### `test_load_invalid_json(tmp_path)`
Test loading an invalid JSON file.

#### `test_load_invalid_yaml(tmp_path)`
Test loading an invalid YAML file.

#### `test_merge_with_defaults(valid_config_dict)`
Test merging loaded config with defaults.

#### `test_save_config_json(tmp_path, valid_config_dict)`
Test saving configuration to JSON file.

#### `test_save_config_yaml(tmp_path, valid_config_dict)`
Test saving configuration to YAML file.

#### `test_load_default_config()`
Test creating loader with default configuration.

---

## tests/test_config_manager.py

**Description**: Tests for configuration manager functionality.

- **Lines of code**: 186
- **Classes**: 0
- **Functions**: 13

### Functions

#### `sample_config()`
Fixture providing a sample mission configuration.

#### `manager()`
Fixture providing a ConfigManager instance.

#### `test_init_manager()`
Test manager initialization.

#### `test_load_config(tmp_path, manager, sample_config)`
Test loading a configuration from file.

#### `test_save_config(tmp_path, manager, sample_config)`
Test saving a configuration to file.

#### `test_save_without_active_config(tmp_path, manager)`
Test that saving without an active configuration raises an error.

#### `test_create_from_template(manager)`
Test creating a configuration from a template.

#### `test_create_from_nonexistent_template(manager)`
Test that creating from a non-existent template raises an error.

#### `test_validate_config(manager)`
Test configuration validation.

#### `test_validate_invalid_config(manager)`
Test validation of invalid configuration.

#### `test_update_config(manager, sample_config)`
Test updating configuration values.

#### `test_update_without_active_config(manager)`
Test that updating without an active configuration raises an error.

#### `test_update_with_invalid_data(manager, sample_config)`
Test updating with invalid configuration data.

---

## tests/test_config_models.py

**Description**: Tests for mission configuration data models.

- **Lines of code**: 207
- **Classes**: 0
- **Functions**: 8

### Functions

#### `test_valid_mission_config()`
Test creation of a valid mission configuration.

#### `test_invalid_payload_mass()`
Test validation of payload mass against dry mass.

#### `test_invalid_orbit_parameters()`
Test validation of orbit parameters.

#### `test_invalid_eccentricity()`
Test validation of orbit eccentricity.

#### `test_cost_factors_validation()`
Test validation of cost factors.

#### `test_isru_target_validation()`
Test validation of ISRU targets.

#### `test_optional_description()`
Test that description is optional.

#### `test_empty_isru_targets()`
Test that ISRU targets can be empty.

---

## tests/test_config_registry.py

**Description**: Tests for configuration registry functionality.

- **Lines of code**: 184
- **Classes**: 0
- **Functions**: 13

### Functions

#### `registry()`
Fixture providing a ConfigRegistry instance.

#### `sample_config()`
Fixture providing a sample mission configuration.

#### `test_default_templates(registry)`
Test that default templates are loaded correctly.

#### `test_register_template(registry, sample_config)`
Test registering a new template.

#### `test_register_duplicate_template(registry, sample_config)`
Test that registering a duplicate default template raises an error.

#### `test_get_nonexistent_template(registry)`
Test that getting a non-existent template raises an error.

#### `test_template_isolation(registry, sample_config)`
Test that templates are properly isolated when copied.

#### `test_load_template_file(tmp_path, registry, sample_config)`
Test loading a template from a file.

#### `test_load_invalid_template_file(tmp_path, registry)`
Test that loading an invalid template file raises an error.

#### `test_load_templates_dir(tmp_path, sample_config)`
Test loading templates from a directory.

#### `test_load_nonexistent_templates_dir(registry)`
Test that loading from a non-existent directory raises an error.

#### `test_save_template(tmp_path, registry, sample_config)`
Test saving a template to a file.

#### `test_save_nonexistent_template(tmp_path, registry)`
Test that saving a non-existent template raises an error.

---

## tests/test_continuous_thrust.py

**Description**: Tests for continuous-thrust propagator.

- **Lines of code**: 310
- **Classes**: 5
- **Functions**: 1

### Classes

#### `TestContinuousThrustBasics`
Test basic continuous-thrust propagator functionality.

**Methods:**
- `test_dynamics_function()`: Test continuous dynamics function.
- `test_low_thrust_transfer_basic()`: Test basic low-thrust transfer calculation.
- `test_optimize_thrust_angle()`: Test thrust angle optimization for target radius.

#### `TestContinuousThrustModel`
Test JAX-compatible continuous-thrust model.

**Methods:**
- `test_model_initialization()`: Test model initialization.
- `test_compute_trajectory()`: Test trajectory computation.

#### `TestOptimizationIntegration`
Test integration with differentiable optimization.

**Methods:**
- `test_continuous_thrust_loss()`: Test continuous-thrust loss function.
- `test_optimization_integration_fast()`: Test fast optimization integration (reduced parameters).

#### `TestPhysicsValidation`
Test physics validation and accuracy.

**Methods:**
- `test_energy_conservation_check()`: Test that energy changes are reasonable for continuous thrust.
- `test_mass_conservation()`: Test mass is conserved according to rocket equation.

#### `TestDemonstration`
Test demonstration function.

**Methods:**
- `test_demonstration_runs()`: Test that demonstration function runs without errors.

### Functions

#### `test_accuracy_caveats_documented()`
Ensure accuracy caveats are properly documented.

---

## tests/test_cost_learning_curves.py

**Description**: Tests for learning curves and environmental costs in cost models.

- **Lines of code**: 432
- **Classes**: 5
- **Functions**: 0

### Classes

#### `TestLearningCurveFunctions`
Test Wright's law learning curve implementation.

**Methods:**
- `test_launch_price_base_year()`: Test launch price calculation for base year.
- `test_launch_price_past_year()`: Test launch price calculation for past year.
- `test_launch_price_future_year_reduction()`: Test launch price reduction in future years.
- `test_learning_curve_mathematics()`: Test Wright's law mathematical correctness.
- `test_learning_rate_validation()`: Test learning rate parameter validation.
- `test_production_growth_impact()`: Test impact of different production base levels.

#### `TestCO2CostCalculation`
Test CO₂ environmental cost calculations.

**Methods:**
- `test_basic_co2_cost_calculation()`: Test basic CO₂ cost calculation.
- `test_co2_cost_zero_emissions()`: Test CO₂ cost with zero emissions.
- `test_co2_cost_zero_price()`: Test CO₂ cost with zero carbon price.
- `test_co2_cost_realistic_values()`: Test CO₂ cost with realistic mission parameters.
- `test_co2_cost_high_emission_scenario()`: Test CO₂ cost with high-emission launcher.

#### `TestCostFactorsConfiguration`
Test CostFactors configuration with new parameters.

**Methods:**
- `test_cost_factors_default_values()`: Test CostFactors with default environmental and learning parameters.
- `test_cost_factors_custom_values()`: Test CostFactors with custom environmental and learning parameters.
- `test_cost_factors_validation()`: Test CostFactors parameter validation.

#### `TestFinancialParametersIntegration`
Test FinancialParameters with environmental cost integration.

**Methods:**
- `test_financial_parameters_defaults()`: Test FinancialParameters with default environmental values.
- `test_financial_parameters_total_cost()`: Test total cost calculation with environmental costs.
- `test_financial_parameters_validation()`: Test FinancialParameters validation.

#### `TestCostCalculatorIntegration`
Test CostCalculator integration with learning curves and environmental costs.

**Methods:**
- `test_cost_calculator_initialization()`: Test CostCalculator initialization with new parameters.
- `test_cost_calculator_learning_curve_integration()`: Test cost calculation with learning curve adjustments.
- `test_cost_breakdown_new_components()`: Test cost breakdown with new environmental and learning curve components.
- `test_environmental_cost_impact()`: Test impact of different environmental cost scenarios.

---

## tests/test_economics_core.py

**Description**: Core unit tests for economics modules to improve coverage.

- **Lines of code**: 219
- **Classes**: 8
- **Functions**: 0

### Classes

#### `TestNPVAnalyzer`
Test NPVAnalyzer class.

**Methods:**
- `test_npv_analyzer_creation()`: Test creating an NPV analyzer.
- `test_npv_calculation_positive()`: Test NPV calculation with positive cash flows.

#### `TestROICalculator`
Test ROICalculator class.

**Methods:**
- `test_roi_calculator_creation()`: Test creating an ROI calculator.
- `test_roi_calculation()`: Test ROI calculation.

#### `TestCashFlowModel`
Test CashFlowModel class.

**Methods:**
- `test_cash_flow_model_creation()`: Test creating a cash flow model.
- `test_add_cash_flow()`: Test adding cash flows.

#### `TestFinancialParameters`
Test FinancialParameters class.

**Methods:**
- `test_financial_parameters_creation()`: Test creating financial parameters.
- `test_financial_parameters_validation()`: Test financial parameters validation.

#### `TestCashFlow`
Test CashFlow dataclass.

**Methods:**
- `test_cash_flow_creation()`: Test creating a cash flow.
- `test_cash_flow_validation()`: Test cash flow validation.

#### `TestMissionCostModel`
Test MissionCostModel class.

**Methods:**
- `test_mission_cost_model_creation()`: Test creating a mission cost model.
- `test_cost_calculation()`: Test basic cost calculation.

#### `TestISRUBenefitAnalyzer`
Test ISRUBenefitAnalyzer class.

**Methods:**
- `test_isru_analyzer_creation()`: Test creating an ISRU benefit analyzer.
- `test_basic_functionality()`: Test basic ISRU analyzer functionality.

#### `TestSensitivityAnalyzer`
Test SensitivityAnalyzer class.

**Methods:**
- `test_sensitivity_analyzer_creation()`: Test creating a sensitivity analyzer.
- `test_parameter_variation()`: Test parameter variation calculation.

---

## tests/test_economics_modules.py

**Description**: Economics Modules Test Suite

- **Lines of code**: 905
- **Classes**: 5
- **Functions**: 1

### Classes

#### `TestFinancialModels`
Test financial models module functionality and realism.

**Methods:**
- `test_financial_parameters_validation()`: Test FinancialParameters initialization and validation.
- `test_cash_flow_model_realistic_scenarios()`: Test CashFlowModel with realistic space mission scenarios.
- `test_npv_calculation_accuracy()`: Test NPV calculation accuracy and realism.
- `test_irr_calculation_accuracy()`: Test IRR calculation accuracy and realism.
- `test_roi_calculation_scenarios()`: Test ROI calculation for different scenarios.
- `test_payback_period_calculation()`: Test payback period calculation accuracy.

#### `TestCostModels`
Test cost models module functionality and realism.

**Methods:**
- `test_mission_cost_model_initialization()`: Test MissionCostModel initialization and basic functionality.
- `test_realistic_mission_cost_estimation()`: Test mission cost estimation with realistic parameters.
- `test_cost_scaling_factors()`: Test cost scaling with different parameters.
- `test_technology_readiness_impact()`: Test impact of technology readiness on costs.
- `test_launch_cost_realism()`: Test launch cost calculations for realism.

#### `TestISRUBenefits`
Test ISRU benefits analysis module.

**Methods:**
- `test_isru_analyzer_initialization()`: Test ISRUBenefitAnalyzer initialization.
- `test_isru_resource_properties()`: Test ISRU resource properties and calculations.
- `test_isru_facility_scaling()`: Test ISRU facility scaling economics.
- `test_isru_economic_realism()`: Test ISRU economic analysis for realistic results.

#### `TestSensitivityAnalysis`
Test sensitivity analysis module.

**Methods:**
- `test_sensitivity_analyzer_initialization()`: Test EconomicSensitivityAnalyzer initialization.
- `test_monte_carlo_simulation_basic()`: Test basic Monte Carlo simulation functionality.
- `test_parameter_distribution_validation()`: Test parameter distribution validation.

#### `TestEconomicReporting`
Test economic reporting module.

**Methods:**
- `test_economic_reporter_initialization()`: Test EconomicReporter initialization.
- `test_financial_summary_creation()`: Test FinancialSummary data structure.
- `test_executive_summary_generation()`: Test executive summary generation.
- `test_data_export_functionality()`: Test data export functionality.

### Functions

#### `test_economics_modules_summary()`
Summary test for all economics modules.

---

## tests/test_environment.py

**Description**: Basic smoke tests to verify the Python environment and dependencies.

- **Lines of code**: 84
- **Classes**: 0
- **Functions**: 7

### Functions

#### `test_scipy_version()`
Verify SciPy version is compatible with PyKEP.

#### `test_jax_configuration()`
Verify JAX is properly configured.

#### `test_pykep_basic()`
Verify PyKEP basic functionality.

#### `test_pygmo_basic()`
Verify PyGMO basic functionality.

#### `test_diffrax_basic()`
Verify Diffrax basic functionality.

#### `test_plotly_basic()`
Verify Plotly basic functionality.

#### `test_poliastro_basic()`
Verify Poliastro basic functionality.

---

## tests/test_final_functionality.py

**Description**: Final Real Functionality Test Suite - All Issues Fixed

- **Lines of code**: 593
- **Classes**: 5
- **Functions**: 1

### Classes

#### `TestPyKEPRealFunctionality`
Test real PyKEP functionality without mocking.

**Methods:**
- `test_lambert_problem_realistic_transfer()`: Test real Lambert problem with known working geometry.
- `test_planet_ephemeris_earth()`: Test real planet ephemeris calculations for Earth.
- `test_orbital_elements_conversion()`: Test real orbital elements to Cartesian conversion.
- `test_mu_constants()`: Test PyKEP gravitational parameter constants.

#### `TestPyGMORealFunctionality`
Test real PyGMO functionality without mocking.

**Methods:**
- `test_single_objective_optimization()`: Test real single-objective optimization.
- `test_multi_objective_optimization_realistic()`: Test real multi-objective optimization with better diversity.
- `test_algorithm_convergence()`: Test real algorithm convergence behavior.

#### `TestConfigurationRealFunctionality`
Test real configuration functionality.

**Methods:**
- `test_cost_factors_with_parameters()`: Test cost factors with explicit parameters.
- `test_cost_factors_edge_cases()`: Test cost factors with edge cases.

#### `TestEconomicAnalysisRealFunctionality`
Test real economic analysis without mocking.

**Methods:**
- `test_real_npv_calculation()`: Test real NPV calculation with realistic cash flows.
- `test_real_irr_calculation_corrected()`: Test real IRR calculation using numerical methods.
- `test_real_mission_cost_estimation()`: Test real mission cost estimation.

#### `TestIntegrationRealFunctionality`
Test real integration between all modules.

**Methods:**
- `test_real_trajectory_optimization_integration_improved()`: Test real integration with improved diversity.
- `test_real_simplified_mission_analysis()`: Test real simplified mission analysis workflow.

### Functions

#### `test_environment_setup()`
Test that the environment is properly configured.

---

## tests/test_helpers.py

**Description**: Test helper classes and utilities for replacing complex dependencies in tests.

- **Lines of code**: 222
- **Classes**: 3
- **Functions**: 2

### Classes

#### `SimpleTrajectory`
Simplified trajectory for testing purposes.

**Methods:**
- `__init__()`: No docstring
- `add_maneuver()`: Add a maneuver to the trajectory.

#### `SimpleLunarTransfer`
Simplified LunarTransfer implementation for testing without PyKEP.

**Methods:**
- `__init__()`: Initialize simplified lunar transfer.
- `generate_transfer()`: Generate a simplified transfer trajectory.

#### `SimpleOptimizationProblem`
Simplified optimization problem for testing PyGMO functionality.

**Methods:**
- `__init__()`: Initialize with specified number of objectives and parameters.
- `get_nobj()`: Get number of objectives.
- `get_bounds()`: Get parameter bounds.
- `fitness()`: Calculate fitness for given parameters.

### Functions

#### `create_mock_pykep()`
Create a mock PyKEP module for testing.

#### `create_mock_pygmo()`
Create a mock PyGMO module for testing.

---

## tests/test_integration_tasks_3_4_5.py

**Description**: Comprehensive integration test suite for Tasks 3, 4, and 5

- **Lines of code**: 810
- **Classes**: 4
- **Functions**: 2

### Classes

#### `TestTask3Task4Integration`
Test integration between trajectory generation and optimization.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_trajectory_generation_in_optimization()`: Test that optimization properly uses trajectory generation.
- `test_optimization_trajectory_parameter_flow()`: Test parameter flow from optimization to trajectory generation.
- `test_optimization_convergence_with_trajectory_data()`: Test optimization convergence using realistic trajectory data.

#### `TestTask3Task5Integration`
Test integration between trajectory generation and economic analysis.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_trajectory_parameters_to_cost_calculation()`: Test trajectory parameters feeding into cost calculations.
- `test_trajectory_derived_financial_analysis()`: Test financial analysis based on trajectory-derived parameters.
- `test_mission_economics_sensitivity_to_trajectory()`: Test how mission economics change with different trajectory parameters.

#### `TestTask4Task5Integration`
Test integration between optimization and economic analysis.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_optimization_with_economic_objectives()`: Test optimization using economic objectives from Task 5.
- `test_cost_calculator_integration()`: Test cost calculator integration with optimization.
- `test_pareto_front_with_economic_trade_offs()`: Test Pareto front generation with economic trade-offs.

#### `TestFullSystemIntegration`
Test complete end-to-end system integration across all three tasks.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_end_to_end_mission_optimization_workflow()`: Test complete end-to-end mission optimization workflow.
- `test_data_consistency_across_modules()`: Test data consistency and format compatibility across modules.
- `test_error_propagation_across_modules()`: Test error handling and propagation across module boundaries.
- `test_performance_integration()`: Test performance characteristics of integrated system.

### Functions

#### `test_integration_environment()`
Test that integration environment is properly configured.

#### `test_integration_module_imports()`
Test that all required modules for integration can be imported.

---

## tests/test_multi_mission_optimization.py

**Description**: Comprehensive tests for multi-mission constellation optimization.

- **Lines of code**: 497
- **Classes**: 8
- **Functions**: 0

### Classes

#### `TestMultiMissionGenome`
Test MultiMissionGenome dataclass and encoding/decoding.

**Methods:**
- `test_genome_creation()`: Test basic genome creation and validation.
- `test_genome_auto_initialization()`: Test automatic initialization of missing parameters.
- `test_decision_vector_encoding_decoding()`: Test conversion to/from PyGMO decision vector.
- `test_mission_parameters_extraction()`: Test extraction of individual mission parameters.
- `test_constellation_geometry_validation()`: Test constellation geometry validation.
- `test_invalid_genome_creation()`: Test error handling for invalid genome parameters.

#### `TestMultiMissionProblem`
Test MultiMissionProblem PyGMO interface.

**Methods:**
- `test_problem_creation()`: Test basic multi-mission problem creation.
- `test_problem_bounds()`: Test decision variable bounds.
- `test_problem_objectives()`: Test number of objectives.
- `test_fitness_evaluation_structure()`: Test fitness evaluation returns correct structure.
- `test_penalty_handling()`: Test penalty values for invalid solutions.

#### `TestMultiMissionOptimizer`
Test MultiMissionOptimizer functionality.

**Methods:**
- `test_optimizer_creation_single_mission()`: Test optimizer creation for single mission (backward compatibility).
- `test_optimizer_creation_multi_mission()`: Test optimizer creation for multi-mission constellation.
- `test_optimization_execution()`: Test basic optimization execution.

#### `TestBackwardCompatibility`
Test backward compatibility with single-mission code.

**Methods:**
- `test_create_backward_compatible_problem_single()`: Test backward compatible problem creation for single mission.
- `test_create_backward_compatible_problem_multi()`: Test backward compatible problem creation for multi-mission.
- `test_migrate_single_to_multi_config()`: Test configuration migration from single to multi-mission.

#### `TestConstellationOptimization`
Test high-level constellation optimization function.

**Methods:**
- `test_optimize_constellation_basic()`: Test basic constellation optimization.

#### `TestDimensionalityScaling`
Test dimensionality scaling with different constellation sizes.

**Methods:**
- `test_decision_vector_scaling()`: Test decision vector scaling with different K values.
- `test_population_scaling_recommendations()`: Test population scaling recommendations.

#### `TestRealOptimizationSmall`
Test with real optimization on small problem (fast execution).

**Methods:**
- `test_single_vs_multi_mission_comparison()`: Compare single mission vs 1-mission constellation (should be similar).

#### `TestErrorHandling`
Test error handling and edge cases.

**Methods:**
- `test_invalid_fitness_evaluation()`: Test handling of invalid fitness evaluations.
- `test_genome_validation_errors()`: Test genome validation error handling.
- `test_optimizer_no_results()`: Test optimizer behavior with no results.

---

## tests/test_optimization_basic.py

**Description**: Basic unit tests for optimization modules to improve coverage.

- **Lines of code**: 200
- **Classes**: 6
- **Functions**: 0

### Classes

#### `TestLunarMissionProblem`
Test LunarMissionProblem class.

**Methods:**
- `test_problem_creation()`: Test creating a lunar mission problem.
- `test_problem_bounds()`: Test problem bounds.

#### `TestGlobalOptimizer`
Test GlobalOptimizer class.

**Methods:**
- `test_global_optimizer_creation()`: Test creating a global optimizer.
- `test_optimizer_configuration()`: Test optimizer configuration validation.

#### `TestParetoAnalyzer`
Test ParetoAnalyzer class.

**Methods:**
- `test_pareto_analyzer_creation()`: Test creating a Pareto analyzer.
- `test_pareto_front_extraction()`: Test Pareto front extraction.

#### `TestCostCalculator`
Test CostCalculator class.

**Methods:**
- `test_cost_calculator_creation()`: Test creating a cost calculator.
- `test_mission_cost_calculation()`: Test mission cost calculation.

#### `TestEconomicObjectives`
Test EconomicObjectives class.

**Methods:**
- `test_economic_objectives_creation()`: Test creating economic objectives.
- `test_objectives_to_list()`: Test converting objectives to list format.

#### `TestOptimizationIntegration`
Test integration between optimization modules.

**Methods:**
- `test_problem_fitness_evaluation()`: Test fitness evaluation of lunar mission problem.
- `test_out_of_bounds_handling()`: Test handling of out-of-bounds parameters.

---

## tests/test_optimization_modules.py

**Description**: Optimization Modules Test Suite

- **Lines of code**: 964
- **Classes**: 5
- **Functions**: 3

### Classes

#### `TestLunarMissionProblem`
Test LunarMissionProblem functionality and realism.

**Methods:**
- `test_lunar_mission_problem_initialization()`: Test LunarMissionProblem initialization.
- `test_fitness_evaluation_realism()`: Test fitness evaluation for realistic results.
- `test_parameter_bounds_validation()`: Test parameter bounds and constraint validation.
- `test_fitness_caching_mechanism()`: Test fitness evaluation caching for performance.

#### `TestGlobalOptimizer`
Test GlobalOptimizer functionality and convergence.

**Methods:**
- `test_global_optimizer_initialization()`: Test GlobalOptimizer initialization.
- `test_optimization_with_mock_problem()`: Test optimization with mock problem for basic functionality.
- `test_solution_ranking_and_selection()`: Test solution ranking and selection functionality.
- `test_convergence_detection()`: Test optimization convergence detection.

#### `TestParetoAnalysis`
Test Pareto analysis functionality.

**Methods:**
- `test_dominance_relation()`: Test Pareto dominance relation.
- `test_pareto_analyzer_initialization()`: Test ParetoAnalyzer initialization.
- `test_pareto_front_analysis()`: Test Pareto front analysis functionality.
- `test_solution_preference_ranking()`: Test solution ranking by user preferences.
- `test_hypervolume_calculation()`: Test hypervolume calculation for Pareto front quality.

#### `TestCostIntegration`
Test cost integration functionality.

**Methods:**
- `test_cost_calculator_initialization()`: Test CostCalculator initialization.
- `test_mission_cost_calculation_realism()`: Test mission cost calculation for realistic results.
- `test_cost_sensitivity_analysis()`: Test cost sensitivity to different parameters.

#### `TestOptimizationIntegration`
Test integrated optimization functionality.

**Methods:**
- `test_optimize_lunar_mission_function()`: Test the high-level optimize_lunar_mission function.
- `test_optimization_performance_metrics()`: Test optimization performance and timing.

### Functions

#### `dominates(solution1, solution2)`
Check if solution1 dominates solution2 (minimization).

#### `calculate_hypervolume(pareto_front, reference_point)`
Simple hypervolume calculation for 2D case.

#### `test_optimization_modules_summary()`
Summary test for all optimization modules.

---

## tests/test_physics_validation.py

**Description**: Physics Validation Test Suite

- **Lines of code**: 637
- **Classes**: 5
- **Functions**: 1

### Classes

#### `TestOrbitalMechanicsPhysics`
Test fundamental orbital mechanics physics validation.

**Methods:**
- `test_circular_velocity_earth()`: Test circular orbital velocity around Earth.
- `test_escape_velocity_validation()`: Test escape velocity calculations.
- `test_orbital_period_validation()`: Test orbital period calculations using Kepler's third law.
- `test_energy_conservation_principle()`: Test orbital energy conservation principles.
- `test_vis_viva_equation()`: Test the vis-viva equation: v^2 = mu*(2/r - 1/a).

#### `TestDeltaVValidation`
Test delta-v calculations and realistic ranges.

**Methods:**
- `test_hohmann_transfer_deltav()`: Test Hohmann transfer delta-v calculations.
- `test_lunar_transfer_deltav_ranges()`: Test realistic delta-v ranges for lunar transfers.
- `test_interplanetary_deltav_ranges()`: Test delta-v ranges for interplanetary missions.

#### `TestSpacecraftEngineering`
Test spacecraft engineering constraints and limits.

**Methods:**
- `test_mass_ratio_validation()`: Test spacecraft mass ratios and Tsiolkovsky rocket equation.
- `test_specific_impulse_ranges()`: Test specific impulse ranges for different propulsion systems.
- `test_thrust_to_weight_ratios()`: Test realistic thrust-to-weight ratios.
- `test_spacecraft_mass_components()`: Test spacecraft mass component validation.

#### `TestUnitConsistencyValidation`
Test unit consistency throughout calculations.

**Methods:**
- `test_distance_unit_consistency()`: Test distance unit conversions and consistency.
- `test_velocity_unit_consistency()`: Test velocity unit conversions and consistency.
- `test_time_unit_consistency()`: Test time unit conversions and consistency.
- `test_energy_unit_consistency()`: Test energy unit consistency and conservation.

#### `TestMissionConstraintValidation`
Test mission-level constraints and feasibility.

**Methods:**
- `test_transfer_time_feasibility()`: Test transfer time constraints and feasibility.
- `test_mission_delta_v_budgets()`: Test complete mission delta-v budgets.
- `test_propellant_mass_fraction_limits()`: Test propellant mass fraction limits for different missions.

### Functions

#### `test_physics_validation_summary()`
Summary test to ensure all physics validations are working.

---

## tests/test_prd_compliance.py

**Description**: PRD Compliance Test Suite

- **Lines of code**: 546
- **Classes**: 6
- **Functions**: 1

### Classes

#### `TestPRDWorkflow1MissionConfiguration`
Test PRD Workflow 1: Mission Configuration Loading

**Methods:**
- `test_mission_configuration_creation()`: Test creation of mission configuration with all required parameters.
- `test_configuration_validation()`: Test configuration validation and error handling.

#### `TestPRDWorkflow2GlobalOptimization`
Test PRD Workflow 2: Global Optimization and Pareto Front Generation

**Methods:**
- `test_pareto_front_analysis()`: Test Pareto front generation for multi-objective optimization.
- `test_global_optimizer_initialization()`: Test global optimizer setup with minimal parameters.

#### `TestPRDWorkflow3DifferentiableOptimization`
Test PRD Workflow 3: Local Differentiable Optimization

**Methods:**
- `test_jax_differentiable_optimization()`: Test JAX-based differentiable optimization.
- `test_gradient_computation()`: Test gradient computation for differentiable optimization.

#### `TestPRDWorkflow4EconomicAnalysis`
Test PRD Workflow 4: Economic Analysis (ROI, NPV, IRR)

**Methods:**
- `test_financial_metrics_calculation()`: Test calculation of core financial metrics.
- `test_isru_benefits_analysis()`: Test ISRU (In-Situ Resource Utilization) benefits calculation.
- `test_sensitivity_analysis()`: Test economic sensitivity analysis functionality.

#### `TestPRDWorkflow5Visualization`
Test PRD Workflow 5: Interactive Visualization & Dashboards

**Methods:**
- `test_economic_visualization_creation()`: Test creation of economic visualization dashboards.
- `test_interactive_dashboard_components()`: Test interactive dashboard component creation.
- `test_trajectory_visualization_data()`: Test trajectory data generation for visualization.

#### `TestPRDIntegrationWorkflow`
Test complete PRD workflow integration

**Methods:**
- `test_end_to_end_workflow_simulation()`: Test complete workflow simulation with minimal computation.
- `test_workflow_data_compatibility()`: Test that data flows correctly between workflow steps.

### Functions

#### `test_prd_compliance_summary()`
Summary test to verify all PRD requirements are covered.

---

## tests/test_ray_optimization.py

**Description**: Test suite for Ray-based parallel optimization.

- **Lines of code**: 424
- **Classes**: 4
- **Functions**: 0

### Classes

#### `TestRayAvailability`
Test Ray availability and graceful fallback.

**Methods:**
- `test_ray_import()`: Test Ray import status.
- `test_create_ray_optimizer_fallback()`: Test fallback to regular optimizer when Ray unavailable.

#### `TestFitnessWorker`
Test Ray fitness worker functionality.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `teardown_method()`: Cleanup after tests.
- `test_worker_initialization()`: Test fitness worker initialization.
- `test_single_evaluation()`: Test single fitness evaluation.
- `test_batch_evaluation()`: Test batch fitness evaluation.
- `test_out_of_bounds_handling()`: Test handling of out-of-bounds parameters.
- `test_worker_caching()`: Test worker-level caching.

#### `TestRayParallelOptimizer`
Test Ray parallel optimizer functionality.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `teardown_method()`: Cleanup after tests.
- `test_optimizer_initialization()`: Test Ray optimizer initialization.
- `test_small_optimization_run()`: Test small optimization run for correctness.
- `test_worker_statistics()`: Test collection of worker statistics.
- `test_performance_vs_sequential()`: Test performance comparison with sequential optimizer.

#### `TestRayIntegration`
Test integration of Ray optimization with existing code.

**Methods:**
- `test_create_ray_optimizer_with_config()`: Test creating Ray optimizer with configuration.
- `test_optimization_correctness()`: Test that Ray optimization produces correct results.
- `test_ray_cleanup()`: Test proper Ray cleanup after optimization.

---

## tests/test_real_fast_comprehensive.py

**Description**: Comprehensive Fast Real Tests - No Mocking

- **Lines of code**: 381
- **Classes**: 4
- **Functions**: 2

### Classes

#### `TestRealTrajectoryCore`
Fast real trajectory tests without mocks.

**Methods:**
- `test_earth_moon_trajectory_real()`: Test real Earth-Moon trajectory generation.
- `test_lunar_transfer_real()`: Test real LunarTransfer implementation.
- `test_orbit_parameters_real()`: Test real orbit parameter validation.

#### `TestRealOptimizationCore`
Fast real optimization tests without mocks.

**Methods:**
- `setup_method()`: Setup cost factors for optimization tests.
- `test_lunar_mission_problem_real()`: Test real LunarMissionProblem without mocks.
- `test_real_fitness_evaluation()`: Test real fitness evaluation.
- `test_global_optimizer_real()`: Test real GlobalOptimizer with minimal parameters.
- `test_pareto_analyzer_real()`: Test real ParetoAnalyzer without mocks.

#### `TestRealIntegrationCore`
Fast real integration tests without mocks.

**Methods:**
- `setup_method()`: Setup integration test components.
- `test_optimizer_initialization_real()`: Test real LunarHorizonOptimizer initialization.
- `test_mission_analysis_real()`: Test real mission analysis with minimal parameters.

#### `TestRealEconomicsCore`
Fast real economics tests without mocks.

**Methods:**
- `test_financial_parameters_real()`: Test real FinancialParameters validation.
- `test_roi_calculator_real()`: Test real ROI calculator.
- `test_isru_analyzer_real()`: Test real ISRU benefits analyzer.

### Functions

#### `test_performance_all_modules()`
Test that all real implementations execute quickly.

#### `test_no_mocking_verification()`
Verify that no mocking is used in any tests.

---

## tests/test_real_integration_fast.py

**Description**: Fast Real Integration Tests - No Mocking

- **Lines of code**: 380
- **Classes**: 2
- **Functions**: 1

### Classes

#### `TestRealIntegrationFast`
Fast tests using real integration implementations.

**Methods:**
- `setup_method()`: Setup for each test method.
- `test_real_optimizer_initialization()`: Test real LunarHorizonOptimizer initialization without mocks.
- `test_real_mission_analysis_minimal()`: Test real mission analysis with minimal parameters for speed.
- `test_real_data_flow_between_modules()`: Test real data flow between actual modules (no mocks).
- `test_real_economic_analysis_integration()`: Test real economic analysis integration without mocks.
- `test_real_configuration_validation()`: Test real configuration validation without mocks.
- `test_real_export_functionality()`: Test real export functionality without mocks.

#### `TestRealPerformanceIntegration`
Test real performance characteristics without mocks.

**Methods:**
- `test_real_memory_efficiency()`: Test real memory usage during analysis.
- `test_real_execution_speed()`: Test real execution speed without mocks.
- `test_real_concurrent_analysis()`: Test real concurrent analysis without mocks.

### Functions

#### `test_real_integration_performance_summary()`
Test overall real integration performance.

---

## tests/test_real_optimization_fast.py

**Description**: Fast Real Optimization Tests - No Mocking

- **Lines of code**: 294
- **Classes**: 2
- **Functions**: 1

### Classes

#### `TestRealOptimizationFast`
Fast tests using real optimization implementations.

**Methods:**
- `setup_method()`: Setup for each test method.
- `test_lunar_mission_problem_real()`: Test real LunarMissionProblem without mocks.
- `test_real_fitness_evaluation_fast()`: Test real fitness evaluation with minimal computation.
- `test_global_optimizer_real_minimal()`: Test real GlobalOptimizer with minimal parameters for speed.
- `test_pareto_analyzer_real()`: Test real ParetoAnalyzer without mocks.
- `test_cost_integrator_real()`: Test real CostIntegrator without mocks.
- `test_optimization_performance_real()`: Test that real optimization executes quickly.

#### `TestRealOptimizationValidation`
Test real optimization validation without mocks.

**Methods:**
- `test_population_size_validation_real()`: Test real population size validation.
- `test_objective_validation_real()`: Test real objective function validation.

### Functions

#### `test_optimization_integration_real()`
Test real optimization integration without mocks.

---

## tests/test_real_trajectory_fast.py

**Description**: Fast Real Trajectory Tests - No Mocking

- **Lines of code**: 264
- **Classes**: 2
- **Functions**: 1

### Classes

#### `TestRealTrajectoryFast`
Fast tests using real trajectory implementations.

**Methods:**
- `test_earth_moon_trajectory_minimal()`: Test real Earth-Moon trajectory generation with minimal parameters.
- `test_lunar_transfer_real_implementation()`: Test real LunarTransfer class with fast parameters.
- `test_transfer_windows_calculation_fast()`: Test real transfer window calculations with minimal search range.
- `test_orbit_parameters_validation()`: Test real orbit parameter validation.
- `test_trajectory_optimization_minimal()`: Test real trajectory optimization with minimal parameters.
- `test_trajectory_validation_real()`: Test real trajectory validation without mocks.

#### `TestRealPhysicsValidation`
Test real physics validation without mocking.

**Methods:**
- `test_delta_v_physics_validation()`: Test real delta-v physics calculations.
- `test_time_of_flight_validation()`: Test real time-of-flight validation.

### Functions

#### `test_trajectory_performance_real()`
Test that real trajectory functions execute quickly.

---

## tests/test_real_working_demo.py

**Description**: Working Demo: Real Implementation Tests - No Mocking

- **Lines of code**: 198
- **Classes**: 1
- **Functions**: 1

### Classes

#### `TestRealWorkingDemo`
Minimal working real implementation tests.

**Methods:**
- `test_real_trajectory_generation()`: Test real trajectory generation - NO MOCKS.
- `test_real_optimization_problem()`: Test real optimization problem - NO MOCKS.
- `test_real_economics_calculation()`: Test real economics calculation - NO MOCKS.
- `test_real_performance_validation()`: Test that real implementations are fast.
- `test_real_implementation_verification()`: Verify all tests use real implementations.

### Functions

#### `test_summary_real_vs_mock()`
Summary of real implementation approach.

---

## tests/test_simple_coverage.py

**Description**: Simple coverage tests - just import modules to boost coverage.

- **Lines of code**: 198
- **Classes**: 0
- **Functions**: 8

### Functions

#### `test_import_all_major_modules()`
Import all major modules to exercise import-time code.

#### `test_basic_imports_with_minimal_usage()`
Test basic imports with minimal safe usage.

#### `test_financial_models_minimal()`
Test financial models with minimal functionality.

#### `test_spacecraft_config_minimal()`
Test spacecraft config with correct API.

#### `test_visualization_imports()`
Test visualization module imports.

#### `test_extensibility_imports()`
Test extensibility module imports.

#### `test_trajectory_calculations_minimal()`
Test trajectory calculations with minimal functionality.

#### `test_optimization_basic()`
Test optimization modules basic functionality.

---

## tests/test_target_state.py

- **Lines of code**: 167
- **Classes**: 0
- **Functions**: 8

### Functions

#### `test_target_state_basic()`
Test basic target state calculation.

#### `test_target_state_velocity_matching()`
Test that target state properly matches Moon's velocity.

#### `test_target_state_invalid_inputs()`
Test error handling for invalid inputs.

#### `test_physical_constants()`
Verify physical constants are in correct ranges and relationships.

#### `test_circular_orbit_velocities()`
Test circular orbit velocity calculations at different altitudes.

#### `test_target_state_edge_cases()`
Test target state calculation with edge cases.

#### `test_target_state_energy()`
Test energy of the resulting orbit relative to the Moon.

#### `test_target_state_angular_momentum()`
Test conservation of angular momentum in target orbit.

---

## tests/test_task_10_extensibility.py

**Description**: Comprehensive test suite for Task 10 - Extensibility Interface.

- **Lines of code**: 718
- **Classes**: 10
- **Functions**: 0

### Classes

#### `TestExtensionMetadata`
Test extension metadata functionality.

**Methods:**
- `test_extension_metadata_creation()`: Test basic extension metadata creation.
- `test_extension_metadata_with_dependencies()`: Test extension metadata with dependencies.

#### `MockExtension`
Mock extension for testing.

**Methods:**
- `initialize()`: No docstring
- `validate_configuration()`: No docstring
- `get_capabilities()`: No docstring

#### `TestBaseExtension`
Test base extension functionality.

**Methods:**
- `test_base_extension_creation()`: Test base extension creation and properties.
- `test_extension_enable_disable()`: Test extension enable/disable functionality.
- `test_extension_status()`: Test extension status reporting.

#### `TestExtensionRegistry`
Test extension registry functionality.

**Methods:**
- `setup_method()`: Set up test registry.
- `test_registry_initialization()`: Test registry initialization.
- `test_register_extension_class()`: Test extension class registration.
- `test_get_extensions_by_type()`: Test getting extensions by type.
- `test_unregister_extension()`: Test extension unregistration.
- `test_list_registered_extensions()`: Test listing registered extensions.

#### `TestExtensionManager`
Test extension manager functionality.

**Methods:**
- `setup_method()`: Set up test manager.
- `test_manager_initialization()`: Test manager initialization.
- `test_register_extension()`: Test extension registration with manager.
- `test_unregister_extension()`: Test extension unregistration.
- `test_get_extensions_by_type()`: Test getting extensions by type from manager.
- `test_enable_disable_extension()`: Test enabling/disabling extensions through manager.

#### `TestDataTransformLayer`
Test data transformation layer.

**Methods:**
- `setup_method()`: Set up test transformer.
- `test_normalize_trajectory_state()`: Test trajectory state normalization.
- `test_normalize_optimization_result()`: Test optimization result normalization.
- `test_normalize_cost_breakdown()`: Test cost breakdown normalization.
- `test_unit_conversion()`: Test unit conversion functionality.
- `test_validate_extension_data()`: Test extension data validation.

#### `TestPluginInterface`
Test plugin interface functionality.

**Methods:**
- `setup_method()`: Set up test plugin interface.
- `test_plugin_interface_initialization()`: Test plugin interface initialization.
- `test_register_plugin()`: Test plugin registration.
- `test_get_trajectory_planners()`: Test getting trajectory planners.
- `test_plugin_result_creation()`: Test plugin result creation.

#### `TestLunarDescentExtension`
Test lunar descent extension example.

**Methods:**
- `setup_method()`: Set up test extension.
- `test_lunar_descent_initialization()`: Test lunar descent extension initialization.
- `test_lunar_descent_validation()`: Test configuration validation.
- `test_lunar_descent_capabilities()`: Test getting capabilities.
- `test_lunar_descent_trajectory_planning()`: Test trajectory planning functionality.
- `test_lunar_descent_cost_estimation()`: Test cost estimation.

#### `TestCustomCostModel`
Test custom cost model extension example.

**Methods:**
- `setup_method()`: Set up test extension.
- `test_custom_cost_model_initialization()`: Test custom cost model initialization.
- `test_custom_cost_model_validation()`: Test configuration validation.
- `test_custom_cost_model_capabilities()`: Test getting capabilities.
- `test_mission_cost_estimation()`: Test mission cost estimation.
- `test_cost_breakdown_analysis()`: Test detailed cost breakdown.
- `test_cost_driver_analysis()`: Test cost driver analysis.

#### `TestExtensibilityIntegration`
Test integration between extensibility components.

**Methods:**
- `setup_method()`: Set up integration test components.
- `test_end_to_end_extension_workflow()`: Test complete extension workflow.
- `test_multiple_extension_types()`: Test managing multiple extension types.
- `test_data_transformation_integration()`: Test data transformation with extensions.

---

## tests/test_task_3_trajectory_generation.py

**Description**: Comprehensive test suite for Task 3: Enhanced Trajectory Generation

- **Lines of code**: 727
- **Classes**: 8
- **Functions**: 3

### Classes

#### `TestLambertSolver`
Test suite for Lambert problem solver.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_lambert_solver_initialization()`: Test Lambert solver initialization.
- `test_lambert_solution_earth_orbit()`: Test Lambert solution for Earth orbit transfer.
- `test_lambert_multiple_revolutions()`: Test multiple revolution Lambert solutions.
- `test_lambert_deltav_calculation()`: Test delta-v calculation for Lambert transfer.

#### `TestPatchedConicsApproximation`
Test suite for patched conics approximation.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_patched_conics_initialization()`: Test patched conics initialization.
- `test_earth_moon_trajectory_calculation()`: Test Earth-Moon trajectory calculation.

#### `TestTrajectoryWindowAnalyzer`
Test suite for trajectory window analysis.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_analyzer_initialization()`: Test trajectory window analyzer initialization.
- `test_find_transfer_windows_mock()`: Test transfer window finding with mocked trajectory generation.
- `test_datetime_to_pykep_epoch_conversion()`: Test datetime to PyKEP epoch conversion.
- `test_c3_energy_calculation()`: Test C3 energy calculation.

#### `TestNBodyPropagator`
Test suite for N-body dynamics propagation.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_propagator_initialization()`: Test N-body propagator initialization.
- `test_trajectory_propagation_mock()`: Test trajectory propagation with mocked integrator.
- `test_nbody_dynamics_function()`: Test n-body dynamics function.

#### `TestNumericalIntegrator`
Test suite for numerical integration methods.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_integrator_initialization()`: Test numerical integrator initialization.
- `test_simple_harmonic_oscillator()`: Test integration with simple harmonic oscillator.
- `test_energy_conservation_orbit()`: Test energy conservation in orbital mechanics.

#### `TestTrajectoryIO`
Test suite for trajectory I/O operations.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_trajectory_io_initialization()`: Test trajectory I/O initialization.
- `test_trajectory_save_load_json()`: Test trajectory save and load in JSON format.
- `test_propagation_result_save_load()`: Test propagation result save and load.

#### `TestTrajectoryOptimization`
Test suite for trajectory optimization.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_optimizer_initialization()`: Test trajectory optimizer initialization.
- `test_single_objective_optimization_mock()`: Test single objective optimization with mocked trajectory generation.
- `test_pareto_front_analysis_mock()`: Test Pareto front analysis with mocked trajectory generation.

#### `TestTask3Integration`
Integration tests for Task 3 modules.

**Methods:**
- `test_end_to_end_trajectory_generation_mock()`: Test end-to-end trajectory generation workflow.
- `test_module_imports()`: Test that all Task 3 modules can be imported.

### Functions

#### `sample_orbit_state()`
Fixture providing sample orbit state.

#### `sample_trajectory()`
Fixture providing sample trajectory.

#### `test_configuration()`
Test configuration and environment setup.

---

## tests/test_task_4_global_optimization.py

**Description**: Comprehensive test suite for Task 4: Global Optimization Module

- **Lines of code**: 938
- **Classes**: 6
- **Functions**: 1

### Classes

#### `TestLunarMissionProblem`
Test suite for PyGMO lunar mission problem implementation.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_problem_initialization()`: Test lunar mission problem initialization.
- `test_get_bounds()`: Test optimization bounds.
- `test_get_nobj()`: Test number of objectives.
- `test_get_nec()`: Test number of equality constraints.
- `test_get_nic()`: Test number of inequality constraints.
- `test_fitness_evaluation_real()`: Test fitness evaluation with real implementations - NO MOCKING.
- `test_fitness_bounds_checking()`: Test fitness evaluation with invalid bounds.
- `test_caching_mechanism()`: Test trajectory caching for performance.
- `test_problem_name()`: Test problem name for PyGMO.

#### `TestGlobalOptimizer`
Test suite for PyGMO global optimizer.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_optimizer_initialization()`: Test global optimizer initialization.
- `test_nsga2_algorithm_setup()`: Test NSGA-II algorithm configuration.
- `test_optimization_execution_real()`: Test optimization execution with real implementation - fast version.
- `test_convergence_monitoring()`: Test optimization convergence monitoring.
- `test_best_solutions_extraction()`: Test extraction of best solutions using real optimization.

#### `TestParetoAnalyzer`
Test suite for Pareto front analysis tools.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_analyzer_initialization()`: Test Pareto analyzer initialization.
- `test_pareto_front_analysis()`: Test Pareto front analysis functionality.
- `test_solution_ranking()`: Test solution ranking by preference.
- `test_normalization_methods()`: Test different normalization methods.
- `test_pareto_dominance_check()`: Test Pareto dominance relationships.

#### `TestCostIntegration`
Test suite for cost integration with optimization.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_cost_calculator_initialization()`: Test cost calculator initialization.
- `test_mission_cost_calculation()`: Test mission cost calculation.
- `test_trajectory_cost_calculation()`: Test trajectory-specific cost calculation.
- `test_cost_scaling_factors()`: Test cost scaling with different parameters.
- `test_cost_breakdown_details()`: Test detailed cost breakdown.

#### `TestTask4Integration`
Integration tests for Task 4 modules.

**Methods:**
- `test_optimization_pipeline_integration()`: Test complete optimization pipeline integration.
- `test_module_imports()`: Test that all Task 4 modules can be imported.
- `test_end_to_end_optimization_real()`: Test end-to-end optimization workflow with real implementation - fast version.

#### `TestTask4Performance`
Performance tests for Task 4 modules.

**Methods:**
- `test_fitness_evaluation_performance()`: Test fitness evaluation performance.
- `test_optimization_memory_usage()`: Test optimization memory usage with real implementation - fast version.

### Functions

#### `test_task4_configuration()`
Test Task 4 configuration and environment setup.

---

## tests/test_task_5_economic_analysis.py

**Description**: Comprehensive test suite for Task 5: Basic Economic Analysis Module

- **Lines of code**: 1032
- **Classes**: 6
- **Functions**: 1

### Classes

#### `TestFinancialModels`
Test suite for core financial analysis models.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_financial_parameters_initialization()`: Test financial parameters initialization.
- `test_cash_flow_model_initialization()`: Test cash flow model initialization.
- `test_cash_flow_creation()`: Test individual cash flow creation.
- `test_development_costs_addition()`: Test adding development costs to cash flow model.
- `test_launch_costs_addition()`: Test adding launch costs.
- `test_operational_costs_addition()`: Test adding operational costs.
- `test_revenue_stream_addition()`: Test adding revenue streams.
- `test_npv_calculation()`: Test Net Present Value calculation.
- `test_irr_calculation()`: Test Internal Rate of Return calculation.
- `test_payback_period_calculation()`: Test payback period calculation.
- `test_roi_calculation()`: Test Return on Investment calculation.

#### `TestCostModels`
Test suite for mission cost modeling.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_mission_cost_model_initialization()`: Test mission cost model initialization.
- `test_total_mission_cost_estimation()`: Test total mission cost estimation.
- `test_cost_scaling_factors()`: Test cost scaling with different parameters.
- `test_launch_cost_optimization()`: Test launch vehicle cost optimization.
- `test_operational_cost_modeling()`: Test operational cost modeling.

#### `TestISRUBenefits`
Test suite for ISRU benefits analysis.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_analyzer_initialization()`: Test ISRU analyzer initialization.
- `test_resource_properties()`: Test lunar resource properties.
- `test_isru_economic_analysis()`: Test comprehensive ISRU economic analysis.
- `test_isru_vs_earth_supply_comparison()`: Test ISRU vs Earth supply comparison.
- `test_resource_value_calculation()`: Test resource value calculation.
- `test_facility_scaling_analysis()`: Test ISRU facility scaling analysis.

#### `TestSensitivityAnalysis`
Test suite for sensitivity and risk analysis.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_analyzer_initialization()`: Test sensitivity analyzer initialization.
- `test_one_way_sensitivity_analysis()`: Test one-way sensitivity analysis.
- `test_scenario_analysis()`: Test scenario analysis.
- `test_monte_carlo_simulation()`: Test Monte Carlo simulation.
- `test_tornado_diagram_data()`: Test tornado diagram data generation.

#### `TestEconomicReporting`
Test suite for economic reporting and data export.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_reporter_initialization()`: Test economic reporter initialization.
- `test_financial_summary_creation()`: Test financial summary data structure.
- `test_executive_summary_generation()`: Test executive summary report generation.
- `test_json_export()`: Test JSON data export.
- `test_csv_export()`: Test CSV data export.
- `test_detailed_financial_report()`: Test detailed financial analysis report.
- `test_comparative_analysis_report()`: Test comparative analysis report for multiple alternatives.

#### `TestTask5Integration`
Integration tests for Task 5 modules.

**Methods:**
- `test_end_to_end_economic_analysis()`: Test complete end-to-end economic analysis workflow.
- `test_module_imports()`: Test that all Task 5 modules can be imported.
- `test_cross_module_data_flow()`: Test data flow between Task 5 modules.

### Functions

#### `test_task5_configuration()`
Test Task 5 configuration and environment setup.

---

## tests/test_task_6_visualization.py

**Description**: Comprehensive test suite for Task 6: Visualization Module

- **Lines of code**: 1048
- **Classes**: 6
- **Functions**: 2

### Classes

#### `TestTrajectoryVisualization`
Test suite for trajectory visualization module.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_trajectory_visualizer_initialization()`: Test trajectory visualizer initialization.
- `test_trajectory_plot_config_defaults()`: Test trajectory plot configuration defaults.
- `test_create_3d_trajectory_plot_with_sample_data()`: Test 3D trajectory plot creation with realistic sample data.
- `test_create_transfer_window_plot_with_real_data()`: Test transfer window plot with real transfer window analyzer.
- `test_orbital_elements_calculation_sanity()`: Test orbital elements calculation with realistic data.
- `test_quick_trajectory_plot_function()`: Test quick trajectory plot creation function.

#### `TestOptimizationVisualization`
Test suite for optimization visualization module.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_optimization_visualizer_initialization()`: Test optimization visualizer initialization.
- `test_pareto_front_plot_with_realistic_data()`: Test Pareto front plot with realistic optimization data.
- `test_solution_comparison_plot()`: Test solution comparison visualization.
- `test_preference_analysis_plot()`: Test preference-based solution ranking visualization.
- `test_quick_pareto_plot_function()`: Test quick Pareto plot creation function.

#### `TestEconomicVisualization`
Test suite for economic visualization module.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_economic_visualizer_initialization()`: Test economic visualizer initialization.
- `test_financial_dashboard_with_realistic_data()`: Test financial dashboard with realistic lunar mission data.
- `test_cost_analysis_dashboard()`: Test cost analysis dashboard visualization.
- `test_isru_analysis_dashboard()`: Test ISRU economic analysis dashboard.
- `test_quick_financial_dashboard_function()`: Test quick financial dashboard creation function.

#### `TestMissionVisualization`
Test suite for mission visualization module.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_mission_visualizer_initialization()`: Test mission visualizer initialization.
- `test_mission_timeline_with_realistic_phases()`: Test mission timeline with realistic lunar mission phases.
- `test_resource_utilization_chart()`: Test resource utilization visualization.
- `test_mission_dashboard()`: Test comprehensive mission dashboard.
- `test_sample_mission_timeline_function()`: Test sample mission timeline creation function.

#### `TestComprehensiveDashboard`
Test suite for comprehensive dashboard integration.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_comprehensive_dashboard_initialization()`: Test comprehensive dashboard initialization.
- `test_executive_dashboard_creation()`: Test executive dashboard with realistic mission data.
- `test_technical_dashboard_creation()`: Test technical dashboard creation.
- `test_sample_dashboard_function()`: Test sample dashboard creation function.

#### `TestVisualizationIntegration`
Test suite for visualization module integration and sanity checks.

**Methods:**
- `setup_method()`: Setup integration test fixtures.
- `test_visualization_module_imports()`: Test that all visualization modules can be imported.
- `test_realistic_value_ranges()`: Test that all visualizations handle realistic value ranges properly.
- `test_plot_output_validation()`: Test that plot outputs are valid Plotly figures.
- `test_visualization_output_manual()`: Manual test for visual inspection of plots (skip in automated tests).

### Functions

#### `test_orbit_altitude_range(altitude)`
Test visualization with different orbit altitudes.

#### `test_transfer_time_range(transfer_time)`
Test visualization with different transfer times.

---

## tests/test_task_7_integration.py

**Description**: Test suite for Task 7: MVP Integration.

- **Lines of code**: 445
- **Classes**: 3
- **Functions**: 0

### Classes

#### `TestTask7MVPIntegration`
Test complete MVP integration functionality.

**Methods:**
- `setup_method()`: Setup test fixtures.
- `test_initialization_all_components()`: Test that all system components initialize correctly.
- `test_configuration_compatibility()`: Test configuration system compatibility across modules.
- `test_end_to_end_pipeline_minimal()`: Test minimal end-to-end pipeline execution.
- `test_data_flow_between_modules()`: Test data flow and compatibility between system modules.
- `test_error_handling_and_recovery()`: Test system error handling and recovery mechanisms.
- `test_caching_and_performance()`: Test caching mechanisms and performance optimizations.
- `test_export_functionality()`: Test results export functionality.
- `test_visualization_integration()`: Test visualization component integration.
- `test_logging_and_monitoring()`: Test logging and monitoring capabilities.
- `test_configuration_file_workflow()`: Test configuration file-based workflow.

#### `TestTask7CLIIntegration`
Test CLI interface integration.

**Methods:**
- `test_cli_validation_command()`: Test CLI validation functionality.
- `test_cli_config_generation()`: Test CLI configuration generation.

#### `TestTask7PerformanceIntegration`
Test system performance and scalability.

**Methods:**
- `test_memory_usage()`: Test memory usage during analysis.
- `test_concurrent_analysis()`: Test system behavior with concurrent analyses.

---

## tests/test_task_7_mvp_integration.py

**Description**: Task 7: MVP Integration - Comprehensive Test Suite

- **Lines of code**: 616
- **Classes**: 7
- **Functions**: 1

### Classes

#### `TestLunarHorizonOptimizerInitialization`
Test initialization of the main optimizer class.

**Methods:**
- `test_default_initialization()`: Test initialization with default parameters.
- `test_custom_configuration_initialization()`: Test initialization with custom configurations.

#### `TestIntegratedAnalysisWorkflow`
Test the complete integrated analysis workflow.

**Methods:**
- `optimizer()`: Create optimizer instance for testing.
- `optimization_config()`: Create optimization configuration for testing.
- `test_trajectory_analysis_component()`: Test trajectory analysis component.
- `test_optimization_component()`: Test optimization component.
- `test_economic_analysis_component()`: Test economic analysis component.
- `test_visualization_component()`: Test visualization component.
- `test_end_to_end_analysis()`: Test complete end-to-end analysis workflow.

#### `TestConfigurationManagement`
Test configuration management and validation.

**Methods:**
- `test_default_configurations()`: Test default configuration creation.
- `test_optimization_configuration()`: Test optimization configuration validation.

#### `TestDataExportAndResults`
Test data export and results management.

**Methods:**
- `sample_results()`: Create sample analysis results for testing.
- `test_analysis_results_structure()`: Test AnalysisResults data structure.
- `test_export_functionality()`: Test export functionality.

#### `TestErrorHandlingAndRobustness`
Test error handling and system robustness.

**Methods:**
- `test_missing_dependency_handling()`: Test handling of missing dependencies.
- `test_invalid_configuration_handling()`: Test handling of invalid configurations.
- `test_optimization_failure_handling()`: Test handling of optimization failures.

#### `TestPerformanceAndScalability`
Test performance characteristics and scalability.

**Methods:**
- `test_memory_usage_reasonable()`: Test that memory usage is reasonable for typical problems.
- `test_small_problem_performance()`: Test performance on small optimization problems.

#### `TestSystemIntegration`
Test complete system integration scenarios.

**Methods:**
- `test_typical_mission_scenario()`: Test a typical lunar mission analysis scenario.
- `test_configuration_consistency()`: Test that configurations remain consistent through analysis.

### Functions

#### `test_integration_summary()`
Comprehensive integration test summary.

---

## tests/test_task_8_differentiable_optimization.py

**Description**: Test suite for Task 8: JAX Differentiable Optimization Module

- **Lines of code**: 1370
- **Classes**: 10
- **Functions**: 0

### Classes

#### `TestJAXInfrastructure`
Test JAX infrastructure and environment setup.

**Methods:**
- `test_jax_availability()`: Test that JAX is available and properly configured.
- `test_jax_environment_validation()`: Test JAX environment validation.
- `test_jax_device_info()`: Test JAX device information retrieval.
- `test_basic_jax_operations()`: Test basic JAX operations work correctly.
- `test_performance_optimization_imports()`: Test that performance optimization module imports correctly.
- `test_jit_compilation()`: Test JIT compilation functionality.

#### `TestTrajectoryModel`
Test JAX-based differentiable trajectory models.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_model_initialization()`: Test trajectory model initialization.
- `test_orbital_velocity_calculation()`: Test orbital velocity calculations.
- `test_orbital_energy_calculation()`: Test orbital energy calculations.
- `test_hohmann_transfer_calculation()`: Test Hohmann transfer calculations.
- `test_lambert_solver_simple()`: Test simplified Lambert solver.
- `test_trajectory_cost_calculation()`: Test complete trajectory cost calculation.
- `test_evaluate_trajectory()`: Test trajectory evaluation interface.
- `test_gradient_computation()`: Test automatic differentiation of trajectory model.
- `test_jit_compilation_performance()`: Test JIT compilation performance benefits.

#### `TestEconomicModel`
Test JAX-based differentiable economic models.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_model_initialization()`: Test economic model initialization.
- `test_launch_cost_model()`: Test launch cost calculations.
- `test_operations_cost_model()`: Test operations cost calculations.
- `test_npv_calculation()`: Test Net Present Value calculations.
- `test_roi_calculation()`: Test Return on Investment calculations.
- `test_economic_cost_calculation()`: Test complete economic cost calculation.
- `test_evaluate_economics()`: Test economics evaluation interface.
- `test_gradient_computation()`: Test automatic differentiation of economic model.

#### `TestCombinedModel`
Test combined trajectory-economic optimization models.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_combined_model_creation()`: Test combined model creation with different weights.
- `test_combined_model_evaluation()`: Test combined model evaluation.
- `test_combined_model_gradient()`: Test gradient computation of combined model.
- `test_weight_sensitivity()`: Test sensitivity to different weight combinations.

#### `TestDifferentiableOptimizer`
Test JAX-based differentiable optimizer.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_optimizer_initialization()`: Test optimizer initialization with different configurations.
- `test_simple_optimization()`: Test optimization of simple quadratic function.
- `test_trajectory_optimization()`: Test optimization of trajectory-economic problem.
- `test_optimization_result_structure()`: Test optimization result structure and metrics.
- `test_batch_optimization()`: Test batch optimization with multiple initial points.
- `test_optimization_comparison()`: Test optimization result comparison analysis.

#### `TestOptimizationDemonstration`
Test complete optimization demonstration.

**Methods:**
- `test_demonstration_initialization()`: Test optimization demonstration initialization.
- `test_initial_guess_generation()`: Test initial guess generation.
- `test_solution_evaluation()`: Test solution evaluation functionality.
- `test_complete_demonstration()`: Test complete optimization demonstration workflow.
- `test_solution_comparison()`: Test solution comparison functionality.
- `test_quick_demo_function()`: Test quick demonstration function.

#### `TestIntegrationAndPerformance`
Test integration and performance aspects.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_module_import_performance()`: Test module import performance.
- `test_end_to_end_performance()`: Test end-to-end optimization performance.
- `test_memory_usage()`: Test reasonable memory usage.
- `test_numerical_stability()`: Test numerical stability with extreme parameter values.

#### `TestPerformanceOptimization`
Test performance optimization features.

**Methods:**
- `test_jit_optimizer_functionality()`: Test JIT optimizer function compilation.
- `test_batch_optimizer_functionality()`: Test batch optimizer basic functionality.
- `test_memory_optimizer_functionality()`: Test memory optimizer basic functionality.
- `test_performance_benchmark_functionality()`: Test performance benchmark basic functionality.
- `test_enhanced_jax_optimizer()`: Test enhanced JAX optimizer with performance features.

#### `TestResultComparison`
Test result comparison and evaluation functionality.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_result_comparator_initialization()`: Test result comparator initialization.
- `test_optimization_result_comparison()`: Test comparison of optimization results.
- `test_convergence_analysis()`: Test convergence analysis functionality.
- `test_solution_ranking()`: Test solution ranking functionality.
- `test_pareto_front_computation()`: Test Pareto front computation.
- `test_utility_functions()`: Test utility functions for result comparison.

#### `TestComparisonDemonstration`
Test comparison demonstration functionality.

**Methods:**
- `test_comparison_demo_initialization()`: Test comparison demo initialization.
- `test_single_comparison_demo()`: Test single comparison demonstration.
- `test_convergence_analysis_demo()`: Test convergence analysis demonstration.
- `test_solution_ranking_demo()`: Test solution ranking demonstration.
- `test_pareto_analysis_demo()`: Test Pareto analysis demonstration.
- `test_method_benchmark_demo()`: Test method benchmarking demonstration.
- `test_comprehensive_analysis_demo()`: Test comprehensive analysis demonstration.
- `test_comparison_demo_runner()`: Test comparison demo runner function.

---

## tests/test_task_9_enhanced_economics.py

**Description**: Test suite for Task 9: Enhanced Economic Analysis Module

- **Lines of code**: 531
- **Classes**: 3
- **Functions**: 0

### Classes

#### `TestTimeBasedISRUModels`
Test time-based ISRU production models.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_production_profile_validation()`: Test production profile validation.
- `test_production_rate_calculation()`: Test production rate calculation over time.
- `test_cumulative_production()`: Test cumulative production calculation.
- `test_time_dependent_economics()`: Test economic calculations with time-dependent production.
- `test_facility_deployment_optimization()`: Test ISRU facility deployment optimization.
- `test_production_forecast()`: Test ISRU production forecast generation.

#### `TestScenarioComparison`
Test advanced scenario comparison tools.

**Methods:**
- `setUp()`: Set up test fixtures.
- `test_scenario_addition()`: Test adding scenarios to comparator.
- `test_single_scenario_analysis()`: Test analysis of a single scenario.
- `test_scenario_comparison()`: Test comparison of multiple scenarios.
- `test_scenario_ranking()`: Test multi-criteria scenario ranking.
- `test_decision_matrix_generation()`: Test decision matrix generation.
- `test_monte_carlo_simulation()`: Test Monte Carlo simulation for uncertainty analysis.
- `test_scenario_comparison_report()`: Test comprehensive comparison report generation.

#### `TestEnhancedEconomicsIntegration`
Test integration with existing economic modules.

**Methods:**
- `test_isru_model_integration()`: Test integration of time-based ISRU with existing analyzers.
- `test_financial_model_compatibility()`: Test compatibility with existing financial models.

---

## tests/test_trajectory_basic.py

**Description**: Basic unit tests for trajectory modules to improve coverage.

- **Lines of code**: 219
- **Classes**: 6
- **Functions**: 0

### Classes

#### `TestTrajectoryConstants`
Test trajectory constants module.

**Methods:**
- `test_earth_constants()`: Test Earth gravitational parameter and radius.
- `test_moon_constants()`: Test Moon gravitational parameter and radius.
- `test_relative_sizes()`: Test relative sizes of celestial bodies.

#### `TestOrbitalElements`
Test orbital elements calculations.

**Methods:**
- `test_orbital_period_calculation()`: Test orbital period calculation.
- `test_velocity_at_point()`: Test velocity calculation at orbital point.
- `test_anomaly_conversions()`: Test anomaly conversion functions.

#### `TestOrbitState`
Test OrbitState class.

**Methods:**
- `test_orbit_state_creation()`: Test creating an orbit state.
- `test_orbit_state_properties()`: Test orbit state calculated properties.

#### `TestManeuver`
Test Maneuver class.

**Methods:**
- `test_maneuver_creation()`: Test creating a maneuver.
- `test_maneuver_magnitude()`: Test maneuver magnitude calculation.

#### `TestTargetState`
Test target state calculations.

**Methods:**
- `test_target_state_calculation()`: Test target state calculation.

#### `TestTrajectoryModels`
Test trajectory models and basic functionality.

**Methods:**
- `test_circular_orbit_velocity()`: Test circular orbit velocity calculation.
- `test_escape_velocity()`: Test escape velocity calculation.
- `test_sphere_of_influence()`: Test basic sphere of influence concepts.

---

## tests/test_trajectory_modules.py

**Description**: Trajectory Modules Test Suite

- **Lines of code**: 758
- **Classes**: 5
- **Functions**: 1

### Classes

#### `TestLambertSolver`
Test Lambert problem solver functionality and physics validation.

**Methods:**
- `test_lambert_solver_initialization()`: Test LambertSolver initialization.
- `test_lambert_problem_earth_orbit()`: Test Lambert problem for Earth orbital transfer.
- `test_lambert_problem_lunar_transfer()`: Test Lambert problem for trans-lunar injection scenario.
- `test_lambert_energy_conservation()`: Test energy conservation in Lambert solutions.
- `test_lambert_short_vs_long_way()`: Test Lambert solver for short-way vs long-way transfers.

#### `TestNBodyIntegration`
Test N-body integration functionality and accuracy.

**Methods:**
- `test_numerical_integrator_initialization()`: Test NumericalIntegrator initialization.
- `test_earth_moon_nbody_propagator()`: Test EarthMoonNBodyPropagator functionality.
- `test_energy_conservation_nbody()`: Test energy conservation in N-body propagation.
- `test_trajectory_io_functionality()`: Test trajectory I/O functionality.

#### `TestEarthMoonTrajectories`
Test Earth-Moon trajectory generation functionality.

**Methods:**
- `test_generate_earth_moon_trajectory_lambert()`: Test Earth-Moon trajectory generation using Lambert solver.
- `test_generate_earth_moon_trajectory_patched_conics()`: Test Earth-Moon trajectory generation using patched conics.
- `test_patched_conics_approximation()`: Test PatchedConicsApproximation functionality.
- `test_optimal_timing_calculator()`: Test OptimalTimingCalculator functionality.

#### `TestTransferWindowAnalysis`
Test transfer window analysis functionality.

**Methods:**
- `test_trajectory_window_analyzer_initialization()`: Test TrajectoryWindowAnalyzer initialization.
- `test_find_transfer_windows()`: Test transfer window finding functionality.
- `test_transfer_window_optimization()`: Test transfer window optimization functionality.

#### `TestTrajectoryOptimization`
Test trajectory optimization functionality.

**Methods:**
- `test_trajectory_optimizer_pareto_analysis()`: Test TrajectoryOptimizer Pareto front analysis functionality.

### Functions

#### `test_trajectory_modules_summary()`
Summary test for all trajectory modules.

---

## tests/test_utils_simplified.py

**Description**: Simplified unit tests for utils modules to achieve 80%+ coverage.

- **Lines of code**: 240
- **Classes**: 6
- **Functions**: 0

### Classes

#### `TestDistanceConversions`
Test distance unit conversions.

**Methods:**
- `test_km_to_m()`: Test kilometers to meters conversion.
- `test_m_to_km()`: Test meters to kilometers conversion.
- `test_distance_conversion_roundtrip()`: Test roundtrip distance conversions.

#### `TestAngleConversions`
Test angle unit conversions.

**Methods:**
- `test_deg_to_rad()`: Test degrees to radians conversion.
- `test_rad_to_deg()`: Test radians to degrees conversion.
- `test_angle_conversion_roundtrip()`: Test roundtrip angle conversions.

#### `TestTimeConversions`
Test time unit conversions.

**Methods:**
- `test_days_to_seconds()`: Test days to seconds conversion.
- `test_seconds_to_days()`: Test seconds to days conversion.
- `test_time_conversion_roundtrip()`: Test roundtrip time conversions.

#### `TestVelocityConversions`
Test velocity unit conversions.

**Methods:**
- `test_mps_to_kmps()`: Test m/s to km/s conversion.
- `test_kmps_to_mps()`: Test km/s to m/s conversion.
- `test_velocity_conversion_roundtrip()`: Test roundtrip velocity conversions.

#### `TestDateTimeConversions`
Test datetime conversion functions.

**Methods:**
- `test_datetime_to_mjd2000()`: Test datetime to MJD2000 conversion.
- `test_datetime_to_j2000()`: Test datetime to J2000 conversion.
- `test_datetime_to_pykep_epoch()`: Test datetime to PyKEP epoch conversion.
- `test_pykep_epoch_to_datetime()`: Test PyKEP epoch to datetime conversion.
- `test_datetime_conversion_roundtrip()`: Test roundtrip datetime conversions.
- `test_naive_datetime_error()`: Test that naive datetime raises error.

#### `TestEdgeCases`
Test edge cases and error conditions.

**Methods:**
- `test_negative_values()`: Test conversions with negative values.
- `test_zero_values()`: Test conversions with zero values.
- `test_large_values()`: Test conversions with large values.
- `test_precision_limits()`: Test precision limits for conversions.
- `test_array_type_preservation()`: Test that array types are preserved.

---

## tests/trajectory/test_celestial_bodies.py

**Description**: Tests for celestial body state calculations.

- **Lines of code**: 202
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TestCelestialBodies`
Tests for celestial body state calculations using SPICE.

**Methods:**
- `setup_class()`: Initialize celestial bodies for all tests.
- `test_earth_state_heliocentric()`: Verify Earth's heliocentric state vector calculation.
- `test_moon_state_heliocentric()`: Verify Moon's heliocentric state vector calculation.
- `test_moon_state_earth_centered()`: Verify Moon's geocentric state vector calculation.
- `test_invalid_epoch()`: Verify proper error handling for invalid epochs.
- `test_local_frame()`: Test local frame transformations.

---

## tests/trajectory/test_elements.py

**Description**: Unit tests for orbital elements utility functions.

- **Lines of code**: 259
- **Classes**: 3
- **Functions**: 0

### Classes

#### `TestOrbitalPeriod`
Test suite for orbital period calculations.

**Methods:**
- `test_leo_period()`: Test orbital period calculation for Low Earth Orbit.
- `test_lunar_period()`: Test orbital period calculation for lunar orbit.

#### `TestOrbitalVelocity`
Test suite for orbital velocity calculations.

**Methods:**
- `test_circular_orbit_velocity()`: Test velocity components in a circular orbit.
- `test_elliptical_orbit_velocity()`: Test velocity components in an elliptical orbit.
- `test_velocity_components_perpendicular()`: Test velocity components at key orbital points.

#### `TestAnomalyConversion`
Test suite for anomaly conversion functions.

**Methods:**
- `test_circular_mean_to_true()`: Test mean to true anomaly conversion in circular orbit.
- `test_elliptical_mean_to_true()`: Test mean to true anomaly conversion in elliptical orbit.
- `test_circular_true_to_mean()`: Test true to mean anomaly conversion in circular orbit.
- `test_elliptical_true_to_mean()`: Test true to mean anomaly conversion in elliptical orbit.
- `test_anomaly_conversion_roundtrip()`: Test consistency of anomaly conversions.

---

## tests/trajectory/test_epoch_conversions.py

**Description**: Tests for epoch conversion utilities.

- **Lines of code**: 90
- **Classes**: 1
- **Functions**: 3

### Classes

#### `TestPhysicalConstants`
Test suite for physical constants validation.

**Methods:**
- `test_earth_gravitational_parameter()`: Verify Earth's gravitational parameter (mu) conversion.
- `test_earth_radius()`: Verify Earth radius conversion from PyKEP.

### Functions

#### `test_datetime_to_mjd2000()`
Test conversion from datetime to MJD2000.

#### `test_datetime_to_pykep()`
Test conversion from datetime to PyKEP epoch.

#### `test_pykep_epoch_roundtrip()`
Test roundtrip conversion between datetime and PyKEP epoch.

---

## tests/trajectory/test_hohmann_transfer.py

**Description**: Tests for Hohmann transfer calculations.

- **Lines of code**: 208
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TestHohmannTransfer`
Test suite for Hohmann transfer calculations.

**Methods:**
- `setup_method()`: Set up test fixtures with common orbital parameters.
- `test_transfer_components()`: Test individual components of Hohmann transfer calculation.
- `test_transfer_estimation()`: Test complete Hohmann transfer calculations.
- `test_invalid_radii()`: Test Hohmann transfer validation with invalid radii.
- `test_transfer_vectors()`: Test Hohmann transfer position and velocity vectors.
- `test_circular_velocities()`: Test circular orbit velocities for Hohmann transfer.

---

## tests/trajectory/test_input_validation.py

**Description**: Tests for input parameter validation in trajectory generation.

- **Lines of code**: 185
- **Classes**: 1
- **Functions**: 1

### Classes

#### `TestLunarTransferValidation`
Test suite for lunar transfer parameter validation.

**Methods:**
- `setup_method()`: Set up test fixtures.
- `test_orbit_altitude_validation()`: Test validation of initial orbit altitude constraints.
- `test_time_of_flight_validation()`: Test validation of time of flight constraints.
- `test_delta_v_constraints()`: Test validation of delta-v constraints.
- `test_max_revolutions_validation()`: Test validation of maximum revolutions parameter.

### Functions

#### `test_invalid_time_of_flight()`
Test validation of time of flight.

---

## tests/trajectory/test_lambert_solver.py

**Description**: Tests for Lambert problem solver.

- **Lines of code**: 532
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TestLambertSolver`
Test suite for Lambert problem solver with progressive complexity.

**Methods:**
- `setup_method()`: Set up common test parameters.
- `verify_solution()`: Verify Lambert solution physics and units.
- `test_quarter_orbit_transfer()`: Test simple 90° planar transfer in circular orbit.
- `test_non_planar_transfer()`: Test transfer between inclined orbits.
- `test_hohmann_transfer()`: Test near-Hohmann transfer from LEO to GEO.
- `test_multi_revolution()`: Test Lambert solver with multiple revolutions.
- `test_invalid_inputs()`: Test error handling for invalid inputs.
- `test_circular_orbit_validation()`: Test Lambert solver for circular orbit velocity validation.
- `test_coplanar_transfer_validation()`: Test Lambert solver for coplanar transfer validation.
- `test_hohmann_velocity_checks()`: Test velocity calculations specific to Hohmann-like transfers.
- `test_lunar_transfer()`: Test Lambert solver for Earth-Moon transfer trajectory.

---

## tests/trajectory/test_lunar_transfer.py

**Description**: Tests for lunar transfer trajectory generation.

- **Lines of code**: 604
- **Classes**: 2
- **Functions**: 0

### Classes

#### `TestBasicPhysics`
Test basic physics calculations and vector operations.

**Methods:**
- `test_vector_operations()`: Test fundamental vector operations used in orbital mechanics.
- `test_basic_orbital_mechanics()`: Test basic orbital mechanics relationships.
- `test_escape_velocity()`: Test escape velocity calculations.
- `test_hohmann_transfer()`: Test basic Hohmann transfer calculations.
- `test_basic_perturbation()`: Test basic perturbation calculations.
- `test_physical_constants()`: Test that physical constants are properly defined and have reasonable values.
- `test_orbital_velocities()`: Test calculation of characteristic orbital velocities.
- `test_minimum_energy_transfer()`: Test minimum energy transfer calculations.

#### `TestLunarTransfer`
Test lunar transfer trajectory generation.

**Methods:**
- `setup_method()`: Set up test fixtures.
- `test_circular_velocity()`: Test circular orbit velocity calculation.
- `test_target_state_basic()`: Test basic target state calculation.
- `test_target_state_velocity_matching()`: Test target state velocity matching Moon's velocity.
- `test_target_state_distance()`: Test target state distance from Moon center.
- `test_optimal_phase_simple()`: Test optimal phase finding with simple circular orbit.
- `test_optimal_phase_velocity()`: Test optimal phase velocity constraints.
- `test_lunar_transfer_components()`: Test individual components of lunar transfer.
- `test_loi_delta_v_calculation()`: Test lunar orbit insertion delta-v calculation.
- `test_transfer_orbit_energy()`: Test energy of transfer orbit relative to Earth.
- `test_intermediate_states()`: Test intermediate states during transfer.
- `test_lunar_transfer_generation()`: Test complete lunar transfer trajectory generation.
- `test_multiple_revolutions()`: Test lunar transfer with multiple revolutions.
- `test_edge_case_phase_angles()`: Test transfer generation with edge case phase angles.
- `test_solution_physics_validation()`: Test physics validation of transfer solutions.
- `test_transfer_solution_evaluation()`: Test evaluation of transfer solutions.
- `test_boundary_conditions()`: Test transfer generation at boundary conditions.
- `test_revolution_count_impact()`: Test impact of revolution count on transfer solutions.

---

## tests/trajectory/test_orbit_state.py

**Description**: Tests for orbit state conversions and units.

- **Lines of code**: 182
- **Classes**: 2
- **Functions**: 0

### Classes

#### `TestOrbitStateConversion`
Test suite for orbit state conversions.

**Methods:**
- `test_circular_orbit()`: Test conversion of circular OrbitState to PyKEP planet.
- `test_elliptical_orbit()`: Test conversion of elliptical OrbitState to PyKEP planet.

#### `TestCelestialBodyStates`
Test celestial body state calculations.

**Methods:**
- `test_moon_state()`: Test Moon state vector calculations.
- `test_earth_state()`: Test Earth state vector calculations.

---

## tests/trajectory/test_propagator.py

**Description**: Unit tests for the TrajectoryPropagator class.

- **Lines of code**: 81
- **Classes**: 0
- **Functions**: 4

### Functions

#### `propagator()`
Create a TrajectoryPropagator instance with Earth as the central body.

#### `test_moon_gravity(propagator)`
Test that Moon's gravity affects the trajectory.

#### `test_energy_conservation(propagator)`
Test that energy is approximately conserved during propagation.

#### `test_invalid_inputs(propagator)`
Test that invalid inputs raise appropriate exceptions.

---

## tests/trajectory/test_trajectory_models.py

**Description**: Tests for trajectory model validation and functionality.

- **Lines of code**: 293
- **Classes**: 3
- **Functions**: 0

### Classes

#### `TestOrbitState`
Test suite for OrbitState model validation.

**Methods:**
- `test_valid_parameters()`: Test OrbitState creation with valid parameters.
- `test_invalid_semi_major_axis()`: Test validation of semi-major axis.
- `test_invalid_eccentricity()`: Test validation of eccentricity.
- `test_invalid_inclination()`: Test validation of inclination.
- `test_invalid_angles()`: Test validation of orbital angles.

#### `TestManeuver`
Test suite for Maneuver model validation.

**Methods:**
- `test_valid_parameters()`: Test Maneuver creation with valid parameters.
- `test_invalid_delta_v()`: Test validation of delta-v vector.
- `test_invalid_epoch()`: Test validation of maneuver epoch.

#### `TestTrajectory`
Test suite for Trajectory model validation.

**Methods:**
- `setup_method()`: Set up test fixtures.
- `test_valid_parameters()`: Test Trajectory creation with valid parameters.
- `test_invalid_time_order()`: Test validation of trajectory time ordering.
- `test_invalid_maneuver_timing()`: Test validation of maneuver timing.

---

## tests/trajectory/test_unit_conversions.py

**Description**: Unit conversion test suite.

- **Lines of code**: 371
- **Classes**: 3
- **Functions**: 0

### Classes

#### `TestBasicUnitConversions`
Test suite for basic unit conversions.

**Methods:**
- `test_distance_conversions()`: Test distance unit conversions.
- `test_velocity_conversions()`: Test velocity unit conversions.
- `test_gravitational_parameters()`: Test gravitational parameter conversions.
- `test_time_duration_conversions()`: Test time duration conversions.
- `test_edge_cases()`: Test edge cases and potential error conditions.

#### `TestEpochConversions`
Test suite for epoch and time format conversions.

**Methods:**
- `test_mjd2000_conversion()`: Verify datetime to MJD2000 conversion.
- `test_j2000_conversion()`: Verify datetime to J2000 conversion.
- `test_pykep_epoch_conversion()`: Verify datetime to PyKEP epoch conversion.

#### `TestTrajectoryUnitConversions`
Test unit conversions in trajectory generation components.

**Methods:**
- `test_orbit_state_units()`: Test unit conversions in OrbitState class.
- `test_maneuver_units()`: Test unit conversions in Maneuver class.
- `test_transfer_trajectory_units()`: Test unit conversions in transfer trajectory generation.
- `test_hohmann_estimate_units()`: Test unit handling in Hohmann transfer estimation.

---

## tests/trajectory/test_validator.py

**Description**: Test module for trajectory validation.

- **Lines of code**: 165
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TestTrajectoryValidator`
Test suite for TrajectoryValidator class.

**Methods:**
- `setup_method()`: Set up test fixtures before each test method.
- `test_valid_inputs()`: Test that valid inputs pass validation.
- `test_invalid_earth_altitude()`: Test that invalid Earth orbit altitudes raise exceptions.
- `test_invalid_moon_altitude()`: Test that invalid lunar orbit altitudes raise exceptions.
- `test_invalid_transfer_time()`: Test that invalid transfer times raise exceptions.
- `test_delta_v_validation()`: Test validation of delta-v values.
- `test_edge_cases()`: Test edge cases with minimum and maximum valid values.
- `test_unit_conversions()`: Test that unit conversions are handled correctly.

---

# Module: trajectory

## trajectory/celestial_bodies.py

**Description**: Celestial body definitions and state calculations.

- **Lines of code**: 198
- **Classes**: 1
- **Functions**: 0

### Classes

#### `CelestialBody`
Provides methods to calculate state vectors of celestial bodies.

**Methods:**
- `__init__()`: Initialize a CelestialBody instance.
- `get_earth_state()`: Get Earth's heliocentric state vector at the specified epoch.
- `get_moon_state()`: Get Moon's heliocentric state vector at the specified epoch.
- `get_moon_state_earth_centered()`: Get Moon's state vector relative to Earth at the specified epoch.
- `create_local_frame()`: Create a local orbital reference frame.

---

## trajectory/constants.py

**Description**: Physical constants and unit definitions for trajectory calculations.

- **Lines of code**: 123
- **Classes**: 3
- **Functions**: 0

### Classes

#### `Units`
Unit conversion constants from PyKEP.

#### `PhysicalConstants`
Physical constants in PyKEP native units.

#### `EphemerisLimits`
Time limits for ephemeris calculations.

---

## trajectory/continuous_thrust.py

**Description**: Minimal continuous-thrust propagator using JAX/Diffrax.

- **Lines of code**: 137
- **Classes**: 0
- **Functions**: 3

### Functions

#### `continuous_dynamics(t, state, args)`
Edelbaum planar continuous-thrust dynamics.

#### `low_thrust_transfer(start_state, target_state, T, Isp, tf_guess, alpha_constant, mu)`
Continuous-thrust transfer with constant thrust angle.

#### `optimize_thrust_angle(start_state, target_radius, T, Isp, tf)`
Find optimal constant thrust angle to reach target radius using simple gradient descent.

---

## trajectory/defaults.py

**Description**: Default values and limits for trajectory calculations.

- **Lines of code**: 114
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TransferDefaults`
Default values and limits for transfer trajectory generation.

**Methods:**
- `validate_earth_orbit()`: Validate Earth orbit altitude.
- `validate_moon_orbit()`: Validate lunar orbit altitude.
- `validate_transfer_time()`: Validate transfer time.

---

## trajectory/earth_moon_trajectories.py

**Description**: Earth-Moon trajectory generation functions for Task 3.2 completion.

- **Lines of code**: 651
- **Classes**: 3
- **Functions**: 2

### Classes

#### `LambertSolver`
Lambert problem solver for Earth-Moon trajectory generation.

**Methods:**
- `__init__()`: Initialize Lambert solver.
- `solve_lambert()`: Solve Lambert problem for given position vectors and time.
- `solve_multiple_revolution()`: Solve Lambert problem for multiple revolution cases.
- `calculate_transfer_deltav()`: Calculate delta-v required for Lambert transfer.

#### `PatchedConicsApproximation`
Patched conics approximation for Earth-Moon trajectories.

**Methods:**
- `__init__()`: Initialize patched conics approximation.
- `calculate_trajectory()`: Calculate trajectory using patched conics approximation.
- `_calculate_earth_escape()`: Calculate Earth escape phase.
- `_calculate_earth_moon_transfer()`: Calculate Earth-Moon transfer phase.
- `_calculate_moon_capture()`: Calculate Moon capture phase.

#### `OptimalTimingCalculator`
Calculator for optimal departure and arrival timing.

**Methods:**
- `__init__()`: Initialize optimal timing calculator.
- `find_optimal_departure_time()`: Find optimal departure time within search period.
- `calculate_launch_windows()`: Calculate multiple launch windows for a given month.
- `analyze_timing_sensitivity()`: Analyze sensitivity to timing variations.

### Functions

#### `generate_earth_moon_trajectory(departure_epoch, earth_orbit_alt, moon_orbit_alt, transfer_time, method)`
Convenience function for Earth-Moon trajectory generation.

#### `find_optimal_launch_window(target_date, window_days, earth_orbit_alt, moon_orbit_alt)`
Find optimal launch window around target date.

---

## trajectory/elements.py

**Description**: Utility functions for orbital calculations.

- **Lines of code**: 122
- **Classes**: 0
- **Functions**: 4

### Functions

#### `orbital_period(semi_major_axis, mu)`
Calculate orbital period using Kepler's Third Law.

#### `velocity_at_point(semi_major_axis, eccentricity, true_anomaly, mu)`
Calculate radial and tangential velocity components at a point in orbit.

#### `mean_to_true_anomaly(mean_anomaly, eccentricity)`
Convert mean anomaly to true anomaly using iterative solver.

#### `true_to_mean_anomaly(true_anomaly, eccentricity)`
Convert true anomaly to mean anomaly.

---

## trajectory/generator.py

**Description**: Trajectory generation functions for Earth-Moon transfers.

- **Lines of code**: 236
- **Classes**: 0
- **Functions**: 3

### Functions

#### `generate_lunar_transfer(departure_time, time_of_flight, initial_orbit_alt, final_orbit_alt, max_tli_dv, min_tli_dv, max_revs)`
Generate a lunar transfer trajectory.

#### `optimize_departure_time(reference_epoch, earth_orbit_altitude, moon_orbit_altitude, transfer_time, search_window)`
Find optimal departure time for minimum delta-v transfer.

#### `estimate_hohmann_transfer_dv(r1, r2)`
Calculate delta-v and time of flight for a Hohmann transfer orbit.

---

## trajectory/lambert_solver.py

**Description**: Lambert problem solver for trajectory calculations.

- **Lines of code**: 218
- **Classes**: 0
- **Functions**: 7

### Functions

#### `get_num_solutions(max_revolutions)`
Calculate number of solutions for given maximum revolutions.

#### `solve_lambert(r1, r2, tof, mu, max_revolutions, prograde, solution_index)`
Solve Lambert's problem using PyKEP.

#### `get_all_solutions(r1, r2, tof, mu, max_revolutions, prograde)`
Get all possible solutions for the Lambert problem.

#### `_validate_lambert_inputs(r1, r2, tof, mu)`
Validate inputs for Lambert problem solver.

#### `_prepare_position_vectors(r1, r2)`
Prepare position vectors for PyKEP format.

#### `_solve_lambert_problem(r1, r2, tof, mu, max_revolutions, prograde)`
Create and solve Lambert problem using PyKEP.

#### `_extract_lambert_solutions(lambert, max_revolutions, solution_index)`
Extract velocity solutions from Lambert problem.

---

## trajectory/lunar_transfer.py

**Description**: Lunar transfer trajectory generation module.

- **Lines of code**: 528
- **Classes**: 2
- **Functions**: 0

### Classes

#### `LunarTrajectory`
Simple concrete implementation for lunar transfers.

**Methods:**
- `__init__()`: Initialize lunar trajectory with departure and arrival states.
- `validate_trajectory()`: Validate the lunar trajectory.
- `add_maneuver()`: Add a maneuver to the trajectory.
- `get_total_delta_v()`: Calculate total delta-v cost of all maneuvers.
- `trajectory_data()`: Get trajectory data for visualization and analysis integration.

#### `LunarTransfer`
Generates lunar transfer trajectories using PyKEP.

**Methods:**
- `__init__()`: Initialize lunar transfer trajectory generator.
- `generate_transfer()`: Generate lunar transfer trajectory.
- `_validate_and_prepare_inputs()`: Validate inputs and prepare transfer parameters.
- `_calculate_moon_states()`: Calculate Moon states at departure and arrival.
- `_find_optimal_departure()`: Find optimal departure point and initial orbit conditions.
- `_build_trajectory()`: Build complete trajectory with maneuvers.
- `_calculate_maneuvers()`: Calculate TLI and LOI maneuver delta-v values using Lambert solver.
- `_add_maneuvers_to_trajectory()`: Add TLI and LOI maneuvers to the trajectory.

---

## trajectory/maneuver.py

**Description**: Orbital maneuver representation and calculations.

- **Lines of code**: 131
- **Classes**: 1
- **Functions**: 0

### Classes

#### `Maneuver`
Represents an impulsive orbital maneuver.

**Methods:**
- `__post_init__()`: Validate maneuver parameters after initialization.
- `magnitude()`: Get the magnitude of the delta-v vector in km/s.
- `get_delta_v_si()`: Get the delta-v vector in SI units (m/s).
- `get_delta_v_ms()`: Get the delta-v vector in m/s (alias for get_delta_v_si).
- `apply_to_velocity()`: Apply the maneuver to a velocity vector.
- `scale()`: Create a new maneuver with delta-v scaled by a factor.
- `reverse()`: Create a new maneuver with reversed delta-v direction.
- `__str__()`: String representation of the maneuver.

---

## trajectory/models.py

**Description**: Legacy imports for backward compatibility.

- **Lines of code**: 13
- **Classes**: 0
- **Functions**: 0

---

## trajectory/nbody_dynamics.py

**Description**: N-body dynamics module for enhanced trajectory propagation.

- **Lines of code**: 383
- **Classes**: 2
- **Functions**: 1

### Classes

#### `NBodyPropagator`
N-body gravitational dynamics propagator for accurate trajectory calculation.

**Methods:**
- `__init__()`: Initialize the N-body propagator.
- `propagate_trajectory()`: Propagate trajectory using n-body dynamics.
- `_nbody_dynamics()`: Compute derivatives for n-body gravitational dynamics.
- `calculate_trajectory_accuracy()`: Calculate accuracy improvement of n-body vs two-body propagation.

#### `HighFidelityPropagator`
High-fidelity propagator combining PyKEP and n-body dynamics.

**Methods:**
- `__init__()`: Initialize high-fidelity propagator.
- `propagate_adaptive()`: Adaptive propagation switching between two-body and n-body.
- `compare_propagation_methods()`: Compare different propagation methods.

### Functions

#### `enhanced_trajectory_propagation(initial_position, initial_velocity, time_of_flight, high_fidelity)`
Enhanced trajectory propagation for Task 3 completion.

---

## trajectory/nbody_integration.py

**Description**: N-body dynamics integration and trajectory I/O for Task 3.3 completion.

- **Lines of code**: 973
- **Classes**: 4
- **Functions**: 2

### Classes

#### `NumericalIntegrator`
Numerical integration methods for trajectory propagation.

**Methods:**
- `__init__()`: Initialize numerical integrator.
- `integrate_trajectory()`: Integrate trajectory using specified method.
- `_integrate_scipy()`: Integrate using SciPy solve_ivp.
- `_integrate_custom()`: Integrate using custom methods.
- `_integrate_rk4()`: Fourth-order Runge-Kutta integration.
- `_integrate_verlet()`: Verlet integration (for position/velocity problems).

#### `EarthMoonNBodyPropagator`
Complete Earth-Moon n-body propagator with multiple body effects.

**Methods:**
- `__init__()`: Initialize Earth-Moon n-body propagator.
- `propagate_spacecraft()`: Propagate spacecraft trajectory in Earth-Moon system.
- `_nbody_dynamics()`: Compute n-body dynamics for Earth-Moon-Sun system.
- `_calculate_perturbations()`: Calculate additional perturbations (simplified).
- `compare_with_twobody()`: Compare n-body propagation with two-body propagation.
- `_propagate_twobody()`: Propagate using two-body dynamics for comparison.

#### `TrajectoryIO`
Trajectory serialization and I/O utilities.

**Methods:**
- `__init__()`: Initialize trajectory I/O manager.
- `save_trajectory()`: Save trajectory to file.
- `load_trajectory()`: Load trajectory from file.
- `save_propagation_result()`: Save propagation result to file.
- `load_propagation_result()`: Load propagation result from file.
- `list_trajectories()`: List all saved trajectories.
- `_trajectory_to_dict()`: Convert trajectory object to dictionary.
- `_dict_to_trajectory()`: Convert dictionary to trajectory object.
- `_save_json()`: Save data to JSON file.
- `_load_json()`: Load data from JSON file.
- `_save_pickle()`: Save data to pickle file.
- `_load_pickle()`: Load data from pickle file.
- `_save_npz()`: Save numerical data to NPZ file.
- `_load_npz()`: Load data from NPZ file.

#### `TrajectoryComparison`
Utility class for comparing and analyzing trajectories.

**Methods:**
- `__init__()`: Initialize trajectory comparison utility.
- `compare_trajectories()`: Compare two trajectory solutions.
- `analyze_accuracy()`: Analyze accuracy of computed trajectory vs reference.

### Functions

#### `create_nbody_propagator(include_sun, include_perturbations)`
Create Earth-Moon n-body propagator.

#### `create_trajectory_io(base_directory)`
Create trajectory I/O manager.

---

## trajectory/orbit_state.py

**Description**: Orbital state representation and calculations.

- **Lines of code**: 325
- **Classes**: 1
- **Functions**: 0

### Classes

#### `OrbitState`
Represents the state of an orbiting body.

**Methods:**
- `__post_init__()`: Validate orbital elements after initialization.
- `position()`: Calculate the position vector in the inertial frame.
- `velocity()`: Calculate the velocity vector in the inertial frame.
- `_get_rotation_matrix()`: Get the rotation matrix from perifocal to inertial frame.
- `from_state_vectors()`: Create an OrbitState from position and velocity vectors.
- `to_pykep()`: Convert to PyKEP planet object.
- `get_state_vectors()`: Get position and velocity vectors.
- `get_state_vectors_km()`: Get position and velocity vectors in km and km/s.

---

## trajectory/phase_optimization.py

**Description**: Phase angle optimization for lunar transfer trajectories.

- **Lines of code**: 306
- **Classes**: 0
- **Functions**: 4

### Functions

#### `calculate_initial_position(r_park, phase, moon_h_unit)`
Calculate initial position vector for given phase angle.

#### `evaluate_transfer_solution(r1, moon_pos, moon_vel, transfer_time, orbit_radius, max_revs)`
Evaluate a transfer solution for given initial conditions.

#### `find_optimal_phase(r_park, moon_pos, moon_vel, transfer_time, orbit_radius, max_revs, num_samples)`
Find the optimal phase angle for lunar transfer departure.

#### `_rotation_matrix(axis, angle)`
Create rotation matrix for rotating around axis by angle.

---

## trajectory/propagator.py

**Description**: Trajectory propagation module.

- **Lines of code**: 183
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TrajectoryPropagator`
Handles trajectory propagation with gravity effects.

**Methods:**
- `__init__()`: Initialize propagator.
- `propagate_to_target()`: Propagate spacecraft trajectory to target using high-precision integration.
- `_calculate_energy()`: Calculate the specific energy of a spacecraft at a given position and velocity.

---

## trajectory/target_state.py

**Description**: Target state calculations for lunar transfer trajectories.

- **Lines of code**: 155
- **Classes**: 0
- **Functions**: 2

### Functions

#### `calculate_target_state(moon_pos, moon_vel, orbit_radius)`
Calculate target state for lunar orbit insertion.

#### `_rotation_matrix(axis, angle)`
Create rotation matrix for rotating around axis by angle.

---

## trajectory/trajectory_base.py

**Description**: Base trajectory class and core trajectory functionality.

- **Lines of code**: 225
- **Classes**: 1
- **Functions**: 0

### Classes

#### `Trajectory`
Base class for orbital trajectories.

**Methods:**
- `__post_init__()`: Initialize derived attributes and validate inputs.
- `validate_trajectory()`: Validate the complete trajectory including maneuvers.
- `propagate_to()`: Propagate the trajectory to a specific epoch.
- `_propagate_segment()`: Propagate a single trajectory segment without maneuvers.
- `get_state_at()`: Get the orbital state at a specific epoch.
- `add_maneuver()`: Add a maneuver to the trajectory.
- `get_total_delta_v()`: Calculate total delta-v cost of all maneuvers.
- `__str__()`: String representation of the trajectory.

---

## trajectory/trajectory_optimization.py

**Description**: Trajectory optimization module for Task 3 completion.

- **Lines of code**: 453
- **Classes**: 1
- **Functions**: 2

### Classes

#### `TrajectoryOptimizer`
Advanced trajectory optimization for Earth-Moon transfers.

**Methods:**
- `__init__()`: Initialize trajectory optimizer.
- `optimize_single_objective()`: Optimize trajectory for a single objective.
- `pareto_front_analysis()`: Generate Pareto front for multi-objective optimization.
- `optimize_with_constraints()`: Optimize trajectory with constraints.
- `_find_pareto_front()`: Find Pareto-optimal solutions from a set of solutions.
- `_check_constraints()`: Check if solution satisfies constraints.

### Functions

#### `optimize_trajectory_parameters(epoch, optimization_type)`
Convenience function for trajectory optimization.

#### `batch_trajectory_optimization(epochs, objective, max_workers)`
Perform batch optimization for multiple epochs.

---

## trajectory/trajectory_physics.py

**Description**: Legacy trajectory physics module - DEPRECATED.

- **Lines of code**: 50
- **Classes**: 0
- **Functions**: 0

---

## trajectory/trajectory_validator.py

**Description**: Consolidated trajectory validation module.

- **Lines of code**: 427
- **Classes**: 5
- **Functions**: 5

### Classes

#### `TrajectoryValidator`
Validates trajectory parameters and constraints.

**Methods:**
- `__init__()`: Initialize validator with constraints.
- `validate_inputs()`: Validate input parameters for trajectory generation.
- `validate_delta_v()`: Validate delta-v values against typical mission constraints.
- `validate_epoch()`: Validate epoch is within supported range.
- `validate_orbit_altitude()`: Validate orbit altitude is within reasonable range.

#### `_FallbackPK`
No docstring

#### `_FallbackTransferDefaults`
No docstring

#### `_FallbackEphemerisLimits`
No docstring

#### `_FallbackOrbitState`
No docstring

**Methods:**
- `__init__()`: No docstring

### Functions

#### `validate_epoch(dt, allow_none)`
Validate epoch is within supported range.

#### `validate_orbit_altitude(altitude, min_alt, max_alt)`
Validate orbit altitude is within reasonable range.

#### `validate_transfer_parameters(tof_days, max_revs, min_dv, max_dv)`
Validate transfer trajectory parameters.

#### `validate_initial_orbit(orbit)`
Validate initial orbit specification.

#### `validate_final_orbit(final_radius, initial_radius)`
Validate final orbit radius.

---

## trajectory/transfer_parameters.py

**Description**: Transfer parameters for trajectory generation.

- **Lines of code**: 97
- **Classes**: 1
- **Functions**: 0

### Classes

#### `TransferParameters`
Parameters for generating transfer trajectories.

**Methods:**
- `__post_init__()`: Validate parameters after initialization.
- `validate()`: Validate all parameters.
- `get_initial_state()`: Convert initial orbit specification to OrbitState.

---

## trajectory/transfer_window_analysis.py

**Description**: Transfer Window Analysis Module for Task 3 completion.

- **Lines of code**: 354
- **Classes**: 2
- **Functions**: 2

### Classes

#### `TransferWindow`
Represents a transfer window opportunity.

**Methods:**
- `__init__()`: No docstring
- `__str__()`: No docstring

#### `TrajectoryWindowAnalyzer`
Analyzes Earth-Moon transfer windows for Task 3 implementation.

**Methods:**
- `__init__()`: Initialize the trajectory window analyzer.
- `find_transfer_windows()`: Find optimal transfer windows in a given time period.
- `optimize_launch_window()`: Optimize launch window around a target date.
- `analyze_trajectory_sensitivity()`: Analyze trajectory sensitivity to parameter variations.
- `_datetime_to_pykep_epoch()`: Convert datetime to PyKEP epoch (days since J2000).
- `_calculate_c3_energy()`: Calculate characteristic energy (C3) for the transfer.

### Functions

#### `generate_multiple_transfer_options(start_date, end_date, max_options)`
Generate multiple transfer options for Task 3 completion.

#### `analyze_launch_opportunities(target_year)`
Analyze launch opportunities for a given year.

---

## trajectory/validator.py

**Description**: Backward compatibility module for trajectory validation.

- **Lines of code**: 22
- **Classes**: 0
- **Functions**: 0

---

## trajectory/validators.py

**Description**: Backward compatibility module for trajectory validation functions.

- **Lines of code**: 38
- **Classes**: 0
- **Functions**: 0

---

# Module: trajectory/validation

## trajectory/validation/constraint_validation.py

**Description**: Constraint validation functions for trajectory calculations.

- **Lines of code**: 132
- **Classes**: 0
- **Functions**: 1

### Functions

#### `validate_trajectory_constraints(r1, v1, r2, v2, tof)`
Validate physical constraints for the complete trajectory.

---

## trajectory/validation/physics_validation.py

**Description**: Physics validation functions for trajectory calculations.

- **Lines of code**: 246
- **Classes**: 0
- **Functions**: 4

### Functions

#### `validate_basic_orbital_mechanics(r, v, mu)`
Validate basic orbital mechanics relationships for a state vector.

#### `validate_transfer_time(r1, r2, tof, mu)`
Validate if transfer time is physically reasonable.

#### `validate_solution_physics(r1, v1, r2, v2, transfer_time)`
Validate the physics of a transfer solution.

#### `calculate_circular_velocity(radius, mu)`
Calculate circular orbit velocity.

---

## trajectory/validation/vector_validation.py

**Description**: Vector validation functions for trajectory calculations.

- **Lines of code**: 190
- **Classes**: 0
- **Functions**: 4

### Functions

#### `validate_vector_units(vector, name, expected_magnitude_range, unit)`
Validate that a vector's magnitude falls within expected range and has correct units.

#### `validate_delta_v(delta_v, max_delta_v)`
Validate delta-v vector for reasonableness.

#### `validate_state_vector(position, velocity)`
Validate state vector components.

#### `propagate_orbit(position, velocity, dt)`
Simple two-body orbit propagation.

---

# Module: utils

## utils/unit_conversions.py

**Description**: Unit conversion utilities for trajectory calculations.

- **Lines of code**: 313
- **Classes**: 0
- **Functions**: 16

### Functions

#### `ensure_array(x)`
Convert input to numpy array if it isn't already.

#### `restore_type(x, arr)`
Restore original type of input after array operations.

#### `datetime_to_mjd2000(dt)`
Convert datetime to Modified Julian Date 2000 (MJD2000).

#### `datetime_to_j2000(dt)`
Convert datetime to days since J2000 epoch.

#### `datetime_to_pykep_epoch(dt)`
Convert datetime to PyKEP epoch (MJD2000).

#### `km_to_m(km)`
Convert kilometers to meters.

#### `m_to_km(m)`
Convert meters to kilometers.

#### `kmps_to_mps(kmps)`
Convert kilometers per second to meters per second.

#### `mps_to_kmps(mps)`
Convert meters per second to kilometers per second.

#### `deg_to_rad(deg)`
Convert degrees to radians.

#### `rad_to_deg(rad)`
Convert radians to degrees.

#### `km3s2_to_m3s2(mu)`
Convert gravitational parameter from km³/s² to m³/s².

#### `m3s2_to_km3s2(mu)`
Convert gravitational parameter from m³/s² to km³/s².

#### `days_to_seconds(days)`
Convert days to seconds.

#### `seconds_to_days(seconds)`
Convert seconds to days.

#### `pykep_epoch_to_datetime(epoch)`
Convert PyKEP epoch (MJD2000) to datetime.

---

# Module: visualization

## visualization/dashboard.py

**Description**: Comprehensive Mission Analysis Dashboard Module.

- **Lines of code**: 963
- **Classes**: 3
- **Functions**: 1

### Classes

#### `DashboardTheme`
Theme configuration for comprehensive dashboard.

#### `MissionAnalysisData`
Complete mission analysis data container.

**Methods:**
- `__post_init__()`: No docstring

#### `ComprehensiveDashboard`
Comprehensive mission analysis dashboard combining all visualization modules.

**Methods:**
- `__init__()`: Initialize comprehensive dashboard.
- `create_executive_dashboard()`: Create executive summary dashboard with key metrics and insights.
- `create_technical_dashboard()`: Create technical analysis dashboard with detailed engineering data.
- `create_comparison_dashboard()`: Create scenario comparison dashboard.
- `create_interactive_explorer()`: Create interactive mission explorer with drill-down capabilities.
- `_add_mission_overview()`: Add mission overview table.
- `_add_financial_kpis()`: Add financial KPI indicators.
- `_add_trajectory_summary()`: Add trajectory analysis summary.
- `_add_optimization_summary()`: Add optimization results summary.
- `_add_cost_summary()`: Add cost breakdown pie chart.
- `_add_timeline_status()`: Add timeline status chart.
- `_add_risk_summary()`: Add risk assessment summary.
- `_add_performance_indicators()`: Add performance indicators.
- `_add_decision_support()`: Add decision support recommendations.
- `_add_3d_trajectory_placeholder()`: Placeholder for 3D trajectory visualization.
- `_add_pareto_front()`: Add Pareto front visualization.
- `_add_sensitivity_analysis()`: Add sensitivity analysis visualization.
- `_add_critical_path()`: Add critical path visualization.
- `_extract_comparison_data()`: Extract data for scenario comparison.
- `_add_financial_comparison()`: Add financial comparison chart.
- `_add_performance_comparison()`: Add performance comparison chart.
- `_add_risk_comparison()`: Add risk comparison chart.
- `_add_cost_comparison()`: Add cost comparison chart.
- `_add_timeline_comparison()`: Add timeline comparison chart.
- `_add_tradeoff_analysis()`: Add trade-off analysis chart.
- `_add_parameter_explorer()`: Add parameter explorer.
- `_add_interactive_timeline()`: Add interactive timeline.
- `_add_cost_benefit_explorer()`: Add cost-benefit explorer.
- `_add_sensitivity_explorer()`: Add sensitivity explorer.
- `_create_empty_plot()`: Create empty plot with message.

### Functions

#### `create_sample_dashboard()`
Create a sample comprehensive dashboard for demonstration.

---

## visualization/economic_visualization.py

**Description**: Economic Analysis Visualization Module.

- **Lines of code**: 1117
- **Classes**: 2
- **Functions**: 1

### Classes

#### `DashboardConfig`
Configuration for economic visualization dashboards.

#### `EconomicVisualizer`
Comprehensive economic analysis visualization using Plotly.

**Methods:**
- `__init__()`: Initialize economic visualizer.
- `create_scenario_comparison()`: Create scenario comparison visualization.
- `create_financial_dashboard()`: Create comprehensive financial dashboard.
- `create_cost_analysis_dashboard()`: Create detailed cost analysis dashboard.
- `create_isru_analysis_dashboard()`: Create ISRU economic analysis dashboard.
- `create_sensitivity_analysis_dashboard()`: Create sensitivity and risk analysis dashboard.
- `_add_financial_indicators()`: Add financial KPI indicators.
- `_add_investment_revenue_chart()`: Add investment vs revenue bar chart.
- `_add_cash_flow_timeline()`: Add cash flow timeline chart.
- `_add_cost_breakdown_pie()`: Add cost breakdown pie chart.
- `_add_roi_analysis()`: Add ROI analysis chart.
- `_add_risk_assessment()`: Add risk assessment scatter plot.
- `_add_placeholder_chart()`: Add placeholder for missing data.

### Functions

#### `create_quick_financial_dashboard(npv, irr, roi, payback_years, total_investment, total_revenue)`
Quick function to create a simple financial dashboard.

---

## visualization/integrated_dashboard.py

**Description**: Integrated Dashboard for Mission Analysis

- **Lines of code**: 298
- **Classes**: 0
- **Functions**: 3

### Functions

#### `create_mission_dashboard(trajectory_results, economic_results, title, show_3d, show_economics)`
Create integrated mission analysis dashboard.

#### `create_optimization_dashboard(optimization_results, title)`
Create optimization results dashboard.

#### `create_comparison_dashboard(scenarios, title)`
Create multi-scenario comparison dashboard.

---

## visualization/mission_visualization.py

**Description**: Mission Timeline and Milestone Visualization Module.

- **Lines of code**: 870
- **Classes**: 4
- **Functions**: 1

### Classes

#### `TimelineConfig`
Configuration for mission timeline visualization.

#### `MissionPhase`
Represents a mission phase or task.

**Methods:**
- `__post_init__()`: No docstring

#### `MissionMilestone`
Represents a mission milestone.

**Methods:**
- `__post_init__()`: No docstring

#### `MissionVisualizer`
Mission timeline and milestone visualization using Plotly.

**Methods:**
- `__init__()`: Initialize mission visualizer.
- `create_mission_timeline()`: Create comprehensive mission timeline visualization.
- `create_resource_utilization_chart()`: Create resource utilization visualization.
- `create_critical_path_analysis()`: Create critical path analysis visualization.
- `create_mission_dashboard()`: Create comprehensive mission dashboard.
- `create_risk_timeline()`: Create risk assessment timeline.
- `_add_gantt_phases()`: Add mission phases as Gantt chart bars.
- `_add_milestones()`: Add milestones to timeline.
- `_add_dependencies()`: Add dependency arrows between phases.
- `_add_dashboard_timeline()`: Add simplified timeline for dashboard.
- `_add_phase_status_chart()`: Add phase status bar chart.
- `_add_upcoming_milestones_table()`: Add upcoming milestones table.
- `_calculate_critical_path()`: Calculate critical path (simplified implementation).
- `_add_dependency_arrows()`: Add dependency arrows for critical path visualization.
- `_get_phase_color()`: Get color for phase category.
- `_configure_timeline_layout()`: Configure timeline layout.
- `_create_empty_plot()`: Create empty plot with message.

### Functions

#### `create_sample_mission_timeline()`
Create a sample lunar mission timeline for demonstration.

---

## visualization/optimization_visualization.py

**Description**: Optimization Results Visualization Module.

- **Lines of code**: 1138
- **Classes**: 2
- **Functions**: 1

### Classes

#### `ParetoPlotConfig`
Configuration for optimization visualization plots.

**Methods:**
- `__post_init__()`: No docstring

#### `OptimizationVisualizer`
Interactive optimization results visualization using Plotly.

**Methods:**
- `__init__()`: Initialize optimization visualizer.
- `create_pareto_front_plot()`: Create comprehensive Pareto front visualization.
- `create_optimization_convergence_plot()`: Create optimization convergence visualization.
- `create_solution_comparison_plot()`: Create detailed solution comparison visualization.
- `_prepare_convergence_data()`: Prepare data for convergence analysis plots.
- `_extract_convergence_metrics()`: Extract convergence metrics from generation history.
- `_create_convergence_subplot_layout()`: Create subplot layout for convergence analysis.
- `_add_best_solutions_evolution()`: Add best solutions evolution subplot.
- `_add_hypervolume_convergence()`: Add hypervolume convergence subplot.
- `_add_solution_count_plot()`: Add solution count per generation subplot.
- `_add_objective_space_coverage()`: Add objective space coverage subplot if applicable.
- `_update_convergence_layout()`: Update axes labels and overall layout for convergence plot.
- `_prepare_comparison_data()`: Prepare data for solution comparison plots.
- `_create_comparison_subplot_layout()`: Create subplot layout for solution comparison.
- `_add_objective_comparison()`: Add objective values comparison subplot.
- `_add_parameter_comparison()`: Add parameter values comparison subplot.
- `_add_ranking_table()`: Add solution ranking table subplot.
- `_build_ranking_data()`: Build ranking data for table display.
- `_build_table_headers()`: Build table header values.
- `_add_tradeoff_analysis()`: Add trade-off analysis subplot if applicable.
- `create_preference_analysis_plot()`: Create preference-based solution ranking visualization.
- `_setup_preference_data()`: Setup initial data for preference analysis.
- `_add_preference_ranking_plot()`: Add preference ranking bar plot.
- `_add_weighted_objectives_plot()`: Add weighted objectives scatter plot.
- `_calculate_weighted_values()`: Calculate weighted values for an objective.
- `_extract_objective_value()`: Extract objective value handling both dict and list formats.
- `_add_sensitivity_plot()`: Add sensitivity analysis heatmap.
- `_add_top_solutions_plot()`: Add top solutions comparison plot.
- `_create_2d_pareto_plot()`: Create 2D Pareto front plot.
- `_create_3d_pareto_plot()`: Create 3D Pareto front plot.
- `_create_parallel_coordinates_plot()`: Create parallel coordinates plot for >3 objectives.
- `_create_empty_plot()`: Create empty plot with message.
- `_calculate_hypervolume()`: Calculate hypervolume (simplified implementation).
- `_calculate_preference_sensitivity()`: Calculate sensitivity of rankings to weight changes.

### Functions

#### `create_quick_pareto_plot(optimization_result, objective_names)`
Quick function to create a simple Pareto front plot.

---

## visualization/trajectory_visualization.py

**Description**: Interactive 3D Trajectory Visualization Module.

- **Lines of code**: 662
- **Classes**: 2
- **Functions**: 1

### Classes

#### `TrajectoryPlotConfig`
Configuration for trajectory visualization plots.

#### `TrajectoryVisualizer`
Interactive 3D trajectory visualization using Plotly.

**Methods:**
- `__init__()`: Initialize trajectory visualizer.
- `create_3d_trajectory_plot()`: Create comprehensive 3D trajectory visualization.
- `create_transfer_window_plot()`: Create transfer window opportunity visualization.
- `create_orbital_elements_plot()`: Create orbital elements evolution visualization.
- `create_trajectory_comparison()`: Create comparative visualization of multiple trajectories.
- `_add_celestial_bodies()`: Add Earth and Moon to 3D plot.
- `_add_trajectory_path()`: Add trajectory path to 3D plot.
- `_configure_3d_layout()`: Configure 3D plot layout.
- `_calculate_orbital_elements_evolution()`: Calculate orbital elements evolution from position/velocity data.

### Functions

#### `create_quick_trajectory_plot(earth_orbit_alt, moon_orbit_alt, transfer_time, departure_epoch)`
Quick function to create a simple trajectory plot.

---
