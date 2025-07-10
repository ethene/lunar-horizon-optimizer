"""Economic sensitivity analysis module for Task 5 completion.

This module provides comprehensive sensitivity and scenario analysis for
lunar mission economics, including Monte Carlo simulation and tornado diagrams.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)


class EconomicSensitivityAnalyzer:
    """Economic sensitivity and scenario analysis for lunar missions.

    This class provides comprehensive sensitivity analysis including one-way
    sensitivity, scenario analysis, and Monte Carlo simulation.
    """

    def __init__(
        self, base_model_function: Callable[..., dict[str, float]] | None = None
    ) -> None:
        """Initialize economic sensitivity analyzer.

        Args:
            base_model_function: Function that calculates economic metrics given parameters
        """
        self.base_model_function = base_model_function
        self.sensitivity_results: dict[str, Any] = {}

        logger.info("Initialized EconomicSensitivityAnalyzer")

    def one_way_sensitivity(
        self,
        base_parameters: dict[str, float],
        variable_ranges: dict[str, tuple[float, float]],
        num_points: int = 20,
    ) -> dict[str, Any]:
        """Perform one-way sensitivity analysis.

        Args:
            base_parameters: Base case parameter values
            variable_ranges: Dictionary of {parameter: (min_value, max_value)}
            num_points: Number of points to evaluate for each variable

        Returns
        -------
            One-way sensitivity analysis results
        """
        logger.info(
            f"Performing one-way sensitivity analysis for {len(variable_ranges)} variables"
        )

        if not self.base_model_function:
            msg = "Model function not provided"
            raise ValueError(msg)

        # Calculate base case
        base_result = self.base_model_function(base_parameters)
        base_npv = (
            base_result.get("npv", 0) if isinstance(base_result, dict) else base_result
        )

        sensitivity_results: dict[str, Any] = {
            "base_npv": base_npv,
            "base_parameters": base_parameters,
            "variables": {},
        }

        for variable, (min_val, max_val) in variable_ranges.items():
            # Create parameter sweep
            values = np.linspace(min_val, max_val, num_points)
            npv_values = []

            for value in values:
                # Modify parameters
                modified_params = base_parameters.copy()
                modified_params[variable] = value

                # Calculate result
                result = self.base_model_function(modified_params)
                npv = result.get("npv", 0) if isinstance(result, dict) else result
                npv_values.append(npv)

            # Calculate sensitivity metrics
            npv_range = max(npv_values) - min(npv_values)
            param_range = max_val - min_val
            sensitivity = npv_range / base_npv if base_npv != 0 else 0

            # Calculate elasticity at base point
            base_value = base_parameters[variable]
            elasticity = self._calculate_elasticity(
                values,
                npv_values,
                base_value,
                base_npv,
            )

            sensitivity_results["variables"][variable] = {
                "parameter_values": values.tolist(),
                "npv_values": npv_values,
                "sensitivity": sensitivity,
                "elasticity": elasticity,
                "npv_range": npv_range,
                "parameter_range": param_range,
            }

        # Rank variables by sensitivity
        ranked_variables = sorted(
            sensitivity_results["variables"].items(),
            key=lambda x: abs(x[1]["sensitivity"]),
            reverse=True,
        )

        sensitivity_results["ranking"] = [var for var, _ in ranked_variables]

        self.sensitivity_results["one_way"] = sensitivity_results
        logger.info(
            f"One-way sensitivity analysis complete. Most sensitive: {ranked_variables[0][0]}"
        )

        return sensitivity_results

    def tornado_diagram_data(
        self,
        base_parameters: dict[str, float],
        variable_ranges: dict[str, tuple[float, float]],
    ) -> dict[str, Any]:
        """Generate data for tornado diagram visualization.

        Args:
            base_parameters: Base case parameters
            variable_ranges: Variable ranges for analysis

        Returns
        -------
            Tornado diagram data
        """
        logger.info("Generating tornado diagram data")

        if not self.base_model_function:
            msg = "Model function not provided"
            raise ValueError(msg)

        base_result = self.base_model_function(base_parameters)
        base_npv = (
            base_result.get("npv", 0) if isinstance(base_result, dict) else base_result
        )

        tornado_data: dict[str, Any] = {
            "base_npv": base_npv,
            "variables": {},
        }

        for variable, (min_val, max_val) in variable_ranges.items():
            # Calculate low case
            low_params = base_parameters.copy()
            low_params[variable] = min_val
            low_result = self.base_model_function(low_params)
            low_npv = (
                low_result.get("npv", 0) if isinstance(low_result, dict) else low_result
            )

            # Calculate high case
            high_params = base_parameters.copy()
            high_params[variable] = max_val
            high_result = self.base_model_function(high_params)
            high_npv = (
                high_result.get("npv", 0)
                if isinstance(high_result, dict)
                else high_result
            )

            # Calculate impacts
            low_impact = low_npv - base_npv
            high_impact = high_npv - base_npv
            total_swing = abs(high_impact - low_impact)

            tornado_data["variables"][variable] = {
                "low_value": min_val,
                "high_value": max_val,
                "low_npv": low_npv,
                "high_npv": high_npv,
                "low_impact": low_impact,
                "high_impact": high_impact,
                "total_swing": total_swing,
            }

        # Sort by total swing (tornado ordering)
        sorted_variables = sorted(
            tornado_data["variables"].items(),
            key=lambda x: x[1]["total_swing"],
            reverse=True,
        )

        tornado_data["sorted_variables"] = sorted_variables

        return tornado_data

    def scenario_analysis(
        self, scenarios: dict[str, dict[str, float]], base_parameters: dict[str, float]
    ) -> dict[str, Any]:
        """Perform scenario analysis with multiple defined scenarios.

        Args:
            scenarios: Dictionary of scenario definitions
            base_parameters: Base case parameters

        Returns
        -------
            Scenario analysis results
        """
        logger.info(f"Performing scenario analysis for {len(scenarios)} scenarios")

        if not self.base_model_function:
            msg = "Model function not provided"
            raise ValueError(msg)

        # Calculate base case
        base_result = self.base_model_function(base_parameters)
        base_npv = (
            base_result.get("npv", 0) if isinstance(base_result, dict) else base_result
        )

        scenario_results: dict[str, Any] = {
            "base_case": {
                "parameters": base_parameters,
                "result": base_result,
                "npv": base_npv,
            },
            "scenarios": {},
        }

        for scenario_name, scenario_params in scenarios.items():
            # Merge scenario parameters with base parameters
            merged_params = base_parameters.copy()
            merged_params.update(scenario_params)

            # Calculate scenario result
            scenario_result = self.base_model_function(merged_params)
            scenario_npv = (
                scenario_result.get("npv", 0)
                if isinstance(scenario_result, dict)
                else scenario_result
            )

            # Calculate differences from base case
            npv_change = scenario_npv - base_npv
            npv_change_percent = (npv_change / base_npv * 100) if base_npv != 0 else 0

            scenario_results["scenarios"][scenario_name] = {
                "parameters": merged_params,
                "result": scenario_result,
                "npv": scenario_npv,
                "npv_change": npv_change,
                "npv_change_percent": npv_change_percent,
            }

        # Summary statistics
        scenario_npvs = [s["npv"] for s in scenario_results["scenarios"].values()]
        scenario_results["summary"] = {
            "min_npv": min(scenario_npvs),
            "max_npv": max(scenario_npvs),
            "mean_npv": np.mean(scenario_npvs),
            "std_npv": np.std(scenario_npvs),
            "best_scenario": max(
                scenario_results["scenarios"].items(), key=lambda x: x[1]["npv"]
            )[0],
            "worst_scenario": min(
                scenario_results["scenarios"].items(), key=lambda x: x[1]["npv"]
            )[0],
        }

        logger.info(
            f"Scenario analysis complete. Range: ${min(scenario_npvs)/1e6:.1f}M to ${max(scenario_npvs)/1e6:.1f}M"
        )

        return scenario_results

    def monte_carlo_simulation(
        self,
        base_parameters: dict[str, float],
        variable_distributions: dict[str, dict[str, Any]],
        num_simulations: int = 10000,
        confidence_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Perform Monte Carlo simulation for risk analysis.

        Args:
            base_parameters: Base case parameters
            variable_distributions: Parameter distributions {'param': {'type': 'normal', 'mean': x, 'std': y}}
            num_simulations: Number of Monte Carlo simulations
            confidence_levels: Confidence levels for analysis (default: [0.05, 0.1, 0.5, 0.9, 0.95])

        Returns
        -------
            Monte Carlo simulation results
        """
        if confidence_levels is None:
            confidence_levels = [0.05, 0.1, 0.5, 0.9, 0.95]

        logger.info(
            f"Performing Monte Carlo simulation with {num_simulations} iterations"
        )

        if not self.base_model_function:
            msg = "Model function not provided"
            raise ValueError(msg)

        # Generate random samples
        samples = self._generate_monte_carlo_samples(
            variable_distributions,
            num_simulations,
        )

        # Run simulations
        results = []
        valid_results = 0

        for i in range(num_simulations):
            # Create parameter set for this simulation
            sim_params = base_parameters.copy()
            for param_name, sample_values in samples.items():
                sim_params[param_name] = sample_values[i]

            try:
                # Calculate result
                result = self.base_model_function(sim_params)
                npv = result.get("npv", 0) if isinstance(result, dict) else result
                results.append(npv)
                valid_results += 1
            except Exception as e:
                logger.warning(f"Simulation {i} failed: {e}")
                results.append(np.nan)

        # Remove invalid results
        valid_results_array = np.array([r for r in results if not np.isnan(r)])

        # Calculate statistics
        percentiles = np.percentile(
            valid_results_array, [p * 100 for p in confidence_levels]
        )

        mc_results = {
            "num_simulations": num_simulations,
            "valid_simulations": valid_results,
            "results": valid_results_array.tolist(),
            "statistics": {
                "mean": np.mean(valid_results_array),
                "median": np.median(valid_results_array),
                "std": np.std(valid_results_array),
                "min": np.min(valid_results_array),
                "max": np.max(valid_results_array),
                "skewness": self._calculate_skewness(valid_results_array),
                "kurtosis": self._calculate_kurtosis(valid_results_array),
            },
            "percentiles": {
                f"p{int(cl * 100)}": percentiles[i]
                for i, cl in enumerate(confidence_levels)
            },
            "risk_metrics": {
                "probability_positive_npv": np.mean(valid_results_array > 0),
                "value_at_risk_5%": percentiles[0],  # 5th percentile
                "expected_shortfall_5%": np.mean(
                    valid_results_array[valid_results_array <= percentiles[0]]
                ),
                "coefficient_of_variation": (
                    np.std(valid_results_array) / np.mean(valid_results_array)
                    if np.mean(valid_results_array) != 0
                    else 0
                ),
            },
        }

        # Correlation analysis
        if len(variable_distributions) > 1:
            mc_results["correlation_analysis"] = self._analyze_parameter_correlations(
                samples,
                valid_results_array,
            )

        logger.info(
            f"Monte Carlo simulation complete. Mean NPV: ${mc_results['statistics']['mean']/1e6:.1f}M, "
            f"P(NPV > 0) = {mc_results['risk_metrics']['probability_positive_npv']:.1%}"
        )

        return mc_results

    def _calculate_elasticity(
        self,
        param_values: np.ndarray[np.float64, np.dtype[np.float64]],
        npv_values: list[float],
        base_param: float,
        base_npv: float,
    ) -> float:
        """Calculate elasticity at base point."""
        # Find closest points to base value
        idx = np.argmin(np.abs(param_values - base_param))

        if idx == 0:
            # Use forward difference
            delta_npv = npv_values[idx + 1] - npv_values[idx]
            delta_param = param_values[idx + 1] - param_values[idx]
        elif idx == len(param_values) - 1:
            # Use backward difference
            delta_npv = npv_values[idx] - npv_values[idx - 1]
            delta_param = param_values[idx] - param_values[idx - 1]
        else:
            # Use central difference
            delta_npv = npv_values[idx + 1] - npv_values[idx - 1]
            delta_param = param_values[idx + 1] - param_values[idx - 1]

        if delta_param != 0 and base_npv != 0 and base_param != 0:
            elasticity = (delta_npv / base_npv) / (delta_param / base_param)
        else:
            elasticity = 0.0

        return float(elasticity)

    def _generate_monte_carlo_samples(
        self, distributions: dict[str, dict[str, Any]], num_samples: int
    ) -> dict[str, np.ndarray[np.float64, np.dtype[np.float64]]]:
        """Generate Monte Carlo samples from specified distributions."""
        samples = {}

        for param_name, dist_spec in distributions.items():
            dist_type = dist_spec["type"].lower()

            if dist_type == "normal":
                samples[param_name] = np.random.normal(
                    dist_spec["mean"],
                    dist_spec["std"],
                    num_samples,
                )
            elif dist_type == "uniform":
                samples[param_name] = np.random.uniform(
                    dist_spec["min"],
                    dist_spec["max"],
                    num_samples,
                )
            elif dist_type == "triang":
                # Use scipy.stats.triang for triangular distribution
                c = (dist_spec["mode"] - dist_spec["min"]) / (
                    dist_spec["max"] - dist_spec["min"]
                )
                samples[param_name] = stats.triang.rvs(
                    c=c,
                    loc=dist_spec["min"],
                    scale=dist_spec["max"] - dist_spec["min"],
                    size=num_samples,
                )
            elif dist_type == "lognormal":
                samples[param_name] = np.random.lognormal(
                    dist_spec["mean"],
                    dist_spec["sigma"],
                    num_samples,
                )
            else:
                msg = f"Unsupported distribution type: {dist_type}"
                raise ValueError(msg)

        return samples

    def _calculate_skewness(
        self, data: np.ndarray[np.float64, np.dtype[np.float64]]
    ) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _calculate_kurtosis(
        self, data: np.ndarray[np.float64, np.dtype[np.float64]]
    ) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)  # Excess kurtosis

    def _analyze_parameter_correlations(
        self,
        samples: dict[str, np.ndarray[np.float64, np.dtype[np.float64]]],
        results: np.ndarray[np.float64, np.dtype[np.float64]],
    ) -> dict[str, Any]:
        """Analyze correlations between parameters and results."""
        correlations = {}

        for param_name, param_values in samples.items():
            correlation = np.corrcoef(param_values, results)[0, 1]
            correlations[param_name] = correlation

        # Rank by absolute correlation
        ranked_correlations = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "correlations": correlations,
            "ranked_by_abs_correlation": ranked_correlations,
            "strongest_positive_correlation": max(
                correlations.items(), key=lambda x: x[1]
            ),
            "strongest_negative_correlation": min(
                correlations.items(), key=lambda x: x[1]
            ),
        }

    def comprehensive_sensitivity_report(
        self,
        base_parameters: dict[str, float],
        variable_ranges: dict[str, tuple[float, float]],
        scenarios: dict[str, dict[str, float]] | None = None,
        distributions: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive sensitivity analysis report.

        Args:
            base_parameters: Base case parameters
            variable_ranges: Parameter ranges for sensitivity analysis
            scenarios: Predefined scenarios (optional)
            distributions: Parameter distributions for Monte Carlo (optional)

        Returns
        -------
            Comprehensive sensitivity analysis report
        """
        logger.info("Generating comprehensive sensitivity analysis report")

        report = {
            "base_parameters": base_parameters,
            "analysis_timestamp": np.datetime64("now").item().isoformat(),
        }

        # One-way sensitivity analysis
        report["one_way_sensitivity"] = self.one_way_sensitivity(
            base_parameters,
            variable_ranges,
        )

        # Tornado diagram
        report["tornado_diagram"] = self.tornado_diagram_data(
            base_parameters,
            variable_ranges,
        )

        # Scenario analysis (if scenarios provided)
        if scenarios:
            report["scenario_analysis"] = self.scenario_analysis(
                scenarios,
                base_parameters,
            )

        # Monte Carlo simulation (if distributions provided)
        if distributions:
            report["monte_carlo"] = self.monte_carlo_simulation(
                base_parameters,
                distributions,
            )

        # Summary and recommendations
        report["summary"] = self._generate_sensitivity_summary(report)

        logger.info("Comprehensive sensitivity analysis report completed")
        return report

    def _generate_sensitivity_summary(self, report: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of sensitivity analysis results."""
        summary: dict[str, Any] = {
            "key_findings": [],
            "risk_assessment": {},
            "recommendations": [],
        }

        # One-way sensitivity findings
        if "one_way_sensitivity" in report:
            most_sensitive = report["one_way_sensitivity"]["ranking"][0]
            sensitivity_value = report["one_way_sensitivity"]["variables"][
                most_sensitive
            ]["sensitivity"]
            summary["key_findings"].append(
                f"Most sensitive parameter: {most_sensitive} (sensitivity: {sensitivity_value:.2f})",
            )

        # Monte Carlo findings
        if "monte_carlo" in report:
            mc_results = report["monte_carlo"]
            prob_positive = mc_results["risk_metrics"]["probability_positive_npv"]
            var_5 = mc_results["risk_metrics"]["value_at_risk_5%"]

            summary["risk_assessment"] = {
                "probability_of_success": prob_positive,
                "value_at_risk_5_percent": var_5,
                "risk_level": (
                    "High"
                    if prob_positive < 0.6
                    else "Medium" if prob_positive < 0.8 else "Low"
                ),
            }

            summary["key_findings"].append(
                f"Probability of positive NPV: {prob_positive:.1%}",
            )

        # Recommendations based on analysis
        if summary["risk_assessment"].get("risk_level") == "High":
            summary["recommendations"].append(
                "High risk identified. Consider risk mitigation strategies.",
            )

        if "one_way_sensitivity" in report:
            top_3_sensitive = report["one_way_sensitivity"]["ranking"][:3]
            summary["recommendations"].append(
                f"Focus risk management on top 3 sensitive parameters: {', '.join(top_3_sensitive)}",
            )

        return summary


def create_lunar_mission_scenarios() -> dict[str, dict[str, float]]:
    """Create predefined scenarios for lunar mission analysis.

    Returns
    -------
        Dictionary of scenario definitions
    """
    return {
        "optimistic": {
            "development_cost_multiplier": 0.8,
            "launch_cost_multiplier": 0.7,
            "revenue_multiplier": 1.3,
            "schedule_multiplier": 0.9,
        },
        "pessimistic": {
            "development_cost_multiplier": 1.5,
            "launch_cost_multiplier": 1.3,
            "revenue_multiplier": 0.7,
            "schedule_multiplier": 1.4,
        },
        "conservative": {
            "development_cost_multiplier": 1.2,
            "launch_cost_multiplier": 1.1,
            "revenue_multiplier": 0.9,
            "schedule_multiplier": 1.1,
        },
        "aggressive": {
            "development_cost_multiplier": 0.9,
            "launch_cost_multiplier": 0.8,
            "revenue_multiplier": 1.2,
            "schedule_multiplier": 0.8,
        },
    }


def create_parameter_distributions() -> dict[str, dict[str, Any]]:
    """Create parameter distributions for Monte Carlo analysis.

    Returns
    -------
        Dictionary of parameter distribution specifications
    """
    return {
        "development_cost_multiplier": {
            "type": "triang",
            "min": 0.8,
            "mode": 1.0,
            "max": 1.8,
        },
        "launch_cost_multiplier": {
            "type": "triang",
            "min": 0.7,
            "mode": 1.0,
            "max": 1.5,
        },
        "revenue_multiplier": {
            "type": "triang",
            "min": 0.6,
            "mode": 1.0,
            "max": 1.4,
        },
        "discount_rate": {
            "type": "normal",
            "mean": 0.08,
            "std": 0.02,
        },
        "operational_efficiency": {
            "type": "normal",
            "mean": 0.85,
            "std": 0.1,
        },
    }
