"""Advanced scenario comparison tools for Task 9.

This module provides comprehensive tools for comparing multiple mission scenarios
with different economic assumptions, risk profiles, and strategic choices.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from src.economics.financial_models import (
    CashFlowModel,
    NPVAnalyzer,
    ROICalculator,
    FinancialParameters,
)
from src.economics.cost_models import MissionCostModel, CostBreakdown
from src.economics.isru_benefits import ISRUBenefitAnalyzer
from src.economics.sensitivity_analysis import EconomicSensitivityAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ScenarioDefinition:
    """Definition of a mission scenario for comparison."""

    name: str
    description: str

    # Mission parameters
    spacecraft_mass: float
    mission_duration_years: float
    launch_date: datetime

    # Technology choices
    propulsion_type: str = "chemical"  # chemical, electric, nuclear
    isru_enabled: bool = True
    technology_readiness: int = 3  # 1-4 scale

    # Economic assumptions
    financial_parameters: FinancialParameters = field(
        default_factory=FinancialParameters
    )
    cost_overrun_factor: float = 1.2  # Expected cost overrun
    revenue_uncertainty: float = 0.3  # Revenue uncertainty (std dev as fraction)

    # Strategic options
    public_private_partnership: bool = False
    international_collaboration: bool = False
    commercial_payload_capacity: float = 0.0  # kg available for commercial use

    # Risk factors
    technical_risk: float = 0.2  # 0-1 scale
    schedule_risk: float = 0.15  # 0-1 scale
    market_risk: float = 0.25  # 0-1 scale

    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResults:
    """Results from scenario analysis."""

    scenario_name: str

    # Financial metrics
    npv: float
    irr: float
    payback_period: float
    total_cost: float
    total_revenue: float
    roi: float

    # Risk-adjusted metrics
    risk_adjusted_npv: float
    probability_of_success: float
    value_at_risk: float  # 95% VaR

    # Performance metrics
    cost_per_kg_delivered: float
    revenue_per_year: float
    break_even_year: Optional[int]

    # Detailed results
    cost_breakdown: CostBreakdown
    cash_flow_profile: List[Tuple[datetime, float]]
    sensitivity_results: Dict[str, Any]
    monte_carlo_results: Optional[Dict[str, Any]] = None


class AdvancedScenarioComparator:
    """Advanced tools for comparing multiple mission scenarios.

    This class provides comprehensive scenario comparison including financial
    analysis, risk assessment, and multi-criteria decision analysis.
    """

    def __init__(self):
        """Initialize scenario comparator."""
        self.scenarios: List[ScenarioDefinition] = []
        self.results: Dict[str, ScenarioResults] = {}

        # Initialize component analyzers
        self.financial_analyzer = NPVAnalyzer()
        self.roi_calculator = ROICalculator()
        self.cost_model = MissionCostModel()
        self.isru_analyzer = ISRUBenefitAnalyzer()
        self.sensitivity_analyzer = EconomicSensitivityAnalyzer()

        logger.info("Initialized AdvancedScenarioComparator")

    def add_scenario(self, scenario: ScenarioDefinition) -> None:
        """Add a scenario for comparison.

        Args:
            scenario: Scenario definition
        """
        self.scenarios.append(scenario)
        logger.info(f"Added scenario: {scenario.name}")

    def analyze_scenario(
        self,
        scenario: ScenarioDefinition,
        run_monte_carlo: bool = True,
        monte_carlo_iterations: int = 1000,
    ) -> ScenarioResults:
        """Analyze a single scenario.

        Args:
            scenario: Scenario to analyze
            run_monte_carlo: Whether to run Monte Carlo simulation
            monte_carlo_iterations: Number of Monte Carlo iterations

        Returns:
            Scenario analysis results
        """
        logger.info(f"Analyzing scenario: {scenario.name}")

        # Calculate costs
        cost_breakdown = self._calculate_scenario_costs(scenario)

        # Generate cash flows
        cash_flow_model = self._generate_cash_flows(scenario, cost_breakdown)

        # Calculate financial metrics
        npv = self.financial_analyzer.calculate_npv(
            cash_flow_model, scenario.launch_date
        )

        irr = self.financial_analyzer.calculate_irr(
            cash_flow_model, scenario.launch_date
        )

        payback_period = self.financial_analyzer.calculate_payback_period(
            cash_flow_model, scenario.launch_date
        )

        # Calculate revenues
        total_revenue = self._calculate_scenario_revenue(scenario)

        # Calculate ROI
        total_cost = cost_breakdown.total * scenario.cost_overrun_factor
        roi = self.roi_calculator.calculate_simple_roi(total_revenue, total_cost)

        # Risk analysis
        risk_adjusted_npv = self._calculate_risk_adjusted_npv(npv, scenario)

        probability_of_success = self._calculate_success_probability(scenario)

        # Performance metrics
        cost_per_kg = total_cost / scenario.spacecraft_mass
        revenue_per_year = total_revenue / scenario.mission_duration_years

        # Extract cash flow profile
        cash_flow_profile = [(cf.date, cf.amount) for cf in cash_flow_model.cash_flows]

        # Sensitivity analysis
        sensitivity_results = self._run_sensitivity_analysis(
            scenario, npv, total_cost, total_revenue
        )

        # Store intermediate results for Monte Carlo access
        temp_results = {"total_cost": total_cost, "total_revenue": total_revenue}

        # Monte Carlo simulation
        monte_carlo_results = None
        if run_monte_carlo:
            monte_carlo_results = self._run_monte_carlo_simulation(
                scenario, monte_carlo_iterations, temp_results
            )

        # Determine break-even year
        break_even_year = self._calculate_break_even_year(cash_flow_profile)

        results = ScenarioResults(
            scenario_name=scenario.name,
            npv=npv,
            irr=irr,
            payback_period=payback_period,
            total_cost=total_cost,
            total_revenue=total_revenue,
            roi=roi,
            risk_adjusted_npv=risk_adjusted_npv,
            probability_of_success=probability_of_success,
            value_at_risk=(
                monte_carlo_results.get("var_95", 0) if monte_carlo_results else 0
            ),
            cost_per_kg_delivered=cost_per_kg,
            revenue_per_year=revenue_per_year,
            break_even_year=break_even_year,
            cost_breakdown=cost_breakdown,
            cash_flow_profile=cash_flow_profile,
            sensitivity_results=sensitivity_results,
            monte_carlo_results=monte_carlo_results,
        )

        self.results[scenario.name] = results
        return results

    def compare_all_scenarios(
        self, run_monte_carlo: bool = True, monte_carlo_iterations: int = 1000
    ) -> pd.DataFrame:
        """Compare all scenarios and return summary DataFrame.

        Args:
            run_monte_carlo: Whether to run Monte Carlo simulation
            monte_carlo_iterations: Number of Monte Carlo iterations

        Returns:
            DataFrame with scenario comparison
        """
        # Analyze all scenarios
        for scenario in self.scenarios:
            if scenario.name not in self.results:
                self.analyze_scenario(scenario, run_monte_carlo, monte_carlo_iterations)

        # Build comparison DataFrame
        comparison_data = []
        for scenario_name, results in self.results.items():
            comparison_data.append(
                {
                    "Scenario": scenario_name,
                    "NPV ($M)": results.npv / 1e6,
                    "IRR (%)": results.irr * 100,
                    "Payback (years)": results.payback_period,
                    "Total Cost ($M)": results.total_cost / 1e6,
                    "Total Revenue ($M)": results.total_revenue / 1e6,
                    "ROI (%)": results.roi * 100,
                    "Risk-Adjusted NPV ($M)": results.risk_adjusted_npv / 1e6,
                    "Success Probability (%)": results.probability_of_success * 100,
                    "Value at Risk ($M)": results.value_at_risk / 1e6,
                    "Cost per kg ($)": results.cost_per_kg_delivered,
                    "Break-even Year": results.break_even_year,
                }
            )

        df = pd.DataFrame(comparison_data)
        df.set_index("Scenario", inplace=True)

        return df

    def rank_scenarios(
        self, criteria_weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """Rank scenarios using multi-criteria decision analysis.

        Args:
            criteria_weights: Weights for different criteria (default: equal weights)

        Returns:
            List of (scenario_name, score) tuples sorted by score
        """
        if criteria_weights is None:
            criteria_weights = {
                "npv": 0.3,
                "irr": 0.2,
                "risk": 0.2,
                "strategic": 0.15,
                "flexibility": 0.15,
            }

        scores = []

        for scenario in self.scenarios:
            if scenario.name not in self.results:
                continue

            results = self.results[scenario.name]

            # Normalize metrics (0-1 scale)
            all_npvs = [r.npv for r in self.results.values()]
            npv_score = (results.npv - min(all_npvs)) / (max(all_npvs) - min(all_npvs))

            all_irrs = [r.irr for r in self.results.values()]
            irr_score = (results.irr - min(all_irrs)) / (max(all_irrs) - min(all_irrs))

            # Risk score (inverted - lower risk is better)
            risk_score = results.probability_of_success

            # Strategic score
            strategic_score = 0.0
            if scenario.public_private_partnership:
                strategic_score += 0.3
            if scenario.international_collaboration:
                strategic_score += 0.3
            if scenario.isru_enabled:
                strategic_score += 0.4

            # Flexibility score
            flexibility_score = (
                scenario.commercial_payload_capacity / scenario.spacecraft_mass
            )

            # Calculate weighted score
            total_score = (
                criteria_weights.get("npv", 0) * npv_score
                + criteria_weights.get("irr", 0) * irr_score
                + criteria_weights.get("risk", 0) * risk_score
                + criteria_weights.get("strategic", 0) * strategic_score
                + criteria_weights.get("flexibility", 0) * flexibility_score
            )

            scores.append((scenario.name, total_score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def generate_decision_matrix(self) -> pd.DataFrame:
        """Generate decision matrix for scenario comparison.

        Returns:
            Decision matrix DataFrame
        """
        matrix_data = []

        for scenario in self.scenarios:
            if scenario.name not in self.results:
                continue

            results = self.results[scenario.name]

            matrix_data.append(
                {
                    "Scenario": scenario.name,
                    # Financial attractiveness
                    "Financial Score": self._normalize_metric(
                        results.npv, [r.npv for r in self.results.values()]
                    ),
                    # Technical feasibility
                    "Technical Score": 1.0 - scenario.technical_risk,
                    # Strategic alignment
                    "Strategic Score": self._calculate_strategic_score(scenario),
                    # Risk profile
                    "Risk Score": results.probability_of_success,
                    # Implementation complexity
                    "Complexity Score": 1.0 - (scenario.technology_readiness / 4.0),
                    # Market potential
                    "Market Score": 1.0 - scenario.market_risk,
                }
            )

        df = pd.DataFrame(matrix_data)
        df.set_index("Scenario", inplace=True)

        return df

    def _calculate_scenario_costs(self, scenario: ScenarioDefinition) -> CostBreakdown:
        """Calculate costs for a scenario."""
        complexity_map = {
            "chemical": "moderate",
            "electric": "complex",
            "nuclear": "flagship",
        }

        complexity = complexity_map.get(scenario.propulsion_type, "moderate")

        return self.cost_model.estimate_total_mission_cost(
            spacecraft_mass=scenario.spacecraft_mass,
            mission_duration_years=scenario.mission_duration_years,
            technology_readiness=scenario.technology_readiness,
            complexity=complexity,
            schedule="nominal",
        )

    def _generate_cash_flows(
        self, scenario: ScenarioDefinition, cost_breakdown: CostBreakdown
    ) -> CashFlowModel:
        """Generate cash flows for scenario."""
        cash_flow_model = CashFlowModel(scenario.financial_parameters)

        # Add development costs
        cash_flow_model.add_development_costs(
            total_cost=cost_breakdown.development,
            start_date=scenario.launch_date,
            duration_months=36,  # 3 years development
        )

        # Add launch costs
        cash_flow_model.add_cash_flow(
            amount=-cost_breakdown.launch,
            date=scenario.launch_date,
            category="launch",
            description="Launch costs",
        )

        # Add operational costs
        cash_flow_model.add_operational_costs(
            monthly_cost=cost_breakdown.operations
            / (scenario.mission_duration_years * 12),
            start_date=scenario.launch_date,
            duration_months=int(scenario.mission_duration_years * 12),
        )

        # Add revenues
        if scenario.isru_enabled:
            # ISRU revenues
            annual_revenue = self._calculate_isru_revenue(scenario)
            monthly_revenue = annual_revenue / 12
            cash_flow_model.add_revenue_stream(
                monthly_revenue=monthly_revenue,
                start_date=scenario.launch_date,
                duration_months=int(scenario.mission_duration_years * 12),
            )

        if scenario.commercial_payload_capacity > 0:
            # Commercial payload revenues
            commercial_revenue = scenario.commercial_payload_capacity * 50000  # $50k/kg
            cash_flow_model.add_cash_flow(
                amount=commercial_revenue,
                date=scenario.launch_date,
                category="commercial",
                description="Commercial payload delivery",
            )

        return cash_flow_model

    def _calculate_scenario_revenue(self, scenario: ScenarioDefinition) -> float:
        """Calculate total revenue for scenario."""
        total_revenue = 0.0

        if scenario.isru_enabled:
            # Simplified ISRU revenue calculation
            annual_isru_revenue = self._calculate_isru_revenue(scenario)
            total_revenue += annual_isru_revenue * scenario.mission_duration_years

        if scenario.commercial_payload_capacity > 0:
            total_revenue += scenario.commercial_payload_capacity * 50000

        # Apply revenue uncertainty
        uncertainty_factor = 1.0 + np.random.normal(0, scenario.revenue_uncertainty)
        total_revenue *= max(0, uncertainty_factor)

        return total_revenue

    def _calculate_isru_revenue(self, scenario: ScenarioDefinition) -> float:
        """Calculate annual ISRU revenue."""
        # Simplified calculation - in practice would use TimeBasedISRUModel
        base_revenue = 10e6  # $10M/year base

        # Adjust for technology readiness
        tech_factor = scenario.technology_readiness / 4.0

        # Adjust for mission duration (economies of scale)
        scale_factor = np.log1p(scenario.mission_duration_years) / np.log(10)

        return base_revenue * tech_factor * scale_factor

    def _calculate_risk_adjusted_npv(
        self, base_npv: float, scenario: ScenarioDefinition
    ) -> float:
        """Calculate risk-adjusted NPV."""
        # Combined risk factor
        total_risk = 1.0 - (
            (1.0 - scenario.technical_risk)
            * (1.0 - scenario.schedule_risk)
            * (1.0 - scenario.market_risk)
        )

        # Risk adjustment
        risk_discount = 1.0 - total_risk * 0.5  # 50% NPV reduction for full risk

        return base_npv * risk_discount

    def _calculate_success_probability(self, scenario: ScenarioDefinition) -> float:
        """Calculate probability of mission success."""
        # Technical success probability
        tech_success = 1.0 - scenario.technical_risk

        # Schedule success probability (meeting timeline)
        schedule_success = 1.0 - scenario.schedule_risk

        # Market success probability
        market_success = 1.0 - scenario.market_risk

        # Combined probability (assuming independence)
        return tech_success * schedule_success * market_success

    def _run_sensitivity_analysis(
        self,
        scenario: ScenarioDefinition,
        base_npv: float,
        base_cost: float,
        base_revenue: float,
    ) -> Dict[str, Any]:
        """Run sensitivity analysis for scenario."""
        # Define sensitivity parameters
        sensitivity_params = {
            "cost_overrun": (0.8, 1.5),  # -20% to +50%
            "revenue_uncertainty": (0.7, 1.3),  # -30% to +30%
            "discount_rate": (0.05, 0.15),  # 5% to 15%
            "mission_duration": (0.8, 1.2),  # -20% to +20%
        }

        # Create sensitivity function
        def sensitivity_model(params: Dict[str, float]) -> float:
            adjusted_cost = base_cost * params.get("cost_overrun", 1.0)
            adjusted_revenue = base_revenue * params.get("revenue_uncertainty", 1.0)
            adjusted_npv = adjusted_revenue - adjusted_cost

            # Apply discount rate adjustment
            discount_factor = params.get("discount_rate", 0.08) / 0.08
            adjusted_npv /= discount_factor

            # Apply duration adjustment
            duration_factor = params.get("mission_duration", 1.0)
            adjusted_npv *= duration_factor

            return adjusted_npv

        # Run one-way sensitivity
        base_params = {
            "cost_overrun": 1.0,
            "revenue_uncertainty": 1.0,
            "discount_rate": 0.08,
            "mission_duration": 1.0,
        }

        self.sensitivity_analyzer.base_model_function = sensitivity_model

        return self.sensitivity_analyzer.one_way_sensitivity(
            base_params, sensitivity_params
        )

    def _run_monte_carlo_simulation(
        self,
        scenario: ScenarioDefinition,
        iterations: int,
        temp_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for scenario."""
        npv_results = []

        for _ in range(iterations):
            # Sample uncertain parameters
            cost_factor = np.random.normal(scenario.cost_overrun_factor, 0.1)
            revenue_factor = np.random.normal(1.0, scenario.revenue_uncertainty)

            # Recalculate with sampled parameters
            if temp_results:
                sampled_cost = temp_results["total_cost"] * cost_factor
                sampled_revenue = temp_results["total_revenue"] * revenue_factor
            else:
                # Fallback to stored results if available
                sampled_cost = self.results[scenario.name].total_cost * cost_factor
                sampled_revenue = (
                    self.results[scenario.name].total_revenue * revenue_factor
                )
            sampled_npv = sampled_revenue - sampled_cost

            npv_results.append(sampled_npv)

        # Calculate statistics
        npv_array = np.array(npv_results)

        return {
            "mean_npv": np.mean(npv_array),
            "std_npv": np.std(npv_array),
            "var_95": np.percentile(npv_array, 5),  # 95% Value at Risk
            "probability_positive": np.sum(npv_array > 0) / iterations,
            "distribution": npv_array,
        }

    def _calculate_break_even_year(
        self, cash_flow_profile: List[Tuple[datetime, float]]
    ) -> Optional[int]:
        """Calculate break-even year from cash flow profile."""
        if not cash_flow_profile:
            return None

        start_date = cash_flow_profile[0][0]
        cumulative_cf = 0.0

        for date, amount in cash_flow_profile:
            cumulative_cf += amount
            if cumulative_cf > 0:
                years_elapsed = (date - start_date).days / 365.25
                return int(np.ceil(years_elapsed))

        return None  # Never breaks even

    def _normalize_metric(self, value: float, all_values: List[float]) -> float:
        """Normalize metric to 0-1 scale."""
        if not all_values:
            return 0.5

        min_val = min(all_values)
        max_val = max(all_values)

        if max_val == min_val:
            return 0.5

        return (value - min_val) / (max_val - min_val)

    def _calculate_strategic_score(self, scenario: ScenarioDefinition) -> float:
        """Calculate strategic alignment score."""
        score = 0.0

        if scenario.public_private_partnership:
            score += 0.25
        if scenario.international_collaboration:
            score += 0.25
        if scenario.isru_enabled:
            score += 0.3
        if scenario.commercial_payload_capacity > 0:
            score += 0.2

        return score


def create_scenario_comparison_report(
    comparator: AdvancedScenarioComparator, output_format: str = "detailed"
) -> Dict[str, Any]:
    """Create comprehensive scenario comparison report.

    Args:
        comparator: Scenario comparator with analyzed scenarios
        output_format: Report format (detailed, summary, executive)

    Returns:
        Report dictionary
    """
    # Get comparison DataFrame
    comparison_df = comparator.compare_all_scenarios()

    # Get rankings
    rankings = comparator.rank_scenarios()

    # Get decision matrix
    decision_matrix = comparator.generate_decision_matrix()

    report = {
        "summary": {
            "total_scenarios": len(comparator.scenarios),
            "best_npv_scenario": comparison_df["NPV ($M)"].idxmax(),
            "best_irr_scenario": comparison_df["IRR (%)"].idxmax(),
            "lowest_risk_scenario": comparison_df["Success Probability (%)"].idxmax(),
            "recommended_scenario": rankings[0][0] if rankings else None,
        },
        "comparison_table": comparison_df.to_dict(),
        "rankings": rankings,
        "decision_matrix": decision_matrix.to_dict(),
    }

    if output_format == "detailed":
        # Add detailed results for each scenario
        report["detailed_results"] = {}
        for scenario_name, results in comparator.results.items():
            report["detailed_results"][scenario_name] = {
                "cost_breakdown": {
                    "development": results.cost_breakdown.development,
                    "launch": results.cost_breakdown.launch,
                    "operations": results.cost_breakdown.operations,
                    "total": results.cost_breakdown.total,
                },
                "sensitivity_analysis": results.sensitivity_results,
                "monte_carlo": results.monte_carlo_results,
            }

    return report
