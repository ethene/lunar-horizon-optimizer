"""Financial models for economic analysis - Task 5 core implementation.

This module provides fundamental financial analysis tools including ROI, NPV,
and cash flow modeling for lunar mission economic evaluation.
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CashFlow:
    """Represents a cash flow event in mission economics."""

    amount: float  # Cash flow amount (positive = income, negative = cost)
    date: datetime  # Date of cash flow
    category: str  # Category (development, launch, operations, revenue, etc.)
    description: str = ""  # Optional description

    def __post_init__(self):
        """Validate cash flow data."""
        if not isinstance(self.date, datetime):
            msg = "Cash flow date must be a datetime object"
            raise ValueError(msg)


@dataclass
class FinancialParameters:
    """Financial parameters for economic analysis."""

    discount_rate: float = 0.08  # Annual discount rate (8%)
    inflation_rate: float = 0.03  # Annual inflation rate (3%)
    tax_rate: float = 0.25  # Corporate tax rate (25%)
    risk_premium: float = 0.02  # Risk premium for space projects (2%)
    project_duration_years: int = 10  # Project duration

    def __post_init__(self):
        """Validate financial parameters."""
        if not 0 <= self.discount_rate <= 1:
            msg = "Discount rate must be between 0 and 1"
            raise ValueError(msg)
        if not 0 <= self.tax_rate <= 1:
            msg = "Tax rate must be between 0 and 1"
            raise ValueError(msg)


class CashFlowModel:
    """Cash flow modeling for lunar mission economics.

    This class manages the complete cash flow profile of a lunar mission,
    including development costs, launch costs, operational expenses, and revenues.
    """

    def __init__(self, financial_params: FinancialParameters = None) -> None:
        """Initialize cash flow model.

        Args:
            financial_params: Financial parameters for analysis
        """
        self.financial_params = financial_params or FinancialParameters()
        self.cash_flows: list[CashFlow] = []

        logger.info(f"Initialized CashFlowModel with {self.financial_params.discount_rate:.1%} discount rate")

    def add_cash_flow(self, amount: float, date: datetime, category: str, description: str = "") -> None:
        """Add a cash flow event.

        Args:
            amount: Cash flow amount (positive = income, negative = cost)
            date: Date of cash flow
            category: Category of cash flow
            description: Optional description
        """
        cash_flow = CashFlow(amount, date, category, description)
        self.cash_flows.append(cash_flow)
        logger.debug(f"Added cash flow: {category} ${amount:,.0f} on {date.strftime('%Y-%m-%d')}")

    def add_development_costs(self, total_cost: float, start_date: datetime, duration_months: int) -> None:
        """Add development costs spread over development period.

        Args:
            total_cost: Total development cost
            start_date: Development start date
            duration_months: Development duration in months
        """
        monthly_cost = total_cost / duration_months

        for month in range(duration_months):
            cash_flow_date = start_date + timedelta(days=30 * month)
            self.add_cash_flow(-monthly_cost, cash_flow_date, "development",
                             f"Development cost month {month + 1}")

        logger.info(f"Added development costs: ${total_cost:,.0f} over {duration_months} months")

    def add_launch_costs(self, cost_per_launch: float, launch_dates: list[datetime]) -> None:
        """Add launch costs for multiple launches.

        Args:
            cost_per_launch: Cost per launch
            launch_dates: List of launch dates
        """
        for i, date in enumerate(launch_dates):
            self.add_cash_flow(-cost_per_launch, date, "launch", f"Launch {i + 1}")

        total_launch_cost = cost_per_launch * len(launch_dates)
        logger.info(f"Added launch costs: ${total_launch_cost:,.0f} for {len(launch_dates)} launches")

    def add_operational_costs(self, monthly_cost: float, start_date: datetime, duration_months: int) -> None:
        """Add operational costs over mission duration.

        Args:
            monthly_cost: Monthly operational cost
            start_date: Operations start date
            duration_months: Duration of operations in months
        """
        for month in range(duration_months):
            cash_flow_date = start_date + timedelta(days=30 * month)
            # Apply inflation to operational costs
            inflated_cost = monthly_cost * (1 + self.financial_params.inflation_rate) ** (month / 12)
            self.add_cash_flow(-inflated_cost, cash_flow_date, "operations",
                             f"Operations month {month + 1}")

        total_ops_cost = sum(monthly_cost * (1 + self.financial_params.inflation_rate) ** (m / 12)
                           for m in range(duration_months))
        logger.info(f"Added operational costs: ${total_ops_cost:,.0f} over {duration_months} months")

    def add_revenue_stream(self, monthly_revenue: float, start_date: datetime, duration_months: int) -> None:
        """Add revenue stream over mission duration.

        Args:
            monthly_revenue: Monthly revenue
            start_date: Revenue start date
            duration_months: Duration of revenue generation
        """
        for month in range(duration_months):
            revenue_date = start_date + timedelta(days=30 * month)
            # Apply inflation to revenues
            inflated_revenue = monthly_revenue * (1 + self.financial_params.inflation_rate) ** (month / 12)
            self.add_cash_flow(inflated_revenue, revenue_date, "revenue",
                             f"Revenue month {month + 1}")

        total_revenue = sum(monthly_revenue * (1 + self.financial_params.inflation_rate) ** (m / 12)
                          for m in range(duration_months))
        logger.info(f"Added revenue stream: ${total_revenue:,.0f} over {duration_months} months")

    def get_cash_flows_by_category(self, category: str) -> list[CashFlow]:
        """Get cash flows by category.

        Args:
            category: Category to filter by

        Returns
        -------
            List of cash flows in the specified category
        """
        return [cf for cf in self.cash_flows if cf.category == category]

    def get_total_by_category(self) -> dict[str, float]:
        """Get total cash flows by category.

        Returns
        -------
            Dictionary with total cash flows by category
        """
        totals = {}
        for cash_flow in self.cash_flows:
            if cash_flow.category not in totals:
                totals[cash_flow.category] = 0
            totals[cash_flow.category] += cash_flow.amount

        return totals

    def get_annual_cash_flows(self) -> dict[int, float]:
        """Get annual cash flow totals.

        Returns
        -------
            Dictionary with annual cash flow totals by year
        """
        annual_flows = {}
        for cash_flow in self.cash_flows:
            year = cash_flow.date.year
            if year not in annual_flows:
                annual_flows[year] = 0
            annual_flows[year] += cash_flow.amount

        return annual_flows


class NPVAnalyzer:
    """Net Present Value (NPV) analysis for lunar missions.

    This class provides comprehensive NPV analysis including sensitivity
    analysis and scenario modeling for economic evaluation.
    """

    def __init__(self, financial_params: FinancialParameters = None) -> None:
        """Initialize NPV analyzer.

        Args:
            financial_params: Financial parameters for analysis
        """
        self.financial_params = financial_params or FinancialParameters()

        logger.info(f"Initialized NPVAnalyzer with {self.financial_params.discount_rate:.1%} discount rate")

    def calculate_npv(self, cash_flow_model: CashFlowModel, reference_date: datetime | None = None) -> float:
        """Calculate Net Present Value of cash flows.

        Args:
            cash_flow_model: Cash flow model containing all cash flows
            reference_date: Reference date for NPV calculation (default: earliest cash flow date)

        Returns
        -------
            Net Present Value
        """
        if not cash_flow_model.cash_flows:
            return 0.0

        if reference_date is None:
            reference_date = min(cf.date for cf in cash_flow_model.cash_flows)

        npv = 0.0
        discount_rate = self.financial_params.discount_rate

        for cash_flow in cash_flow_model.cash_flows:
            # Calculate years from reference date
            years_from_reference = (cash_flow.date - reference_date).days / 365.25

            # Calculate present value
            present_value = cash_flow.amount / (1 + discount_rate) ** years_from_reference
            npv += present_value

        logger.info(f"Calculated NPV: ${npv:,.0f}")
        return npv

    def calculate_irr(self, cash_flow_model: CashFlowModel, reference_date: datetime | None = None) -> float:
        """Calculate Internal Rate of Return (IRR).

        Args:
            cash_flow_model: Cash flow model
            reference_date: Reference date for IRR calculation

        Returns
        -------
            Internal Rate of Return (as decimal)
        """
        if not cash_flow_model.cash_flows:
            return 0.0

        if reference_date is None:
            reference_date = min(cf.date for cf in cash_flow_model.cash_flows)

        # Prepare cash flows for IRR calculation
        cash_flows_by_time = {}
        for cf in cash_flow_model.cash_flows:
            years = (cf.date - reference_date).days / 365.25
            if years not in cash_flows_by_time:
                cash_flows_by_time[years] = 0
            cash_flows_by_time[years] += cf.amount

        # Convert to arrays
        times = np.array(sorted(cash_flows_by_time.keys()))
        flows = np.array([cash_flows_by_time[t] for t in times])

        # Use Newton-Raphson method to find IRR
        irr = self._calculate_irr_newton_raphson(times, flows)

        logger.info(f"Calculated IRR: {irr:.1%}")
        return irr

    def _calculate_irr_newton_raphson(self, times: np.ndarray, flows: np.ndarray,
                                    initial_guess: float = 0.1, max_iterations: int = 100) -> float:
        """Calculate IRR using Newton-Raphson method."""
        rate = initial_guess

        for _ in range(max_iterations):
            # Calculate NPV and its derivative
            npv = np.sum(flows / (1 + rate) ** times)
            npv_derivative = np.sum(-times * flows / (1 + rate) ** (times + 1))

            if abs(npv) < 1e-6:  # Convergence threshold
                return rate

            if abs(npv_derivative) < 1e-10:  # Avoid division by zero
                break

            # Newton-Raphson update
            rate_new = rate - npv / npv_derivative

            if abs(rate_new - rate) < 1e-6:
                return rate_new

            rate = rate_new

        # If no convergence, return best estimate
        return rate

    def calculate_payback_period(self, cash_flow_model: CashFlowModel, reference_date: datetime | None = None) -> float:
        """Calculate payback period in years.

        Args:
            cash_flow_model: Cash flow model
            reference_date: Reference date for calculation

        Returns
        -------
            Payback period in years (returns inf if never breaks even)
        """
        if not cash_flow_model.cash_flows:
            return float("inf")

        if reference_date is None:
            reference_date = min(cf.date for cf in cash_flow_model.cash_flows)

        # Sort cash flows by date
        sorted_flows = sorted(cash_flow_model.cash_flows, key=lambda cf: cf.date)

        cumulative_cash_flow = 0.0
        for cash_flow in sorted_flows:
            cumulative_cash_flow += cash_flow.amount

            if cumulative_cash_flow >= 0:  # Break-even point
                years = (cash_flow.date - reference_date).days / 365.25
                logger.info(f"Calculated payback period: {years:.1f} years")
                return years

        logger.info("Project never reaches payback")
        return float("inf")

    def sensitivity_analysis(self, cash_flow_model: CashFlowModel,
                           variable_ranges: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
        """Perform sensitivity analysis on NPV.

        Args:
            cash_flow_model: Base cash flow model
            variable_ranges: Dictionary of variable names and their (min, max) ranges

        Returns
        -------
            Dictionary with sensitivity results
        """
        logger.info("Performing NPV sensitivity analysis")

        base_npv = self.calculate_npv(cash_flow_model)
        sensitivity_results = {"base_npv": base_npv}

        for variable, (min_val, max_val) in variable_ranges.items():
            npv_values = []
            variable_values = np.linspace(min_val, max_val, 10)

            for value in variable_values:
                # Create modified cash flow model
                modified_model = self._apply_sensitivity_change(cash_flow_model, variable, value)
                npv = self.calculate_npv(modified_model)
                npv_values.append(npv)

            sensitivity_results[variable] = {
                "variable_values": variable_values.tolist(),
                "npv_values": npv_values,
                "sensitivity": (max(npv_values) - min(npv_values)) / (max_val - min_val)
            }

        return sensitivity_results

    def _apply_sensitivity_change(self, base_model: CashFlowModel, variable: str, value: float) -> CashFlowModel:
        """Apply sensitivity change to cash flow model."""
        # This is a simplified implementation
        # In practice, you'd modify specific cash flows based on the variable
        modified_model = CashFlowModel(self.financial_params)

        for cf in base_model.cash_flows:
            modified_amount = cf.amount

            # Apply scaling based on variable type
            if variable == "development_cost_multiplier":
                if cf.category == "development":
                    modified_amount *= value
            elif variable == "revenue_multiplier":
                if cf.category == "revenue":
                    modified_amount *= value
            elif variable == "operational_cost_multiplier":
                if cf.category == "operations":
                    modified_amount *= value

            modified_model.add_cash_flow(modified_amount, cf.date, cf.category, cf.description)

        return modified_model


class ROICalculator:
    """Return on Investment (ROI) calculator for lunar missions.

    This class provides various ROI calculations and comparative analysis
    for lunar mission investment evaluation.
    """

    def __init__(self) -> None:
        """Initialize ROI calculator."""
        logger.info("Initialized ROICalculator")

    def calculate_simple_roi(self, total_investment: float, total_return: float) -> float:
        """Calculate simple ROI.

        Args:
            total_investment: Total investment amount
            total_return: Total return amount

        Returns
        -------
            ROI as decimal (e.g., 0.15 = 15%)
        """
        if total_investment == 0:
            return 0.0

        roi = (total_return - total_investment) / total_investment
        logger.info(f"Simple ROI: {roi:.1%}")
        return roi

    def calculate_annualized_roi(self, initial_investment: float, final_value: float, years: float) -> float:
        """Calculate annualized ROI.

        Args:
            initial_investment: Initial investment
            final_value: Final value
            years: Investment period in years

        Returns
        -------
            Annualized ROI as decimal
        """
        if initial_investment <= 0 or years <= 0:
            return 0.0

        annualized_roi = (final_value / initial_investment) ** (1 / years) - 1
        logger.info(f"Annualized ROI: {annualized_roi:.1%}")
        return annualized_roi

    def calculate_risk_adjusted_roi(self, roi: float, risk_free_rate: float, beta: float) -> float:
        """Calculate risk-adjusted ROI using CAPM.

        Args:
            roi: Raw ROI
            risk_free_rate: Risk-free rate
            beta: Beta coefficient for risk adjustment

        Returns
        -------
            Risk-adjusted ROI
        """
        market_premium = 0.08  # Assumed market risk premium
        required_return = risk_free_rate + beta * market_premium
        risk_adjusted_roi = roi - required_return

        logger.info(f"Risk-adjusted ROI: {risk_adjusted_roi:.1%}")
        return risk_adjusted_roi

    def compare_investments(self, investments: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Compare multiple investment options.

        Args:
            investments: Dictionary of investment options with their metrics

        Returns
        -------
            Comparison analysis
        """
        logger.info(f"Comparing {len(investments)} investment options")

        comparison = {
            "investments": investments,
            "rankings": {},
            "best_options": {}
        }

        # Rank by different metrics
        metrics = ["roi", "npv", "irr", "payback_period"]

        for metric in metrics:
            if all(metric in inv for inv in investments.values()):
                # Sort investments by metric (lower is better for payback_period)
                reverse_order = metric != "payback_period"
                sorted_investments = sorted(
                    investments.items(),
                    key=lambda x: x[1][metric],
                    reverse=reverse_order
                )

                comparison["rankings"][metric] = [name for name, _ in sorted_investments]
                comparison["best_options"][metric] = sorted_investments[0][0]

        return comparison


def create_mission_cash_flow_model(mission_config: dict[str, Any]) -> CashFlowModel:
    """Create a complete cash flow model for a lunar mission.

    Args:
        mission_config: Mission configuration with cost and revenue parameters

    Returns
    -------
        Configured cash flow model
    """
    financial_params = FinancialParameters(
        discount_rate=mission_config.get("discount_rate", 0.08),
        inflation_rate=mission_config.get("inflation_rate", 0.03),
        project_duration_years=mission_config.get("duration_years", 10)
    )

    model = CashFlowModel(financial_params)

    # Add development costs
    if "development" in mission_config:
        dev_config = mission_config["development"]
        model.add_development_costs(
            total_cost=dev_config["total_cost"],
            start_date=datetime.fromisoformat(dev_config["start_date"]),
            duration_months=dev_config["duration_months"]
        )

    # Add launch costs
    if "launches" in mission_config:
        launch_config = mission_config["launches"]
        launch_dates = [datetime.fromisoformat(date) for date in launch_config["dates"]]
        model.add_launch_costs(
            cost_per_launch=launch_config["cost_per_launch"],
            launch_dates=launch_dates
        )

    # Add operational costs
    if "operations" in mission_config:
        ops_config = mission_config["operations"]
        model.add_operational_costs(
            monthly_cost=ops_config["monthly_cost"],
            start_date=datetime.fromisoformat(ops_config["start_date"]),
            duration_months=ops_config["duration_months"]
        )

    # Add revenue streams
    if "revenue" in mission_config:
        rev_config = mission_config["revenue"]
        model.add_revenue_stream(
            monthly_revenue=rev_config["monthly_revenue"],
            start_date=datetime.fromisoformat(rev_config["start_date"]),
            duration_months=rev_config["duration_months"]
        )

    return model
