"""
Economic Analysis Visualization Module.

Provides comprehensive visualization for lunar mission economic analysis including
financial dashboards, cost breakdowns, ROI analysis, sensitivity visualization,
and professional economic reporting charts.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any
from dataclasses import dataclass

from src.economics.financial_models import CashFlowModel
from src.economics.cost_models import MissionCostModel, CostBreakdown
from src.economics.isru_benefits import ISRUBenefitAnalyzer
from src.economics.reporting import FinancialSummary


@dataclass
class DashboardConfig:
    """Configuration for economic visualization dashboards."""

    # Dashboard layout
    width: int = 1400
    height: int = 1000
    title: str = "Lunar Mission Economic Analysis"

    # Color scheme
    primary_color: str = "#2E86AB"
    secondary_color: str = "#A23B72"
    success_color: str = "#059862"
    warning_color: str = "#F18F01"
    danger_color: str = "#C73E1D"

    # Chart styling
    show_grid: bool = True
    grid_color: str = "#E5E5E5"
    background_color: str = "#FFFFFF"
    text_color: str = "#333333"

    # Interactive features
    enable_hover: bool = True
    enable_zoom: bool = True
    show_legend: bool = True

    # Professional styling
    theme: str = "plotly_white"
    font_family: str = "Arial, sans-serif"
    title_font_size: int = 16
    axis_font_size: int = 12


class EconomicVisualizer:
    """
    Comprehensive economic analysis visualization using Plotly.

    Provides professional-grade visualization of lunar mission economics including:
    - Financial dashboard with key metrics
    - Cost breakdown analysis
    - Cash flow visualization
    - ROI and sensitivity analysis
    - ISRU benefit analysis
    - Risk assessment charts
    """

    def __init__(self, config: DashboardConfig | None = None) -> None:
        """
        Initialize economic visualizer.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.cost_model = MissionCostModel()
        self.isru_analyzer = ISRUBenefitAnalyzer()

    def create_financial_dashboard(
        self,
        financial_summary: FinancialSummary,
        cash_flow_model: CashFlowModel | None = None,
        cost_breakdown: CostBreakdown | None = None
    ) -> go.Figure:
        """
        Create comprehensive financial dashboard.

        Args:
            financial_summary: Financial summary data
            cash_flow_model: Optional cash flow model for detailed analysis
            cost_breakdown: Optional cost breakdown for detailed view

        Returns
        -------
            Plotly Figure with financial dashboard
        """
        # Create 2x3 subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Key Financial Metrics",
                "Investment vs Revenue",
                "Cash Flow Timeline",
                "Cost Breakdown",
                "ROI Analysis",
                "Risk Assessment"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Key Financial Metrics (Indicators)
        self._add_financial_indicators(fig, financial_summary, row=1, col=1)

        # 2. Investment vs Revenue
        self._add_investment_revenue_chart(fig, financial_summary, row=1, col=2)

        # 3. Cash Flow Timeline
        if cash_flow_model:
            self._add_cash_flow_timeline(fig, cash_flow_model, row=2, col=1)
        else:
            self._add_placeholder_chart(fig, "Cash Flow Data Not Available", row=2, col=1)

        # 4. Cost Breakdown
        if cost_breakdown:
            self._add_cost_breakdown_pie(fig, cost_breakdown, row=2, col=2)
        else:
            self._add_placeholder_chart(fig, "Cost Breakdown Data Not Available", row=2, col=2)

        # 5. ROI Analysis
        self._add_roi_analysis(fig, financial_summary, row=3, col=1)

        # 6. Risk Assessment
        self._add_risk_assessment(fig, financial_summary, row=3, col=2)

        # Update layout
        fig.update_layout(
            title={
                "text": f"Financial Dashboard - {self.config.title}",
                "x": 0.5,
                "font": {"size": 20, "family": self.config.font_family}
            },
            template=self.config.theme,
            height=self.config.height,
            width=self.config.width,
            showlegend=self.config.show_legend,
            font={"family": self.config.font_family, "color": self.config.text_color}
        )

        return fig

    def create_cost_analysis_dashboard(
        self,
        cost_breakdown: CostBreakdown,
        comparison_scenarios: list[dict[str, Any]] | None = None
    ) -> go.Figure:
        """
        Create detailed cost analysis dashboard.

        Args:
            cost_breakdown: Detailed cost breakdown
            comparison_scenarios: Optional cost scenarios for comparison

        Returns
        -------
            Plotly Figure with cost analysis dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Cost Category Breakdown",
                "Cost Trends Over Time",
                "Cost per Unit Analysis",
                "Scenario Comparison"
            ],
            specs=[
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )

        # 1. Cost Category Breakdown (Pie Chart)
        categories = ["Development", "Launch", "Spacecraft", "Operations", "Ground Systems", "Contingency"]
        values = [
            cost_breakdown.development,
            cost_breakdown.launch,
            cost_breakdown.spacecraft,
            cost_breakdown.operations,
            cost_breakdown.ground_systems,
            cost_breakdown.contingency
        ]

        colors = [self.config.primary_color, self.config.secondary_color,
                 self.config.success_color, self.config.warning_color,
                 "#8B5A83", "#6B7280"]

        fig.add_trace(
            go.Pie(
                labels=categories,
                values=values,
                marker={"colors": colors},
                textinfo="label+percent+value",
                texttemplate="%{label}<br>%{percent}<br>$%{value:.1f}M",
                hovertemplate="%{label}<br>Cost: $%{value:.1f}M<br>Percentage: %{percent}<extra></extra>"
            ),
            row=1, col=1
        )

        # 2. Cost Trends Over Time (simulated)
        years = list(range(2025, 2035))
        dev_costs = [cost_breakdown.development * (0.8 + 0.4 * np.exp(-0.5 * i)) for i in range(len(years))]
        ops_costs = [cost_breakdown.operations * (0.5 + 0.5 * i / len(years)) for i in range(len(years))]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=dev_costs,
                mode="lines+markers",
                name="Development Costs",
                line={"color": self.config.primary_color, "width": 3},
                marker={"size": 8}
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=ops_costs,
                mode="lines+markers",
                name="Operations Costs",
                line={"color": self.config.secondary_color, "width": 3},
                marker={"size": 8}
            ),
            row=1, col=2
        )

        # 3. Cost per Unit Analysis
        units = ["Per kg to LEO", "Per kg to Moon", "Per Mission Day", "Per Crew Member"]
        unit_costs = [
            cost_breakdown.launch / 5000,  # Assuming 5000 kg payload
            cost_breakdown.total / 1000,   # Assuming 1000 kg lunar payload
            cost_breakdown.operations / 365,  # Per day
            cost_breakdown.total / 4       # Assuming 4 crew members
        ]

        fig.add_trace(
            go.Bar(
                x=units,
                y=unit_costs,
                marker={"color": [self.config.primary_color, self.config.secondary_color,
                                  self.config.success_color, self.config.warning_color]},
                text=[f"${v:.1f}K" for v in unit_costs],
                textposition="auto"
            ),
            row=2, col=1
        )

        # 4. Scenario Comparison
        if comparison_scenarios:
            scenario_names = [s.get("name", f"Scenario {i+1}") for i, s in enumerate(comparison_scenarios)]
            scenario_costs = [s.get("total_cost", 0) / 1e6 for s in comparison_scenarios]  # Convert to millions

            # Add baseline
            scenario_names.insert(0, "Baseline")
            scenario_costs.insert(0, cost_breakdown.total)

            colors_scenarios = [self.config.primary_color] + [self.config.secondary_color] * len(comparison_scenarios)

            fig.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=scenario_costs,
                    marker={"color": colors_scenarios},
                    text=[f"${v:.0f}M" for v in scenario_costs],
                    textposition="auto"
                ),
                row=2, col=2
            )
        else:
            self._add_placeholder_chart(fig, "No Comparison Scenarios Available", row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f"Cost Analysis Dashboard - Total: ${cost_breakdown.total:.0f}M",
            template=self.config.theme,
            height=800,
            width=1400
        )

        return fig

    def create_isru_analysis_dashboard(
        self,
        isru_analysis: dict[str, Any],
        resource_name: str = "water_ice"
    ) -> go.Figure:
        """
        Create ISRU economic analysis dashboard.

        Args:
            isru_analysis: ISRU analysis results
            resource_name: Name of primary resource

        Returns
        -------
            Plotly Figure with ISRU analysis dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f"ISRU {resource_name.title()} Economics",
                "Production Timeline",
                "Break-even Analysis",
                "Revenue Streams"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "pie"}]
            ]
        )

        # Extract data
        financial_metrics = isru_analysis.get("financial_metrics", {})
        production_profile = isru_analysis.get("production_profile", {})
        break_even = isru_analysis.get("break_even_analysis", {})
        revenue_streams = isru_analysis.get("revenue_streams", {})

        # 1. ISRU Key Metrics
        npv = financial_metrics.get("npv", 0) / 1e6  # Convert to millions
        roi = financial_metrics.get("roi", 0) * 100  # Convert to percentage
        break_even.get("payback_period_months", 0) / 12  # Convert to years

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=npv,
                title={"text": "ISRU NPV ($M)"},
                number={"suffix": "M", "font": {"size": 20}},
                delta={"reference": 0, "position": "bottom"},
                domain={"x": [0, 0.5], "y": [0.7, 1]}
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=roi,
                title={"text": "ISRU ROI (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": self.config.success_color},
                    "steps": [
                        {"range": [0, 25], "color": "lightgray"},
                        {"range": [25, 50], "color": "yellow"},
                        {"range": [50, 100], "color": "lightgreen"}
                    ],
                    "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90}
                },
                domain={"x": [0, 0.5], "y": [0, 0.6]}
            ),
            row=1, col=1
        )

        # 2. Production Timeline
        if "monthly_production" in production_profile:
            months = list(range(1, len(production_profile["monthly_production"]) + 1))
            production = production_profile["monthly_production"]
            cumulative = np.cumsum(production)

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=production,
                    mode="lines+markers",
                    name="Monthly Production",
                    line={"color": self.config.primary_color, "width": 3},
                    yaxis="y"
                ),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=cumulative,
                    mode="lines+markers",
                    name="Cumulative Production",
                    line={"color": self.config.secondary_color, "width": 3},
                    yaxis="y2"
                ),
                row=1, col=2
            )
        else:
            self._add_placeholder_chart(fig, "Production Data Not Available", row=1, col=2)

        # 3. Break-even Analysis
        if "monthly_cash_flow" in break_even:
            months = list(range(1, len(break_even["monthly_cash_flow"]) + 1))
            cash_flows = break_even["monthly_cash_flow"]
            cumulative_cf = np.cumsum(cash_flows)

            # Find break-even point
            break_even_month = None
            for i, cf in enumerate(cumulative_cf):
                if cf > 0:
                    break_even_month = i + 1
                    break

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=cumulative_cf,
                    mode="lines+markers",
                    name="Cumulative Cash Flow",
                    line={"color": self.config.success_color, "width": 3},
                    fill="tonexty"
                ),
                row=2, col=1
            )

            # Add break-even line
            fig.add_hline(
                y=0, line_dash="dash", line_color="red",
                annotation_text="Break-even", row=2, col=1
            )

            if break_even_month:
                fig.add_vline(
                    x=break_even_month, line_dash="dash", line_color="green",
                    annotation_text=f"Break-even: Month {break_even_month}",
                    row=2, col=1
                )
        else:
            self._add_placeholder_chart(fig, "Break-even Data Not Available", row=2, col=1)

        # 4. Revenue Streams
        if revenue_streams:
            stream_names = list(revenue_streams.keys())
            stream_values = list(revenue_streams.values())

            fig.add_trace(
                go.Pie(
                    labels=stream_names,
                    values=stream_values,
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>Revenue: $%{value:.1f}M<br>Percentage: %{percent}<extra></extra>"
                ),
                row=2, col=2
            )
        else:
            self._add_placeholder_chart(fig, "Revenue Stream Data Not Available", row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f"ISRU Economic Analysis - {resource_name.title()}",
            template=self.config.theme,
            height=900,
            width=1400
        )

        return fig

    def create_sensitivity_analysis_dashboard(
        self,
        sensitivity_results: dict[str, Any],
        monte_carlo_results: dict[str, Any] | None = None
    ) -> go.Figure:
        """
        Create sensitivity and risk analysis dashboard.

        Args:
            sensitivity_results: Sensitivity analysis results
            monte_carlo_results: Optional Monte Carlo simulation results

        Returns
        -------
            Plotly Figure with sensitivity analysis dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Parameter Sensitivity",
                "Monte Carlo Distribution",
                "Risk Metrics",
                "Scenario Analysis"
            ],
            specs=[
                [{"type": "bar"}, {"type": "histogram"}],
                [{"type": "table"}, {"type": "scatter"}]
            ]
        )

        # 1. Parameter Sensitivity
        if "parameter_sensitivity" in sensitivity_results:
            params = list(sensitivity_results["parameter_sensitivity"].keys())
            sensitivities = list(sensitivity_results["parameter_sensitivity"].values())

            # Sort by absolute sensitivity
            sorted_pairs = sorted(zip(params, sensitivities, strict=False), key=lambda x: abs(x[1]), reverse=True)
            params_sorted, sens_sorted = zip(*sorted_pairs, strict=False)

            colors = [self.config.success_color if s > 0 else self.config.danger_color for s in sens_sorted]

            fig.add_trace(
                go.Bar(
                    x=list(params_sorted),
                    y=list(sens_sorted),
                    marker={"color": colors},
                    text=[f"{s:.2f}" for s in sens_sorted],
                    textposition="auto"
                ),
                row=1, col=1
            )
        else:
            self._add_placeholder_chart(fig, "Sensitivity Data Not Available", row=1, col=1)

        # 2. Monte Carlo Distribution
        if monte_carlo_results and "npv_distribution" in monte_carlo_results:
            npv_dist = monte_carlo_results["npv_distribution"]

            fig.add_trace(
                go.Histogram(
                    x=npv_dist,
                    nbinsx=50,
                    name="NPV Distribution",
                    marker={"color": self.config.primary_color, "opacity": 0.7},
                    histnorm="probability density"
                ),
                row=1, col=2
            )

            # Add mean and percentiles
            mean_npv = np.mean(npv_dist)
            p5 = np.percentile(npv_dist, 5)
            p95 = np.percentile(npv_dist, 95)

            fig.add_vline(x=mean_npv, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: ${mean_npv/1e6:.1f}M", row=1, col=2)
            fig.add_vline(x=p5, line_dash="dot", line_color="orange",
                         annotation_text=f"5%: ${p5/1e6:.1f}M", row=1, col=2)
            fig.add_vline(x=p95, line_dash="dot", line_color="orange",
                         annotation_text=f"95%: ${p95/1e6:.1f}M", row=1, col=2)
        else:
            self._add_placeholder_chart(fig, "Monte Carlo Data Not Available", row=1, col=2)

        # 3. Risk Metrics Table
        if monte_carlo_results and "risk_metrics" in monte_carlo_results:
            risk_metrics = monte_carlo_results["risk_metrics"]

            metrics_data = [
                ["Probability of Positive NPV", f"{risk_metrics.get('probability_positive_npv', 0):.1%}"],
                ["Value at Risk (5%)", f"${risk_metrics.get('value_at_risk_5%', 0)/1e6:.1f}M"],
                ["Expected Shortfall", f"${risk_metrics.get('expected_shortfall', 0)/1e6:.1f}M"],
                ["Standard Deviation", f"${risk_metrics.get('std_deviation', 0)/1e6:.1f}M"],
                ["Coefficient of Variation", f"{risk_metrics.get('coefficient_variation', 0):.2f}"]
            ]

            fig.add_trace(
                go.Table(
                    header={
                        "values": ["Risk Metric", "Value"],
                        "fill_color": self.config.primary_color,
                        "align": "left",
                        "font": {"color": "white", "size": 12}
                    },
                    cells={
                        "values": list(zip(*metrics_data, strict=False)),
                        "fill_color": "lightblue",
                        "align": "left",
                        "font": {"size": 11}
                    }
                ),
                row=2, col=1
            )
        else:
            self._add_placeholder_chart(fig, "Risk Metrics Not Available", row=2, col=1)

        # 4. Scenario Analysis
        if "scenario_analysis" in sensitivity_results:
            scenarios = sensitivity_results["scenario_analysis"]
            scenario_names = list(scenarios.keys())
            scenario_npvs = [s.get("npv", 0)/1e6 for s in scenarios.values()]
            scenario_probs = [s.get("probability", 0) for s in scenarios.values()]

            fig.add_trace(
                go.Scatter(
                    x=scenario_probs,
                    y=scenario_npvs,
                    mode="markers+text",
                    text=scenario_names,
                    textposition="top center",
                    marker={
                        "size": [p*100 for p in scenario_probs],  # Size by probability
                        "color": scenario_npvs,
                        "colorscale": "RdYlGn",
                        "showscale": True,
                        "colorbar": {"title": "NPV ($M)"}
                    },
                    name="Scenarios"
                ),
                row=2, col=2
            )
        else:
            self._add_placeholder_chart(fig, "Scenario Data Not Available", row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Economic Sensitivity & Risk Analysis",
            template=self.config.theme,
            height=900,
            width=1400
        )

        return fig

    def _add_financial_indicators(
        self,
        fig: go.Figure,
        financial_summary: FinancialSummary,
        row: int,
        col: int
    ) -> None:
        """Add financial KPI indicators."""
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=financial_summary.net_present_value / 1e6,
                title={"text": "NPV ($M)"},
                number={"suffix": "M", "font": {"size": 16}},
                delta={"reference": 0, "position": "bottom"},
                domain={"x": [0, 0.25], "y": [0.8, 1]}
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=financial_summary.internal_rate_of_return * 100,
                title={"text": "IRR (%)"},
                number={"suffix": "%", "font": {"size": 16}},
                domain={"x": [0.25, 0.5], "y": [0.8, 1]}
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Indicator(
                mode="number",
                value=financial_summary.payback_period_years,
                title={"text": "Payback (Years)"},
                number={"suffix": " yr", "font": {"size": 16}},
                domain={"x": [0, 0.25], "y": [0.4, 0.7]}
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=financial_summary.probability_of_success * 100,
                title={"text": "Success Probability"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": self.config.success_color},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "lightgreen"}
                    ]
                },
                domain={"x": [0.25, 0.5], "y": [0.4, 0.7]}
            ),
            row=row, col=col
        )

    def _add_investment_revenue_chart(
        self,
        fig: go.Figure,
        financial_summary: FinancialSummary,
        row: int,
        col: int
    ) -> None:
        """Add investment vs revenue bar chart."""
        categories = ["Investment", "Revenue", "Net Benefit"]
        values = [
            financial_summary.total_investment / 1e6,
            financial_summary.total_revenue / 1e6,
            financial_summary.net_present_value / 1e6
        ]
        colors = [self.config.danger_color, self.config.success_color, self.config.primary_color]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                marker={"color": colors},
                text=[f"${v:.0f}M" for v in values],
                textposition="auto",
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_cash_flow_timeline(
        self,
        fig: go.Figure,
        cash_flow_model: CashFlowModel,
        row: int,
        col: int
    ) -> None:
        """Add cash flow timeline chart."""
        # Extract cash flows (simplified - would need actual implementation)
        # This is a placeholder implementation
        months = list(range(1, 61))  # 5 years
        cash_flows = [(-10 if i < 24 else 5) * (1 + 0.1 * np.random.randn()) for i in months]
        cumulative = np.cumsum(cash_flows)

        fig.add_trace(
            go.Scatter(
                x=months,
                y=cash_flows,
                mode="lines",
                name="Monthly Cash Flow",
                line={"color": self.config.primary_color, "width": 2},
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=months,
                y=cumulative,
                mode="lines",
                name="Cumulative Cash Flow",
                line={"color": self.config.secondary_color, "width": 2},
                yaxis="y2",
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_cost_breakdown_pie(
        self,
        fig: go.Figure,
        cost_breakdown: CostBreakdown,
        row: int,
        col: int
    ) -> None:
        """Add cost breakdown pie chart."""
        labels = ["Development", "Launch", "Spacecraft", "Operations", "Ground", "Contingency"]
        values = [
            cost_breakdown.development,
            cost_breakdown.launch,
            cost_breakdown.spacecraft,
            cost_breakdown.operations,
            cost_breakdown.ground_systems,
            cost_breakdown.contingency
        ]

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                textinfo="label+percent",
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_roi_analysis(
        self,
        fig: go.Figure,
        financial_summary: FinancialSummary,
        row: int,
        col: int
    ) -> None:
        """Add ROI analysis chart."""
        years = list(range(1, 11))
        roi_values = [financial_summary.return_on_investment * (1 - np.exp(-0.3 * y)) for y in years]

        fig.add_trace(
            go.Bar(
                x=years,
                y=roi_values,
                marker={"color": self.config.success_color},
                text=[f"{v:.1%}" for v in roi_values],
                textposition="auto",
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_risk_assessment(
        self,
        fig: go.Figure,
        financial_summary: FinancialSummary,
        row: int,
        col: int
    ) -> None:
        """Add risk assessment scatter plot."""
        # Simulated risk vs return analysis
        scenarios = ["Conservative", "Base Case", "Optimistic", "Aggressive"]
        returns = [0.08, 0.15, 0.25, 0.35]
        risks = [0.05, 0.12, 0.22, 0.35]

        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode="markers+text",
                text=scenarios,
                textposition="top center",
                marker={
                    "size": 15,
                    "color": [self.config.success_color, self.config.primary_color,
                          self.config.warning_color, self.config.danger_color]
                },
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_placeholder_chart(
        self,
        fig: go.Figure,
        message: str,
        row: int,
        col: int
    ) -> None:
        """Add placeholder for missing data."""
        fig.add_annotation(
            text=message,
            xref=f"x{col if row > 1 or col > 1 else ''}",
            yref=f"y{col if row > 1 or col > 1 else ''}",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 14, "color": "gray"},
            row=row, col=col
        )


def create_quick_financial_dashboard(
    npv: float,
    irr: float,
    roi: float,
    payback_years: float,
    total_investment: float,
    total_revenue: float
) -> go.Figure:
    """
    Quick function to create a simple financial dashboard.

    Args:
        npv: Net Present Value
        irr: Internal Rate of Return
        roi: Return on Investment
        payback_years: Payback period in years
        total_investment: Total investment required
        total_revenue: Total expected revenue

    Returns
    -------
        Plotly Figure with financial dashboard
    """
    # Create financial summary
    financial_summary = FinancialSummary(
        total_investment=total_investment,
        total_revenue=total_revenue,
        net_present_value=npv,
        internal_rate_of_return=irr,
        return_on_investment=roi,
        payback_period_years=payback_years,
        mission_duration_years=8,
        probability_of_success=0.75
    )

    # Create visualizer and dashboard
    visualizer = EconomicVisualizer()
    return visualizer.create_financial_dashboard(financial_summary)
