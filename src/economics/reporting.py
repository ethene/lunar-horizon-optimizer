"""Economic reporting and summary generation for Task 5 completion.

This module provides comprehensive reporting capabilities for economic analysis
results, including financial summaries, executive reports, and data export.
"""

import json
import csv
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FinancialSummary:
    """Financial summary data structure for lunar mission economics."""
    
    # Basic financial metrics
    total_investment: float = 0.0
    total_revenue: float = 0.0
    net_present_value: float = 0.0
    internal_rate_of_return: float = 0.0
    return_on_investment: float = 0.0
    payback_period_years: float = 0.0
    
    # Cost breakdown
    development_cost: float = 0.0
    launch_cost: float = 0.0
    operational_cost: float = 0.0
    contingency_cost: float = 0.0
    
    # Revenue breakdown
    primary_revenue: float = 0.0
    secondary_revenue: float = 0.0
    isru_benefits: float = 0.0
    
    # Risk metrics
    probability_of_success: float = 0.0
    value_at_risk_5_percent: float = 0.0
    expected_shortfall: float = 0.0
    
    # Mission parameters
    mission_duration_years: int = 0
    spacecraft_mass_kg: float = 0.0
    launch_date: str = ""
    
    # Analysis metadata
    analysis_date: str = ""
    analyst: str = ""
    confidence_level: float = 0.0
    
    def __post_init__(self):
        """Set default analysis date if not provided."""
        if not self.analysis_date:
            self.analysis_date = datetime.now().isoformat()


class EconomicReporter:
    """Comprehensive economic reporting for lunar mission analysis.
    
    This class generates various types of reports including executive summaries,
    detailed financial analysis, and comparative studies.
    """
    
    def __init__(self, output_directory: str = "economic_reports"):
        """Initialize economic reporter.
        
        Args:
            output_directory: Directory for report output
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report templates and formatting
        self.currency_format = "${:,.0f}"
        self.percent_format = "{:.1%}"
        self.number_format = "{:,.1f}"
        
        logger.info(f"Initialized EconomicReporter with output directory: {self.output_dir}")
    
    def generate_executive_summary(self,
                                 financial_summary: FinancialSummary,
                                 analysis_results: Dict[str, Any] = None) -> str:
        """Generate executive summary report.
        
        Args:
            financial_summary: Financial summary data
            analysis_results: Additional analysis results
            
        Returns:
            Executive summary as formatted string
        """
        logger.info("Generating executive summary report")
        
        # Determine project viability
        viability = self._assess_project_viability(financial_summary)
        
        # Generate summary text
        summary = f"""
LUNAR MISSION ECONOMIC ANALYSIS - EXECUTIVE SUMMARY
===================================================

Analysis Date: {financial_summary.analysis_date}
Analyst: {financial_summary.analyst or 'Economic Analysis Team'}
Confidence Level: {self.percent_format.format(financial_summary.confidence_level)}

PROJECT OVERVIEW
----------------
Mission Duration: {financial_summary.mission_duration_years} years
Spacecraft Mass: {self.number_format.format(financial_summary.spacecraft_mass_kg)} kg
Launch Date: {financial_summary.launch_date or 'TBD'}

FINANCIAL HIGHLIGHTS
--------------------
Total Investment Required: {self.currency_format.format(financial_summary.total_investment)}
Expected Total Revenue: {self.currency_format.format(financial_summary.total_revenue)}
Net Present Value (NPV): {self.currency_format.format(financial_summary.net_present_value)}
Internal Rate of Return: {self.percent_format.format(financial_summary.internal_rate_of_return)}
Return on Investment: {self.percent_format.format(financial_summary.return_on_investment)}
Payback Period: {financial_summary.payback_period_years:.1f} years

COST BREAKDOWN
--------------
Development: {self.currency_format.format(financial_summary.development_cost)} ({self.percent_format.format(financial_summary.development_cost/financial_summary.total_investment if financial_summary.total_investment > 0 else 0)})
Launch: {self.currency_format.format(financial_summary.launch_cost)} ({self.percent_format.format(financial_summary.launch_cost/financial_summary.total_investment if financial_summary.total_investment > 0 else 0)})
Operations: {self.currency_format.format(financial_summary.operational_cost)} ({self.percent_format.format(financial_summary.operational_cost/financial_summary.total_investment if financial_summary.total_investment > 0 else 0)})
Contingency: {self.currency_format.format(financial_summary.contingency_cost)} ({self.percent_format.format(financial_summary.contingency_cost/financial_summary.total_investment if financial_summary.total_investment > 0 else 0)})

REVENUE SOURCES
---------------
Primary Revenue: {self.currency_format.format(financial_summary.primary_revenue)}
Secondary Revenue: {self.currency_format.format(financial_summary.secondary_revenue)}
ISRU Benefits: {self.currency_format.format(financial_summary.isru_benefits)}

RISK ASSESSMENT
---------------
Probability of Success: {self.percent_format.format(financial_summary.probability_of_success)}
Value at Risk (5%): {self.currency_format.format(financial_summary.value_at_risk_5_percent)}
Risk Level: {viability['risk_level']}

RECOMMENDATION
--------------
Project Viability: {viability['recommendation']}
Key Justification: {viability['justification']}

{self._generate_key_insights(financial_summary, analysis_results)}
"""
        
        return summary
    
    def generate_detailed_financial_report(self,
                                         analysis_results: Dict[str, Any],
                                         include_sensitivity: bool = True,
                                         include_scenarios: bool = True) -> str:
        """Generate detailed financial analysis report.
        
        Args:
            analysis_results: Complete analysis results
            include_sensitivity: Include sensitivity analysis
            include_scenarios: Include scenario analysis
            
        Returns:
            Detailed report as formatted string
        """
        logger.info("Generating detailed financial report")
        
        report = f"""
LUNAR MISSION ECONOMIC ANALYSIS - DETAILED REPORT
==================================================

Analysis Timestamp: {datetime.now().isoformat()}

"""
        
        # Cash flow analysis
        if 'cash_flow_analysis' in analysis_results:
            report += self._format_cash_flow_section(analysis_results['cash_flow_analysis'])
        
        # NPV analysis
        if 'npv_analysis' in analysis_results:
            report += self._format_npv_section(analysis_results['npv_analysis'])
        
        # Cost analysis
        if 'cost_analysis' in analysis_results:
            report += self._format_cost_section(analysis_results['cost_analysis'])
        
        # ISRU benefits
        if 'isru_analysis' in analysis_results:
            report += self._format_isru_section(analysis_results['isru_analysis'])
        
        # Sensitivity analysis
        if include_sensitivity and 'sensitivity_analysis' in analysis_results:
            report += self._format_sensitivity_section(analysis_results['sensitivity_analysis'])
        
        # Scenario analysis
        if include_scenarios and 'scenario_analysis' in analysis_results:
            report += self._format_scenario_section(analysis_results['scenario_analysis'])
        
        return report
    
    def generate_comparison_report(self,
                                 alternatives: Dict[str, FinancialSummary],
                                 criteria_weights: Dict[str, float] = None) -> str:
        """Generate comparative analysis report for multiple alternatives.
        
        Args:
            alternatives: Dictionary of alternative analyses
            criteria_weights: Weights for decision criteria
            
        Returns:
            Comparison report as formatted string
        """
        logger.info(f"Generating comparison report for {len(alternatives)} alternatives")
        
        if not alternatives:
            return "No alternatives provided for comparison."
        
        # Default criteria weights
        if criteria_weights is None:
            criteria_weights = {
                'npv': 0.3,
                'roi': 0.25,
                'risk': 0.2,
                'payback': 0.15,
                'strategic_value': 0.1
            }
        
        report = f"""
LUNAR MISSION ALTERNATIVES COMPARISON
=====================================

Analysis Date: {datetime.now().isoformat()}
Number of Alternatives: {len(alternatives)}

ALTERNATIVE OVERVIEW
--------------------
"""
        
        # Summary table
        report += self._create_comparison_table(alternatives)
        
        # Detailed comparison
        report += "\nDETAILED COMPARISON\n" + "="*19 + "\n"
        
        for name, summary in alternatives.items():
            report += f"\n{name.upper()}\n" + "-" * len(name) + "\n"
            report += f"NPV: {self.currency_format.format(summary.net_present_value)}\n"
            report += f"ROI: {self.percent_format.format(summary.return_on_investment)}\n"
            report += f"Payback: {summary.payback_period_years:.1f} years\n"
            report += f"Success Probability: {self.percent_format.format(summary.probability_of_success)}\n"
            report += f"Investment Required: {self.currency_format.format(summary.total_investment)}\n"
        
        # Ranking and recommendation
        ranking = self._rank_alternatives(alternatives, criteria_weights)
        report += "\nRANKING AND RECOMMENDATION\n" + "="*27 + "\n"
        
        for i, (name, score) in enumerate(ranking, 1):
            report += f"{i}. {name} (Score: {score:.2f})\n"
        
        best_alternative = ranking[0][0]
        report += f"\nRECOMMENDED ALTERNATIVE: {best_alternative}\n"
        report += self._justify_recommendation(alternatives[best_alternative], alternatives)
        
        return report
    
    def export_to_csv(self,
                     data: Union[FinancialSummary, Dict[str, Any], List[Dict[str, Any]]],
                     filename: str) -> Path:
        """Export data to CSV format.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.csv"
        
        logger.info(f"Exporting data to CSV: {filepath}")
        
        with open(filepath, 'w', newline='') as csvfile:
            if isinstance(data, FinancialSummary):
                # Export financial summary
                writer = csv.DictWriter(csvfile, fieldnames=asdict(data).keys())
                writer.writeheader()
                writer.writerow(asdict(data))
            
            elif isinstance(data, dict):
                # Export dictionary data
                if data:
                    first_key = next(iter(data.keys()))
                    if isinstance(data[first_key], dict):
                        # Nested dictionary - flatten
                        flattened_data = []
                        for key, value in data.items():
                            if isinstance(value, dict):
                                row = {'name': key}
                                row.update(value)
                                flattened_data.append(row)
                        
                        if flattened_data:
                            writer = csv.DictWriter(csvfile, fieldnames=flattened_data[0].keys())
                            writer.writeheader()
                            writer.writerows(flattened_data)
                    else:
                        # Simple dictionary
                        writer = csv.writer(csvfile)
                        writer.writerow(['Key', 'Value'])
                        for key, value in data.items():
                            writer.writerow([key, value])
            
            elif isinstance(data, list):
                # Export list of dictionaries
                if data and isinstance(data[0], dict):
                    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        
        logger.info(f"Data exported successfully to {filepath}")
        return filepath
    
    def export_to_json(self,
                      data: Any,
                      filename: str,
                      pretty_print: bool = True) -> Path:
        """Export data to JSON format.
        
        Args:
            data: Data to export
            filename: Output filename
            pretty_print: Format JSON for readability
            
        Returns:
            Path to exported file
        """
        filepath = self.output_dir / f"{filename}.json"
        
        logger.info(f"Exporting data to JSON: {filepath}")
        
        # Convert dataclass to dict if necessary
        if isinstance(data, FinancialSummary):
            export_data = asdict(data)
        else:
            export_data = data
        
        with open(filepath, 'w') as jsonfile:
            if pretty_print:
                json.dump(export_data, jsonfile, indent=2, default=str)
            else:
                json.dump(export_data, jsonfile, default=str)
        
        logger.info(f"Data exported successfully to {filepath}")
        return filepath
    
    def generate_dashboard_data(self,
                              financial_summary: FinancialSummary,
                              analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate data structure for economic dashboard visualization.
        
        Args:
            financial_summary: Financial summary
            analysis_results: Additional analysis results
            
        Returns:
            Dashboard data structure
        """
        logger.info("Generating dashboard data")
        
        dashboard_data = {
            'summary_metrics': {
                'npv': financial_summary.net_present_value,
                'irr': financial_summary.internal_rate_of_return,
                'roi': financial_summary.return_on_investment,
                'payback_years': financial_summary.payback_period_years,
                'success_probability': financial_summary.probability_of_success
            },
            'cost_breakdown': {
                'Development': financial_summary.development_cost,
                'Launch': financial_summary.launch_cost,
                'Operations': financial_summary.operational_cost,
                'Contingency': financial_summary.contingency_cost
            },
            'revenue_breakdown': {
                'Primary Revenue': financial_summary.primary_revenue,
                'Secondary Revenue': financial_summary.secondary_revenue,
                'ISRU Benefits': financial_summary.isru_benefits
            },
            'risk_metrics': {
                'Value at Risk (5%)': financial_summary.value_at_risk_5_percent,
                'Expected Shortfall': financial_summary.expected_shortfall,
                'Probability of Success': financial_summary.probability_of_success
            }
        }
        
        # Add time series data if available
        if analysis_results and 'cash_flow_model' in analysis_results:
            cash_flows = analysis_results['cash_flow_model']
            if hasattr(cash_flows, 'get_annual_cash_flows'):
                dashboard_data['annual_cash_flows'] = cash_flows.get_annual_cash_flows()
        
        # Add sensitivity data if available
        if analysis_results and 'sensitivity_analysis' in analysis_results:
            sens_analysis = analysis_results['sensitivity_analysis']
            if 'variables' in sens_analysis:
                dashboard_data['sensitivity_data'] = {
                    name: {
                        'sensitivity': data['sensitivity'],
                        'parameter_values': data['parameter_values'],
                        'npv_values': data['npv_values']
                    }
                    for name, data in sens_analysis['variables'].items()
                }
        
        return dashboard_data
    
    def _assess_project_viability(self, summary: FinancialSummary) -> Dict[str, str]:
        """Assess overall project viability."""
        # Scoring based on financial metrics
        npv_score = 1 if summary.net_present_value > 0 else 0
        roi_score = 1 if summary.return_on_investment > 0.15 else 0.5 if summary.return_on_investment > 0 else 0
        payback_score = 1 if summary.payback_period_years < 7 else 0.5 if summary.payback_period_years < 10 else 0
        risk_score = 1 if summary.probability_of_success > 0.7 else 0.5 if summary.probability_of_success > 0.5 else 0
        
        total_score = (npv_score + roi_score + payback_score + risk_score) / 4
        
        if total_score >= 0.75:
            recommendation = "RECOMMENDED"
            risk_level = "Low"
            justification = "Strong financial metrics and acceptable risk profile"
        elif total_score >= 0.5:
            recommendation = "CONDITIONAL"
            risk_level = "Medium"
            justification = "Moderate financial returns with manageable risks"
        else:
            recommendation = "NOT RECOMMENDED"
            risk_level = "High"
            justification = "Poor financial outlook and/or high risk profile"
        
        return {
            'recommendation': recommendation,
            'risk_level': risk_level,
            'justification': justification,
            'score': total_score
        }
    
    def _generate_key_insights(self,
                             summary: FinancialSummary,
                             analysis_results: Dict[str, Any] = None) -> str:
        """Generate key insights section."""
        insights = ["KEY INSIGHTS", "="*12, ""]
        
        # Financial insights
        if summary.net_present_value > 0:
            insights.append(f"• Project creates {self.currency_format.format(summary.net_present_value)} in shareholder value")
        else:
            insights.append(f"• Project destroys {self.currency_format.format(abs(summary.net_present_value))} in shareholder value")
        
        if summary.return_on_investment > 0.2:
            insights.append(f"• High ROI of {self.percent_format.format(summary.return_on_investment)} indicates strong returns")
        elif summary.return_on_investment > 0.1:
            insights.append(f"• Moderate ROI of {self.percent_format.format(summary.return_on_investment)} provides acceptable returns")
        else:
            insights.append(f"• Low ROI of {self.percent_format.format(summary.return_on_investment)} may not justify investment")
        
        # Risk insights
        if summary.probability_of_success < 0.6:
            insights.append("• High execution risk requires careful risk management")
        
        # Cost insights
        total_cost = summary.development_cost + summary.launch_cost + summary.operational_cost
        if summary.development_cost / total_cost > 0.6:
            insights.append("• Development costs dominate - focus on controlling R&D expenses")
        if summary.launch_cost / total_cost > 0.4:
            insights.append("• Launch costs are significant - consider cost reduction strategies")
        
        return "\n".join(insights)
    
    def _format_cash_flow_section(self, cash_flow_data: Dict[str, Any]) -> str:
        """Format cash flow analysis section."""
        section = "\nCASH FLOW ANALYSIS\n" + "="*18 + "\n"
        
        if 'annual_cash_flows' in cash_flow_data:
            section += "\nAnnual Cash Flows:\n"
            for year, amount in cash_flow_data['annual_cash_flows'].items():
                section += f"Year {year}: {self.currency_format.format(amount)}\n"
        
        return section
    
    def _format_npv_section(self, npv_data: Dict[str, Any]) -> str:
        """Format NPV analysis section."""
        section = "\nNET PRESENT VALUE ANALYSIS\n" + "="*26 + "\n"
        
        section += f"NPV: {self.currency_format.format(npv_data.get('npv', 0))}\n"
        section += f"IRR: {self.percent_format.format(npv_data.get('irr', 0))}\n"
        section += f"Payback Period: {npv_data.get('payback_period', 0):.1f} years\n"
        
        return section
    
    def _format_cost_section(self, cost_data: Dict[str, Any]) -> str:
        """Format cost analysis section."""
        section = "\nCOST ANALYSIS\n" + "="*13 + "\n"
        
        if 'cost_breakdown' in cost_data:
            for category, amount in cost_data['cost_breakdown'].items():
                section += f"{category}: {self.currency_format.format(amount)}\n"
        
        return section
    
    def _format_isru_section(self, isru_data: Dict[str, Any]) -> str:
        """Format ISRU analysis section."""
        section = "\nISRU BENEFITS ANALYSIS\n" + "="*22 + "\n"
        
        section += f"Total ISRU Value: {self.currency_format.format(isru_data.get('total_value', 0))}\n"
        section += f"Break-even Production: {isru_data.get('break_even_kg', 0):,.0f} kg\n"
        
        return section
    
    def _format_sensitivity_section(self, sens_data: Dict[str, Any]) -> str:
        """Format sensitivity analysis section."""
        section = "\nSENSITIVITY ANALYSIS\n" + "="*20 + "\n"
        
        if 'ranking' in sens_data:
            section += "\nMost Sensitive Parameters:\n"
            for i, param in enumerate(sens_data['ranking'][:5], 1):
                sensitivity = sens_data['variables'][param]['sensitivity']
                section += f"{i}. {param}: {sensitivity:.2f}\n"
        
        return section
    
    def _format_scenario_section(self, scenario_data: Dict[str, Any]) -> str:
        """Format scenario analysis section."""
        section = "\nSCENARIO ANALYSIS\n" + "="*17 + "\n"
        
        if 'scenarios' in scenario_data:
            for name, data in scenario_data['scenarios'].items():
                npv = data.get('npv', 0)
                section += f"{name.title()}: {self.currency_format.format(npv)}\n"
        
        return section
    
    def _create_comparison_table(self, alternatives: Dict[str, FinancialSummary]) -> str:
        """Create comparison table for alternatives."""
        table = "\n"
        table += f"{'Alternative':<15} {'NPV ($M)':<12} {'ROI':<8} {'Payback':<10} {'Success %':<10}\n"
        table += "-" * 65 + "\n"
        
        for name, summary in alternatives.items():
            table += f"{name:<15} "
            table += f"{summary.net_present_value/1e6:<12.1f} "
            table += f"{summary.return_on_investment:<8.1%} "
            table += f"{summary.payback_period_years:<10.1f} "
            table += f"{summary.probability_of_success:<10.1%}\n"
        
        return table
    
    def _rank_alternatives(self,
                          alternatives: Dict[str, FinancialSummary],
                          weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """Rank alternatives using weighted scoring."""
        scores = {}
        
        # Normalize metrics for scoring
        npvs = [s.net_present_value for s in alternatives.values()]
        rois = [s.return_on_investment for s in alternatives.values()]
        paybacks = [s.payback_period_years for s in alternatives.values()]
        probs = [s.probability_of_success for s in alternatives.values()]
        
        npv_max, npv_min = max(npvs), min(npvs)
        roi_max, roi_min = max(rois), min(rois)
        payback_max, payback_min = max(paybacks), min(paybacks)
        prob_max, prob_min = max(probs), min(probs)
        
        for name, summary in alternatives.items():
            # Normalize scores (0-1 scale)
            npv_score = (summary.net_present_value - npv_min) / (npv_max - npv_min) if npv_max > npv_min else 0.5
            roi_score = (summary.return_on_investment - roi_min) / (roi_max - roi_min) if roi_max > roi_min else 0.5
            payback_score = 1 - (summary.payback_period_years - payback_min) / (payback_max - payback_min) if payback_max > payback_min else 0.5  # Lower is better
            prob_score = (summary.probability_of_success - prob_min) / (prob_max - prob_min) if prob_max > prob_min else 0.5
            
            # Calculate weighted score
            total_score = (npv_score * weights.get('npv', 0.25) +
                          roi_score * weights.get('roi', 0.25) +
                          payback_score * weights.get('payback', 0.25) +
                          prob_score * weights.get('risk', 0.25))
            
            scores[name] = total_score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _justify_recommendation(self,
                              best_alternative: FinancialSummary,
                              all_alternatives: Dict[str, FinancialSummary]) -> str:
        """Justify the recommended alternative."""
        justification = f"\nJUSTIFICATION:\n"
        
        # Compare with other alternatives
        npvs = [s.net_present_value for s in all_alternatives.values()]
        if best_alternative.net_present_value == max(npvs):
            justification += "• Highest NPV among alternatives\n"
        
        rois = [s.return_on_investment for s in all_alternatives.values()]
        if best_alternative.return_on_investment == max(rois):
            justification += "• Best return on investment\n"
        
        paybacks = [s.payback_period_years for s in all_alternatives.values()]
        if best_alternative.payback_period_years == min(paybacks):
            justification += "• Shortest payback period\n"
        
        probs = [s.probability_of_success for s in all_alternatives.values()]
        if best_alternative.probability_of_success == max(probs):
            justification += "• Highest probability of success\n"
        
        return justification


def create_financial_summary_from_analysis(analysis_results: Dict[str, Any]) -> FinancialSummary:
    """Create FinancialSummary from analysis results.
    
    Args:
        analysis_results: Complete analysis results dictionary
        
    Returns:
        FinancialSummary object
    """
    # Extract data from various analysis components
    npv_data = analysis_results.get('npv_analysis', {})
    cost_data = analysis_results.get('cost_analysis', {})
    revenue_data = analysis_results.get('revenue_analysis', {})
    risk_data = analysis_results.get('risk_analysis', {})
    mission_data = analysis_results.get('mission_parameters', {})
    
    return FinancialSummary(
        total_investment=cost_data.get('total_cost', 0),
        total_revenue=revenue_data.get('total_revenue', 0),
        net_present_value=npv_data.get('npv', 0),
        internal_rate_of_return=npv_data.get('irr', 0),
        return_on_investment=npv_data.get('roi', 0),
        payback_period_years=npv_data.get('payback_period', 0),
        
        development_cost=cost_data.get('development_cost', 0),
        launch_cost=cost_data.get('launch_cost', 0),
        operational_cost=cost_data.get('operational_cost', 0),
        contingency_cost=cost_data.get('contingency_cost', 0),
        
        primary_revenue=revenue_data.get('primary_revenue', 0),
        secondary_revenue=revenue_data.get('secondary_revenue', 0),
        isru_benefits=revenue_data.get('isru_benefits', 0),
        
        probability_of_success=risk_data.get('probability_of_success', 0),
        value_at_risk_5_percent=risk_data.get('value_at_risk_5%', 0),
        expected_shortfall=risk_data.get('expected_shortfall', 0),
        
        mission_duration_years=mission_data.get('duration_years', 0),
        spacecraft_mass_kg=mission_data.get('spacecraft_mass', 0),
        launch_date=mission_data.get('launch_date', ''),
        
        confidence_level=analysis_results.get('confidence_level', 0.8)
    )