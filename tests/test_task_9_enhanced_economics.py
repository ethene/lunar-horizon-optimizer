"""
Test suite for Task 9: Enhanced Economic Analysis Module

This test suite validates the implementation of advanced economic analysis
features including time-dependent ISRU models, scenario comparison tools,
and enhanced financial reporting capabilities.

Features tested:
- Time-based ISRU production models with ramp-up and maintenance
- Advanced scenario comparison and ranking
- Multi-criteria decision analysis
- Enhanced financial reporting
- Integration with existing economic modules
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Local imports
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economics.advanced_isru_models import (
    ProductionProfile,
    ISRUFacility,
    TimeBasedISRUModel,
    create_isru_production_forecast,
)
from src.economics.scenario_comparison import (
    ScenarioDefinition,
    ScenarioResults,
    AdvancedScenarioComparator,
    create_scenario_comparison_report,
)
from src.economics.financial_models import FinancialParameters


class TestTimeBasedISRUModels(unittest.TestCase):
    """Test time-based ISRU production models."""

    def setUp(self):
        """Set up test fixtures."""
        self.start_date = datetime(2030, 1, 1)

        # Create test production profile
        self.water_profile = ProductionProfile(
            resource_type="water_ice",
            initial_capacity=100,  # kg/day
            peak_capacity=1000,  # kg/day
            ramp_up_months=12,
            maintenance_frequency_days=90,
            maintenance_duration_days=3,
            efficiency_degradation_rate=0.02,
            learning_curve_factor=0.1,
        )

        # Create test facility
        self.test_facility = ISRUFacility(
            name="Test Water Plant",
            resources_produced=["water_ice", "oxygen", "hydrogen"],
            capital_cost=50e6,
            operational_cost_per_day=10000,
            lifetime_years=15,
            production_profiles={"water_ice": self.water_profile},
            startup_date=self.start_date,
        )

        # Create model
        self.model = TimeBasedISRUModel()

    def test_production_profile_validation(self):
        """Test production profile validation."""
        # Test invalid peak capacity
        with self.assertRaises(ValueError):
            ProductionProfile(
                resource_type="test",
                initial_capacity=1000,
                peak_capacity=100,  # Less than initial
                ramp_up_months=12,
                maintenance_frequency_days=90,
                maintenance_duration_days=3,
                efficiency_degradation_rate=0.02,
                learning_curve_factor=0.1,
            )

        # Test invalid degradation rate
        with self.assertRaises(ValueError):
            ProductionProfile(
                resource_type="test",
                initial_capacity=100,
                peak_capacity=1000,
                ramp_up_months=12,
                maintenance_frequency_days=90,
                maintenance_duration_days=3,
                efficiency_degradation_rate=1.5,  # > 1
                learning_curve_factor=0.1,
            )

    def test_production_rate_calculation(self):
        """Test production rate calculation over time."""
        self.model.add_facility(self.test_facility)

        # Test before startup
        pre_start = self.start_date - timedelta(days=10)
        rate = self.model.calculate_production_rate(
            self.test_facility, "water_ice", pre_start
        )
        self.assertEqual(rate, 0.0)

        # Test at startup
        rate_start = self.model.calculate_production_rate(
            self.test_facility, "water_ice", self.start_date
        )
        self.assertAlmostEqual(rate_start, 100.0, delta=1.0)  # Initial capacity

        # Test during ramp-up (6 months)
        mid_ramp = self.start_date + timedelta(days=180)
        rate_mid = self.model.calculate_production_rate(
            self.test_facility, "water_ice", mid_ramp
        )
        self.assertGreater(rate_mid, 100.0)
        self.assertLess(rate_mid, 1000.0)

        # Test at peak (after ramp-up)
        peak_date = self.start_date + timedelta(days=400)
        rate_peak = self.model.calculate_production_rate(
            self.test_facility, "water_ice", peak_date
        )
        # Should be close to peak with learning curve benefit
        self.assertGreater(rate_peak, 1000.0)

        # Test during maintenance
        maintenance_date = self.start_date + timedelta(days=87)  # First maintenance
        rate_maintenance = self.model.calculate_production_rate(
            self.test_facility, "water_ice", maintenance_date
        )
        self.assertEqual(rate_maintenance, 0.0)

    def test_cumulative_production(self):
        """Test cumulative production calculation."""
        self.model.add_facility(self.test_facility)

        # Calculate production for 1 year
        end_date = self.start_date + timedelta(days=365)
        results = self.model.calculate_cumulative_production(
            "water_ice", self.start_date, end_date, time_step_days=1
        )

        self.assertIn("cumulative_production", results)
        self.assertIn("average_daily_production", results)
        self.assertIn("peak_production", results)
        self.assertIn("production_days", results)
        self.assertIn("downtime_days", results)

        # Check that cumulative production is positive
        self.assertGreater(results["cumulative_production"], 0)

        # Check that there are some downtime days (maintenance)
        self.assertGreater(results["downtime_days"], 0)

        # Check that peak > average (due to ramp-up)
        self.assertGreater(
            results["peak_production"], results["average_daily_production"]
        )

    def test_time_dependent_economics(self):
        """Test economic calculations with time-dependent production."""
        self.model.add_facility(self.test_facility)

        end_date = self.start_date + timedelta(days=5 * 365)  # 5 years
        economics = self.model.calculate_time_dependent_economics(
            "water_ice", self.start_date, end_date
        )

        # Check all required fields
        self.assertIn("cumulative_production", economics)
        self.assertIn("total_revenue", economics)
        self.assertIn("total_discounted_revenue", economics)
        self.assertIn("total_capital_cost", economics)
        self.assertIn("total_operational_cost", economics)
        self.assertIn("net_revenue", economics)
        self.assertIn("roi", economics)

        # Verify capital cost matches facility
        self.assertEqual(economics["total_capital_cost"], 50e6)

        # Verify operational costs are accumulated
        expected_op_cost = 10000 * 5 * 365  # Daily cost * days
        self.assertAlmostEqual(
            economics["total_operational_cost"],
            expected_op_cost,
            delta=expected_op_cost * 0.1,  # 10% tolerance
        )

        # Check that discounted revenue < total revenue
        self.assertLess(
            economics["total_discounted_revenue"], economics["total_revenue"]
        )

    def test_facility_deployment_optimization(self):
        """Test ISRU facility deployment optimization."""
        resources = ["water_ice", "oxygen", "hydrogen"]
        budget = 200e6  # $200M budget

        results = self.model.optimize_facility_deployment(
            resources=resources,
            budget=budget,
            start_date=self.start_date,
            analysis_period_years=10,
        )

        self.assertIn("selected_facilities", results)
        self.assertIn("deployment_schedule", results)
        self.assertIn("total_capital_cost", results)
        self.assertIn("remaining_budget", results)
        self.assertIn("combined_economics", results)

        # Check budget constraint
        self.assertLessEqual(results["total_capital_cost"], budget)

        # Check that at least one facility was selected
        self.assertGreater(len(results["selected_facilities"]), 0)

    def test_production_forecast(self):
        """Test ISRU production forecast generation."""
        resources = ["water_ice", "oxygen"]
        forecast = create_isru_production_forecast(
            resources=resources, start_date=self.start_date, forecast_years=10
        )

        self.assertIn("water_ice", forecast)
        self.assertIn("oxygen", forecast)

        # Check water ice forecast
        water_forecast = forecast["water_ice"]
        self.assertIn("dates", water_forecast)
        self.assertIn("daily_production", water_forecast)
        self.assertIn("cumulative_production", water_forecast)
        self.assertIn("economics", water_forecast)

        # Check that production increases over time (ramp-up)
        daily_prod = water_forecast["daily_production"]
        self.assertGreater(daily_prod[-1], daily_prod[0])


class TestScenarioComparison(unittest.TestCase):
    """Test advanced scenario comparison tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.comparator = AdvancedScenarioComparator()
        self.launch_date = datetime(2030, 1, 1)

        # Create base scenario
        self.base_scenario = ScenarioDefinition(
            name="Base Case",
            description="Standard lunar mission",
            spacecraft_mass=5000,  # kg
            mission_duration_years=10,
            launch_date=self.launch_date,
            propulsion_type="chemical",
            isru_enabled=True,
            technology_readiness=3,
            financial_parameters=FinancialParameters(discount_rate=0.08, tax_rate=0.25),
            cost_overrun_factor=1.2,
            revenue_uncertainty=0.2,
            technical_risk=0.15,
            schedule_risk=0.10,
            market_risk=0.20,
        )

        # Create alternative scenarios
        self.electric_scenario = ScenarioDefinition(
            name="Electric Propulsion",
            description="Mission with electric propulsion",
            spacecraft_mass=4000,  # Less mass due to higher ISP
            mission_duration_years=12,  # Longer due to slower transit
            launch_date=self.launch_date,
            propulsion_type="electric",
            isru_enabled=True,
            technology_readiness=3,
            cost_overrun_factor=1.15,  # Lower cost risk
            technical_risk=0.20,  # Higher technical risk
            schedule_risk=0.15,
            market_risk=0.20,
        )

        self.ppp_scenario = ScenarioDefinition(
            name="Public-Private Partnership",
            description="Mission with PPP model",
            spacecraft_mass=5000,
            mission_duration_years=10,
            launch_date=self.launch_date,
            propulsion_type="chemical",
            isru_enabled=True,
            technology_readiness=3,
            public_private_partnership=True,
            commercial_payload_capacity=500,  # 500 kg commercial
            cost_overrun_factor=1.1,  # Better cost control
            technical_risk=0.10,  # Lower risk with partners
            schedule_risk=0.08,
            market_risk=0.15,
        )

    def test_scenario_addition(self):
        """Test adding scenarios to comparator."""
        self.comparator.add_scenario(self.base_scenario)
        self.assertEqual(len(self.comparator.scenarios), 1)

        self.comparator.add_scenario(self.electric_scenario)
        self.assertEqual(len(self.comparator.scenarios), 2)

    def test_single_scenario_analysis(self):
        """Test analysis of a single scenario."""
        self.comparator.add_scenario(self.base_scenario)

        results = self.comparator.analyze_scenario(
            self.base_scenario, run_monte_carlo=False  # Skip for speed
        )

        self.assertIsInstance(results, ScenarioResults)
        self.assertEqual(results.scenario_name, "Base Case")

        # Check financial metrics
        self.assertIsNotNone(results.npv)
        self.assertIsNotNone(results.irr)
        self.assertIsNotNone(results.payback_period)
        self.assertIsNotNone(results.roi)

        # Check risk metrics
        self.assertIsNotNone(results.risk_adjusted_npv)
        self.assertIsNotNone(results.probability_of_success)

        # Check performance metrics
        self.assertGreater(results.cost_per_kg_delivered, 0)
        self.assertGreater(results.revenue_per_year, 0)

        # Verify risk adjustment reduces NPV
        self.assertLess(results.risk_adjusted_npv, results.npv)

    def test_scenario_comparison(self):
        """Test comparison of multiple scenarios."""
        # Add all scenarios
        self.comparator.add_scenario(self.base_scenario)
        self.comparator.add_scenario(self.electric_scenario)
        self.comparator.add_scenario(self.ppp_scenario)

        # Compare scenarios
        comparison_df = self.comparator.compare_all_scenarios(run_monte_carlo=False)

        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 3)

        # Check required columns
        required_columns = [
            "NPV ($M)",
            "IRR (%)",
            "Payback (years)",
            "Total Cost ($M)",
            "ROI (%)",
            "Success Probability (%)",
        ]
        for col in required_columns:
            self.assertIn(col, comparison_df.columns)

    def test_scenario_ranking(self):
        """Test multi-criteria scenario ranking."""
        # Add scenarios
        self.comparator.add_scenario(self.base_scenario)
        self.comparator.add_scenario(self.electric_scenario)
        self.comparator.add_scenario(self.ppp_scenario)

        # Analyze scenarios
        self.comparator.compare_all_scenarios(run_monte_carlo=False)

        # Rank scenarios
        rankings = self.comparator.rank_scenarios()

        self.assertEqual(len(rankings), 3)

        # Check ranking structure
        for rank in rankings:
            self.assertEqual(len(rank), 2)  # (name, score)
            self.assertIsInstance(rank[0], str)
            self.assertIsInstance(rank[1], float)
            self.assertGreaterEqual(rank[1], 0.0)
            self.assertLessEqual(rank[1], 1.0)

        # Check that rankings are sorted
        scores = [r[1] for r in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_decision_matrix_generation(self):
        """Test decision matrix generation."""
        # Add scenarios
        self.comparator.add_scenario(self.base_scenario)
        self.comparator.add_scenario(self.electric_scenario)
        self.comparator.add_scenario(self.ppp_scenario)

        # Analyze scenarios
        self.comparator.compare_all_scenarios(run_monte_carlo=False)

        # Generate decision matrix
        decision_matrix = self.comparator.generate_decision_matrix()

        self.assertIsInstance(decision_matrix, pd.DataFrame)
        self.assertEqual(len(decision_matrix), 3)

        # Check score columns
        score_columns = [
            "Financial Score",
            "Technical Score",
            "Strategic Score",
            "Risk Score",
            "Complexity Score",
            "Market Score",
        ]
        for col in score_columns:
            self.assertIn(col, decision_matrix.columns)
            # Check all scores are between 0 and 1
            self.assertTrue(all(0 <= v <= 1 for v in decision_matrix[col]))

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation for uncertainty analysis."""
        self.comparator.add_scenario(self.base_scenario)

        # Run analysis with Monte Carlo
        results = self.comparator.analyze_scenario(
            self.base_scenario,
            run_monte_carlo=True,
            monte_carlo_iterations=100,  # Reduced for test speed
        )

        self.assertIsNotNone(results.monte_carlo_results)
        mc_results = results.monte_carlo_results

        # Check Monte Carlo outputs
        self.assertIn("mean_npv", mc_results)
        self.assertIn("std_npv", mc_results)
        self.assertIn("var_95", mc_results)
        self.assertIn("probability_positive", mc_results)
        self.assertIn("distribution", mc_results)

        # Check distribution size
        self.assertEqual(len(mc_results["distribution"]), 100)

        # Check VaR is populated in results
        self.assertGreater(results.value_at_risk, 0)

    def test_scenario_comparison_report(self):
        """Test comprehensive comparison report generation."""
        # Add scenarios
        self.comparator.add_scenario(self.base_scenario)
        self.comparator.add_scenario(self.electric_scenario)
        self.comparator.add_scenario(self.ppp_scenario)

        # Generate report
        report = create_scenario_comparison_report(
            self.comparator, output_format="detailed"
        )

        self.assertIn("summary", report)
        self.assertIn("comparison_table", report)
        self.assertIn("rankings", report)
        self.assertIn("decision_matrix", report)
        self.assertIn("detailed_results", report)

        # Check summary
        summary = report["summary"]
        self.assertIn("total_scenarios", summary)
        self.assertIn("best_npv_scenario", summary)
        self.assertIn("recommended_scenario", summary)

        # Check detailed results
        self.assertEqual(len(report["detailed_results"]), 3)


class TestEnhancedEconomicsIntegration(unittest.TestCase):
    """Test integration with existing economic modules."""

    def test_isru_model_integration(self):
        """Test integration of time-based ISRU with existing analyzers."""
        from src.economics.isru_benefits import ISRUBenefitAnalyzer

        # Create base analyzer
        base_analyzer = ISRUBenefitAnalyzer()

        # Create time-based model with base analyzer
        time_model = TimeBasedISRUModel(base_analyzer)

        # Verify resource properties are accessible
        self.assertIsNotNone(time_model.base_analyzer)
        self.assertIn("water_ice", time_model.base_analyzer.resource_model.resources)

    def test_financial_model_compatibility(self):
        """Test compatibility with existing financial models."""
        from src.economics.financial_models import CashFlowModel, NPVAnalyzer

        # Create scenario
        _ = ScenarioDefinition(
            name="Test",
            description="Test scenario",
            spacecraft_mass=5000,
            mission_duration_years=10,
            launch_date=datetime(2030, 1, 1),
        )

        # Create comparator
        comparator = AdvancedScenarioComparator()

        # Verify financial analyzer is properly initialized
        self.assertIsNotNone(comparator.financial_analyzer)

        # Verify it can calculate standard metrics
        cash_flow_model = CashFlowModel()
        cash_flow_model.add_cash_flow(-100e6, datetime(2030, 1, 1), "investment")
        cash_flow_model.add_cash_flow(50e6, datetime(2035, 1, 1), "revenue")

        npv = comparator.financial_analyzer.calculate_npv(
            cash_flow_model, datetime(2030, 1, 1)
        )
        self.assertIsInstance(npv, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
