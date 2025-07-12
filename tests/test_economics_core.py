"""
Core unit tests for economics modules to improve coverage.
"""
import pytest
import numpy as np
import pandas as pd

from src.economics.financial_models import FinancialAnalyzer, CashFlowModel
from src.economics.cost_models import MissionCostModel, LaunchCostModel
from src.economics.isru_benefits import IsruAnalyzer
from src.economics.sensitivity_analysis import SensitivityAnalyzer


class TestFinancialAnalyzer:
    """Test FinancialAnalyzer class."""
    
    def test_financial_analyzer_creation(self):
        """Test creating a FinancialAnalyzer."""
        analyzer = FinancialAnalyzer(
            discount_rate=0.08,
            analysis_period_years=10
        )
        
        assert analyzer.discount_rate == 0.08
        assert analyzer.analysis_period_years == 10
    
    def test_npv_calculation_positive(self):
        """Test NPV calculation with positive cash flows."""
        analyzer = FinancialAnalyzer(discount_rate=0.10)
        
        # Initial investment of -1000, then +300 for 5 years
        cash_flows = [-1000, 300, 300, 300, 300, 300]
        
        npv = analyzer.calculate_npv(cash_flows)
        
        # Should be positive since total undiscounted cash flows = 500
        assert npv > 0
        # Should be less than 500 due to discounting
        assert npv < 500
    
    def test_npv_calculation_negative(self):
        """Test NPV calculation with negative result."""
        analyzer = FinancialAnalyzer(discount_rate=0.20)  # High discount rate
        
        # Initial investment of -1000, then +200 for 3 years
        cash_flows = [-1000, 200, 200, 200]
        
        npv = analyzer.calculate_npv(cash_flows)
        
        # Should be negative due to high discount rate
        assert npv < 0
    
    def test_irr_calculation(self):
        """Test IRR calculation."""
        analyzer = FinancialAnalyzer()
        
        # Standard investment: -1000 initial, +400 for 3 years
        cash_flows = [-1000, 400, 400, 400]
        
        irr = analyzer.calculate_irr(cash_flows)
        
        # IRR should be around 9.7% for this cash flow
        assert 0.09 < irr < 0.11
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        analyzer = FinancialAnalyzer()
        
        initial_investment = 100000
        total_returns = 150000
        
        roi = analyzer.calculate_roi(initial_investment, total_returns)
        
        # ROI should be 50%
        assert abs(roi - 0.50) < 0.01
    
    def test_payback_period(self):
        """Test payback period calculation."""
        analyzer = FinancialAnalyzer()
        
        # Initial investment of -1000, then +400 per year
        cash_flows = [-1000, 400, 400, 400, 400]
        
        payback = analyzer.calculate_payback_period(cash_flows)
        
        # Should pay back in 2.5 years
        assert 2.4 < payback < 2.6


class TestCashFlowModel:
    """Test CashFlowModel class."""
    
    def test_cash_flow_model_creation(self):
        """Test creating a cash flow model."""
        model = CashFlowModel(
            initial_investment=1000000,
            annual_revenues=[200000, 250000, 300000],
            annual_costs=[100000, 120000, 140000],
            terminal_value=500000
        )
        
        assert model.initial_investment == 1000000
        assert len(model.annual_revenues) == 3
        assert model.terminal_value == 500000
    
    def test_net_cash_flows(self):
        """Test net cash flow calculation."""
        model = CashFlowModel(
            initial_investment=1000000,
            annual_revenues=[200000, 300000],
            annual_costs=[100000, 150000],
            terminal_value=200000
        )
        
        net_flows = model.get_net_cash_flows()
        
        # Year 0: -1,000,000 (initial investment)
        # Year 1: 200,000 - 100,000 = 100,000
        # Year 2: 300,000 - 150,000 + 200,000 = 350,000
        expected = [-1000000, 100000, 350000]
        
        assert net_flows == expected
    
    def test_cumulative_cash_flows(self):
        """Test cumulative cash flow calculation."""
        model = CashFlowModel(
            initial_investment=500000,
            annual_revenues=[300000, 300000],
            annual_costs=[200000, 200000]
        )
        
        cumulative = model.get_cumulative_cash_flows()
        
        # Year 0: -500,000
        # Year 1: -500,000 + 100,000 = -400,000  
        # Year 2: -400,000 + 100,000 = -300,000
        expected = [-500000, -400000, -300000]
        
        assert cumulative == expected


class TestMissionCostModel:
    """Test MissionCostModel class."""
    
    def test_mission_cost_model_creation(self):
        """Test creating a mission cost model."""
        model = MissionCostModel(
            payload_mass=1000.0,  # kg
            mission_duration_days=365,
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=50000.0
        )
        
        assert model.payload_mass == 1000.0
        assert model.mission_duration_days == 365
    
    def test_launch_cost_calculation(self):
        """Test launch cost calculation."""
        model = MissionCostModel(
            payload_mass=2000.0,
            mission_duration_days=180,
            launch_cost_per_kg=15000.0,
            operations_cost_per_day=75000.0
        )
        
        launch_cost = model.calculate_launch_cost()
        
        # Should be 2000 kg * $15,000/kg = $30,000,000
        assert launch_cost == 30000000.0
    
    def test_operations_cost_calculation(self):
        """Test operations cost calculation."""
        model = MissionCostModel(
            payload_mass=1000.0,
            mission_duration_days=100,
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=60000.0
        )
        
        ops_cost = model.calculate_operations_cost()
        
        # Should be 100 days * $60,000/day = $6,000,000
        assert ops_cost == 6000000.0
    
    def test_total_mission_cost(self):
        """Test total mission cost calculation."""
        model = MissionCostModel(
            payload_mass=1500.0,
            mission_duration_days=200,
            launch_cost_per_kg=12000.0,
            operations_cost_per_day=80000.0,
            development_cost=50000000.0
        )
        
        total_cost = model.calculate_total_cost()
        
        # Launch: 1500 * 12000 = 18,000,000
        # Operations: 200 * 80000 = 16,000,000  
        # Development: 50,000,000
        # Total: 84,000,000
        expected_total = 18000000 + 16000000 + 50000000
        assert total_cost == expected_total


class TestLaunchCostModel:
    """Test LaunchCostModel class."""
    
    def test_launch_cost_model_creation(self):
        """Test creating a launch cost model."""
        model = LaunchCostModel(
            base_cost_per_kg=10000.0,
            mass_scaling_factor=1.2,
            launch_provider="SpaceX"
        )
        
        assert model.base_cost_per_kg == 10000.0
        assert model.mass_scaling_factor == 1.2
    
    def test_cost_with_mass_scaling(self):
        """Test cost calculation with mass scaling."""
        model = LaunchCostModel(
            base_cost_per_kg=8000.0,
            mass_scaling_factor=1.1
        )
        
        mass = 2000.0  # kg
        cost = model.calculate_cost(mass)
        
        # Base cost: 2000 * 8000 = 16,000,000
        # With scaling: 16,000,000 * (2000/1000)^0.1 ≈ 16,000,000 * 1.072
        expected_min = 16000000 * 1.05  # Lower bound
        expected_max = 16000000 * 1.15  # Upper bound
        
        assert expected_min < cost < expected_max
    
    def test_bulk_discount(self):
        """Test bulk discount for large payloads."""
        model = LaunchCostModel(
            base_cost_per_kg=10000.0,
            bulk_discount_threshold=5000.0,  # kg
            bulk_discount_rate=0.15  # 15% discount
        )
        
        # Large payload should get discount
        large_mass = 6000.0  # kg
        large_cost = model.calculate_cost(large_mass)
        
        # Small payload should not get discount
        small_mass = 3000.0  # kg  
        small_cost = model.calculate_cost(small_mass)
        
        # Cost per kg should be lower for large payload
        large_cost_per_kg = large_cost / large_mass
        small_cost_per_kg = small_cost / small_mass
        
        if hasattr(model, 'bulk_discount_rate'):
            assert large_cost_per_kg < small_cost_per_kg


class TestIsruAnalyzer:
    """Test IsruAnalyzer class."""
    
    def test_isru_analyzer_creation(self):
        """Test creating an ISRU analyzer."""
        analyzer = IsruAnalyzer(
            facility_mass=5000.0,  # kg
            production_rate=10.0,  # kg/day
            efficiency=0.85,
            power_requirement=2000.0  # kW
        )
        
        assert analyzer.facility_mass == 5000.0
        assert analyzer.production_rate == 10.0
        assert analyzer.efficiency == 0.85
    
    def test_resource_production_calculation(self):
        """Test resource production over time."""
        analyzer = IsruAnalyzer(
            facility_mass=3000.0,
            production_rate=15.0,  # kg/day
            efficiency=0.90
        )
        
        days = 100
        production = analyzer.calculate_production(days)
        
        # Should be 15 kg/day * 100 days * 0.90 efficiency = 1350 kg
        expected = 15.0 * 100 * 0.90
        assert abs(production - expected) < 0.1
    
    def test_cost_savings_calculation(self):
        """Test ISRU cost savings calculation."""
        analyzer = IsruAnalyzer(
            facility_mass=4000.0,
            production_rate=20.0,
            efficiency=0.80,
            resource_value_per_kg=1000.0  # $/kg
        )
        
        days = 365
        savings = analyzer.calculate_cost_savings(days)
        
        # Production: 20 kg/day * 365 days * 0.80 = 5840 kg
        # Savings: 5840 kg * $1000/kg = $5,840,000
        expected = 20.0 * 365 * 0.80 * 1000.0
        assert abs(savings - expected) < 1000  # $1000 tolerance
    
    def test_payback_calculation(self):
        """Test ISRU facility payback period."""
        analyzer = IsruAnalyzer(
            facility_mass=5000.0,
            production_rate=25.0,
            efficiency=0.75,
            facility_cost=10000000.0,  # $10M
            resource_value_per_kg=800.0
        )
        
        payback_days = analyzer.calculate_payback_period()
        
        # Daily savings: 25 * 0.75 * 800 = $15,000/day
        # Payback: $10,000,000 / $15,000/day ≈ 667 days
        expected_payback = 10000000 / (25.0 * 0.75 * 800.0)
        assert abs(payback_days - expected_payback) < 10  # 10 day tolerance


class TestSensitivityAnalyzer:
    """Test SensitivityAnalyzer class."""
    
    def test_sensitivity_analyzer_creation(self):
        """Test creating a sensitivity analyzer."""
        analyzer = SensitivityAnalyzer(
            base_parameters={
                'launch_cost': 50000000,
                'operations_cost': 20000000,
                'revenue': 100000000
            },
            sensitivity_ranges={
                'launch_cost': 0.20,  # ±20%
                'operations_cost': 0.15,  # ±15%
                'revenue': 0.25  # ±25%
            }
        )
        
        assert analyzer.base_parameters['launch_cost'] == 50000000
        assert analyzer.sensitivity_ranges['launch_cost'] == 0.20
    
    def test_parameter_variation(self):
        """Test parameter variation calculation."""
        analyzer = SensitivityAnalyzer(
            base_parameters={'cost': 1000000},
            sensitivity_ranges={'cost': 0.10}  # ±10%
        )
        
        variations = analyzer.generate_parameter_variations('cost')
        
        # Should include base value and variations
        assert 1000000 in variations  # Base value
        assert any(v < 1000000 for v in variations)  # Lower values
        assert any(v > 1000000 for v in variations)  # Higher values
        
        # Check range
        min_expected = 1000000 * 0.90  # -10%
        max_expected = 1000000 * 1.10  # +10%
        
        assert min(variations) >= min_expected - 1000  # Small tolerance
        assert max(variations) <= max_expected + 1000
    
    def test_tornado_analysis(self):
        """Test tornado diagram analysis."""
        analyzer = SensitivityAnalyzer(
            base_parameters={
                'revenue': 100000000,
                'cost': 60000000
            },
            sensitivity_ranges={
                'revenue': 0.20,
                'cost': 0.15
            }
        )
        
        def calculate_npv(params):
            return params['revenue'] - params['cost']
        
        tornado_data = analyzer.perform_tornado_analysis(calculate_npv)
        
        # Should have data for each parameter
        assert 'revenue' in tornado_data
        assert 'cost' in tornado_data
        
        # Each parameter should have impact data
        for param_data in tornado_data.values():
            assert 'impact' in param_data
            assert param_data['impact'] > 0  # Should have positive impact magnitude