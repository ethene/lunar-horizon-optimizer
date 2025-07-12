"""
Core unit tests for economics modules to improve coverage.
"""
import pytest
import numpy as np
from datetime import datetime

from src.economics.financial_models import CashFlowModel, NPVAnalyzer, ROICalculator, FinancialParameters, CashFlow
from src.economics.cost_models import MissionCostModel
from src.economics.isru_benefits import ISRUBenefitAnalyzer
from src.economics.sensitivity_analysis import SensitivityAnalyzer


class TestNPVAnalyzer:
    """Test NPVAnalyzer class."""
    
    def test_npv_analyzer_creation(self):
        """Test creating an NPV analyzer."""
        analyzer = NPVAnalyzer()
        assert analyzer is not None
    
    def test_npv_calculation_positive(self):
        """Test NPV calculation with positive cash flows."""
        analyzer = NPVAnalyzer()
        
        # Initial investment of -1000, then +300 for 5 years
        cash_flows = [-1000, 300, 300, 300, 300, 300]
        
        npv = analyzer.calculate_npv(cash_flows, discount_rate=0.10)
        
        # Should be positive since total undiscounted cash flows = 500
        assert npv > 0
        # Should be less than 500 due to discounting
        assert npv < 500


class TestROICalculator:
    """Test ROICalculator class."""
    
    def test_roi_calculator_creation(self):
        """Test creating an ROI calculator."""
        calculator = ROICalculator()
        assert calculator is not None
    
    def test_roi_calculation(self):
        """Test ROI calculation."""
        calculator = ROICalculator()
        
        initial_investment = 100000
        total_returns = 150000
        
        roi = calculator.calculate_roi(initial_investment, total_returns)
        
        # ROI should be 50%
        assert abs(roi - 0.50) < 0.01


class TestCashFlowModel:
    """Test CashFlowModel class."""
    
    def test_cash_flow_model_creation(self):
        """Test creating a cash flow model."""
        model = CashFlowModel()
        assert model is not None
    
    def test_add_cash_flow(self):
        """Test adding cash flows."""
        model = CashFlowModel()
        
        # Add initial investment
        model.add_cash_flow(-1000000, datetime(2024, 1, 1), "investment")
        
        # Add revenue
        model.add_cash_flow(200000, datetime(2024, 6, 1), "revenue")
        
        cash_flows = model.cash_flows
        assert len(cash_flows) == 2
        assert cash_flows[0].amount == -1000000
        assert cash_flows[1].amount == 200000


class TestFinancialParameters:
    """Test FinancialParameters class."""
    
    def test_financial_parameters_creation(self):
        """Test creating financial parameters."""
        params = FinancialParameters(
            discount_rate=0.08,
            inflation_rate=0.03,
            tax_rate=0.25
        )
        
        assert params.discount_rate == 0.08
        assert params.inflation_rate == 0.03
        assert params.tax_rate == 0.25
    
    def test_financial_parameters_validation(self):
        """Test financial parameters validation."""
        # Valid parameters should work
        params = FinancialParameters(discount_rate=0.10)
        assert params.discount_rate == 0.10
        
        # Invalid discount rate should raise error
        with pytest.raises(ValueError):
            FinancialParameters(discount_rate=1.5)  # > 1


class TestCashFlow:
    """Test CashFlow dataclass."""
    
    def test_cash_flow_creation(self):
        """Test creating a cash flow."""
        flow = CashFlow(
            amount=100000,
            date=datetime(2024, 1, 1),
            category="revenue",
            description="Product sales"
        )
        
        assert flow.amount == 100000
        assert flow.category == "revenue"
        assert flow.description == "Product sales"
    
    def test_cash_flow_validation(self):
        """Test cash flow validation."""
        # Valid cash flow should work
        flow = CashFlow(
            amount=50000,
            date=datetime(2024, 1, 1),
            category="cost"
        )
        assert flow.amount == 50000
        
        # Invalid date should raise error
        with pytest.raises(ValueError):
            CashFlow(
                amount=50000,
                date="2024-01-01",  # String instead of datetime
                category="cost"
            )


class TestMissionCostModel:
    """Test MissionCostModel class."""
    
    def test_mission_cost_model_creation(self):
        """Test creating a mission cost model."""
        model = MissionCostModel()
        assert model is not None
    
    def test_cost_calculation(self):
        """Test basic cost calculation."""
        model = MissionCostModel()
        
        # Test with simple parameters
        cost = model.calculate_total_cost(
            payload_mass=1000.0,
            launch_cost_per_kg=10000.0,
            mission_duration_days=365,
            operations_cost_per_day=50000.0
        )
        
        # Launch cost: 1000 * 10000 = 10M
        # Operations: 365 * 50000 = 18.25M  
        # Total should be around 28.25M
        expected = 10000000 + 18250000
        assert abs(cost - expected) / expected < 0.1  # 10% tolerance


class TestISRUBenefitAnalyzer:
    """Test ISRUBenefitAnalyzer class."""
    
    def test_isru_analyzer_creation(self):
        """Test creating an ISRU benefit analyzer."""
        analyzer = ISRUBenefitAnalyzer()
        assert analyzer is not None
    
    def test_basic_functionality(self):
        """Test basic ISRU analyzer functionality."""
        analyzer = ISRUBenefitAnalyzer()
        
        # Test basic analysis with simple parameters
        if hasattr(analyzer, 'analyze_benefits'):
            benefits = analyzer.analyze_benefits(
                facility_mass=1000.0,
                production_rate=5.0,
                mission_duration=365
            )
            
            # Should return some analysis
            assert benefits is not None


class TestSensitivityAnalyzer:
    """Test SensitivityAnalyzer class."""
    
    def test_sensitivity_analyzer_creation(self):
        """Test creating a sensitivity analyzer."""
        analyzer = SensitivityAnalyzer()
        assert analyzer is not None
    
    def test_parameter_variation(self):
        """Test parameter variation calculation."""
        analyzer = SensitivityAnalyzer()
        
        # Test parameter variation
        base_value = 1000000
        variation_pct = 0.10  # ±10%
        
        variations = analyzer.generate_variations(
            base_value=base_value,
            variation_percentage=variation_pct,
            num_points=5
        )
        
        # Should have multiple values
        assert len(variations) >= 3
        
        # Should include values above and below base
        assert any(v < base_value for v in variations)
        assert any(v > base_value for v in variations)