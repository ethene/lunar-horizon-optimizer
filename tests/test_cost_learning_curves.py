"""Tests for learning curves and environmental costs in cost models.

Comprehensive unit tests without mocking for Wright's law learning curves
and CO₂ environmental cost calculations.

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
"""

import pytest
import math
from datetime import datetime

from src.config.costs import CostFactors
from src.economics.financial_models import FinancialParameters
from src.optimization.cost_integration import launch_price, co2_cost, CostCalculator


class TestLearningCurveFunctions:
    """Test Wright's law learning curve implementation."""

    def test_launch_price_base_year(self):
        """Test launch price calculation for base year."""
        base_price = 10000.0
        base_year = 2024

        # Same year should return base price
        result = launch_price(
            year=2024, base_price=base_price, learning_rate=0.90, base_year=base_year
        )

        assert result == base_price

    def test_launch_price_past_year(self):
        """Test launch price calculation for past year."""
        base_price = 10000.0
        base_year = 2024

        # Past year should return base price
        result = launch_price(
            year=2020, base_price=base_price, learning_rate=0.90, base_year=base_year
        )

        assert result == base_price

    def test_launch_price_future_year_reduction(self):
        """Test launch price reduction in future years."""
        base_price = 10000.0
        base_year = 2024
        learning_rate = 0.90  # 10% reduction per doubling

        # Future year should have reduced price
        result = launch_price(
            year=2030,
            base_price=base_price,
            learning_rate=learning_rate,
            base_year=base_year,
            cumulative_units_base=10,
        )

        # Price should be reduced due to learning curve
        assert result < base_price
        assert result > 0

    def test_learning_curve_mathematics(self):
        """Test Wright's law mathematical correctness."""
        base_price = 1000.0
        learning_rate = 0.80  # 20% reduction per doubling

        # Test with known values
        result = launch_price(
            year=2026,  # 2 years later
            base_price=base_price,
            learning_rate=learning_rate,
            base_year=2024,
            cumulative_units_base=100,
        )

        # With 20% annual growth for 2 years: 100 * 1.2^2 = 144 units
        # Learning exponent: ln(0.8) / ln(2) ≈ -0.322
        # Cost ratio: (144/100)^(-0.322) ≈ 0.866
        # Expected: 1000 * 0.866 ≈ 866

        expected_ratio = (144 / 100) ** (math.log(learning_rate) / math.log(2))
        expected_price = base_price * expected_ratio

        assert abs(result - expected_price) < 1.0  # Within $1

    def test_learning_rate_validation(self):
        """Test learning rate parameter validation."""
        # Test with valid learning rates
        for lr in [0.70, 0.80, 0.90, 0.95]:
            result = launch_price(
                year=2025, base_price=1000.0, learning_rate=lr, base_year=2024
            )
            assert result > 0
            assert result <= 1000.0  # Should be reduced or same

    def test_production_growth_impact(self):
        """Test impact of different production base levels."""
        base_price = 5000.0

        # Test different scenarios relative to learning curve progress
        # Scenario 1: Low base production (learning curve just starting)
        low_production_price = launch_price(
            year=2027,
            base_price=base_price,
            learning_rate=0.85,
            base_year=2024,
            cumulative_units_base=5,  # Low base production
        )

        # Scenario 2: High base production (learning curve more advanced)
        high_production_price = launch_price(
            year=2027,
            base_price=base_price,
            learning_rate=0.85,
            base_year=2022,  # Earlier base year for same target year
            cumulative_units_base=50,  # High base production
        )

        # With earlier base year and higher production, should see more cost reduction
        assert high_production_price < low_production_price


class TestCO2CostCalculation:
    """Test CO₂ environmental cost calculations."""

    def test_basic_co2_cost_calculation(self):
        """Test basic CO₂ cost calculation."""
        payload_mass = 1000.0  # kg
        co2_per_kg = 2.5  # tCO₂/kg
        price_per_ton = 50.0  # $/tCO₂

        result = co2_cost(payload_mass, co2_per_kg, price_per_ton)

        # Expected: 1000 * 2.5 * 50 = 125,000
        expected = 125000.0
        assert result == expected

    def test_co2_cost_zero_emissions(self):
        """Test CO₂ cost with zero emissions."""
        result = co2_cost(payload_mass_kg=1000.0, co2_per_kg=0.0, price_per_ton=100.0)

        assert result == 0.0

    def test_co2_cost_zero_price(self):
        """Test CO₂ cost with zero carbon price."""
        result = co2_cost(payload_mass_kg=1000.0, co2_per_kg=3.0, price_per_ton=0.0)

        assert result == 0.0

    def test_co2_cost_realistic_values(self):
        """Test CO₂ cost with realistic mission parameters."""
        # Realistic lunar mission parameters
        payload_mass = 2000.0  # kg
        co2_per_kg = 1.8  # tCO₂/kg (efficient launcher)
        price_per_ton = 75.0  # $/tCO₂ (projected 2030 price)

        result = co2_cost(payload_mass, co2_per_kg, price_per_ton)

        # Expected: 2000 * 1.8 * 75 = 270,000
        expected = 270000.0
        assert result == expected

    def test_co2_cost_high_emission_scenario(self):
        """Test CO₂ cost with high-emission launcher."""
        # High-emission scenario
        payload_mass = 500.0  # kg
        co2_per_kg = 4.0  # tCO₂/kg (less efficient launcher)
        price_per_ton = 150.0  # $/tCO₂ (high carbon price)

        result = co2_cost(payload_mass, co2_per_kg, price_per_ton)

        # Expected: 500 * 4.0 * 150 = 300,000
        expected = 300000.0
        assert result == expected


class TestCostFactorsConfiguration:
    """Test CostFactors configuration with new parameters."""

    def test_cost_factors_default_values(self):
        """Test CostFactors with default environmental and learning parameters."""
        cost_factors = CostFactors(
            launch_cost_per_kg=8000.0,
            operations_cost_per_day=75000.0,
            development_cost=800000000.0,
        )

        # Check default values
        assert cost_factors.learning_rate == 0.90
        assert cost_factors.carbon_price_per_ton_co2 == 50.0
        assert cost_factors.co2_emissions_per_kg_payload == 2.5
        assert cost_factors.environmental_compliance_factor == 1.1
        assert cost_factors.base_production_year == 2024
        assert cost_factors.cumulative_production_units == 10

    def test_cost_factors_custom_values(self):
        """Test CostFactors with custom environmental and learning parameters."""
        cost_factors = CostFactors(
            launch_cost_per_kg=12000.0,
            operations_cost_per_day=90000.0,
            development_cost=1200000000.0,
            learning_rate=0.85,
            carbon_price_per_ton_co2=75.0,
            co2_emissions_per_kg_payload=3.0,
            environmental_compliance_factor=1.15,
            base_production_year=2023,
            cumulative_production_units=15,
        )

        # Check custom values
        assert cost_factors.learning_rate == 0.85
        assert cost_factors.carbon_price_per_ton_co2 == 75.0
        assert cost_factors.co2_emissions_per_kg_payload == 3.0
        assert cost_factors.environmental_compliance_factor == 1.15
        assert cost_factors.base_production_year == 2023
        assert cost_factors.cumulative_production_units == 15

    def test_cost_factors_validation(self):
        """Test CostFactors parameter validation."""
        # Test valid ranges
        cost_factors = CostFactors(
            launch_cost_per_kg=5000.0,
            operations_cost_per_day=50000.0,
            development_cost=500000000.0,
            learning_rate=0.75,  # Valid range
            carbon_price_per_ton_co2=0.0,  # Valid minimum
            environmental_compliance_factor=1.0,  # Valid minimum
        )
        assert cost_factors.learning_rate == 0.75

        # Test invalid learning rate
        with pytest.raises(ValueError):
            CostFactors(
                launch_cost_per_kg=5000.0,
                operations_cost_per_day=50000.0,
                development_cost=500000000.0,
                learning_rate=1.5,  # Invalid: > 1.0
            )


class TestFinancialParametersIntegration:
    """Test FinancialParameters with environmental cost integration."""

    def test_financial_parameters_defaults(self):
        """Test FinancialParameters with default environmental values."""
        params = FinancialParameters()

        # Check environmental defaults
        assert params.carbon_price_per_ton_co2 == 50.0
        assert params.environmental_compliance_factor == 1.1
        assert params.learning_rate == 0.90
        assert params.mission_year == 2025

    def test_financial_parameters_total_cost(self):
        """Test total cost calculation with environmental costs."""
        params = FinancialParameters(
            carbon_price_per_ton_co2=100.0, environmental_compliance_factor=1.2
        )

        base_cost = 1000000.0  # $1M base cost
        payload_mass = 1500.0  # kg
        co2_emissions = 2.0  # tCO₂/kg

        total_cost = params.total_cost(
            base_cost=base_cost,
            payload_mass_kg=payload_mass,
            co2_emissions_per_kg=co2_emissions,
        )

        # Environmental cost: 1500 * 2.0 * 100 * 1.2 = 360,000
        expected_environmental = 1500 * 2.0 * 100.0 * 1.2
        expected_total = base_cost + expected_environmental

        assert total_cost == expected_total
        assert total_cost > base_cost  # Environmental cost added

    def test_financial_parameters_validation(self):
        """Test FinancialParameters validation."""
        # Test valid parameters
        params = FinancialParameters(
            carbon_price_per_ton_co2=0.0,  # Valid minimum
            learning_rate=0.8,  # Valid range
        )
        assert params.carbon_price_per_ton_co2 == 0.0

        # Test invalid carbon price
        with pytest.raises(ValueError):
            FinancialParameters(carbon_price_per_ton_co2=-10.0)

        # Test invalid learning rate
        with pytest.raises(ValueError):
            FinancialParameters(learning_rate=0.3)  # Too low


class TestCostCalculatorIntegration:
    """Test CostCalculator integration with learning curves and environmental costs."""

    def test_cost_calculator_initialization(self):
        """Test CostCalculator initialization with new parameters."""
        cost_factors = CostFactors(
            launch_cost_per_kg=9000.0,
            operations_cost_per_day=80000.0,
            development_cost=900000000.0,
            learning_rate=0.88,
            carbon_price_per_ton_co2=60.0,
        )

        calculator = CostCalculator(cost_factors=cost_factors, mission_year=2027)

        assert calculator.cost_factors.learning_rate == 0.88
        assert calculator.cost_factors.carbon_price_per_ton_co2 == 60.0
        assert calculator.mission_year == 2027

    def test_cost_calculator_learning_curve_integration(self):
        """Test cost calculation with learning curve adjustments."""
        cost_factors = CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1000000000.0,
            learning_rate=0.85,  # Aggressive learning
            carbon_price_per_ton_co2=50.0,
        )

        # Test with base year (no learning curve effect)
        calculator_2024 = CostCalculator(cost_factors=cost_factors, mission_year=2024)
        cost_2024 = calculator_2024.calculate_mission_cost(
            total_dv=3200.0,
            transfer_time=4.5,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
        )

        # Test with future year (learning curve effect)
        calculator_2028 = CostCalculator(cost_factors=cost_factors, mission_year=2028)
        cost_2028 = calculator_2028.calculate_mission_cost(
            total_dv=3200.0,
            transfer_time=4.5,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
        )

        # Future cost should be lower due to learning curve
        assert cost_2028 < cost_2024

    def test_cost_breakdown_new_components(self):
        """Test cost breakdown with new environmental and learning curve components."""
        cost_factors = CostFactors(
            launch_cost_per_kg=8000.0,
            operations_cost_per_day=75000.0,
            development_cost=800000000.0,
            learning_rate=0.90,
            carbon_price_per_ton_co2=75.0,
            co2_emissions_per_kg_payload=3.0,
        )

        calculator = CostCalculator(cost_factors=cost_factors, mission_year=2026)

        breakdown = calculator.calculate_cost_breakdown(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=500.0,
            moon_orbit_alt=150.0,
        )

        # Check new components exist
        assert "environmental_cost" in breakdown
        assert "learning_curve_savings" in breakdown
        assert "learning_curve_adjustment" in breakdown
        assert "environmental_fraction" in breakdown

        # Environmental cost should be positive
        assert breakdown["environmental_cost"] > 0

        # Learning curve adjustment should be <= 1.0 (cost reduction)
        assert breakdown["learning_curve_adjustment"] <= 1.0

        # Environmental fraction should be reasonable
        assert 0 <= breakdown["environmental_fraction"] <= 0.1  # Typically < 10%

    def test_environmental_cost_impact(self):
        """Test impact of different environmental cost scenarios."""
        # Low carbon price scenario
        low_carbon_factors = CostFactors(
            launch_cost_per_kg=8000.0,
            operations_cost_per_day=75000.0,
            development_cost=800000000.0,
            carbon_price_per_ton_co2=25.0,  # Low carbon price
        )

        # High carbon price scenario
        high_carbon_factors = CostFactors(
            launch_cost_per_kg=8000.0,
            operations_cost_per_day=75000.0,
            development_cost=800000000.0,
            carbon_price_per_ton_co2=150.0,  # High carbon price
        )

        low_calculator = CostCalculator(cost_factors=low_carbon_factors)
        high_calculator = CostCalculator(cost_factors=high_carbon_factors)

        # Same mission parameters
        mission_params = {
            "total_dv": 3200.0,
            "transfer_time": 4.5,
            "earth_orbit_alt": 400.0,
            "moon_orbit_alt": 100.0,
        }

        low_cost = low_calculator.calculate_mission_cost(**mission_params)
        high_cost = high_calculator.calculate_mission_cost(**mission_params)

        # High carbon price should result in higher total cost
        assert high_cost > low_cost

        # Difference should be reasonable (environmental cost impact)
        cost_difference = high_cost - low_cost
        assert cost_difference > 0
        assert cost_difference < low_cost * 0.2  # Less than 20% increase


if __name__ == "__main__":
    # Run basic functionality tests
    print("🧪 Testing Learning Curves and Environmental Costs")
    print("=" * 50)

    pytest.main([__file__, "-v", "--tb=short"])

    print("\n✅ All learning curve and environmental cost tests completed!")
