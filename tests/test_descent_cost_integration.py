"""Comprehensive tests for powered descent cost integration with JSON/YAML scenarios.

This module tests the complete integration of powered descent costs into the
economic framework, verifying both JSON and YAML scenario loading and ensuring
realistic, sensible results without using any mocks.
"""

import json
import tempfile
from pathlib import Path
import pytest
import yaml
import numpy as np

from src.optimization.cost_integration import CostCalculator, create_cost_calculator
from src.config.costs import CostFactors
from src.config.loader import ConfigLoader
from src.economics.financial_models import CashFlowModel, NPVAnalyzer
from datetime import datetime, timedelta


class TestDescentCostIntegration:
    """Test suite for descent cost integration functionality."""
    
    def test_basic_descent_cost_calculation(self):
        """Test basic descent cost calculation with default parameters."""
        calculator = CostCalculator()
        
        # Get breakdown without descent params to verify no descent costs
        breakdown_without = calculator.calculate_cost_breakdown(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            descent_params=None
        )
        
        # Calculate with descent params
        descent_params = {
            'thrust': 15000.0,
            'isp': 300.0,
            'burn_time': 300.0
        }
        
        breakdown_with = calculator.calculate_cost_breakdown(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            descent_params=descent_params
        )
        
        # Verify descent costs are added in breakdown
        assert breakdown_with['descent_propellant_cost'] > 0
        assert breakdown_with['lander_hardware_cost'] > 0
        assert breakdown_with['total_cost'] > breakdown_without['total_cost']
        
        # The difference should be the descent costs
        difference = breakdown_with['total_cost'] - breakdown_without['total_cost']
        descent_total = breakdown_with['descent_propellant_cost'] + breakdown_with['lander_hardware_cost']
        
        # Account for contingency on descent costs
        contingency_factor = 1 + (calculator.cost_factors.contingency_percentage / 100)
        expected_difference = descent_total * contingency_factor
        
        # Verify the difference matches expected (within rounding)
        assert abs(difference - expected_difference) < 100  # Within $100 for rounding
    
    def test_descent_cost_breakdown(self):
        """Test detailed cost breakdown includes descent components."""
        calculator = create_cost_calculator(
            propellant_unit_cost=20.0,
            lander_fixed_cost=12e6
        )
        
        descent_params = {
            'thrust': 18000.0,
            'isp': 330.0,
            'burn_time': 420.0
        }
        
        breakdown = calculator.calculate_cost_breakdown(
            total_dv=3200.0,
            transfer_time=5.5,
            earth_orbit_alt=450.0,
            moon_orbit_alt=110.0,
            descent_params=descent_params
        )
        
        # Verify all descent cost components exist
        assert 'descent_propellant_cost' in breakdown
        assert 'lander_hardware_cost' in breakdown
        assert 'descent_propellant_fraction' in breakdown
        assert 'lander_hardware_fraction' in breakdown
        
        # Verify values are sensible
        assert breakdown['descent_propellant_cost'] > 0
        assert breakdown['lander_hardware_cost'] == 12e6
        assert 0 < breakdown['descent_propellant_fraction'] < 0.1  # Less than 10%
        assert 0 < breakdown['lander_hardware_fraction'] < 0.2  # Less than 20%
        
        # Verify total includes descent costs
        total_descent = breakdown['descent_propellant_cost'] + breakdown['lander_hardware_cost']
        assert total_descent > 12e6  # At least the hardware cost
    
    def test_propellant_mass_calculation(self):
        """Test rocket equation implementation for propellant mass."""
        calculator = CostCalculator()
        
        # Test with different thrust and ISP combinations
        test_cases = [
            {'thrust': 10000.0, 'isp': 250.0, 'burn_time': 300.0},
            {'thrust': 20000.0, 'isp': 350.0, 'burn_time': 400.0},
            {'thrust': 30000.0, 'isp': 450.0, 'burn_time': 500.0},
        ]
        
        for params in test_cases:
            propellant_cost, _ = calculator._calculate_descent_costs(params)
            
            # Calculate expected propellant mass
            g = 9.81
            mass_flow_rate = params['thrust'] / (params['isp'] * g)
            expected_mass = mass_flow_rate * params['burn_time']
            
            # Propellant cost should match mass * unit cost
            expected_cost = min(max(expected_mass, 50.0), 2000.0) * calculator.propellant_unit_cost
            assert abs(propellant_cost - expected_cost) < 1.0
    
    def test_parameter_sensitivity(self):
        """Test cost sensitivity to descent parameters."""
        calculator = create_cost_calculator(
            propellant_unit_cost=25.0,
            lander_fixed_cost=15e6
        )
        
        base_params = {
            'thrust': 15000.0,
            'isp': 300.0,
            'burn_time': 300.0
        }
        
        base_cost = calculator.calculate_mission_cost(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            descent_params=base_params
        )
        
        # Test thrust variation
        high_thrust_params = base_params.copy()
        high_thrust_params['thrust'] = 25000.0
        
        high_thrust_cost = calculator.calculate_mission_cost(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            descent_params=high_thrust_params
        )
        
        # Higher thrust should increase propellant cost slightly
        assert high_thrust_cost > base_cost
        
        # Test ISP variation
        high_isp_params = base_params.copy()
        high_isp_params['isp'] = 400.0
        
        high_isp_cost = calculator.calculate_mission_cost(
            total_dv=3500.0,
            transfer_time=5.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            descent_params=high_isp_params
        )
        
        # Higher ISP should decrease propellant cost
        assert high_isp_cost < base_cost
    
    def test_json_scenario_with_descent(self):
        """Test loading and using descent parameters from JSON scenario."""
        # Create a JSON scenario with descent parameters
        scenario_json = {
            "mission": {
                "name": "Powered Descent Test Mission",
                "description": "Test mission with powered descent costs",
                "transfer_time": 5.0
            },
            "spacecraft": {
                "dry_mass": 5000.0,
                "payload_mass": 1500.0,
                "max_propellant_mass": 3500.0,
                "specific_impulse": 320.0
            },
            "costs": {
                "launch_cost_per_kg": 8000.0,
                "operations_cost_per_day": 75000.0,
                "development_cost": 800000000.0,
                "contingency_percentage": 15.0,
                "propellant_unit_cost": 20.0,
                "lander_fixed_cost": 12000000.0
            },
            "descent_parameters": {
                "thrust": 18000.0,
                "isp": 330.0,
                "burn_time": 420.0
            },
            "orbit": {
                "semi_major_axis": 384400.0,
                "eccentricity": 0.0,
                "inclination": 0.0
            }
        }
        
        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scenario_json, f, indent=2)
            temp_path = Path(f.name)
        
        try:
            # Create calculator from scenario
            cost_params = scenario_json['costs']
            calculator = create_cost_calculator(
                launch_cost_per_kg=cost_params['launch_cost_per_kg'],
                operations_cost_per_day=cost_params['operations_cost_per_day'],
                development_cost=cost_params['development_cost'],
                contingency_percentage=cost_params['contingency_percentage'],
                propellant_unit_cost=cost_params.get('propellant_unit_cost', 25.0),
                lander_fixed_cost=cost_params.get('lander_fixed_cost', 15e6)
            )
            
            # Calculate mission cost with descent parameters
            cost = calculator.calculate_mission_cost(
                total_dv=3200.0,
                transfer_time=scenario_json['mission']['transfer_time'],
                earth_orbit_alt=400.0,
                moon_orbit_alt=100.0,
                descent_params=scenario_json.get('descent_parameters')
            )
            
            # Verify reasonable cost
            assert 150e6 < cost < 250e6  # Between $150M and $250M
            
            # Verify cost breakdown
            breakdown = calculator.calculate_cost_breakdown(
                total_dv=3200.0,
                transfer_time=scenario_json['mission']['transfer_time'],
                earth_orbit_alt=400.0,
                moon_orbit_alt=100.0,
                descent_params=scenario_json.get('descent_parameters')
            )
            
            assert breakdown['lander_hardware_cost'] == 12e6
            assert breakdown['descent_propellant_cost'] > 0
            
        finally:
            temp_path.unlink()
    
    def test_yaml_scenario_with_descent(self):
        """Test loading and using descent parameters from YAML scenario."""
        # Create a YAML scenario with descent parameters
        scenario_yaml = """
mission:
  name: Powered Descent YAML Test
  description: Test mission with powered descent costs in YAML
  transfer_time: 6.0

spacecraft:
  dry_mass: 6000.0
  payload_mass: 2000.0
  max_propellant_mass: 4000.0
  specific_impulse: 340.0

costs:
  launch_cost_per_kg: 7500.0
  operations_cost_per_day: 80000.0
  development_cost: 900000000.0
  contingency_percentage: 18.0
  propellant_unit_cost: 22.0
  lander_fixed_cost: 14000000.0

descent_parameters:
  thrust: 22000.0
  isp: 350.0
  burn_time: 480.0

orbit:
  semi_major_axis: 384400.0
  eccentricity: 0.0
  inclination: 0.0
"""
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(scenario_yaml)
            temp_path = Path(f.name)
        
        try:
            # Load YAML
            with open(temp_path, 'r') as f:
                scenario_data = yaml.safe_load(f)
            
            # Create calculator from scenario
            cost_params = scenario_data['costs']
            calculator = create_cost_calculator(
                launch_cost_per_kg=cost_params['launch_cost_per_kg'],
                operations_cost_per_day=cost_params['operations_cost_per_day'],
                development_cost=cost_params['development_cost'],
                contingency_percentage=cost_params['contingency_percentage'],
                propellant_unit_cost=cost_params.get('propellant_unit_cost', 25.0),
                lander_fixed_cost=cost_params.get('lander_fixed_cost', 15e6)
            )
            
            # Calculate mission cost
            cost = calculator.calculate_mission_cost(
                total_dv=3400.0,
                transfer_time=scenario_data['mission']['transfer_time'],
                earth_orbit_alt=450.0,
                moon_orbit_alt=120.0,
                descent_params=scenario_data.get('descent_parameters')
            )
            
            # Verify reasonable cost
            assert 160e6 < cost < 280e6
            
            # Verify descent costs
            breakdown = calculator.calculate_cost_breakdown(
                total_dv=3400.0,
                transfer_time=scenario_data['mission']['transfer_time'],
                earth_orbit_alt=450.0,
                moon_orbit_alt=120.0,
                descent_params=scenario_data.get('descent_parameters')
            )
            
            assert breakdown['lander_hardware_cost'] == 14e6
            assert breakdown['descent_propellant_cost'] > 0
            
            # Higher thrust and burn time should mean more propellant
            assert breakdown['descent_propellant_cost'] > 30000  # At least $30k
            
        finally:
            temp_path.unlink()
    
    def test_edge_cases_and_bounds(self):
        """Test edge cases and parameter bounds."""
        calculator = CostCalculator()
        
        # Test with minimum bounds
        min_params = {
            'thrust': 1000.0,  # Very low thrust
            'isp': 200.0,      # Low ISP
            'burn_time': 60.0  # Short burn
        }
        
        propellant_cost, hardware_cost = calculator._calculate_descent_costs(min_params)
        
        # Should still return valid costs
        assert propellant_cost >= 50.0 * calculator.propellant_unit_cost  # Min mass bound
        assert hardware_cost == calculator.lander_fixed_cost
        
        # Test with maximum bounds
        max_params = {
            'thrust': 100000.0,  # Very high thrust
            'isp': 500.0,        # High ISP
            'burn_time': 1800.0  # 30 minute burn
        }
        
        propellant_cost, hardware_cost = calculator._calculate_descent_costs(max_params)
        
        # Should be capped at maximum
        assert propellant_cost <= 2000.0 * calculator.propellant_unit_cost  # Max mass bound
        assert hardware_cost == calculator.lander_fixed_cost
    
    def test_financial_integration_with_descent(self):
        """Test integration with financial models including descent costs."""
        # Create cost calculator with descent parameters
        calculator = create_cost_calculator(
            launch_cost_per_kg=9000.0,
            operations_cost_per_day=85000.0,
            development_cost=850000000.0,
            contingency_percentage=20.0,
            propellant_unit_cost=24.0,
            lander_fixed_cost=13500000.0
        )
        
        # Calculate mission cost with descent
        descent_params = {
            'thrust': 20000.0,
            'isp': 340.0,
            'burn_time': 450.0
        }
        
        total_cost = calculator.calculate_mission_cost(
            total_dv=3300.0,
            transfer_time=5.5,
            earth_orbit_alt=420.0,
            moon_orbit_alt=105.0,
            descent_params=descent_params
        )
        
        # Create cash flow model
        cash_flow_model = CashFlowModel()
        
        # Add development costs
        start_date = datetime.now()
        cash_flow_model.add_development_costs(
            total_cost=850000000.0,
            start_date=start_date,
            duration_months=36
        )
        
        # Add launch costs (including descent hardware)
        launch_date = start_date + timedelta(days=1095)  # 3 years later
        cash_flow_model.add_launch_costs(
            cost_per_launch=total_cost,
            launch_dates=[launch_date]
        )
        
        # Add revenue stream
        revenue_start = launch_date + timedelta(days=30)
        cash_flow_model.add_revenue_stream(
            monthly_revenue=15000000.0,
            start_date=revenue_start,
            duration_months=60
        )
        
        # Calculate NPV
        npv_analyzer = NPVAnalyzer()
        npv = npv_analyzer.calculate_npv(cash_flow_model)
        
        # NPV should be reasonable
        assert isinstance(npv, float)
        assert -2e9 < npv < 2e9  # Between -$2B and $2B (wider range for large projects)
        
        # Calculate payback period
        payback = npv_analyzer.calculate_payback_period(cash_flow_model)
        
        # Should have a reasonable payback period or infinite if not profitable
        if payback != float('inf'):
            assert 3.0 < payback < 15.0  # Between 3 and 15 years
    
    def test_multi_mission_descent_costs(self):
        """Test descent costs in multi-mission scenarios."""
        calculator = create_cost_calculator(
            propellant_unit_cost=21.0,
            lander_fixed_cost=11000000.0
        )
        
        # Define three missions with different descent profiles
        missions = [
            {
                'dv': 3100.0,
                'time': 5.0,
                'descent': {'thrust': 12000.0, 'isp': 300.0, 'burn_time': 360.0}
            },
            {
                'dv': 3300.0,
                'time': 5.5,
                'descent': {'thrust': 18000.0, 'isp': 320.0, 'burn_time': 420.0}
            },
            {
                'dv': 3500.0,
                'time': 6.0,
                'descent': {'thrust': 24000.0, 'isp': 340.0, 'burn_time': 480.0}
            }
        ]
        
        total_constellation_cost = 0.0
        total_descent_cost = 0.0
        
        for i, mission in enumerate(missions):
            cost = calculator.calculate_mission_cost(
                total_dv=mission['dv'],
                transfer_time=mission['time'],
                earth_orbit_alt=400.0 + i * 50,
                moon_orbit_alt=100.0 + i * 10,
                descent_params=mission['descent']
            )
            
            breakdown = calculator.calculate_cost_breakdown(
                total_dv=mission['dv'],
                transfer_time=mission['time'],
                earth_orbit_alt=400.0 + i * 50,
                moon_orbit_alt=100.0 + i * 10,
                descent_params=mission['descent']
            )
            
            total_constellation_cost += cost
            total_descent_cost += (breakdown['descent_propellant_cost'] + 
                                   breakdown['lander_hardware_cost'])
        
        # Verify total costs are reasonable
        assert total_constellation_cost > 300e6  # More than $300M for 3 missions
        assert total_descent_cost > 30e6  # At least 3 landers worth ($33M)
        
        # Descent should be significant but not dominant fraction
        descent_fraction = total_descent_cost / total_constellation_cost
        assert 0.03 < descent_fraction < 0.35  # Between 3% and 35% (wider range)


class TestDocumentationAndExamples:
    """Test documentation and example scenarios."""
    
    def test_create_example_scenarios(self):
        """Create and verify example scenario files."""
        # Example JSON scenario
        json_scenario = {
            "scenario_type": "powered_descent_mission",
            "version": "1.0",
            "metadata": {
                "created_by": "Lunar Horizon Optimizer",
                "description": "Example scenario demonstrating powered descent cost integration"
            },
            "mission": {
                "name": "Artemis Cargo Lander",
                "description": "Commercial cargo delivery to lunar surface with powered descent",
                "transfer_time": 5.2,
                "launch_date": "2026-03-15"
            },
            "spacecraft": {
                "dry_mass": 7500.0,
                "payload_mass": 2500.0,
                "max_propellant_mass": 5000.0,
                "specific_impulse": 325.0
            },
            "costs": {
                "launch_cost_per_kg": 6500.0,
                "operations_cost_per_day": 65000.0,
                "development_cost": 750000000.0,
                "contingency_percentage": 25.0,
                "propellant_unit_cost": 18.0,
                "lander_fixed_cost": 10000000.0,
                "learning_rate": 0.85,
                "carbon_price_per_ton_co2": 75.0
            },
            "descent_parameters": {
                "thrust": 16000.0,
                "isp": 315.0,
                "burn_time": 380.0,
                "landing_site": "Shackleton Crater Rim",
                "guidance_mode": "terrain_relative_navigation"
            },
            "orbit": {
                "semi_major_axis": 1837.4,
                "eccentricity": 0.0,
                "inclination": 90.0,
                "raan": 0.0,
                "argument_of_periapsis": 0.0,
                "true_anomaly": 0.0
            },
            "optimization": {
                "objectives": ["minimize_cost", "minimize_risk"],
                "constraints": {
                    "max_g_load": 4.0,
                    "min_landing_accuracy": 100.0,
                    "max_slope_angle": 15.0
                }
            }
        }
        
        # Verify the scenario is valid
        assert json_scenario['descent_parameters']['thrust'] > 0
        assert json_scenario['descent_parameters']['isp'] > 0
        assert json_scenario['descent_parameters']['burn_time'] > 0
        assert json_scenario['costs']['propellant_unit_cost'] > 0
        assert json_scenario['costs']['lander_fixed_cost'] > 0
        
        # Example YAML scenario
        yaml_scenario = """
# Lunar Horizon Optimizer - Powered Descent Scenario Example
scenario_type: powered_descent_mission
version: '1.0'

metadata:
  created_by: Lunar Horizon Optimizer
  description: YAML example for powered descent cost analysis
  tags:
    - powered_descent
    - commercial_lunar
    - cost_optimization

mission:
  name: Blue Origin Lunar Cargo
  description: >
    Commercial lunar cargo mission using Blue Moon lander
    with emphasis on reusability and cost efficiency
  transfer_time: 4.8  # days
  launch_date: 2027-09-20
  mission_type: cargo_delivery

spacecraft:
  dry_mass: 8200.0      # kg - Blue Moon derivative
  payload_mass: 3300.0   # kg - Increased cargo capacity
  max_propellant_mass: 6500.0  # kg
  specific_impulse: 335.0      # s - BE-7 engine performance

costs:
  # Launch costs
  launch_cost_per_kg: 5500.0   # $/kg - New Glenn launch
  
  # Operations
  operations_cost_per_day: 55000.0  # $/day
  
  # Development (amortized)
  development_cost: 650000000.0  # $650M program cost
  
  # Risk factors
  contingency_percentage: 22.0   # %
  
  # Descent-specific costs
  propellant_unit_cost: 16.0     # $/kg - Liquid methane/LOX
  lander_fixed_cost: 8500000.0   # $8.5M - Reusable lander
  
  # Environmental factors
  learning_rate: 0.88            # Wright's law learning
  carbon_price_per_ton_co2: 80.0 # $/tCO2
  co2_emissions_per_kg_payload: 1.2  # tCO2/kg

descent_parameters:
  # BE-7 engine cluster configuration
  thrust: 24000.0       # N - 24 kN total thrust
  isp: 345.0           # s - Vacuum specific impulse
  burn_time: 420.0     # s - 7 minute powered descent
  
  # Landing profile
  landing_site: Mare Imbrium
  landing_accuracy: 50.0  # meters
  
  # Guidance parameters
  guidance_mode: precision_landing
  hazard_avoidance: true
  throttle_range:
    min: 0.2  # 20% minimum throttle
    max: 1.0  # 100% maximum throttle

orbit:
  # Near-rectilinear halo orbit (NRHO)
  semi_major_axis: 7152.0  # km
  eccentricity: 0.95
  inclination: 57.0        # degrees
  raan: 0.0               # degrees
  argument_of_periapsis: 180.0  # degrees
  true_anomaly: 0.0       # degrees

optimization:
  objectives:
    - minimize_total_cost
    - maximize_payload_fraction
    - minimize_landing_error
    
  constraints:
    max_deceleration: 3.5    # g
    min_hover_time: 30.0     # seconds
    max_descent_angle: 70.0  # degrees
    min_fuel_margin: 0.05    # 5% reserve
    
  algorithm:
    name: NSGA-II
    population_size: 100
    generations: 50

# Economic analysis parameters
economics:
  discount_rate: 0.06      # 6% annual
  inflation_rate: 0.025    # 2.5% annual
  project_duration: 10     # years
  
  revenue_model:
    cargo_delivery_rate: 2500000.0  # $/metric ton
    annual_missions: 4              # missions/year
    
  market_growth:
    initial_demand: 10.0    # metric tons/year
    growth_rate: 0.15       # 15% annual growth
"""
        
        # Parse and verify YAML
        yaml_data = yaml.safe_load(yaml_scenario)
        
        assert yaml_data['descent_parameters']['thrust'] == 24000.0
        assert yaml_data['descent_parameters']['isp'] == 345.0
        assert yaml_data['costs']['propellant_unit_cost'] == 16.0
        assert yaml_data['costs']['lander_fixed_cost'] == 8500000.0
        
        # Both scenarios should produce similar cost structures
        for scenario_data in [json_scenario, yaml_data]:
            cost_params = scenario_data['costs']
            calculator = create_cost_calculator(
                launch_cost_per_kg=cost_params['launch_cost_per_kg'],
                operations_cost_per_day=cost_params['operations_cost_per_day'],
                development_cost=cost_params['development_cost'],
                contingency_percentage=cost_params['contingency_percentage'],
                propellant_unit_cost=cost_params['propellant_unit_cost'],
                lander_fixed_cost=cost_params['lander_fixed_cost']
            )
            
            cost = calculator.calculate_mission_cost(
                total_dv=3250.0,
                transfer_time=scenario_data['mission']['transfer_time'],
                earth_orbit_alt=400.0,
                moon_orbit_alt=100.0,
                descent_params=scenario_data['descent_parameters']
            )
            
            # Both should produce reasonable costs
            assert 100e6 < cost < 300e6


def test_real_world_validation():
    """Validate against real-world mission cost estimates."""
    # Based on public information about lunar lander costs
    # Apollo LM: ~$2B in today's dollars for development
    # Modern estimates: $100-200M per lander
    
    calculator = create_cost_calculator(
        launch_cost_per_kg=7000.0,      # SpaceX/Blue Origin era
        operations_cost_per_day=100000.0,
        development_cost=2000000000.0,   # $2B development like Apollo
        contingency_percentage=30.0,     # Higher for lunar missions
        propellant_unit_cost=20.0,       # Modern propellant costs
        lander_fixed_cost=150000000.0   # $150M per lander (modern estimate)
    )
    
    # Typical lunar descent profile
    descent_params = {
        'thrust': 45000.0,   # 45 kN (similar to Apollo LM descent stage)
        'isp': 311.0,        # Hypergolic propellants
        'burn_time': 720.0   # 12 minutes (Apollo was ~12 minutes)
    }
    
    cost = calculator.calculate_mission_cost(
        total_dv=3800.0,     # Typical Earth-Moon transfer
        transfer_time=4.0,   # 4 days
        earth_orbit_alt=300.0,
        moon_orbit_alt=100.0,
        descent_params=descent_params
    )
    
    breakdown = calculator.calculate_cost_breakdown(
        total_dv=3800.0,
        transfer_time=4.0,
        earth_orbit_alt=300.0,
        moon_orbit_alt=100.0,
        descent_params=descent_params
    )
    
    # Validate against known ranges
    assert 300e6 < cost < 600e6  # $300-600M range for modern lunar mission
    
    # Lander should be significant portion
    lander_fraction = breakdown['lander_hardware_fraction']
    assert 0.2 < lander_fraction < 0.5  # 20-50% of total cost
    
    # Descent propellant should be smaller fraction
    propellant_fraction = breakdown['descent_propellant_fraction']
    assert propellant_fraction < 0.01  # Less than 1% for propellant alone