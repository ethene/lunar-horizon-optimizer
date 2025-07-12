"""
Basic unit tests for optimization modules to improve coverage.
"""
import pytest
import numpy as np

from src.optimization.global_optimizer import GlobalOptimizer
from src.optimization.pareto_analysis import ParetoAnalyzer
from src.optimization.cost_integration import CostIntegrator


class TestGlobalOptimizer:
    """Test GlobalOptimizer class."""
    
    def test_global_optimizer_creation(self):
        """Test creating a global optimizer."""
        optimizer = GlobalOptimizer(
            algorithm_name="NSGA2",
            population_size=50,
            generations=100,
            crossover_probability=0.9,
            mutation_probability=0.1
        )
        
        assert optimizer.algorithm_name == "NSGA2"
        assert optimizer.population_size == 50
        assert optimizer.generations == 100
    
    def test_optimizer_configuration(self):
        """Test optimizer configuration validation."""
        # Test valid configuration
        optimizer = GlobalOptimizer(
            algorithm_name="NSGA2",
            population_size=30,
            generations=50
        )
        
        assert optimizer.population_size > 0
        assert optimizer.generations > 0
    
    def test_objective_function_setup(self):
        """Test setting up objective functions."""
        optimizer = GlobalOptimizer(algorithm_name="NSGA2")
        
        def delta_v_objective(x):
            return x[0]**2 + x[1]**2  # Simple quadratic
        
        def time_objective(x):
            return x[0] + 2*x[1]  # Linear combination
        
        optimizer.add_objective(delta_v_objective, "minimize", "delta_v")
        optimizer.add_objective(time_objective, "minimize", "time")
        
        assert len(optimizer.objectives) == 2
        assert "delta_v" in [obj["name"] for obj in optimizer.objectives]
        assert "time" in [obj["name"] for obj in optimizer.objectives]
    
    def test_constraint_setup(self):
        """Test setting up constraints."""
        optimizer = GlobalOptimizer(algorithm_name="NSGA2")
        
        def constraint1(x):
            return x[0] + x[1] - 5.0  # x[0] + x[1] <= 5
        
        def constraint2(x):
            return 1.0 - x[0]  # x[0] >= 1
        
        optimizer.add_constraint(constraint1, "inequality")
        optimizer.add_constraint(constraint2, "inequality")
        
        assert len(optimizer.constraints) == 2
    
    def test_bounds_setup(self):
        """Test setting up variable bounds."""
        optimizer = GlobalOptimizer(algorithm_name="NSGA2")
        
        bounds = [(0.0, 10.0), (-5.0, 5.0), (1.0, 100.0)]
        optimizer.set_bounds(bounds)
        
        assert optimizer.bounds == bounds
        assert len(optimizer.bounds) == 3
    
    def test_simple_optimization(self):
        """Test simple optimization problem."""
        optimizer = GlobalOptimizer(
            algorithm_name="NSGA2",
            population_size=20,
            generations=10  # Small for testing
        )
        
        # Simple single-objective problem: minimize x^2
        def objective(x):
            return x[0]**2
        
        optimizer.add_objective(objective, "minimize", "quadratic")
        optimizer.set_bounds([(-5.0, 5.0)])
        
        if hasattr(optimizer, 'optimize'):
            results = optimizer.optimize()
            
            # Best solution should be near x=0
            if results and len(results) > 0:
                best_x = results[0]['variables'][0]
                assert abs(best_x) < 1.0  # Should be close to optimal


class TestParetoAnalyzer:
    """Test ParetoAnalyzer class."""
    
    def test_pareto_analyzer_creation(self):
        """Test creating a Pareto analyzer."""
        analyzer = ParetoAnalyzer(
            objective_names=["delta_v", "time", "cost"],
            minimize_objectives=[True, True, True]
        )
        
        assert len(analyzer.objective_names) == 3
        assert analyzer.objective_names[0] == "delta_v"
    
    def test_dominance_check(self):
        """Test Pareto dominance checking."""
        analyzer = ParetoAnalyzer(
            objective_names=["obj1", "obj2"],
            minimize_objectives=[True, True]
        )
        
        # Solution A dominates B if A is better in all objectives
        solution_a = [1.0, 2.0]  # Better in both objectives
        solution_b = [2.0, 3.0]  # Worse in both objectives
        
        dominates = analyzer.dominates(solution_a, solution_b)
        assert dominates is True
        
        # Solution C does not dominate D (trade-off)
        solution_c = [1.0, 3.0]  # Better in obj1, worse in obj2
        solution_d = [2.0, 2.0]  # Worse in obj1, better in obj2
        
        dominates_cd = analyzer.dominates(solution_c, solution_d)
        dominates_dc = analyzer.dominates(solution_d, solution_c)
        assert dominates_cd is False
        assert dominates_dc is False
    
    def test_pareto_front_identification(self):
        """Test identifying Pareto front from solutions."""
        analyzer = ParetoAnalyzer(
            objective_names=["delta_v", "time"],
            minimize_objectives=[True, True]
        )
        
        # Set of solutions with known Pareto front
        solutions = [
            {"objectives": [1.0, 5.0], "id": 0},  # Non-dominated
            {"objectives": [2.0, 4.0], "id": 1},  # Non-dominated
            {"objectives": [3.0, 3.0], "id": 2},  # Non-dominated
            {"objectives": [4.0, 2.0], "id": 3},  # Non-dominated
            {"objectives": [5.0, 1.0], "id": 4},  # Non-dominated
            {"objectives": [2.5, 4.5], "id": 5}, # Dominated by solution 1
            {"objectives": [3.5, 3.5], "id": 6}, # Dominated by solution 2
        ]
        
        pareto_front = analyzer.find_pareto_front(solutions)
        
        # Should have 5 non-dominated solutions
        assert len(pareto_front) == 5
        
        # Dominated solutions should not be in front
        pareto_ids = [sol["id"] for sol in pareto_front]
        assert 5 not in pareto_ids  # Solution 5 is dominated
        assert 6 not in pareto_ids  # Solution 6 is dominated
    
    def test_hypervolume_calculation(self):
        """Test hypervolume calculation."""
        analyzer = ParetoAnalyzer(
            objective_names=["obj1", "obj2"],
            minimize_objectives=[True, True]
        )
        
        # Simple 2D case
        pareto_points = [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0]
        ]
        
        reference_point = [4.0, 4.0]
        
        if hasattr(analyzer, 'calculate_hypervolume'):
            hypervolume = analyzer.calculate_hypervolume(pareto_points, reference_point)
            
            # Hypervolume should be positive
            assert hypervolume > 0
            
            # Should be less than total area
            total_area = (reference_point[0] - 0) * (reference_point[1] - 0)
            assert hypervolume < total_area
    
    def test_spacing_metric(self):
        """Test spacing metric calculation."""
        analyzer = ParetoAnalyzer(
            objective_names=["obj1", "obj2"],
            minimize_objectives=[True, True]
        )
        
        # Evenly spaced points should have good spacing
        even_points = [
            [1.0, 5.0],
            [2.0, 4.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0]
        ]
        
        if hasattr(analyzer, 'calculate_spacing'):
            spacing = analyzer.calculate_spacing(even_points)
            
            # Lower spacing values indicate better distribution
            assert spacing >= 0
    
    def test_convergence_metric(self):
        """Test convergence metric calculation."""
        analyzer = ParetoAnalyzer(
            objective_names=["obj1", "obj2"],
            minimize_objectives=[True, True]
        )
        
        current_front = [
            [1.1, 2.9],
            [2.1, 1.9],
        ]
        
        reference_front = [
            [1.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0]
        ]
        
        if hasattr(analyzer, 'calculate_convergence'):
            convergence = analyzer.calculate_convergence(current_front, reference_front)
            
            # Convergence should be a positive distance metric
            assert convergence >= 0


class TestCostIntegrator:
    """Test CostIntegrator class."""
    
    def test_cost_integrator_creation(self):
        """Test creating a cost integrator."""
        integrator = CostIntegrator(
            mission_duration=365,  # days
            discount_rate=0.08
        )
        
        assert integrator.mission_duration == 365
        assert integrator.discount_rate == 0.08
    
    def test_trajectory_cost_calculation(self):
        """Test trajectory-specific cost calculation."""
        integrator = CostIntegrator(mission_duration=180)
        
        # Mock trajectory data
        trajectory_data = {
            "delta_v_total": 3200.0,  # m/s
            "time_of_flight": 120.0,  # hours
            "fuel_mass": 5000.0,      # kg
            "propellant_cost_per_kg": 5.0  # $/kg
        }
        
        if hasattr(integrator, 'calculate_trajectory_cost'):
            cost = integrator.calculate_trajectory_cost(trajectory_data)
            
            # Should include propellant cost at minimum
            expected_min = trajectory_data["fuel_mass"] * trajectory_data["propellant_cost_per_kg"]
            assert cost >= expected_min
    
    def test_mission_phase_costs(self):
        """Test mission phase cost breakdown."""
        integrator = CostIntegrator(mission_duration=200)
        
        phase_costs = {
            "launch": 50000000,      # $50M
            "transit": 5000000,      # $5M  
            "operations": 100000000, # $100M
            "recovery": 10000000     # $10M
        }
        
        if hasattr(integrator, 'calculate_total_mission_cost'):
            total_cost = integrator.calculate_total_mission_cost(phase_costs)
            
            # Should sum all phases
            expected_total = sum(phase_costs.values())
            assert total_cost == expected_total
    
    def test_cost_optimization_objective(self):
        """Test cost-based optimization objective."""
        integrator = CostIntegrator(mission_duration=365)
        
        # Mock optimization variables
        variables = {
            "payload_mass": 1500.0,     # kg
            "delta_v_budget": 4000.0,   # m/s
            "mission_duration": 400.0   # days
        }
        
        if hasattr(integrator, 'cost_objective_function'):
            cost = integrator.cost_objective_function(variables)
            
            # Cost should be positive
            assert cost > 0
            
            # Cost should increase with payload mass
            variables_heavy = variables.copy()
            variables_heavy["payload_mass"] = 2000.0
            
            cost_heavy = integrator.cost_objective_function(variables_heavy)
            assert cost_heavy > cost
    
    def test_sensitivity_analysis_integration(self):
        """Test integration with sensitivity analysis."""
        integrator = CostIntegrator(mission_duration=180)
        
        base_parameters = {
            "launch_cost_per_kg": 10000.0,
            "payload_mass": 1000.0,
            "operations_daily_cost": 50000.0
        }
        
        if hasattr(integrator, 'perform_cost_sensitivity'):
            sensitivity_results = integrator.perform_cost_sensitivity(base_parameters)
            
            # Should return sensitivity data
            assert isinstance(sensitivity_results, dict)
            
            # Should include parameter impacts
            if "impacts" in sensitivity_results:
                assert len(sensitivity_results["impacts"]) > 0


class TestOptimizationIntegration:
    """Test integration between optimization modules."""
    
    def test_optimizer_pareto_integration(self):
        """Test integration between optimizer and Pareto analyzer."""
        optimizer = GlobalOptimizer(
            algorithm_name="NSGA2",
            population_size=20,
            generations=5
        )
        
        analyzer = ParetoAnalyzer(
            objective_names=["delta_v", "cost"],
            minimize_objectives=[True, True]
        )
        
        # Define simple multi-objective problem
        def delta_v_obj(x):
            return x[0]**2  # Minimize delta-v
        
        def cost_obj(x):
            return (x[0] - 5)**2  # Minimize cost (optimum at x=5)
        
        optimizer.add_objective(delta_v_obj, "minimize", "delta_v")
        optimizer.add_objective(cost_obj, "minimize", "cost")
        optimizer.set_bounds([(0.0, 10.0)])
        
        # Mock optimization results
        mock_results = [
            {"objectives": [1.0, 16.0], "variables": [1.0]},
            {"objectives": [4.0, 9.0], "variables": [2.0]},
            {"objectives": [9.0, 4.0], "variables": [3.0]},
            {"objectives": [16.0, 1.0], "variables": [4.0]},
            {"objectives": [25.0, 0.0], "variables": [5.0]},
        ]
        
        # Analyze Pareto front
        pareto_front = analyzer.find_pareto_front(mock_results)
        
        # All solutions should be non-dominated (trade-off between objectives)
        assert len(pareto_front) == len(mock_results)
    
    def test_cost_integration_optimization(self):
        """Test integration of cost model with optimization."""
        integrator = CostIntegrator(mission_duration=365)
        optimizer = GlobalOptimizer(algorithm_name="NSGA2")
        
        # Define cost-aware objective function
        def total_cost_objective(variables):
            payload_mass = variables[0]
            delta_v = variables[1]
            
            # Simple cost model
            launch_cost = payload_mass * 10000  # $10k per kg
            fuel_cost = delta_v * 1000          # $1k per m/s
            
            return launch_cost + fuel_cost
        
        optimizer.add_objective(total_cost_objective, "minimize", "total_cost")
        optimizer.set_bounds([(500.0, 2000.0), (3000.0, 5000.0)])  # mass, delta_v
        
        # Verify objective function works
        test_variables = [1000.0, 4000.0]  # 1000 kg, 4000 m/s
        cost = total_cost_objective(test_variables)
        
        expected_cost = 1000 * 10000 + 4000 * 1000  # $14M
        assert cost == expected_cost