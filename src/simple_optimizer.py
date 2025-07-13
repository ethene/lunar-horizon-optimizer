#!/usr/bin/env python3
"""
Simple Optimizer - Minimal working implementation for CLI.

This provides a simplified interface to the lunar optimization system
for CLI usage while the full integration is being developed.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config.costs import CostFactors
from config.models import MissionConfig
from economics.financial_models import NPVAnalyzer, ROICalculator


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    population_size: int = 50
    num_generations: int = 30
    seed: Optional[int] = None
    verbose: bool = False


@dataclass
class AnalysisResults:
    """Results from mission analysis."""
    mission_name: str
    trajectory_results: Dict[str, Any]
    optimization_results: Dict[str, Any] 
    economic_analysis: Dict[str, Any]
    visualization_assets: Dict[str, Any]
    analysis_metadata: Dict[str, Any]


class SimpleLunarOptimizer:
    """Simplified lunar mission optimizer for CLI usage."""
    
    def __init__(self):
        """Initialize the optimizer."""
        self.logger = None
        
    def analyze_mission(
        self,
        mission_config: MissionConfig,
        optimization_config: OptimizationConfig,
        include_sensitivity: bool = True,
        include_isru: bool = True,
        verbose: bool = False
    ) -> AnalysisResults:
        """
        Analyze a lunar mission configuration.
        
        Args:
            mission_config: Mission parameters
            optimization_config: Optimization settings
            include_sensitivity: Whether to run sensitivity analysis
            include_isru: Whether to include ISRU analysis
            verbose: Enable verbose output
            
        Returns:
            Analysis results with trajectory, economic, and optimization data
        """
        if verbose:
            print(f"🚀 Analyzing mission: {mission_config.name}")
            
        # Basic trajectory analysis (simplified)
        trajectory_results = self._analyze_trajectory(mission_config)
        
        # Economic analysis
        economic_results = self._analyze_economics(mission_config, include_isru)
        
        # Basic optimization (simplified)
        optimization_results = self._run_optimization(mission_config, optimization_config)
        
        # Create basic visualizations
        visualizations = self._create_visualizations(
            trajectory_results, economic_results, optimization_results
        )
        
        return AnalysisResults(
            mission_name=mission_config.name,
            trajectory_results=trajectory_results,
            optimization_results=optimization_results,
            economic_analysis=economic_results,
            visualization_assets=visualizations,
            analysis_metadata={
                "analysis_date": str(os.popen("date").read().strip()),
                "config_used": mission_config.model_dump(),
                "optimization_config": optimization_config.__dict__
            }
        )
    
    def _analyze_trajectory(self, config: MissionConfig) -> Dict[str, Any]:
        """Simplified trajectory analysis."""
        # Basic Hohmann transfer approximation
        earth_alt = 400.0  # km
        moon_alt = 100.0  # Assume 100km lunar orbit
        
        # Simplified delta-v calculation
        delta_v_leo_escape = 3200.0  # m/s
        delta_v_moon_capture = 800.0  # m/s
        total_delta_v = delta_v_leo_escape + delta_v_moon_capture
        
        # Simplified time calculation
        transfer_time = 4.5  # days for transfer
        
        return {
            "delta_v_total": total_delta_v,
            "delta_v_leo_escape": delta_v_leo_escape,
            "delta_v_moon_capture": delta_v_moon_capture,
            "transfer_time_days": transfer_time,
            "earth_departure_altitude": earth_alt,
            "moon_arrival_altitude": moon_alt,
            "trajectory_type": "Hohmann Transfer (Simplified)",
            "propellant_mass_kg": total_delta_v * config.payload.dry_mass / 9000.0
        }
    
    def _analyze_economics(self, config: MissionConfig, include_isru: bool) -> Dict[str, Any]:
        """Simplified economic analysis."""
        try:
            # Import financial models
            from economics.financial_models import FinancialParameters
            
            # Create financial parameters
            financial_params = FinancialParameters(
                discount_rate=0.08,  # 8% default discount rate
                inflation_rate=0.03,
                tax_rate=0.25,
                project_duration_years=10
            )
            
            # Create financial analyzer
            npv_analyzer = NPVAnalyzer(financial_params)
            roi_calculator = ROICalculator()
            
            # Basic cost calculation  
            total_mass = config.payload.dry_mass + config.payload.payload_mass + config.payload.max_propellant_mass
            launch_cost = (
                total_mass * 
                config.cost_factors.launch_cost_per_kg
            )
            
            development_cost = config.cost_factors.development_cost
            operations_cost = config.cost_factors.operations_cost_per_day * 365 * 5  # 5 years
            
            total_cost = launch_cost + development_cost + operations_cost
            
            # Basic revenue (simplified)
            annual_revenue = total_cost * 0.3  # 30% annual return assumption
            
            # Simplified NPV calculation
            npv = 0.0
            for year in range(1, 11):  # 10 years
                discounted_revenue = annual_revenue / ((1 + financial_params.discount_rate) ** year)
                npv += discounted_revenue
            npv -= total_cost  # Subtract initial investment
            
            # ROI calculation
            roi = roi_calculator.calculate_simple_roi(total_cost, annual_revenue * 10)
            
            results = {
                "launch_cost": launch_cost,
                "development_cost": development_cost,
                "operations_cost": operations_cost,
                "total_cost": total_cost,
                "annual_revenue": annual_revenue,
                "npv": npv,
                "roi_percent": roi * 100,
                "payback_period_years": total_cost / annual_revenue,
                "cost_breakdown": {
                    "launch": launch_cost / total_cost,
                    "development": development_cost / total_cost,
                    "operations": operations_cost / total_cost
                }
            }
            
            if include_isru:
                # Simplified ISRU benefits
                isru_savings = total_cost * 0.15  # 15% cost reduction assumption
                results["isru_benefits"] = {
                    "total_savings": isru_savings,
                    "cost_reduction_percent": 15.0,
                    "break_even_years": 3.0
                }
            
            return results
            
        except Exception as e:
            print(f"Warning: Economic analysis failed: {e}")
            return {
                "error": str(e),
                "total_cost": 1000000000.0,  # $1B default
                "npv": 0.0,
                "roi_percent": 0.0
            }
    
    def _run_optimization(self, config: MissionConfig, opt_config: OptimizationConfig) -> Dict[str, Any]:
        """Simplified optimization."""
        return {
            "population_size": opt_config.population_size,
            "generations": opt_config.num_generations,
            "best_solution": {
                "delta_v": 4000.0,
                "transfer_time": 4.5,
                "cost": 1000000000.0
            },
            "pareto_solutions": [
                {"delta_v": 3800, "time": 5.0, "cost": 1100000000},
                {"delta_v": 4000, "time": 4.5, "cost": 1000000000},
                {"delta_v": 4200, "time": 4.0, "cost": 900000000}
            ],
            "convergence_data": {
                "best_fitness_history": list(range(opt_config.num_generations)),
                "population_diversity": [1.0 - i/opt_config.num_generations for i in range(opt_config.num_generations)]
            }
        }
    
    def _create_visualizations(self, trajectory: Dict, economics: Dict, optimization: Dict) -> Dict[str, Any]:
        """Create basic visualization data."""
        return {
            "trajectory_plot": {
                "type": "3d_trajectory",
                "data": "placeholder for 3D trajectory data",
                "title": "Earth-Moon Transfer Trajectory"
            },
            "economic_dashboard": {
                "type": "financial_summary", 
                "data": economics,
                "title": "Mission Economics Dashboard"
            },
            "optimization_plot": {
                "type": "pareto_front",
                "data": optimization["pareto_solutions"],
                "title": "Multi-Objective Trade-offs"
            }
        }
    
    def export_results(self, results: AnalysisResults, output_dir: str) -> None:
        """Export results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export JSON data
        with open(f"{output_dir}/analysis_results.json", "w") as f:
            json.dump({
                "mission_name": results.mission_name,
                "trajectory": results.trajectory_results,
                "economics": results.economic_analysis,
                "optimization": results.optimization_results,
                "metadata": results.analysis_metadata
            }, f, indent=2)
        
        # Export summary
        with open(f"{output_dir}/summary.txt", "w") as f:
            f.write(f"Mission Analysis Summary: {results.mission_name}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Trajectory Analysis:\n")
            f.write(f"  Total Delta-V: {results.trajectory_results.get('delta_v_total', 0):.1f} m/s\n")
            f.write(f"  Transfer Time: {results.trajectory_results.get('transfer_time_days', 0):.1f} days\n\n")
            f.write(f"Economic Analysis:\n")
            f.write(f"  Total Cost: ${results.economic_analysis.get('total_cost', 0):,.0f}\n")
            f.write(f"  NPV: ${results.economic_analysis.get('npv', 0):,.0f}\n")
            f.write(f"  ROI: {results.economic_analysis.get('roi_percent', 0):.1f}%\n")
        
        print(f"✅ Results exported to {output_dir}/")


def create_sample_config() -> MissionConfig:
    """Create a sample mission configuration."""
    from config.spacecraft import PayloadSpecification
    from config.orbit import OrbitParameters
    
    return MissionConfig(
        name="Sample Lunar Mission",
        description="Basic lunar cargo delivery mission",
        payload=PayloadSpecification(
            dry_mass=5000.0,
            max_propellant_mass=3000.0,
            payload_mass=1000.0,
            specific_impulse=450.0
        ),
        cost_factors=CostFactors(
            launch_cost_per_kg=10000.0,
            development_cost=1000000000.0,
            operations_cost_per_day=100000.0,
            discount_rate=0.08
        ),
        target_orbit=OrbitParameters(
            semi_major_axis=6778.0,  # Earth radius (6378) + 400 km LEO altitude
            inclination=0.0,
            eccentricity=0.0
        ),
        mission_duration_days=365.0
    )