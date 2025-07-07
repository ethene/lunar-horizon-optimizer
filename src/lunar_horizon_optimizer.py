#!/usr/bin/env python3
"""
Lunar Horizon Optimizer - Main Integration Module
=================================================

This is the main entry point for the Lunar Horizon Optimizer system, providing
a unified interface that integrates trajectory generation, multi-objective 
optimization, economic analysis, and interactive visualization.

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0-rc1
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
from dataclasses import dataclass
import json

# Task 3: Trajectory Generation
from trajectory.lunar_transfer import LunarTransfer
from trajectory.earth_moon_trajectories import generate_earth_moon_trajectory
from trajectory.transfer_window_analysis import TrajectoryWindowAnalyzer

# Task 4: Global Optimization  
from optimization.global_optimizer import GlobalOptimizer, LunarMissionProblem
from optimization.pareto_analysis import ParetoAnalyzer

# Task 5: Economic Analysis
from economics.financial_models import CashFlowModel, NPVAnalyzer, FinancialParameters
from economics.cost_models import MissionCostModel
from economics.isru_benefits import ISRUBenefitAnalyzer
from economics.sensitivity_analysis import EconomicSensitivityAnalyzer
from economics.reporting import EconomicReporter, FinancialSummary

# Task 6: Visualization
from visualization.dashboard import ComprehensiveDashboard, MissionAnalysisData
from visualization.trajectory_visualization import TrajectoryVisualizer
from visualization.optimization_visualization import OptimizationVisualizer
from visualization.economic_visualization import EconomicVisualizer

# Configuration
from config.mission_config import MissionConfig
from config.costs import CostFactors
from config.spacecraft import SpacecraftConfig

# Utilities
from utils.unit_conversions import km_to_m, days_to_seconds
from constants import PhysicalConstants

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters."""
    population_size: int = 100
    num_generations: int = 100
    seed: Optional[int] = 42
    min_earth_alt: float = 200.0
    max_earth_alt: float = 1000.0
    min_moon_alt: float = 50.0
    max_moon_alt: float = 500.0
    min_transfer_time: float = 3.0
    max_transfer_time: float = 10.0


@dataclass
class AnalysisResults:
    """Container for complete mission analysis results."""
    mission_name: str
    trajectory_results: Dict[str, Any]
    optimization_results: Dict[str, Any]
    economic_analysis: Dict[str, Any]
    visualization_assets: Dict[str, Any]
    analysis_metadata: Dict[str, Any]


class LunarHorizonOptimizer:
    """
    Main integration class for the Lunar Horizon Optimizer system.
    
    This class provides a unified interface for complete lunar mission analysis,
    integrating trajectory generation, multi-objective optimization, economic
    analysis, and interactive visualization.
    """
    
    def __init__(self, 
                 mission_config: Optional[MissionConfig] = None,
                 cost_factors: Optional[CostFactors] = None,
                 spacecraft_config: Optional[SpacecraftConfig] = None):
        """
        Initialize the Lunar Horizon Optimizer.
        
        Args:
            mission_config: Mission configuration parameters
            cost_factors: Economic cost factors
            spacecraft_config: Spacecraft specifications
        """
        # Initialize configuration
        self.mission_config = mission_config or self._create_default_mission_config()
        self.cost_factors = cost_factors or self._create_default_cost_factors()
        self.spacecraft_config = spacecraft_config or self._create_default_spacecraft_config()
        
        # Initialize core components
        self._initialize_components()
        
        # Analysis cache
        self._analysis_cache = {}
        
        logger.info("Lunar Horizon Optimizer initialized successfully")
    
    def _initialize_components(self):
        """Initialize all core analysis components."""
        # Task 3: Trajectory components
        self.lunar_transfer = LunarTransfer()
        self.window_analyzer = TrajectoryWindowAnalyzer()
        
        # Task 4: Optimization components
        self.pareto_analyzer = ParetoAnalyzer()
        
        # Task 5: Economic components
        financial_params = FinancialParameters(
            discount_rate=0.08,
            inflation_rate=0.03,
            tax_rate=0.25,
            project_duration_years=10
        )
        self.npv_analyzer = NPVAnalyzer(financial_params)
        self.cost_model = MissionCostModel()
        self.isru_analyzer = ISRUBenefitAnalyzer()
        self.sensitivity_analyzer = EconomicSensitivityAnalyzer(self._economic_model)
        self.economic_reporter = EconomicReporter()
        
        # Task 6: Visualization components
        self.dashboard = ComprehensiveDashboard()
        self.trajectory_viz = TrajectoryVisualizer()
        self.optimization_viz = OptimizationVisualizer()
        self.economic_viz = EconomicVisualizer()
    
    def analyze_mission(self, 
                       mission_name: str = "Lunar Mission Analysis",
                       optimization_config: Optional[OptimizationConfig] = None,
                       include_sensitivity: bool = True,
                       include_isru: bool = True,
                       verbose: bool = True) -> AnalysisResults:
        """
        Perform comprehensive lunar mission analysis.
        
        Args:
            mission_name: Name for the mission analysis
            optimization_config: Optimization parameters
            include_sensitivity: Whether to include sensitivity analysis
            include_isru: Whether to include ISRU economic analysis
            verbose: Enable detailed progress logging
            
        Returns:
            Complete analysis results including all modules
        """
        if verbose:
            logger.info(f"🚀 Starting comprehensive analysis for: {mission_name}")
        
        opt_config = optimization_config or OptimizationConfig()
        
        # Step 1: Trajectory Analysis
        if verbose:
            logger.info("📊 Step 1: Performing trajectory analysis...")
        trajectory_results = self._analyze_trajectories(opt_config, verbose)
        
        # Step 2: Multi-objective Optimization
        if verbose:
            logger.info("🎯 Step 2: Running multi-objective optimization...")
        optimization_results = self._perform_optimization(opt_config, verbose)
        
        # Step 3: Economic Analysis
        if verbose:
            logger.info("💰 Step 3: Conducting economic analysis...")
        economic_results = self._analyze_economics(
            optimization_results, 
            include_sensitivity, 
            include_isru, 
            verbose
        )
        
        # Step 4: Generate Visualizations
        if verbose:
            logger.info("📈 Step 4: Creating comprehensive visualizations...")
        visualization_assets = self._create_visualizations(
            trajectory_results,
            optimization_results, 
            economic_results,
            mission_name
        )
        
        # Step 5: Compile Results
        analysis_results = AnalysisResults(
            mission_name=mission_name,
            trajectory_results=trajectory_results,
            optimization_results=optimization_results,
            economic_analysis=economic_results,
            visualization_assets=visualization_assets,
            analysis_metadata={
                'analysis_date': datetime.now().isoformat(),
                'configuration': {
                    'mission_config': self.mission_config.__dict__,
                    'cost_factors': self.cost_factors.__dict__,
                    'optimization_config': opt_config.__dict__
                },
                'performance_metrics': self._calculate_performance_metrics()
            }
        )
        
        if verbose:
            logger.info("✅ Comprehensive mission analysis completed successfully!")
            self._print_analysis_summary(analysis_results)
        
        return analysis_results
    
    def _analyze_trajectories(self, opt_config: OptimizationConfig, verbose: bool) -> Dict[str, Any]:
        """Analyze trajectory options and transfer windows."""
        results = {}
        
        # Baseline trajectory
        baseline_trajectory, baseline_dv = generate_earth_moon_trajectory(
            departure_epoch=10000.0,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            method='lambert'
        )
        
        results['baseline'] = {
            'trajectory': baseline_trajectory,
            'total_dv': baseline_dv,
            'transfer_time': 4.5,
            'earth_orbit_alt': 400.0,
            'moon_orbit_alt': 100.0
        }
        
        # Transfer window analysis
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 7, 1)
        
        transfer_windows = self.window_analyzer.find_transfer_windows(
            start_date=start_date,
            end_date=end_date,
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            time_step=2.0
        )
        
        results['transfer_windows'] = {
            'windows': transfer_windows,
            'analysis_period': f"{start_date} to {end_date}",
            'best_window': min(transfer_windows, key=lambda w: w.total_dv) if transfer_windows else None
        }
        
        if verbose:
            logger.info(f"   - Baseline trajectory: ΔV = {baseline_dv:.0f} m/s")
            logger.info(f"   - Transfer windows analyzed: {len(transfer_windows)} windows found")
        
        return results
    
    def _perform_optimization(self, opt_config: OptimizationConfig, verbose: bool) -> Dict[str, Any]:
        """Perform multi-objective optimization."""
        # Create optimization problem
        problem = LunarMissionProblem(
            cost_factors=self.cost_factors,
            min_earth_alt=opt_config.min_earth_alt,
            max_earth_alt=opt_config.max_earth_alt,
            min_moon_alt=opt_config.min_moon_alt,
            max_moon_alt=opt_config.max_moon_alt,
            min_transfer_time=opt_config.min_transfer_time,
            max_transfer_time=opt_config.max_transfer_time,
            reference_epoch=10000.0
        )
        
        # Run optimization
        optimizer = GlobalOptimizer(
            problem=problem,
            population_size=opt_config.population_size,
            num_generations=opt_config.num_generations,
            seed=opt_config.seed
        )
        
        optimization_results = optimizer.optimize(verbose=verbose)
        
        # Analyze Pareto front
        analyzed_results = self.pareto_analyzer.analyze_pareto_front(optimization_results)
        
        # Get best solutions with different preferences
        preference_sets = [
            ([0.6, 0.2, 0.2], "Delta-V Focused"),
            ([0.2, 0.6, 0.2], "Time Focused"), 
            ([0.2, 0.2, 0.6], "Cost Focused"),
            ([0.33, 0.33, 0.34], "Balanced")
        ]
        
        best_solutions = {}
        for weights, label in preference_sets:
            solutions = optimizer.get_best_solutions(
                num_solutions=3,
                preference_weights=weights
            )
            best_solutions[label] = solutions
        
        results = {
            'raw_results': optimization_results,
            'analyzed_results': analyzed_results,
            'best_solutions': best_solutions,
            'pareto_front_size': len(optimization_results.get('pareto_solutions', [])),
            'optimization_config': opt_config.__dict__
        }
        
        if verbose:
            logger.info(f"   - Pareto solutions found: {results['pareto_front_size']}")
            logger.info(f"   - Cache efficiency: {optimization_results.get('cache_stats', {}).get('hit_rate', 0):.1%}")
        
        return results
    
    def _analyze_economics(self, optimization_results: Dict[str, Any], 
                          include_sensitivity: bool, include_isru: bool, 
                          verbose: bool) -> Dict[str, Any]:
        """Perform comprehensive economic analysis."""
        results = {}
        
        # Analyze top Pareto solutions economically
        pareto_solutions = optimization_results['raw_results'].get('pareto_solutions', [])
        top_solutions = pareto_solutions[:5] if pareto_solutions else []
        
        economic_analyses = []
        for i, solution in enumerate(top_solutions):
            analysis = self._analyze_solution_economics(solution, f"Solution_{i+1}")
            economic_analyses.append(analysis)
        
        results['solution_analyses'] = economic_analyses
        
        # ISRU Analysis
        if include_isru:
            isru_analysis = self.isru_analyzer.analyze_isru_economics(
                resource_name='water_ice',
                facility_scale='commercial',
                operation_duration_months=60
            )
            results['isru_analysis'] = isru_analysis
            
            if verbose:
                npv = isru_analysis['financial_metrics']['npv']
                logger.info(f"   - ISRU Analysis: NPV = ${npv/1e6:.1f}M")
        
        # Sensitivity Analysis
        if include_sensitivity:
            base_params = {
                'cost_multiplier': 1.0,
                'revenue_multiplier': 1.0,
                'development_duration_multiplier': 1.0
            }
            
            distributions = {
                'cost_multiplier': {'type': 'triangular', 'min': 0.8, 'mode': 1.0, 'max': 1.5},
                'revenue_multiplier': {'type': 'normal', 'mean': 1.0, 'std': 0.2},
                'development_duration_multiplier': {'type': 'uniform', 'min': 0.9, 'max': 1.3}
            }
            
            mc_results = self.sensitivity_analyzer.monte_carlo_simulation(
                base_params, distributions, 1000
            )
            results['sensitivity_analysis'] = mc_results
            
            if verbose:
                mean_npv = mc_results['statistics']['mean']
                prob_positive = mc_results['risk_metrics']['probability_positive_npv']
                logger.info(f"   - Monte Carlo: Mean NPV = ${mean_npv/1e6:.1f}M, P(NPV>0) = {prob_positive:.1%}")
        
        # Generate economic reports
        if economic_analyses:
            best_analysis = economic_analyses[0]
            exec_summary = self.economic_reporter.generate_executive_summary(
                best_analysis['financial_summary']
            )
            results['executive_summary'] = exec_summary
        
        return results
    
    def _analyze_solution_economics(self, solution: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Analyze economics for a single optimization solution."""
        params = solution.get('parameters', {})
        objectives = solution.get('objectives', {})
        
        # Create cash flow model
        financial_params = FinancialParameters(
            discount_rate=0.08,
            inflation_rate=0.03,
            tax_rate=0.25,
            project_duration_years=8
        )
        
        cash_model = CashFlowModel(financial_params)
        start_date = datetime(2025, 1, 1)
        
        # Estimate costs based on solution parameters
        spacecraft_mass = self.spacecraft_config.dry_mass
        mission_duration = params.get('transfer_time', 4.5) / 365.0 * 2  # Round trip
        
        cost_breakdown = self.cost_model.estimate_total_mission_cost(
            spacecraft_mass=spacecraft_mass,
            mission_duration_years=mission_duration,
            technology_readiness=3,
            complexity='moderate',
            schedule='nominal'
        )
        
        # Add cash flows
        cash_model.add_development_costs(cost_breakdown.development, start_date, 24)
        cash_model.add_launch_costs(cost_breakdown.launch, [start_date + timedelta(days=730)])
        cash_model.add_operational_costs(cost_breakdown.operations, start_date + timedelta(days=730), 36)
        
        # Add simplified revenue streams
        annual_revenue = 10e6  # Simplified revenue assumption
        cash_model.add_revenue_stream(annual_revenue, start_date + timedelta(days=760), 48)
        
        # Calculate financial metrics
        npv = self.npv_analyzer.calculate_npv(cash_model)
        irr = self.npv_analyzer.calculate_irr(cash_model)
        payback = self.npv_analyzer.calculate_payback_period(cash_model)
        
        financial_summary = FinancialSummary(
            total_investment=cost_breakdown.total,
            total_revenue=annual_revenue * 4,  # 4 years of revenue
            net_present_value=npv,
            internal_rate_of_return=irr,
            return_on_investment=(annual_revenue * 4 - cost_breakdown.total) / cost_breakdown.total,
            payback_period_years=payback,
            mission_duration_years=mission_duration,
            probability_of_success=0.75
        )
        
        return {
            'label': label,
            'solution': solution,
            'cost_breakdown': cost_breakdown,
            'financial_summary': financial_summary,
            'cash_model': cash_model
        }
    
    def _create_visualizations(self, trajectory_results: Dict[str, Any],
                             optimization_results: Dict[str, Any],
                             economic_results: Dict[str, Any],
                             mission_name: str) -> Dict[str, Any]:
        """Create comprehensive visualizations."""
        visualizations = {}
        
        # Create mission data for dashboard
        mission_data = MissionAnalysisData(
            mission_name=mission_name,
            trajectory_data=trajectory_results.get('baseline', {}),
            optimization_results=optimization_results.get('analyzed_results'),
            financial_summary=economic_results.get('solution_analyses', [{}])[0].get('financial_summary'),
            cost_breakdown=economic_results.get('solution_analyses', [{}])[0].get('cost_breakdown')
        )
        
        # Executive dashboard
        try:
            exec_dashboard = self.dashboard.create_executive_dashboard(mission_data)
            visualizations['executive_dashboard'] = exec_dashboard
        except Exception as e:
            logger.warning(f"Could not create executive dashboard: {e}")
            visualizations['executive_dashboard'] = None
        
        # Technical dashboard
        try:
            tech_dashboard = self.dashboard.create_technical_dashboard(mission_data)
            visualizations['technical_dashboard'] = tech_dashboard
        except Exception as e:
            logger.warning(f"Could not create technical dashboard: {e}")
            visualizations['technical_dashboard'] = None
        
        # Individual visualizations
        try:
            # Trajectory visualization
            baseline_trajectory = trajectory_results.get('baseline', {}).get('trajectory')
            if baseline_trajectory and hasattr(baseline_trajectory, 'trajectory_data'):
                traj_plot = self.trajectory_viz.create_3d_trajectory_plot(
                    baseline_trajectory.trajectory_data
                )
                visualizations['trajectory_plot'] = traj_plot
        except Exception as e:
            logger.warning(f"Could not create trajectory plot: {e}")
        
        try:
            # Pareto front visualization
            opt_result = optimization_results.get('analyzed_results')
            if opt_result:
                pareto_plot = self.optimization_viz.create_pareto_front_plot(
                    opt_result,
                    objective_names=["Delta-V (m/s)", "Transfer Time (days)", "Cost ($)"]
                )
                visualizations['pareto_plot'] = pareto_plot
        except Exception as e:
            logger.warning(f"Could not create Pareto plot: {e}")
        
        try:
            # Economic dashboard
            financial_summary = economic_results.get('solution_analyses', [{}])[0].get('financial_summary')
            if financial_summary:
                econ_dashboard = self.economic_viz.create_financial_dashboard(financial_summary)
                visualizations['economic_dashboard'] = econ_dashboard
        except Exception as e:
            logger.warning(f"Could not create economic dashboard: {e}")
        
        return visualizations
    
    def _economic_model(self, params: Dict[str, float]) -> Dict[str, float]:
        """Economic model for sensitivity analysis."""
        base_cost = 500e6
        base_revenue = 750e6
        
        total_cost = base_cost * params.get('cost_multiplier', 1.0)
        total_revenue = base_revenue * params.get('revenue_multiplier', 1.0)
        
        npv = total_revenue - total_cost
        
        return {'npv': npv}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate system performance metrics."""
        return {
            'modules_loaded': {
                'trajectory': True,
                'optimization': True,
                'economics': True,
                'visualization': True
            },
            'version': '1.0.0-rc1',
            'capabilities': [
                'Earth-Moon trajectory generation',
                'Multi-objective optimization',
                'Economic analysis with NPV/IRR',
                'Interactive visualization',
                'Integrated workflow'
            ]
        }
    
    def _print_analysis_summary(self, results: AnalysisResults):
        """Print a summary of analysis results."""
        print("\n" + "="*60)
        print(f"LUNAR MISSION ANALYSIS SUMMARY: {results.mission_name}")
        print("="*60)
        
        # Trajectory summary
        baseline = results.trajectory_results.get('baseline', {})
        if baseline:
            print(f"📊 TRAJECTORY ANALYSIS:")
            print(f"   • Baseline ΔV: {baseline.get('total_dv', 0):.0f} m/s")
            print(f"   • Transfer time: {baseline.get('transfer_time', 0):.1f} days")
            print(f"   • Earth orbit: {baseline.get('earth_orbit_alt', 0):.0f} km")
            print(f"   • Moon orbit: {baseline.get('moon_orbit_alt', 0):.0f} km")
        
        # Optimization summary
        opt_results = results.optimization_results
        pareto_size = opt_results.get('pareto_front_size', 0)
        print(f"\n🎯 OPTIMIZATION RESULTS:")
        print(f"   • Pareto solutions: {pareto_size}")
        
        # Economic summary
        econ_analyses = results.economic_analysis.get('solution_analyses', [])
        if econ_analyses:
            best_analysis = econ_analyses[0]
            financial_summary = best_analysis.get('financial_summary')
            if financial_summary:
                print(f"\n💰 ECONOMIC ANALYSIS (Best Solution):")
                print(f"   • NPV: ${financial_summary.net_present_value/1e6:.1f}M")
                print(f"   • IRR: {financial_summary.internal_rate_of_return:.1%}")
                print(f"   • ROI: {financial_summary.return_on_investment:.1%}")
                print(f"   • Payback: {financial_summary.payback_period_years:.1f} years")
        
        # Visualization summary
        viz_count = len([v for v in results.visualization_assets.values() if v is not None])
        print(f"\n📈 VISUALIZATION ASSETS:")
        print(f"   • Created visualizations: {viz_count}")
        
        print("\n" + "="*60)
        print("Analysis completed successfully! 🚀")
        print("="*60)
    
    def export_results(self, results: AnalysisResults, output_dir: str = "output"):
        """Export analysis results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export metadata
        metadata_path = os.path.join(output_dir, "analysis_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(results.analysis_metadata, f, indent=2, default=str)
        
        # Export financial summary
        econ_analyses = results.economic_analysis.get('solution_analyses', [])
        if econ_analyses:
            financial_summary = econ_analyses[0].get('financial_summary')
            if financial_summary:
                summary_path = os.path.join(output_dir, "financial_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(financial_summary.__dict__, f, indent=2, default=str)
        
        # Export visualizations
        for name, fig in results.visualization_assets.items():
            if fig is not None:
                try:
                    html_path = os.path.join(output_dir, f"{name}.html")
                    fig.write_html(html_path)
                except Exception as e:
                    logger.warning(f"Could not export {name}: {e}")
        
        logger.info(f"Results exported to {output_dir}")
    
    @staticmethod
    def _create_default_mission_config() -> MissionConfig:
        """Create default mission configuration."""
        return MissionConfig(
            name="Default Lunar Mission",
            earth_orbit_alt=400.0,
            moon_orbit_alt=100.0,
            transfer_time=4.5,
            departure_epoch=10000.0
        )
    
    @staticmethod
    def _create_default_cost_factors() -> CostFactors:
        """Create default cost factors."""
        return CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=100000.0,
            development_cost=1e9,
            contingency_percentage=20.0
        )
    
    @staticmethod  
    def _create_default_spacecraft_config() -> SpacecraftConfig:
        """Create default spacecraft configuration."""
        return SpacecraftConfig(
            name="Default Lunar Spacecraft",
            dry_mass=5000.0,
            propellant_mass=3000.0,
            payload_mass=1000.0,
            power_system_mass=500.0,
            propulsion_isp=320.0
        )


def main():
    """
    Main function demonstrating the integrated Lunar Horizon Optimizer.
    """
    print("🚀 Lunar Horizon Optimizer - Integrated Mission Analysis")
    print("="*60)
    
    # Initialize system
    optimizer = LunarHorizonOptimizer()
    
    # Configure analysis
    opt_config = OptimizationConfig(
        population_size=50,  # Reduced for demo
        num_generations=25,  # Reduced for demo
        seed=42
    )
    
    # Run comprehensive analysis
    results = optimizer.analyze_mission(
        mission_name="Artemis Lunar Base Mission",
        optimization_config=opt_config,
        include_sensitivity=True,
        include_isru=True,
        verbose=True
    )
    
    # Export results
    optimizer.export_results(results, "mission_analysis_output")
    
    # Display visualizations if available
    if results.visualization_assets.get('executive_dashboard'):
        print("\n📈 Opening executive dashboard...")
        results.visualization_assets['executive_dashboard'].show()
    
    if results.visualization_assets.get('technical_dashboard'):
        print("📊 Opening technical dashboard...")
        results.visualization_assets['technical_dashboard'].show()
    
    print("\n✅ Integrated mission analysis completed successfully!")
    return results


if __name__ == "__main__":
    results = main()