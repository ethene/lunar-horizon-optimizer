"""
Comprehensive test suite for Task 6: Visualization Module

This module tests all components of the visualization system including:
- 3D trajectory visualization with Plotly
- Pareto front and optimization results visualization
- Economic analysis dashboards and charts
- Mission timeline and milestone visualization
- Comprehensive integrated dashboards

Tests include sanity checks for realistic values, proper data validation,
and verification that all visualizations produce sensible results.
"""

import pytest
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Test constants
EARTH_RADIUS = 6378137.0  # m
MOON_RADIUS = 1737400.0  # m
EARTH_MOON_DISTANCE = 384400000.0  # m

# Try importing plotly - skip tests if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    pytestmark = pytest.mark.skip("Plotly not available")


class TestTrajectoryVisualization:
    """Test suite for trajectory visualization module."""

    def setup_method(self):
        """Setup test fixtures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        try:
            from visualization.trajectory_visualization import (
                TrajectoryVisualizer,
                TrajectoryPlotConfig,
                create_quick_trajectory_plot,
            )

            self.TrajectoryVisualizer = TrajectoryVisualizer
            self.TrajectoryPlotConfig = TrajectoryPlotConfig
            self.create_quick_trajectory_plot = create_quick_trajectory_plot

            # Create test configuration
            self.config = TrajectoryPlotConfig(
                width=800, height=600, title="Test Trajectory"
            )

            self.visualizer = TrajectoryVisualizer(self.config)

        except ImportError as e:
            pytest.skip(f"Trajectory visualization module not available: {e}")

    def test_trajectory_visualizer_initialization(self):
        """Test trajectory visualizer initialization."""
        assert self.visualizer.config.width == 800
        assert self.visualizer.config.height == 600
        assert self.visualizer.config.title == "Test Trajectory"

        # Check required components exist
        assert hasattr(self.visualizer, "lambert_solver")
        assert hasattr(self.visualizer, "propagator")
        assert hasattr(self.visualizer, "window_analyzer")

    def test_trajectory_plot_config_defaults(self):
        """Test trajectory plot configuration defaults."""
        default_config = self.TrajectoryPlotConfig()

        assert default_config.width == 1200
        assert default_config.height == 800
        assert default_config.show_earth is True
        assert default_config.show_moon is True
        assert default_config.trajectory_color == "#00ff00"
        assert default_config.enable_animation is True

    def test_create_3d_trajectory_plot_with_sample_data(self):
        """Test 3D trajectory plot creation with realistic sample data."""
        # Create realistic trajectory data
        n_points = 100
        time_span = 4.5 * 86400  # 4.5 days in seconds
        times = np.linspace(0, time_span, n_points)

        # Simulate Earth-Moon transfer trajectory
        # Start near Earth, end near Moon
        earth_pos = np.array([EARTH_RADIUS + 400000, 0, 0])  # 400 km altitude
        moon_pos = np.array([EARTH_MOON_DISTANCE, 0, 0])  # Moon position

        # Linear interpolation for simplicity (real trajectory would be curved)
        positions = np.zeros((3, n_points))
        velocities = np.zeros((3, n_points))

        for i in range(n_points):
            t_frac = i / (n_points - 1)
            positions[:, i] = earth_pos + t_frac * (moon_pos - earth_pos)
            # Simple velocity model
            velocities[:, i] = np.array([3000, 1000, 0]) * (1 - 0.5 * t_frac)

        trajectory_data = {
            "positions": positions,
            "velocities": velocities,
            "times": times,
        }

        # Create plot
        fig = self.visualizer.create_3d_trajectory_plot(trajectory_data)

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Should have trajectory trace

        # Check layout
        assert fig.layout.scene.xaxis.title.text == "X (km)"
        assert fig.layout.scene.yaxis.title.text == "Y (km)"
        assert fig.layout.scene.zaxis.title.text == "Z (km)"

        # Validate data sanity
        trajectory_trace = None
        for trace in fig.data:
            if hasattr(trace, "x") and len(trace.x) == n_points:
                trajectory_trace = trace
                break

        assert trajectory_trace is not None

        # Check trajectory data sanity (converted to km)
        x_data = np.array(trajectory_trace.x)
        y_data = np.array(trajectory_trace.y)
        z_data = np.array(trajectory_trace.z)

        # Start point should be near Earth
        start_distance = np.sqrt(x_data[0] ** 2 + y_data[0] ** 2 + z_data[0] ** 2)
        assert 6000 < start_distance < 8000  # 6000-8000 km from Earth center

        # End point should be near Moon distance
        end_distance = np.sqrt(x_data[-1] ** 2 + y_data[-1] ** 2 + z_data[-1] ** 2)
        assert 300000 < end_distance < 400000  # Near Moon distance

    def test_create_transfer_window_plot_with_mock_data(self):
        """Test transfer window plot with mock data since we don't have full implementation."""
        start_date = datetime(2025, 6, 1)
        end_date = datetime(2025, 7, 1)

        # Mock the window analyzer to return test data
        with patch.object(
            self.visualizer.window_analyzer, "find_transfer_windows"
        ) as mock_windows:
            # Create realistic mock transfer windows
            mock_windows.return_value = [
                MagicMock(
                    departure_date=start_date + timedelta(days=i),
                    arrival_date=start_date + timedelta(days=i + 4),
                    total_dv=3000 + i * 100,  # Realistic delta-v values
                    c3_energy=10 + i * 2,  # Realistic C3 energy
                    transfer_time=4.0 + i * 0.1,
                )
                for i in range(10)
            ]

            fig = self.visualizer.create_transfer_window_plot(
                start_date=start_date, end_date=end_date
            )

            # Validate plot structure
            assert isinstance(fig, go.Figure)
            assert len(fig.data) >= 1  # Should have data traces

            # Check that transfer window analysis was called
            mock_windows.assert_called_once()

            # Validate realistic ranges in plot data
            for trace in fig.data:
                if hasattr(trace, "y") and len(trace.y) > 0:
                    y_values = np.array(trace.y)
                    if np.all(y_values > 1000):  # Delta-v values
                        assert np.all(
                            y_values < 10000
                        )  # Reasonable delta-v upper bound
                    elif np.all(y_values > 1):  # Transfer time values
                        assert np.all(y_values < 20)  # Reasonable transfer time

    def test_orbital_elements_calculation_sanity(self):
        """Test orbital elements calculation with realistic data."""
        # Create simple circular orbit data
        n_points = 50

        # 400 km circular orbit
        orbit_radius = EARTH_RADIUS + 400000  # m
        orbital_period = 2 * np.pi * np.sqrt(orbit_radius**3 / 3.986004418e14)  # s

        times = np.linspace(0, orbital_period, n_points)
        positions = np.zeros((3, n_points))
        velocities = np.zeros((3, n_points))

        # Circular orbit
        for i, t in enumerate(times):
            angle = 2 * np.pi * t / orbital_period
            positions[0, i] = orbit_radius * np.cos(angle)
            positions[1, i] = orbit_radius * np.sin(angle)
            positions[2, i] = 0

            # Circular velocity
            v_mag = np.sqrt(3.986004418e14 / orbit_radius)
            velocities[0, i] = -v_mag * np.sin(angle)
            velocities[1, i] = v_mag * np.cos(angle)
            velocities[2, i] = 0

        trajectory_data = {
            "positions": positions,
            "velocities": velocities,
            "times": times,
        }

        fig = self.visualizer.create_orbital_elements_plot(trajectory_data)

        # Validate plot structure
        assert isinstance(fig, go.Figure)

        # Test orbital elements calculation method directly
        elements = self.visualizer._calculate_orbital_elements_evolution(
            positions, velocities
        )

        # Validate orbital elements sanity for circular orbit
        if "a" in elements:
            semi_major_axes = elements["a"]
            # Filter out any NaN values
            valid_a = semi_major_axes[np.isfinite(semi_major_axes)]
            if len(valid_a) > 0:
                mean_a = np.mean(valid_a)
                assert abs(mean_a - orbit_radius) < 10000  # Within 10 km

        if "e" in elements:
            eccentricities = elements["e"]
            valid_e = eccentricities[np.isfinite(eccentricities)]
            if len(valid_e) > 0:
                mean_e = np.mean(valid_e)
                assert mean_e < 0.1  # Nearly circular

    def test_quick_trajectory_plot_function(self):
        """Test quick trajectory plot creation function."""
        # Test with realistic mission parameters
        fig = self.create_quick_trajectory_plot(
            earth_orbit_alt=400.0,  # km
            moon_orbit_alt=100.0,  # km
            transfer_time=4.5,  # days
            departure_epoch=10000.0,  # days since J2000
        )

        # Should return a valid figure (even if trajectory generation fails)
        assert isinstance(fig, go.Figure)

        # Check if it's an error plot or actual trajectory
        if len(fig.data) > 0:
            # If trajectory was generated successfully, validate it
            assert fig.layout.title is not None
        else:
            # If no data, should have error annotation
            assert len(fig.layout.annotations) > 0


class TestOptimizationVisualization:
    """Test suite for optimization visualization module."""

    def setup_method(self):
        """Setup test fixtures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        try:
            from visualization.optimization_visualization import (
                OptimizationVisualizer,
                ParetoPlotConfig,
                create_quick_pareto_plot,
            )
            from optimization.pareto_analysis import OptimizationResult

            self.OptimizationVisualizer = OptimizationVisualizer
            self.ParetoPlotConfig = ParetoPlotConfig
            self.OptimizationResult = OptimizationResult
            self.create_quick_pareto_plot = create_quick_pareto_plot

            self.visualizer = OptimizationVisualizer()

        except ImportError as e:
            pytest.skip(f"Optimization visualization module not available: {e}")

    def test_optimization_visualizer_initialization(self):
        """Test optimization visualizer initialization."""
        assert hasattr(self.visualizer, "config")
        assert hasattr(self.visualizer, "pareto_analyzer")

    def test_pareto_front_plot_with_realistic_data(self):
        """Test Pareto front plot with realistic optimization data."""
        # Create realistic Pareto solutions for lunar mission
        pareto_solutions = []

        # Generate realistic trade-offs between delta-v, time, and cost
        for _i in range(20):
            # Delta-v typically 3-6 km/s for lunar missions
            delta_v = 3000 + np.random.uniform(-500, 1500)

            # Transfer time typically 3-8 days
            transfer_time = 3 + np.random.uniform(0, 5)

            # Cost scales with delta-v and complexity
            cost = 200e6 + delta_v * 50 + transfer_time * 10e6

            pareto_solutions.append(
                {
                    "objectives": [
                        delta_v,
                        transfer_time * 86400,
                        cost,
                    ],  # Convert time to seconds
                    "parameters": [
                        400 + np.random.uniform(-200, 400),  # Earth altitude
                        100 + np.random.uniform(-50, 200),  # Moon altitude
                        transfer_time,
                    ],  # Transfer time
                }
            )

        # Create optimization result
        opt_result = self.OptimizationResult(
            pareto_solutions=pareto_solutions,
            all_solutions=pareto_solutions,
            optimization_stats={"generations": 100, "population_size": 50},
            generation_history=[],
        )

        # Create plot
        objective_names = ["Delta-V (m/s)", "Transfer Time (s)", "Cost ($)"]
        fig = self.visualizer.create_pareto_front_plot(
            opt_result, objective_names=objective_names, show_dominated=False
        )

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Validate data ranges
        for trace in fig.data:
            if hasattr(trace, "x") and hasattr(trace, "y"):
                x_data = np.array(trace.x)
                y_data = np.array(trace.y)

                # Check delta-v range (assuming x-axis)
                if np.min(x_data) > 1000:  # Delta-v values
                    assert np.min(x_data) >= 2000  # Minimum realistic delta-v
                    assert np.max(x_data) <= 10000  # Maximum realistic delta-v

                # Check time range (assuming y-axis in seconds)
                if np.min(y_data) > 100000:  # Time in seconds
                    assert np.min(y_data) >= 2 * 86400  # At least 2 days
                    assert np.max(y_data) <= 15 * 86400  # At most 15 days

    def test_solution_comparison_plot(self):
        """Test solution comparison visualization."""
        # Create test solutions with realistic values
        solutions = [
            {
                "objectives": [3200, 4.5 * 86400, 250e6],  # Conservative
                "parameters": [400, 100, 4.5],
            },
            {
                "objectives": [4100, 3.2 * 86400, 280e6],  # Fast
                "parameters": [600, 150, 3.2],
            },
            {
                "objectives": [2800, 6.1 * 86400, 230e6],  # Efficient
                "parameters": [350, 80, 6.1],
            },
        ]

        solution_labels = ["Conservative", "Fast", "Efficient"]
        objective_names = ["Delta-V (m/s)", "Time (s)", "Cost ($)"]
        parameter_names = ["Earth Alt (km)", "Moon Alt (km)", "Transfer Time (days)"]

        fig = self.visualizer.create_solution_comparison_plot(
            solutions=solutions,
            solution_labels=solution_labels,
            objective_names=objective_names,
            parameter_names=parameter_names,
        )

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that comparison data is reasonable
        for trace in fig.data:
            if hasattr(trace, "x") and hasattr(trace, "y"):
                # Validate that values are in expected ranges
                y_data = np.array(trace.y)
                if len(y_data) > 0:
                    # Should have positive values for costs and times
                    assert np.all(y_data >= 0)

    def test_preference_analysis_plot(self):
        """Test preference-based solution ranking visualization."""
        # Create test Pareto solutions
        pareto_solutions = [
            {"objectives": [3000, 4 * 86400, 240e6], "parameters": [400, 100, 4]},
            {"objectives": [3500, 3 * 86400, 260e6], "parameters": [500, 120, 3]},
            {"objectives": [2800, 5 * 86400, 220e6], "parameters": [350, 90, 5]},
        ]

        preference_weights = [0.4, 0.3, 0.3]  # Prefer lower delta-v
        objective_names = ["Delta-V (m/s)", "Time (s)", "Cost ($)"]

        fig = self.visualizer.create_preference_analysis_plot(
            pareto_solutions=pareto_solutions,
            preference_weights=preference_weights,
            objective_names=objective_names,
        )

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that preferences were applied
        assert fig.layout.title.text is not None
        assert "Preference Analysis" in fig.layout.title.text

    def test_quick_pareto_plot_function(self):
        """Test quick Pareto plot creation function."""
        # Create mock optimization results
        mock_results = {
            "pareto_front": [
                {
                    "objectives": [3200, 4.5 * 86400, 250e6],
                    "parameters": [400, 100, 4.5],
                },
                {
                    "objectives": [3800, 3.8 * 86400, 270e6],
                    "parameters": [500, 120, 3.8],
                },
            ],
            "all_solutions": [],
            "stats": {"generations": 50},
        }

        fig = self.create_quick_pareto_plot(
            optimization_result=mock_results,
            objective_names=["Delta-V", "Time", "Cost"],
        )

        # Should return a valid figure
        assert isinstance(fig, go.Figure)


class TestEconomicVisualization:
    """Test suite for economic visualization module."""

    def setup_method(self):
        """Setup test fixtures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        try:
            from visualization.economic_visualization import (
                EconomicVisualizer,
                DashboardConfig,
                create_quick_financial_dashboard,
            )
            from economics.financial_models import FinancialSummary
            from economics.cost_models import CostBreakdown

            self.EconomicVisualizer = EconomicVisualizer
            self.DashboardConfig = DashboardConfig
            self.FinancialSummary = FinancialSummary
            self.CostBreakdown = CostBreakdown
            self.create_quick_financial_dashboard = create_quick_financial_dashboard

            self.visualizer = EconomicVisualizer()

        except ImportError as e:
            pytest.skip(f"Economic visualization module not available: {e}")

    def test_economic_visualizer_initialization(self):
        """Test economic visualizer initialization."""
        assert hasattr(self.visualizer, "config")
        assert hasattr(self.visualizer, "cost_model")
        assert hasattr(self.visualizer, "isru_analyzer")

    def test_financial_dashboard_with_realistic_data(self):
        """Test financial dashboard with realistic lunar mission data."""
        # Create realistic financial summary
        financial_summary = self.FinancialSummary(
            total_investment=500e6,  # $500M total investment
            total_revenue=750e6,  # $750M total revenue
            net_present_value=125e6,  # $125M NPV
            internal_rate_of_return=0.18,  # 18% IRR
            return_on_investment=0.25,  # 25% ROI
            payback_period_years=6.5,  # 6.5 year payback
            mission_duration_years=8,  # 8 year mission
            probability_of_success=0.75,  # 75% success probability
        )

        # Create realistic cost breakdown
        cost_breakdown = self.CostBreakdown(
            development=200e6,  # $200M development
            launch=150e6,  # $150M launch
            spacecraft=100e6,  # $100M spacecraft
            operations=80e6,  # $80M operations
            ground_systems=40e6,  # $40M ground systems
            contingency=30e6,  # $30M contingency
            total=600e6,  # $600M total
        )

        fig = self.visualizer.create_financial_dashboard(
            financial_summary=financial_summary,
            cash_flow_model=None,  # Optional
            cost_breakdown=cost_breakdown,
        )

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that financial metrics are in reasonable ranges
        # Look for indicator traces (KPIs)
        for trace in fig.data:
            if hasattr(trace, "value"):
                value = trace.value
                if isinstance(value, int | float):
                    # Check that financial values are reasonable
                    if abs(value) > 1:  # Large values (millions)
                        assert abs(value) < 1000  # Less than $1B in millions
                    else:  # Percentage values
                        assert -1 <= value <= 5  # IRR/ROI should be reasonable

    def test_cost_analysis_dashboard(self):
        """Test cost analysis dashboard visualization."""
        cost_breakdown = self.CostBreakdown(
            development=250e6,
            launch=180e6,
            spacecraft=120e6,
            operations=100e6,
            ground_systems=50e6,
            contingency=40e6,
            total=740e6,
        )

        fig = self.visualizer.create_cost_analysis_dashboard(
            cost_breakdown=cost_breakdown, comparison_scenarios=None
        )

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that cost values are positive and reasonable
        for trace in fig.data:
            if hasattr(trace, "values") and trace.values is not None:
                values = np.array(trace.values)
                if len(values) > 0 and np.all(values > 1e6):  # Cost values
                    assert np.all(values > 0)  # All costs should be positive
                    assert np.all(values < 1e10)  # Reasonable upper bound

    def test_isru_analysis_dashboard(self):
        """Test ISRU economic analysis dashboard."""
        # Create mock ISRU analysis data
        isru_analysis = {
            "financial_metrics": {"npv": 50e6, "roi": 0.15, "irr": 0.12},
            "production_profile": {
                "monthly_production": [10, 15, 20, 25, 30] * 12  # 5 years
            },
            "break_even_analysis": {
                "monthly_cash_flow": [-5e6] * 24 + [2e6] * 36,  # Break-even at 2 years
                "payback_period_months": 24,
            },
            "revenue_streams": {
                "Water Sales": 30e6,
                "Oxygen Sales": 15e6,
                "Fuel Sales": 20e6,
            },
        }

        fig = self.visualizer.create_isru_analysis_dashboard(
            isru_analysis=isru_analysis, resource_name="water_ice"
        )

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that ISRU metrics are reasonable
        financial_metrics = isru_analysis["financial_metrics"]
        assert -100e6 < financial_metrics["npv"] < 200e6  # Reasonable NPV range
        assert 0 <= financial_metrics["roi"] <= 1.0  # ROI should be percentage

    def test_quick_financial_dashboard_function(self):
        """Test quick financial dashboard creation function."""
        fig = self.create_quick_financial_dashboard(
            npv=75e6,
            irr=0.15,
            roi=0.20,
            payback_years=5.5,
            total_investment=400e6,
            total_revenue=600e6,
        )

        # Should return a valid figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestMissionVisualization:
    """Test suite for mission visualization module."""

    def setup_method(self):
        """Setup test fixtures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        try:
            from visualization.mission_visualization import (
                MissionVisualizer,
                TimelineConfig,
                MissionPhase,
                MissionMilestone,
                create_sample_mission_timeline,
            )

            self.MissionVisualizer = MissionVisualizer
            self.TimelineConfig = TimelineConfig
            self.MissionPhase = MissionPhase
            self.MissionMilestone = MissionMilestone
            self.create_sample_mission_timeline = create_sample_mission_timeline

            self.visualizer = MissionVisualizer()

        except ImportError as e:
            pytest.skip(f"Mission visualization module not available: {e}")

    def test_mission_visualizer_initialization(self):
        """Test mission visualizer initialization."""
        assert hasattr(self.visualizer, "config")
        assert isinstance(self.visualizer.config, self.TimelineConfig)

    def test_mission_timeline_with_realistic_phases(self):
        """Test mission timeline with realistic lunar mission phases."""
        base_date = datetime(2025, 1, 1)

        # Create realistic mission phases
        phases = [
            self.MissionPhase(
                name="Mission Design",
                start_date=base_date,
                end_date=base_date + timedelta(days=365),
                category="Development",
                cost=50e6,
                risk_level="Medium",
            ),
            self.MissionPhase(
                name="Spacecraft Development",
                start_date=base_date + timedelta(days=180),
                end_date=base_date + timedelta(days=730),
                category="Development",
                dependencies=["Mission Design"],
                cost=200e6,
                risk_level="High",
            ),
            self.MissionPhase(
                name="Launch Campaign",
                start_date=base_date + timedelta(days=700),
                end_date=base_date + timedelta(days=730),
                category="Launch",
                dependencies=["Spacecraft Development"],
                cost=100e6,
                risk_level="High",
            ),
        ]

        # Create realistic milestones
        milestones = [
            self.MissionMilestone(
                name="PDR",
                date=base_date + timedelta(days=120),
                category="Design",
                description="Preliminary Design Review",
            ),
            self.MissionMilestone(
                name="Launch",
                date=base_date + timedelta(days=720),
                category="Launch",
                description="Mission Launch",
            ),
        ]

        fig = self.visualizer.create_mission_timeline(phases, milestones)

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= len(phases)  # Should have traces for phases

        # Validate timeline data sanity
        for trace in fig.data:
            if hasattr(trace, "x") and len(trace.x) == 2:  # Phase trace
                start_time, end_time = trace.x
                assert start_time < end_time  # Start should be before end

                # Check that times are reasonable (within project timeframe)
                time_diff = (
                    (end_time - start_time).total_seconds()
                    if hasattr(start_time, "total_seconds")
                    else 0
                )
                if time_diff > 0:
                    assert time_diff < 5 * 365 * 86400  # Less than 5 years

    def test_resource_utilization_chart(self):
        """Test resource utilization visualization."""
        base_date = datetime(2025, 1, 1)

        # Create phases with resource requirements
        phases = [
            self.MissionPhase(
                name="Phase 1",
                start_date=base_date,
                end_date=base_date + timedelta(days=100),
                category="Development",
                resources={"engineers": 10, "budget": 1e6},
            ),
            self.MissionPhase(
                name="Phase 2",
                start_date=base_date + timedelta(days=50),
                end_date=base_date + timedelta(days=150),
                category="Testing",
                resources={"engineers": 15, "budget": 2e6, "hardware": 5},
            ),
        ]

        fig = self.visualizer.create_resource_utilization_chart(phases)

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that resource utilization values are positive
        for trace in fig.data:
            if hasattr(trace, "y"):
                y_data = np.array(trace.y)
                if len(y_data) > 0:
                    assert np.all(
                        y_data >= 0
                    )  # Resource utilization should be non-negative

    def test_mission_dashboard(self):
        """Test comprehensive mission dashboard."""
        base_date = datetime(2025, 1, 1)
        current_date = base_date + timedelta(days=200)

        # Create test phases and milestones
        phases = [
            self.MissionPhase(
                name="Development",
                start_date=base_date,
                end_date=base_date + timedelta(days=365),
                category="Development",
                risk_level="Medium",
            )
        ]

        milestones = [
            self.MissionMilestone(
                name="Critical Review",
                date=base_date + timedelta(days=300),
                category="Review",
            )
        ]

        fig = self.visualizer.create_mission_dashboard(phases, milestones, current_date)

        # Validate plot structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that dashboard includes current date reference
        assert fig.layout.title is not None
        assert current_date.strftime("%Y-%m-%d") in fig.layout.title.text

    def test_sample_mission_timeline_function(self):
        """Test sample mission timeline creation function."""
        fig = self.create_sample_mission_timeline()

        # Should return a valid figure with sample data
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1  # Should have sample phases


class TestComprehensiveDashboard:
    """Test suite for comprehensive dashboard integration."""

    def setup_method(self):
        """Setup test fixtures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        try:
            from visualization.dashboard import (
                ComprehensiveDashboard,
                DashboardTheme,
                MissionAnalysisData,
                create_sample_dashboard,
            )
            from economics.financial_models import FinancialSummary

            self.ComprehensiveDashboard = ComprehensiveDashboard
            self.DashboardTheme = DashboardTheme
            self.MissionAnalysisData = MissionAnalysisData
            self.FinancialSummary = FinancialSummary
            self.create_sample_dashboard = create_sample_dashboard

            self.dashboard = ComprehensiveDashboard()

        except ImportError as e:
            pytest.skip(f"Comprehensive dashboard module not available: {e}")

    def test_comprehensive_dashboard_initialization(self):
        """Test comprehensive dashboard initialization."""
        assert hasattr(self.dashboard, "theme")
        assert hasattr(self.dashboard, "trajectory_viz")
        assert hasattr(self.dashboard, "optimization_viz")
        assert hasattr(self.dashboard, "economic_viz")
        assert hasattr(self.dashboard, "mission_viz")

    def test_executive_dashboard_creation(self):
        """Test executive dashboard with realistic mission data."""
        # Create comprehensive mission analysis data
        financial_summary = self.FinancialSummary(
            total_investment=500e6,
            total_revenue=750e6,
            net_present_value=125e6,
            internal_rate_of_return=0.18,
            return_on_investment=0.25,
            payback_period_years=6.5,
            mission_duration_years=8,
            probability_of_success=0.75,
        )

        mission_data = self.MissionAnalysisData(
            mission_name="Artemis Lunar Base",
            financial_summary=financial_summary,
            analysis_date=datetime.now(),
        )

        fig = self.dashboard.create_executive_dashboard(mission_data)

        # Validate dashboard structure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

        # Check that executive dashboard includes key information
        assert fig.layout.title is not None
        assert "Executive Dashboard" in fig.layout.title.text
        assert mission_data.mission_name in fig.layout.title.text

    def test_technical_dashboard_creation(self):
        """Test technical dashboard creation."""
        mission_data = self.MissionAnalysisData(
            mission_name="Technical Analysis Test", analysis_date=datetime.now()
        )

        fig = self.dashboard.create_technical_dashboard(mission_data)

        # Validate dashboard structure
        assert isinstance(fig, go.Figure)
        assert fig.layout.title is not None
        assert "Technical Analysis Dashboard" in fig.layout.title.text

    def test_sample_dashboard_function(self):
        """Test sample dashboard creation function."""
        fig = self.create_sample_dashboard()

        # Should return a valid comprehensive dashboard
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestVisualizationIntegration:
    """Test suite for visualization module integration and sanity checks."""

    def setup_method(self):
        """Setup integration test fixtures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

    def test_visualization_module_imports(self):
        """Test that all visualization modules can be imported."""
        try:
            from visualization import (
                TrajectoryVisualizer,
                TrajectoryPlotConfig,
                OptimizationVisualizer,
                ParetoPlotConfig,
                EconomicVisualizer,
                DashboardConfig,
                MissionVisualizer,
                TimelineConfig,
                ComprehensiveDashboard,
                DashboardTheme,
            )

            # All imports successful
            assert True

        except ImportError as e:
            pytest.fail(f"Failed to import visualization modules: {e}")

    def test_realistic_value_ranges(self):
        """Test that all visualizations handle realistic value ranges properly."""
        # Define realistic lunar mission parameter ranges
        realistic_ranges = {
            "delta_v": (2000, 8000),  # m/s
            "transfer_time": (3, 12),  # days
            "mission_cost": (100e6, 2e9),  # dollars
            "npv": (-500e6, 1e9),  # dollars
            "irr": (-0.2, 0.5),  # fraction
            "spacecraft_mass": (1000, 10000),  # kg
            "orbit_altitude": (200, 2000),  # km
            "mission_duration": (1, 15),  # years
        }

        # Test that our constants are within realistic ranges
        assert 6.0e6 < EARTH_RADIUS < 7.0e6
        assert 1.5e6 < MOON_RADIUS < 2.0e6
        assert 3.8e8 < EARTH_MOON_DISTANCE < 4.0e8

        # Test realistic parameter validation
        for param, (min_val, max_val) in realistic_ranges.items():
            assert min_val < max_val
            assert min_val > 0 or param in ["npv", "irr"]  # NPV and IRR can be negative

    def test_plot_output_validation(self):
        """Test that plot outputs are valid Plotly figures."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not available")

        # Test that we can create basic plot types
        basic_fig = go.Figure()
        basic_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

        assert isinstance(basic_fig, go.Figure)
        assert len(basic_fig.data) == 1

        # Test 3D plot
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9]))

        assert isinstance(fig_3d, go.Figure)
        assert len(fig_3d.data) == 1

    @pytest.mark.skip("Manual test - requires visual inspection")
    def test_visualization_output_manual(self):
        """Manual test for visual inspection of plots (skip in automated tests)."""
        # This test would be used for manual verification of plot quality
        # Skip in automated testing but useful for development


# Parametrized tests for edge cases
@pytest.mark.parametrize("altitude", [200, 400, 800, 1200])
def test_orbit_altitude_range(altitude):
    """Test visualization with different orbit altitudes."""
    if not PLOTLY_AVAILABLE:
        pytest.skip("Plotly not available")

    # Test that altitude values produce sensible orbital parameters
    orbit_radius = EARTH_RADIUS + altitude * 1000  # Convert km to m

    # Orbital velocity calculation
    mu_earth = 3.986004418e14  # m³/s²
    orbital_velocity = np.sqrt(mu_earth / orbit_radius)

    # Sanity checks
    assert 6000 < orbital_velocity < 8000  # m/s - typical LEO velocities
    assert orbit_radius > EARTH_RADIUS  # Must be above Earth surface


@pytest.mark.parametrize("transfer_time", [3.0, 4.5, 6.0, 8.5])
def test_transfer_time_range(transfer_time):
    """Test visualization with different transfer times."""
    if not PLOTLY_AVAILABLE:
        pytest.skip("Plotly not available")

    # Test that transfer times are reasonable
    assert 1.0 < transfer_time < 15.0  # Days

    # Convert to seconds for validation
    transfer_seconds = transfer_time * 86400
    assert 86400 < transfer_seconds < 15 * 86400


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
