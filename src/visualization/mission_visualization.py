"""
Mission Timeline and Milestone Visualization Module.

Provides comprehensive visualization for lunar mission timelines including
project phases, milestones, critical path analysis, resource allocation,
and interactive mission planning tools.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta



@dataclass
class TimelineConfig:
    """Configuration for mission timeline visualization."""

    # Timeline layout
    width: int = 1400
    height: int = 800
    title: str = "Lunar Mission Timeline"

    # Phase colors
    development_color: str = "#FF6B6B"
    testing_color: str = "#4ECDC4"
    launch_color: str = "#45B7D1"
    operations_color: str = "#96CEB4"
    completion_color: str = "#FFEAA7"

    # Milestone styling
    milestone_size: int = 12
    milestone_color: str = "#FF7675"
    critical_path_color: str = "#E84393"
    critical_path_width: int = 4

    # Interactive features
    enable_hover: bool = True
    show_dependencies: bool = True
    enable_drag: bool = False

    # Professional styling
    theme: str = "plotly_white"
    font_family: str = "Arial, sans-serif"
    grid_color: str = "#E5E5E5"
    background_color: str = "#FFFFFF"


@dataclass
class MissionPhase:
    """Represents a mission phase or task."""

    name: str
    start_date: datetime
    end_date: datetime
    category: str
    dependencies: list[str] | None = None
    resources: dict[str, float] | None = None
    cost: float = 0.0
    risk_level: str = "Medium"
    critical_path: bool = False

    def __post_init__(self) -> None:
        if self.dependencies is None:
            self.dependencies = []
        if self.resources is None:
            self.resources = {}


@dataclass
class MissionMilestone:
    """Represents a mission milestone."""

    name: str
    date: datetime
    category: str
    description: str = ""
    completion_criteria: list[str] | None = None
    risk_level: str = "Medium"

    def __post_init__(self) -> None:
        if self.completion_criteria is None:
            self.completion_criteria = []


class MissionVisualizer:
    """
    Mission timeline and milestone visualization using Plotly.

    Provides comprehensive visualization of lunar mission planning including:
    - Interactive Gantt charts with phases and dependencies
    - Milestone tracking and completion status
    - Resource allocation and utilization charts
    - Critical path analysis
    - Risk assessment timelines
    """

    def __init__(self, config: TimelineConfig | None = None) -> None:
        """
        Initialize mission visualizer.

        Args:
            config: Timeline visualization configuration
        """
        self.config = config or TimelineConfig()

    def create_mission_timeline(
        self,
        phases: list[MissionPhase],
        milestones: list[MissionMilestone],
        title: str | None = None
    ) -> go.Figure:
        """
        Create comprehensive mission timeline visualization.

        Args:
            phases: List of mission phases
            milestones: List of mission milestones
            title: Optional timeline title

        Returns
        -------
            Plotly Figure with mission timeline
        """
        if not phases:
            return self._create_empty_plot("No mission phases provided")

        fig = go.Figure()

        # Add phases as Gantt bars
        self._add_gantt_phases(fig, phases)

        # Add milestones
        if milestones:
            self._add_milestones(fig, milestones)

        # Add dependencies if enabled
        if self.config.show_dependencies:
            self._add_dependencies(fig, phases)

        # Configure layout
        self._configure_timeline_layout(fig, phases, title or self.config.title)

        return fig

    def create_resource_utilization_chart(
        self,
        phases: list[MissionPhase],
        resource_types: list[str] | None = None
    ) -> go.Figure:
        """
        Create resource utilization visualization.

        Args:
            phases: List of mission phases with resource data
            resource_types: List of resource types to track

        Returns
        -------
            Plotly Figure with resource utilization
        """
        if not phases:
            return self._create_empty_plot("No phase data provided")

        # Extract all resource types if not specified
        if resource_types is None:
            resource_types_set: set[str] = set()
            for phase in phases:
                if phase.resources:
                    resource_types_set.update(phase.resources.keys())
            resource_types = sorted(resource_types_set)

        if not resource_types:
            return self._create_empty_plot("No resource data available")

        # Create subplots for each resource type
        fig = make_subplots(
            rows=len(resource_types), cols=1,
            subplot_titles=[f"{res.title()} Utilization" for res in resource_types],
            vertical_spacing=0.1,
            shared_xaxes=True
        )

        # Generate timeline data
        start_date = min(phase.start_date for phase in phases)
        end_date = max(phase.end_date for phase in phases)
        date_range = pd.date_range(start_date, end_date, freq="D")

        colors = px.colors.qualitative.Set1[:len(resource_types)]

        for idx, resource_type in enumerate(resource_types):
            # Calculate daily resource utilization
            daily_utilization = np.zeros(len(date_range))

            for phase in phases:
                if phase.resources and resource_type in phase.resources:
                    phase_start_idx = max(0, (phase.start_date - start_date).days)
                    phase_end_idx = min(len(date_range), (phase.end_date - start_date).days + 1)

                    if phase_end_idx > phase_start_idx and phase.resources:
                        daily_amount = phase.resources[resource_type] / (phase_end_idx - phase_start_idx)
                        daily_utilization[phase_start_idx:phase_end_idx] += daily_amount

            # Add utilization trace
            fig.add_trace(
                go.Scatter(
                    x=date_range,
                    y=daily_utilization,
                    mode="lines",
                    name=resource_type.title(),
                    fill="tonexty" if idx == 0 else "tonexty",
                    line={"color": colors[idx % len(colors)], "width": 2},
                    showlegend=True
                ),
                row=idx + 1, col=1
            )

            # Add peak utilization annotation
            peak_util = np.max(daily_utilization)
            peak_date = date_range[np.argmax(daily_utilization)]

            fig.add_annotation(
                x=peak_date,
                y=peak_util,
                text=f"Peak: {peak_util:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors[idx % len(colors)],
                row=idx + 1, col=1
            )

        # Update layout
        fig.update_layout(
            title="Mission Resource Utilization Over Time",
            template=self.config.theme,
            height=200 * len(resource_types) + 200,
            width=self.config.width,
            showlegend=True
        )

        return fig

    def create_critical_path_analysis(
        self,
        phases: list[MissionPhase]
    ) -> go.Figure:
        """
        Create critical path analysis visualization.

        Args:
            phases: List of mission phases with dependencies

        Returns
        -------
            Plotly Figure with critical path analysis
        """
        if not phases:
            return self._create_empty_plot("No phase data provided")

        # Calculate critical path (simplified implementation)
        self._calculate_critical_path(phases)

        fig = go.Figure()

        # Add all phases
        for phase in phases:
            color = self.config.critical_path_color if phase.critical_path else self._get_phase_color(phase.category)
            width = self.config.critical_path_width if phase.critical_path else 2

            fig.add_trace(
                go.Scatter(
                    x=[phase.start_date, phase.end_date],
                    y=[phase.name, phase.name],
                    mode="lines+markers",
                    name=phase.name,
                    line={"color": color, "width": width},
                    marker={"size": 8},
                    hovertemplate=f"<b>{phase.name}</b><br>"
                                f"Duration: {(phase.end_date - phase.start_date).days} days<br>"
                                f"Critical: {'Yes' if phase.critical_path else 'No'}<br>"
                                f"Risk: {phase.risk_level}<extra></extra>"
                )
            )

        # Add dependency arrows
        self._add_dependency_arrows(fig, phases)

        # Update layout
        fig.update_layout(
            title="Critical Path Analysis",
            xaxis_title="Timeline",
            yaxis_title="Mission Phases",
            template=self.config.theme,
            height=max(600, len(phases) * 40),
            width=self.config.width,
            showlegend=False
        )

        return fig

    def create_mission_dashboard(
        self,
        phases: list[MissionPhase],
        milestones: list[MissionMilestone],
        current_date: datetime | None = None
    ) -> go.Figure:
        """
        Create comprehensive mission dashboard.

        Args:
            phases: List of mission phases
            milestones: List of mission milestones
            current_date: Current date for progress tracking

        Returns
        -------
            Plotly Figure with mission dashboard
        """
        if current_date is None:
            current_date = datetime.now()

        # Create 2x2 subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Mission Timeline",
                "Phase Status",
                "Upcoming Milestones",
                "Risk Assessment"
            ],
            specs=[
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.15
        )

        # 1. Mission Timeline (Top, full width)
        self._add_dashboard_timeline(fig, phases, milestones, current_date, row=1, col=1)

        # 2. Phase Status
        self._add_phase_status_chart(fig, phases, current_date, row=2, col=1)

        # 3. Upcoming Milestones
        self._add_upcoming_milestones_table(fig, milestones, current_date, row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f"Mission Dashboard - {current_date.strftime('%Y-%m-%d')}",
            template=self.config.theme,
            height=900,
            width=self.config.width,
            showlegend=True
        )

        return fig

    def create_risk_timeline(
        self,
        phases: list[MissionPhase],
        risk_events: list[dict[str, Any]] | None = None
    ) -> go.Figure:
        """
        Create risk assessment timeline.

        Args:
            phases: List of mission phases with risk levels
            risk_events: Optional list of specific risk events

        Returns
        -------
            Plotly Figure with risk timeline
        """
        fig = go.Figure()

        # Risk level colors
        risk_colors = {
            "Low": "#2ECC71",
            "Medium": "#F39C12",
            "High": "#E74C3C",
            "Critical": "#8E44AD"
        }

        # Add phase risk levels
        for phase in phases:
            color = risk_colors.get(phase.risk_level, "#95A5A6")

            fig.add_trace(
                go.Scatter(
                    x=[phase.start_date, phase.end_date],
                    y=[phase.name, phase.name],
                    mode="lines+markers",
                    name=f"{phase.name} ({phase.risk_level})",
                    line={"color": color, "width": 6},
                    marker={"size": 10, "color": color},
                    hovertemplate=f"<b>{phase.name}</b><br>"
                                f"Risk Level: {phase.risk_level}<br>"
                                f"Duration: {(phase.end_date - phase.start_date).days} days<extra></extra>"
                )
            )

        # Add risk events if provided
        if risk_events:
            for event in risk_events:
                event_date = event.get("date")
                event_name = event.get("name", "Risk Event")
                event_risk = event.get("risk_level", "Medium")

                if event_date:
                    fig.add_vline(
                        x=event_date,
                        line_dash="dash",
                        line_color=risk_colors.get(event_risk, "#95A5A6"),
                        annotation_text=event_name,
                        annotation_position="top"
                    )

        # Update layout
        fig.update_layout(
            title="Mission Risk Timeline",
            xaxis_title="Timeline",
            yaxis_title="Mission Phases",
            template=self.config.theme,
            height=max(600, len(phases) * 40),
            width=self.config.width,
            showlegend=True
        )

        return fig

    def _add_gantt_phases(self, fig: go.Figure, phases: list[MissionPhase]) -> None:
        """Add mission phases as Gantt chart bars."""
        for phase in phases:
            color = self._get_phase_color(phase.category)

            fig.add_trace(
                go.Scatter(
                    x=[phase.start_date, phase.end_date],
                    y=[phase.name, phase.name],
                    mode="lines+markers",
                    name=phase.name,
                    line={"color": color, "width": 8},
                    marker={"size": 10, "color": color},
                    hovertemplate=f"<b>{phase.name}</b><br>"
                                f"Start: {phase.start_date.strftime('%Y-%m-%d')}<br>"
                                f"End: {phase.end_date.strftime('%Y-%m-%d')}<br>"
                                f"Duration: {(phase.end_date - phase.start_date).days} days<br>"
                                f"Category: {phase.category}<br>"
                                f"Cost: ${phase.cost/1e6:.1f}M<extra></extra>"
                )
            )

    def _add_milestones(self, fig: go.Figure, milestones: list[MissionMilestone]) -> None:
        """Add milestones to timeline."""
        milestone_y_positions: dict[str, str] = {}

        for milestone in milestones:
            # Create unique y-position for milestone
            if milestone.category not in milestone_y_positions:
                milestone_y_positions[milestone.category] = f"Milestone {len(milestone_y_positions) + 1}"

            y_pos = milestone_y_positions[milestone.category]

            fig.add_trace(
                go.Scatter(
                    x=[milestone.date],
                    y=[y_pos],
                    mode="markers+text",
                    name=milestone.name,
                    text=[milestone.name],
                    textposition="top center",
                    marker={
                        "size": self.config.milestone_size,
                        "color": self.config.milestone_color,
                        "symbol": "diamond",
                        "line": {"width": 2, "color": "white"}
                    },
                    hovertemplate=f"<b>{milestone.name}</b><br>"
                                f"Date: {milestone.date.strftime('%Y-%m-%d')}<br>"
                                f"Category: {milestone.category}<br>"
                                f"Description: {milestone.description}<extra></extra>"
                )
            )

    def _add_dependencies(self, fig: go.Figure, phases: list[MissionPhase]) -> None:
        """Add dependency arrows between phases."""
        phase_dict = {phase.name: phase for phase in phases}

        for phase in phases:
            if phase.dependencies:
                for dep_name in phase.dependencies:
                    if dep_name in phase_dict:
                        dep_phase = phase_dict[dep_name]

                    # Add dependency arrow
                    fig.add_annotation(
                        x=dep_phase.end_date,
                        y=dep_phase.name,
                        ax=phase.start_date,
                        ay=phase.name,
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="gray",
                        opacity=0.6
                    )

    def _add_dashboard_timeline(
        self,
        fig: go.Figure,
        phases: list[MissionPhase],
        milestones: list[MissionMilestone],
        current_date: datetime,
        row: int,
        col: int
    ) -> None:
        """Add simplified timeline for dashboard."""
        # Add current date line
        fig.add_vline(
            x=current_date,
            line_dash="dash",
            line_color="red",
            annotation_text="Today",
            annotation_position="top",
            row=row
        )

        # Add key phases only (first 5)
        for i, phase in enumerate(phases[:5]):
            color = self._get_phase_color(phase.category)

            fig.add_trace(
                go.Scatter(
                    x=[phase.start_date, phase.end_date],
                    y=[i, i],
                    mode="lines",
                    name=phase.name,
                    line={"color": color, "width": 6},
                    showlegend=False
                ),
                row=row, col=col
            )

    def _add_phase_status_chart(
        self,
        fig: go.Figure,
        phases: list[MissionPhase],
        current_date: datetime,
        row: int,
        col: int
    ) -> None:
        """Add phase status bar chart."""
        # Calculate phase status
        status_counts = {"Not Started": 0, "In Progress": 0, "Completed": 0}

        for phase in phases:
            if current_date < phase.start_date:
                status_counts["Not Started"] += 1
            elif current_date > phase.end_date:
                status_counts["Completed"] += 1
            else:
                status_counts["In Progress"] += 1

        statuses = list(status_counts.keys())
        counts = list(status_counts.values())
        colors = [self.config.development_color, self.config.launch_color, self.config.completion_color]

        fig.add_trace(
            go.Bar(
                x=statuses,
                y=counts,
                marker={"color": colors},
                text=counts,
                textposition="auto",
                showlegend=False
            ),
            row=row, col=col
        )

    def _add_upcoming_milestones_table(
        self,
        fig: go.Figure,
        milestones: list[MissionMilestone],
        current_date: datetime,
        row: int,
        col: int
    ) -> None:
        """Add upcoming milestones table."""
        # Filter upcoming milestones
        upcoming = [m for m in milestones if m.date >= current_date]
        upcoming.sort(key=lambda x: x.date)

        if upcoming:
            table_data = []
            for milestone in upcoming[:5]:  # Top 5 upcoming
                days_until = (milestone.date - current_date).days
                table_data.append([
                    milestone.name,
                    milestone.date.strftime("%Y-%m-%d"),
                    f"{days_until} days",
                    milestone.category
                ])

            fig.add_trace(
                go.Table(
                    header={
                        "values": ["Milestone", "Date", "Days Until", "Category"],
                        "fill_color": self.config.development_color,
                        "align": "left",
                        "font": {"color": "white", "size": 12}
                    },
                    cells={
                        "values": list(zip(*table_data, strict=False)),
                        "fill_color": "lightblue",
                        "align": "left",
                        "font": {"size": 11}
                    }
                ),
                row=row, col=col
            )
        else:
            # Add empty table message
            fig.add_annotation(
                text="No upcoming milestones",
                xref=f"x{col}", yref=f"y{col}",
                x=0.5, y=0.5, showarrow=False,
                font={"size": 14, "color": "gray"},
                row=row, col=col
            )

    def _calculate_critical_path(self, phases: list[MissionPhase]) -> list[MissionPhase]:
        """Calculate critical path (simplified implementation)."""
        # Mark phases on critical path based on dependencies and duration
        phase_dict = {phase.name: phase for phase in phases}

        # Simple heuristic: longest dependent chain
        for phase in phases:
            if not phase.dependencies:
                phase.critical_path = True
            else:
                # Check if any dependency is on critical path
                for dep_name in phase.dependencies:
                    if dep_name in phase_dict and phase_dict[dep_name].critical_path:
                        phase.critical_path = True
                        break

        return [phase for phase in phases if phase.critical_path]

    def _add_dependency_arrows(self, fig: go.Figure, phases: list[MissionPhase]) -> None:
        """Add dependency arrows for critical path visualization."""
        phase_dict = {phase.name: phase for phase in phases}

        for phase in phases:
            if phase.dependencies:
                for dep_name in phase.dependencies:
                    if dep_name in phase_dict:
                        dep_phase = phase_dict[dep_name]

                    arrow_color = self.config.critical_path_color if (
                        phase.critical_path and dep_phase.critical_path
                    ) else "gray"

                    fig.add_annotation(
                        x=dep_phase.end_date,
                        y=dep_phase.name,
                        ax=phase.start_date,
                        ay=phase.name,
                        arrowhead=2,
                        arrowcolor=arrow_color,
                        arrowwidth=2 if phase.critical_path else 1
                    )

    def _get_phase_color(self, category: str) -> str:
        """Get color for phase category."""
        color_map = {
            "Development": self.config.development_color,
            "Testing": self.config.testing_color,
            "Launch": self.config.launch_color,
            "Operations": self.config.operations_color,
            "Completion": self.config.completion_color
        }
        return color_map.get(category, "#95A5A6")

    def _configure_timeline_layout(
        self,
        fig: go.Figure,
        phases: list[MissionPhase],
        title: str
    ) -> None:
        """Configure timeline layout."""
        fig.update_layout(
            title=title,
            xaxis_title="Timeline",
            yaxis_title="Mission Phases",
            template=self.config.theme,
            height=max(self.config.height, len(phases) * 50),
            width=self.config.width,
            showlegend=False,
            font={"family": self.config.font_family},
            hovermode="closest"
        )

        # Set y-axis to show all phases
        phase_names = [phase.name for phase in phases]
        fig.update_yaxes(categoryorder="array", categoryarray=phase_names)

    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16, "color": "gray"}
        )
        return fig


def create_sample_mission_timeline() -> go.Figure:
    """
    Create a sample lunar mission timeline for demonstration.

    Returns
    -------
        Plotly Figure with sample mission timeline
    """
    # Define sample mission phases
    base_date = datetime(2025, 1, 1)

    phases = [
        MissionPhase(
            name="Mission Design",
            start_date=base_date,
            end_date=base_date + timedelta(days=365),
            category="Development",
            cost=50e6,
            risk_level="Medium"
        ),
        MissionPhase(
            name="Spacecraft Development",
            start_date=base_date + timedelta(days=180),
            end_date=base_date + timedelta(days=730),
            category="Development",
            dependencies=["Mission Design"],
            cost=200e6,
            risk_level="High"
        ),
        MissionPhase(
            name="Integration & Testing",
            start_date=base_date + timedelta(days=600),
            end_date=base_date + timedelta(days=900),
            category="Testing",
            dependencies=["Spacecraft Development"],
            cost=75e6,
            risk_level="Medium"
        ),
        MissionPhase(
            name="Launch Campaign",
            start_date=base_date + timedelta(days=870),
            end_date=base_date + timedelta(days=930),
            category="Launch",
            dependencies=["Integration & Testing"],
            cost=100e6,
            risk_level="High"
        ),
        MissionPhase(
            name="Lunar Operations",
            start_date=base_date + timedelta(days=935),
            end_date=base_date + timedelta(days=1300),
            category="Operations",
            dependencies=["Launch Campaign"],
            cost=150e6,
            risk_level="Medium"
        )
    ]

    # Define sample milestones
    milestones = [
        MissionMilestone(
            name="PDR",
            date=base_date + timedelta(days=120),
            category="Design",
            description="Preliminary Design Review"
        ),
        MissionMilestone(
            name="CDR",
            date=base_date + timedelta(days=240),
            category="Design",
            description="Critical Design Review"
        ),
        MissionMilestone(
            name="Launch",
            date=base_date + timedelta(days=900),
            category="Launch",
            description="Mission Launch"
        ),
        MissionMilestone(
            name="Lunar Arrival",
            date=base_date + timedelta(days=906),
            category="Operations",
            description="Lunar Orbit Insertion"
        )
    ]

    # Create visualizer and timeline
    visualizer = MissionVisualizer()
    return visualizer.create_mission_timeline(phases, milestones)
