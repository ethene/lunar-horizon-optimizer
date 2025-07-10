"""
Optimization Results Visualization Module.

Provides comprehensive visualization for multi-objective optimization results including
Pareto front analysis, optimization convergence tracking, solution comparison,
and interactive decision-making support tools.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.optimization.pareto_analysis import OptimizationResult, ParetoAnalyzer


@dataclass
class ParetoPlotConfig:
    """Configuration for optimization visualization plots."""

    # Plot dimensions and layout
    width: int = 1200
    height: int = 800
    title: str = "Multi-Objective Optimization Results"

    # Pareto front visualization
    pareto_color: str = "#ff6b6b"
    pareto_size: int = 8
    dominated_color: str = "#4ecdc4"
    dominated_size: int = 5
    dominated_opacity: float = 0.6

    # Convergence tracking
    show_convergence: bool = True
    generation_colors: list[str] | None = None

    # Interactive features
    enable_hover: bool = True
    show_preference_lines: bool = True
    enable_selection: bool = True

    # Styling
    background_color: str = "#ffffff"
    grid_color: str = "#e0e0e0"
    text_color: str = "#333333"
    theme: str = "plotly_white"

    def __post_init__(self) -> None:
        if self.generation_colors is None:
            self.generation_colors = px.colors.qualitative.Set3


class OptimizationVisualizer:
    """
    Interactive optimization results visualization using Plotly.

    Provides comprehensive visualization of multi-objective optimization including:
    - Pareto front analysis and ranking
    - Optimization convergence tracking
    - Solution comparison and selection tools
    - Interactive decision-making support
    """

    def __init__(self, config: ParetoPlotConfig | None = None) -> None:
        """
        Initialize optimization visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or ParetoPlotConfig()
        self.pareto_analyzer = ParetoAnalyzer()

    def create_pareto_front_plot(
        self,
        optimization_result: OptimizationResult,
        objective_names: list[str] | None = None,
        show_dominated: bool = True,
        preference_weights: list[float] | None = None,
    ) -> go.Figure:
        """
        Create comprehensive Pareto front visualization.

        Args:
            optimization_result: Results from optimization run
            objective_names: Names for objectives (default: ["Objective 1", ...])
            show_dominated: Whether to show dominated solutions
            preference_weights: User preference weights for ranking

        Returns
        -------
            Plotly Figure with Pareto front visualization
        """
        pareto_solutions = optimization_result.pareto_solutions
        all_solutions = optimization_result.all_solutions

        if not pareto_solutions:
            return self._create_empty_plot("No Pareto solutions found")

        n_objectives = len(pareto_solutions[0]["objectives"])

        if objective_names is None:
            objective_names = [f"Objective {i+1}" for i in range(n_objectives)]

        if n_objectives == 2:
            return self._create_2d_pareto_plot(
                pareto_solutions, all_solutions, objective_names,
                show_dominated, preference_weights,
            )
        if n_objectives == 3:
            return self._create_3d_pareto_plot(
                pareto_solutions, all_solutions, objective_names,
                show_dominated, preference_weights,
            )
        return self._create_parallel_coordinates_plot(
            pareto_solutions, all_solutions, objective_names, preference_weights,
        )

    def create_optimization_convergence_plot(
        self,
        optimization_result: OptimizationResult,
        objective_names: list[str] | None = None,
    ) -> go.Figure:
        """
        Create optimization convergence visualization.

        Args:
            optimization_result: Results from optimization run
            objective_names: Names for objectives

        Returns
        -------
            Plotly Figure with convergence analysis
        """
        if not hasattr(optimization_result, "generation_history"):
            return self._create_empty_plot("No convergence data available")

        generation_history = optimization_result.generation_history
        n_generations = len(generation_history)
        n_objectives = len(generation_history[0][0]["objectives"])

        if objective_names is None:
            objective_names = [f"Objective {i+1}" for i in range(n_objectives)]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Best Solutions Evolution",
                "Hypervolume Convergence",
                "Solution Count per Generation",
                "Objective Space Coverage",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # Track best solutions per generation
        generations = list(range(n_generations))
        best_objectives: list[list[float]] = [[] for _ in range(n_objectives)]
        hypervolumes = []
        solution_counts = []

        for _gen, solutions in enumerate(generation_history):
            if solutions:
                # Find best solution for each objective
                for obj_idx in range(n_objectives):
                    obj_values = [sol["objectives"][obj_idx] for sol in solutions]
                    best_objectives[obj_idx].append(min(obj_values))

                # Calculate hypervolume (simplified)
                pareto_sols = self.pareto_analyzer.find_pareto_front(solutions)
                if len(pareto_sols) > 1 and n_objectives <= 3:
                    hv = self._calculate_hypervolume(pareto_sols, n_objectives)
                    hypervolumes.append(hv)
                else:
                    hypervolumes.append(0)

                solution_counts.append(len(solutions))
            else:
                for obj_idx in range(n_objectives):
                    best_objectives[obj_idx].append(np.nan)
                hypervolumes.append(0)
                solution_counts.append(0)

        # Plot 1: Best Solutions Evolution
        colors = px.colors.qualitative.Set1[:n_objectives]
        for _obj_idx, (values, name, color) in enumerate(zip(best_objectives, objective_names, colors, strict=False)):
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=values,
                    mode="lines+markers",
                    name=f"Best {name}",
                    line={"color": color, "width": 2},
                    marker={"size": 6},
                ),
                row=1, col=1,
            )

        # Plot 2: Hypervolume Convergence
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=hypervolumes,
                mode="lines+markers",
                name="Hypervolume",
                line={"color": "purple", "width": 3},
                marker={"size": 6},
                showlegend=False,
            ),
            row=1, col=2,
        )

        # Plot 3: Solution Count per Generation
        fig.add_trace(
            go.Bar(
                x=generations,
                y=solution_counts,
                name="Solution Count",
                marker={"color": "lightblue"},
                showlegend=False,
            ),
            row=2, col=1,
        )

        # Plot 4: Objective Space Coverage
        if n_objectives >= 2:
            # Use final generation solutions
            final_solutions = generation_history[-1] if generation_history else []
            if final_solutions:
                obj1_vals = [sol["objectives"][0] for sol in final_solutions]
                obj2_vals = [sol["objectives"][1] for sol in final_solutions]

                fig.add_trace(
                    go.Scatter(
                        x=obj1_vals,
                        y=obj2_vals,
                        mode="markers",
                        name="Final Generation",
                        marker={
                            "size": 6,
                            "color": "red",
                            "opacity": 0.7,
                        },
                        showlegend=False,
                    ),
                    row=2, col=2,
                )

        # Update layout
        fig.update_xaxes(title_text="Generation", row=1, col=1)
        fig.update_yaxes(title_text="Objective Value", row=1, col=1)

        fig.update_xaxes(title_text="Generation", row=1, col=2)
        fig.update_yaxes(title_text="Hypervolume", row=1, col=2)

        fig.update_xaxes(title_text="Generation", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        if n_objectives >= 2:
            fig.update_xaxes(title_text=objective_names[0], row=2, col=2)
            fig.update_yaxes(title_text=objective_names[1], row=2, col=2)

        fig.update_layout(
            title="Optimization Convergence Analysis",
            template=self.config.theme,
            height=800,
            width=1400,
            showlegend=True,
        )

        return fig

    def create_solution_comparison_plot(
        self,
        solutions: list[dict[str, Any]],
        solution_labels: list[str] | None = None,
        objective_names: list[str] | None = None,
        parameter_names: list[str] | None = None,
    ) -> go.Figure:
        """
        Create detailed solution comparison visualization.

        Args:
            solutions: List of solutions to compare
            solution_labels: Labels for each solution
            objective_names: Names for objectives
            parameter_names: Names for parameters

        Returns
        -------
            Plotly Figure with solution comparison
        """
        if not solutions:
            return self._create_empty_plot("No solutions provided for comparison")

        n_solutions = len(solutions)
        n_objectives = len(solutions[0]["objectives"])
        n_parameters = len(solutions[0]["parameters"])

        if solution_labels is None:
            solution_labels = [f"Solution {i+1}" for i in range(n_solutions)]
        if objective_names is None:
            objective_names = [f"Objective {i+1}" for i in range(n_objectives)]
        if parameter_names is None:
            parameter_names = [f"Parameter {i+1}" for i in range(n_parameters)]

        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Objective Values Comparison",
                "Parameter Values Comparison",
                "Solution Ranking",
                "Trade-off Analysis",
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "scatter"}],
            ],
        )

        # Plot 1: Objective Values Comparison
        px.colors.qualitative.Set1[:n_solutions]

        for obj_idx, obj_name in enumerate(objective_names):
            values = [sol["objectives"][obj_idx] for sol in solutions]

            fig.add_trace(
                go.Bar(
                    x=solution_labels,
                    y=values,
                    name=obj_name,
                    text=[f"{v:.2e}" for v in values],
                    textposition="auto",
                ),
                row=1, col=1,
            )

        # Plot 2: Parameter Values Comparison
        for param_idx, param_name in enumerate(parameter_names):
            values = [sol["parameters"][param_idx] for sol in solutions]

            fig.add_trace(
                go.Bar(
                    x=solution_labels,
                    y=values,
                    name=param_name,
                    text=[f"{v:.2f}" for v in values],
                    textposition="auto",
                    showlegend=False,
                ),
                row=1, col=2,
            )

        # Plot 3: Solution Ranking Table
        ranking_data = []
        for i, (sol, label) in enumerate(zip(solutions, solution_labels, strict=False)):
            ranking_data.append([
                label,
                f"{sol['objectives'][0]:.2e}",
                f"{sol['objectives'][1] if n_objectives > 1 else 'N/A'}",
                f"{sol['objectives'][2] if n_objectives > 2 else 'N/A'}",
                f"Rank {i+1}",
            ])

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Solution", obj_names[0] if len(obj_names := objective_names) > 0 else "Obj1",
                           obj_names[1] if len(obj_names) > 1 else "Obj2",
                           obj_names[2] if len(obj_names) > 2 else "Obj3", "Ranking"],
                    "fill_color": "lightblue",
                    "align": "center",
                    "font": {"size": 12},
                },
                cells={
                    "values": list(zip(*ranking_data, strict=False)),
                    "fill_color": "white",
                    "align": "center",
                    "font": {"size": 11},
                },
            ),
            row=2, col=1,
        )

        # Plot 4: Trade-off Analysis (if >=2 objectives)
        if n_objectives >= 2:
            obj1_vals = [sol["objectives"][0] for sol in solutions]
            obj2_vals = [sol["objectives"][1] for sol in solutions]

            fig.add_trace(
                go.Scatter(
                    x=obj1_vals,
                    y=obj2_vals,
                    mode="markers+text",
                    text=solution_labels,
                    textposition="top center",
                    marker={
                        "size": 12,
                        "color": list(range(n_solutions)),
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Solution Index", "x": 1.0},
                    },
                    name="Solutions",
                    showlegend=False,
                ),
                row=2, col=2,
            )

        # Update layout
        fig.update_layout(
            title="Solution Comparison Analysis",
            template=self.config.theme,
            height=1000,
            width=1400,
            showlegend=True,
        )

        return fig

    def create_preference_analysis_plot(
        self,
        pareto_solutions: list[dict[str, Any]],
        preference_weights: list[float],
        objective_names: list[str] | None = None,
    ) -> go.Figure:
        """
        Create preference-based solution ranking visualization.

        Args:
            pareto_solutions: List of Pareto-optimal solutions
            preference_weights: User preference weights for objectives
            objective_names: Names for objectives

        Returns
        -------
            Plotly Figure with preference analysis
        """
        if not pareto_solutions:
            return self._create_empty_plot("No Pareto solutions provided")

        n_objectives = len(pareto_solutions[0]["objectives"])

        if objective_names is None:
            objective_names = [f"Objective {i+1}" for i in range(n_objectives)]

        # Rank solutions by preference
        ranked_solutions = self.pareto_analyzer.rank_solutions_by_preference(
            pareto_solutions, preference_weights,
        )

        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Preference-Based Ranking",
                "Weighted Objective Values",
                "Preference Sensitivity",
                "Top Solutions Comparison",
            ],
        )

        # Extract data
        scores = [score for score, _ in ranked_solutions]
        solution_indices = list(range(len(ranked_solutions)))

        # Plot 1: Preference-Based Ranking
        fig.add_trace(
            go.Bar(
                x=solution_indices,
                y=scores,
                name="Preference Score",
                marker={
                    "color": scores,
                    "colorscale": "RdYlGn",
                    "showscale": True,
                    "colorbar": {"title": "Preference Score", "x": 0.48},
                },
                text=[f"{s:.3f}" for s in scores],
                textposition="auto",
            ),
            row=1, col=1,
        )

        # Plot 2: Weighted Objective Values
        colors = px.colors.qualitative.Set1[:n_objectives]
        for obj_idx, (obj_name, weight, color) in enumerate(zip(objective_names, preference_weights, colors, strict=False)):
            weighted_values = []
            for _, solution in ranked_solutions:
                obj_val = solution["objectives"][obj_idx]
                # Normalize and weight (simplified)
                weighted_val = obj_val * weight
                weighted_values.append(weighted_val)

            fig.add_trace(
                go.Scatter(
                    x=solution_indices,
                    y=weighted_values,
                    mode="lines+markers",
                    name=f"{obj_name} (w={weight:.2f})",
                    line={"color": color, "width": 2},
                    marker={"size": 6},
                ),
                row=1, col=2,
            )

        # Plot 3: Preference Sensitivity Analysis
        # Vary weights slightly and show impact
        sensitivity_data = self._calculate_preference_sensitivity(
            pareto_solutions, preference_weights,
        )

        weight_variations = sensitivity_data["weight_variations"]
        ranking_changes = sensitivity_data["ranking_changes"]

        fig.add_trace(
            go.Heatmap(
                z=ranking_changes,
                x=[f"±{v:.1%}" for v in weight_variations],
                y=objective_names,
                colorscale="RdBu",
                name="Ranking Sensitivity",
                showscale=True,
                colorbar={"title": "Rank Change", "x": 1.0},
            ),
            row=2, col=1,
        )

        # Plot 4: Top Solutions Comparison
        top_n = min(5, len(ranked_solutions))
        top_solutions = ranked_solutions[:top_n]

        for obj_idx, obj_name in enumerate(objective_names):
            values = [sol["objectives"][obj_idx] for _, sol in top_solutions]
            labels = [f"Sol {i+1}" for i in range(top_n)]

            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=f"Top {obj_name}",
                    showlegend=False,
                ),
                row=2, col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Preference Analysis (Weights: {preference_weights})",
            template=self.config.theme,
            height=1000,
            width=1400,
        )

        return fig

    def _create_2d_pareto_plot(
        self,
        pareto_solutions: list[dict[str, Any]],
        all_solutions: list[dict[str, Any]],
        objective_names: list[str],
        show_dominated: bool,
        preference_weights: list[float] | None,
    ) -> go.Figure:
        """Create 2D Pareto front plot."""
        fig = go.Figure()

        # Plot dominated solutions
        if show_dominated and all_solutions:
            dominated_sols = [sol for sol in all_solutions
                            if sol not in pareto_solutions]
            if dominated_sols:
                obj1_dom = [sol["objectives"][0] for sol in dominated_sols]
                obj2_dom = [sol["objectives"][1] for sol in dominated_sols]

                fig.add_trace(
                    go.Scatter(
                        x=obj1_dom,
                        y=obj2_dom,
                        mode="markers",
                        name="Dominated Solutions",
                        marker={
                            "size": self.config.dominated_size,
                            "color": self.config.dominated_color,
                            "opacity": self.config.dominated_opacity,
                        },
                    ),
                )

        # Plot Pareto front
        obj1_pareto = [sol["objectives"][0] for sol in pareto_solutions]
        obj2_pareto = [sol["objectives"][1] for sol in pareto_solutions]

        # Color by preference if weights provided
        if preference_weights:
            ranked_solutions = self.pareto_analyzer.rank_solutions_by_preference(
                pareto_solutions, preference_weights,
            )
            [score for score, _ in ranked_solutions]
            # Map back to original order
            score_map = {id(sol): score for score, sol in ranked_solutions}
            colors = [score_map.get(id(sol), 0) for sol in pareto_solutions]

            fig.add_trace(
                go.Scatter(
                    x=obj1_pareto,
                    y=obj2_pareto,
                    mode="markers+lines",
                    name="Pareto Front",
                    marker={
                        "size": self.config.pareto_size,
                        "color": colors,
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Preference Score"},
                    },
                    line={"color": self.config.pareto_color, "width": 2},
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=obj1_pareto,
                    y=obj2_pareto,
                    mode="markers+lines",
                    name="Pareto Front",
                    marker={
                        "size": self.config.pareto_size,
                        "color": self.config.pareto_color,
                    },
                    line={"color": self.config.pareto_color, "width": 2},
                ),
            )

        fig.update_layout(
            title=self.config.title,
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def _create_3d_pareto_plot(
        self,
        pareto_solutions: list[dict[str, Any]],
        all_solutions: list[dict[str, Any]],
        objective_names: list[str],
        show_dominated: bool,
        preference_weights: list[float] | None,
    ) -> go.Figure:
        """Create 3D Pareto front plot."""
        fig = go.Figure()

        # Plot dominated solutions
        if show_dominated and all_solutions:
            dominated_sols = [sol for sol in all_solutions
                            if sol not in pareto_solutions]
            if dominated_sols:
                obj1_dom = [sol["objectives"][0] for sol in dominated_sols]
                obj2_dom = [sol["objectives"][1] for sol in dominated_sols]
                obj3_dom = [sol["objectives"][2] for sol in dominated_sols]

                fig.add_trace(
                    go.Scatter3d(
                        x=obj1_dom,
                        y=obj2_dom,
                        z=obj3_dom,
                        mode="markers",
                        name="Dominated Solutions",
                        marker={
                            "size": self.config.dominated_size,
                            "color": self.config.dominated_color,
                            "opacity": self.config.dominated_opacity,
                        },
                    ),
                )

        # Plot Pareto front
        obj1_pareto = [sol["objectives"][0] for sol in pareto_solutions]
        obj2_pareto = [sol["objectives"][1] for sol in pareto_solutions]
        obj3_pareto = [sol["objectives"][2] for sol in pareto_solutions]

        fig.add_trace(
            go.Scatter3d(
                x=obj1_pareto,
                y=obj2_pareto,
                z=obj3_pareto,
                mode="markers",
                name="Pareto Front",
                marker={
                    "size": self.config.pareto_size,
                    "color": self.config.pareto_color,
                },
            ),
        )

        fig.update_layout(
            title=self.config.title,
            scene={
                "xaxis_title": objective_names[0],
                "yaxis_title": objective_names[1],
                "zaxis_title": objective_names[2],
            },
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def _create_parallel_coordinates_plot(
        self,
        pareto_solutions: list[dict[str, Any]],
        all_solutions: list[dict[str, Any]],
        objective_names: list[str],
        preference_weights: list[float] | None,
    ) -> go.Figure:
        """Create parallel coordinates plot for >3 objectives."""
        fig = go.Figure()

        # Prepare data
        obj_data = []
        for sol in pareto_solutions:
            obj_data.append(sol["objectives"])

        obj_array = np.array(obj_data)

        # Create dimensions for parallel coordinates
        dimensions = []
        for i, name in enumerate(objective_names):
            dimensions.append(
                {
                    "range": [obj_array[:, i].min(), obj_array[:, i].max()],
                    "label": name,
                    "values": obj_array[:, i],
                },
            )

        fig.add_trace(
            go.Parcoords(
                line={
                    "color": list(range(len(pareto_solutions))),
                    "colorscale": "Viridis",
                    "showscale": True,
                },
                dimensions=dimensions,
            ),
        )

        fig.update_layout(
            title=f"Pareto Front - {len(objective_names)} Objectives",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16, "color": self.config.text_color},
        )
        return fig

    def _calculate_hypervolume(
        self,
        solutions: list[dict[str, Any]],
        n_objectives: int,
    ) -> float:
        """Calculate hypervolume (simplified implementation)."""
        if n_objectives > 3 or len(solutions) < 2:
            return 0.0

        # Simple hypervolume calculation for demonstration
        objectives = np.array([sol["objectives"][:n_objectives] for sol in solutions])

        # Use max values as reference point
        ref_point = objectives.max(axis=0) * 1.1

        # Simplified hypervolume calculation
        volume = 0.0
        for sol_obj in objectives:
            contribution = np.prod(ref_point - sol_obj)
            if contribution > 0:
                volume += contribution

        return volume

    def _calculate_preference_sensitivity(
        self,
        pareto_solutions: list[dict[str, Any]],
        base_weights: list[float],
    ) -> dict[str, Any]:
        """Calculate sensitivity of rankings to weight changes."""
        n_objectives = len(base_weights)
        weight_variations = [0.1, 0.2, 0.3]  # ±10%, ±20%, ±30%

        # Get base ranking
        base_ranking = self.pareto_analyzer.rank_solutions_by_preference(
            pareto_solutions, base_weights,
        )
        base_order = [id(sol) for _, sol in base_ranking]

        ranking_changes = []

        for obj_idx in range(n_objectives):
            obj_changes = []

            for variation in weight_variations:
                # Increase weight for this objective
                varied_weights = base_weights.copy()
                varied_weights[obj_idx] *= (1 + variation)
                # Normalize
                total = sum(varied_weights)
                varied_weights = [w/total for w in varied_weights]

                # Get new ranking
                new_ranking = self.pareto_analyzer.rank_solutions_by_preference(
                    pareto_solutions, varied_weights,
                )
                new_order = [id(sol) for _, sol in new_ranking]

                # Calculate rank changes
                rank_change = 0
                for i, sol_id in enumerate(base_order):
                    old_rank = i
                    new_rank = new_order.index(sol_id) if sol_id in new_order else len(new_order)
                    rank_change += abs(new_rank - old_rank)

                obj_changes.append(rank_change / len(base_order))

            ranking_changes.append(obj_changes)

        return {
            "weight_variations": weight_variations,
            "ranking_changes": ranking_changes,
        }


def create_quick_pareto_plot(
    optimization_result: dict[str, Any],
    objective_names: list[str] | None = None,
) -> go.Figure:
    """
    Quick function to create a simple Pareto front plot.

    Args:
        optimization_result: Results from optimization run
        objective_names: Names for objectives

    Returns
    -------
        Plotly Figure with Pareto front visualization
    """
    try:
        # Create visualizer
        visualizer = OptimizationVisualizer()

        # Convert to OptimizationResult if needed
        if isinstance(optimization_result, dict):
            pareto_front = optimization_result.get("pareto_front", [])
            if not pareto_front:
                return visualizer._create_empty_plot("No Pareto solutions in results")

            # Convert to expected format
            pareto_solutions = []
            for sol in pareto_front:
                if isinstance(sol, dict) and "objectives" in sol:
                    pareto_solutions.append(sol)
                else:
                    # Assume it's a list [objectives, parameters]
                    pareto_solutions.append({
                        "objectives": sol[0] if isinstance(sol, list | tuple) else sol,
                        "parameters": sol[1] if isinstance(sol, list | tuple) and len(sol) > 1 else [],
                    })

            opt_result = OptimizationResult(
                pareto_solutions=pareto_solutions,
                all_solutions=optimization_result.get("all_solutions", pareto_solutions),
                optimization_stats=optimization_result.get("stats", {}),
                generation_history=optimization_result.get("generation_history", []),
            )
        else:
            opt_result = optimization_result

        return visualizer.create_pareto_front_plot(
            opt_result,
            objective_names=objective_names,
        )

    except Exception as e:
        # Create error plot
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization failed: {e!s}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 16, "color": "red"},
        )
        return fig
