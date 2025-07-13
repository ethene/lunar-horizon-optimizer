#!/usr/bin/env python3
"""Enhanced Progress Tracking Module.

This module provides rich progress tracking capabilities for lunar mission
analysis with real-time updates, phase tracking, and solution monitoring.
"""

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn,
    TextColumn,
)
from rich.table import Table
from rich.layout import Layout
from rich.text import Text


@dataclass
class OptimizationSolution:
    """Represents a solution in the optimization process."""

    generation: int
    individual_id: int
    objectives: List[float]
    parameters: List[float]
    delta_v: float = 0.0
    cost: float = 0.0
    transfer_time: float = 0.0

    def __post_init__(self):
        # Extract common metrics from objectives if available
        if len(self.objectives) >= 2:
            self.delta_v = self.objectives[0] if self.objectives[0] > 0 else 0.0
            self.cost = self.objectives[1] if len(self.objectives) > 1 else 0.0
            self.transfer_time = self.objectives[2] if len(self.objectives) > 2 else 0.0


@dataclass
class AnalysisPhase:
    """Represents a phase in the analysis pipeline."""

    name: str
    description: str
    weight: float  # Relative weight in overall progress (0-1)
    estimated_duration: float  # Estimated duration in seconds
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    progress: float = 0.0  # Phase-specific progress (0-1)

    @property
    def duration(self) -> Optional[float]:
        """Get actual duration if phase is completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time if phase is running."""
        if self.start_time and not self.end_time:
            return time.time() - self.start_time
        return self.duration


class EnhancedProgressTracker:
    """Enhanced progress tracker with rich terminal output."""

    def __init__(self, scenario_name: str = "Lunar Mission Analysis"):
        """Initialize the progress tracker.

        Args:
            scenario_name: Name of the scenario being analyzed
        """
        self.scenario_name = scenario_name
        self.console = Console()

        # Analysis phases
        self.phases = [
            AnalysisPhase(
                "Initialization",
                "Loading configuration and setting up analysis",
                0.05,
                2,
            ),
            AnalysisPhase(
                "Trajectory Generation",
                "Computing Earth-Moon transfer trajectories",
                0.15,
                10,
            ),
            AnalysisPhase(
                "Global Optimization",
                "Multi-objective optimization with PyGMO",
                0.50,
                60,
            ),
            AnalysisPhase(
                "Local Refinement", "JAX-based gradient refinement (optional)", 0.15, 15
            ),
            AnalysisPhase(
                "Economic Analysis", "Cost modeling and financial analysis", 0.10, 8
            ),
            AnalysisPhase(
                "Risk Assessment",
                "Monte Carlo sensitivity analysis (optional)",
                0.15,
                12,
            ),
            AnalysisPhase("Visualization", "Generating plots and dashboards", 0.05, 3),
            AnalysisPhase("Export", "Saving results and generating reports", 0.05, 2),
        ]

        self.current_phase_index = 0
        self.overall_start_time = time.time()
        self.optimization_solutions: List[OptimizationSolution] = []
        self.best_solutions: List[OptimizationSolution] = []

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.main_task: Optional[TaskID] = None
        self.phase_task: Optional[TaskID] = None

        # Threading
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None

        # Callbacks
        self.optimization_callback: Optional[Callable] = None

    def start(self, skip_phases: Optional[List[str]] = None) -> None:
        """Start the progress tracking.

        Args:
            skip_phases: List of phase names to skip (e.g., ["Local Refinement", "Risk Assessment"])
        """
        # Filter out skipped phases
        if skip_phases:
            self.phases = [p for p in self.phases if p.name not in skip_phases]

        # Renormalize weights
        total_weight = sum(p.weight for p in self.phases)
        for phase in self.phases:
            phase.weight = phase.weight / total_weight

        # Set up progress tracking
        total_steps = sum(p.estimated_duration for p in self.phases)
        self.main_task = self.progress.add_task(
            f"🚀 {self.scenario_name}", total=total_steps
        )

        # Start the first phase
        self._start_current_phase()

    def _start_current_phase(self) -> None:
        """Start the current phase."""
        if self.current_phase_index >= len(self.phases):
            return

        phase = self.phases[self.current_phase_index]
        phase.status = "running"
        phase.start_time = time.time()

        self.phase_task = self.progress.add_task(
            f"  📊 {phase.description}", total=phase.estimated_duration
        )

    def update_phase_progress(
        self, progress: float, message: Optional[str] = None
    ) -> None:
        """Update the progress of the current phase.

        Args:
            progress: Progress as a fraction (0.0 to 1.0)
            message: Optional status message
        """
        if self.current_phase_index >= len(self.phases):
            return

        phase = self.phases[self.current_phase_index]
        phase.progress = max(0.0, min(1.0, progress))

        if self.phase_task is not None:
            completed = phase.progress * phase.estimated_duration
            self.progress.update(self.phase_task, completed=completed)

            if message:
                self.progress.update(self.phase_task, description=f"  📊 {message}")

    def update_phase(self, phase_name: str, progress: float) -> None:
        """Compatibility method for LunarHorizonOptimizer integration.

        Args:
            phase_name: Name of the phase to update
            progress: Progress percentage (0.0 to 1.0)
        """
        # Find the matching phase by name
        phase_mapping = {
            "Trajectory Analysis": "Trajectory Generation",
            "Multi-objective Optimization": "Global Optimization",
            "Economic Analysis": "Economic Analysis",
            "Visualization Generation": "Visualization",
            "Results Compilation": "Export",
        }

        mapped_name = phase_mapping.get(phase_name, phase_name)

        # Find the phase index
        target_phase_index = None
        for i, phase in enumerate(self.phases):
            if (
                mapped_name.lower() in phase.name.lower()
                or phase.name.lower() in mapped_name.lower()
            ):
                target_phase_index = i
                break

        if target_phase_index is not None:
            # Complete previous phases if needed
            while self.current_phase_index < target_phase_index:
                self.complete_current_phase()

            # Update the current phase
            if self.current_phase_index == target_phase_index:
                self.update_phase_progress(
                    progress, f"{phase_name} - {int(progress*100)}% complete"
                )

    def complete_current_phase(self) -> None:
        """Mark the current phase as completed and move to the next."""
        if self.current_phase_index >= len(self.phases):
            return

        phase = self.phases[self.current_phase_index]
        phase.status = "completed"
        phase.end_time = time.time()
        phase.progress = 1.0

        if self.phase_task is not None:
            self.progress.update(self.phase_task, completed=phase.estimated_duration)
            self.progress.remove_task(self.phase_task)

        # Update overall progress
        if self.main_task is not None:
            completed_weight = sum(
                p.weight for p in self.phases[: self.current_phase_index + 1]
            )
            total_steps = sum(p.estimated_duration for p in self.phases)
            self.progress.update(
                self.main_task, completed=completed_weight * total_steps
            )

        # Move to next phase
        self.current_phase_index += 1
        if self.current_phase_index < len(self.phases):
            self._start_current_phase()

    def fail_current_phase(self, error_message: str) -> None:
        """Mark the current phase as failed."""
        if self.current_phase_index >= len(self.phases):
            return

        phase = self.phases[self.current_phase_index]
        phase.status = "failed"
        phase.end_time = time.time()

        if self.phase_task is not None:
            self.progress.update(
                self.phase_task,
                description=f"  ❌ {phase.description} - {error_message}",
            )

    def add_optimization_solution(self, solution: OptimizationSolution) -> None:
        """Add a new optimization solution."""
        self.optimization_solutions.append(solution)

        # Update best solutions (keep top 5 by delta-v)
        self.best_solutions = sorted(
            self.optimization_solutions,
            key=lambda s: s.delta_v if s.delta_v > 0 else float("inf"),
        )[:5]

    def create_live_display(self) -> Layout:
        """Create a live display layout showing progress and results."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=8),
            Layout(name="solutions", size=12),
            Layout(name="footer", size=2),
        )

        # Header
        header_text = Text(
            f"🌙 Lunar Horizon Optimizer - {self.scenario_name}",
            style="bold magenta",
            justify="center",
        )
        layout["header"].update(Panel(header_text))

        # Progress
        layout["progress"].update(self.progress)

        # Solutions table
        solutions_table = self._create_solutions_table()
        layout["solutions"].update(Panel(solutions_table, title="🏆 Top Solutions"))

        # Footer
        elapsed = datetime.now() - datetime.fromtimestamp(self.overall_start_time)
        footer_text = Text(
            f"Elapsed: {elapsed} | Press Ctrl+C to stop", style="dim", justify="center"
        )
        layout["footer"].update(footer_text)

        return layout

    def _create_solutions_table(self) -> Table:
        """Create a table showing the best solutions."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Rank", style="cyan", width=4)
        table.add_column("Generation", style="green", width=10)
        table.add_column("Δv (m/s)", style="yellow", width=12)
        table.add_column("Cost ($M)", style="red", width=12)
        table.add_column("Time (days)", style="blue", width=12)

        if not self.best_solutions:
            table.add_row("—", "—", "—", "—", "—")
            return table

        for i, solution in enumerate(self.best_solutions, 1):
            table.add_row(
                str(i),
                str(solution.generation),
                f"{solution.delta_v:,.0f}" if solution.delta_v > 0 else "—",
                f"{solution.cost/1e6:.1f}" if solution.cost > 0 else "—",
                f"{solution.transfer_time:.2f}" if solution.transfer_time > 0 else "—",
            )

        return table

    @contextmanager
    def live_progress(self):
        """Context manager for live progress display."""
        try:
            layout = self.create_live_display()
            with Live(layout, refresh_per_second=2, screen=True):
                yield self
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Analysis interrupted by user[/yellow]")
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the progress tracker."""
        self._stop_event.set()

        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis progress."""
        total_elapsed = time.time() - self.overall_start_time
        completed_phases = [p for p in self.phases if p.status == "completed"]

        return {
            "scenario_name": self.scenario_name,
            "total_elapsed": total_elapsed,
            "phases_completed": len(completed_phases),
            "total_phases": len(self.phases),
            "current_phase": (
                self.phases[self.current_phase_index].name
                if self.current_phase_index < len(self.phases)
                else "Completed"
            ),
            "solutions_found": len(self.optimization_solutions),
            "best_solution": self.best_solutions[0] if self.best_solutions else None,
        }

    def print_summary(self) -> None:
        """Print a final summary of the analysis."""
        summary = self.get_summary()

        self.console.print("\n" + "=" * 60)
        self.console.print(
            f"🎯 Analysis Summary: {summary['scenario_name']}", style="bold green"
        )
        self.console.print("=" * 60)

        # Timing information
        elapsed_str = str(timedelta(seconds=int(summary["total_elapsed"])))
        self.console.print(f"⏱️  Total Time: {elapsed_str}")
        self.console.print(
            f"📊 Phases Completed: {summary['phases_completed']}/{summary['total_phases']}"
        )

        # Best solution
        if summary["best_solution"]:
            best = summary["best_solution"]
            self.console.print("\n🏆 Best Solution Found:")
            self.console.print(f"   • Delta-v: {best.delta_v:,.0f} m/s")
            if best.cost > 0:
                self.console.print(f"   • Cost: ${best.cost/1e6:.1f}M")
            if best.transfer_time > 0:
                self.console.print(f"   • Transfer Time: {best.transfer_time:.2f} days")

        # Phase breakdown
        self.console.print("\n📋 Phase Details:")
        for phase in self.phases:
            status_icon = {
                "completed": "✅",
                "running": "🔄",
                "pending": "⏳",
                "failed": "❌",
            }
            icon = status_icon.get(phase.status, "❓")
            duration_str = f"{phase.duration:.1f}s" if phase.duration else "—"
            self.console.print(f"   {icon} {phase.name}: {duration_str}")


class OptimizationCallback:
    """Callback for optimization progress updates."""

    def __init__(self, progress_tracker: EnhancedProgressTracker):
        self.tracker = progress_tracker
        self.generation = 0
        self.last_update = time.time()

    def __call__(self, population, fitness_values, generation=None):
        """Callback function for optimization updates."""
        if generation is not None:
            self.generation = generation

        # Throttle updates to avoid overwhelming the display
        now = time.time()
        if now - self.last_update < 0.5:  # Update at most every 0.5 seconds
            return
        self.last_update = now

        # Process solutions
        if hasattr(population, "__len__") and hasattr(fitness_values, "__len__"):
            for i, (individual, fitness) in enumerate(zip(population, fitness_values, strict=False)):
                solution = OptimizationSolution(
                    generation=self.generation,
                    individual_id=i,
                    objectives=(
                        list(fitness) if hasattr(fitness, "__iter__") else [fitness]
                    ),
                    parameters=(
                        list(individual)
                        if hasattr(individual, "__iter__")
                        else [individual]
                    ),
                )
                self.tracker.add_optimization_solution(solution)

        # Update phase progress based on generation
        # This is a rough estimate - actual implementation should get this from the optimizer
        if hasattr(self.tracker, "optimization_config"):
            total_gens = getattr(
                self.tracker.optimization_config, "num_generations", 50
            )
            progress = min(1.0, self.generation / total_gens)
            self.tracker.update_phase_progress(
                progress, f"Generation {self.generation}/{total_gens}"
            )
