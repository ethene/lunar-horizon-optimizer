#!/usr/bin/env python3
"""Output Management and Export Module.

This module handles result organization, export functionality,
and report generation for lunar mission analysis.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from rich.console import Console
from rich.panel import Panel


class OutputManager:
    """Manages analysis outputs, exports, and report generation."""

    def __init__(self, base_output_dir: Optional[Path] = None):
        """Initialize the output manager.

        Args:
            base_output_dir: Base directory for outputs. If None, uses default.
        """
        self.console = Console()

        if base_output_dir is None:
            # Default to results/ directory relative to project root
            current_file = Path(__file__)
            project_root = current_file.parents[
                2
            ]  # src/cli/output_manager.py -> project root
            self.base_output_dir = project_root / "results"
        else:
            self.base_output_dir = Path(base_output_dir)

        self.base_output_dir.mkdir(exist_ok=True)

        # Set up Plotly for static image export
        try:
            import kaleido  # noqa

            self.pdf_export_available = True
        except ImportError:
            self.pdf_export_available = False
            self.console.print(
                "[yellow]Warning: Kaleido not available. PDF export disabled.[/yellow]"
            )

    def create_output_directory(
        self, scenario_name: str, timestamp: Optional[str] = None
    ) -> Path:
        """Create a timestamped output directory for a scenario.

        Args:
            scenario_name: Name of the scenario
            timestamp: Optional timestamp string. If None, uses current time.

        Returns:
            Path to the created output directory
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clean scenario name for use as directory name
        clean_name = "".join(
            c for c in scenario_name if c.isalnum() or c in "._- "
        ).strip()
        clean_name = clean_name.replace(" ", "_")

        output_dir = self.base_output_dir / f"{timestamp}_{clean_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "figures").mkdir(exist_ok=True)
        (output_dir / "data").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)

        return output_dir

    def save_analysis_results(self, output_dir: Path, results: Any) -> None:
        """Save analysis results to the output directory.

        Args:
            output_dir: Output directory path
            results: Analysis results object
        """
        # Save main results as JSON
        results_file = output_dir / "data" / "analysis_results.json"

        try:
            # Convert results to serializable format
            if hasattr(results, "to_dict"):
                results_dict = results.to_dict()
            elif hasattr(results, "__dict__"):
                results_dict = self._make_serializable(results.__dict__)
            else:
                results_dict = {"raw_results": str(results)}

            with open(results_file, "w") as f:
                json.dump(results_dict, f, indent=2, default=str)

            self.console.print(
                f"[green]✅ Analysis results saved to {results_file}[/green]"
            )

        except Exception as e:
            self.console.print(f"[red]❌ Failed to save analysis results: {e}[/red]")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "tolist"):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def generate_html_dashboard(
        self, output_dir: Path, results: Any, scenario_metadata: Dict[str, Any]
    ) -> Optional[Path]:
        """Generate an HTML dashboard with interactive plots.

        Args:
            output_dir: Output directory path
            results: Analysis results
            scenario_metadata: Scenario metadata

        Returns:
            Path to the generated HTML file, or None if failed
        """
        try:
            from src.visualization.integrated_dashboard import IntegratedDashboard

            dashboard = IntegratedDashboard()

            # Generate dashboard HTML
            html_content = dashboard.create_dashboard(results, scenario_metadata)

            html_file = output_dir / "dashboard.html"
            with open(html_file, "w") as f:
                f.write(html_content)

            self.console.print(
                f"[green]✅ Interactive dashboard saved to {html_file}[/green]"
            )
            return html_file

        except Exception as e:
            self.console.print(f"[red]❌ Failed to generate HTML dashboard: {e}[/red]")
            return None

    def export_figures_to_pdf(
        self, output_dir: Path, results: Any
    ) -> Optional[List[Path]]:
        """Export key figures to PDF format.

        Args:
            output_dir: Output directory path
            results: Analysis results

        Returns:
            List of paths to generated PDF files, or None if failed
        """
        if not self.pdf_export_available:
            self.console.print(
                "[yellow]PDF export not available (Kaleido not installed)[/yellow]"
            )
            return None

        try:
            pdf_files = []
            figures_dir = output_dir / "figures"

            # Extract and export key figures
            figures_to_export = self._extract_key_figures(results)

            for fig_name, figure in figures_to_export.items():
                if figure is not None:
                    pdf_file = figures_dir / f"{fig_name}.pdf"
                    figure.write_image(str(pdf_file), format="pdf")
                    pdf_files.append(pdf_file)

            if pdf_files:
                self.console.print(
                    f"[green]✅ {len(pdf_files)} figures exported to PDF[/green]"
                )
            else:
                self.console.print(
                    "[yellow]No figures available for PDF export[/yellow]"
                )

            return pdf_files

        except Exception as e:
            self.console.print(f"[red]❌ Failed to export figures to PDF: {e}[/red]")
            return None

    def _extract_key_figures(self, results: Any) -> Dict[str, Optional[go.Figure]]:
        """Extract key figures from analysis results.

        Args:
            results: Analysis results

        Returns:
            Dictionary mapping figure names to Plotly figures
        """
        figures = {}

        try:
            # Try to extract common figure types
            if hasattr(results, "trajectory_plot"):
                figures["trajectory"] = results.trajectory_plot

            if hasattr(results, "pareto_front_plot"):
                figures["pareto_front"] = results.pareto_front_plot

            if hasattr(results, "cost_breakdown_plot"):
                figures["cost_breakdown"] = results.cost_breakdown_plot

            if hasattr(results, "sensitivity_plot"):
                figures["sensitivity_analysis"] = results.sensitivity_plot

            # If results has a figures attribute
            if hasattr(results, "figures") and isinstance(results.figures, dict):
                figures.update(results.figures)

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Failed to extract some figures: {e}[/yellow]"
            )

        return figures

    def generate_summary_report(
        self,
        output_dir: Path,
        results: Any,
        scenario_metadata: Dict[str, Any],
        analysis_summary: Dict[str, Any],
    ) -> Optional[Path]:
        """Generate a text summary report.

        Args:
            output_dir: Output directory path
            results: Analysis results
            scenario_metadata: Scenario metadata
            analysis_summary: Analysis execution summary

        Returns:
            Path to the generated report file, or None if failed
        """
        try:
            report_file = output_dir / "reports" / "summary_report.txt"

            with open(report_file, "w") as f:
                f.write(
                    self._format_summary_report(
                        results, scenario_metadata, analysis_summary
                    )
                )

            self.console.print(
                f"[green]✅ Summary report saved to {report_file}[/green]"
            )
            return report_file

        except Exception as e:
            self.console.print(f"[red]❌ Failed to generate summary report: {e}[/red]")
            return None

    def _format_summary_report(
        self,
        results: Any,
        scenario_metadata: Dict[str, Any],
        analysis_summary: Dict[str, Any],
    ) -> str:
        """Format the summary report content."""
        report_lines = []

        # Header
        report_lines.extend(
            [
                "=" * 80,
                "🌙 LUNAR HORIZON OPTIMIZER - ANALYSIS SUMMARY REPORT",
                "=" * 80,
                "",
                f"Scenario: {scenario_metadata.get('name', 'Unknown')}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Analysis Duration: {analysis_summary.get('total_elapsed', 0):.1f} seconds",
                "",
            ]
        )

        # Scenario Information
        report_lines.extend(
            [
                "📋 SCENARIO DETAILS",
                "-" * 40,
                f"ID: {scenario_metadata.get('id', 'N/A')}",
                f"Description: {scenario_metadata.get('description', 'N/A')}",
                f"Mission Type: {scenario_metadata.get('mission_type', 'N/A')}",
                f"Complexity: {scenario_metadata.get('complexity', 'N/A')}",
                "",
            ]
        )

        # Analysis Results
        if hasattr(results, "best_solution") and results.best_solution:
            solution = results.best_solution
            report_lines.extend(
                [
                    "🏆 BEST SOLUTION FOUND",
                    "-" * 40,
                    f"Delta-v: {getattr(solution, 'delta_v', 'N/A')} m/s",
                    f"Transfer Time: {getattr(solution, 'transfer_time', 'N/A')} days",
                    f"Total Cost: ${getattr(solution, 'total_cost', 0)/1e6:.1f}M",
                    "",
                ]
            )

        # Economic Analysis
        if hasattr(results, "economic_results"):
            econ = results.economic_results
            report_lines.extend(
                [
                    "💰 ECONOMIC ANALYSIS",
                    "-" * 40,
                    f"NPV: ${getattr(econ, 'npv', 0)/1e6:.1f}M",
                    f"ROI: {getattr(econ, 'roi', 0)*100:.1f}%",
                    f"IRR: {getattr(econ, 'irr', 0)*100:.1f}%",
                    f"Payback Period: {getattr(econ, 'payback_period', 0):.1f} years",
                    "",
                ]
            )

        # Performance Summary
        report_lines.extend(
            [
                "📊 ANALYSIS PERFORMANCE",
                "-" * 40,
                f"Phases Completed: {analysis_summary.get('phases_completed', 0)}/{analysis_summary.get('total_phases', 0)}",
                f"Solutions Evaluated: {analysis_summary.get('solutions_found', 0)}",
                "",
            ]
        )

        # Modules Used
        modules_used = scenario_metadata.get("modules_used", [])
        if modules_used:
            report_lines.extend(
                [
                    "🔧 MODULES UTILIZED",
                    "-" * 40,
                ]
            )
            for module in modules_used:
                report_lines.append(f"• {module}")
            report_lines.append("")

        # Footer
        report_lines.extend(
            [
                "=" * 80,
                "Report generated by Lunar Horizon Optimizer",
                "For more details, see the interactive dashboard and data files",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)

    def save_configuration(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        scenario_metadata: Dict[str, Any],
    ) -> None:
        """Save the scenario configuration and metadata.

        Args:
            output_dir: Output directory path
            config: Scenario configuration
            scenario_metadata: Scenario metadata
        """
        try:
            # Save original configuration
            config_file = output_dir / "data" / "scenario_config.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            # Save metadata
            metadata_file = output_dir / "data" / "scenario_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(scenario_metadata, f, indent=2, default=str)

            self.console.print("[green]✅ Configuration saved[/green]")

        except Exception as e:
            self.console.print(f"[red]❌ Failed to save configuration: {e}[/red]")

    def create_output_summary(self, output_dir: Path) -> None:
        """Create a summary of all generated outputs.

        Args:
            output_dir: Output directory path
        """
        summary_lines = [
            "📁 OUTPUT SUMMARY",
            "=" * 50,
            "",
            "Generated files and directories:",
            "",
        ]

        # List all generated files
        for root, _dirs, files in os.walk(output_dir):
            root_path = Path(root)
            relative_root = root_path.relative_to(output_dir)

            if relative_root != Path("."):
                summary_lines.append(f"📂 {relative_root}/")

            for file in files:
                file_path = root_path / file
                file_size = file_path.stat().st_size
                size_str = self._format_file_size(file_size)
                relative_file = file_path.relative_to(output_dir)
                summary_lines.append(f"   📄 {relative_file} ({size_str})")

        summary_lines.extend(
            [
                "",
                "🎯 Key files:",
                "• dashboard.html - Interactive analysis dashboard",
                "• reports/summary_report.txt - Text summary report",
                "• data/analysis_results.json - Complete analysis data",
                "• figures/*.pdf - Exported figures (if enabled)",
                "",
            ]
        )

        # Save summary
        summary_file = output_dir / "OUTPUT_SUMMARY.txt"
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))

        # Also print to console
        self.console.print(
            Panel(
                "\n".join(summary_lines),
                title="📁 Output Summary",
                border_style="green",
            )
        )

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    def open_dashboard(self, html_file: Path) -> None:
        """Attempt to open the dashboard in the default browser.

        Args:
            html_file: Path to the HTML dashboard file
        """
        try:
            import webbrowser

            webbrowser.open(f"file://{html_file.absolute()}")
            self.console.print(
                f"[green]🌐 Dashboard opened in browser: {html_file}[/green]"
            )
        except Exception as e:
            self.console.print(
                f"[yellow]Could not open browser automatically: {e}[/yellow]"
            )
            self.console.print(f"[blue]Please open manually: {html_file}[/blue]")

    def cleanup_old_results(self, keep_recent: int = 10) -> None:
        """Clean up old result directories, keeping only the most recent ones.

        Args:
            keep_recent: Number of recent result directories to keep
        """
        try:
            result_dirs = [d for d in self.base_output_dir.iterdir() if d.is_dir()]
            result_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            dirs_to_remove = result_dirs[keep_recent:]

            for dir_to_remove in dirs_to_remove:
                shutil.rmtree(dir_to_remove)
                self.console.print(
                    f"[dim]Cleaned up old results: {dir_to_remove.name}[/dim]"
                )

            if dirs_to_remove:
                self.console.print(
                    f"[green]✅ Cleaned up {len(dirs_to_remove)} old result directories[/green]"
                )

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Failed to cleanup old results: {e}[/yellow]"
            )
