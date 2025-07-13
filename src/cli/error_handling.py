#!/usr/bin/env python3
"""Error Handling and Validation Module.

This module provides comprehensive error handling, validation,
and user-friendly error messages for the CLI.
"""

import sys
import traceback
from enum import Enum
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel


class ErrorLevel(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LunarOptError(Exception):
    """Base exception for Lunar Horizon Optimizer CLI errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "LO_ERROR",
        suggestions: Optional[List[str]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.suggestions = suggestions or []
        super().__init__(self.message)


class ScenarioError(LunarOptError):
    """Errors related to scenario management."""

    def __init__(
        self,
        message: str,
        scenario_id: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.scenario_id = scenario_id
        super().__init__(message, "LO_SCENARIO", suggestions)


class ConfigurationError(LunarOptError):
    """Errors related to configuration validation."""

    def __init__(
        self,
        message: str,
        config_field: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.config_field = config_field
        super().__init__(message, "LO_CONFIG", suggestions)


class AnalysisError(LunarOptError):
    """Errors during analysis execution."""

    def __init__(
        self,
        message: str,
        phase: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.phase = phase
        super().__init__(message, "LO_ANALYSIS", suggestions)


class DependencyError(LunarOptError):
    """Errors related to missing or incompatible dependencies."""

    def __init__(
        self,
        message: str,
        package: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.package = package
        super().__init__(message, "LO_DEPENDENCY", suggestions)


class ErrorHandler:
    """Centralized error handling and user messaging."""

    def __init__(self, console: Optional[Console] = None, verbose: bool = False):
        self.console = console or Console()
        self.verbose = verbose

    def handle_error(self, error: Exception, context: Optional[str] = None) -> int:
        """Handle an error and return appropriate exit code.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        if isinstance(error, KeyboardInterrupt):
            return self._handle_keyboard_interrupt()
        elif isinstance(error, LunarOptError):
            return self._handle_lunar_opt_error(error, context)
        elif isinstance(error, ImportError):
            return self._handle_import_error(error)
        elif isinstance(error, FileNotFoundError):
            return self._handle_file_not_found_error(error, context)
        elif isinstance(error, PermissionError):
            return self._handle_permission_error(error, context)
        else:
            return self._handle_unexpected_error(error, context)

    def _handle_keyboard_interrupt(self) -> int:
        """Handle user interruption (Ctrl+C)."""
        self.console.print("\n[yellow]⚠️  Operation interrupted by user[/yellow]")
        return 130  # Standard exit code for SIGINT

    def _handle_lunar_opt_error(
        self, error: LunarOptError, context: Optional[str]
    ) -> int:
        """Handle application-specific errors."""
        error_color = "red"

        # Create error message
        title = f"❌ {error.error_code}"
        if context:
            title += f" ({context})"

        message_lines = [f"[bold]{error.message}[/bold]"]

        # Add suggestions if available
        if error.suggestions:
            message_lines.extend(["", "[bold blue]💡 Suggestions:[/bold blue]"])
            for suggestion in error.suggestions:
                message_lines.append(f"  • {suggestion}")

        # Add specific error details
        if isinstance(error, ScenarioError) and error.scenario_id:
            message_lines.extend(["", f"[dim]Scenario ID: {error.scenario_id}[/dim]"])
        elif isinstance(error, ConfigurationError) and error.config_field:
            message_lines.extend(
                ["", f"[dim]Configuration field: {error.config_field}[/dim]"]
            )
        elif isinstance(error, AnalysisError) and error.phase:
            message_lines.extend(["", f"[dim]Analysis phase: {error.phase}[/dim]"])
        elif isinstance(error, DependencyError) and error.package:
            message_lines.extend(["", f"[dim]Package: {error.package}[/dim]"])

        self.console.print(
            Panel("\n".join(message_lines), title=title, border_style=error_color)
        )

        return 1

    def _handle_import_error(self, error: ImportError) -> int:
        """Handle import/dependency errors."""
        package_name = str(error).split("'")[1] if "'" in str(error) else "unknown"

        suggestions = []

        # Provide specific suggestions for known packages
        dependency_suggestions = {
            "click": ["pip install click>=8.1.0"],
            "rich": ["pip install rich>=13.7.0"],
            "pydantic": ["pip install pydantic>=2.0.0"],
            "plotly": ["pip install plotly>=5.24.1"],
            "kaleido": ["pip install kaleido (for PDF export)"],
            "pygmo": ["conda install -c conda-forge pygmo"],
            "pykep": ["conda install -c conda-forge pykep"],
            "jax": ["pip install jax jaxlib"],
        }

        if package_name in dependency_suggestions:
            suggestions.extend(dependency_suggestions[package_name])
        else:
            suggestions.extend(
                [
                    "Check that all required dependencies are installed",
                    "Run: pip install -r requirements.txt",
                    "For conda packages: conda install -c conda-forge <package>",
                ]
            )

        dep_error = DependencyError(
            f"Required package '{package_name}' is not available",
            package=package_name,
            suggestions=suggestions,
        )

        return self._handle_lunar_opt_error(dep_error, "Dependency Check")

    def _handle_file_not_found_error(
        self, error: FileNotFoundError, context: Optional[str]
    ) -> int:
        """Handle file not found errors."""
        filename = str(error).split("'")[1] if "'" in str(error) else "unknown file"

        suggestions = [
            "Check that the file path is correct",
            "Verify file permissions",
            "Ensure you're running from the correct directory",
        ]

        # Scenario-specific suggestions
        if "scenario" in filename.lower() or filename.endswith(".json"):
            suggestions.extend(
                [
                    "List available scenarios with: lunar-opt run list",
                    "Check the scenarios/ directory for available files",
                ]
            )

        file_error = LunarOptError(
            f"File not found: {filename}",
            error_code="LO_FILE_NOT_FOUND",
            suggestions=suggestions,
        )

        return self._handle_lunar_opt_error(file_error, context)

    def _handle_permission_error(
        self, error: PermissionError, context: Optional[str]
    ) -> int:
        """Handle permission errors."""
        suggestions = [
            "Check file/directory permissions",
            "Ensure you have write access to the output directory",
            "Try running with appropriate permissions",
        ]

        perm_error = LunarOptError(
            f"Permission denied: {error}",
            error_code="LO_PERMISSION",
            suggestions=suggestions,
        )

        return self._handle_lunar_opt_error(perm_error, context)

    def _handle_unexpected_error(self, error: Exception, context: Optional[str]) -> int:
        """Handle unexpected errors."""
        error_type = type(error).__name__

        if self.verbose:
            # Show full traceback in verbose mode
            self.console.print("\n[red]💥 Unexpected Error (with traceback):[/red]")
            self.console.print(f"[red]{traceback.format_exc()}[/red]")
        else:
            # Show user-friendly error message
            suggestions = [
                "Try running with --verbose for more details",
                "Check the documentation for troubleshooting",
                "Report this issue if it persists",
            ]

            unexpected_error = LunarOptError(
                f"Unexpected {error_type}: {error}",
                error_code="LO_UNEXPECTED",
                suggestions=suggestions,
            )

            self._handle_lunar_opt_error(unexpected_error, context)

        return 1

    def validate_environment(self) -> List[str]:
        """Validate the environment and return a list of issues.

        Returns:
            List of validation error messages (empty if all OK)
        """
        issues = []

        # Check required Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ is required")

        # Check critical dependencies
        required_packages = ["numpy", "scipy", "plotly", "pydantic", "click", "rich"]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Required package '{package}' is not installed")

        # Check optional but important packages
        optional_packages = {
            "pygmo": "Global optimization capabilities will be limited",
            "pykep": "Trajectory generation will not work",
            "jax": "Differentiable optimization will not be available",
            "kaleido": "PDF export will not be available",
        }

        for package, consequence in optional_packages.items():
            try:
                __import__(package)
            except ImportError:
                issues.append(f"Optional package '{package}' missing: {consequence}")

        return issues

    def print_validation_results(self, issues: List[str]) -> bool:
        """Print environment validation results.

        Args:
            issues: List of validation issues

        Returns:
            True if validation passed, False otherwise
        """
        if not issues:
            self.console.print("[green]✅ Environment validation passed![/green]")
            return True

        # Separate critical from non-critical issues
        critical_issues = [i for i in issues if "Required" in i or "Python" in i]
        warning_issues = [i for i in issues if i not in critical_issues]

        if critical_issues:
            self.console.print("\n[red]❌ Critical Issues Found:[/red]")
            for issue in critical_issues:
                self.console.print(f"  • {issue}")

        if warning_issues:
            self.console.print("\n[yellow]⚠️  Warnings:[/yellow]")
            for issue in warning_issues:
                self.console.print(f"  • {issue}")

        return len(critical_issues) == 0


def safe_execute(
    func, error_handler: Optional[ErrorHandler] = None, context: Optional[str] = None
):
    """Safely execute a function with error handling.

    Args:
        func: Function to execute
        error_handler: Error handler instance
        context: Context description for error messages

    Returns:
        Result of function execution, or exits with error code
    """
    if error_handler is None:
        error_handler = ErrorHandler()

    try:
        return func()
    except Exception as e:
        exit_code = error_handler.handle_error(e, context)
        sys.exit(exit_code)
