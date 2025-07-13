#!/usr/bin/env python3
"""Scenario Discovery and Management Module.

This module handles discovery, validation, and metadata extraction
for lunar mission analysis scenarios.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table


@dataclass
class ScenarioMetadata:
    """Metadata for a lunar mission scenario."""

    id: str
    name: str
    description: str
    file_path: Path
    complexity: str = "Intermediate"
    runtime_estimate: str = "~60 seconds"
    mission_type: str = "General"
    modules_used: List[str] = None
    expected_results: Dict[str, Any] = None

    def __post_init__(self):
        if self.modules_used is None:
            self.modules_used = []
        if self.expected_results is None:
            self.expected_results = {}


class ScenarioConfig(BaseModel):
    """Pydantic model for scenario configuration validation."""

    mission: Dict[str, Any] = Field(..., description="Mission configuration")
    spacecraft: Dict[str, Any] = Field(..., description="Spacecraft parameters")
    costs: Dict[str, Any] = Field(..., description="Cost model parameters")
    orbit: Dict[str, Any] = Field(..., description="Orbital parameters")
    optimization: Dict[str, Any] = Field(..., description="Optimization settings")


class ScenarioManager:
    """Manages lunar mission scenarios and their metadata."""

    def __init__(self, scenarios_dir: Optional[Path] = None):
        """Initialize the scenario manager.

        Args:
            scenarios_dir: Path to scenarios directory. If None, uses default.
        """
        self.console = Console()

        if scenarios_dir is None:
            # Default to scenarios/ directory relative to project root
            current_file = Path(__file__)
            project_root = current_file.parents[
                2
            ]  # src/cli/scenario_manager.py -> project root
            self.scenarios_dir = project_root / "scenarios"
        else:
            self.scenarios_dir = Path(scenarios_dir)

        self.scenarios: Dict[str, ScenarioMetadata] = {}
        self._load_scenarios()

    def _load_scenarios(self) -> None:
        """Discover and load all available scenarios."""
        if not self.scenarios_dir.exists():
            self.console.print(
                f"[red]Scenarios directory not found: {self.scenarios_dir}[/red]"
            )
            return

        json_files = list(self.scenarios_dir.glob("*.json"))

        for json_file in json_files:
            try:
                scenario = self._parse_scenario_file(json_file)
                if scenario:
                    self.scenarios[scenario.id] = scenario
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load {json_file.name}: {e}[/yellow]"
                )

    def _parse_scenario_file(self, file_path: Path) -> Optional[ScenarioMetadata]:
        """Parse a scenario JSON file and extract metadata."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Validate the configuration structure
            ScenarioConfig(**data)

            # Extract scenario ID from filename (e.g., "01_basic_transfer" from "01_basic_transfer.json")
            scenario_id = file_path.stem

            # Extract metadata from the configuration
            mission_info = data.get("mission", {})
            name = mission_info.get("name", scenario_id.replace("_", " ").title())
            description = mission_info.get("description", "No description available")

            # Determine complexity and modules based on configuration
            complexity = self._determine_complexity(data)
            modules_used = self._identify_modules(data)
            runtime_estimate = self._estimate_runtime(data)
            mission_type = self._classify_mission_type(data)
            expected_results = self._extract_expected_results(data)

            return ScenarioMetadata(
                id=scenario_id,
                name=name,
                description=description,
                file_path=file_path,
                complexity=complexity,
                runtime_estimate=runtime_estimate,
                mission_type=mission_type,
                modules_used=modules_used,
                expected_results=expected_results,
            )

        except Exception as e:
            self.console.print(f"[red]Error parsing {file_path}: {e}[/red]")
            return None

    def _determine_complexity(self, config: Dict[str, Any]) -> str:
        """Determine scenario complexity based on configuration."""
        opt_config = config.get("optimization", {})
        population = opt_config.get("population_size", 20)
        generations = opt_config.get("num_generations", 10)

        # Simple heuristic based on computational requirements
        complexity_score = population * generations

        if complexity_score < 500:
            return "Beginner"
        elif complexity_score < 1500:
            return "Intermediate"
        else:
            return "Advanced"

    def _identify_modules(self, config: Dict[str, Any]) -> List[str]:
        """Identify which modules will be used based on configuration."""
        modules = ["trajectory.lunar_transfer", "optimization.global_optimizer"]

        # Check for economic analysis requirements
        if config.get("costs"):
            modules.append("economics.financial_models")

        # Check for advanced features
        spacecraft = config.get("spacecraft", {})
        if spacecraft.get("max_propellant_mass", 0) > 10000:
            modules.append("economics.advanced_isru_models")

        # Check for orbital complexity
        orbit = config.get("orbit", {})
        if orbit.get("inclination", 0) > 60:  # Polar or high-inclination orbits
            modules.append("trajectory.transfer_window_analysis")

        return modules

    def _estimate_runtime(self, config: Dict[str, Any]) -> str:
        """Estimate runtime based on configuration complexity."""
        opt_config = config.get("optimization", {})
        population = opt_config.get("population_size", 20)
        generations = opt_config.get("num_generations", 10)

        # Runtime estimation based on computational load
        total_evaluations = population * generations

        if total_evaluations < 300:
            return "~15-30 seconds"
        elif total_evaluations < 800:
            return "~30-60 seconds"
        elif total_evaluations < 1500:
            return "~1-2 minutes"
        else:
            return "~2-5 minutes"

    def _classify_mission_type(self, config: Dict[str, Any]) -> str:
        """Classify the mission type based on configuration."""
        mission_info = config.get("mission", {})
        name = mission_info.get("name", "").lower()

        if "cargo" in name or "delivery" in name:
            return "Cargo Delivery"
        elif "mining" in name or "isru" in name:
            return "Resource Extraction"
        elif "constellation" in name:
            return "Multi-Spacecraft"
        elif "risk" in name:
            return "Risk Analysis"
        elif "pareto" in name or "optimization" in name:
            return "Trade Study"
        else:
            return "General Mission"

    def _extract_expected_results(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract expected results based on scenario type and configuration."""
        results = {}

        # Basic trajectory metrics
        results["delta_v_range"] = "20,000-30,000 m/s"
        results["transfer_time"] = (
            f"{config.get('mission', {}).get('transfer_time', 4.5)} days"
        )

        # Cost estimates based on spacecraft mass and complexity
        spacecraft = config.get("spacecraft", {})
        total_mass = (
            spacecraft.get("dry_mass", 0)
            + spacecraft.get("max_propellant_mass", 0)
            + spacecraft.get("payload_mass", 0)
        )

        if total_mass > 25000:  # Heavy mission
            results["mission_cost"] = "$5-10 billion"
            results["roi_range"] = "20-40%"
        elif total_mass > 15000:  # Medium mission
            results["mission_cost"] = "$3-6 billion"
            results["roi_range"] = "15-30%"
        else:  # Light mission
            results["mission_cost"] = "$1-3 billion"
            results["roi_range"] = "10-25%"

        return results

    def list_scenarios(self, detailed: bool = False) -> None:
        """Display available scenarios in a formatted table."""
        if not self.scenarios:
            self.console.print("[red]No scenarios found![/red]")
            return

        table = Table(title="🚀 Available Lunar Mission Scenarios")

        if detailed:
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="bright_white")
            table.add_column("Type", style="green")
            table.add_column("Complexity", style="yellow")
            table.add_column("Runtime", style="blue")
            table.add_column("Description", style="dim", max_width=40)

            for scenario in self.scenarios.values():
                table.add_row(
                    scenario.id,
                    scenario.name,
                    scenario.mission_type,
                    scenario.complexity,
                    scenario.runtime_estimate,
                    scenario.description,
                )
        else:
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="bright_white")
            table.add_column("Description", style="dim", max_width=60)

            for scenario in self.scenarios.values():
                table.add_row(scenario.id, scenario.name, scenario.description)

        self.console.print(table)

    def get_scenario(self, scenario_id: str) -> Optional[ScenarioMetadata]:
        """Get a specific scenario by ID."""
        return self.scenarios.get(scenario_id)

    def validate_scenario(self, scenario_id: str) -> bool:
        """Validate that a scenario exists and is properly configured."""
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            return False

        try:
            with open(scenario.file_path, "r") as f:
                data = json.load(f)
            ScenarioConfig(**data)
            return True
        except Exception as e:
            self.console.print(f"[red]Scenario validation failed: {e}[/red]")
            return False

    def get_scenario_config(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Load and return the configuration for a specific scenario."""
        scenario = self.get_scenario(scenario_id)
        if not scenario:
            return None

        try:
            with open(scenario.file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.console.print(f"[red]Failed to load scenario config: {e}[/red]")
            return None

    def search_scenarios(self, query: str) -> List[ScenarioMetadata]:
        """Search scenarios by name or description."""
        query = query.lower()
        results = []

        for scenario in self.scenarios.values():
            if (
                query in scenario.name.lower()
                or query in scenario.description.lower()
                or query in scenario.mission_type.lower()
            ):
                results.append(scenario)

        return results

    def get_scenarios_by_type(self, mission_type: str) -> List[ScenarioMetadata]:
        """Get all scenarios of a specific mission type."""
        return [s for s in self.scenarios.values() if s.mission_type == mission_type]

    def get_scenarios_by_complexity(self, complexity: str) -> List[ScenarioMetadata]:
        """Get all scenarios of a specific complexity level."""
        return [s for s in self.scenarios.values() if s.complexity == complexity]
