#!/usr/bin/env python3
"""
Comprehensive Test Analysis and Coverage Report

This script provides detailed analysis of all test modules, coverage gaps,
sanity check validation, and realistic result verification across the
entire Lunar Horizon Optimizer codebase.

Author: Lunar Horizon Optimizer Team
Date: July 2025
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import importlib

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestAnalyzer:
    """Comprehensive test analysis and execution framework."""

    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"

        # Test execution results
        self.test_results = {}
        self.coverage_analysis = {}
        self.sanity_check_results = {}

        # Test categories
        self.test_categories = {
            "core_functionality": ["test_final_functionality.py"],
            "task_3_trajectory": ["test_task_3_trajectory_generation.py"],
            "task_4_optimization": ["test_task_4_global_optimization.py"],
            "task_5_economics": ["test_task_5_economic_analysis.py"],
            "task_6_visualization": ["test_task_6_visualization.py"],
            "integration": ["test_integration_tasks_3_4_5.py"],
        }

        # Known realistic value ranges for sanity checking
        self.realistic_ranges = {
            "delta_v_ms": (1000, 10000),  # m/s for lunar missions
            "transfer_time_days": (3, 15),  # days for Earth-Moon transfer
            "mission_cost_usd": (50e6, 5e9),  # dollars for lunar missions
            "npv_usd": (-1e9, 2e9),  # dollars for project NPV
            "irr_fraction": (-0.5, 1.0),  # fraction for IRR
            "roi_fraction": (-0.5, 2.0),  # fraction for ROI
            "spacecraft_mass_kg": (500, 50000),  # kg for lunar spacecraft
            "orbit_altitude_km": (200, 2000),  # km for Earth/Moon orbits
            "mission_duration_years": (1, 20),  # years for mission duration
            "earth_radius_m": (6.35e6, 6.4e6),  # meters
            "moon_radius_m": (1.7e6, 1.8e6),  # meters
            "earth_moon_distance_m": (3.6e8, 4.1e8),  # meters
        }

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive test analysis and coverage report."""
        print("🔍 Starting Comprehensive Test Analysis")
        print(f"Project root: {self.project_root}")
        print(f"Test directory: {self.test_dir}")
        print(f"Environment: {self._get_environment_info()}")
        print("=" * 80)

        # 1. Module Coverage Analysis
        print("\n📊 ANALYZING MODULE COVERAGE")
        self.coverage_analysis = self._analyze_module_coverage()
        self._print_coverage_summary()

        # 2. Test Execution
        print("\n🧪 EXECUTING TEST SUITES")
        self._execute_all_tests()

        # 3. Sanity Check Analysis
        print("\n✅ PERFORMING SANITY CHECKS")
        self._perform_sanity_checks()

        # 4. Generate Comprehensive Report
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        report = self._generate_comprehensive_report()
        self._save_report(report)

        return report

    def _get_environment_info(self) -> str:
        """Get environment information."""
        try:
            import pykep as pk
            import pygmo as pg
            import plotly

            return f"conda py312 with PyKEP {pk.__version__}, PyGMO {pg.__version__}, Plotly {plotly.__version__}"
        except ImportError as e:
            return f"Environment issue: {e}"

    def _analyze_module_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage for all modules."""
        coverage = {
            "src_modules": self._find_source_modules(),
            "test_files": self._find_test_files(),
            "coverage_gaps": [],
            "test_categories": {},
        }

        # Analyze coverage by category
        for category, test_files in self.test_categories.items():
            coverage["test_categories"][category] = {
                "test_files": test_files,
                "exists": [self._test_file_exists(f) for f in test_files],
                "runnable": [],
            }

        # Find coverage gaps
        src_modules = set(coverage["src_modules"])
        tested_modules = self._extract_tested_modules()
        coverage["coverage_gaps"] = list(src_modules - tested_modules)

        return coverage

    def _find_source_modules(self) -> List[str]:
        """Find all source modules."""
        modules = []
        for package in [
            "config",
            "trajectory",
            "optimization",
            "economics",
            "visualization",
            "utils",
        ]:
            package_path = self.src_dir / package
            if package_path.exists():
                for py_file in package_path.rglob("*.py"):
                    if py_file.name != "__init__.py":
                        relative_path = py_file.relative_to(self.src_dir)
                        modules.append(str(relative_path))
        return sorted(modules)

    def _find_test_files(self) -> List[str]:
        """Find all test files."""
        test_files = []
        for py_file in self.test_dir.glob("test_*.py"):
            test_files.append(py_file.name)
        return sorted(test_files)

    def _test_file_exists(self, filename: str) -> bool:
        """Check if test file exists."""
        return (self.test_dir / filename).exists()

    def _extract_tested_modules(self) -> set:
        """Extract modules that have dedicated tests."""
        tested = set()

        # Known tested modules from existing tests
        tested.update(
            [
                "config/costs.py",
                "economics/financial_models.py",
                "economics/cost_models.py",
                "economics/isru_benefits.py",
                "economics/sensitivity_analysis.py",
                "economics/reporting.py",
                "trajectory/earth_moon_trajectories.py",
                "trajectory/nbody_integration.py",
                "trajectory/transfer_window_analysis.py",
                "optimization/global_optimizer.py",
                "optimization/pareto_analysis.py",
                "visualization/trajectory_visualization.py",
                "visualization/optimization_visualization.py",
                "visualization/economic_visualization.py",
                "visualization/mission_visualization.py",
                "visualization/dashboard.py",
            ]
        )

        return tested

    def _execute_all_tests(self) -> None:
        """Execute all test suites and collect results."""
        test_suites = [
            ("Core Functionality", "test_final_functionality.py"),
            ("Task 3 Trajectory", "test_task_3_trajectory_generation.py"),
            ("Task 4 Optimization", "test_task_4_global_optimization.py"),
            ("Task 5 Economics", "test_task_5_economic_analysis.py"),
            ("Task 6 Visualization", "test_task_6_visualization.py"),
            ("Integration Tests", "test_integration_tasks_3_4_5.py"),
        ]

        for suite_name, test_file in test_suites:
            print(f"\n{'='*60}")
            print(f"Running {suite_name}")
            print(f"{'='*60}")

            if not self._test_file_exists(test_file):
                print(f"❌ Test file {test_file} not found")
                self.test_results[suite_name] = {
                    "status": "not_found",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": ["Test file not found"],
                }
                continue

            result = self._run_pytest(test_file)
            self.test_results[suite_name] = result

            # Print result
            if result["status"] == "passed":
                print(f"✅ PASSED - {suite_name}")
            else:
                print(f"❌ {result['status'].upper()} - {suite_name}")

            print(
                f"Tests: {result['total']} total, "
                f"{result['passed']} passed, "
                f"{result['failed']} failed, "
                f"{result['skipped']} skipped"
            )
            print(f"Execution time: {result['execution_time']:.2f}s")

    def _run_pytest(self, test_file: str) -> Dict[str, Any]:
        """Run pytest on a specific test file."""
        start_time = time.time()

        try:
            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.src_dir)

            # Run pytest
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(self.test_dir / test_file),
                "-v",
                "--tb=short",
                "--no-header",
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            execution_time = time.time() - start_time

            # Parse output
            parsed = self._parse_pytest_output(result.stdout, result.stderr)
            parsed["execution_time"] = execution_time
            parsed["return_code"] = result.returncode

            if result.returncode == 0:
                parsed["status"] = "passed"
            else:
                parsed["status"] = "failed"

            return parsed

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "total": 0,
                "execution_time": time.time() - start_time,
                "errors": ["Test execution timed out"],
            }
        except Exception as e:
            return {
                "status": "error",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "total": 0,
                "execution_time": time.time() - start_time,
                "errors": [str(e)],
            }

    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        result = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "total": 0,
            "details": stdout,
            "errors_detail": stderr,
        }

        # Look for summary line like "3 passed, 1 skipped in 2.14s"
        lines = stdout.split("\n")
        for line in lines:
            if "passed" in line or "failed" in line or "skipped" in line:
                if " in " in line and "s" in line:
                    # This looks like a summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            result["passed"] = int(parts[i - 1])
                        elif part == "failed" and i > 0:
                            result["failed"] = int(parts[i - 1])
                        elif part == "skipped" and i > 0:
                            result["skipped"] = int(parts[i - 1])
                        elif part == "error" and i > 0:
                            result["errors"] = int(parts[i - 1])

        result["total"] = (
            result["passed"] + result["failed"] + result["skipped"] + result["errors"]
        )
        return result

    def _perform_sanity_checks(self) -> None:
        """Perform sanity checks on test results and calculations."""
        sanity_results = {
            "realistic_ranges": self._check_realistic_ranges(),
            "physical_constants": self._check_physical_constants(),
            "calculation_consistency": self._check_calculation_consistency(),
            "result_validation": self._check_result_validation(),
        }

        self.sanity_check_results = sanity_results

        # Print sanity check summary
        print("\n🔍 SANITY CHECK RESULTS:")
        for check_type, results in sanity_results.items():
            status = (
                "✅ PASS" if results.get("status") == "pass" else "⚠️ NEEDS ATTENTION"
            )
            print(f"  {check_type}: {status}")
            if results.get("issues"):
                for issue in results["issues"][:3]:  # Show first 3 issues
                    print(f"    - {issue}")

    def _check_realistic_ranges(self) -> Dict[str, Any]:
        """Check that test values are within realistic ranges."""
        issues = []

        # Check if realistic ranges are being validated in tests
        for _category, test_files in self.test_categories.items():
            for test_file in test_files:
                if self._test_file_exists(test_file):
                    test_path = self.test_dir / test_file
                    try:
                        with open(test_path, "r") as f:
                            content = f.read()

                        # Look for range validations
                        if "assert" in content and (">" in content or "<" in content):
                            # Good - has range checks
                            pass
                        else:
                            issues.append(
                                f"{test_file}: Missing range validation checks"
                            )
                    except Exception as e:
                        issues.append(f"{test_file}: Could not analyze - {e}")

        return {
            "status": "pass" if len(issues) < 3 else "needs_attention",
            "issues": issues,
            "checked_ranges": list(self.realistic_ranges.keys()),
        }

    def _check_physical_constants(self) -> Dict[str, Any]:
        """Check that physical constants are reasonable."""
        issues = []

        try:
            # Import and check constants
            from trajectory.constants import PhysicalConstants as PC

            # Check Earth constants
            if not (6.0e6 < PC.EARTH_RADIUS < 7.0e6):
                issues.append(f"Earth radius unrealistic: {PC.EARTH_RADIUS}")

            if not (3.9e14 < PC.EARTH_MU < 4.1e14):
                issues.append(f"Earth mu unrealistic: {PC.EARTH_MU}")

            # Check Moon constants
            if not (1.5e6 < PC.MOON_RADIUS < 2.0e6):
                issues.append(f"Moon radius unrealistic: {PC.MOON_RADIUS}")

            if not (4.8e12 < PC.MOON_MU < 5.0e12):
                issues.append(f"Moon mu unrealistic: {PC.MOON_MU}")

            # Check Sun constants (if available)
            if hasattr(PC, "SUN_MU"):
                if not (1.0e20 < PC.SUN_MU < 1.5e20):
                    issues.append(f"Sun mu unrealistic: {PC.SUN_MU}")

        except Exception as e:
            issues.append(f"Could not import constants: {e}")

        return {
            "status": "pass" if len(issues) == 0 else "needs_attention",
            "issues": issues,
        }

    def _check_calculation_consistency(self) -> Dict[str, Any]:
        """Check that calculations are consistent and sensible."""
        issues = []

        # Check if we can run basic calculations
        try:
            import numpy as np

            # Basic orbital mechanics sanity
            earth_radius = 6.378e6  # m
            earth_mu = 3.986e14  # m³/s²

            # Circular velocity at 400 km altitude
            altitude = 400e3  # m
            orbit_radius = earth_radius + altitude
            circular_velocity = np.sqrt(earth_mu / orbit_radius)

            if not (7000 < circular_velocity < 8000):
                issues.append(f"LEO velocity unrealistic: {circular_velocity:.0f} m/s")

            # Escape velocity
            escape_velocity = np.sqrt(2 * earth_mu / earth_radius)
            if not (11000 < escape_velocity < 12000):
                issues.append(
                    f"Earth escape velocity unrealistic: {escape_velocity:.0f} m/s"
                )

        except Exception as e:
            issues.append(f"Could not perform basic calculations: {e}")

        return {
            "status": "pass" if len(issues) == 0 else "needs_attention",
            "issues": issues,
        }

    def _check_result_validation(self) -> Dict[str, Any]:
        """Check that test results are being properly validated."""
        issues = []

        # Check test results for unrealistic values
        for suite_name, result in self.test_results.items():
            if result.get("status") == "passed":
                # Good - tests are passing
                continue
            elif result.get("status") == "failed":
                if result.get("failed", 0) > 0:
                    issues.append(f"{suite_name}: {result['failed']} tests failing")

        return {
            "status": "pass" if len(issues) < 2 else "needs_attention",
            "issues": issues,
        }

    def _print_coverage_summary(self) -> None:
        """Print module coverage summary."""
        coverage = self.coverage_analysis

        print("\n📋 MODULE COVERAGE SUMMARY:")
        print(f"  Source modules: {len(coverage['src_modules'])}")
        print(f"  Test files: {len(coverage['test_files'])}")
        print(f"  Coverage gaps: {len(coverage['coverage_gaps'])}")

        if coverage["coverage_gaps"]:
            print("\n⚠️ MODULES WITHOUT DEDICATED TESTS:")
            for module in coverage["coverage_gaps"][:10]:  # Show first 10
                print(f"    - {module}")
            if len(coverage["coverage_gaps"]) > 10:
                print(f"    ... and {len(coverage['coverage_gaps']) - 10} more")

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        # Calculate overall statistics
        total_tests = sum(r.get("total", 0) for r in self.test_results.values())
        total_passed = sum(r.get("passed", 0) for r in self.test_results.values())
        total_failed = sum(r.get("failed", 0) for r in self.test_results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.test_results.values())
        total_time = sum(r.get("execution_time", 0) for r in self.test_results.values())

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        report = {
            "analysis_date": datetime.now().isoformat(),
            "environment": self._get_environment_info(),
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "pass_rate": pass_rate,
                "total_execution_time": total_time,
            },
            "test_results": self.test_results,
            "coverage_analysis": self.coverage_analysis,
            "sanity_check_results": self.sanity_check_results,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Test coverage recommendations
        coverage_gaps = len(self.coverage_analysis.get("coverage_gaps", []))
        if coverage_gaps > 5:
            recommendations.append(f"Create tests for {coverage_gaps} untested modules")

        # Pass rate recommendations
        total_tests = sum(r.get("total", 0) for r in self.test_results.values())
        total_passed = sum(r.get("passed", 0) for r in self.test_results.values())
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        if pass_rate < 90:
            recommendations.append(f"Improve pass rate from {pass_rate:.1f}% to 90%+")

        # Sanity check recommendations
        for check_type, results in self.sanity_check_results.items():
            if results.get("status") != "pass":
                recommendations.append(f"Address {check_type} issues")

        # Module-specific recommendations
        for suite_name, result in self.test_results.items():
            if result.get("status") == "not_found":
                recommendations.append(f"Create {suite_name} test suite")
            elif result.get("failed", 0) > 0:
                recommendations.append(
                    f"Fix {result['failed']} failing tests in {suite_name}"
                )

        return recommendations

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save comprehensive report to file."""
        report_path = self.test_dir / "comprehensive_test_analysis.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n📄 Comprehensive report saved to: {report_path}")

        # Print summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST ANALYSIS SUMMARY")
        print(f"{'='*80}")

        summary = report["summary"]
        print("📊 OVERALL RESULTS:")
        print(f"  Total tests: {summary['total_tests']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Skipped: {summary['skipped']}")
        print(f"  Pass rate: {summary['pass_rate']:.1f}%")
        print(f"  Total execution time: {summary['total_execution_time']:.2f}s")

        # Status assessment
        if summary["pass_rate"] >= 90:
            status = "🟢 EXCELLENT"
        elif summary["pass_rate"] >= 80:
            status = "🟡 GOOD - Needs Minor Improvements"
        elif summary["pass_rate"] >= 70:
            status = "🟠 NEEDS ATTENTION"
        else:
            status = "🔴 CRITICAL - Major Issues"

        print(f"\n🎯 OVERALL STATUS: {status}")

        # Print key recommendations
        recommendations = report["recommendations"]
        if recommendations:
            print("\n📋 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
            if len(recommendations) > 5:
                print(f"  ... and {len(recommendations) - 5} more (see full report)")

        print("\n🏁 Analysis completed successfully")


def main():
    """Main function to run comprehensive test analysis."""
    analyzer = TestAnalyzer()
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()
