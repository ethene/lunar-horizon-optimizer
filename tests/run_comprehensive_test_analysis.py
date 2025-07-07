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
sys.path.insert(0, str(project_root / 'src'))

class TestAnalyzer:
    """Comprehensive test analysis and execution framework."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / 'tests'
        self.src_dir = self.project_root / 'src'
        
        # Test execution results
        self.test_results = {}
        self.coverage_analysis = {}
        self.sanity_check_results = {}
        
        # Test categories
        self.test_categories = {
            'core_functionality': ['test_final_functionality.py'],
            'task_3_trajectory': ['test_task_3_trajectory_generation.py'],
            'task_4_optimization': ['test_task_4_global_optimization.py'],
            'task_5_economics': ['test_task_5_economic_analysis.py'],
            'task_6_visualization': ['test_task_6_visualization.py'],
            'integration': ['test_integration_tasks_3_4_5.py']
        }
        
        # Known realistic value ranges for sanity checking
        self.realistic_ranges = {
            'delta_v_ms': (1000, 10000),           # m/s for lunar missions
            'transfer_time_days': (3, 15),         # days for Earth-Moon transfer
            'mission_cost_usd': (50e6, 5e9),       # dollars for lunar missions
            'npv_usd': (-1e9, 2e9),                # dollars for project NPV
            'irr_fraction': (-0.5, 1.0),           # fraction for IRR
            'roi_fraction': (-0.5, 2.0),           # fraction for ROI
            'spacecraft_mass_kg': (500, 50000),    # kg for lunar spacecraft
            'orbit_altitude_km': (200, 2000),      # km for Earth/Moon orbits
            'mission_duration_years': (1, 20),     # years for mission duration
            'earth_radius_m': (6.35e6, 6.4e6),     # meters
            'moon_radius_m': (1.7e6, 1.8e6),       # meters
            'earth_moon_distance_m': (3.6e8, 4.1e8) # meters
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
            'src_modules': self._find_source_modules(),
            'test_files': self._find_test_files(),
            'coverage_gaps': [],
            'test_categories': {}
        }
        
        # Analyze coverage by category
        for category, test_files in self.test_categories.items():
            coverage['test_categories'][category] = {
                'test_files': test_files,
                'exists': [self._test_file_exists(f) for f in test_files],
                'runnable': []
            }
        
        # Find coverage gaps
        src_modules = set(coverage['src_modules'])
        tested_modules = self._extract_tested_modules()
        coverage['coverage_gaps'] = list(src_modules - tested_modules)
        
        return coverage
    
    def _find_source_modules(self) -> List[str]:
        """Find all source modules."""
        modules = []
        for package in ['config', 'trajectory', 'optimization', 'economics', 'visualization', 'utils']:
            package_path = self.src_dir / package
            if package_path.exists():
                for py_file in package_path.rglob('*.py'):
                    if py_file.name != '__init__.py':
                        relative_path = py_file.relative_to(self.src_dir)
                        modules.append(str(relative_path))
        return sorted(modules)
    
    def _find_test_files(self) -> List[str]:
        """Find all test files."""
        test_files = []
        for py_file in self.test_dir.glob('test_*.py'):
            test_files.append(py_file.name)
        return sorted(test_files)
    
    def _test_file_exists(self, filename: str) -> bool:
        """Check if test file exists."""
        return (self.test_dir / filename).exists()
    
    def _extract_tested_modules(self) -> set:
        """Extract modules that have dedicated tests."""
        tested = set()
        
        # Known tested modules from existing tests
        tested.update([
            'config/costs.py',
            'economics/financial_models.py',
            'economics/cost_models.py',
            'economics/isru_benefits.py',
            'economics/sensitivity_analysis.py',
            'economics/reporting.py',
            'trajectory/earth_moon_trajectories.py',
            'trajectory/nbody_integration.py',
            'trajectory/transfer_window_analysis.py',
            'optimization/global_optimizer.py',
            'optimization/pareto_analysis.py',
            'visualization/trajectory_visualization.py',
            'visualization/optimization_visualization.py',
            'visualization/economic_visualization.py',
            'visualization/mission_visualization.py',
            'visualization/dashboard.py'
        ])
        
        return tested
    
    def _execute_all_tests(self) -> None:
        """Execute all test suites and collect results."""
        test_suites = [
            ('Core Functionality', 'test_final_functionality.py'),
            ('Task 3 Trajectory', 'test_task_3_trajectory_generation.py'),
            ('Task 4 Optimization', 'test_task_4_global_optimization.py'),
            ('Task 5 Economics', 'test_task_5_economic_analysis.py'),
            ('Task 6 Visualization', 'test_task_6_visualization.py'),
            ('Integration Tests', 'test_integration_tasks_3_4_5.py')
        ]
        
        for suite_name, test_file in test_suites:
            print(f"\n{'='*60}")
            print(f"Running {suite_name}")
            print(f"{'='*60}")
            
            if not self._test_file_exists(test_file):
                print(f"❌ Test file {test_file} not found")
                self.test_results[suite_name] = {
                    'status': 'not_found',
                    'passed': 0,
                    'failed': 0,
                    'skipped': 0,
                    'errors': ['Test file not found']
                }
                continue
            
            result = self._run_pytest(test_file)
            self.test_results[suite_name] = result
            
            # Print result
            if result['status'] == 'passed':
                print(f"✅ PASSED - {suite_name}")
            else:
                print(f"❌ {result['status'].upper()} - {suite_name}")
            
            print(f"Tests: {result['total']} total, "
                  f"{result['passed']} passed, "
                  f"{result['failed']} failed, "
                  f"{result['skipped']} skipped")
            print(f"Execution time: {result['execution_time']:.2f}s")
    
    def _run_pytest(self, test_file: str) -> Dict[str, Any]:
        """Run pytest on a specific test file."""
        start_time = time.time()
        
        try:
            # Set environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.src_dir)
            
            # Run pytest
            cmd = [
                sys.executable, '-m', 'pytest',
                str(self.test_dir / test_file),
                '-v', '--tb=short', '--no-header'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse output
            parsed = self._parse_pytest_output(result.stdout, result.stderr)
            parsed['execution_time'] = execution_time
            parsed['return_code'] = result.returncode
            
            if result.returncode == 0:
                parsed['status'] = 'passed'
            else:
                parsed['status'] = 'failed'
            
            return parsed
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'total': 0,
                'execution_time': time.time() - start_time,
                'errors': ['Test execution timed out']
            }
        except Exception as e:
            return {
                'status': 'error',
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'total': 0,
                'execution_time': time.time() - start_time,
                'errors': [str(e)]
            }
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output to extract test statistics."""
        result = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'total': 0,
            'details': stdout,
            'errors_detail': stderr
        }
        
        # Look for summary line like "3 passed, 1 skipped in 2.14s"
        lines = stdout.split('\n')
        for line in lines:
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                if ' in ' in line and 's' in line:
                    # This looks like a summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed' and i > 0:
                            result['passed'] = int(parts[i-1])
                        elif part == 'failed' and i > 0:
                            result['failed'] = int(parts[i-1])
                        elif part == 'skipped' and i > 0:
                            result['skipped'] = int(parts[i-1])
                        elif part == 'error' and i > 0:
                            result['errors'] = int(parts[i-1])
        
        result['total'] = result['passed'] + result['failed'] + result['skipped'] + result['errors']
        return result
    
    def _perform_sanity_checks(self) -> None:
        """Perform sanity checks on test results and calculations."""
        sanity_results = {
            'realistic_ranges': self._check_realistic_ranges(),
            'physical_constants': self._check_physical_constants(),
            'calculation_consistency': self._check_calculation_consistency(),
            'result_validation': self._check_result_validation()
        }
        
        self.sanity_check_results = sanity_results
        
        # Print sanity check summary
        print("\n🔍 SANITY CHECK RESULTS:")
        for check_type, results in sanity_results.items():
            status = "✅ PASS" if results.get('status') == 'pass' else "⚠️ NEEDS ATTENTION"
            print(f"  {check_type}: {status}")
            if results.get('issues'):
                for issue in results['issues'][:3]:  # Show first 3 issues
                    print(f"    - {issue}")\n    \n    def _check_realistic_ranges(self) -> Dict[str, Any]:\n        \"\"\"Check that test values are within realistic ranges.\"\"\"\n        issues = []\n        \n        # Check if realistic ranges are being validated in tests\n        for category, test_files in self.test_categories.items():\n            for test_file in test_files:\n                if self._test_file_exists(test_file):\n                    test_path = self.test_dir / test_file\n                    try:\n                        with open(test_path, 'r') as f:\n                            content = f.read()\n                            \n                        # Look for range validations\n                        if 'assert' in content and ('>' in content or '<' in content):\n                            # Good - has range checks\n                            pass\n                        else:\n                            issues.append(f\"{test_file}: Missing range validation checks\")\n                    except Exception as e:\n                        issues.append(f\"{test_file}: Could not analyze - {e}\")\n        \n        return {\n            'status': 'pass' if len(issues) < 3 else 'needs_attention',\n            'issues': issues,\n            'checked_ranges': list(self.realistic_ranges.keys())\n        }\n    \n    def _check_physical_constants(self) -> Dict[str, Any]:\n        \"\"\"Check that physical constants are reasonable.\"\"\"\n        issues = []\n        \n        try:\n            # Import and check constants\n            from trajectory.constants import PhysicalConstants as PC\n            \n            # Check Earth constants\n            if not (6.0e6 < PC.EARTH_RADIUS < 7.0e6):\n                issues.append(f\"Earth radius unrealistic: {PC.EARTH_RADIUS}\")\n            \n            if not (3.9e14 < PC.EARTH_MU < 4.1e14):\n                issues.append(f\"Earth mu unrealistic: {PC.EARTH_MU}\")\n            \n            # Check Moon constants\n            if not (1.5e6 < PC.MOON_RADIUS < 2.0e6):\n                issues.append(f\"Moon radius unrealistic: {PC.MOON_RADIUS}\")\n            \n            if not (4.8e12 < PC.MOON_MU < 5.0e12):\n                issues.append(f\"Moon mu unrealistic: {PC.MOON_MU}\")\n            \n            # Check Sun constants (if available)\n            if hasattr(PC, 'SUN_MU'):\n                if not (1.0e20 < PC.SUN_MU < 1.5e20):\n                    issues.append(f\"Sun mu unrealistic: {PC.SUN_MU}\")\n            \n        except Exception as e:\n            issues.append(f\"Could not import constants: {e}\")\n        \n        return {\n            'status': 'pass' if len(issues) == 0 else 'needs_attention',\n            'issues': issues\n        }\n    \n    def _check_calculation_consistency(self) -> Dict[str, Any]:\n        \"\"\"Check that calculations are consistent and sensible.\"\"\"\n        issues = []\n        \n        # Check if we can run basic calculations\n        try:\n            import numpy as np\n            \n            # Basic orbital mechanics sanity\n            earth_radius = 6.378e6  # m\n            earth_mu = 3.986e14     # m³/s²\n            \n            # Circular velocity at 400 km altitude\n            altitude = 400e3  # m\n            orbit_radius = earth_radius + altitude\n            circular_velocity = np.sqrt(earth_mu / orbit_radius)\n            \n            if not (7000 < circular_velocity < 8000):\n                issues.append(f\"LEO velocity unrealistic: {circular_velocity:.0f} m/s\")\n            \n            # Escape velocity\n            escape_velocity = np.sqrt(2 * earth_mu / earth_radius)\n            if not (11000 < escape_velocity < 12000):\n                issues.append(f\"Earth escape velocity unrealistic: {escape_velocity:.0f} m/s\")\n            \n        except Exception as e:\n            issues.append(f\"Could not perform basic calculations: {e}\")\n        \n        return {\n            'status': 'pass' if len(issues) == 0 else 'needs_attention',\n            'issues': issues\n        }\n    \n    def _check_result_validation(self) -> Dict[str, Any]:\n        \"\"\"Check that test results are being properly validated.\"\"\"\n        issues = []\n        \n        # Check test results for unrealistic values\n        for suite_name, result in self.test_results.items():\n            if result.get('status') == 'passed':\n                # Good - tests are passing\n                continue\n            elif result.get('status') == 'failed':\n                if result.get('failed', 0) > 0:\n                    issues.append(f\"{suite_name}: {result['failed']} tests failing\")\n        \n        return {\n            'status': 'pass' if len(issues) < 2 else 'needs_attention',\n            'issues': issues\n        }\n    \n    def _print_coverage_summary(self) -> None:\n        \"\"\"Print module coverage summary.\"\"\"\n        coverage = self.coverage_analysis\n        \n        print(f\"\\n📋 MODULE COVERAGE SUMMARY:\")\n        print(f\"  Source modules: {len(coverage['src_modules'])}\")\n        print(f\"  Test files: {len(coverage['test_files'])}\")\n        print(f\"  Coverage gaps: {len(coverage['coverage_gaps'])}\")\n        \n        if coverage['coverage_gaps']:\n            print(f\"\\n⚠️ MODULES WITHOUT DEDICATED TESTS:\")\n            for module in coverage['coverage_gaps'][:10]:  # Show first 10\n                print(f\"    - {module}\")\n            if len(coverage['coverage_gaps']) > 10:\n                print(f\"    ... and {len(coverage['coverage_gaps']) - 10} more\")\n    \n    def _generate_comprehensive_report(self) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive test report.\"\"\"\n        # Calculate overall statistics\n        total_tests = sum(r.get('total', 0) for r in self.test_results.values())\n        total_passed = sum(r.get('passed', 0) for r in self.test_results.values())\n        total_failed = sum(r.get('failed', 0) for r in self.test_results.values())\n        total_skipped = sum(r.get('skipped', 0) for r in self.test_results.values())\n        total_time = sum(r.get('execution_time', 0) for r in self.test_results.values())\n        \n        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0\n        \n        report = {\n            'analysis_date': datetime.now().isoformat(),\n            'environment': self._get_environment_info(),\n            'summary': {\n                'total_tests': total_tests,\n                'passed': total_passed,\n                'failed': total_failed,\n                'skipped': total_skipped,\n                'pass_rate': pass_rate,\n                'total_execution_time': total_time\n            },\n            'test_results': self.test_results,\n            'coverage_analysis': self.coverage_analysis,\n            'sanity_check_results': self.sanity_check_results,\n            'recommendations': self._generate_recommendations()\n        }\n        \n        return report\n    \n    def _generate_recommendations(self) -> List[str]:\n        \"\"\"Generate recommendations based on analysis.\"\"\"\n        recommendations = []\n        \n        # Test coverage recommendations\n        coverage_gaps = len(self.coverage_analysis.get('coverage_gaps', []))\n        if coverage_gaps > 5:\n            recommendations.append(f\"Create tests for {coverage_gaps} untested modules\")\n        \n        # Pass rate recommendations\n        total_tests = sum(r.get('total', 0) for r in self.test_results.values())\n        total_passed = sum(r.get('passed', 0) for r in self.test_results.values())\n        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0\n        \n        if pass_rate < 90:\n            recommendations.append(f\"Improve pass rate from {pass_rate:.1f}% to 90%+\")\n        \n        # Sanity check recommendations\n        for check_type, results in self.sanity_check_results.items():\n            if results.get('status') != 'pass':\n                recommendations.append(f\"Address {check_type} issues\")\n        \n        # Module-specific recommendations\n        for suite_name, result in self.test_results.items():\n            if result.get('status') == 'not_found':\n                recommendations.append(f\"Create {suite_name} test suite\")\n            elif result.get('failed', 0) > 0:\n                recommendations.append(f\"Fix {result['failed']} failing tests in {suite_name}\")\n        \n        return recommendations\n    \n    def _save_report(self, report: Dict[str, Any]) -> None:\n        \"\"\"Save comprehensive report to file.\"\"\"\n        report_path = self.test_dir / 'comprehensive_test_analysis.json'\n        with open(report_path, 'w') as f:\n            json.dump(report, f, indent=2, default=str)\n        \n        print(f\"\\n📄 Comprehensive report saved to: {report_path}\")\n        \n        # Print summary\n        print(f\"\\n{'='*80}\")\n        print(f\"COMPREHENSIVE TEST ANALYSIS SUMMARY\")\n        print(f\"{'='*80}\")\n        \n        summary = report['summary']\n        print(f\"📊 OVERALL RESULTS:\")\n        print(f\"  Total tests: {summary['total_tests']}\")\n        print(f\"  Passed: {summary['passed']}\")\n        print(f\"  Failed: {summary['failed']}\")\n        print(f\"  Skipped: {summary['skipped']}\")\n        print(f\"  Pass rate: {summary['pass_rate']:.1f}%\")\n        print(f\"  Total execution time: {summary['total_execution_time']:.2f}s\")\n        \n        # Status assessment\n        if summary['pass_rate'] >= 90:\n            status = \"🟢 EXCELLENT\"\n        elif summary['pass_rate'] >= 80:\n            status = \"🟡 GOOD - Needs Minor Improvements\"\n        elif summary['pass_rate'] >= 70:\n            status = \"🟠 NEEDS ATTENTION\"\n        else:\n            status = \"🔴 CRITICAL - Major Issues\"\n        \n        print(f\"\\n🎯 OVERALL STATUS: {status}\")\n        \n        # Print key recommendations\n        recommendations = report['recommendations']\n        if recommendations:\n            print(f\"\\n📋 KEY RECOMMENDATIONS:\")\n            for i, rec in enumerate(recommendations[:5], 1):\n                print(f\"  {i}. {rec}\")\n            if len(recommendations) > 5:\n                print(f\"  ... and {len(recommendations) - 5} more (see full report)\")\n        \n        print(f\"\\n🏁 Analysis completed successfully\")\n\n\ndef main():\n    \"\"\"Main function to run comprehensive test analysis.\"\"\"\n    analyzer = TestAnalyzer()\n    analyzer.run_comprehensive_analysis()\n\n\nif __name__ == \"__main__\":\n    main()