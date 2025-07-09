#!/usr/bin/env python3
"""
Comprehensive test runner and validation script for Tasks 3, 4, and 5

This script runs all test suites and validates results for sanity and correctness.
It provides comprehensive reporting on test coverage, performance, and integration.
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class ComprehensiveTestRunner:
    """Comprehensive test runner with validation and reporting."""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.results = {}
        self.start_time = None

    def run_test_suite(self, test_file, suite_name):
        """Run a specific test suite and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")

        test_path = self.test_dir / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_file}")
            return False

        start_time = time.time()

        try:
            # Run pytest with verbose output and capture results
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "--capture=no",
                "--disable-warnings"
            ], check=False, capture_output=True, text=True, timeout=300)

            end_time = time.time()
            execution_time = end_time - start_time

            # Parse results
            output_lines = result.stdout.split("\n")
            error_lines = result.stderr.split("\n") if result.stderr else []

            # Count test results
            passed_count = sum(1 for line in output_lines if " PASSED" in line)
            failed_count = sum(1 for line in output_lines if " FAILED" in line)
            skipped_count = sum(1 for line in output_lines if " SKIPPED" in line)
            error_count = sum(1 for line in output_lines if " ERROR" in line)

            total_tests = passed_count + failed_count + skipped_count + error_count

            success = result.returncode == 0 and failed_count == 0 and error_count == 0

            self.results[suite_name] = {
                "success": success,
                "return_code": result.returncode,
                "execution_time": execution_time,
                "total_tests": total_tests,
                "passed": passed_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "errors": error_count,
                "output": result.stdout,
                "stderr": result.stderr
            }

            # Print summary
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status} - {suite_name}")
            print(f"Tests: {total_tests} total, {passed_count} passed, {failed_count} failed, {skipped_count} skipped")
            print(f"Execution time: {execution_time:.2f}s")

            if failed_count > 0 or error_count > 0:
                print(f"❌ Failed tests: {failed_count}, Errors: {error_count}")
                if result.stderr:
                    print(f"Stderr: {result.stderr[:500]}...")

            return success

        except subprocess.TimeoutExpired:
            print(f"❌ Test suite {suite_name} timed out after 300 seconds")
            self.results[suite_name] = {
                "success": False,
                "error": "timeout",
                "execution_time": 300
            }
            return False

        except Exception as e:
            print(f"❌ Error running {suite_name}: {e}")
            self.results[suite_name] = {
                "success": False,
                "error": str(e),
                "execution_time": 0
            }
            return False

    def validate_test_sanity(self):
        """Validate that test results are sane and correct."""
        print(f"\n{'='*60}")
        print("VALIDATING TEST SANITY AND CORRECTNESS")
        print(f"{'='*60}")

        validation_results = {}

        # 1. Check test coverage
        total_tests = sum(r.get("total_tests", 0) for r in self.results.values())
        total_passed = sum(r.get("passed", 0) for r in self.results.values())
        total_failed = sum(r.get("failed", 0) for r in self.results.values())
        total_skipped = sum(r.get("skipped", 0) for r in self.results.values())

        print("\n📊 OVERALL TEST COVERAGE:")
        print(f"  Total tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Skipped: {total_skipped}")

        pass_rate = total_passed / total_tests if total_tests > 0 else 0
        print(f"  Pass rate: {pass_rate:.1%}")

        validation_results["coverage"] = {
            "total_tests": total_tests,
            "pass_rate": pass_rate,
            "sufficient_coverage": total_tests >= 50,  # Expect at least 50 tests
            "good_pass_rate": pass_rate >= 0.8  # Expect 80%+ pass rate
        }

        # 2. Check test execution performance
        total_time = sum(r.get("execution_time", 0) for r in self.results.values())
        avg_time_per_test = total_time / total_tests if total_tests > 0 else 0

        print("\n⏱️ PERFORMANCE METRICS:")
        print(f"  Total execution time: {total_time:.2f}s")
        print(f"  Average time per test: {avg_time_per_test:.3f}s")

        validation_results["performance"] = {
            "total_time": total_time,
            "avg_time_per_test": avg_time_per_test,
            "reasonable_total_time": total_time < 600,  # Less than 10 minutes
            "reasonable_avg_time": avg_time_per_test < 5.0  # Less than 5s per test
        }

        # 3. Check module coverage
        expected_modules = [
            "Task 3: Trajectory Generation",
            "Task 4: Global Optimization",
            "Task 5: Economic Analysis",
            "Integration Tests"
        ]

        modules_tested = list(self.results.keys())
        module_coverage = len(modules_tested) / len(expected_modules)

        print("\n🧩 MODULE COVERAGE:")
        print(f"  Expected modules: {len(expected_modules)}")
        print(f"  Tested modules: {len(modules_tested)}")
        print(f"  Coverage: {module_coverage:.1%}")

        for module in expected_modules:
            tested = any(module.lower() in tested_module.lower() for tested_module in modules_tested)
            status = "✅" if tested else "❌"
            print(f"  {status} {module}")

        validation_results["modules"] = {
            "expected_count": len(expected_modules),
            "tested_count": len(modules_tested),
            "coverage": module_coverage,
            "complete_coverage": module_coverage >= 1.0
        }

        # 4. Validate critical test functionality
        print("\n🔍 CRITICAL FUNCTIONALITY VALIDATION:")

        critical_checks = {
            "trajectory_generation": self._check_trajectory_tests(),
            "optimization": self._check_optimization_tests(),
            "economic_analysis": self._check_economic_tests(),
            "integration": self._check_integration_tests()
        }

        for check_name, check_result in critical_checks.items():
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name.replace('_', ' ').title()}")

        validation_results["critical_functionality"] = critical_checks

        # 5. Overall validation score
        all_checks = [
            validation_results["coverage"]["sufficient_coverage"],
            validation_results["coverage"]["good_pass_rate"],
            validation_results["performance"]["reasonable_total_time"],
            validation_results["performance"]["reasonable_avg_time"],
            validation_results["modules"]["complete_coverage"],
            all(critical_checks.values())
        ]

        validation_score = sum(all_checks) / len(all_checks)
        overall_pass = validation_score >= 0.8

        print("\n🎯 OVERALL VALIDATION:")
        print(f"  Validation score: {validation_score:.1%}")
        print(f"  Status: {'✅ PASS' if overall_pass else '❌ FAIL'}")

        validation_results["overall"] = {
            "score": validation_score,
            "pass": overall_pass
        }

        return validation_results

    def _check_trajectory_tests(self):
        """Check if trajectory generation tests cover key functionality."""
        task3_results = self.results.get("Task 3: Trajectory Generation", {})

        # Look for key test indicators in output
        output = task3_results.get("output", "")

        key_tests = [
            "lambert",
            "nbody",
            "trajectory",
            "integration",
            "propagation"
        ]

        tests_found = sum(1 for test in key_tests if test.lower() in output.lower())
        return tests_found >= 3 and task3_results.get("success", False)

    def _check_optimization_tests(self):
        """Check if optimization tests cover key functionality."""
        task4_results = self.results.get("Task 4: Global Optimization", {})

        output = task4_results.get("output", "")

        key_tests = [
            "nsga",
            "pygmo",
            "pareto",
            "optimization",
            "fitness"
        ]

        tests_found = sum(1 for test in key_tests if test.lower() in output.lower())
        return tests_found >= 3 and task4_results.get("success", False)

    def _check_economic_tests(self):
        """Check if economic analysis tests cover key functionality."""
        task5_results = self.results.get("Task 5: Economic Analysis", {})

        output = task5_results.get("output", "")

        key_tests = [
            "npv",
            "financial",
            "cost",
            "isru",
            "economic"
        ]

        tests_found = sum(1 for test in key_tests if test.lower() in output.lower())
        return tests_found >= 3 and task5_results.get("success", False)

    def _check_integration_tests(self):
        """Check if integration tests cover key functionality."""
        integration_results = self.results.get("Integration Tests", {})

        output = integration_results.get("output", "")

        key_tests = [
            "integration",
            "end_to_end",
            "workflow",
            "task"
        ]

        tests_found = sum(1 for test in key_tests if test.lower() in output.lower())
        return tests_found >= 2 and integration_results.get("success", False)

    def generate_report(self, validation_results):
        """Generate comprehensive test report."""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST REPORT")
        print(f"{'='*60}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_execution_time": sum(r.get("execution_time", 0) for r in self.results.values()),
                "total_tests": sum(r.get("total_tests", 0) for r in self.results.values()),
                "total_passed": sum(r.get("passed", 0) for r in self.results.values()),
                "total_failed": sum(r.get("failed", 0) for r in self.results.values()),
                "total_skipped": sum(r.get("skipped", 0) for r in self.results.values()),
                "overall_success": validation_results["overall"]["pass"]
            },
            "test_suites": self.results,
            "validation": validation_results
        }

        # Print summary
        print("📋 EXECUTIVE SUMMARY:")
        print(f"  Total test suites: {len(self.results)}")
        print(f"  Total tests: {report['summary']['total_tests']}")
        print(f"  Total passed: {report['summary']['total_passed']}")
        print(f"  Total failed: {report['summary']['total_failed']}")
        print(f"  Total skipped: {report['summary']['total_skipped']}")
        print(f"  Total execution time: {report['summary']['total_execution_time']:.2f}s")
        print(f"  Validation score: {validation_results['overall']['score']:.1%}")
        print(f"  Overall status: {'✅ SUCCESS' if report['summary']['overall_success'] else '❌ FAILURE'}")

        # Save detailed report
        report_path = self.test_dir / "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Detailed report saved to: {report_path}")

        return report

    def run_all_tests(self):
        """Run all test suites and generate comprehensive report."""
        print("🚀 Starting Comprehensive Test Suite for Tasks 3, 4, and 5")
        print(f"Project root: {self.project_root}")
        print(f"Test directory: {self.test_dir}")

        self.start_time = time.time()

        # Define test suites to run
        test_suites = [
            ("test_task_3_trajectory_generation.py", "Task 3: Trajectory Generation"),
            ("test_task_4_global_optimization.py", "Task 4: Global Optimization"),
            ("test_task_5_economic_analysis.py", "Task 5: Economic Analysis"),
            ("test_integration_tasks_3_4_5.py", "Integration Tests")
        ]

        # Run each test suite
        all_success = True
        for test_file, suite_name in test_suites:
            success = self.run_test_suite(test_file, suite_name)
            if not success:
                all_success = False

        # Validate results
        validation_results = self.validate_test_sanity()

        # Generate final report
        report = self.generate_report(validation_results)

        total_time = time.time() - self.start_time
        print(f"\n🏁 Testing completed in {total_time:.2f}s")

        if report["summary"]["overall_success"]:
            print("🎉 ALL TESTS PASSED WITH VALIDATION!")
            return 0
        print("💥 TEST VALIDATION FAILED!")
        return 1


def main():
    """Main function to run comprehensive tests."""
    runner = ComprehensiveTestRunner()
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
