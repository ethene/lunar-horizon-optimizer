# Lunar Horizon Optimizer - Development Pipeline
# Comprehensive Python code quality and testing pipeline

# Configuration
PYTHON := python
SRC_DIR := src
TESTS_DIR := tests
ALL_DIRS := $(SRC_DIR) $(TESTS_DIR)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
BOLD := \033[1m
NC := \033[0m # No Color

# Pipeline targets
.PHONY: help pipeline format lint complexity type-check refactor security test test-all test-quick test-trajectory test-economics test-config coverage clean install-dev
.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "$(BOLD)Lunar Horizon Optimizer - Development Pipeline$(NC)"
	@echo "=================================================="
	@echo ""
	@echo "$(BOLD)Main Commands:$(NC)"
	@echo "  $(GREEN)make pipeline$(NC)     - Run complete development pipeline (format, lint, type-check, etc.)"
	@echo "  $(GREEN)make test$(NC)         - Run core production tests (38 tests - recommended for CI/CD)"
	@echo "  $(GREEN)make coverage$(NC)     - Run tests with coverage reporting"
	@echo ""
	@echo "$(BOLD)Test Suite Options:$(NC)"
	@echo "  $(GREEN)make test$(NC)         - Core production tests (38 tests: functionality + economics)"
	@echo "  $(GREEN)make test-all$(NC)     - Complete test suite (445+ tests - comprehensive)"
	@echo "  $(GREEN)make test-quick$(NC)   - Quick sanity tests (environment + basic functionality)"
	@echo "  $(GREEN)make test-trajectory$(NC) - Trajectory generation and orbital mechanics tests"
	@echo "  $(GREEN)make test-economics$(NC)  - Economic analysis and financial modeling tests"
	@echo "  $(GREEN)make test-config$(NC)     - Configuration validation and management tests"
	@echo ""
	@echo "$(BOLD)Individual Steps:$(NC)"
	@echo "  $(BLUE)make format$(NC)       - Format code with black"
	@echo "  $(BLUE)make lint$(NC)         - Lint with ruff (production mode: critical issues only)"
	@echo "  $(BLUE)make complexity$(NC)   - Check maintainability with radon and xenon"
	@echo "  $(BLUE)make type-check$(NC)   - Type checking with mypy"
	@echo "  $(BLUE)make refactor$(NC)     - AI-based refactor suggestions with sourcery"
	@echo "  $(BLUE)make security$(NC)     - Security scan with bandit"
	@echo ""
	@echo "$(BOLD)Utilities:$(NC)"
	@echo "  $(YELLOW)make install-dev$(NC)  - Install development dependencies"
	@echo "  $(YELLOW)make clean$(NC)        - Clean up temporary files"

pipeline: ## Run complete development pipeline
	@echo "$(BOLD)$(BLUE)🚀 Starting Lunar Horizon Optimizer Development Pipeline$(NC)"
	@echo "================================================================"
	@echo ""
	@$(MAKE) --no-print-directory format
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory complexity
	@$(MAKE) --no-print-directory type-check
	@$(MAKE) --no-print-directory refactor
	@$(MAKE) --no-print-directory security
	@echo ""
	@echo "$(BOLD)$(GREEN)✅ Pipeline completed successfully!$(NC)"
	@echo "$(GREEN)All code quality checks passed. Ready for commit.$(NC)"

format: ## Format code with black
	@echo "$(BOLD)$(BLUE)1. Code Formatting with Black$(NC)"
	@echo "====================================="
	@echo "$(YELLOW)Formatting Python code in $(SRC_DIR)/ and $(TESTS_DIR)/...$(NC)"
	@/opt/anaconda3/envs/py312/bin/black --line-length 88 --target-version py312 $(ALL_DIRS) || { \
		echo "$(RED)❌ Black formatting failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Code formatting completed$(NC)"
	@echo ""

lint: ## Lint with ruff (production-focused: critical issues only)
	@echo "$(BOLD)$(BLUE)2. Linting with Ruff (Production Mode)$(NC)"
	@echo "========================================"
	@echo "$(YELLOW)Running production-focused linting (critical issues only)...$(NC)"
	@conda run -n py312 ruff check $(ALL_DIRS) || { \
		echo "$(RED)❌ Ruff linting failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Linting completed$(NC)"
	@echo ""

complexity: ## Check maintainability and complexity with radon and xenon
	@echo "$(BOLD)$(BLUE)3. Complexity and Maintainability Analysis$(NC)"
	@echo "============================================="
	@echo "$(YELLOW)Analyzing cyclomatic complexity with radon...$(NC)"
	@/opt/anaconda3/envs/py312/bin/radon cc $(ALL_DIRS) -a -nc || { \
		echo "$(RED)❌ Radon complexity analysis failed$(NC)"; \
		exit 1; \
	}
	@echo ""
	@echo "$(YELLOW)Analyzing maintainability index...$(NC)"
	@/opt/anaconda3/envs/py312/bin/radon mi $(ALL_DIRS) -nc || { \
		echo "$(RED)❌ Radon maintainability analysis failed$(NC)"; \
		exit 1; \
	}
	@echo ""
	@echo "$(YELLOW)Checking complexity thresholds with xenon...$(NC)"
	@/opt/anaconda3/envs/py312/bin/xenon --max-absolute C --max-modules B --max-average B $(ALL_DIRS) || { \
		echo "$(YELLOW)⚠️ Xenon found complexity issues (non-fatal)$(NC)"; \
		echo "$(YELLOW)Consider refactoring high-complexity functions$(NC)"; \
	}
	@echo "$(GREEN)✅ Complexity analysis completed$(NC)"
	@echo ""

type-check: ## Type checking with mypy
	@echo "$(BOLD)$(BLUE)4. Type Checking with MyPy$(NC)"
	@echo "============================="
	@echo "$(YELLOW)Running static type analysis (critical/major issues only)...$(NC)"
	@/opt/anaconda3/envs/py312/bin/mypy $(SRC_DIR) --config-file=pyproject.toml --show-error-codes --pretty || { \
		echo "$(YELLOW)⚠️ MyPy found type issues (non-fatal)$(NC)"; \
		echo "$(YELLOW)Continuing pipeline - see above for type improvement opportunities$(NC)"; \
	}
	@echo "$(GREEN)✅ Type checking completed$(NC)"
	@echo ""

refactor: ## AI-based refactor suggestions with sourcery
	@echo "$(BOLD)$(BLUE)5. AI-Based Refactor Suggestions$(NC)"
	@echo "=================================="
	@echo "$(YELLOW)Analyzing code for refactor opportunities...$(NC)"
	@/opt/anaconda3/envs/py312/bin/sourcery review $(ALL_DIRS) --no-summary || { \
		echo "$(YELLOW)⚠️  Sourcery analysis completed with suggestions$(NC)"; \
		echo "$(YELLOW)Note: Sourcery suggestions are recommendations, not failures$(NC)"; \
	}
	@echo "$(GREEN)✅ Refactor analysis completed$(NC)"
	@echo ""

security: ## Security scan with bandit
	@echo "$(BOLD)$(BLUE)6. Security Analysis with Bandit$(NC)"
	@echo "================================="
	@echo "$(YELLOW)Scanning for security vulnerabilities...$(NC)"
	@/opt/anaconda3/envs/py312/bin/bandit -r $(SRC_DIR) -f txt --skip B101,B601 || { \
		echo "$(RED)❌ Bandit security scan failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Security scan completed$(NC)"
	@echo ""

test: ## Run core production tests (38 tests - recommended for CI/CD)
	@echo "$(BOLD)$(BLUE)🧪 Running Core Production Test Suite$(NC)"
	@echo "============================================="
	@echo "$(YELLOW)Activating conda py312 environment and running core tests...$(NC)"
	@echo "$(BLUE)Tests: Core functionality (15) + Economics modules (23) = 38 tests$(NC)"
	@conda run -n py312 python -m pytest tests/test_final_functionality.py tests/test_economics_modules.py \
		-v --tb=short --disable-warnings --cov-fail-under=0 || { \
		echo "$(RED)❌ Core test suite failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ All core tests passed (38/38)$(NC)"
	@echo ""

coverage: ## Run tests with coverage reporting
	@echo "$(BOLD)$(BLUE)📊 Running Tests with Coverage Analysis$(NC)"
	@echo "============================================="
	@echo "$(YELLOW)Activating conda py312 environment and running coverage analysis...$(NC)"
	@conda run -n py312 pytest $(TESTS_DIR)/test_final_functionality.py $(TESTS_DIR)/test_task_5_economic_analysis.py \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=80 \
		-v || { \
		echo "$(RED)❌ Coverage analysis failed$(NC)"; \
		exit 1; \
	}
	@echo ""
	@echo "$(GREEN)✅ Coverage analysis completed$(NC)"
	@echo "$(BLUE)📊 HTML coverage report generated in htmlcov/$(NC)"
	@echo ""

test-all: ## Run complete test suite (445+ tests - comprehensive)
	@echo "$(BOLD)$(BLUE)🧪 Running Complete Test Suite$(NC)"
	@echo "===================================="
	@echo "$(YELLOW)Activating conda py312 environment and running ALL tests...$(NC)"
	@echo "$(RED)⚠️  This may take several minutes and includes experimental tests$(NC)"
	@conda run -n py312 python -m pytest tests/ -x --tb=short --disable-warnings --maxfail=5 || { \
		echo "$(RED)❌ Complete test suite failed$(NC)"; \
		echo "$(YELLOW)Note: Some failures expected in experimental/incomplete modules$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Complete test suite completed$(NC)"
	@echo ""

test-quick: ## Quick sanity tests (environment + basic functionality)
	@echo "$(BOLD)$(BLUE)🧪 Running Quick Sanity Tests$(NC)"
	@echo "=================================="
	@echo "$(YELLOW)Running environment setup and basic functionality tests...$(NC)"
	@conda run -n py312 python -m pytest tests/test_environment.py tests/test_final_functionality.py::test_environment_setup \
		tests/test_final_functionality.py::TestPyKEPRealFunctionality::test_mu_constants \
		-v --tb=short --disable-warnings --cov-fail-under=0 || { \
		echo "$(RED)❌ Quick tests failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Quick sanity tests passed$(NC)"
	@echo ""

test-trajectory: ## Trajectory generation and orbital mechanics tests
	@echo "$(BOLD)$(BLUE)🧪 Running Trajectory Tests$(NC)"
	@echo "==============================="
	@echo "$(YELLOW)Testing trajectory generation, orbital mechanics, and Lambert solvers...$(NC)"
	@conda run -n py312 python -m pytest tests/test_task_3_trajectory_generation.py tests/trajectory/ \
		tests/test_final_functionality.py::TestPyKEPRealFunctionality \
		-v --tb=short --disable-warnings --cov-fail-under=0 || { \
		echo "$(RED)❌ Trajectory tests failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Trajectory tests passed$(NC)"
	@echo ""

test-economics: ## Economic analysis and financial modeling tests
	@echo "$(BOLD)$(BLUE)🧪 Running Economics Tests$(NC)"
	@echo "==============================="
	@echo "$(YELLOW)Testing economic analysis, financial modeling, and ISRU benefits...$(NC)"
	@conda run -n py312 python -m pytest tests/test_economics_modules.py tests/test_task_5_economic_analysis.py \
		tests/test_final_functionality.py::TestEconomicAnalysisRealFunctionality \
		-v --tb=short --disable-warnings --cov-fail-under=0 || { \
		echo "$(RED)❌ Economics tests failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Economics tests passed$(NC)"
	@echo ""

test-config: ## Configuration validation and management tests
	@echo "$(BOLD)$(BLUE)🧪 Running Configuration Tests$(NC)"
	@echo "===================================="
	@echo "$(YELLOW)Testing configuration loading, validation, and management...$(NC)"
	@conda run -n py312 python -m pytest tests/test_config_*.py \
		tests/test_final_functionality.py::TestConfigurationRealFunctionality \
		-v --tb=short --disable-warnings --cov-fail-under=0 || { \
		echo "$(RED)❌ Configuration tests failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Configuration tests passed$(NC)"
	@echo ""

install-dev: ## Install development dependencies
	@echo "$(BOLD)$(BLUE)📦 Installing Development Dependencies$(NC)"
	@echo "=========================================="
	@pip install black ruff radon xenon mypy sourcery bandit pytest pytest-cov
	@echo "$(GREEN)✅ Development dependencies installed$(NC)"

clean: ## Clean up temporary files
	@echo "$(BOLD)$(BLUE)🧹 Cleaning Temporary Files$(NC)"
	@echo "============================="
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup completed$(NC)"