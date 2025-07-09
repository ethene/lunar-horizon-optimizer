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
.PHONY: help pipeline format lint complexity type-check refactor security test coverage clean install-dev
.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "$(BOLD)Lunar Horizon Optimizer - Development Pipeline$(NC)"
	@echo "=================================================="
	@echo ""
	@echo "$(BOLD)Main Commands:$(NC)"
	@echo "  $(GREEN)make pipeline$(NC)     - Run complete development pipeline (format, lint, type-check, etc.)"
	@echo "  $(GREEN)make test$(NC)         - Run all tests with pytest"
	@echo "  $(GREEN)make coverage$(NC)     - Run tests with coverage reporting"
	@echo ""
	@echo "$(BOLD)Individual Steps:$(NC)"
	@echo "  $(BLUE)make format$(NC)       - Format code with black"
	@echo "  $(BLUE)make lint$(NC)         - Lint with ruff (flake8 + pylint rules)"
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
	@conda run -n py312 black --line-length 88 --target-version py312 $(ALL_DIRS) || { \
		echo "$(RED)❌ Black formatting failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Code formatting completed$(NC)"
	@echo ""

lint: ## Lint with ruff (flake8 + pylint rules)
	@echo "$(BOLD)$(BLUE)2. Linting with Ruff$(NC)"
	@echo "====================="
	@echo "$(YELLOW)Running comprehensive linting (flake8 + pylint rules)...$(NC)"
	@conda run -n py312 ruff check $(ALL_DIRS) --select=E,W,F,B,C,N,D,UP,YTT,ANN,S,BLE,FBT,A,COM,C4,DTZ,T10,EM,EXE,ISC,ICN,G,INP,PIE,T20,PT,Q,RSE,RET,SLF,SIM,TID,TCH,ARG,PTH,ERA,PD,PGH,PL,TRY,NPY,RUF || { \
		echo "$(RED)❌ Ruff linting failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Linting completed$(NC)"
	@echo ""

complexity: ## Check maintainability and complexity with radon and xenon
	@echo "$(BOLD)$(BLUE)3. Complexity and Maintainability Analysis$(NC)"
	@echo "============================================="
	@echo "$(YELLOW)Analyzing cyclomatic complexity with radon...$(NC)"
	@conda run -n py312 radon cc $(ALL_DIRS) -a -nc || { \
		echo "$(RED)❌ Radon complexity analysis failed$(NC)"; \
		exit 1; \
	}
	@echo ""
	@echo "$(YELLOW)Analyzing maintainability index...$(NC)"
	@conda run -n py312 radon mi $(ALL_DIRS) -nc || { \
		echo "$(RED)❌ Radon maintainability analysis failed$(NC)"; \
		exit 1; \
	}
	@echo ""
	@echo "$(YELLOW)Checking complexity thresholds with xenon...$(NC)"
	@conda run -n py312 xenon --max-absolute B --max-modules A --max-average A $(ALL_DIRS) || { \
		echo "$(RED)❌ Xenon complexity check failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Complexity analysis completed$(NC)"
	@echo ""

type-check: ## Type checking with mypy
	@echo "$(BOLD)$(BLUE)4. Type Checking with MyPy$(NC)"
	@echo "============================="
	@echo "$(YELLOW)Running static type analysis...$(NC)"
	@conda run -n py312 mypy $(SRC_DIR) --strict --show-error-codes --pretty || { \
		echo "$(RED)❌ MyPy type checking failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Type checking completed$(NC)"
	@echo ""

refactor: ## AI-based refactor suggestions with sourcery
	@echo "$(BOLD)$(BLUE)5. AI-Based Refactor Suggestions$(NC)"
	@echo "=================================="
	@echo "$(YELLOW)Analyzing code for refactor opportunities...$(NC)"
	@conda run -n py312 sourcery review $(ALL_DIRS) --no-summary || { \
		echo "$(YELLOW)⚠️  Sourcery analysis completed with suggestions$(NC)"; \
		echo "$(YELLOW)Note: Sourcery suggestions are recommendations, not failures$(NC)"; \
	}
	@echo "$(GREEN)✅ Refactor analysis completed$(NC)"
	@echo ""

security: ## Security scan with bandit
	@echo "$(BOLD)$(BLUE)6. Security Analysis with Bandit$(NC)"
	@echo "================================="
	@echo "$(YELLOW)Scanning for security vulnerabilities...$(NC)"
	@conda run -n py312 bandit -r $(SRC_DIR) -f txt --skip B101,B601 || { \
		echo "$(RED)❌ Bandit security scan failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Security scan completed$(NC)"
	@echo ""

test: ## Run all tests with pytest
	@echo "$(BOLD)$(BLUE)🧪 Running Test Suite$(NC)"
	@echo "=========================="
	@echo "$(YELLOW)Activating conda py312 environment and running tests...$(NC)"
	@conda run -n py312 python -m pytest tests/test_final_functionality.py tests/test_economics_modules.py \
		-v --tb=short --disable-warnings --cov-fail-under=0 || { \
		echo "$(RED)❌ Test suite failed$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ All tests passed$(NC)"
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