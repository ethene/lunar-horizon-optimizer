#!/bin/bash
# Lunar Horizon Optimizer - Development Pipeline Script
# Comprehensive Python code quality and testing pipeline

set -e  # Exit on any error

# Configuration
SRC_DIR="src"
TESTS_DIR="tests"
ALL_DIRS="$SRC_DIR $TESTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "$(printf '=%.0s' {1..${#1}})"
}

print_step() {
    echo -e "${YELLOW}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to show help
show_help() {
    echo -e "${BOLD}Lunar Horizon Optimizer - Development Pipeline${NC}"
    echo "=================================================="
    echo ""
    echo -e "${BOLD}Usage:${NC}"
    echo "  $0 [command]"
    echo ""
    echo -e "${BOLD}Commands:${NC}"
    echo -e "  ${GREEN}pipeline${NC}     - Run complete development pipeline (default)"
    echo -e "  ${GREEN}test${NC}         - Run all tests with pytest"
    echo -e "  ${GREEN}coverage${NC}     - Run tests with coverage reporting"
    echo -e "  ${BLUE}format${NC}       - Format code with black"
    echo -e "  ${BLUE}lint${NC}         - Lint with ruff"
    echo -e "  ${BLUE}complexity${NC}   - Check complexity with radon and xenon"
    echo -e "  ${BLUE}type-check${NC}   - Type checking with mypy"
    echo -e "  ${BLUE}refactor${NC}     - AI-based refactor suggestions with sourcery"
    echo -e "  ${BLUE}security${NC}     - Security scan with bandit"
    echo -e "  ${YELLOW}install-dev${NC}  - Install development dependencies"
    echo -e "  ${YELLOW}clean${NC}        - Clean up temporary files"
    echo -e "  ${YELLOW}help${NC}         - Show this help message"
    echo ""
}

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

# Function to run code formatting
run_format() {
    print_header "1. Code Formatting with Black"
    print_step "Formatting Python code in $SRC_DIR/ and $TESTS_DIR/..."
    
    if black --line-length 88 --target-version py312 $ALL_DIRS; then
        print_success "Code formatting completed"
    else
        print_error "Black formatting failed"
        exit 1
    fi
    echo ""
}

# Function to run linting
run_lint() {
    print_header "2. Linting with Ruff"
    print_step "Running comprehensive linting (flake8 + pylint rules)..."
    
    if ruff check $ALL_DIRS --select=E,W,F,B,C,N,D,UP,YTT,ANN,S,BLE,FBT,A,COM,C4,DTZ,T10,EM,EXE,ISC,ICN,G,INP,PIE,T20,PT,Q,RSE,RET,SLF,SIM,TID,TCH,ARG,PTH,ERA,PD,PGH,PL,TRY,NPY,RUF; then
        print_success "Linting completed"
    else
        print_error "Ruff linting failed"
        exit 1
    fi
    echo ""
}

# Function to run complexity analysis
run_complexity() {
    print_header "3. Complexity and Maintainability Analysis"
    print_step "Analyzing cyclomatic complexity with radon..."
    
    if radon cc $ALL_DIRS -a -nc; then
        echo ""
        print_step "Analyzing maintainability index..."
        if radon mi $ALL_DIRS -nc; then
            echo ""
            print_step "Checking complexity thresholds with xenon..."
            if xenon --max-absolute B --max-modules A --max-average A $ALL_DIRS; then
                print_success "Complexity analysis completed"
            else
                print_error "Xenon complexity check failed"
                exit 1
            fi
        else
            print_error "Radon maintainability analysis failed"
            exit 1
        fi
    else
        print_error "Radon complexity analysis failed"
        exit 1
    fi
    echo ""
}

# Function to run type checking
run_type_check() {
    print_header "4. Type Checking with MyPy"
    print_step "Running static type analysis..."
    
    if mypy $SRC_DIR --strict --show-error-codes --pretty; then
        print_success "Type checking completed"
    else
        print_error "MyPy type checking failed"
        exit 1
    fi
    echo ""
}

# Function to run refactor suggestions
run_refactor() {
    print_header "5. AI-Based Refactor Suggestions"
    print_step "Analyzing code for refactor opportunities..."
    
    if sourcery review $ALL_DIRS --no-summary; then
        print_success "Refactor analysis completed"
    else
        print_warning "Sourcery analysis completed with suggestions"
        print_warning "Note: Sourcery suggestions are recommendations, not failures"
    fi
    echo ""
}

# Function to run security scan
run_security() {
    print_header "6. Security Analysis with Bandit"
    print_step "Scanning for security vulnerabilities..."
    
    if bandit -r $SRC_DIR -f text --skip B101,B601; then
        print_success "Security scan completed"
    else
        print_error "Bandit security scan failed"
        exit 1
    fi
    echo ""
}

# Function to run the complete pipeline
run_pipeline() {
    echo -e "${BOLD}${BLUE}🚀 Starting Lunar Horizon Optimizer Development Pipeline${NC}"
    echo "================================================================"
    echo ""
    
    # Check required commands
    check_command black
    check_command ruff
    check_command radon
    check_command xenon
    check_command mypy
    check_command sourcery
    check_command bandit
    
    # Run all steps
    run_format
    run_lint
    run_complexity
    run_type_check
    run_refactor
    run_security
    
    echo ""
    echo -e "${BOLD}${GREEN}✅ Pipeline completed successfully!${NC}"
    echo -e "${GREEN}All code quality checks passed. Ready for commit.${NC}"
}

# Function to run tests
run_tests() {
    print_header "🧪 Running Test Suite"
    print_step "Activating conda py312 environment and running tests..."
    
    if conda run -n py312 python tests/run_working_tests.py; then
        print_success "All tests passed"
    else
        print_error "Test suite failed"
        exit 1
    fi
    echo ""
}

# Function to run coverage
run_coverage() {
    print_header "📊 Running Tests with Coverage Analysis"
    print_step "Activating conda py312 environment and running coverage analysis..."
    
    if conda run -n py312 pytest tests/test_final_functionality.py tests/test_task_5_economic_analysis.py \
        --cov=$SRC_DIR \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        --cov-fail-under=80 \
        -v; then
        echo ""
        print_success "Coverage analysis completed"
        echo -e "${BLUE}📊 HTML coverage report generated in htmlcov/${NC}"
    else
        print_error "Coverage analysis failed"
        exit 1
    fi
    echo ""
}

# Function to install development dependencies
install_dev() {
    print_header "📦 Installing Development Dependencies"
    print_step "Installing required packages..."
    
    if pip install black ruff radon xenon mypy sourcery bandit pytest pytest-cov; then
        print_success "Development dependencies installed"
    else
        print_error "Failed to install development dependencies"
        exit 1
    fi
}

# Function to clean temporary files
clean() {
    print_header "🧹 Cleaning Temporary Files"
    print_step "Removing cache and temporary files..."
    
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name ".coverage" -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-pipeline}" in
    "pipeline")
        run_pipeline
        ;;
    "format")
        check_command black
        run_format
        ;;
    "lint")
        check_command ruff
        run_lint
        ;;
    "complexity")
        check_command radon
        check_command xenon
        run_complexity
        ;;
    "type-check")
        check_command mypy
        run_type_check
        ;;
    "refactor")
        check_command sourcery
        run_refactor
        ;;
    "security")
        check_command bandit
        run_security
        ;;
    "test")
        run_tests
        ;;
    "coverage")
        check_command pytest
        run_coverage
        ;;
    "install-dev")
        install_dev
        ;;
    "clean")
        clean
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac