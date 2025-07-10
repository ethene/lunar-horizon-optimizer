# CLAUDE.md - Project Working Rules

## Project Overview

The **Lunar Horizon Optimizer** is an integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions. This document establishes the working rules, coding standards, and development practices for this project when working with Claude Code.

## Project Structure & Architecture

### Core Components
```
src/
├── config/          # Mission configuration and parameter management
├── trajectory/      # Orbital mechanics and trajectory calculations  
├── optimization/    # Global optimization with PyGMO (Task 4) ✅
├── economics/       # Economic analysis and financial modeling (Task 5) ✅
├── utils/           # Utility functions and unit conversions
└── constants/       # Physical constants and parameters

tests/               # Comprehensive test suite (415 total tests, production core: 100% pass rate)
docs/                # Project documentation (comprehensive)
tasks/               # Development task management
scripts/             # Utility scripts and PRD
```

### Key Technologies
- **PyKEP 2.6** - High-fidelity orbital mechanics
- **PyGMO 2.19.6** - Global optimization algorithms (NSGA-II)
- **JAX 0.5.3** + **Diffrax 0.7.0** - Differentiable programming
- **SciPy 1.13.1** - Scientific computing foundation
- **Plotly 5.24.1** - Interactive visualization
- **Pydantic** - Configuration validation
- **SpiceyPy** - NASA SPICE toolkit for ephemeris calculations

### Development Environment Requirements
- **Python Environment**: conda py312 environment (required - ALREADY CONFIGURED)
- **Local Environment**: The py312 environment is pre-configured with PyKEP, PyGMO, and all dependencies
- **Testing Environment**: ALWAYS use `conda activate py312` for all testing operations
- **Environment Setup** (if needed on new systems): 
  ```bash
  conda create -n py312 python=3.12 -y
  conda activate py312
  conda install -c conda-forge pykep pygmo astropy spiceypy -y
  pip install -r requirements.txt
  ```
- **CRITICAL**: All testing must be done in the py312 environment to ensure PyKEP/PyGMO compatibility

## Development Standards

### Code Quality Requirements

#### 1. **Testing is Mandatory**
- **Test Suite Size**: 415 total tests across 34 test files
- **Production Tests**: 38 core tests (100% pass rate required for commits)
- **Test Types**: Unit tests, integration tests, and validation tests
- **Test Location**: All tests in `tests/` directory with clear naming
- **Before Committing**: Always run `make test` in conda py312 environment and ensure 100% pass rate
- **Known Issues**: Some trajectory/optimization tests have failures due to Task 3 incompleteness
- **NO MOCKING RULE**: NEVER use mocks when real functionality exists - always examine existing code first
- **Real Implementation First**: Use actual classes and functions instead of mocks for all testing

#### 2. **Code Style & Formatting** 
- **Linting**: Run `flake8` for code quality checks
- **Formatting**: Use `black` for consistent code formatting  
- **Type Hints**: All functions must include type hints
- **Docstrings**: All public functions/classes require comprehensive docstrings

#### 3. **Architecture Principles**
- **Single Responsibility**: Each module has one clear purpose
- **Unit Consistency**: Always use PyKEP native units (meters, m/s, radians)
- **Error Handling**: Comprehensive validation and meaningful error messages
- **Logging**: Use structured logging for debugging and monitoring

### Makefile Commands Management

#### Makefile Command Policy
- **Comprehensive Documentation**: Always document all available make commands
- **Consistency**: Maintain a standard format for command naming and usage
- **Testing Requirement**: Every command should have an associated testing mechanism
- **Commit Policy**: After every fix or change, run tests and commit immediately to maintain rollback capability

#### Available Make Commands

##### **Main Development Commands**
- **`make help`**: Display comprehensive help with all available commands
- **`make pipeline`**: Run complete development pipeline (format, lint, type-check, security, etc.)
- **`make test`**: Run production test suite (38 curated tests) in conda py312 environment
- **`make coverage`**: Run tests with comprehensive coverage reporting and HTML output

##### **Code Quality Commands**
- **`make format`**: Format code with Black (line length 88, Python 3.12 target)
- **`make lint`**: Comprehensive linting with Ruff (flake8 + pylint rules)
- **`make complexity`**: Analyze code complexity with Radon and Xenon
- **`make type-check`**: Static type analysis with MyPy in strict mode
- **`make refactor`**: AI-based refactor suggestions with Sourcery
- **`make security`**: Security vulnerability scan with Bandit

##### **Utility Commands**
- **`make install-dev`**: Install all development dependencies
- **`make clean`**: Remove temporary files (__pycache__, .pytest_cache, etc.)

#### Command Usage Guidelines

##### **Pre-Commit Workflow (MANDATORY)**
```bash
# Always run before committing changes:
conda activate py312       # Activate required environment
make test                 # Ensure all tests pass
make pipeline            # Run complete quality checks
git add .               # Stage changes
git commit -m "descriptive message"  # Commit with clear message
```

##### **Environment Requirements**
- **CRITICAL**: Always activate `conda py312` environment before running ANY make commands
- All make commands automatically use `conda run -n py312` to ensure environment consistency
- Never run make commands without proper conda environment activation

##### **Command Execution Order**
1. **Development**: `make format` → `make lint` → `make test`
2. **Pre-Commit**: `make pipeline` (runs all quality checks)
3. **Coverage Analysis**: `make coverage` (generates HTML reports)
4. **Cleanup**: `make clean` (when needed)

#### Git Commit Policy and Rollback Strategy

##### **Mandatory Commit Policy**
- **After Every Major Change**: ALWAYS commit immediately after completing any significant modification
- **Test Before Commit**: Run `make test` and ensure 100% pass rate before committing
- **Quality Checks**: Run `make pipeline` to verify code quality standards
- **Descriptive Messages**: Use clear, descriptive commit messages following conventional commit format

##### **Rollback and State Management**
- **Frequent Commits**: Enable easy rollback to previous working states
- **State Comparison**: Use `git diff` and `git log` to compare with previous states
- **Rollback Commands**:
  ```bash
  git log --oneline -10        # View recent commits
  git diff HEAD~1              # Compare with previous commit
  git reset --hard HEAD~1      # Rollback to previous commit (DESTRUCTIVE)
  git revert <commit-hash>     # Safe rollback creating new commit
  git stash                    # Temporarily save uncommitted changes
  ```

##### **Branch Protection Strategy**
- **Main Branch**: Always keep main branch in working state
- **Feature Branches**: Use for experimental changes
- **Testing**: Ensure all tests pass before merging to main
- **Backup**: Regular commits provide restore points

#### Error Handling and Recovery

##### **Test Failures**
```bash
# If make test fails:
conda activate py312                    # Ensure correct environment
python -m pytest tests/ -v --tb=short  # Run with detailed output
# Fix issues, then re-run make test
```

##### **Pipeline Failures**
```bash
# If make pipeline fails:
make format     # Fix formatting issues
make lint       # Address linting errors
make type-check # Resolve type issues
make security   # Fix security vulnerabilities
```

##### **Environment Issues**
```bash
# If conda environment issues:
conda activate py312
conda install -c conda-forge pykep pygmo astropy spiceypy -y
pip install -r requirements.txt
```

#### Testing Strategy

##### **Production Test Suite (38 tests) - RECOMMENDED**
- **Command**: `make test`
- **Coverage**: Core functionality (15) + Economics modules (23) = 38 tests
- **Requirement**: 100% pass rate for commits (CI/CD ready)
- **Time**: ~5 seconds execution
- **Purpose**: Essential validation for production readiness

##### **Comprehensive Test Suite Options**

**Core Test Suites:**
- **`make test`** - Production tests (38 tests: functionality + economics) - 100% pass rate required
- **`make test-all`** - Complete test suite (415 tests - includes trajectory/optimization modules with some known failures)
- **`make test-quick`** - Quick sanity tests (9 tests: environment + basic functionality) - 100% pass rate

**Specialized Test Suites:**
- **`make test-trajectory`** - Trajectory generation and orbital mechanics tests
- **`make test-economics`** - Economic analysis and financial modeling tests (64 tests)
- **`make test-config`** - Configuration validation and management tests

**Test Suite Details:**
```bash
# Production tests (recommended for daily use)
make test                    # 38 tests, ~5s, 100% must pass

# Quick environment validation
make test-quick             # 9 tests, ~7s, environment + basic functionality

# Comprehensive testing (development)
make test-all               # 415 tests, ~60s, includes trajectory/optimization modules (some failures expected)

# Domain-specific testing
make test-economics         # 64 tests, ~4s, financial modeling + ISRU + sensitivity
make test-trajectory        # varies, orbital mechanics + Lambert solvers
make test-config           # varies, configuration loading + validation
```

##### **Coverage Analysis**
- **Command**: `make coverage`
- **Output**: Terminal report + HTML in `htmlcov/`
- **Threshold**: 80% minimum coverage required
- **Review**: Check `htmlcov/index.html` for detailed analysis

#### Performance and Monitoring

##### **Execution Times**
- `make test`: ~5 seconds (38 production tests)
- `make test-quick`: ~7 seconds (9 sanity tests)
- `make test-economics`: ~4 seconds (64 economics tests)
- `make test-all`: ~60 seconds (415 comprehensive tests, some failures expected in trajectory/optimization)
- `make lint`: <1 second (production-focused)
- `make coverage`: ~45 seconds
- `make pipeline`: ~2-3 minutes (includes all steps)

##### **Resource Usage**
- Memory: ~500MB peak during testing
- Disk: ~100MB for coverage reports
- CPU: Multi-core utilization during parallel tests

#### Troubleshooting Guide

##### **Common Issues**
1. **Environment not activated**: Always run `conda activate py312`
2. **Missing dependencies**: Run `make install-dev`
3. **Cache issues**: Run `make clean` to clear temporary files
4. **Test failures**: Check specific test output and fix underlying issues
5. **Type errors**: Review MyPy output and add proper type annotations

##### **Debug Commands**
```bash
# Verbose test output:
conda activate py312
python -m pytest tests/test_final_functionality.py -v -s

# Check environment:
conda info --envs
conda list -n py312

# Verify tools:
conda run -n py312 python --version
conda run -n py312 pytest --version
```

#### Test Suite Status (Current State)

##### **Overall Test Metrics**
- **Total Tests**: 415 tests across 34 test files
- **Production Core**: 38 tests (100% pass rate) ✅
- **Quick Validation**: 9 tests (100% pass rate) ✅
- **Economics Module**: 23 tests (100% pass rate) ✅
- **Trajectory Module**: ~130 tests (77% pass rate) ⚠️
- **Optimization Module**: ~49 tests (69% pass rate) ⚠️

##### **Test Status by Module**
| Module | Tests | Status | Pass Rate | Notes |
|--------|-------|--------|-----------|-------|
| **Production Core** | 38 | ✅ Ready | 100% | CI/CD ready, commit requirement |
| **Environment** | 9 | ✅ Ready | 100% | PyKEP/PyGMO validation |
| **Economics** | 64 | ✅ Ready | 100% | Financial models, ISRU, sensitivity |
| **Configuration** | ~20 | ✅ Ready | 95% | Config management |
| **Trajectory** | ~130 | ⚠️ Partial | 77% | Task 3 incompleteness, NaN issues |
| **Optimization** | ~49 | ✅ Ready | 93% | PyGMO integration complete, no mocking |
| **Visualization** | ~37 | ⚠️ Partial | 62% | Minor dashboard issues |
| **Integration** | ~38 | ⚠️ Pending | Unknown | Task 7 dependencies |

##### **Known Test Issues**
1. **Trajectory Module Failures** (~30 tests):
   - Lambert solver NaN velocities in edge cases
   - Missing attributes in LunarTrajectory class
   - Incomplete Task 3.2 & 3.3 implementations

2. **Optimization Module** (FIXED ✅):
   - Previously had ~15 test failures
   - Now 93% pass rate (28/30 tests passing)
   - All tests use real implementations (NO MOCKING)
   - Fast execution with minimal parameters

3. **Expected Behavior**:
   - Production tests must always pass (commit requirement)
   - Trajectory/optimization test failures are tracked issues
   - Full test suite passes after Task 3 completion

##### **Testing Workflow**
```bash
# Daily development (required)
make test                    # Must pass 100% before commits

# Module-specific testing
make test-economics         # Financial models (should pass 100%)
make test-trajectory        # Orbital mechanics (known failures)
make test-quick             # Environment validation (must pass)

# Full validation (optional)
make test-all               # Complete suite (some failures expected)
```

## Task Tracking & Development Status

### Task Management System

The project uses a structured task tracking system with multiple components:

- **Primary Source**: `/tasks/tasks.json` - Central JSON file containing all task definitions and status
- **CLI Tool**: `node scripts/dev.js` - Comprehensive task management command-line interface
- **Task Files**: `/tasks/task_XXX.txt` - Detailed individual task descriptions
- **Documentation**: `/docs/task_X_documentation.md` - Extended documentation for specific tasks

### Current Development Status

| Task ID | Title | Status | Test Coverage | Implementation Notes |
|---------|-------|--------|---------------|----------------------|
| **1** | Setup Project Repository | ✅ Done | 100% | Environment fully configured |
| **2** | Mission Configuration Module | ✅ Done | 95% | Complete with validation |
| **3** | Trajectory Generation Module | ✅ Done | 77% | **All subtasks complete** - PyKEP + Lambert + N-body |
| **4** | Global Optimization Module | ✅ Done | 93% | PyGMO NSGA-II complete, real optimization tests |
| **5** | Basic Economic Analysis Module | ✅ Done | 100% | ROI, NPV, IRR, ISRU analysis complete |
| **6** | Basic Visualization Module | ⏱️ Pending | 62% | Dashboard framework exists |
| **7** | MVP Integration | ⏱️ Ready | Unknown | **READY TO START** - all dependencies complete |
| **8** | Local Differentiable Optimization | ⏱️ Deferred | 0% | JAX/Diffrax future enhancement |
| **9** | Enhanced Economic Analysis | ⏱️ Deferred | N/A | Advanced features planned |
| **10** | Extensibility Interface | ⏱️ Deferred | N/A | Plugin system planned |

**Status Legend**:
- ✅ **Done**: Fully implemented and tested
- ✅ **Functional***: Implementation complete but marked pending in tasks.json
- ⏱️ **Partial**: Some subtasks complete
- ⏱️ **Pending**: Not yet started
- ⏱️ **Deferred**: Planned for future phases

### Task Management Commands

```bash
# View all tasks with current status
node scripts/dev.js list --with-subtasks

# Check specific task details
node scripts/dev.js show --id=3

# Update task status
node scripts/dev.js set-status --id=4 --status=done

# Find next actionable task
node scripts/dev.js next

# Validate task dependencies
node scripts/dev.js validate-dependencies

# Generate subtasks with research
node scripts/dev.js research --task-id=3
```

### Task Development Workflow

1. **Start Task**: Review requirements and dependencies
   ```bash
   node scripts/dev.js show --id=X
   node scripts/dev.js validate-dependencies --task-id=X
   ```

2. **Implementation**: Follow task strategy and test requirements
   - Check existing code first (NO MOCKING RULE)
   - Implement with proper testing
   - Ensure PyKEP units consistency

3. **Testing**: Validate implementation
   ```bash
   conda activate py312
   make test                  # Core tests must pass
   make test-<module>         # Module-specific tests
   ```

4. **Update Status**: Mark task/subtask complete
   ```bash
   node scripts/dev.js set-status --id=X --status=done
   # For subtasks:
   node scripts/dev.js set-status --parent-id=3 --subtask-index=0 --status=done
   ```

5. **Commit**: Reference task in commit message
   ```bash
   git commit -m "Complete Task X: Brief description
   
   - Implementation details
   - Test coverage: XX%
   - Refs: Task #X in tasks.json"
   ```

### Task Dependencies & Critical Path

**Critical Path** (must be completed in order):
1. Task 3 (Trajectory Generation) → Blocks Task 7 (MVP Integration)
2. Tasks 3, 4, 5 → All needed for Task 7
3. Task 7 → Required before Tasks 8-10

**Current Status - NO BLOCKERS**:
- ✅ **Task 3**: COMPLETE - All trajectory generation implemented
- ✅ **Task 4**: COMPLETE - PyGMO optimization fully functional  
- ✅ **Task 5**: COMPLETE - Economic analysis 100% tested
- ⏱️ **Task 6**: Pending - Visualization module needs completion
- 🚀 **Task 7**: READY TO START - MVP integration can begin immediately

### Task Status Reconciliation

**Note on Status Discrepancy**: 
- Tasks 4 & 5 are marked ✅ in architecture comments due to functional implementation
- tasks.json shows "pending" as they await formal validation and Task 3 integration
- Use `node scripts/dev.js` to update official status when validated

### Integration with Development Workflow

1. **Pre-Development**: Always check task status first
2. **During Development**: Update subtask status as completed
3. **Post-Implementation**: Run full test suite and update task status
4. **Documentation**: Update this table when task status changes

### Next Steps Priority

Based on updated status and dependencies:
1. 🚀 **START Task 7** - MVP Integration (highest priority - all dependencies met)
2. **Complete Task 6** - Visualization module (can proceed in parallel)
3. **Consider Task 8** - JAX/Diffrax optimization (after Task 7)
4. **Plan Tasks 9-10** - Future enhancements (after MVP)