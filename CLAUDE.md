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

tests/               # Comprehensive test suite (83% success rate)
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
- **Python Environment**: conda py312 environment (required)
- **PyKEP Installation**: Must use conda-forge channel for PyKEP 2.6
- **Environment Setup**: 
  ```bash
  conda create -n py312 python=3.12 -y
  conda activate py312
  conda install -c conda-forge pykep pygmo astropy spiceypy -y
  pip install -r requirements.txt
  ```

## Development Standards

### Code Quality Requirements

#### 1. **Testing is Mandatory**
- **Coverage**: Maintain >90% test coverage for all new code
- **Test Types**: Unit tests, integration tests, and validation tests
- **Test Location**: All tests in `tests/` directory with clear naming
- **Before Committing**: Always run `pytest tests/` and ensure all tests pass

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

### File Organization Rules

#### Source Code (`src/`)
- **Modular Design**: Keep modules focused and under 300 lines
- **Clear Imports**: Explicit imports, avoid wildcard imports
- **Constants**: Use constants from `src/constants/` for all physical values
- **Unit Conversions**: Use utilities from `src/utils/unit_conversions.py`

#### Configuration (`src/config/`)
- **Pydantic Models**: All configuration uses Pydantic for validation
- **Backward Compatibility**: Maintain compatibility while deprecating old patterns
- **Validation**: Comprehensive validation with clear error messages
- **Documentation**: Document all configuration options

#### Tests (`tests/`)
- **Mirror Structure**: Test structure mirrors source structure
- **Naming Convention**: `test_<module_name>.py`
- **Test Categories**: Organize by importance (Critical, High, Medium)
- **Data**: Use realistic test data and edge cases

### Development Workflow

#### 1. **Before Starting Work**
```bash
# Always verify current state
pytest tests/                    # Ensure tests pass
python scripts/verify_dependencies.py  # Check dependencies
git status                      # Check repository state
```

#### 2. **Making Changes**
- **Small Commits**: Make focused, atomic commits
- **Test-Driven**: Write tests before implementing features
- **Documentation**: Update relevant documentation
- **Validation**: Run linting and type checking

#### 3. **Before Committing**
```bash
# Required checks
flake8 src/ tests/              # Linting
mypy src/                       # Type checking  
black src/ tests/               # Code formatting
pytest tests/ -v                # Full test suite
```

## Current Development Status (July 2025)

### ✅ COMPLETED COMPONENTS
- **Task 1**: Project setup and environment configuration ✅
- **Task 2**: Mission configuration module with Pydantic validation ✅
- **Task 3**: Enhanced Trajectory Generation ✅
  - PyKEP integration ✅
  - Earth-Moon trajectory functions ✅ 
  - N-body dynamics and integration ✅
- **Task 4**: Global Optimization Module ✅
  - PyGMO/NSGA-II implementation ✅
  - Pareto front analysis ✅
  - Cost integration ✅
- **Task 5**: Basic Economic Analysis ✅
  - Financial modeling (NPV, IRR, ROI) ✅
  - Cost estimation and ISRU analysis ✅
  - Testing validated (29/38 tests passing) ✅

### 🔄 CURRENT PRIORITIES
1. **Task 6**: Visualization Module (Plotly 3D) - **NEXT**
2. **Task 7**: MVP Integration
3. **Optional**: Tasks 3-4 test enhancement

### 📊 PROJECT STATUS
- **Version**: 0.9.0 (Near-MVP)
- **Test Success Rate**: 83% (44/53 tests passing)
- **Code Quality**: All import issues resolved, absolute imports implemented
- **Documentation**: Comprehensive and current

## Code Patterns & Best Practices

### Unit Handling
```python
# CORRECT: Use PyKEP native units
from src.utils.unit_conversions import km_to_m, kmps_to_mps
position_m = km_to_m(position_km)
velocity_ms = kmps_to_mps(velocity_kmps)

# WRONG: Don't use magic numbers
position_m = position_km * 1000  # Avoid this
```

### Configuration Usage
```python
# CORRECT: Use Pydantic models
from src.config.mission_config import MissionConfig
config = MissionConfig(name="lunar_mission", ...)

# CORRECT: Validate inputs
@validator('altitude')
def validate_altitude(cls, v):
    if v < 200:
        raise ValueError("Altitude must be > 200 km")
    return v
```

### Error Handling
```python
# CORRECT: Comprehensive error handling
def calculate_trajectory(params):
    try:
        validate_parameters(params)
        result = perform_calculation(params)
        return result
    except ValidationError as e:
        logger.error(f"Parameter validation failed: {e}")
        raise
    except CalculationError as e:
        logger.error(f"Trajectory calculation failed: {e}")
        raise
```

### Testing Patterns
```python
# CORRECT: Comprehensive test structure
class TestTrajectoryCalculation:
    def test_valid_inputs(self):
        """Test with valid inputs."""
        pass
        
    def test_edge_cases(self):
        """Test boundary conditions.""" 
        pass
        
    def test_invalid_inputs(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            invalid_calculation()
```

## Refactoring Guidelines

### ✅ COMPLETED REFACTORING (July 2025)

#### Phase 1: Consolidation ✅
- ✅ Merged duplicate validation modules
- ✅ Removed duplicate propagation methods  
- ✅ Cleaned up deprecated configuration models

#### Phase 2: Extraction ✅
- ✅ Split large files (`trajectory_physics.py`, `config/manager.py`)
- ✅ Extracted complex methods from `lunar_transfer.py`
- ✅ Created focused, single-purpose modules
- ✅ **Major Achievement**: Resolved all import dependencies with absolute imports

### 🔄 OPTIONAL FUTURE REFACTORING

#### Phase 3: Architecture (Low Priority)
- Implement dependency injection patterns
- Add strategy pattern for propagation  
- Create factory pattern for trajectory generation

### Refactoring Success Criteria ✅
1. **Test First**: ✅ All tests maintained through refactoring (83% success rate)
2. **Small Steps**: ✅ Incremental changes with preserved functionality
3. **Document Changes**: ✅ All architectural changes documented
4. **Backward Compatibility**: ✅ API compatibility maintained
5. **Import Resolution**: ✅ All relative imports converted to absolute imports

## Integration Guidelines

### PyKEP Integration
```python
# CORRECT: Proper PyKEP usage
import pykep as pk
from src.constants import PhysicalConstants as PC

# Use PyKEP's gravitational parameters
mu = pk.MU_EARTH  # Consistent with library
planet = pk.planet.jpl_lp('earth')
```

### JAX/Diffrax Usage
```python
# CORRECT: JAX for differentiable optimization
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm

# Use JAX arrays for automatic differentiation
state = jnp.array([x, y, z, vx, vy, vz])
```

### Economic Analysis
```python
# CORRECT: Economic modeling structure
from src.config.costs import CostFactors
from src.config.isru import IsruCapabilities

def calculate_mission_roi(costs: CostFactors, isru: IsruCapabilities):
    """Calculate mission return on investment."""
    pass
```

## Debugging & Troubleshooting

### Common Issues
1. **Unit Mismatches**: Always verify units are consistent (PyKEP native)
2. **SPICE Kernel Errors**: Ensure `de430.bsp` is in `data/spice/`
3. **Test Failures**: Check imports and configuration after refactoring
4. **Dependency Issues**: Run `scripts/verify_dependencies.py`

### Debugging Tools
```python
# Use structured logging
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Trajectory state: {state}")

# Validate intermediate results
assert np.isfinite(position).all(), "Position contains invalid values"
```

## Performance Considerations

### Optimization Guidelines
1. **Vectorization**: Use NumPy/JAX vectorized operations
2. **Caching**: Cache expensive calculations (orbital propagation)
3. **Memory**: Minimize large array copies
4. **Profiling**: Profile before optimizing

### Memory Management
```python
# CORRECT: Efficient array handling
import jax.numpy as jnp

# Use JAX for automatic differentiation
trajectory = jnp.array(trajectory_data)

# WRONG: Don't create unnecessary copies
trajectory_copy = np.copy(trajectory)  # Avoid unless necessary
```

## Documentation Requirements

### Code Documentation
- **Docstrings**: NumPy-style docstrings for all public functions
- **Type Hints**: Complete type annotations
- **Examples**: Include usage examples in docstrings
- **Units**: Always specify units in docstrings

### Project Documentation
- **Update with Changes**: Keep docs current with implementation
- **Architecture Decisions**: Document major design choices
- **Migration Guides**: Provide guides for API changes
- **Examples**: Maintain working examples

## Version Control

### Commit Standards
```
feat: add lunar transfer optimization module
fix: resolve unit conversion error in trajectory calculation  
refactor: split trajectory_physics.py into focused modules
test: add comprehensive tests for orbital elements
docs: update trajectory module documentation
```

### Branch Strategy
- **main**: Production-ready code
- **feature/**: New feature development
- **refactor/**: Code restructuring
- **hotfix/**: Critical bug fixes

## Documentation Index & Memory Maintenance

### 📚 Critical Documentation Files (Update After Each Major Step)

#### **Project Status Documents** (Always Keep Current)
1. **`docs/development_status.md`** - Overall project progress and completion status
   - 🔄 Update after: Task completion, major feature implementation, testing milestones
   - Contains: Task completion status, version info, capability overview

2. **`docs/DOCUMENTATION_SUMMARY.md`** - Master documentation inventory  
   - 🔄 Update after: New documentation creation, major documentation changes
   - Contains: Complete documentation status, quality metrics, future needs

#### **Task-Specific Documentation** (Update When Modifying Related Tasks)
3. **`docs/task_3_documentation.md`** - Trajectory generation (Task 3)
   - 🔄 Update after: Trajectory module changes, algorithm improvements
   
4. **`docs/task_4_documentation.md`** - Global optimization (Task 4)
   - 🔄 Update after: Optimization algorithm changes, PyGMO integration updates

5. **`docs/task_5_documentation.md`** - Economic analysis (Task 5) ✅ COMPLETE
   - 🔄 Update after: Economic model changes, financial algorithm updates
   - Status: Complete & tested (29/38 tests passing)

6. **`docs/api_reference.md`** - Complete API documentation
   - 🔄 Update after: API changes, new module creation, interface modifications

7. **`docs/integration_guide.md`** - Cross-module integration patterns
   - 🔄 Update after: New module integration, workflow changes

#### **Testing Documentation** (Update After Testing Changes)
8. **`tests/README.md`** - Test suite organization and instructions
   - 🔄 Update after: Test infrastructure changes, new test suites

9. **`tests/TEST_STATUS_UPDATE.md`** - Latest testing achievements and status
   - 🔄 Update after: Major testing improvements, test fixes, coverage changes
   - Current: 83% success rate documented

10. **`tests/TESTING_AUDIT.md`** - Comprehensive testing analysis
    - 🔄 Update after: Major testing overhauls, infrastructure changes

### 🎯 Memory Maintenance Protocol

#### **After Each Development Session:**
1. **Update Development Status**: Reflect current task progress in `docs/development_status.md`
2. **Update Testing Status**: If tests change, update relevant testing documentation
3. **Version Tracking**: Update version numbers and completion dates
4. **Todo Management**: Keep todo list current with actual progress

#### **After Task Completion:**
1. **Complete Task Documentation**: Finalize task-specific documentation
2. **Update API Reference**: Add new APIs and examples  
3. **Update Integration Guide**: Document new integration patterns
4. **Update Architecture**: Reflect new modules in project structure

#### **After Major Milestones:**
1. **Comprehensive Status Update**: Update all status documents
2. **Documentation Summary**: Update master documentation inventory
3. **Testing Documentation**: Reflect testing achievements and infrastructure
4. **Project Health**: Update risk assessments and future planning

#### **Before Starting New Major Work:**
1. **Review Current Status**: Check `docs/development_status.md` for current state
2. **Check Dependencies**: Verify what's completed vs. what's needed
3. **Update Priorities**: Adjust task priorities based on current progress
4. **Set Success Criteria**: Define clear completion criteria for new work

### 📋 Quick Reference Commands

#### **Status Verification**
```bash
# Check current test status
python tests/run_working_tests.py

# Verify environment
python scripts/verify_dependencies.py

# Check documentation currency
ls -la docs/ | grep -E "(development_status|DOCUMENTATION_SUMMARY)"
```

#### **Documentation Updates**
```bash
# After task completion
vim docs/development_status.md          # Update task status
vim docs/task_X_documentation.md        # Update specific task docs
vim docs/api_reference.md               # Update API documentation

# After testing improvements  
vim tests/README.md                     # Update test instructions
vim tests/TEST_STATUS_UPDATE.md         # Document testing achievements
```

## Contact & Support

### Development Questions
- **Primary**: Review `docs/development_status.md` for current project state
- **Technical**: Check `docs/api_reference.md` for implementation details
- **Testing**: Review `tests/README.md` for test execution guidance
- **Integration**: Consult `docs/integration_guide.md` for module connections

### Code Issues
- **Testing**: `python tests/run_working_tests.py` (recommended test runner)
- **Environment**: `python scripts/verify_dependencies.py`
- **Linting**: `flake8 src/ tests/`
- **Status Check**: Review latest documentation in `docs/` directory

---

*This document serves as the authoritative guide for development practices in the Lunar Horizon Optimizer project. All contributors must follow these guidelines to ensure code quality, consistency, and project success.*