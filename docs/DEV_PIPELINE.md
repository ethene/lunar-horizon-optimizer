# Development Pipeline - Lunar Horizon Optimizer

## Overview

This development pipeline provides comprehensive code quality checks for the Lunar Horizon Optimizer project. It includes formatting, linting, complexity analysis, type checking, AI-based refactor suggestions, security scanning, and testing.

## Quick Start

### Using Makefile (Recommended)
```bash
# Run complete pipeline
make pipeline

# Run individual steps
make format        # Format code with Black
make lint          # Lint with Ruff
make complexity    # Check complexity with Radon/Xenon
make type-check    # Type check with MyPy
make refactor      # Get AI suggestions with Sourcery
make security      # Security scan with Bandit

# Testing
make test          # Run tests
make coverage      # Run tests with coverage

# Utilities
make install-dev   # Install development dependencies
make clean         # Clean temporary files
make help          # Show all available commands
```

### Using Shell Script
```bash
# Run complete pipeline
./dev-pipeline.sh pipeline

# Run individual steps
./dev-pipeline.sh format
./dev-pipeline.sh lint
./dev-pipeline.sh complexity
./dev-pipeline.sh type-check
./dev-pipeline.sh refactor
./dev-pipeline.sh security

# Testing
./dev-pipeline.sh test
./dev-pipeline.sh coverage

# Utilities
./dev-pipeline.sh install-dev
./dev-pipeline.sh clean
./dev-pipeline.sh help
```

## Pipeline Steps

### 1. Code Formatting (Black)
- **Tool**: [Black](https://black.readthedocs.io/)
- **Purpose**: Automatic code formatting
- **Configuration**: 88 character line length, Python 3.12 target
- **Files**: `src/` and `tests/` directories
- **Behavior**: Automatically formats code to consistent style

### 2. Linting (Ruff)
- **Tool**: [Ruff](https://docs.astral.sh/ruff/)
- **Purpose**: Fast linting with flake8 + pylint rules
- **Rules**: Comprehensive rule set including:
  - Code style (E, W)
  - Logic errors (F)
  - Security issues (S)
  - Code complexity (C)
  - Naming conventions (N)
  - Documentation (D)
  - Type annotations (ANN)
  - And many more
- **Speed**: ~10-100x faster than traditional linters

### 3. Complexity Analysis (Radon + Xenon)
- **Tools**: [Radon](https://radon.readthedocs.io/) + [Xenon](https://xenon.readthedocs.io/)
- **Purpose**: Code complexity and maintainability analysis
- **Metrics**:
  - Cyclomatic complexity
  - Maintainability index
  - Halstead metrics
- **Thresholds**: 
  - Max absolute complexity: B
  - Max module complexity: A
  - Max average complexity: A

### 4. Type Checking (MyPy)
- **Tool**: [MyPy](https://mypy.readthedocs.io/)
- **Purpose**: Static type analysis
- **Mode**: Strict mode with comprehensive checks
- **Coverage**: `src/` directory only
- **Features**: Error codes, pretty output

### 5. AI Refactor Suggestions (Sourcery)
- **Tool**: [Sourcery](https://sourcery.ai/)
- **Purpose**: AI-powered code quality suggestions
- **Behavior**: Provides refactoring recommendations
- **Note**: Suggestions are recommendations, not requirements

### 6. Security Scanning (Bandit)
- **Tool**: [Bandit](https://bandit.readthedocs.io/)
- **Purpose**: Security vulnerability detection
- **Scope**: `src/` directory
- **Exclusions**: B101 (assert usage), B601 (shell injection in controlled contexts)

## Testing

### Regular Testing
```bash
make test
# or
./dev-pipeline.sh test
```
- Uses conda py312 environment
- Runs the working test suite
- 44/53 tests passing (83% success rate)
- Zero failures

### Coverage Testing
```bash
make coverage
# or
./dev-pipeline.sh coverage
```
- Generates coverage reports
- HTML report in `htmlcov/`
- Minimum 80% coverage required
- Terminal output with missing lines

## Development Dependencies

### Required Tools
Install with: `make install-dev` or `pip install`:

```
black>=23.12.0      # Code formatting
ruff>=0.1.8         # Fast linting
radon>=6.0.1        # Complexity analysis
xenon>=0.9.0        # Complexity thresholds
mypy>=1.8.0         # Type checking
sourcery>=1.14.0    # AI refactor suggestions
bandit>=1.7.5       # Security scanning
pytest>=7.4.0       # Testing framework
pytest-cov>=4.1.0   # Coverage reporting
```

### Environment Requirements
- **Python**: 3.12+
- **Environment**: conda py312 (for testing)
- **OS**: Cross-platform (Linux, macOS, Windows)

## Configuration Files

### pyproject.toml
Central configuration for all tools:
- Black formatting settings
- Ruff linting rules and exclusions
- MyPy type checking configuration
- Pytest settings
- Coverage configuration
- Bandit security settings

### .pre-commit-config.yaml
Git pre-commit hooks configuration:
- Runs subset of pipeline on commit
- Automatic code formatting
- Basic linting and security checks
- Install with: `pre-commit install`

## Integration with IDEs

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true
}
```

### PyCharm
1. Install plugins: Black, Ruff, MyPy
2. Configure external tools for pipeline commands
3. Set up file watchers for automatic formatting

## Continuous Integration

### GitHub Actions
Add to `.github/workflows/quality.yml`:
```yaml
name: Code Quality
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -e .[dev]
      - run: make pipeline
      - run: make coverage
```

### Pre-commit.ci
Already configured in `.pre-commit-config.yaml`:
- Runs on every PR
- Auto-fixes formatting issues
- Weekly dependency updates

## Troubleshooting

### Common Issues

1. **Command not found**
   ```bash
   make install-dev
   # or manually install: pip install black ruff radon xenon mypy sourcery bandit
   ```

2. **MyPy errors with external libraries**
   - Configuration ignores missing imports for PyKEP, PyGMO, etc.
   - Add new libraries to `pyproject.toml` if needed

3. **Sourcery login required**
   ```bash
   sourcery login
   # Follow authentication flow
   ```

4. **Test environment issues**
   ```bash
   conda activate py312
   # Ensure conda environment is active
   ```

### Performance Tips

1. **Parallel execution**: Pipeline steps run sequentially for clarity, but can be parallelized
2. **Incremental checks**: Use `git diff` to check only changed files
3. **Caching**: MyPy and Ruff support caching for faster subsequent runs

## Pipeline Philosophy

### Goals
1. **Quality**: Comprehensive code quality checks
2. **Speed**: Fast feedback loop for developers
3. **Consistency**: Uniform code style and standards
4. **Security**: Early detection of vulnerabilities
5. **Maintainability**: Code complexity management

### Best Practices
1. **Fail Fast**: Pipeline stops on first failure
2. **Visible Output**: All tool output shown in terminal
3. **Configurable**: Centralized configuration in `pyproject.toml`
4. **Extensible**: Easy to add new tools or modify settings
5. **Documentation**: Clear explanations of each step

## Advanced Usage

### Custom Rules
Modify `pyproject.toml` to customize:
- Ruff rule selection
- MyPy strictness levels
- Complexity thresholds
- Coverage requirements

### Integration Testing
```bash
# Run specific test categories
pytest tests/ -m integration
pytest tests/ -m unit
pytest tests/ --cov=src --cov-report=html
```

### Performance Profiling
```bash
# Profile code complexity
radon cc src/ -a -nc -s
radon mi src/ -nc -s

# Profile test performance
pytest tests/ --durations=10
```

---

This development pipeline ensures the Lunar Horizon Optimizer maintains high code quality, security, and maintainability standards while providing fast feedback to developers.