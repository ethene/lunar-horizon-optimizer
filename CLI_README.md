# 🌙 Lunar Horizon Optimizer - Modern CLI

A comprehensive, user-friendly command-line interface for lunar mission analysis providing scenario-based workflows with rich progress tracking and automated report generation.

## ✨ Key Features

- **🎯 Scenario-Based Workflows**: Predefined mission scenarios with one-command execution
- **📊 Rich Progress Tracking**: Real-time optimization monitoring with live solution updates
- **📈 Automated Reporting**: Interactive HTML dashboards and PDF exports
- **🔧 Modular Architecture**: Clean separation of concerns with extensible components
- **🛡️ Robust Error Handling**: User-friendly error messages with helpful suggestions
- **⚡ Performance Options**: Parallel processing, GPU acceleration, and configurable parameters

## 🚀 Quick Start

### 1. Installation
```bash
# Install CLI dependencies
pip install click rich pydantic

# For conda users (recommended):
conda activate py312
pip install click rich pydantic
```

### 2. Verify Installation
```bash
python lunar_opt.py validate
```

### 3. List Available Scenarios
```bash
python lunar_opt.py run list
```

### 4. Run Your First Analysis
```bash
python lunar_opt.py run scenario 01_basic_transfer
```

## 📋 Command Structure

```
lunar_opt.py
├── run                    # Modern scenario-based interface
│   ├── list              # List available scenarios
│   ├── scenario NAME     # Run specific scenario
│   └── info NAME         # Show scenario details
├── analyze               # Legacy commands (backward compatibility)
│   └── config FILE       # Analyze using config file
├── validate              # Environment validation
└── sample                # Quick sample analysis
```

## 🎯 Available Scenarios

| Scenario ID | Name | Type | Complexity | Runtime |
|-------------|------|------|------------|---------|
| `01_basic_transfer` | Apollo-class Cargo | Cargo Delivery | Beginner | ~30s |
| `06_isru_economics` | Water Mining Business Case | Resource Extraction | Intermediate | ~45s |
| `04_pareto_optimization` | Multi-Objective Trade Study | Trade Study | Intermediate | ~1-2min |
| `08_risk_analysis` | Monte Carlo Risk Analysis | Risk Analysis | Advanced | ~2-3min |

*[See complete list with `python lunar_opt.py run list --detailed`]*

## 🏃‍♂️ Common Usage Patterns

### Quick Testing
```bash
# Fast analysis for development/testing
python lunar_opt.py run scenario 01_basic_transfer \
  --gens 5 --population 10 --no-sensitivity
```

### Production Analysis
```bash
# High-fidelity analysis with all features
python lunar_opt.py run scenario 06_isru_economics \
  --gens 50 --risk --refine --export-pdf --open-dashboard
```

### Comparative Studies
```bash
# Run multiple scenarios for comparison
python lunar_opt.py run scenario 01_basic_transfer --output basic_results
python lunar_opt.py run scenario 03_propulsion_comparison --output propulsion_results
```

## 📊 Output Structure

Every analysis generates organized outputs:

```
results/TIMESTAMP_scenario_name/
├── dashboard.html              # 🌐 Interactive dashboard
├── reports/
│   └── summary_report.txt     # 📄 Executive summary
├── data/
│   ├── analysis_results.json  # 📊 Complete results data
│   └── scenario_config.json   # ⚙️ Configuration used
├── figures/                   # 📈 PDF exports (optional)
│   ├── trajectory.pdf
│   ├── pareto_front.pdf
│   └── cost_breakdown.pdf
└── OUTPUT_SUMMARY.txt         # 📋 File listing
```

## 🔧 Advanced Options

### Parameter Override
```bash
# Override optimization parameters
python lunar_opt.py run scenario 04_pareto_optimization \
  --gens 30 \
  --population 50 \
  --seed 42
```

### Analysis Customization
```bash
# Enable/disable analysis phases
python lunar_opt.py run scenario 06_isru_economics \
  --risk \           # Enable Monte Carlo risk analysis
  --refine \         # Enable JAX gradient refinement
  --no-sensitivity \ # Skip sensitivity analysis
  --no-isru          # Skip ISRU analysis
```

### Performance Tuning
```bash
# Performance options
python lunar_opt.py run scenario 09_complete_mission \
  --parallel \       # Use parallel optimization (default)
  --gpu \           # Enable GPU acceleration
  --export-pdf      # Export figures to PDF
```

## 📈 Progress Monitoring

The CLI provides rich, real-time progress tracking:

```
🌙 Lunar Horizon Optimizer - Apollo-class Lunar Cargo Mission
┌──────────────────────────────────────────────────────────┐
│ ⠋ 🚀 Apollo-class Lunar Cargo Mission ████████░░ 80% │
│ ⠋   📊 Global Optimization - Generation 15/25          │
└──────────────────────────────────────────────────────────┘

🏆 Top Solutions
┌──────┬────────────┬─────────────┬─────────────┬──────────────┐
│ Rank │ Generation │ Δv (m/s)    │ Cost ($M)   │ Time (days)  │
├──────┼────────────┼─────────────┼─────────────┼──────────────┤
│ 1    │ 14         │ 22,456      │ 3,245.2     │ 4.5          │
│ 2    │ 13         │ 22,891      │ 3,156.7     │ 4.7          │
│ 3    │ 15         │ 23,123      │ 3,089.4     │ 4.8          │
└──────┴────────────┴─────────────┴─────────────┴──────────────┘

Elapsed: 0:01:23 | Press Ctrl+C to stop
```

## 🛠️ Development Integration

### Programmatic Usage
```python
from src.cli.main import cli
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(cli, ['run', 'scenario', '01_basic_transfer'])
```

### Custom Scenarios
1. Create new JSON file in `scenarios/` directory
2. Follow existing scenario format
3. The CLI automatically discovers new scenarios

### Result Processing
```python
import json

# Load results for post-processing
with open('results/TIMESTAMP_scenario/data/analysis_results.json') as f:
    results = json.load(f)
```

## 🐛 Troubleshooting

### Environment Issues
```bash
# Check dependencies
python lunar_opt.py validate

# Test basic functionality
python lunar_opt.py sample
```

### Common Errors

**"Scenario not found"**
```bash
# List available scenarios
python lunar_opt.py run list

# Check exact scenario ID
python lunar_opt.py run info SCENARIO_ID
```

**"Import errors"**
```bash
# Install missing dependencies
pip install click rich pydantic

# For conda users
conda activate py312
```

**"Analysis hangs"**
- Normal behavior during optimization phases
- Use `--verbose` for detailed progress
- Press Enter to see current status

### Getting Help
```bash
# General help
python lunar_opt.py --help

# Command-specific help
python lunar_opt.py run --help
python lunar_opt.py run scenario --help

# Verbose output for debugging
python lunar_opt.py --verbose run scenario SCENARIO_NAME
```

## 🎯 Comparison: Old vs New CLI

| Feature | Old CLI | New CLI |
|---------|---------|---------|
| **Interface** | argparse | Click (modern) |
| **Scenarios** | Manual config files | Auto-discovered scenarios |
| **Progress** | Basic text output | Rich progress bars + live updates |
| **Results** | Basic file output | Organized directories + dashboards |
| **Errors** | Python tracebacks | User-friendly messages |
| **Help** | Standard help text | Rich help with examples |

### Migration Path
```bash
# Old way
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output results

# New way (equivalent)
python lunar_opt.py run scenario 01_basic_transfer

# Legacy compatibility (still works)
python lunar_opt.py analyze config scenarios/01_basic_transfer.json
```

## 📚 Documentation

### Core Documentation
- **[CLI Help Reference](docs/CLI_HELP_REFERENCE.md)** - Complete command reference and examples
- **[User Guide](docs/guides/NEW_CLI_USER_GUIDE.md)** - Comprehensive usage guide with tutorials
- **[Scenario Catalog](docs/USE_CASES.md)** - Detailed scenario descriptions and use cases

### Technical Documentation  
- **[Technical Implementation](docs/technical/)** - Architecture and implementation details
- **[API Reference](docs/api_reference.md)** - Programmatic interface documentation
- **[Testing Guide](tests/TEST_SUITE_DOCUMENTATION.md)** - Test suite documentation

### Quick References
- **[Project Status](docs/PROJECT_STATUS.md)** - Current development status and roadmap
- **[Task Documentation](docs/INDEX.md)** - Task tracking and development progress

## 🏗️ Architecture

The new CLI is built with modular components:

```
src/cli/
├── main.py              # Click-based CLI interface
├── scenario_manager.py  # Scenario discovery & validation
├── progress_tracker.py  # Rich progress tracking
├── output_manager.py    # Result organization & export
├── error_handling.py    # User-friendly error handling
└── __init__.py         # Package exports
```

### Design Principles
- **User-Friendly**: Clear commands, helpful errors, rich output
- **Modular**: Separated concerns, reusable components
- **Extensible**: Easy to add new scenarios and features
- **Backward Compatible**: Legacy CLI still works
- **Production Ready**: Robust error handling, comprehensive testing

## 🤝 Contributing

### Adding New Scenarios
1. Create JSON file in `scenarios/` following existing format
2. CLI automatically discovers and validates new scenarios
3. Test with `python lunar_opt.py run info NEW_SCENARIO_ID`

### Extending CLI Features
1. Components are in `src/cli/` with clear interfaces
2. Follow existing patterns for consistency
3. Add comprehensive error handling
4. Update documentation

---

**🌙 Ready to explore lunar missions?** Start with `python lunar_opt.py run list` to see what's possible!