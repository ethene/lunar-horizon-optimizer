# 🌙 Lunar Horizon Optimizer - CLI Help Reference

Complete command-line interface reference for the Lunar Horizon Optimizer.

## Table of Contents
- [Quick Start](#quick-start)
- [Main Commands](#main-commands)
- [Scenario Management](#scenario-management)
- [Parameters & Options](#parameters--options)
- [Examples](#examples)
- [Environment Setup](#environment-setup)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Make executable and test
chmod +x lunar_opt.py
./lunar_opt.py --help

# Validate environment
./lunar_opt.py validate

# List available scenarios
./lunar_opt.py run list

# Run basic analysis
./lunar_opt.py run scenario 01_basic_transfer
```

## Main Commands

### `./lunar_opt.py --help`
Display comprehensive help with core capabilities and common usage patterns.

### `./lunar_opt.py validate`
Validate installation and environment, checking for:
- Required dependencies (numpy, scipy, plotly, pydantic, click, rich)
- Optional packages (pygmo, pykep, jax, kaleido)
- Python version compatibility (3.8+)

### `./lunar_opt.py sample`
Run a quick sample analysis to verify system functionality.

### `./lunar_opt.py run --help`
Show detailed help for scenario management commands.

## Scenario Management

### `./lunar_opt.py run list [OPTIONS]`
List all available analysis scenarios.

**Options:**
- `--detailed, -d` - Show detailed information including modules and expected results
- `--type, -t TYPE` - Filter by mission type (cargo, resource, trade, etc.)
- `--complexity, -c LEVEL` - Filter by complexity (beginner, intermediate, advanced)

**Examples:**
```bash
./lunar_opt.py run list                          # Basic listing
./lunar_opt.py run list --detailed               # Full details
./lunar_opt.py run list --type="cargo"           # Cargo missions only
./lunar_opt.py run list --complexity="beginner"  # Beginner scenarios
```

### `./lunar_opt.py run info SCENARIO_NAME`
Show detailed information about a specific scenario.

**Example:**
```bash
./lunar_opt.py run info 01_basic_transfer
```

### `./lunar_opt.py run scenario SCENARIO_NAME [OPTIONS]`
Execute a complete analysis pipeline for the specified scenario.

## Parameters & Options

### Core Options
- `--output, -o DIR` - Custom output directory (default: auto-generated with timestamp)
- `--verbose, -v` - Enable detailed logging and debug output
- `--help` - Show help for any command

### Optimization Parameters
- `--gens N` - Number of optimization generations (default: 25)
- `--population, -p N` - Population size (default: 40, must be multiple of 4)
- `--seed N` - Random seed for reproducible results
- `--parallel/--sequential` - Use parallel or sequential optimization (default: parallel)

### Analysis Control
- `--refine` - Enable JAX gradient-based local refinement
- `--risk` - Enable Monte Carlo risk analysis and sensitivity
- `--no-sensitivity` - Skip economic sensitivity analysis
- `--no-isru` - Skip ISRU (In-Situ Resource Utilization) analysis

### Output & Visualization
- `--export-pdf` - Export figures to PDF format (requires Kaleido)
- `--open-dashboard` - Open interactive dashboard in browser after completion
- `--gpu` - Enable GPU acceleration for JAX operations (if available)

### Parameter Constraints
- **Population size**: Must be multiple of 4 (NSGA-II requirement), minimum 8
- **Generations**: Higher values = better accuracy but longer runtime
- **GPU acceleration**: Requires JAX with CUDA/Metal support

## Examples

### Testing & Development
```bash
# Quick test run (fast execution)
./lunar_opt.py run scenario 01_basic_transfer \
  --gens 5 --population 8 --no-sensitivity --no-isru

# Environment validation
./lunar_opt.py validate

# Scenario exploration
./lunar_opt.py run list --detailed
./lunar_opt.py run info 06_isru_economics
```

### Standard Analysis
```bash
# Default parameters (recommended)
./lunar_opt.py run scenario 01_basic_transfer

# With custom output directory
./lunar_opt.py run scenario 02_launch_windows \
  --output my_analysis_results

# Reproducible results
./lunar_opt.py run scenario 03_propulsion_comparison \
  --seed 42
```

### Comprehensive Analysis
```bash
# High-fidelity analysis with all features
./lunar_opt.py run scenario 06_isru_economics \
  --gens 100 --population 80 --risk --refine \
  --export-pdf --open-dashboard

# Advanced optimization with GPU
./lunar_opt.py run scenario 04_pareto_optimization \
  --gens 50 --population 60 --refine --gpu \
  --export-pdf

# Risk analysis focus
./lunar_opt.py run scenario 08_risk_analysis \
  --risk --refine --export-pdf
```

### Batch Processing
```bash
# Run multiple scenarios
for scenario in 01_basic_transfer 02_launch_windows 03_propulsion_comparison; do
  ./lunar_opt.py run scenario $scenario --gens 10 --population 12
done

# Compare different parameter sets
./lunar_opt.py run scenario 01_basic_transfer --gens 25 --output results_25gen
./lunar_opt.py run scenario 01_basic_transfer --gens 50 --output results_50gen
```

## Environment Setup

### Prerequisites
```bash
# Ensure conda environment is available
conda activate py312

# Verify dependencies
./lunar_opt.py validate
```

### Installation Check
```bash
# Core dependencies (required)
python -c "import numpy, scipy, plotly, pydantic, click, rich; print('✅ Core deps OK')"

# Optimization dependencies (required)
python -c "import pygmo, pykep; print('✅ Optimization deps OK')"

# Optional dependencies
python -c "import jax; print('✅ JAX available')" 2>/dev/null || echo "⚠️ JAX not available"
python -c "import kaleido; print('✅ PDF export available')" 2>/dev/null || echo "⚠️ PDF export not available"
```

## Available Scenarios

| ID | Name | Type | Complexity | Runtime | Description |
|----|------|------|------------|---------|-------------|
| `01_basic_transfer` | Apollo-class Cargo | Cargo Delivery | Beginner | ~1-2 min | Basic Earth-Moon transfer with delta-v analysis |
| `02_launch_windows` | Artemis Crew Transport | General Mission | Intermediate | ~30-60s | Launch window optimization for crew missions |
| `03_propulsion_comparison` | Chemical vs Electric | General Mission | Intermediate | ~1-2 min | Propulsion technology trade study |
| `04_pareto_optimization` | Gateway Resupply | General Mission | Advanced | ~2-5 min | Multi-objective Pareto front analysis |
| `05_constellation_optimization` | Multi-Satellite | Multi-Mission | Intermediate | ~1-2 min | Constellation deployment optimization |
| `06_isru_economics` | Water Mining | Resource Extraction | Beginner | ~30-60s | ISRU economic analysis and ROI |
| `07_environmental_economics` | Sustainable Operations | General Mission | Intermediate | ~30-60s | Carbon impact and learning curves |
| `08_risk_analysis` | High-Risk Mining | Resource Extraction | Advanced | ~2-5 min | Monte Carlo risk and uncertainty |
| `09_complete_mission` | Lunar Base | General Mission | Advanced | ~2-5 min | End-to-end mission workflow |
| `10_multi_mission_campaign` | Multi-Mission Campaign | General Mission | Advanced | ~2-5 min | Campaign-level analysis |

## Output Structure

Every analysis generates organized outputs:
```
results/TIMESTAMP_scenario_name/
├── dashboard.html              # 🌐 Interactive analysis dashboard
├── reports/
│   └── summary_report.txt     # 📄 Executive summary
├── data/
│   ├── analysis_results.json  # 📊 Complete structured results
│   ├── scenario_config.json   # ⚙️ Configuration used
│   └── scenario_metadata.json # 📋 Scenario information
├── figures/                   # 📈 PDF exports (if enabled)
│   ├── trajectory.pdf
│   ├── pareto_front.pdf
│   └── cost_breakdown.pdf
└── OUTPUT_SUMMARY.txt         # 📋 File listing
```

## Troubleshooting

### Common Issues

**"Scenario not found"**
```bash
# Check available scenarios
./lunar_opt.py run list

# Verify exact scenario ID
./lunar_opt.py run info SCENARIO_ID
```

**"PyKEP/PyGMO import errors"**
```bash
# Ensure conda environment is activated
conda activate py312

# Reinstall if needed
conda install -c conda-forge pykep pygmo
```

**"Population size error"**
```bash
# Use population size that's multiple of 4
./lunar_opt.py run scenario 01_basic_transfer --population 8  # ✅ Valid
./lunar_opt.py run scenario 01_basic_transfer --population 10 # ❌ Invalid
```

**"Analysis hangs"**
- Normal behavior during trajectory generation (~30s) and optimization phases
- Use `--verbose` for detailed progress information
- Press Enter to see current status without interrupting

### Performance Tips

**Fast Testing**
```bash
./lunar_opt.py run scenario SCENARIO_NAME \
  --gens 5 --population 8 --no-sensitivity --no-isru
```

**Balanced Performance**
```bash
./lunar_opt.py run scenario SCENARIO_NAME \
  --gens 25 --population 40
```

**High Accuracy**
```bash
./lunar_opt.py run scenario SCENARIO_NAME \
  --gens 100 --population 80 --risk --refine
```

### Getting Help

```bash
# General help
./lunar_opt.py --help

# Command-specific help
./lunar_opt.py run --help
./lunar_opt.py run scenario --help

# Scenario information
./lunar_opt.py run info SCENARIO_NAME

# Verbose output for debugging
./lunar_opt.py --verbose run scenario SCENARIO_NAME
```

## Documentation References

- **[User Guide](guides/NEW_CLI_USER_GUIDE.md)** - Comprehensive usage guide
- **[CLI Overview](../CLI_README.md)** - Quick start and feature overview
- **[Scenarios](USE_CASES.md)** - Detailed scenario descriptions
- **[Technical Docs](technical/)** - Implementation details
- **[API Reference](api_reference.md)** - Programmatic interface

---

🌙 **Ready to optimize lunar missions?** Start with `./lunar_opt.py run list` to explore what's possible!