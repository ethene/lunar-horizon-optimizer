# Lunar Horizon Optimizer - Modern CLI User Guide

The new Lunar Horizon Optimizer CLI provides a streamlined, user-friendly interface for running lunar mission analysis scenarios with rich progress tracking and automated report generation.

## 🚀 Quick Start

### Installation & Setup

1. **Ensure Python 3.8+ is installed**
2. **Install dependencies:**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # For conda users (recommended for PyKEP/PyGMO):
   conda activate py312
   conda install -c conda-forge pykep pygmo
   ```

3. **Verify installation:**
   ```bash
   python lunar_opt.py validate
   ```

### Basic Usage

```bash
# List available scenarios
python lunar_opt.py run list

# Run a basic scenario
python lunar_opt.py run scenario 01_basic_transfer

# Run with custom parameters
python lunar_opt.py run scenario 06_isru_economics --risk --export-pdf
```

## 📋 Command Reference

### Main Commands

#### `lunar-opt run list`
List all available analysis scenarios.

**Options:**
- `--detailed, -d`: Show detailed scenario information
- `--type, -t TYPE`: Filter by mission type
- `--complexity, -c LEVEL`: Filter by complexity level

**Examples:**
```bash
# Basic listing
lunar-opt run list

# Detailed view with complexity filter
lunar-opt run list --detailed --complexity intermediate

# Filter by mission type
lunar-opt run list --type "cargo delivery"
```

#### `lunar-opt run scenario SCENARIO_NAME`
Run a specific lunar mission analysis scenario.

**Arguments:**
- `SCENARIO_NAME`: ID of scenario (e.g., `01_basic_transfer`)

**Core Options:**
- `--output, -o DIR`: Output directory (auto-generated if not specified)
- `--gens N`: Override number of optimization generations
- `--population, -p N`: Override population size
- `--seed N`: Random seed for reproducible results

**Analysis Options:**
- `--refine`: Enable JAX gradient-based refinement
- `--risk`: Enable Monte Carlo risk analysis
- `--no-sensitivity`: Skip sensitivity analysis
- `--no-isru`: Skip ISRU analysis

**Output Options:**
- `--export-pdf`: Export figures to PDF format
- `--open-dashboard`: Open dashboard in browser after completion

**Performance Options:**
- `--parallel/--sequential`: Use parallel or sequential optimization
- `--gpu`: Enable GPU acceleration (if available)

**Examples:**
```bash
# Basic scenario run
lunar-opt run scenario 01_basic_transfer

# Advanced run with all options
lunar-opt run scenario 06_isru_economics \
  --gens 50 \
  --population 60 \
  --risk \
  --refine \
  --export-pdf \
  --open-dashboard

# Quick test run
lunar-opt run scenario 04_pareto_optimization \
  --gens 10 \
  --population 20 \
  --no-sensitivity
```

#### `lunar-opt run info SCENARIO_NAME`
Show detailed information about a specific scenario.

**Example:**
```bash
lunar-opt run info 01_basic_transfer
```

### Legacy Commands (Backward Compatibility)

#### `lunar-opt analyze config CONFIG_FILE`
Run analysis using a configuration file (legacy mode).

#### `lunar-opt validate`
Validate installation and environment.

#### `lunar-opt sample`
Run a quick sample analysis.

## 🎯 Available Scenarios

### **01_basic_transfer** - Basic Earth-Moon Transfer
- **Type**: Cargo Delivery
- **Complexity**: Beginner  
- **Runtime**: ~30 seconds
- **Description**: Apollo-class lunar cargo delivery mission demonstrating optimal transfer trajectory calculation

### **06_isru_economics** - ISRU Economics Business Case
- **Type**: Resource Extraction
- **Complexity**: Intermediate
- **Runtime**: ~45 seconds  
- **Description**: Economic analysis of lunar water mining demonstrating ROI, NPV, and payback period

### **04_pareto_optimization** - Multi-Objective Trade Study
- **Type**: Trade Study
- **Complexity**: Intermediate
- **Runtime**: ~1-2 minutes
- **Description**: Pareto-optimal solutions balancing multiple mission objectives

### **08_risk_analysis** - Monte Carlo Risk Analysis
- **Type**: Risk Analysis
- **Complexity**: Advanced
- **Runtime**: ~2-3 minutes
- **Description**: Comprehensive uncertainty analysis using Monte Carlo methods

## 📊 Understanding Outputs

### Output Directory Structure
```
results/TIMESTAMP_scenario_name/
├── dashboard.html              # Interactive analysis dashboard
├── reports/
│   └── summary_report.txt     # Text summary of results
├── data/
│   ├── analysis_results.json  # Complete structured results
│   ├── scenario_config.json   # Original configuration
│   └── scenario_metadata.json # Scenario metadata
├── figures/                   # PDF exports (if enabled)
│   ├── trajectory.pdf
│   ├── pareto_front.pdf
│   └── cost_breakdown.pdf
└── OUTPUT_SUMMARY.txt         # Summary of all generated files
```

### Key Result Files

#### **dashboard.html**
Interactive HTML dashboard with:
- 3D trajectory visualizations
- Pareto front plots
- Economic analysis charts
- Parameter sensitivity plots

#### **summary_report.txt**
Human-readable summary containing:
- Best solution metrics (Δv, cost, transfer time)
- Economic analysis (NPV, ROI, IRR)
- Analysis performance statistics
- Modules and phases executed

#### **analysis_results.json**
Complete structured results in JSON format for programmatic access.

## ⚡ Performance Optimization

### Quick Analysis (Development/Testing)
```bash
# Minimal parameters for fast testing
lunar-opt run scenario 01_basic_transfer \
  --gens 5 \
  --population 10 \
  --no-sensitivity \
  --no-isru
```

### Production Analysis (High Accuracy)
```bash
# High-fidelity analysis with all features
lunar-opt run scenario 06_isru_economics \
  --gens 100 \
  --population 80 \
  --risk \
  --refine \
  --parallel \
  --gpu
```

### Parallel Processing
- `--parallel`: Use PyGMO's built-in parallelization
- `--gpu`: Enable JAX GPU acceleration for refinement
- Multiple scenarios can be run simultaneously in different terminals

## 🛠️ Advanced Usage

### Custom Configuration Override
```bash
# Override key parameters while keeping scenario structure
lunar-opt run scenario 01_basic_transfer \
  --gens 25 \
  --population 40 \
  --seed 12345
```

### Reproducible Results
```bash
# Use specific seed for reproducible optimization
lunar-opt run scenario 04_pareto_optimization --seed 42
```

### Export for External Analysis
```bash
# Generate all outputs including PDFs
lunar-opt run scenario 06_isru_economics \
  --export-pdf \
  --open-dashboard
```

## 🐛 Troubleshooting

### Common Issues

#### **"Scenario not found"**
```bash
# Check available scenarios
lunar-opt run list

# Verify scenario ID format (use underscores, not spaces)
lunar-opt run scenario 01_basic_transfer  # ✅ Correct
lunar-opt run scenario "basic transfer"   # ❌ Incorrect
```

#### **"PyKEP/PyGMO import errors"**
```bash
# Ensure conda environment is activated
conda activate py312

# Reinstall if needed
conda install -c conda-forge pykep pygmo
```

#### **"Analysis hangs at 95%"**
- This is usually normal - final phases can take time
- Use `--verbose` for detailed progress information
- Press Enter to see current status without interrupting

#### **"Out of memory errors"**
```bash
# Reduce computational load
lunar-opt run scenario SCENARIO_NAME \
  --gens 20 \
  --population 30 \
  --sequential
```

### Getting Help

```bash
# General help
lunar-opt --help

# Command-specific help
lunar-opt run --help
lunar-opt run scenario --help

# Scenario information
lunar-opt run info SCENARIO_NAME

# Verbose output for debugging
lunar-opt --verbose run scenario SCENARIO_NAME
```

### Environment Validation
```bash
# Check installation
lunar-opt validate

# Quick functionality test
lunar-opt sample
```

## 🔗 Integration with External Tools

### Jupyter Notebooks
```python
# Import results for further analysis
import json
with open('results/TIMESTAMP_scenario/data/analysis_results.json') as f:
    results = json.load(f)
```

### Custom Scripts
```python
# Use CLI programmatically
from src.cli.main import cli
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(cli, ['run', 'scenario', '01_basic_transfer'])
```

## 📈 Best Practices

### Development Workflow
1. **Start with quick tests**: Use minimal parameters during development
2. **Validate with intermediate runs**: Use default parameters for validation
3. **Production analysis**: Use high-fidelity parameters for final results

### Result Management
1. **Use descriptive output directories**: `--output descriptive_name`
2. **Export PDFs for presentations**: `--export-pdf`
3. **Clean up old results**: CLI automatically keeps 10 most recent

### Performance Tips
1. **Use parallel processing**: `--parallel` (default)
2. **Enable GPU when available**: `--gpu` for JAX refinement
3. **Adjust parameters based on accuracy needs**: Higher gens/population = better accuracy but slower

## 🎓 Learning Path

### Beginner
1. Run `lunar-opt validate` to check setup
2. Try `lunar-opt sample` for quick test
3. Run `lunar-opt run scenario 01_basic_transfer`
4. Explore the generated dashboard

### Intermediate  
1. Try different scenarios with `lunar-opt run list`
2. Use `--risk` and `--refine` options
3. Export results with `--export-pdf`
4. Compare multiple scenario results

### Advanced
1. Create custom scenarios by modifying JSON files
2. Use programmatic access to results
3. Integrate with external optimization tools
4. Develop custom visualization workflows

---

For more detailed information, see:
- [Scenario Guide](SCENARIO_GUIDE.md) - Detailed scenario descriptions
- [Technical Documentation](../technical/) - Implementation details
- [API Reference](../api_reference.md) - Programmatic interface