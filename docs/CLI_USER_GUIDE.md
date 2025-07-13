# Lunar Horizon Optimizer - CLI User Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Command Reference](#command-reference)
5. [Configuration Files](#configuration-files)
6. [Real-World Examples](#real-world-examples)
7. [Output & Results](#output--results)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Overview

The Lunar Horizon Optimizer CLI provides comprehensive mission analysis capabilities for Earth-Moon trajectories, integrating:

- **PyKEP-based trajectory calculations** - Real orbital mechanics using NASA ephemeris
- **PyGMO multi-objective optimization** - NSGA-II algorithm for Pareto front analysis
- **Financial modeling** - NPV, IRR, ROI with Wright's law learning curves
- **ISRU economic analysis** - In-Situ Resource Utilization benefits
- **Interactive dashboards** - HTML and Plotly visualizations

### Key Features
- ✅ **Real calculations** - No mocked data, uses actual aerospace engineering models
- ✅ **Fast execution** - Complete analysis in <60 seconds
- ✅ **Production-ready** - 100% test coverage on core modules
- ✅ **Extensible** - Plugin architecture for custom analyses

## Installation & Setup

### Prerequisites
```bash
# REQUIRED: Conda py312 environment with PyKEP/PyGMO
conda activate py312

# Verify environment
python --version  # Should be 3.12.x
python -c "import pykep; print(f'PyKEP {pykep.__version__}')"
python -c "import pygmo; print(f'PyGMO {pygmo.__version__}')"
```

### Quick Install Check
```bash
# Validate installation
python src/cli.py validate

# Expected output:
# ✅ PyKEP: OK (2.6.x)
# ✅ PyGMO: OK (2.19.x)
# ✅ LunarHorizonOptimizer: OK
# ✅ Sample configuration: OK
# 🎉 Environment validation PASSED!
```

## Quick Start

### 1. Run Sample Analysis
```bash
# Quick demo with optimized settings
python src/cli.py sample

# Output:
# 🚀 Running Quick Sample Analysis...
# 🔄 Running quick optimization (20 pop, 10 gen)...
# ✅ Analysis complete!
# 📊 Results saved to: sample_analysis_YYYYMMDD_HHMMSS/
```

### 2. Analyze Pre-configured Scenario
```bash
# Basic lunar transfer mission
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output my_results

# Output includes:
# - analysis_results.json (complete numerical data)
# - dashboard.html (executive visualization)
# - summary.txt (key metrics)
```

### 3. Generate Configuration Template
```bash
# Create customizable mission config
python src/cli.py config --output my_mission.json

# Edit the generated file for your mission parameters
```

## Command Reference

### `analyze` - Run Mission Analysis

Performs comprehensive mission analysis with trajectory optimization, economic modeling, and visualization.

```bash
python src/cli.py analyze [OPTIONS]

Options:
  --config PATH           Mission configuration JSON file
  --output PATH           Output directory for results (default: timestamped)
  --mission-name TEXT     Override mission name from config
  --population-size INT   Optimization population size (must be multiple of 4)
  --generations INT       Number of optimization generations
  --no-sensitivity        Skip sensitivity analysis
  --no-isru              Skip ISRU economic analysis
  --verbose              Enable detailed logging
  --learning-rate FLOAT   Wright's law learning rate (0.7-0.95)
  --carbon-price FLOAT    Carbon price per ton CO2
```

#### Example: Complete Analysis
```bash
python src/cli.py analyze \
  --config scenarios/09_complete_mission.json \
  --output lunar_base_analysis \
  --population-size 100 \
  --generations 50 \
  --verbose
```

### `config` - Generate Configuration

Creates sample configuration files with all available parameters.

```bash
python src/cli.py config [OPTIONS]

Options:
  --output PATH    Output file path (default: sample_mission_config.json)
  --full          Include all optional parameters
```

#### Example Configuration Structure
```json
{
  "mission": {
    "name": "Artemis Lunar Cargo Mission",
    "description": "Deliver supplies to lunar Gateway",
    "transfer_time": 4.5
  },
  "spacecraft": {
    "dry_mass": 5000.0,
    "max_propellant_mass": 3000.0,
    "payload_mass": 1000.0,
    "specific_impulse": 450.0
  },
  "costs": {
    "launch_cost_per_kg": 10000.0,
    "operations_cost_per_day": 100000.0,
    "development_cost": 1000000000.0,
    "contingency_percentage": 20.0,
    "discount_rate": 0.08,
    "learning_rate": 0.85,
    "carbon_price_per_ton_co2": 50.0,
    "co2_emissions_per_kg_payload": 2.5
  },
  "orbit": {
    "semi_major_axis": 6778.0,
    "inclination": 28.5,
    "eccentricity": 0.0
  },
  "optimization": {
    "population_size": 52,
    "num_generations": 30,
    "seed": 42
  }
}
```

### `validate` - Environment Validation

Checks all dependencies and system configuration.

```bash
python src/cli.py validate

# Validates:
# - Python version and environment
# - PyKEP/PyGMO installation
# - Module imports
# - Configuration loading
# - Basic functionality
```

### `sample` - Quick Demo

Runs a fast demonstration analysis with reduced optimization parameters.

```bash
python src/cli.py sample

# Uses:
# - 20 population size
# - 10 generations
# - Default mission parameters
# - Full visualization output
```

## Configuration Files

### Mission Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `transfer_time` | Earth-Moon transfer duration | days | 3.0 - 10.0 |
| `dry_mass` | Spacecraft structure mass | kg | 1000 - 50000 |
| `max_propellant_mass` | Maximum fuel capacity | kg | 500 - 30000 |
| `payload_mass` | Cargo/payload mass | kg | 100 - 20000 |
| `specific_impulse` | Engine efficiency | seconds | 300 - 480 |

### Cost Factors

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `launch_cost_per_kg` | Launch vehicle cost | $/kg | 2000 - 20000 |
| `operations_cost_per_day` | Daily mission ops | $/day | 50000 - 1000000 |
| `development_cost` | Total development | $ | 1e8 - 1e10 |
| `contingency_percentage` | Risk margin | % | 10 - 50 |
| `discount_rate` | Financial discount | decimal | 0.04 - 0.12 |
| `learning_rate` | Wright's law factor | decimal | 0.70 - 0.95 |
| `carbon_price_per_ton_co2` | Environmental cost | $/ton | 0 - 200 |

### Orbit Parameters

| Parameter | Description | Units | Notes |
|-----------|-------------|-------|-------|
| `semi_major_axis` | Orbit size | km | Earth radius + altitude |
| `inclination` | Orbit angle | degrees | 0-180 |
| `eccentricity` | Orbit shape | decimal | 0 = circular, <1 = elliptical |

**Important**: 
- LEO altitude 400km → semi_major_axis = 6778 km (6378 + 400)
- GTO altitude 35786km → semi_major_axis = 42164 km

## Real-World Examples

### 1. Basic Apollo-Style Mission
```bash
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output apollo_analysis

# Characteristics:
# - 5000 kg dry mass
# - 450s Isp (chemical propulsion)
# - 4.5 day transfer
# - $10k/kg launch cost
```

**Expected Results:**
- Total ΔV: ~4000 m/s (realistic for lunar transfer)
- Total Cost: ~$1.5B
- ROI: 200%+ (with commercial payload delivery)
- NPV: Positive with 8% discount rate

### 2. ISRU Mining Operation
```bash
python src/cli.py analyze --config scenarios/06_isru_economics.json --output mining_analysis

# Characteristics:
# - 15000 kg mining equipment
# - ISRU benefits modeling
# - 5-year operation
# - Water ice extraction focus
```

**Expected Results:**
- ISRU Savings: 15-30% cost reduction
- Break-even: 3-4 years
- Long-term NPV: Highly positive
- Risk factors included in Monte Carlo

### 3. Sustainable Operations with Carbon Pricing
```bash
python src/cli.py analyze \
  --config scenarios/07_environmental_economics.json \
  --output green_mission \
  --carbon-price 120 \
  --learning-rate 0.82

# Characteristics:
# - Carbon emissions tracking
# - Wright's law cost reduction
# - Environmental impact pricing
```

**Expected Results:**
- Carbon cost impact: 5-10% of total
- Learning curve benefits: 18% cost reduction per doubling
- Sustainability metrics in dashboard

### 4. High-Risk Venture Analysis
```bash
python src/cli.py analyze --config scenarios/08_risk_analysis.json --output risk_assessment

# Characteristics:
# - 50% contingency
# - 12% discount rate (high risk)
# - Monte Carlo uncertainty
```

**Expected Results:**
- Wide NPV distribution
- P(NPV>0): 60-80%
- Critical risk factors identified
- Sensitivity tornado chart

### 5. Complete Lunar Base Mission
```bash
python src/cli.py analyze \
  --config scenarios/09_complete_mission.json \
  --output lunar_base \
  --population-size 100 \
  --generations 50

# Characteristics:
# - 25000 kg total mass
# - Comprehensive analysis
# - Multi-objective optimization
# - Executive dashboards
```

**Expected Results:**
- Full Pareto front (30+ solutions)
- Trade-off analysis (time vs cost vs fuel)
- Complete financial projections
- Interactive HTML dashboard

## Output & Results

### File Structure
```
output_directory/
├── analysis_results.json    # Complete numerical data
├── dashboard.html          # Executive visualization
├── summary.txt            # Key metrics summary
├── executive_dashboard.html # High-level KPIs (if available)
├── technical_dashboard.html # Engineering details (if available)
├── trajectory_plot.html    # 3D trajectory visualization
├── pareto_plot.html       # Multi-objective trade-offs
└── economic_dashboard.html # Financial analysis details
```

### Dashboard Features

The HTML dashboard includes:

1. **Trajectory Analysis Panel**
   - Total Delta-V (m/s)
   - Transfer time (days)
   - Propellant mass (kg)
   - Trajectory efficiency (%)

2. **Economic Analysis Panel**
   - Total mission cost ($)
   - Net Present Value (NPV)
   - Return on Investment (ROI %)
   - Payback period (years)
   - Cost breakdown chart

3. **Optimization Results Panel**
   - Population size & generations
   - Best solution metrics
   - Pareto front solutions table
   - Convergence history

4. **Mission Summary Panel**
   - Feasibility assessment
   - Economic viability rating
   - Risk assessment level
   - Key recommendations

### Interpreting Results

#### Delta-V Budget
- **<4000 m/s**: Excellent (efficient trajectory)
- **4000-5000 m/s**: Good (standard mission)
- **>5000 m/s**: Challenging (may need staging)

#### Economic Viability
- **ROI >15%**: Strong investment case
- **ROI 10-15%**: Moderate returns
- **ROI <10%**: Marginal viability

#### Optimization Quality
- **Pareto solutions >20**: Good diversity
- **Cache hit rate >50%**: Efficient optimization
- **Convergence achieved**: Reliable results

## Advanced Usage

### Multi-Mission Campaign Analysis
```bash
# Analyze shared infrastructure benefits
python src/cli.py analyze \
  --config scenarios/10_multi_mission_campaign.json \
  --output campaign_analysis \
  --population-size 80 \
  --generations 50
```

### Custom Optimization Parameters
```bash
# Fine-tune optimization bounds
python src/cli.py analyze \
  --config my_mission.json \
  --output custom_results \
  --min-earth-alt 200 \
  --max-earth-alt 1000 \
  --min-moon-alt 50 \
  --max-moon-alt 500 \
  --min-transfer-time 3.0 \
  --max-transfer-time 10.0
```

### Batch Processing
```bash
# Run multiple scenarios
for scenario in scenarios/*.json; do
    output_name=$(basename "$scenario" .json)
    python src/cli.py analyze \
        --config "$scenario" \
        --output "results/$output_name" \
        --population-size 40 \
        --generations 20
done
```

### Integration with CI/CD
```bash
# Validate before deployment
conda run -n py312 make test
conda run -n py312 python src/cli.py validate
conda run -n py312 python src/cli.py analyze --config test_scenario.json
```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: pygmo
```bash
# Solution: Activate correct environment
conda activate py312
conda install -c conda-forge pygmo=2.19.6
```

#### 2. Population Size Error
```
Error: for NSGA-II at least 5 individuals in the population are needed 
and the population size must be a multiple of 4
```
```bash
# Solution: Use valid population size
--population-size 52  # or 56, 60, 64, etc.
```

#### 3. Memory Issues
```bash
# For large optimizations, limit population/generations
--population-size 40 --generations 20

# Or increase system resources
export OMP_NUM_THREADS=4
```

#### 4. Visualization Not Loading
```bash
# Check output directory permissions
ls -la output_directory/
chmod 755 output_directory/*.html

# Open in modern browser (Chrome/Firefox/Safari)
```

### Debug Mode
```bash
# Enable full debug logging
python src/cli.py analyze --config my_mission.json --verbose

# Check specific module logs
grep "DEBUG:src.trajectory" output.log
grep "INFO:src.economics" output.log
```

### Performance Tips

1. **Start Small**: Use 20-40 population for initial runs
2. **Cache Warmup**: First run is slower, subsequent runs faster
3. **Parallel Execution**: Set `OMP_NUM_THREADS` for multi-core
4. **Profile Memory**: Monitor with `htop` during execution

## Best Practices

### 1. Configuration Management
- Version control your mission configs
- Use descriptive names for scenarios
- Document assumptions in descriptions

### 2. Result Validation
- Compare against known missions
- Check physical constraints (ΔV, time)
- Verify economic assumptions

### 3. Optimization Strategy
- Start with low pop/gen for testing
- Increase for production runs
- Use multiple seeds for robustness

### 4. Reporting
- Export dashboards as PDF for reports
- Include sensitivity analysis
- Document uncertainty ranges

## Support & Contributing

### Getting Help
- GitHub Issues: Report bugs or request features
- Documentation: Check `/docs` directory
- Tests: Review `/tests` for usage examples

### Contributing
- Follow CLAUDE.md guidelines
- Maintain 100% test coverage
- Use conda py312 environment
- Run `make pipeline` before commits

## Appendix: Scenario Library

| Scenario | Use Case | Key Features |
|----------|----------|--------------|
| 01_basic_transfer | Apollo-style cargo | Simple, proven approach |
| 02_launch_windows | Crew transport timing | Optimal departure analysis |
| 03_propulsion_comparison | Chemical vs electric | Technology trade study |
| 04_pareto_optimization | Multi-objective | Time/cost/fuel trades |
| 05_constellation | Satellite deployment | Phasing optimization |
| 06_isru_economics | Mining operations | Resource utilization |
| 07_environmental | Sustainable missions | Carbon pricing |
| 08_risk_analysis | High-risk ventures | Monte Carlo simulation |
| 09_complete_mission | Lunar base | Full integration demo |
| 10_multi_mission | Campaign planning | Infrastructure sharing |

---

**Remember**: Always use `conda activate py312` before running any commands!