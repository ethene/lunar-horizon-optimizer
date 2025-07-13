# Lunar Horizon Optimizer - Scenario Library

This directory contains working configuration files for demonstrating real-world lunar mission analysis use cases. Each scenario is designed to showcase specific capabilities of the Lunar Horizon Optimizer platform.

## Quick Start

```bash
# Run any scenario
conda activate py312
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output results/scenario_01

# View results
cat results/scenario_01/summary.txt
```

## Available Scenarios

### ✅ Scenario 1: Basic Earth-Moon Transfer
**File:** `01_basic_transfer.json`
**Runtime:** ~5 seconds
**Purpose:** Demonstrates optimal Earth-Moon transfer trajectory calculation

**Command:**
```bash
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output results/basic_transfer
```

**What it demonstrates:**
- Apollo-class cargo mission configuration
- Delta-v analysis for Earth-Moon transfer
- Economic analysis with realistic cost factors
- Multi-objective optimization (40 pop, 25 gen)

**Expected Results:**
- Total Delta-V: ~4000 m/s
- Transfer Time: 4.5 days
- Total Cost: ~$900M
- Positive ROI and NPV

### 🚧 Scenario 2: Launch Window Analysis
**File:** `02_launch_windows.json` (Coming Soon)
**Runtime:** < 1 minute
**Purpose:** Find optimal launch opportunities over time

### 🚧 Scenario 3: Propulsion Comparison
**File:** `03_propulsion_comparison.json` (Coming Soon)
**Runtime:** < 1 minute
**Purpose:** Compare chemical vs electric propulsion systems

### 🚧 Scenario 4: Pareto Front Analysis
**File:** `04_pareto_optimization.json` (Coming Soon)
**Runtime:** ~1.5 minutes
**Purpose:** Multi-objective trade-off analysis

### 🚧 Scenario 5: Constellation Optimization
**File:** `05_constellation.yaml` (Coming Soon)
**Runtime:** ~1.5 minutes
**Purpose:** Multi-satellite mission optimization

### 🚧 Scenario 6: ISRU Business Case
**File:** `06_isru_economics.json` (Coming Soon)
**Runtime:** < 1 minute
**Purpose:** In-Situ Resource Utilization economic analysis

### 🚧 Scenario 7: Environmental Economics
**File:** `07_environmental_economics.json` (Coming Soon)
**Runtime:** < 1 minute
**Purpose:** Learning curves and carbon pricing impact

### 🚧 Scenario 8: Risk Analysis
**File:** `08_risk_analysis.json` (Coming Soon)
**Runtime:** ~1.5 minutes
**Purpose:** Monte Carlo uncertainty analysis

### 🚧 Scenario 9: Complete Mission
**File:** `09_complete_mission.yaml` (Coming Soon)
**Runtime:** ~2 minutes
**Purpose:** End-to-end mission analysis with dashboards

### 🚧 Scenario 10: Multi-Mission Campaign
**File:** `10_multi_mission_campaign.yaml` (Coming Soon)
**Runtime:** ~2 minutes
**Purpose:** Mission sequence optimization with shared infrastructure

## Scenario Configuration Format

Each scenario is a JSON file with the following structure:

```json
{
  "mission": {
    "name": "Mission Name",
    "description": "Detailed description of the mission purpose",
    "transfer_time": 4.5
  },
  "spacecraft": {
    "dry_mass": 8000.0,
    "max_propellant_mass": 6000.0,
    "payload_mass": 2000.0,
    "specific_impulse": 440.0
  },
  "costs": {
    "launch_cost_per_kg": 8000.0,
    "operations_cost_per_day": 150000.0,
    "development_cost": 500000000.0,
    "contingency_percentage": 25.0,
    "learning_rate": 0.92,
    "carbon_price_per_ton_co2": 45.0
  },
  "orbit": {
    "semi_major_axis": 6778.0,
    "inclination": 28.5,
    "eccentricity": 0.0
  },
  "optimization": {
    "population_size": 40,
    "num_generations": 25,
    "seed": 12345
  }
}
```

## Parameter Guidelines

### Mission Parameters
- **transfer_time**: Nominal transfer duration (days)
- **name**: Descriptive mission identifier
- **description**: Detailed mission purpose and context

### Spacecraft Parameters
- **dry_mass**: Spacecraft dry mass without propellant (kg)
- **max_propellant_mass**: Maximum propellant capacity (kg)
- **payload_mass**: Mission payload mass (kg)
- **specific_impulse**: Propulsion system Isp (seconds)

### Cost Parameters
- **launch_cost_per_kg**: Launch cost per kilogram to LEO (USD/kg)
- **operations_cost_per_day**: Daily operations cost (USD/day)
- **development_cost**: Total development cost (USD)
- **contingency_percentage**: Cost contingency factor (%)
- **learning_rate**: Wright's law learning rate (0.9 = 10% reduction per doubling)
- **carbon_price_per_ton_co2**: Carbon pricing for environmental costs (USD/tCO₂)

### Orbit Parameters
- **semi_major_axis**: Target orbit semi-major axis (km)
- **inclination**: Orbit inclination (degrees)
- **eccentricity**: Orbit eccentricity (0.0 = circular)

### Optimization Parameters
- **population_size**: Genetic algorithm population size
- **num_generations**: Number of optimization generations
- **seed**: Random seed for reproducible results

## Output Files

Each scenario analysis produces:

- **`summary.txt`**: Human-readable mission summary
- **`analysis_results.json`**: Complete structured results data
- **`trajectory_plot.html`**: Interactive trajectory visualization (future)
- **`economic_dashboard.html`**: Financial analysis dashboard (future)

## Performance Guidelines

- **Quick scenarios** (< 1 minute): Basic analysis, parameter sweeps
- **Standard scenarios** (1-2 minutes): Multi-objective optimization, risk analysis
- **Complex scenarios** (2+ minutes): Comprehensive studies, constellation optimization

## Creating Custom Scenarios

1. **Start with an existing scenario**:
   ```bash
   cp scenarios/01_basic_transfer.json scenarios/my_mission.json
   ```

2. **Edit parameters** for your mission requirements

3. **Test the scenario**:
   ```bash
   python src/cli.py analyze --config scenarios/my_mission.json --output results/my_mission
   ```

4. **Validate results** and adjust parameters as needed

## Troubleshooting

### Configuration Errors
- Check JSON syntax with a validator
- Ensure all required fields are present
- Verify parameter ranges (positive masses, valid inclinations, etc.)

### Analysis Failures
- Run environment validation: `python src/cli.py validate`
- Check conda environment: `conda activate py312`
- Enable verbose output: `--verbose` flag

### Performance Issues
- Reduce `population_size` and `num_generations` for faster results
- Use `--no-sensitivity` flag to skip sensitivity analysis
- Monitor system resources during optimization

---

**Next Steps:** Run `python src/cli.py sample` for a quick demo, then try Scenario 1 to see the full workflow in action.