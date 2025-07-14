# CLI Powered Descent Integration Documentation

## Overview

This document describes the implementation of the `--include-descent` CLI flag for the Lunar Horizon Optimizer, which enables powered descent cost optimization and landing analysis.

## Implementation Summary

The powered descent CLI integration adds the ability to include lunar landing costs in the optimization pipeline through a simple command-line flag. When enabled, the system reads descent parameters from scenario configurations and integrates them into the multi-objective optimization process.

## Key Features

### ✅ CLI Flag Integration
- Added `--include-descent` flag to the `lunar-opt run scenario` command
- Flag enables powered descent optimization and cost modeling
- Backward compatible - all existing scenarios continue to work without the flag

### ✅ Configuration Integration  
- Reads `descent_parameters` from JSON/YAML scenario files
- Supports thrust, specific impulse, and burn time parameters
- Graceful handling when descent parameters are missing from scenarios

### ✅ Optimization Pipeline Integration
- Passes descent parameters through the complete analysis pipeline
- Integrates with existing multi-objective optimization (NSGA-II)
- Includes descent costs in economic analysis and visualization

## Usage

### Basic Usage
```bash
# Enable powered descent analysis
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent

# Standard analysis without descent costs (backward compatible)
./lunar_opt.py run scenario 11_powered_descent_mission
```

### Combined with Other Flags
```bash
# Comprehensive analysis with powered descent
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent --gens 50 --risk --refine

# Quick test with powered descent
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent --gens 5 --population 8 --no-sensitivity
```

### Help and Documentation
```bash
# View all available flags including --include-descent
./lunar_opt.py run scenario --help

# Check scenario information
./lunar_opt.py run info 11_powered_descent_mission
```

## Scenario Configuration

### Required Descent Parameters
For powered descent analysis, scenarios must include a `descent_parameters` section:

```json
{
  "descent_parameters": {
    "thrust": 16000.0,      // N - Engine thrust
    "isp": 315.0,           // s - Specific impulse
    "burn_time": 380.0,     // s - Powered descent duration
    "landing_site": "Shackleton Crater Rim",
    "guidance_mode": "terrain_relative_navigation"
  }
}
```

### YAML Example
```yaml
descent_parameters:
  engine_type: BE-7
  thrust: 24000.0          # N - 24 kN total thrust
  isp: 345.0              # s - Vacuum specific impulse  
  burn_time: 420.0        # s - 7 minute powered descent
  fuel: liquid_methane
  oxidizer: liquid_oxygen
```

### Optional Parameters
Additional parameters are supported but not required:
- `engine_type`: Engine designation (informational)
- `landing_site`: Target landing location (informational)
- `guidance_mode`: Landing guidance system (informational)
- `fuel`, `oxidizer`: Propellant types (informational)

## Implementation Details

### CLI Architecture
The integration follows the existing CLI architecture patterns:

1. **Flag Definition**: Added to `src/cli/main.py` in the `run_scenario` command
2. **Parameter Extraction**: Descent parameters extracted from scenario configuration
3. **Pipeline Integration**: Parameters passed through the analysis pipeline
4. **Cost Integration**: Descent costs included in economic optimization

### Code Flow
```python
# 1. CLI flag parsing
@click.option("--include-descent", is_flag=True, 
              help="Enable powered-descent optimization and cost modeling")

# 2. Configuration extraction
if include_descent:
    descent_params = config.get("descent_parameters")

# 3. Pipeline integration  
results = optimizer.analyze_mission(
    descent_params=descent_params,
    # ... other parameters
)

# 4. Optimization integration
problem = LunarMissionProblem(
    descent_params=descent_params,
    # ... other parameters  
)

# 5. Cost calculation
cost = self.cost_calculator.calculate_mission_cost(
    descent_params=self.descent_params,
    # ... other parameters
)
```

### Backward Compatibility
The implementation maintains full backward compatibility:
- All existing scenario calls continue to work unchanged
- `descent_params` parameter defaults to `None` throughout the system
- When `descent_params=None`, descent costs are zero (no impact)
- Existing scenarios without descent parameters work normally

## Error Handling

### Missing Descent Parameters
When `--include-descent` is specified but no descent parameters are found:
```bash
⚠️ --include-descent flag set but no descent_parameters found in scenario
```

### Invalid Parameters
The cost integration module validates parameters and applies bounds:
- Thrust: Must be positive
- ISP: Must be positive  
- Burn time: Must be positive
- Propellant mass: Bounded to 50-2000 kg range

### Graceful Degradation
- System continues to work if descent parameters are malformed
- Falls back to zero descent costs when parameters are invalid
- Does not break existing optimization pipeline

## Testing and Validation

### Available Scenarios with Descent Parameters
- `11_powered_descent_mission.json` - Artemis Commercial Cargo Lander
- `12_powered_descent_mission.yaml` - Blue Origin Lunar Cargo Express

### Test Coverage
- ✅ CLI flag parsing and help text
- ✅ Configuration loading and validation
- ✅ Parameter passing through pipeline
- ✅ Cost integration and calculation
- ✅ Backward compatibility verification

### Validation Commands
```bash
# Test configuration loading
./lunar_opt.py run info 11_powered_descent_mission

# Test flag help text
./lunar_opt.py run scenario --help

# Quick integration test
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent --gens 1 --population 4 --no-sensitivity --no-isru
```

## Cost Integration Details

### Rocket Equation Implementation
The system calculates descent costs using the rocket equation:
```python
mass_flow_rate = thrust / (isp * g)  # kg/s
propellant_mass = mass_flow_rate * burn_time  # kg
descent_propellant_cost = propellant_mass * propellant_unit_cost
```

### Cost Breakdown
Powered descent costs are integrated into the total mission cost:
- **Descent Propellant Cost**: Calculated from thrust, ISP, and burn time
- **Lander Hardware Cost**: Fixed cost for landing system
- **Total Integration**: Added to existing cost objectives in optimization

### Economic Analysis
Descent costs are included in:
- Multi-objective optimization (delta-v, time, cost)
- Economic analysis (NPV, IRR, ROI calculations)
- Cost breakdown visualization
- Financial reporting and dashboards

## Future Enhancements

### Planned Improvements
1. **Multiple Landing Sites**: Support for trajectory optimization to different landing locations
2. **Reusable Landers**: Cost modeling for reusable landing systems
3. **Engine Throttling**: Variable thrust profiles during descent
4. **Trajectory Integration**: Direct integration with powered descent trajectory calculations

### Extension Points
The implementation provides extension points for:
- Additional descent parameters (throttle profiles, guidance systems)
- Alternative cost models (reusability, maintenance, fuel production)
- Advanced trajectory analysis (optimal landing site selection)
- Risk analysis (landing failure probability, cost uncertainty)

## Conclusion

The powered descent CLI integration provides a seamless way to include lunar landing costs in mission optimization. The implementation follows aerospace industry standards, maintains full backward compatibility, and integrates cleanly with the existing optimization pipeline.

Key benefits:
- **Simple Usage**: Single `--include-descent` flag enables functionality
- **Realistic Physics**: Uses rocket equation for propellant calculations
- **Economic Integration**: Includes descent costs in financial analysis
- **Flexible Configuration**: Supports both JSON and YAML scenarios
- **Production Ready**: Comprehensive error handling and validation

The integration enables users to perform complete lunar mission analysis including landing costs, making it suitable for both academic research and commercial mission planning.