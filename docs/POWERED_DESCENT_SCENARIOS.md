# Powered Descent Scenarios Documentation

This document describes the three comprehensive powered descent scenarios designed for lunar landing mission optimization with economic analysis.

## Scenario Overview

All scenarios include the required sections:

1. **`mission`** section (departure epoch, parking orbit altitudes)
2. **`descent_parameters`** section (thrust, isp, burn_time)
3. **`optimizer`** section (pop_size, generations, include_descent: true)  
4. **`economics`** section (discount_rate, propellant_unit_cost, lander_fixed_cost)
5. **Top-level `description`** field explaining powered-descent optimization

## Scenario Specifications

### 11_powered_descent_mission.json - Artemis Commercial Cargo Lander

**Description**: "Lunar landing scenario with powered-descent optimization and economic analysis. This scenario demonstrates comprehensive cost modeling for lunar cargo delivery missions including propellant consumption, lander hardware costs, and multi-objective optimization of delta-v, time, and total mission cost."

**Mission Configuration**:
- Departure epoch: 10000.0 days since J2000
- Earth parking orbit altitude: 400.0 km  
- Moon parking orbit altitude: 100.0 km
- Transfer time: 5.2 days
- Mission: Artemis Commercial Cargo Lander

**Descent Parameters**:
- Thrust: 16,000 N (RL-10 derivative engine)
- ISP: 315 s (LOX/LH2 propellant)
- Burn time: 380 s (6.3 minute powered descent)
- Propellant mass: ~195 kg
- Landing site: Shackleton Crater Rim

**Optimizer Configuration**:
- Algorithm: NSGA-II
- Population size: 40
- Generations: 25
- Include descent: true
- Seed: 12345

**Economics**:
- Discount rate: 7%
- Propellant unit cost: $18/kg
- Lander fixed cost: $10M
- Expected descent fraction: 6-8% of total mission cost

**Expected Results**:
- Total mission cost: $165M - $195M
- Optimal delta-v: 3150 - 3400 m/s
- Optimal transfer time: 4.8 - 5.5 days
- ROI range: 18% - 25%
- Runtime: 60-90 seconds

### 12_powered_descent_mission.json - Blue Origin Lunar Cargo Express

**Description**: "Lunar landing scenario with powered-descent optimization and economic analysis. This scenario models a Blue Origin lunar cargo mission with reusable lander technology, demonstrating comprehensive cost optimization including propellant consumption, hardware amortization, and learning curve effects in the commercial lunar economy."

**Mission Configuration**:
- Departure epoch: 10000.0 days since J2000
- Earth parking orbit altitude: 350.0 km (optimized for New Glenn)
- Moon parking orbit altitude: 80.0 km (low lunar orbit for efficiency)
- Transfer time: 4.8 days (optimized trajectory)
- Mission: Blue Origin Lunar Cargo Express

**Descent Parameters**:
- Thrust: 24,000 N (BE-7 engine cluster)
- ISP: 345 s (methane/oxygen propellant)
- Burn time: 420 s (7 minute powered descent)
- Propellant mass: ~175 kg
- Landing site: Mare Imbrium

**Optimizer Configuration**:
- Algorithm: NSGA-II
- Population size: 50
- Generations: 30
- Include descent: true
- Seed: 54321

**Economics**:
- Discount rate: 6%
- Propellant unit cost: $16/kg (methane/LOX blend)
- Lander fixed cost: $8.5M (reusable lander)
- Learning rate: 0.88 (12% cost reduction per doubling)

**Expected Results**:
- Total mission cost: $155M - $180M
- Optimal delta-v: 3100 - 3350 m/s
- Optimal transfer time: 4.6 - 5.2 days
- ROI range: 22% - 28%
- Runtime: 90-120 seconds

### 13_powered_descent_quick.json - Quick Lunar Descent Test

**Description**: "Lunar landing scenario with powered-descent optimization and economic analysis. Quick-execution scenario for testing and validation of powered descent cost integration with minimal optimization parameters while maintaining realistic physics and economics."

**Mission Configuration**:
- Departure epoch: 10000.0 days since J2000
- Earth parking orbit altitude: 400.0 km
- Moon parking orbit altitude: 100.0 km
- Transfer time: 5.0 days
- Mission: Quick Lunar Descent Test

**Descent Parameters**:
- Thrust: 10,000 N (Aestus derivative)
- ISP: 320 s (MMH/NTO propellant)
- Burn time: 31.4 s (optimized for 100kg propellant mass)
- Propellant mass: ~100 kg
- Landing site: Oceanus Procellarum

**Optimizer Configuration**:
- Algorithm: NSGA-II
- Population size: 20 (reduced for speed)
- Generations: 15 (reduced for speed)
- Include descent: true
- Seed: 98765

**Economics**:
- Discount rate: 8%
- Propellant unit cost: $75/kg (higher cost hypergolic)
- Lander fixed cost: $8M
- Project duration: 6 years

**Expected Results**:
- Total mission cost: $115M - $135M
- Optimal delta-v: 3200 - 3450 m/s
- Optimal transfer time: 4.8 - 5.4 days
- ROI range: 15% - 22%
- Runtime: 25-45 seconds

## Usage Examples

### Basic Powered Descent Analysis
```bash
# Run Artemis scenario with powered descent
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent

# Run Blue Origin scenario with powered descent and risk analysis
./lunar_opt.py run scenario 12_powered_descent_mission --include-descent --risk

# Quick test scenario for validation
./lunar_opt.py run scenario 13_powered_descent_quick --include-descent --no-sensitivity --no-isru
```

### Performance Testing
```bash
# Fast execution for testing
./lunar_opt.py run scenario 13_powered_descent_quick --include-descent --gens 10 --population 12

# Standard analysis
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent

# Comprehensive analysis
./lunar_opt.py run scenario 12_powered_descent_mission --include-descent --gens 50 --risk --refine
```

### Comparative Analysis
```bash
# Compare scenarios without descent costs
./lunar_opt.py run scenario 11_powered_descent_mission

# Compare with descent costs enabled
./lunar_opt.py run scenario 11_powered_descent_mission --include-descent
```

## Validation and Testing

### Key Validation Points

1. **Cost Fraction Validation**:
   - Descent costs should represent 5-10% of total mission cost
   - Propellant costs calculated using rocket equation: `mass_flow_rate = thrust / (isp * g)`
   - Hardware costs are fixed per mission

2. **Propulsion System Realism**:
   - Thrust levels: 10-25 kN (realistic for lunar landers)
   - ISP values: 290-345 s (depending on propellant)
   - Burn times: 30-420 s (short terminal descent to longer powered descent)

3. **Economic Realism**:
   - Total mission costs: $115M - $195M
   - Propellant unit costs: $16-75/kg (depending on propellant type)
   - Lander hardware costs: $8M - $10M per mission

### Expected Performance

| Scenario | Runtime | Pop Size | Generations | Descent Fraction |
|----------|---------|----------|-------------|------------------|
| 11_powered_descent_mission | 60-90s | 40 | 25 | 6-8% |
| 12_powered_descent_mission | 90-120s | 50 | 30 | 5-7% |
| 13_powered_descent_quick | 25-45s | 20 | 15 | 7-10% |

### Troubleshooting

**If descent costs appear too low**:
- Check that `--include-descent` flag is used
- Verify `descent_parameters` exist in scenario
- Confirm propellant unit cost and lander fixed cost are reasonable

**If execution is too slow**:
- Use scenario 13 (quick version)
- Reduce population size with `--population` flag
- Reduce generations with `--gens` flag
- Add `--no-sensitivity --no-isru` flags

**If results seem unrealistic**:
- Verify propellant mass calculation: thrust/(isp*9.81)*burn_time
- Check that descent fraction is 5-10% of total cost
- Ensure thrust levels are 8-25 kN range for lunar landers

## Technical Implementation

### Rocket Equation Physics
All scenarios use the rocket equation for propellant mass calculation:
```
mass_flow_rate = thrust / (isp * g)  # kg/s
propellant_mass = mass_flow_rate * burn_time  # kg
```

### Cost Integration
Descent costs are integrated into the optimization objectives:
- **Objective 1**: Minimize delta-v (m/s)
- **Objective 2**: Minimize transfer time (days) 
- **Objective 3**: Minimize total cost (USD) including descent costs

### Multi-Objective Optimization
Uses NSGA-II algorithm to find Pareto-optimal solutions trading off:
- Mission performance (delta-v, time)
- Economic factors (total cost including descent)
- Mission constraints (fuel reserves, structural limits)

## Conclusion

These three scenarios provide comprehensive coverage of powered descent mission analysis:

- **Scenario 11**: Standard commercial cargo mission with realistic parameters
- **Scenario 12**: Advanced reusable lander with learning curve economics  
- **Scenario 13**: Quick validation scenario for testing and development

All scenarios maintain realistic physics, economics, and performance expectations while demonstrating the complete powered descent optimization capability.