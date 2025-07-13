# Performance and Usage Guide - Real vs Simplified Optimizer

## Overview

The Lunar Horizon Optimizer has two operational modes:

1. **Real Integration (LunarHorizonOptimizer)** - Full PyKEP/PyGMO calculations
2. **Simplified Mode (SimpleLunarOptimizer)** - Fast approximations for demos

## Real Integration Performance

### Expected Execution Times

The real `LunarHorizonOptimizer` performs comprehensive calculations:

| Population | Generations | Expected Time | Use Case |
|------------|-------------|---------------|----------|
| 8 | 5 | 2-5 minutes | Quick test |
| 20 | 10 | 5-10 minutes | Demo run |
| 52 | 30 | 15-30 minutes | Standard analysis |
| 100 | 50 | 30-60 minutes | Production run |
| 200 | 100 | 1-2 hours | Research quality |

### Why It Takes Time

1. **PyKEP Lambert Solver** - Iterative solutions for each trajectory
2. **PyGMO NSGA-II** - Multi-objective genetic algorithm evolution
3. **Cache Building** - First run builds trajectory cache
4. **Real Ephemeris** - NASA SPICE data for accurate positions

### Running Real Analysis

```bash
# ALWAYS use conda environment
conda activate py312

# Quick test (2-5 minutes)
python src/cli.py analyze \
  --config scenarios/01_basic_transfer.json \
  --output quick_test \
  --population-size 8 \
  --generations 5

# Standard analysis (15-30 minutes)
python src/cli.py analyze \
  --config scenarios/09_complete_mission.json \
  --output full_analysis \
  --population-size 52 \
  --generations 30
```

## Performance Optimization Tips

### 1. Start Small
```bash
# Test configuration first
--population-size 8 --generations 5

# Then scale up
--population-size 52 --generations 30
```

### 2. Use Caching
The optimizer caches trajectory calculations. Second runs are faster:
- First run: Builds cache
- Subsequent: 30-50% faster

### 3. Parallel Execution
```bash
# Set thread count
export OMP_NUM_THREADS=4
python src/cli.py analyze --config my_mission.json
```

### 4. Monitor Progress
```bash
# Use verbose mode to see progress
python src/cli.py analyze --config my_mission.json --verbose

# Check system resources
htop  # In another terminal
```

## When to Use Each Mode

### Use Real Integration For:
- **Production analysis** requiring accurate results
- **Scientific papers** needing validated calculations
- **Mission planning** with real trajectory data
- **Economic analysis** with full financial modeling

### Use Simplified Mode For:
- **Quick demos** (<1 minute execution)
- **Testing configurations** before full runs
- **Development** and debugging
- **Teaching** basic concepts

## Understanding the Results

### Real Calculations Include:

1. **Trajectory**
   - Lambert problem solutions
   - Actual Earth-Moon ephemeris
   - Gravity assists calculations
   - N-body perturbations

2. **Optimization**
   - True Pareto fronts
   - Convergence metrics
   - Population diversity
   - Constraint handling

3. **Economics**
   - Cash flow modeling
   - NPV with discounting
   - ISRU benefit analysis
   - Monte Carlo risk

### Validation Checklist

✅ **Delta-V values** should be 3800-4500 m/s for lunar transfers
✅ **Transfer times** typically 3-7 days
✅ **Costs** reflect real launch prices ($2k-20k/kg)
✅ **ROI** depends on mission type (10-200%)
✅ **Pareto solutions** show clear trade-offs

## Example: Running Complete Analysis

```bash
# Step 1: Activate environment
conda activate py312

# Step 2: Validate setup
python src/cli.py validate

# Step 3: Run analysis (expect 20-30 minutes)
time python src/cli.py analyze \
  --config scenarios/09_complete_mission.json \
  --output lunar_base_results \
  --population-size 52 \
  --generations 30 \
  --verbose

# Step 4: Check results
open lunar_base_results/dashboard.html
cat lunar_base_results/summary.txt
```

## Troubleshooting Performance

### If Taking Too Long:

1. **Reduce population/generations**
   ```bash
   --population-size 20 --generations 10
   ```

2. **Check system resources**
   ```bash
   top -o cpu  # macOS
   htop        # Linux
   ```

3. **Use simplified scenarios**
   - Start with scenarios 01-03 (simpler)
   - Avoid 09-10 (complex) initially

4. **Enable debug logging**
   ```bash
   --verbose 2>&1 | tee analysis.log
   grep "Generation" analysis.log  # Track progress
   ```

### Memory Issues:

```bash
# Monitor memory usage
vmstat 1  # Linux
vm_stat 1 # macOS

# Limit optimization scope
--max-transfer-time 5.0  # Reduce search space
```

## Background Execution

For long runs, use background execution:

```bash
# Using nohup
nohup python src/cli.py analyze \
  --config scenarios/10_multi_mission_campaign.json \
  --output campaign_results \
  --population-size 100 \
  --generations 50 \
  > analysis.log 2>&1 &

# Check progress
tail -f analysis.log

# Using screen/tmux
screen -S lunar_analysis
python src/cli.py analyze --config my_mission.json
# Ctrl+A, D to detach
screen -r lunar_analysis  # Reattach
```

## Performance Benchmarks

### Test System: M1 Mac, 16GB RAM

| Scenario | Pop×Gen | Real Time | Simplified Time |
|----------|---------|-----------|-----------------|
| Basic Transfer | 8×5 | 3 min | 0.8 sec |
| Launch Windows | 20×10 | 8 min | 0.8 sec |
| ISRU Economics | 52×30 | 25 min | 0.9 sec |
| Complete Mission | 100×50 | 55 min | 1.0 sec |

### Key Insights:
- **Real optimizer** is 200-3000× slower but provides accurate results
- **First run** is slower due to cache building
- **Memory usage** peaks at 2-3GB for large populations
- **CPU usage** is near 100% during optimization

## Conclusion

The real `LunarHorizonOptimizer` provides scientifically accurate results but requires patience. For production use:

1. **Start with small test runs** to validate configuration
2. **Scale up gradually** based on time availability
3. **Use background execution** for long analyses
4. **Monitor system resources** during execution
5. **Leverage caching** for iterative improvements

The investment in computation time yields professional-grade mission analysis with real orbital mechanics, validated financial models, and comprehensive optimization - suitable for actual mission planning and academic research.