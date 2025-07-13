# CLI Progress Tracking - User Guide

## ✅ What Has Been Fixed

The CLI now provides **real progress tracking** and **controlled debug output** for a much better user experience.

### Before (Issues):
- ❌ Flooded console with debug messages
- ❌ No progress indication
- ❌ Unclear how long to wait
- ❌ No time estimates

### After (Fixed):
- ✅ Clean progress display with real-time updates
- ✅ Accurate time estimates
- ✅ Debug output only when requested
- ✅ Meaningful status messages

## How to Use the Improved CLI

### Standard Usage (Clean Output)
```bash
# Activate conda environment
conda activate py312

# Run analysis with clean progress tracking
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output my_results

# Expected output:
🚀 Starting Lunar Horizon Optimizer Analysis...
📁 Loading configuration from scenarios/01_basic_transfer.json
🎯 Mission: Apollo-class Lunar Cargo Mission
⚙️  Optimization: 52 pop, 30 gen
⏱️  Estimated time: 15-30 minutes
💡 Use --verbose for debug output, otherwise only progress is shown

🔄 Running trajectory analysis | Elapsed: 2.3m | ETA: 12.7m | 15.3%
```

### Debug Mode (When Needed)
```bash
# Enable verbose output for troubleshooting
python src/cli.py analyze --config scenarios/01_basic_transfer.json --output debug_results --verbose

# This shows all internal debug messages and detailed logging
```

## Progress Display Explained

The progress indicator shows:
```
🔄 [Current Phase] | Elapsed: [Time] | ETA: [Estimate] | [Progress]%
```

### Example Phases:
1. **Initializing optimizer** (5%) - Setting up components
2. **Running trajectory analysis** (10-40%) - PyKEP Lambert solver
3. **Multi-objective optimization** (40-80%) - PyGMO NSGA-II evolution
4. **Economic analysis** (80-90%) - NPV, ROI calculations
5. **Creating visualizations** (90-100%) - Dashboard generation

## Time Estimates

The CLI provides realistic time estimates based on optimization parameters:

| Population × Generations | Estimated Time | Use Case |
|--------------------------|----------------|----------|
| 8 × 5 | 1-2 minutes | Quick test |
| 20 × 10 | 2-5 minutes | Demo/development |
| 52 × 30 | 15-30 minutes | Standard analysis |
| 100 × 50 | 30-60 minutes | Production quality |

### Quick Test Example:
```bash
# Fast test run (1-2 minutes)
python src/cli.py analyze \
  --config scenarios/01_basic_transfer.json \
  --output quick_test \
  --population-size 8 \
  --generations 5 \
  --no-sensitivity

# Expected completion: ~30 seconds to 2 minutes
```

### Standard Analysis Example:
```bash
# Production analysis (15-30 minutes)
python src/cli.py analyze \
  --config scenarios/01_basic_transfer.json \
  --output full_analysis \
  --population-size 52 \
  --generations 30

# Expected completion: 15-30 minutes
```

## Progress Features

### 1. Real-Time Updates
- Updates every few seconds during optimization
- Shows actual elapsed time
- Provides dynamic ETA based on current progress

### 2. Clean Console
- No debug spam unless requested
- Clear phase indicators
- Meaningful progress percentages

### 3. Accurate Estimates
- Based on actual population and generation counts
- Adjusts based on system performance
- Warns about long-running analyses

### 4. Controlled Logging
```bash
# Clean output (default)
python src/cli.py analyze --config my_mission.json

# Debug output (when troubleshooting)
python src/cli.py analyze --config my_mission.json --verbose
```

## Sample Command with Progress

The sample command also uses progress tracking:

```bash
python src/cli.py sample

# Output:
🚀 Running Quick Sample Analysis...
   This demonstrates basic lunar mission optimization
⏱️  Estimated time: 2-5 minutes

🔄 Initializing demo | Elapsed: 0s | ETA: 2.3m | 5.0%
🔄 Running quick optimization | Elapsed: 15s | ETA: 1.8m | 25.0%
✅ Analysis completed in 1.2 minutes

🎉 Quick Demo Complete!
   Mission: Quick Demo Mission
   Results saved to quick_demo_results/
```

## Background Execution

For long analyses, you can run in background:

```bash
# Run in background with progress log
nohup python src/cli.py analyze \
  --config scenarios/09_complete_mission.json \
  --output large_analysis \
  --population-size 100 \
  --generations 50 \
  > progress.log 2>&1 &

# Monitor progress
tail -f progress.log
```

## Troubleshooting

### If Progress Seems Stuck:
1. **Normal behavior** - PyKEP calculations can take time per iteration
2. **Check system resources** - Monitor CPU/memory usage
3. **Use smaller parameters** - Reduce population/generations for testing

### If No Progress Display:
1. **Check terminal** - Some terminals don't support real-time updates
2. **Use verbose mode** - Add `--verbose` to see detailed output
3. **Background execution** - Progress may not display in background

### Common Patterns:
```bash
# Quick validation
python src/cli.py analyze --config my_mission.json --population-size 8 --generations 5

# If taking too long, interrupt with Ctrl+C and try smaller parameters
```

## Best Practices

### 1. Start Small
Always test with small parameters first:
```bash
--population-size 8 --generations 5  # 1-2 minutes
```

### 2. Scale Up Gradually
```bash
--population-size 20 --generations 10  # 5-10 minutes
--population-size 52 --generations 30  # 15-30 minutes
```

### 3. Use Progress for Planning
- Check estimated time before starting long runs
- Plan coffee breaks during 30+ minute analyses
- Use background execution for multi-hour runs

### 4. Monitor Resources
```bash
# In another terminal
htop  # or 'top -o cpu' on macOS
```

## Summary

The CLI now provides **professional-grade progress tracking** with:
- ✅ Clean, real-time progress display
- ✅ Accurate time estimates
- ✅ Controlled debug output
- ✅ Meaningful phase indicators
- ✅ Background execution support

This makes the real lunar trajectory optimization system much more user-friendly while maintaining the full power of PyKEP/PyGMO calculations.