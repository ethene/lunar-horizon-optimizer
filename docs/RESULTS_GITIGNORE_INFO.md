# Analysis Results - Git Ignore Configuration

## 🎯 Overview

All analysis results and output directories are automatically excluded from git tracking to keep the repository clean and prevent accidentally committing large result files.

## 📁 Ignored Patterns

### Analysis Output Directories
The following patterns are excluded from git:

```bash
# Generic patterns
*_test*/          # Any directory ending with _test
test_*/           # Any directory starting with test_
*_results/        # Any directory ending with _results
*results*/        # Any directory containing results
analysis_*/       # Analysis output directories
output_*/         # Generic output directories

# Specific CLI patterns
quick_test_*/
production_test_*/
standard_test_*/
research_test_*/
demo_*/
quick_demo*/
```

### Analysis Output Files
Individual result files are also ignored:

```bash
analysis_*.json           # Analysis metadata
financial_summary.json    # Economic results
*_dashboard.html          # Interactive dashboards
*_plot.html              # Trajectory plots
*_visualization.html     # General visualizations
executive_dashboard.html  # Specific dashboard types
technical_dashboard.html
economic_dashboard.html
```

## 🔧 Commands That Generate Ignored Output

### CLI Commands
```bash
# These automatically create ignored directories:
python src/cli.py analyze --output my_test
python src/cli.py analyze --config scenarios/01_basic_transfer.json
python src/cli.py sample
```

### Examples Script
```bash
# These use timestamped output directories (auto-ignored):
python run_analysis_examples.py quick        # → quick_test_HHMMSS/
python run_analysis_examples.py production   # → production_test_HHMMSS/
python run_analysis_examples.py standard     # → standard_test_HHMMSS/
python run_analysis_examples.py research     # → research_test_HHMMSS/
```

## ✅ Verification

### Check What's Ignored
```bash
# See current git status (should not show result directories)
git status

# Test if a result directory would be ignored
mkdir test_example_123
echo "test" > test_example_123/analysis_metadata.json
git status  # Should not show test_example_123/
rm -rf test_example_123
```

### Force Include a Result (Not Recommended)
```bash
# If you need to track a specific result for documentation:
git add -f my_special_results/
```

## 📚 Benefits

### Clean Repository
- No large HTML dashboard files in git history
- No generated JSON results cluttering the repo
- Faster cloning and pulling
- Focus on source code, not outputs

### Easy Development
- Run analyses without worrying about git pollution
- Test different scenarios freely
- Results stay local for your review
- Share code, not generated results

### Production Workflow
- Generate results locally
- Review dashboards and data
- Share analysis code and configurations
- Keep repository lightweight

## 🚨 Important Notes

### Don't Commit Results
- Results are meant to be generated fresh
- Each analysis run may have different timestamps
- HTML dashboards can be large (100KB-1MB+)
- JSON results contain environment-specific paths

### Sharing Results
Instead of committing results, share:
- **Scenario configurations**: `scenarios/*.json`
- **Analysis commands**: CLI commands or script usage
- **Screenshots**: Of key dashboard visualizations
- **Summary data**: Key metrics in documentation

### Backup Strategy
Results are excluded from git but you can:
- Archive important results separately
- Export key metrics to documentation
- Save dashboard screenshots for reports
- Use external storage for large result sets

## 🔍 Current Status

The .gitignore is configured to exclude all analysis results automatically. You can run any analysis command and the results will stay local without affecting git status.

**Last Updated**: 2025-07-13