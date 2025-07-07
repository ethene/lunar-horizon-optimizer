#!/usr/bin/env python3
"""
Lunar Horizon Optimizer - Integrated System Setup
=================================================

Setup script for the complete Lunar Horizon Optimizer integrated system.
This script configures the environment, validates dependencies, and provides
guided setup for the MVP integration.

Author: Lunar Horizon Optimizer Development Team
Date: July 2025
Version: 1.0.0-rc1
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import importlib.util


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class LunarOptimizerSetup:
    """Setup manager for Lunar Horizon Optimizer."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.docs_dir = self.project_root / "docs"
        self.output_dir = self.project_root / "output"
        
        self.required_packages = [
            ("numpy", "1.24.0"),
            ("scipy", "1.10.0"),
            ("pandas", "2.0.0"),
            ("plotly", "5.20.0"),
            ("matplotlib", "3.7.0"),
            ("pydantic", "2.0.0"),
            ("pytest", "7.0.0"),
        ]
        
        self.conda_packages = [
            ("pykep", "2.6"),
            ("pygmo", "2.19.0"),
            ("astropy", "5.3.0"),
            ("spiceypy", "6.0.0"),
        ]
        
        self.setup_status = {
            'python_version': False,
            'conda_environment': False,
            'pip_packages': False,
            'conda_packages': False,
            'src_structure': False,
            'test_validation': False,
            'documentation': False,
            'examples': False
        }
    
    def print_header(self):
        """Print setup header."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}")
        print("🚀 LUNAR HORIZON OPTIMIZER - INTEGRATED SYSTEM SETUP")
        print("="*70)
        print(f"Version: 1.0.0-rc1{Colors.END}")
        print(f"{Colors.WHITE}Complete trajectory optimization and economic analysis platform{Colors.END}\n")
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        print(f"{Colors.BLUE}📋 Checking Python version...{Colors.END}")
        
        version_info = sys.version_info
        current_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
        if version_info.major == 3 and version_info.minor >= 10:
            print(f"   {Colors.GREEN}✅ Python {current_version} - Compatible{Colors.END}")
            self.setup_status['python_version'] = True
            return True
        else:
            print(f"   {Colors.RED}❌ Python {current_version} - Requires Python 3.10+{Colors.END}")
            return False
    
    def check_conda_environment(self) -> bool:
        """Check if running in conda environment."""
        print(f"{Colors.BLUE}🐍 Checking conda environment...{Colors.END}")
        
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            print(f"   {Colors.GREEN}✅ Running in conda environment: {conda_env}{Colors.END}")
            
            # Check if it's py312 environment
            if 'py312' in conda_env.lower():
                print(f"   {Colors.GREEN}✅ Detected py312 environment{Colors.END}")
                self.setup_status['conda_environment'] = True
                return True
            else:
                print(f"   {Colors.YELLOW}⚠️  Not in py312 environment - some features may not work{Colors.END}")
                return False
        else:
            print(f"   {Colors.RED}❌ Not running in conda environment{Colors.END}")
            print(f"   {Colors.YELLOW}Please activate conda py312 environment first{Colors.END}")
            return False
    
    def install_pip_packages(self) -> bool:
        """Install required pip packages."""
        print(f"{Colors.BLUE}📦 Installing pip packages...{Colors.END}")
        
        success = True
        for package, min_version in self.required_packages:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", f"{package}>={min_version}"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"   {Colors.GREEN}✅ {package} >= {min_version}{Colors.END}")
                else:
                    print(f"   {Colors.RED}❌ Failed to install {package}: {result.stderr}{Colors.END}")
                    success = False
                    
            except subprocess.TimeoutExpired:
                print(f"   {Colors.RED}❌ Timeout installing {package}{Colors.END}")
                success = False
            except Exception as e:
                print(f"   {Colors.RED}❌ Error installing {package}: {e}{Colors.END}")
                success = False
        
        if success:
            self.setup_status['pip_packages'] = True
        return success
    
    def check_conda_packages(self) -> bool:
        """Check conda packages installation."""
        print(f"{Colors.BLUE}🔬 Checking conda packages...{Colors.END}")
        
        all_available = True
        for package, min_version in self.conda_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    print(f"   {Colors.GREEN}✅ {package} available{Colors.END}")
                else:
                    print(f"   {Colors.YELLOW}⚠️  {package} not found - install with: conda install -c conda-forge {package}{Colors.END}")
                    all_available = False
            except ImportError:
                print(f"   {Colors.YELLOW}⚠️  {package} not available{Colors.END}")
                all_available = False
        
        if all_available:
            self.setup_status['conda_packages'] = True
        return all_available
    
    def validate_src_structure(self) -> bool:
        """Validate source code structure."""
        print(f"{Colors.BLUE}📁 Validating source structure...{Colors.END}")
        
        required_modules = [
            "lunar_horizon_optimizer.py",
            "cli.py",
            "config/mission_config.py",
            "trajectory/lunar_transfer.py",
            "optimization/global_optimizer.py", 
            "economics/financial_models.py",
            "visualization/dashboard.py"
        ]
        
        all_present = True
        for module in required_modules:
            module_path = self.src_dir / module
            if module_path.exists():
                print(f"   {Colors.GREEN}✅ {module}{Colors.END}")
            else:
                print(f"   {Colors.RED}❌ Missing: {module}{Colors.END}")
                all_present = False
        
        if all_present:
            self.setup_status['src_structure'] = True
        return all_present
    
    def run_basic_validation(self) -> bool:
        """Run basic system validation."""
        print(f"{Colors.BLUE}🧪 Running basic validation tests...{Colors.END}")
        
        try:
            # Add src to path
            sys.path.insert(0, str(self.src_dir))
            
            # Test basic imports
            print("   Testing core imports...")
            
            try:
                from lunar_horizon_optimizer import LunarHorizonOptimizer
                print(f"   {Colors.GREEN}✅ Core optimizer import successful{Colors.END}")
            except ImportError as e:
                print(f"   {Colors.RED}❌ Core optimizer import failed: {e}{Colors.END}")
                return False
            
            try:
                optimizer = LunarHorizonOptimizer()
                print(f"   {Colors.GREEN}✅ Optimizer initialization successful{Colors.END}")
            except Exception as e:
                print(f"   {Colors.YELLOW}⚠️  Optimizer initialization warning: {e}{Colors.END}")
                # Continue - some components may require conda packages
            
            self.setup_status['test_validation'] = True
            return True
            
        except Exception as e:
            print(f"   {Colors.RED}❌ Validation failed: {e}{Colors.END}")
            return False
    
    def create_example_configs(self) -> bool:
        """Create example configuration files."""
        print(f"{Colors.BLUE}📋 Creating example configurations...{Colors.END}")
        
        try:
            # Create output directory
            self.output_dir.mkdir(exist_ok=True)
            
            # Sample mission configuration
            sample_config = {
                "mission": {
                    "name": "Artemis Lunar Base Mission",
                    "earth_orbit_alt": 400.0,
                    "moon_orbit_alt": 100.0,
                    "transfer_time": 4.5,
                    "departure_epoch": 10000.0
                },
                "spacecraft": {
                    "name": "Lunar Base Supply Vehicle",
                    "dry_mass": 5000.0,
                    "propellant_mass": 3000.0,
                    "payload_mass": 1000.0,
                    "power_system_mass": 500.0,
                    "propulsion_isp": 320.0
                },
                "costs": {
                    "launch_cost_per_kg": 10000.0,
                    "operations_cost_per_day": 100000.0,
                    "development_cost": 1000000000.0,
                    "contingency_percentage": 20.0
                },
                "optimization": {
                    "population_size": 100,
                    "num_generations": 100,
                    "seed": 42,
                    "min_earth_alt": 200.0,
                    "max_earth_alt": 1000.0,
                    "min_moon_alt": 50.0,
                    "max_moon_alt": 500.0,
                    "min_transfer_time": 3.0,
                    "max_transfer_time": 10.0
                }
            }
            
            config_path = self.output_dir / "example_mission_config.json"
            with open(config_path, 'w') as f:
                json.dump(sample_config, f, indent=2)
            
            print(f"   {Colors.GREEN}✅ Example config: {config_path}{Colors.END}")
            
            # Create quick start script
            quickstart_script = """#!/usr/bin/env python3
\"\"\"
Quick Start Script for Lunar Horizon Optimizer
\"\"\"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig

def main():
    print("🚀 Lunar Horizon Optimizer - Quick Start")
    print("="*50)
    
    # Create optimizer with defaults
    optimizer = LunarHorizonOptimizer()
    
    # Quick analysis configuration
    opt_config = OptimizationConfig(
        population_size=20,  # Small for quick demo
        num_generations=10   # Small for quick demo
    )
    
    # Run analysis
    results = optimizer.analyze_mission(
        mission_name="Quick Start Demo",
        optimization_config=opt_config,
        include_sensitivity=False,  # Skip for speed
        include_isru=False,         # Skip for speed
        verbose=True
    )
    
    print("✅ Quick start completed!")
    return results

if __name__ == "__main__":
    main()
"""
            
            quickstart_path = self.output_dir / "quickstart.py"
            with open(quickstart_path, 'w') as f:
                f.write(quickstart_script)
            
            # Make executable
            os.chmod(quickstart_path, 0o755)
            
            print(f"   {Colors.GREEN}✅ Quick start script: {quickstart_path}{Colors.END}")
            
            self.setup_status['examples'] = True
            return True
            
        except Exception as e:
            print(f"   {Colors.RED}❌ Failed to create examples: {e}{Colors.END}")
            return False
    
    def create_user_guide(self) -> bool:
        """Create user guide documentation."""
        print(f"{Colors.BLUE}📖 Creating user guide...{Colors.END}")
        
        try:
            user_guide = """# Lunar Horizon Optimizer - User Guide

## Quick Start

1. **Environment Setup**:
   ```bash
   conda activate py312
   python setup_integrated_system.py
   ```

2. **Basic Usage**:
   ```python
   from lunar_horizon_optimizer import LunarHorizonOptimizer
   
   optimizer = LunarHorizonOptimizer()
   results = optimizer.analyze_mission("My Mission")
   ```

3. **Command Line Interface**:
   ```bash
   cd src
   python cli.py analyze --mission-name "My Mission"
   ```

## Configuration

Create a JSON configuration file:
```json
{
  "mission": {
    "name": "Lunar Mission",
    "earth_orbit_alt": 400.0,
    "moon_orbit_alt": 100.0
  },
  "optimization": {
    "population_size": 100,
    "num_generations": 100
  }
}
```

## Advanced Usage

### Custom Analysis
```python
from lunar_horizon_optimizer import LunarHorizonOptimizer, OptimizationConfig

optimizer = LunarHorizonOptimizer()

config = OptimizationConfig(
    population_size=200,
    num_generations=150
)

results = optimizer.analyze_mission(
    mission_name="Advanced Mission",
    optimization_config=config,
    include_sensitivity=True,
    include_isru=True
)
```

### Exporting Results
```python
optimizer.export_results(results, "output_directory")
```

## Visualization

Results include interactive visualizations:
- Executive dashboards
- Technical analysis plots
- 3D trajectory visualizations
- Pareto front analysis
- Economic analysis charts

## Troubleshooting

1. **Import Errors**: Ensure conda py312 environment is activated
2. **PyKEP/PyGMO Issues**: Install with `conda install -c conda-forge pykep pygmo`
3. **Visualization Problems**: Ensure Plotly is installed
4. **Performance Issues**: Reduce population_size and num_generations

For more help, see the comprehensive documentation in the docs/ directory.
"""
            
            guide_path = self.output_dir / "USER_GUIDE.md"
            with open(guide_path, 'w') as f:
                f.write(user_guide)
            
            print(f"   {Colors.GREEN}✅ User guide: {guide_path}{Colors.END}")
            
            self.setup_status['documentation'] = True
            return True
            
        except Exception as e:
            print(f"   {Colors.RED}❌ Failed to create user guide: {e}{Colors.END}")
            return False
    
    def print_setup_summary(self):
        """Print setup completion summary."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}SETUP SUMMARY{Colors.END}")
        print("="*50)
        
        for component, status in self.setup_status.items():
            if status:
                print(f"{Colors.GREEN}✅ {component.replace('_', ' ').title()}{Colors.END}")
            else:
                print(f"{Colors.RED}❌ {component.replace('_', ' ').title()}{Colors.END}")
        
        success_count = sum(self.setup_status.values())
        total_count = len(self.setup_status)
        
        print(f"\n{Colors.BOLD}Overall Progress: {success_count}/{total_count} components ready{Colors.END}")
        
        if success_count == total_count:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SETUP COMPLETED SUCCESSFULLY!{Colors.END}")
            print(f"{Colors.WHITE}Lunar Horizon Optimizer is ready for mission analysis.{Colors.END}")
            self.print_next_steps()
        elif success_count >= total_count * 0.7:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  SETUP MOSTLY COMPLETE{Colors.END}")
            print(f"{Colors.WHITE}Some optional components failed, but core functionality is available.{Colors.END}")
            self.print_next_steps()
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ SETUP INCOMPLETE{Colors.END}")
            print(f"{Colors.WHITE}Please resolve the failed components before proceeding.{Colors.END}")
    
    def print_next_steps(self):
        """Print next steps for users."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}NEXT STEPS{Colors.END}")
        print("="*30)
        print(f"{Colors.WHITE}1. Quick Start:{Colors.END}")
        print(f"   cd output && python quickstart.py")
        print(f"\n{Colors.WHITE}2. CLI Usage:{Colors.END}")
        print(f"   cd src && python cli.py validate")
        print(f"   cd src && python cli.py sample")
        print(f"\n{Colors.WHITE}3. Full Analysis:{Colors.END}")
        print(f"   cd src && python cli.py analyze --config ../output/example_mission_config.json")
        print(f"\n{Colors.WHITE}4. Documentation:{Colors.END}")
        print(f"   See output/USER_GUIDE.md for detailed instructions")
        print(f"   See docs/ directory for comprehensive documentation")
        print(f"\n{Colors.WHITE}5. Integration Testing:{Colors.END}")
        print(f"   pytest tests/test_task_7_mvp_integration.py -v")
    
    def run_full_setup(self):
        """Run complete setup process."""
        self.print_header()
        
        print(f"{Colors.BLUE}{Colors.BOLD}Starting integrated system setup...{Colors.END}\n")
        
        # Setup steps
        self.check_python_version()
        self.check_conda_environment()
        self.install_pip_packages()
        self.check_conda_packages()
        self.validate_src_structure()
        self.run_basic_validation()
        self.create_example_configs()
        self.create_user_guide()
        
        # Summary
        self.print_setup_summary()


def main():
    """Main setup function."""
    setup = LunarOptimizerSetup()
    setup.run_full_setup()


if __name__ == "__main__":
    main()