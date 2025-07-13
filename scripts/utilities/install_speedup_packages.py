#!/usr/bin/env python3
"""
Install and validate speed-up packages for Lunar Horizon Optimizer.

This script ensures all performance optimization packages are properly installed
and configured for maximum calculation speed.
"""

import subprocess
import sys
import importlib
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"   ✅ SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ FAILED: {e.stderr}")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ❌ {package_name}: Not installed")
        return False


def main():
    """Install and validate speed-up packages."""
    print("🚀 Installing Speed-up Packages for Lunar Horizon Optimizer")
    print("=" * 60)
    
    # Check if we're in conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'py312':
        print("⚠️  WARNING: Not in py312 conda environment")
        print("   Run: conda activate py312")
        print()
    else:
        print(f"✅ Using conda environment: {conda_env}")
        print()
    
    # Install speed-up packages
    speedup_packages = [
        ("numba", "conda install -c conda-forge numba -y"),
        ("joblib", "conda install -c conda-forge joblib -y"),
        ("dask", "conda install -c conda-forge dask -y"),
        ("fastparquet", "conda install -c conda-forge fastparquet -y")
    ]
    
    print("📦 Installing Speed-up Packages:")
    print("-" * 40)
    
    success_count = 0
    for package, install_cmd in speedup_packages:
        if run_command(install_cmd, f"Installing {package}"):
            success_count += 1
        print()
    
    # Validate installations
    print("🔍 Validating Installations:")
    print("-" * 40)
    
    validation_packages = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("JAX", "jax"),
        ("PyKEP", "pykep"),
        ("PyGMO", "pygmo"),
        ("Numba", "numba"),
        ("Joblib", "joblib"),
        ("Dask", "dask"),
        ("FastParquet", "fastparquet")
    ]
    
    valid_count = 0
    for display_name, import_name in validation_packages:
        if check_package(display_name, import_name):
            valid_count += 1
    
    print()
    print("🧪 Testing Performance Optimizations:")
    print("-" * 40)
    
    # Test performance module
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from utils.performance import (
            get_optimization_status, 
            enable_performance_optimizations,
            fast_norm,
            jit_compile
        )
        
        # Enable optimizations
        status = enable_performance_optimizations()
        print("   ✅ Performance module imported successfully")
        
        # Test JIT compilation
        import numpy as np
        test_vector = np.array([1.0, 2.0, 3.0])
        norm_result = fast_norm(test_vector)
        print(f"   ✅ JIT functions working: norm([1,2,3]) = {norm_result:.3f}")
        
        # Show optimization status
        opt_status = get_optimization_status()
        print("\n📊 Optimization Status:")
        for package, info in opt_status.items():
            status_icon = "✅" if info['available'] else "❌"
            version = info.get('version', 'N/A')
            print(f"   {status_icon} {package.capitalize()}: {version}")
        
    except Exception as e:
        print(f"   ❌ Performance module test failed: {e}")
    
    print()
    print("=" * 60)
    print("📈 Installation Summary:")
    print(f"   Speed-up packages: {success_count}/{len(speedup_packages)} installed")
    print(f"   Core packages: {valid_count}/{len(validation_packages)} available")
    
    if success_count == len(speedup_packages) and valid_count == len(validation_packages):
        print("\n🎉 All speed-up packages installed successfully!")
        print("💡 Your Lunar Horizon Optimizer is now optimized for maximum performance.")
        print("\nNext steps:")
        print("   1. Run: python src/cli.py validate")
        print("   2. Test: python src/cli.py sample")
        print("   3. Analyze: python src/cli.py analyze --config scenarios/01_basic_transfer.json")
        return 0
    else:
        print("\n⚠️  Some packages failed to install or validate.")
        print("💡 The optimizer will work but may be slower without all optimizations.")
        print("\nTroubleshooting:")
        print("   1. Ensure conda py312 environment is activated")
        print("   2. Update conda: conda update conda")
        print("   3. Clear cache: conda clean --all")
        return 1


if __name__ == "__main__":
    sys.exit(main())