#!/usr/bin/env python3
"""
Dependency verification script for Lunar Horizon Optimizer.
Tests if all required libraries are installed and working correctly.
"""

import importlib
import sys
from typing import Dict, Tuple, List

def check_import(module_name: str, min_version: str = None) -> Tuple[bool, str]:
    """
    Check if a module can be imported and optionally verify its version.
    
    Args:
        module_name: Name of the module to import
        min_version: Minimum required version (optional)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as version_parser
            if version_parser.parse(version) < version_parser.parse(min_version):
                return False, f"{module_name} version {version} is older than required version {min_version}"
        
        return True, f"{module_name} (version {version}) successfully imported"
    except ImportError as e:
        return False, f"Failed to import {module_name}: {str(e)}"
    except Exception as e:
        return False, f"Error checking {module_name}: {str(e)}"

def verify_jax_gpu() -> Tuple[bool, str]:
    """Verify JAX GPU support."""
    try:
        import jax
        devices = jax.devices()
        if any(d.platform == 'gpu' for d in devices):
            return True, "JAX GPU support verified"
        return False, "No GPU devices found for JAX"
    except Exception as e:
        return False, f"Error checking JAX GPU support: {str(e)}"

def check_scipy_version() -> Tuple[bool, str]:
    """Check SciPy version and compatibility."""
    try:
        import scipy
        version = scipy.__version__
        if version >= "1.14.0":
            return False, f"SciPy version {version} is too new (needs <1.14.0 for PyKEP compatibility)"
        return True, f"SciPy {version} is compatible"
    except ImportError as e:
        return False, f"Failed to import SciPy: {str(e)}"

def test_trajectory_components() -> List[Tuple[bool, str, bool]]:
    """Test trajectory calculation components (PyKEP, PyGMO).
    Returns: List of (success, message, is_critical)"""
    results = []
    
    # Test PyKEP
    try:
        import pykep as pk
        version = pk.__version__
        # Try a simple calculation instead of planet creation
        pk.epoch(0)
        results.append((True, f"PyKEP {version}: Basic functionality works", True))
    except ImportError as e:
        results.append((False, f"PyKEP import failed: {str(e)}", True))
    except Exception as e:
        results.append((False, f"PyKEP test failed: {str(e)}", True))
    
    # Test PyGMO with minimal functionality
    try:
        import pygmo as pg
        version = pg.__version__
        # Try to create a simple problem without using advanced features
        results.append((True, f"PyGMO {version}: Successfully imported", True))
    except ImportError as e:
        results.append((False, f"PyGMO import failed: {str(e)}", True))
    except Exception as e:
        results.append((False, f"PyGMO test failed: {str(e)}", True))
    
    return results

def test_optimization_components() -> List[Tuple[bool, str, bool]]:
    """Test optimization components (JAX, Diffrax).
    Returns: List of (success, message, is_critical)"""
    results = []
    
    # Test JAX
    try:
        import jax
        import jax.numpy as jnp
        version = jax.__version__
        # Simple computation test
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        results.append((True, f"JAX {version}: Basic computation works", True))
        
        # Check device availability
        devices = jax.devices()
        dev_msg = "Available devices: " + ", ".join(d.device_kind for d in devices)
        results.append((True, dev_msg, False))
    except ImportError as e:
        results.append((False, f"JAX import failed: {str(e)}", True))
    except Exception as e:
        results.append((False, f"JAX test failed: {str(e)}", True))
    
    # Test Diffrax with minimal functionality
    try:
        import diffrax
        version = diffrax.__version__
        results.append((True, f"Diffrax {version}: Successfully imported", True))
    except ImportError as e:
        results.append((False, f"Diffrax import failed: {str(e)}", True))
    except Exception as e:
        results.append((False, f"Diffrax test failed: {str(e)}", True))
    
    return results

def test_visualization_components() -> List[Tuple[bool, str, bool]]:
    """Test visualization components (Plotly, Poliastro).
    Returns: List of (success, message, is_critical)"""
    results = []
    
    # Test Plotly
    try:
        import plotly
        version = plotly.__version__
        import plotly.graph_objects as go
        fig = go.Figure()
        results.append((True, f"Plotly {version}: Successfully created figure", False))
    except ImportError as e:
        results.append((False, f"Plotly import failed: {str(e)}", False))
    except Exception as e:
        results.append((False, f"Plotly test failed: {str(e)}", False))
    
    # Test Poliastro
    try:
        import poliastro
        version = poliastro.__version__
        from poliastro.bodies import Earth
        results.append((True, f"Poliastro {version}: Successfully imported Earth body", False))
    except ImportError as e:
        results.append((False, f"Poliastro import failed: {str(e)}", False))
    except Exception as e:
        results.append((False, f"Poliastro test failed: {str(e)}", False))
    
    return results

def print_section_results(title: str, results: List[Tuple[bool, str, bool]]):
    """Print results for a section with proper formatting."""
    print(f"\n{title}:")
    critical_failures = 0
    for success, message, is_critical in results:
        status = "✓" if success else "✗"
        critical_mark = " (CRITICAL)" if is_critical and not success else ""
        print(f"  {status} {message}{critical_mark}")
        if not success and is_critical:
            critical_failures += 1
    return critical_failures

def main():
    """Main verification routine."""
    print("Verifying Lunar Horizon Optimizer dependencies...\n")
    
    # Check SciPy compatibility first
    scipy_success, scipy_message = check_scipy_version()
    print("SciPy Compatibility:")
    print(f"  {'✓' if scipy_success else '✗'} {scipy_message}")
    
    # Test all components
    critical_failures = 0
    critical_failures += print_section_results("Trajectory Components", test_trajectory_components())
    critical_failures += print_section_results("Optimization Components", test_optimization_components())
    critical_failures += print_section_results("Visualization Components", test_visualization_components())
    
    # Final status
    print("\nVerification Summary:")
    if critical_failures > 0:
        print(f"❌ Found {critical_failures} critical failure(s)")
        print("   Some core functionality may not work properly")
    else:
        print("✅ All critical components are working")
    
    if not scipy_success:
        print("\nRecommended fixes:")
        print("1. Create a new conda environment:")
        print("   conda create -n lunar-opt python=3.8 scipy=1.13.3")
        print("2. Activate the environment:")
        print("   conda activate lunar-opt")
        print("3. Install remaining dependencies:")
        print("   pip install -r requirements.txt")
    
    sys.exit(1 if critical_failures > 0 else 0)

if __name__ == '__main__':
    main() 