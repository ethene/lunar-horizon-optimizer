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

def verify_basic_functionality() -> List[Tuple[bool, str]]:
    """Run basic functionality tests for each library."""
    results = []
    
    # Test PyKEP basic functionality
    try:
        import pykep as pk
        planet = pk.planet.jpl_lp('earth')
        results.append((True, "PyKEP: Successfully created Earth planet object"))
    except Exception as e:
        results.append((False, f"PyKEP functionality test failed: {str(e)}"))
    
    # Test PyGMO basic functionality
    try:
        import pygmo as pg
        prob = pg.problem(pg.rosenbrock())
        results.append((True, "PyGMO: Successfully created optimization problem"))
    except Exception as e:
        results.append((False, f"PyGMO functionality test failed: {str(e)}"))
    
    # Test JAX basic functionality
    try:
        import jax.numpy as jnp
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        results.append((True, "JAX: Successfully performed basic array operation"))
    except Exception as e:
        results.append((False, f"JAX functionality test failed: {str(e)}"))
    
    # Test Diffrax basic functionality
    try:
        import diffrax
        results.append((True, "Diffrax: Successfully imported"))
    except Exception as e:
        results.append((False, f"Diffrax functionality test failed: {str(e)}"))
    
    # Test Plotly basic functionality
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        results.append((True, "Plotly: Successfully created figure object"))
    except Exception as e:
        results.append((False, f"Plotly functionality test failed: {str(e)}"))
    
    # Test Poliastro basic functionality
    try:
        from astropy import units as u
        from poliastro.bodies import Earth
        earth = Earth
        results.append((True, "Poliastro: Successfully imported Earth body"))
    except Exception as e:
        results.append((False, f"Poliastro functionality test failed: {str(e)}"))
    
    return results

def main():
    """Main verification routine."""
    print("Verifying Lunar Horizon Optimizer dependencies...\n")
    
    # Required dependencies with minimum versions
    dependencies = {
        'pykep': '2.6',
        'pygmo': '2.19',
        'jax': '0.4.13',
        'jaxlib': '0.4.13',
        'diffrax': '0.4.0',
        'plotly': '5.18.0',
        'poliastro': '0.17.0'
    }
    
    # Check all imports
    all_success = True
    print("Checking imports:")
    for module, version in dependencies.items():
        success, message = check_import(module, version)
        print(f"  {'✓' if success else '✗'} {message}")
        all_success = all_success and success
    
    # Check JAX GPU support
    print("\nChecking JAX GPU support:")
    success, message = verify_jax_gpu()
    print(f"  {'✓' if success else '✗'} {message}")
    
    # Run basic functionality tests
    print("\nRunning basic functionality tests:")
    functionality_results = verify_basic_functionality()
    for success, message in functionality_results:
        print(f"  {'✓' if success else '✗'} {message}")
        all_success = all_success and success
    
    # Final status
    print("\nVerification ", "PASSED" if all_success else "FAILED")
    sys.exit(0 if all_success else 1)

if __name__ == '__main__':
    main() 