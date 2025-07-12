Lunar Horizon Optimizer Documentation
=====================================

Welcome to the Lunar Horizon Optimizer documentation. This is an integrated differentiable trajectory optimization and economic analysis platform for LEO-Moon missions.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   USER_GUIDE
   examples

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation:

   api_reference
   integration_guide

.. toctree::
   :maxdepth: 2
   :caption: Project Information:

   PROJECT_STATUS
   PRD_COMPLIANCE
   TESTING_GUIDELINES

.. toctree::
   :maxdepth: 1
   :caption: Task Documentation:

   task_3_documentation
   task_4_documentation
   task_5_documentation
   task_6_documentation
   task_7_documentation
   task_10_extensibility_documentation

Project Overview
================

The Lunar Horizon Optimizer is a production-ready platform that combines:

* **Advanced Trajectory Generation** - PyKEP 2.6 Lambert solvers with N-body dynamics
* **Multi-Objective Optimization** - PyGMO 2.19.6 NSGA-II algorithms 
* **Differentiable Optimization** - JAX 0.5.3 and Diffrax 0.7.0 integration
* **Economic Analysis** - NPV, IRR, ROI with ISRU benefits modeling
* **Interactive Visualization** - Plotly 5.24.1 dashboards and 3D plots
* **Extensible Architecture** - Plugin system for custom functionality

Key Features
============

🛰️ **Trajectory Generation**
   High-fidelity orbital mechanics with Lambert solver integration, multi-body dynamics, and patched conics approximation.

🎯 **Global Optimization**
   Multi-objective optimization using PyGMO with NSGA-II algorithms for Pareto front analysis.

💰 **Economic Analysis**
   Comprehensive financial modeling including NPV, IRR, ROI calculations with ISRU benefits analysis.

📊 **Interactive Visualization**
   Real-time 3D trajectory plots, economic dashboards, and Pareto front exploration.

🔧 **System Integration**
   Unified configuration management, cross-module workflows, and automated pipeline processing.

Installation
============

1. **Environment Setup**::

    conda create -n py312 python=3.12 -y
    conda activate py312

2. **Core Dependencies**::

    conda install -c conda-forge pykep pygmo astropy spiceypy -y

3. **Additional Dependencies**::

    pip install -r requirements.txt

4. **Verification**::

    python scripts/verify_dependencies.py

Quick Start
===========

.. code-block:: python

    from src.lunar_horizon_optimizer import LunarHorizonOptimizer
    
    # Initialize the optimizer
    optimizer = LunarHorizonOptimizer()
    
    # Run complete analysis
    results = optimizer.run_complete_analysis()
    
    # Generate dashboards
    optimizer.create_comprehensive_dashboard(results)

Development Status
==================

✅ **All 10 Tasks Complete** - Full implementation achieved  
✅ **Production Ready** - 415 tests with 100% production core pass rate  
✅ **Clean Pipeline** - 0 linting errors, comprehensive quality checks  
✅ **Documentation Complete** - User guides, API docs, and examples  

Coverage Status
===============

* **Current Coverage**: 50% (4,958 lines covered / 9,619 total lines)
* **Production Tests**: 38 tests with 100% pass rate (required for commits)
* **Total Test Suite**: 458 tests across all modules
* **Target Coverage**: 80% (continuous improvement in progress)

Architecture
============

The system follows a modular architecture with clear separation of concerns:

* ``src/config/`` - Mission configuration and parameter management
* ``src/trajectory/`` - Orbital mechanics and trajectory calculations  
* ``src/optimization/`` - Global and differentiable optimization
* ``src/economics/`` - Economic analysis and financial modeling
* ``src/visualization/`` - Interactive dashboards and plotting
* ``src/extensibility/`` - Plugin system and extension framework

Performance
===========

Typical performance benchmarks:

* **Trajectory Generation**: ~2-5s (Lambert solver + propagation)
* **Global Optimization**: ~15-60s (population size dependent)  
* **Economic Analysis**: ~1-3s (NPV/IRR + sensitivity analysis)
* **Visualization**: ~2-5s (interactive Plotly dashboards)
* **Complete Pipeline**: ~30-90s (full mission analysis)

Contributing
============

We welcome contributions! Please see our development guidelines:

* Follow the NO MOCKING rule - use real implementations only
* Ensure ``make test`` passes (100% required for commits)
* Run ``make pipeline`` for quality checks
* Update relevant documentation
* Use clear, descriptive commit messages

License
=======

This project is licensed under the MIT License. See the LICENSE file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`