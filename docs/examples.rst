Examples and Tutorials
======================

This section provides comprehensive examples demonstrating the Lunar Horizon Optimizer's capabilities.

Quick Start Example
===================

The fastest way to get started with the Lunar Horizon Optimizer:

.. code-block:: bash

    conda activate py312
    python examples/quick_start.py

This example demonstrates:

* Mission configuration setup
* Trajectory generation with Lambert solvers  
* Multi-objective optimization with PyGMO
* Economic analysis (NPV, IRR, ROI)
* Interactive visualization dashboards

Working Example
===============

A simplified example focusing on core functionality:

.. code-block:: bash

    python examples/working_example.py

Features demonstrated:

* Basic trajectory generation
* Simple optimization
* Core visualization
* Streamlined workflow

Integration Examples
====================

Advanced examples showing system integration:

Trajectory Integration
----------------------

.. code-block:: bash

    python examples/simple_trajectory_test.py

* Lambert solver integration
* Trajectory data structures
* Visualization compatibility
* Error handling patterns

Comprehensive Integration  
-------------------------

.. code-block:: bash

    python examples/final_integration_test.py

* Complete system integration validation
* Cross-module workflow automation
* Performance optimization
* PRD compliance measurement

Configuration Examples
=======================

Sample Configuration File
--------------------------

The ``examples/configs/basic_mission.yaml`` file provides a template for mission configuration:

.. code-block:: yaml

    mission:
      name: "Sample Lunar Mission"
      duration_days: 180
      
    spacecraft:
      dry_mass_kg: 1000
      fuel_capacity_kg: 500
      
    trajectory:
      earth_orbit_altitude_km: 400
      lunar_orbit_altitude_km: 100
      transfer_time_days: 3.5
      
    economics:
      launch_cost_per_kg: 10000
      mission_cost_fixed: 50000000
      discount_rate: 0.08

Loading Configuration
---------------------

.. code-block:: python

    from src.config.loader import ConfigLoader
    
    # Load configuration
    config = ConfigLoader.load_yaml('examples/configs/basic_mission.yaml')
    
    # Validate configuration
    validated_config = MissionConfig(**config)

Performance Benchmarks
=======================

Example Runtime Performance
----------------------------

+------------------------+----------------+----------------+----------------+
| Example                | Trajectory Gen | Optimization   | Total Runtime  |
+========================+================+================+================+
| quick_start.py         | ~2s            | ~20s           | ~30s           |
+------------------------+----------------+----------------+----------------+
| working_example.py     | ~1s            | ~10s           | ~15s           |
+------------------------+----------------+----------------+----------------+
| integration_test.py    | ~5s            | ~40s           | ~60s           |
+------------------------+----------------+----------------+----------------+

Troubleshooting
===============

Common Issues and Solutions
---------------------------

Environment Not Activated
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Import errors or missing dependencies

**Solution**:

.. code-block:: bash

    conda activate py312
    python scripts/verify_dependencies.py

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

**Problem**: PyKEP or PyGMO import failures

**Solution**:

.. code-block:: bash

    conda install -c conda-forge pykep pygmo astropy spiceypy -y
    pip install -r requirements.txt

Performance Issues
~~~~~~~~~~~~~~~~~~

**Problem**: Examples running slowly

**Solution**: Reduce complexity parameters:

.. code-block:: python

    # In example files, reduce these values for faster execution:
    population_size = 20  # Default: 50
    num_generations = 10  # Default: 30

Learning Path
=============

Recommended Learning Sequence
-----------------------------

**Beginner Path**:

1. Read the :doc:`USER_GUIDE` 
2. Run ``quick_start.py``
3. Examine ``working_example.py`` source code
4. Modify parameters and re-run examples

**Intermediate Path**:

1. Study ``simple_trajectory_test.py``
2. Understand ``configs/basic_mission.yaml``
3. Run ``advanced_trajectory_test.py``
4. Create custom configurations

**Advanced Path**:

1. Analyze ``integration_test.py``
2. Study ``final_integration_test.py``
3. Develop custom extensions using the :doc:`task_10_extensibility_documentation`
4. Contribute to the project

Next Steps
==========

After running the examples:

* Explore the :doc:`api_reference` for detailed API documentation
* Read the :doc:`integration_guide` for cross-module integration patterns
* Review :doc:`PROJECT_STATUS` for current development status
* Check :doc:`TESTING_GUIDELINES` for contributing guidelines

For detailed example documentation, see the `examples/README.md <https://github.com/lunar-horizon/optimizer/blob/main/examples/README.md>`_ file.