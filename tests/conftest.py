import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config.management.config_manager import ConfigManager

@pytest.fixture
def config_manager():
    """Fixture that provides a ConfigurationManager instance."""
    return ConfigManager()

@pytest.fixture
def sample_config():
    """Fixture that provides a sample valid configuration."""
    return {
        "mission_name": "test_mission",
        "mission_duration": 365,
        "setup_time": 30,
        "target_orbit": {
            "semi_major_axis": 400000,
            "eccentricity": 0.0,
            "inclination": 0.0,
            "raan": 0.0,
            "argument_of_periapsis": 0.0,
            "true_anomaly": 0.0
        },
        "isru_target": {
            "name": "Moon",
            "resource": "water"
        },
        "payload_specification": {
            "dry_mass": 1000,
            "payload_mass": 500,
            "propellant_mass": 200
        }
    }

@pytest.fixture
def invalid_config():
    """Fixture that provides an invalid configuration missing required fields."""
    return {
        "mission_name": "invalid_mission",
        # Missing mission_duration
        "target_orbit": {
            "semi_major_axis": 400000,
            # Missing other orbital parameters
        }
    }

@pytest.fixture
def different_config():
    """Fixture that provides a valid but different configuration for comparison."""
    return {
        "mission_name": "different_mission",
        "mission_duration": 730,  # Different duration
        "setup_time": 30,
        "target_orbit": {
            "semi_major_axis": 500000,  # Different orbit
            "eccentricity": 0.1,
            "inclination": 45.0,
            "raan": 0.0,
            "argument_of_periapsis": 0.0,
            "true_anomaly": 0.0
        },
        "isru_target": {
            "name": "Moon",
            "resource": "water"
        },
        "payload_specification": {
            "dry_mass": 1000,
            "payload_mass": 500,
            "propellant_mass": 200
        }
    }

@pytest.fixture
def missing_fields_config():
    """Fixture that provides a configuration with some fields missing for comparison."""
    return {
        "mission_name": "missing_fields_mission",
        "mission_duration": 365,
        "target_orbit": {
            "semi_major_axis": 400000,
            "eccentricity": 0.0,
            "inclination": 0.0,
            "raan": 0.0,
            "argument_of_periapsis": 0.0,
            "true_anomaly": 0.0
        }
        # Missing isru_target and payload_specification
    } 