"""Tests for configuration manager functionality."""

import pytest
import json
import yaml
from pydantic import ValidationError

from config.models import MissionConfig
from config.management.config_manager import ConfigManager
from config.loader import ConfigurationError
from config.registry import ConfigRegistry


@pytest.fixture
def sample_config():
    """Fixture providing a sample mission configuration."""
    return MissionConfig.model_validate(
        {
            "name": "Test Mission",
            "description": "Test configuration",
            "payload": {
                "dry_mass": 1000.0,
                "payload_mass": 500.0,
                "max_propellant_mass": 2000.0,
                "specific_impulse": 300.0,
            },
            "cost_factors": {
                "launch_cost_per_kg": 10000.0,
                "operations_cost_per_day": 50000.0,
                "development_cost": 1000000.0,
            },
            "mission_duration_days": 180.0,
            "target_orbit": {
                "semi_major_axis": 384400.0,
                "eccentricity": 0.1,
                "inclination": 45.0,
            },
        }
    )


@pytest.fixture
def manager():
    """Fixture providing a ConfigManager instance."""
    return ConfigManager()


def test_init_manager():
    """Test manager initialization."""
    manager = ConfigManager()
    assert manager.active_config is None
    assert isinstance(manager.registry, ConfigRegistry)


def test_load_config(tmp_path, manager, sample_config):
    """Test loading a configuration from file."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(sample_config.model_dump(), f)

    config = manager.load_config(config_file)
    assert isinstance(config, MissionConfig)
    assert config.name == "Test Mission"
    assert manager.active_config == config


def test_save_config(tmp_path, manager, sample_config):
    """Test saving a configuration to file."""
    # Set active config
    manager._active_config = sample_config

    # Save in both formats
    json_file = tmp_path / "config.json"
    yaml_file = tmp_path / "config.yaml"

    manager.save_config(json_file)
    manager.save_config(yaml_file)

    # Verify JSON content
    with open(json_file) as f:
        json_data = json.load(f)
    assert json_data["name"] == "Test Mission"

    # Verify YAML content
    with open(yaml_file) as f:
        yaml_data = yaml.safe_load(f)
    assert yaml_data["name"] == "Test Mission"


def test_save_without_active_config(tmp_path, manager):
    """Test that saving without an active configuration raises an error."""
    with pytest.raises(ConfigurationError) as exc_info:
        manager.save_config(tmp_path / "config.json")
    assert "No active configuration" in str(exc_info.value)


def test_create_from_template(manager):
    """Test creating a configuration from a template."""
    config = manager.create_from_template("lunar_delivery", name="Custom Mission")
    assert isinstance(config, MissionConfig)
    assert config.name == "Custom Mission"
    assert manager.active_config == config


def test_create_from_nonexistent_template(manager):
    """Test that creating from a non-existent template raises an error."""
    with pytest.raises(KeyError) as exc_info:
        manager.create_from_template("nonexistent")
    assert "not found" in str(exc_info.value)


def test_validate_config(manager):
    """Test configuration validation."""
    config_dict = {
        "name": "Test Mission",
        "payload": {
            "dry_mass": 1000.0,
            "payload_mass": 500.0,
            "max_propellant_mass": 2000.0,
            "specific_impulse": 300.0,
        },
        "cost_factors": {
            "launch_cost_per_kg": 10000.0,
            "operations_cost_per_day": 50000.0,
            "development_cost": 1000000.0,
        },
        "mission_duration_days": 180.0,
        "target_orbit": {
            "semi_major_axis": 384400.0,
            "eccentricity": 0.0,
            "inclination": 0.0,
        },
    }

    validated = manager.validate_config(config_dict)
    assert isinstance(validated, MissionConfig)
    assert validated.name == "Test Mission"


def test_validate_invalid_config(manager):
    """Test validation of invalid configuration."""
    invalid_config = {
        "name": "Test Mission",
        # Missing required fields
    }

    with pytest.raises(ValidationError):
        manager.validate_config(invalid_config)


def test_update_config(manager, sample_config):
    """Test updating configuration values."""
    manager._active_config = sample_config

    updates = {
        "name": "Updated Mission",
        "payload": {
            "dry_mass": 1500.0,
            "payload_mass": 500.0,
            "max_propellant_mass": 2000.0,
            "specific_impulse": 300.0,
        },
    }

    updated = manager.update_config(updates)
    assert updated.name == "Updated Mission"
    assert updated.payload.dry_mass == 1500.0
    assert manager.active_config == updated


def test_update_without_active_config(manager):
    """Test that updating without an active configuration raises an error."""
    with pytest.raises(ConfigurationError) as exc_info:
        manager.update_config({"name": "Test"})
    assert "No active configuration" in str(exc_info.value)


def test_update_with_invalid_data(manager, sample_config):
    """Test updating with invalid configuration data."""
    manager._active_config = sample_config

    with pytest.raises(ValidationError):
        manager.update_config(
            {"payload": {"dry_mass": -1000.0}}  # Invalid negative mass
        )
