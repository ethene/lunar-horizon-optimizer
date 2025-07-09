"""Tests for configuration loader functionality."""

import json
import yaml
import pytest
from pathlib import Path
from config.loader import ConfigLoader, ConfigurationError
from config.models import MissionConfig

@pytest.fixture
def valid_config_dict():
    """Fixture providing a valid configuration dictionary."""
    return {
        "name": "Test Mission",
        "description": "A test lunar mission",
        "payload": {
            "dry_mass": 1000.0,
            "payload_mass": 500.0,
            "max_propellant_mass": 2000.0,
            "specific_impulse": 300.0
        },
        "cost_factors": {
            "launch_cost_per_kg": 10000.0,
            "operations_cost_per_day": 50000.0,
            "development_cost": 1000000.0
        },
        "mission_duration_days": 180.0,
        "target_orbit": {
            "semi_major_axis": 384400.0,
            "eccentricity": 0.1,
            "inclination": 45.0
        }
    }

@pytest.fixture
def temp_json_config(tmp_path: Path, valid_config_dict):
    """Fixture creating a temporary JSON configuration file."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(valid_config_dict, f)
    return config_file

@pytest.fixture
def temp_yaml_config(tmp_path: Path, valid_config_dict):
    """Fixture creating a temporary YAML configuration file."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(valid_config_dict, f)
    return config_file

def test_load_json_config(temp_json_config):
    """Test loading a valid JSON configuration file."""
    loader = ConfigLoader()
    config = loader.load_file(temp_json_config)
    assert isinstance(config, MissionConfig)
    assert config.name == "Test Mission"
    assert config.payload.dry_mass == 1000.0

def test_load_yaml_config(temp_yaml_config):
    """Test loading a valid YAML configuration file."""
    loader = ConfigLoader()
    config = loader.load_file(temp_yaml_config)
    assert isinstance(config, MissionConfig)
    assert config.name == "Test Mission"
    assert config.payload.dry_mass == 1000.0

def test_load_nonexistent_file():
    """Test loading a non-existent file."""
    loader = ConfigLoader()
    with pytest.raises(ConfigurationError) as exc_info:
        loader.load_file("nonexistent.json")
    assert "not found" in str(exc_info.value)

def test_load_invalid_format(tmp_path):
    """Test loading a file with unsupported format."""
    invalid_file = tmp_path / "config.txt"
    invalid_file.touch()

    loader = ConfigLoader()
    with pytest.raises(ConfigurationError) as exc_info:
        loader.load_file(invalid_file)
    assert "Unsupported file format" in str(exc_info.value)

def test_load_invalid_json(tmp_path):
    """Test loading an invalid JSON file."""
    config_file = tmp_path / "invalid.json"
    with open(config_file, "w") as f:
        f.write("invalid json content")

    loader = ConfigLoader()
    with pytest.raises(ConfigurationError) as exc_info:
        loader.load_file(config_file)
    assert "Invalid JSON" in str(exc_info.value)

def test_load_invalid_yaml(tmp_path):
    """Test loading an invalid YAML file."""
    config_file = tmp_path / "invalid.yaml"
    with open(config_file, "w") as f:
        f.write("invalid:\nyaml:\ncontent:\n  - [}")

    loader = ConfigLoader()
    with pytest.raises(ConfigurationError) as exc_info:
        loader.load_file(config_file)
    assert "Invalid YAML" in str(exc_info.value)

def test_merge_with_defaults(valid_config_dict):
    """Test merging loaded config with defaults."""
    default_config = {
        "name": "Default Mission",
        "description": "Default description",
        "extra_field": "default value"
    }

    loader = ConfigLoader(default_config=default_config)
    merged = loader._merge_with_defaults(valid_config_dict)

    assert merged["name"] == "Test Mission"  # Overridden by loaded config
    assert merged["extra_field"] == "default value"  # Preserved from defaults

def test_save_config_json(tmp_path, valid_config_dict):
    """Test saving configuration to JSON file."""
    config = MissionConfig(**valid_config_dict)
    output_file = tmp_path / "output.json"

    loader = ConfigLoader()
    loader.save_config(config, output_file)

    assert output_file.exists()
    with open(output_file) as f:
        saved_config = json.load(f)
    assert saved_config["name"] == "Test Mission"

def test_save_config_yaml(tmp_path, valid_config_dict):
    """Test saving configuration to YAML file."""
    config = MissionConfig(**valid_config_dict)
    output_file = tmp_path / "output.yaml"

    loader = ConfigLoader()
    loader.save_config(config, output_file)

    assert output_file.exists()
    with open(output_file) as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["name"] == "Test Mission"

def test_load_default_config():
    """Test creating loader with default configuration."""
    loader = ConfigLoader.load_default_config()
    assert isinstance(loader, ConfigLoader)
    assert loader.default_config["name"] == "Default Lunar Mission"

    # Test loading partial config with defaults
    partial_config = {
        "name": "Custom Mission",
        "payload": {
            "dry_mass": 1500.0,
            "payload_mass": 750.0,
            "max_propellant_mass": 2500.0,
            "specific_impulse": 310.0
        }
    }

    # Save and load partial config
    tmp_path = Path("tests/temp")
    tmp_path.mkdir(exist_ok=True)
    config_file = tmp_path / "partial.json"
    with open(config_file, "w") as f:
        json.dump(partial_config, f)

    config = loader.load_file(config_file)
    assert config.name == "Custom Mission"  # From partial config
    assert config.mission_duration_days == 180.0  # From defaults

    # Cleanup
    config_file.unlink()
    tmp_path.rmdir()
