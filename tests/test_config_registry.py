"""Tests for configuration registry functionality."""

import pytest
from pathlib import Path
import yaml
import json

from src.config.mission_config import MissionConfig
from src.config.registry import ConfigRegistry, ConfigurationError


@pytest.fixture
def registry():
    """Fixture providing a ConfigRegistry instance."""
    return ConfigRegistry()


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


def test_default_templates(registry):
    """Test that default templates are loaded correctly."""
    templates = registry.list_templates()
    assert "lunar_delivery" in templates
    assert "lunar_isru" in templates

    template = registry.get_template("lunar_delivery")
    assert isinstance(template, MissionConfig)
    assert template.name == "Lunar Payload Delivery"


def test_register_template(registry, sample_config):
    """Test registering a new template."""
    registry.register_template("test_template", sample_config)
    assert "test_template" in registry.list_templates()

    loaded = registry.get_template("test_template")
    assert loaded.name == "Test Mission"
    assert loaded.payload.dry_mass == 1000.0


def test_register_duplicate_template(registry, sample_config):
    """Test that registering a duplicate template raises an error."""
    registry.register_template("test", sample_config)
    with pytest.raises(ValueError) as exc_info:
        registry.register_template("test", sample_config)
    assert "already exists" in str(exc_info.value)


def test_get_nonexistent_template(registry):
    """Test that getting a non-existent template raises an error."""
    with pytest.raises(KeyError) as exc_info:
        registry.get_template("nonexistent")
    assert "not found" in str(exc_info.value)


def test_template_isolation(registry, sample_config):
    """Test that templates are properly isolated when copied."""
    registry.register_template("test", sample_config)
    template1 = registry.get_template("test")
    template2 = registry.get_template("test")

    # Modify one copy
    template1.name = "Modified"

    # Original and second copy should be unchanged
    assert registry.get_template("test").name == "Test Mission"
    assert template2.name == "Test Mission"


def test_load_template_file(tmp_path, registry, sample_config):
    """Test loading a template from a file."""
    # Create test files
    json_file = tmp_path / "test.json"
    yaml_file = tmp_path / "test.yaml"

    with open(json_file, "w") as f:
        json.dump(sample_config.model_dump(), f)
    with open(yaml_file, "w") as f:
        yaml.safe_dump(sample_config.model_dump(), f)

    # Test loading both formats
    registry.load_template_file(json_file)
    assert "test" in registry.list_templates()

    registry.load_template_file(yaml_file)
    assert "test" in registry.list_templates()


def test_load_invalid_template_file(tmp_path, registry):
    """Test that loading an invalid template file raises an error."""
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("invalid json content")

    with pytest.raises(ConfigurationError) as exc_info:
        registry.load_template_file(invalid_file)
    assert "Failed to load template" in str(exc_info.value)


def test_load_templates_dir(tmp_path, registry, sample_config):
    """Test loading templates from a directory."""
    # Create test directory with multiple templates
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    for i in range(3):
        config = sample_config.model_copy()
        config.name = f"Template {i}"
        file_path = templates_dir / f"template{i}.json"
        with open(file_path, "w") as f:
            json.dump(config.model_dump(), f)

    registry.load_templates_dir(templates_dir)
    templates = registry.list_templates()
    assert "template0" in templates
    assert "template1" in templates
    assert "template2" in templates


def test_load_nonexistent_templates_dir(registry):
    """Test that loading from a non-existent directory raises an error."""
    with pytest.raises(ConfigurationError) as exc_info:
        registry.load_templates_dir(Path("nonexistent"))
    assert "not found" in str(exc_info.value)


def test_save_template(tmp_path, registry, sample_config):
    """Test saving a template to a file."""
    registry.register_template("test", sample_config)

    # Test saving in both formats
    json_file = tmp_path / "output.json"
    yaml_file = tmp_path / "output.yaml"

    registry.save_template("test", json_file)
    registry.save_template("test", yaml_file)

    assert json_file.exists()
    assert yaml_file.exists()

    # Verify content
    with open(json_file) as f:
        json_data = json.load(f)
    assert json_data["name"] == "Test Mission"

    with open(yaml_file) as f:
        yaml_data = yaml.safe_load(f)
    assert yaml_data["name"] == "Test Mission"


def test_save_nonexistent_template(tmp_path, registry):
    """Test that saving a non-existent template raises an error."""
    with pytest.raises(KeyError) as exc_info:
        registry.save_template("nonexistent", tmp_path / "output.json")
    assert "not found" in str(exc_info.value)
