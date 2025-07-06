"""Tests for mission configuration data models."""

import pytest
from pydantic import ValidationError
from src.config.mission_config import (
    MissionConfig,
    PayloadSpecification,
    CostFactors,
    IsruTarget,
    IsruResourceType
)

def test_valid_mission_config():
    """Test creation of a valid mission configuration."""
    config = MissionConfig(
        name="Test Mission",
        description="A test lunar mission",
        payload=PayloadSpecification(
            dry_mass=1000.0,
            payload_mass=500.0,
            max_propellant_mass=2000.0,
            specific_impulse=300.0
        ),
        cost_factors=CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=50000.0,
            development_cost=1000000.0
        ),
        isru_targets=[
            IsruTarget(
                resource_type=IsruResourceType.WATER,
                target_rate=10.0,
                setup_time_days=30.0,
                market_value_per_kg=1000.0
            )
        ],
        mission_duration_days=180.0,
        target_orbit={
            "semi_major_axis": 384400.0,  # km
            "eccentricity": 0.1,
            "inclination": 45.0
        }
    )
    
    assert config.name == "Test Mission"
    assert config.payload.dry_mass == 1000.0
    assert len(config.isru_targets) == 1
    assert config.isru_targets[0].resource_type == IsruResourceType.WATER

def test_invalid_payload_mass():
    """Test validation of payload mass against dry mass."""
    with pytest.raises(ValidationError) as exc_info:
        PayloadSpecification(
            dry_mass=1000.0,
            payload_mass=1500.0,  # Greater than dry mass
            max_propellant_mass=2000.0,
            specific_impulse=300.0
        )
    assert "Value error, Payload mass" in str(exc_info.value)
    assert "must be less than dry mass" in str(exc_info.value)

def test_invalid_orbit_parameters():
    """Test validation of orbit parameters."""
    with pytest.raises(ValidationError) as exc_info:
        MissionConfig(
            name="Test Mission",
            payload=PayloadSpecification(
                dry_mass=1000.0,
                payload_mass=500.0,
                max_propellant_mass=2000.0,
                specific_impulse=300.0
            ),
            cost_factors=CostFactors(
                launch_cost_per_kg=10000.0,
                operations_cost_per_day=50000.0,
                development_cost=1000000.0
            ),
            mission_duration_days=180.0,
            target_orbit={
                "semi_major_axis": 384400.0,
                # Missing eccentricity and inclination
            }
        )
    assert "Field required" in str(exc_info.value)

def test_invalid_eccentricity():
    """Test validation of orbit eccentricity."""
    with pytest.raises(ValidationError) as exc_info:
        MissionConfig(
            name="Test Mission",
            payload=PayloadSpecification(
                dry_mass=1000.0,
                payload_mass=500.0,
                max_propellant_mass=2000.0,
                specific_impulse=300.0
            ),
            cost_factors=CostFactors(
                launch_cost_per_kg=10000.0,
                operations_cost_per_day=50000.0,
                development_cost=1000000.0
            ),
            mission_duration_days=180.0,
            target_orbit={
                "semi_major_axis": 384400.0,
                "eccentricity": 1.5,  # Invalid: must be < 1
                "inclination": 45.0
            }
        )
    assert "Input should be less than 1" in str(exc_info.value)

def test_cost_factors_validation():
    """Test validation of cost factors."""
    with pytest.raises(ValidationError) as exc_info:
        CostFactors(
            launch_cost_per_kg=-1000.0,  # Invalid: must be positive
            operations_cost_per_day=50000.0,
            development_cost=1000000.0
        )
    
    assert "greater than 0" in str(exc_info.value)

def test_isru_target_validation():
    """Test validation of ISRU targets."""
    with pytest.raises(ValidationError) as exc_info:
        IsruTarget(
            resource_type=IsruResourceType.WATER,
            target_rate=-10.0,  # Invalid: must be positive
            setup_time_days=30.0,
            market_value_per_kg=1000.0
        )
    
    assert "greater than 0" in str(exc_info.value)

def test_optional_description():
    """Test that description is optional."""
    config = MissionConfig(
        name="Test Mission",
        payload=PayloadSpecification(
            dry_mass=1000.0,
            payload_mass=500.0,
            max_propellant_mass=2000.0,
            specific_impulse=300.0
        ),
        cost_factors=CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=50000.0,
            development_cost=1000000.0
        ),
        mission_duration_days=180.0,
        target_orbit={
            "semi_major_axis": 384400.0,
            "eccentricity": 0.1,
            "inclination": 45.0
        }
    )
    
    assert config.description is None

def test_empty_isru_targets():
    """Test that ISRU targets can be empty."""
    config = MissionConfig(
        name="Test Mission",
        payload=PayloadSpecification(
            dry_mass=1000.0,
            payload_mass=500.0,
            max_propellant_mass=2000.0,
            specific_impulse=300.0
        ),
        cost_factors=CostFactors(
            launch_cost_per_kg=10000.0,
            operations_cost_per_day=50000.0,
            development_cost=1000000.0
        ),
        mission_duration_days=180.0,
        target_orbit={
            "semi_major_axis": 384400.0,
            "eccentricity": 0.1,
            "inclination": 45.0
        }
    )
    
    assert len(config.isru_targets) == 0 