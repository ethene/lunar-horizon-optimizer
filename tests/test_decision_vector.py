"""Test suite for decision vector with descent parameters."""

import pytest
import numpy as np
from src.optimization.decision_vector import (
    DescentParameters,
    MissionGenome, 
    LunarMissionProblem
)


class TestDescentParameters:
    """Test DescentParameters dataclass."""
    
    def test_default_initialization(self):
        """Test default parameter values."""
        params = DescentParameters()
        
        assert params.burn_time == 300.0
        assert params.thrust == 15000.0
        assert params.isp == 300.0
    
    def test_custom_initialization(self):
        """Test custom parameter values."""
        params = DescentParameters(
            burn_time=600.0,
            thrust=25000.0,
            isp=350.0
        )
        
        assert params.burn_time == 600.0
        assert params.thrust == 25000.0
        assert params.isp == 350.0
    
    def test_validation_positive_burn_time(self):
        """Test burn_time must be positive."""
        with pytest.raises(ValueError, match="burn_time must be positive"):
            DescentParameters(burn_time=-100.0)
            
        with pytest.raises(ValueError, match="burn_time must be positive"):
            DescentParameters(burn_time=0.0)
    
    def test_validation_positive_thrust(self):
        """Test thrust must be positive.""" 
        with pytest.raises(ValueError, match="thrust must be positive"):
            DescentParameters(thrust=-1000.0)
            
        with pytest.raises(ValueError, match="thrust must be positive"):
            DescentParameters(thrust=0.0)
    
    def test_validation_positive_isp(self):
        """Test isp must be positive."""
        with pytest.raises(ValueError, match="isp must be positive"):
            DescentParameters(isp=-200.0)
            
        with pytest.raises(ValueError, match="isp must be positive"):
            DescentParameters(isp=0.0)


class TestMissionGenome:
    """Test MissionGenome dataclass."""
    
    def test_single_mission_genome_creation(self):
        """Test creating genome for single mission."""
        # Decision vector for 1 mission: 4*1 + 2 + 3 = 9 parameters
        x = [
            10000.0,  # epoch
            400.0,    # earth altitude  
            0.0,      # raan
            1000.0,   # payload mass
            100.0,    # lunar altitude (shared)
            5.0,      # transfer time (shared)
            300.0,    # descent burn time
            15000.0,  # descent thrust
            300.0     # descent isp
        ]
        
        genome = MissionGenome.from_decision_vector(x, num_missions=1)
        
        # Verify base genome properties
        assert genome.num_missions == 1
        assert genome.base_genome.epochs == [10000.0]
        assert genome.base_genome.parking_altitudes == [400.0]
        assert genome.base_genome.plane_raan == [0.0]
        assert genome.base_genome.payload_masses == [1000.0]
        assert genome.base_genome.lunar_altitude == 100.0
        assert genome.base_genome.transfer_time == 5.0
        
        # Verify descent parameters
        assert genome.descent.burn_time == 300.0
        assert genome.descent.thrust == 15000.0
        assert genome.descent.isp == 300.0
    
    def test_multi_mission_genome_creation(self):
        """Test creating genome for multiple missions."""
        # Decision vector for 3 missions: 4*3 + 2 + 3 = 17 parameters
        x = [
            # Epochs
            10000.0, 10050.0, 10100.0,
            # Earth altitudes
            400.0, 500.0, 600.0,
            # RAAN values
            0.0, 120.0, 240.0,
            # Payload masses
            1000.0, 1200.0, 800.0,
            # Shared orbital parameters
            100.0,  # lunar altitude
            5.0,    # transfer time
            # Descent parameters (shared)
            400.0,   # burn time
            20000.0, # thrust
            320.0    # isp
        ]
        
        genome = MissionGenome.from_decision_vector(x, num_missions=3)
        
        # Verify multi-mission structure
        assert genome.num_missions == 3
        assert genome.base_genome.epochs == [10000.0, 10050.0, 10100.0]
        assert genome.base_genome.parking_altitudes == [400.0, 500.0, 600.0]
        assert genome.base_genome.plane_raan == [0.0, 120.0, 240.0]
        assert genome.base_genome.payload_masses == [1000.0, 1200.0, 800.0]
        
        # Verify shared parameters
        assert genome.base_genome.lunar_altitude == 100.0
        assert genome.base_genome.transfer_time == 5.0
        assert genome.descent.burn_time == 400.0
        assert genome.descent.thrust == 20000.0
        assert genome.descent.isp == 320.0
    
    def test_invalid_decision_vector_length(self):
        """Test error handling for incorrect decision vector length."""
        # Too short vector
        x_short = [10000.0, 400.0, 0.0]  # Only 3 parameters
        
        with pytest.raises(ValueError, match="Decision vector length"):
            MissionGenome.from_decision_vector(x_short, num_missions=1)
        
        # Too long vector  
        x_long = [10000.0] * 20  # Too many parameters
        
        with pytest.raises(ValueError, match="Decision vector length"):
            MissionGenome.from_decision_vector(x_long, num_missions=1)
    
    def test_to_decision_vector_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        # Create original decision vector
        original_x = [
            10000.0, 10050.0,  # epochs
            400.0, 500.0,      # altitudes
            0.0, 180.0,        # raan
            1000.0, 1200.0,    # masses
            100.0, 5.0,        # shared orbital
            350.0, 18000.0, 310.0  # descent params
        ]
        
        # Decode to genome and encode back
        genome = MissionGenome.from_decision_vector(original_x, num_missions=2)
        reconstructed_x = genome.to_decision_vector()
        
        # Verify roundtrip accuracy
        np.testing.assert_array_almost_equal(original_x, reconstructed_x)
    
    def test_get_mission_parameters(self):
        """Test getting parameters for individual missions."""
        x = [
            10000.0, 10100.0,  # epochs
            400.0, 600.0,      # altitudes  
            0.0, 180.0,        # raan
            1000.0, 1500.0,    # masses
            150.0, 6.0,        # shared orbital
            450.0, 22000.0, 340.0  # descent
        ]
        
        genome = MissionGenome.from_decision_vector(x, num_missions=2)
        
        # Test first mission parameters
        params_0 = genome.get_mission_parameters(0)
        assert params_0["epoch"] == 10000.0
        assert params_0["earth_orbit_alt"] == 400.0
        assert params_0["plane_raan"] == 0.0
        assert params_0["payload_mass"] == 1000.0
        assert params_0["moon_orbit_alt"] == 150.0
        assert params_0["transfer_time"] == 6.0
        assert params_0["descent_burn_time"] == 450.0
        assert params_0["descent_thrust"] == 22000.0
        assert params_0["descent_isp"] == 340.0
        
        # Test second mission parameters
        params_1 = genome.get_mission_parameters(1)
        assert params_1["epoch"] == 10100.0
        assert params_1["earth_orbit_alt"] == 600.0
        assert params_1["plane_raan"] == 180.0
        assert params_1["payload_mass"] == 1500.0
        # Shared parameters should be same
        assert params_1["moon_orbit_alt"] == 150.0
        assert params_1["transfer_time"] == 6.0
        assert params_1["descent_burn_time"] == 450.0
        assert params_1["descent_thrust"] == 22000.0
        assert params_1["descent_isp"] == 340.0


class TestLunarMissionProblem:
    """Test LunarMissionProblem with descent parameters."""
    
    def test_problem_initialization(self):
        """Test problem initialization with descent bounds."""
        problem = LunarMissionProblem(
            num_missions=2,
            min_burn_time=120.0,
            max_burn_time=800.0,
            min_thrust=2000.0,
            max_thrust=40000.0,
            min_isp=250.0,
            max_isp=400.0
        )
        
        assert problem.num_missions == 2
        assert problem.min_burn_time == 120.0
        assert problem.max_burn_time == 800.0
        assert problem.min_thrust == 2000.0
        assert problem.max_thrust == 40000.0
        assert problem.min_isp == 250.0
        assert problem.max_isp == 400.0
    
    def test_get_bounds_single_mission(self):
        """Test bounds for single mission with descent parameters."""
        problem = LunarMissionProblem(
            num_missions=1,
            min_epoch=9000.0, max_epoch=11000.0,
            min_earth_alt=200.0, max_earth_alt=1000.0,
            min_moon_alt=50.0, max_moon_alt=500.0,
            min_transfer_time=3.0, max_transfer_time=10.0,
            min_payload=500.0, max_payload=2000.0,
            min_burn_time=100.0, max_burn_time=1000.0,
            min_thrust=1000.0, max_thrust=50000.0,
            min_isp=200.0, max_isp=450.0
        )
        
        lower, upper = problem.get_bounds()
        
        # Expected length: 4*1 + 2 + 3 = 9 parameters
        assert len(lower) == 9
        assert len(upper) == 9
        
        # Check bounds structure
        expected_lower = [
            9000.0,   # epoch
            200.0,    # earth altitude
            0.0,      # raan
            500.0,    # payload
            50.0,     # lunar altitude
            3.0,      # transfer time
            100.0,    # burn time
            1000.0,   # thrust
            200.0     # isp
        ]
        
        expected_upper = [
            11000.0,  # epoch
            1000.0,   # earth altitude
            360.0,    # raan
            2000.0,   # payload
            500.0,    # lunar altitude
            10.0,     # transfer time
            1000.0,   # burn time
            50000.0,  # thrust
            450.0     # isp
        ]
        
        np.testing.assert_array_equal(lower, expected_lower)
        np.testing.assert_array_equal(upper, expected_upper)
    
    def test_get_bounds_multi_mission(self):
        """Test bounds for multiple missions."""
        problem = LunarMissionProblem(num_missions=3)
        lower, upper = problem.get_bounds()
        
        # Expected length: 4*3 + 2 + 3 = 17 parameters
        assert len(lower) == 17
        assert len(upper) == 17
        
        # Check that mission-specific bounds are repeated correctly
        # Epochs (3 missions)
        assert lower[0:3] == [9000.0, 9000.0, 9000.0]
        assert upper[0:3] == [11000.0, 11000.0, 11000.0]
        
        # Earth altitudes (3 missions)
        assert lower[3:6] == [200.0, 200.0, 200.0]
        assert upper[3:6] == [1000.0, 1000.0, 1000.0]
        
        # Descent parameters at end (shared)
        assert lower[-3:] == [100.0, 1000.0, 200.0]  # burn_time, thrust, isp
        assert upper[-3:] == [1000.0, 50000.0, 450.0]
    
    def test_decode_valid_parameters(self):
        """Test decoding valid decision vector."""
        problem = LunarMissionProblem(num_missions=1)
        
        # Valid decision vector within bounds
        x = [
            10000.0,  # epoch (within 9000-11000)
            400.0,    # earth alt (within 200-1000)
            45.0,     # raan (within 0-360)
            1200.0,   # payload (within 500-2000)
            100.0,    # lunar alt (within 50-500)
            5.0,      # transfer time (within 3-10)
            300.0,    # burn time (within 100-1000)
            15000.0,  # thrust (within 1000-50000)
            320.0     # isp (within 200-450)
        ]
        
        genome = problem.decode(x)
        
        assert isinstance(genome, MissionGenome)
        assert genome.num_missions == 1
        assert genome.descent.burn_time == 300.0
        assert genome.descent.thrust == 15000.0
        assert genome.descent.isp == 320.0
    
    def test_decode_invalid_parameters(self):
        """Test decoding invalid decision vector raises error."""
        problem = LunarMissionProblem(num_missions=1)
        
        # Invalid decision vector - burn_time too large
        x = [
            10000.0, 400.0, 45.0, 1200.0, 100.0, 5.0,
            1500.0,   # burn_time > max_burn_time (1000)
            15000.0, 320.0
        ]
        
        with pytest.raises(ValueError, match="exceed defined bounds"):
            problem.decode(x)
    
    def test_validate_bounds_valid_genome(self):
        """Test bound validation for valid genome."""
        problem = LunarMissionProblem(num_missions=1)
        
        # Create valid genome
        x = [10000.0, 400.0, 45.0, 1200.0, 100.0, 5.0, 300.0, 15000.0, 320.0]
        genome = MissionGenome.from_decision_vector(x, num_missions=1)
        
        assert problem._validate_bounds(genome) == True
    
    def test_validate_bounds_invalid_descent_params(self):
        """Test bound validation for invalid descent parameters."""
        problem = LunarMissionProblem(
            num_missions=1,
            min_thrust=5000.0,  # Set higher minimum
            max_thrust=30000.0
        )
        
        # Create genome with thrust below minimum
        x = [10000.0, 400.0, 45.0, 1200.0, 100.0, 5.0, 300.0, 2000.0, 320.0]
        genome = MissionGenome.from_decision_vector(x, num_missions=1)
        
        assert problem._validate_bounds(genome) == False
    
    def test_validate_bounds_invalid_base_params(self):
        """Test bound validation for invalid base parameters."""
        problem = LunarMissionProblem(num_missions=1)
        
        # Create genome with lunar altitude above maximum
        x = [10000.0, 400.0, 45.0, 1200.0, 600.0, 5.0, 300.0, 15000.0, 320.0]
        genome = MissionGenome.from_decision_vector(x, num_missions=1)
        
        assert problem._validate_bounds(genome) == False


class TestIntegrationExamples:
    """Test integration patterns and examples."""
    
    def test_realistic_mission_scenario(self):
        """Test realistic lunar mission scenario with descent."""
        # Realistic mission parameters
        problem = LunarMissionProblem(
            num_missions=2,
            min_burn_time=180.0,    # 3 minutes minimum descent
            max_burn_time=600.0,    # 10 minutes maximum descent
            min_thrust=10000.0,     # 10 kN minimum for landing
            max_thrust=30000.0,     # 30 kN maximum thrust
            min_isp=280.0,          # Realistic chemical propulsion
            max_isp=380.0
        )
        
        # Realistic decision vector for two missions
        x = [
            # Mission epochs (50 days apart)
            10000.0, 10050.0,
            # Earth parking orbits (LEO)
            400.0, 450.0,
            # Orbital planes (180° apart for coverage)
            0.0, 180.0,
            # Payload masses (1-1.5 tons)
            1000.0, 1500.0,
            # Shared lunar orbit (100 km altitude)
            100.0,
            # Transfer time (5 days)
            5.0,
            # Descent parameters (realistic)
            300.0,    # 5-minute powered descent
            18000.0,  # 18 kN thrust
            320.0     # 320s specific impulse
        ]
        
        # Decode and validate
        genome = problem.decode(x)
        
        # Verify realistic parameters
        assert genome.num_missions == 2
        assert 180.0 <= genome.descent.burn_time <= 600.0
        assert 10000.0 <= genome.descent.thrust <= 30000.0
        assert 280.0 <= genome.descent.isp <= 380.0
        
        # Check mission separation
        mission_0 = genome.get_mission_parameters(0)
        mission_1 = genome.get_mission_parameters(1)
        
        epoch_separation = abs(mission_1["epoch"] - mission_0["epoch"])
        plane_separation = abs(mission_1["plane_raan"] - mission_0["plane_raan"])
        
        assert epoch_separation == 50.0  # 50 days between launches
        assert plane_separation == 180.0  # Opposite orbital planes
    
    def test_decision_vector_structure_comment(self):
        """Test that decision vector follows documented structure."""
        # Test the structure described in module docstring
        num_missions = 3
        
        # Create decision vector following documented structure
        x = (
            # Mission timing (K values)
            [10000.0, 10030.0, 10060.0] +
            # Earth parking altitudes (K values)  
            [400.0, 500.0, 600.0] +
            # Orbital plane orientations (K values)
            [0.0, 120.0, 240.0] +
            # Payload masses (K values)
            [1000.0, 1200.0, 800.0] +
            # Shared orbital parameters (2 values)
            [100.0, 5.0] +
            # Shared descent parameters (3 values)
            [300.0, 15000.0, 300.0]
        )
        
        # Verify total length matches formula: 4*K + 5
        expected_length = 4 * num_missions + 5
        assert len(x) == expected_length
        assert len(x) == 17  # 4*3 + 5 = 17
        
        # Verify structure can be decoded
        genome = MissionGenome.from_decision_vector(x, num_missions)
        assert genome.num_missions == 3
        
        # Verify all missions have unique timing and planes
        epochs = genome.base_genome.epochs
        raans = genome.base_genome.plane_raan
        
        assert len(set(epochs)) == 3  # All unique epochs
        assert len(set(raans)) == 3   # All unique planes