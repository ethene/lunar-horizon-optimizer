"""Trajectory validation module.

This module provides validation functionality for lunar transfer trajectories,
including input parameter validation and delta-v constraints checking.

Example:
    ```python
    validator = TrajectoryValidator(
        min_earth_alt=200,
        max_earth_alt=1000,
        min_moon_alt=50,
        max_moon_alt=500
    )
    
    # Validate trajectory parameters
    validator.validate_inputs(
        earth_orbit_alt=300,
        moon_orbit_alt=100,
        transfer_time=3.5
    )
    
    # Validate delta-v values
    validator.validate_delta_v(
        tli_dv=3200,  # m/s
        loi_dv=850    # m/s
    )
    ```
"""

class TrajectoryValidator:
    """Validates trajectory parameters and constraints.
    
    This class handles validation of all trajectory-related parameters including:
    - Orbit altitudes (Earth and Moon)
    - Transfer time
    - Delta-v magnitudes
    
    Attributes:
        min_earth_alt (float): Minimum Earth parking orbit altitude [m]
        max_earth_alt (float): Maximum Earth parking orbit altitude [m]
        min_moon_alt (float): Minimum lunar orbit altitude [m]
        max_moon_alt (float): Maximum lunar orbit altitude [m]
        min_transfer_time (float): Minimum transfer time [days]
        max_transfer_time (float): Maximum transfer time [days]
    """
    
    def __init__(self,
                min_earth_alt: float = 200,
                max_earth_alt: float = 1000,
                min_moon_alt: float = 50,
                max_moon_alt: float = 500,
                min_transfer_time: float = 2.0,
                max_transfer_time: float = 7.0):
        """Initialize validator with constraints.
        
        Args:
            min_earth_alt: Minimum Earth parking orbit altitude [km]
            max_earth_alt: Maximum Earth parking orbit altitude [km]
            min_moon_alt: Minimum lunar orbit altitude [km]
            max_moon_alt: Maximum lunar orbit altitude [km]
            min_transfer_time: Minimum transfer time [days]
            max_transfer_time: Maximum transfer time [days]
            
        Note:
            All altitude inputs are in kilometers but stored internally in meters
        """
        self.min_earth_alt = min_earth_alt * 1000  # Convert to meters
        self.max_earth_alt = max_earth_alt * 1000
        self.min_moon_alt = min_moon_alt * 1000
        self.max_moon_alt = max_moon_alt * 1000
        self.min_transfer_time = min_transfer_time
        self.max_transfer_time = max_transfer_time
        
    def validate_inputs(self,
                      earth_orbit_alt: float,
                      moon_orbit_alt: float,
                      transfer_time: float) -> None:
        """Validate input parameters for trajectory generation.
        
        Performs comprehensive validation of all input parameters including:
        - Earth orbit altitude within allowed range
        - Moon orbit altitude within allowed range
        - Transfer time within allowed range
        
        Args:
            earth_orbit_alt: Initial parking orbit altitude [km]
            moon_orbit_alt: Final lunar orbit altitude [km]
            transfer_time: Transfer time [days]
            
        Raises:
            ValueError: If any parameter is outside its allowed range
            
        Note:
            All altitude inputs should be in kilometers
        """
        if transfer_time <= 0:
            raise ValueError("Transfer time must be positive")

        if not (self.min_moon_alt/1000 <= moon_orbit_alt <= self.max_moon_alt/1000):
            raise ValueError(
                f"Moon orbit altitude must be between {self.min_moon_alt/1000:.1f} "
                f"and {self.max_moon_alt/1000:.1f} km"
            )

        if not (self.min_earth_alt/1000 <= earth_orbit_alt <= self.max_earth_alt/1000):
            raise ValueError(
                f"Earth orbit altitude must be between {self.min_earth_alt/1000:.1f} "
                f"and {self.max_earth_alt/1000:.1f} km"
            )
            
        if not (self.min_transfer_time <= transfer_time <= self.max_transfer_time):
            raise ValueError(
                f"Transfer time must be between {self.min_transfer_time} "
                f"and {self.max_transfer_time} days"
            )
            
    def validate_delta_v(self, tli_dv: float, loi_dv: float) -> None:
        """Validate delta-v values against typical mission constraints.
        
        Checks if the Trans-Lunar Injection (TLI) and Lunar Orbit Insertion (LOI)
        delta-v values are within typical mission constraints. These constraints
        are based on historical lunar missions and practical engineering limits.
        
        Args:
            tli_dv: Trans-lunar injection delta-v [m/s]
            loi_dv: Lunar orbit insertion delta-v [m/s]
            
        Raises:
            ValueError: If either delta-v exceeds its typical limit
            
        Note:
            Typical limits are:
            - TLI: 3500 m/s
            - LOI: 1200 m/s
        """
        if tli_dv > 3500:  # Typical TLI delta-v limit
            raise ValueError(f"TLI delta-v {tli_dv:.1f} m/s exceeds limit of 3500 m/s")
        if loi_dv > 1200:  # Typical LOI delta-v limit
            raise ValueError(f"LOI delta-v {loi_dv:.1f} m/s exceeds limit of 1200 m/s") 