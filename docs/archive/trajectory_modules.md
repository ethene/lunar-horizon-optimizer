# Trajectory Calculation Modules Documentation

## Celestial Bodies Module (`celestial_bodies.py`)

### Overview
The Celestial Bodies module provides functionality for calculating and obtaining state vectors of celestial bodies using the SPICE toolkit. It operates in the J2000 heliocentric ecliptic reference frame and handles all necessary unit conversions between different systems.

### Dependencies
- `numpy`: For numerical computations
- `spiceypy`: For celestial body ephemeris calculations
- `pykep`: For astrodynamics calculations
- SPICE kernel: `de430.bsp` (must be present in `data/spice/`)

### Units Convention
All methods return values in PyKEP's native units:
- Distances: meters (m)
- Velocities: meters per second (m/s)
- Times: days since J2000 epoch

Note: While SPICE internally uses kilometers and seconds, all values are automatically converted to PyKEP's native units.

### Key Classes

#### `CelestialBody`
Main class providing methods to calculate state vectors of celestial bodies.

##### Methods

###### `get_earth_state(epoch: float) -> Tuple[list, list]`
Gets Earth's heliocentric state vector.
- **Input**: Time in days since J2000 epoch
- **Returns**: Tuple of (position [x,y,z], velocity [vx,vy,vz])
- **Units**: Position in meters, velocity in m/s

###### `get_moon_state(epoch: float) -> Tuple[list, list]`
Gets Moon's heliocentric state vector.
- **Input**: Time in days since J2000 epoch
- **Returns**: Tuple of (position [x,y,z], velocity [vx,vy,vz])
- **Units**: Position in meters, velocity in m/s

###### `get_moon_state_earth_centered(epoch: float) -> Tuple[list, list]`
Gets Moon's state vector relative to Earth.
- **Input**: Time in days since J2000 epoch
- **Returns**: Tuple of (position [x,y,z], velocity [vx,vy,vz])
- **Units**: Position in meters, velocity in m/s

###### `create_local_frame(r: np.ndarray, v: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
Creates a local orbital reference frame.
- **Inputs**:
  - `r`: Position vector in meters
  - `v`: Optional velocity vector in m/s
- **Returns**: Tuple of unit vectors (x_hat, y_hat, z_hat)
- **Frame Definition**:
  - x_hat: Along position vector
  - z_hat: Along angular momentum (if velocity provided)
  - y_hat: Completes right-handed system

### Global Instances
- `EARTH`: Pre-initialized CelestialBody instance for Earth calculations
- `MOON`: Pre-initialized CelestialBody instance for Moon calculations

### Unit Conversions
The module handles several unit conversions internally:
- SPICE to PyKEP:
  - Kilometers → Meters (×1000)
  - Kilometers/second → Meters/second (×1000)
  - Seconds since J2000 ↔ Days since J2000 (×86400)

### Error Handling
- Type checking for epoch inputs
- SPICE kernel loading verification
- Comprehensive error logging
- Exception propagation with context

### Usage Example
```python
from trajectory.celestial_bodies import CelestialBody

# Get Moon's position and velocity relative to Earth at J2000 epoch
moon_pos, moon_vel = CelestialBody.get_moon_state_earth_centered(0.0)

# Create a local frame from position and velocity vectors
x_hat, y_hat, z_hat = CelestialBody.create_local_frame(moon_pos, moon_vel)
```

### Notes
- All calculations are performed in the J2000 heliocentric ecliptic frame
- SPICE kernel (de430.bsp) must be available in the data directory
- Logging is configured at DEBUG level for detailed operation tracking
- Unit conversions are handled automatically 

### Testing (`test_celestial_bodies.py`)

#### Test Suite Overview
The test suite verifies the accuracy and reliability of celestial body state calculations, focusing on:
- State vector calculations in the J2000 frame
- Unit conversions and consistency
- Physical plausibility of results
- Error handling for invalid inputs

#### Key Test Cases

##### `test_earth_state_heliocentric`
Validates Earth's heliocentric state vector:
- Position magnitude ≈ 1 AU (1.496e11 m ± 5%)
- Velocity magnitude ≈ 29.8 km/s (29800 m/s ± 5%)
- Orthogonality of position and velocity vectors

##### `test_moon_state_heliocentric`
Verifies Moon's heliocentric state vector:
- Position magnitude ≈ 1 AU (similar to Earth)
- Velocity magnitude ≈ 29.8 km/s (± 5%)
- Physical plausibility checks

##### `test_moon_state_earth_centered`
Tests Moon's geocentric state vector:
- Position magnitude ≈ 384,400 km (384.4e6 m ± 10%)
- Velocity magnitude ≈ 1.022 km/s (1022 m/s ± 10%)
- Orbital motion verification (perpendicular vectors)

##### `test_invalid_epoch`
Validates error handling for:
- Far future epochs
- Far past epochs
- Non-numeric inputs

##### `test_local_frame`
Verifies local frame transformations:
- Orthonormality of basis vectors
- Preservation of vector magnitudes
- Correct orientation relative to orbital motion

#### Test Execution
Run the test suite with:
```bash
python -m pytest tests/trajectory/test_celestial_bodies.py -v
```

### Physical Constants and Parameters

#### Orbital Parameters
- Earth's semi-major axis: 1.496e11 meters (1 AU)
- Moon's mean distance from Earth: 384,400 km
- Earth's orbital velocity: ~29.8 km/s
- Moon's orbital velocity relative to Earth: ~1.022 km/s

#### Reference Frames
1. **J2000 Heliocentric Ecliptic**
   - Origin: Solar System Barycenter
   - X-axis: Vernal equinox direction at J2000
   - Z-axis: Perpendicular to ecliptic plane
   - Y-axis: Completes right-handed system

2. **Earth-Centered Frame**
   - Origin: Earth's center
   - Parallel to J2000 frame
   - Used for Moon's relative state calculations

3. **Local Orbital Frame**
   - Origin: Object's position
   - X-axis: Along position vector
   - Z-axis: Along angular momentum
   - Y-axis: Completes right-handed system

### Implementation Notes

#### Unit Conversion Best Practices
1. Always convert to PyKEP native units before calculations:
   ```python
   # Convert from km to m
   position_m = position_km * 1000
   velocity_ms = velocity_kms * 1000
   ```

2. Verify units when interfacing with external libraries:
   ```python
   # SPICE returns km, convert to m for PyKEP
   spice_pos_km, spice_vel_kms = spice.spkezr(...)
   pykep_pos_m = [x * 1000 for x in spice_pos_km]
   pykep_vel_ms = [v * 1000 for v in spice_vel_kms]
   ```

#### Performance Considerations
1. State vector calculations are cached internally by SPICE
2. Local frame transformations use optimized NumPy operations
3. Unit conversions are performed only when necessary

#### Error Handling Patterns
```python
try:
    state = celestial_body.get_earth_state(epoch)
except spice.support_types.SpiceyError as e:
    # Handle SPICE-specific errors (e.g., kernel not loaded)
    raise RuntimeError(f"SPICE error: {str(e)}")
except ValueError as e:
    # Handle invalid input values
    raise ValueError(f"Invalid epoch value: {str(e)}")
```

### Troubleshooting Guide

#### Common Issues and Solutions

1. **SPICE Kernel Not Found**
   - Error: "Unable to load kernel"
   - Solution: Ensure `de430.bsp` is in `data/spice/` directory

2. **Invalid Epoch Values**
   - Error: "Time value not in range"
   - Solution: Use epochs within SPICE kernel coverage (typically 1950-2050)

3. **Unit Conversion Errors**
   - Symptom: Results off by factors of 1000
   - Solution: Verify all values are in meters and m/s before calculations

4. **Reference Frame Mismatches**
   - Symptom: Unexpected orbital orientations
   - Solution: Confirm all vectors are in J2000 frame

#### Validation Checks
1. Position magnitudes should be reasonable:
   - Earth from Sun: ~1 AU (1.496e11 m)
   - Moon from Earth: ~384,400 km (3.844e8 m)

2. Velocity magnitudes should be reasonable:
   - Earth around Sun: ~29.8 km/s
   - Moon around Earth: ~1.022 km/s

3. Orbital motion should show:
   - Position and velocity nearly perpendicular
   - Consistent angular momentum direction

### Future Improvements
1. Add support for additional celestial bodies
2. Implement state propagation capabilities
3. Add perturbation models for increased accuracy
4. Optimize performance for batch calculations
5. Extend unit test coverage for edge cases 

## Constants Module (`constants.py`)

### Overview
The Constants module provides a centralized source of physical constants, unit conversions, and time limits used throughout the trajectory calculations. It ensures consistency by sourcing values from established libraries like PyKEP and Astropy.

### Classes

#### `Units`
Unit conversion constants, primarily sourced from PyKEP.

##### Constants
- Angular Conversions:
  - `DEG2RAD` = 0.017453 (degrees to radians)
  - `RAD2DEG` = 57.29578 (radians to degrees)
- Distance Conversions:
  - `M2KM` = 1e-3 (meters to kilometers)
  - `KM2M` = 1000.0 (kilometers to meters)
- Velocity Conversions:
  - `MS2KMS` = 1e-3 (m/s to km/s)
  - `KMS2MS` = 1000.0 (km/s to m/s)
- Time Conversions:
  - `DAYS2SEC` = 86400 (days to seconds)
  - `SEC2DAYS` = 1.1574e-5 (seconds to days)

#### `PhysicalConstants`
Physical constants in PyKEP native units.

##### Earth Constants
- `MU_EARTH` = 3.986004418e14 m³/s² (gravitational parameter)
- `EARTH_RADIUS` = 6378137 m (equatorial radius)
- `EARTH_ESCAPE_VELOCITY` ≈ 11.2 km/s (calculated)
- `EARTH_ORBITAL_PERIOD` = 31557600 s (365.25 days)

##### Moon Constants
- `MU_MOON` = 4902.800118e9 m³/s² (gravitational parameter)
- `MOON_RADIUS` = 1737.4 km (mean radius)
- `MOON_ORBIT_RADIUS` = 384400 km (mean distance)
- `MOON_ORBITAL_PERIOD` = 2361744 s (27.32 days)
- `MOON_ESCAPE_VELOCITY` ≈ 2.38 km/s (calculated)
- `MOON_ORBITAL_VELOCITY` ≈ 1.022 km/s (calculated)
- `MOON_SOI` ≈ 66,000 km (sphere of influence)

#### `EphemerisLimits`
Time limits for ephemeris calculations.

##### Parameters
- `MIN_YEAR` = 2020 (minimum valid year)
- `MAX_YEAR` = 2050 (maximum valid year)
- `EPOCH_REF` = "2000-01-01 00:00:00" (J2000 reference epoch)

### Unit Conventions
All calculations use consistent units:
- Distances: meters (m)
- Velocities: meters per second (m/s)
- Times: seconds (s) for durations, days since J2000 for epochs
- Angles: radians (rad)

### Reference Standards
1. **PyKEP Constants**
   - Used for astrodynamics calculations
   - Includes Earth parameters, angular conversions, time conversions
   - Provides consistent units across the library

2. **Astropy Constants**
   - Used for general physical constants
   - Provides high-precision values with uncertainty estimates
   - Includes up-to-date astronomical constants

3. **JPL DE440 Ephemeris**
   - Source for Moon constants
   - Provides high-accuracy ephemeris data
   - Used for precise orbital calculations

### Usage Examples

#### Unit Conversions
```python
from trajectory.constants import Units

# Convert degrees to radians
angle_rad = angle_deg * Units.DEG2RAD

# Convert kilometers to meters
distance_m = distance_km * Units.KM2M
```

#### Physical Calculations
```python
from trajectory.constants import PhysicalConstants as PC

# Calculate orbital velocity at given radius
def orbital_velocity(radius_m):
    return np.sqrt(PC.MU_EARTH / radius_m)

# Check if position is within Moon's SOI
def is_in_moon_soi(position_m):
    return np.linalg.norm(position_m) < PC.MOON_SOI
```

#### Time Validation
```python
from trajectory.constants import EphemerisLimits

def validate_epoch(year):
    if not EphemerisLimits.MIN_YEAR <= year <= EphemerisLimits.MAX_YEAR:
        raise ValueError(f"Year {year} outside valid range "
                       f"[{EphemerisLimits.MIN_YEAR}, {EphemerisLimits.MAX_YEAR}]")
```

### Implementation Notes

#### Unit Conversion Best Practices
1. Always use the provided constants for conversions:
   ```python
   # Correct:
   distance_m = distance_km * Units.KM2M
   
   # Avoid:
   distance_m = distance_km * 1000  # Magic numbers are error-prone
   ```

2. Maintain consistent units throughout calculations:
   ```python
   # Convert all inputs to standard units (meters)
   radius_m = altitude_km * Units.KM2M + PhysicalConstants.EARTH_RADIUS
   velocity_ms = velocity_kms * Units.KMS2MS
   ```

#### Validation Checks
Use physical constants for sanity checks:
```python
def validate_orbit(radius_m, velocity_ms):
    # Check if orbit is above Earth's surface
    if radius_m < PhysicalConstants.EARTH_RADIUS:
        raise ValueError("Orbit below Earth's surface")
        
    # Check if velocity exceeds escape velocity
    escape_vel = PhysicalConstants.EARTH_ESCAPE_VELOCITY
    if velocity_ms > escape_vel:
        raise ValueError("Velocity exceeds escape velocity")
``` 

## Unit Conversions Module (`unit_conversions.py`)

### Overview
The Unit Conversions module provides a comprehensive set of utilities for converting between different units and coordinate systems. It maintains type consistency and supports both scalar and array inputs.

### Standard Units (PyKEP Native)
- Distances: meters (m)
- Velocities: meters per second (m/s)
- Angles: radians (internal calculations)
- Gravitational Parameters: m³/s²

### Time Conventions
- **MJD2000**: Days since 2000-01-01 00:00:00 UTC (PyKEP's internal reference)
- **J2000**: Days since 2000-01-01 12:00:00 UTC (standard astronomical epoch)
- Relationship: J2000 = MJD2000 + 0.5

### Core Functions

#### Time Conversions
```python
def datetime_to_mjd2000(dt: datetime) -> float:
    """Convert datetime to Modified Julian Date 2000 (MJD2000)."""
    
def datetime_to_j2000(dt: datetime) -> float:
    """Convert datetime to days since J2000 epoch."""
    
def datetime_to_pykep_epoch(dt: datetime) -> float:
    """Convert datetime to PyKEP epoch (alias for datetime_to_mjd2000)."""
    
def pykep_epoch_to_datetime(epoch: float) -> datetime:
    """Convert PyKEP epoch back to datetime."""
```

#### Distance Conversions
```python
def km_to_m(km: NumericType) -> NumericType:
    """Convert kilometers to meters."""
    
def m_to_km(m: NumericType) -> NumericType:
    """Convert meters to kilometers."""
```

#### Velocity Conversions
```python
def kmps_to_mps(kmps: NumericType) -> NumericType:
    """Convert kilometers per second to meters per second."""
    
def mps_to_kmps(mps: NumericType) -> NumericType:
    """Convert meters per second to kilometers per second."""
```

#### Angular Conversions
```python
def deg_to_rad(deg: NumericType) -> NumericType:
    """Convert degrees to radians."""
    
def rad_to_deg(rad: NumericType) -> NumericType:
    """Convert radians to degrees."""
```

#### Gravitational Parameter Conversions
```python
def km3s2_to_m3s2(mu: float) -> float:
    """Convert gravitational parameter from km³/s² to m³/s²."""
    
def m3s2_to_km3s2(mu: float) -> float:
    """Convert gravitational parameter from m³/s² to km³/s²."""
```

#### Time Duration Conversions
```python
def days_to_seconds(days: float) -> float:
    """Convert days to seconds."""
    
def seconds_to_days(seconds: float) -> float:
    """Convert seconds to days."""
```

### Integration Examples

#### Using with CelestialBody
```python
from datetime import datetime, timezone
from trajectory.celestial_bodies import CelestialBody
from trajectory.constants import PhysicalConstants as PC
from utils.unit_conversions import datetime_to_j2000, mps_to_kmps

# Get Moon's state at a specific time
dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
epoch = datetime_to_j2000(dt)
moon_pos, moon_vel = CelestialBody.get_moon_state(epoch)

# Convert velocities to km/s for display
moon_vel_kms = mps_to_kmps(moon_vel)
```

#### Using with Physical Constants
```python
from trajectory.constants import PhysicalConstants as PC
from utils.unit_conversions import km_to_m, kmps_to_mps

# Convert orbital parameters to standard units
altitude_km = 400  # km
radius_m = km_to_m(altitude_km + PC.EARTH_RADIUS/1000)

# Calculate orbital velocity
orbital_velocity_ms = np.sqrt(PC.MU_EARTH / radius_m)
orbital_velocity_kms = mps_to_kmps(orbital_velocity_ms)
```

#### Time Handling Example
```python
from datetime import datetime, timezone
from utils.unit_conversions import (
    datetime_to_mjd2000,
    datetime_to_j2000,
    pykep_epoch_to_datetime
)

# Convert between different time representations
dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
mjd2000 = datetime_to_mjd2000(dt)      # PyKEP internal time
j2000 = datetime_to_j2000(dt)          # Astronomical epoch
dt_back = pykep_epoch_to_datetime(mjd2000)  # Back to datetime
```

### Best Practices

#### 1. Input Validation
```python
def calculate_orbital_elements(position_km, velocity_kmps):
    # Convert to standard units (meters, m/s)
    position_m = km_to_m(position_km)
    velocity_ms = kmps_to_mps(velocity_kmps)
    
    # Perform calculations in standard units
    # ...
    
    # Convert results back to desired output units
    return m_to_km(result_m)
```

#### 2. Consistent Unit Handling
```python
def propagate_orbit(initial_state, duration_days):
    # Convert all inputs to standard units
    pos_m = km_to_m(initial_state.position_km)
    vel_ms = kmps_to_mps(initial_state.velocity_kmps)
    duration_sec = days_to_seconds(duration_days)
    
    # Calculations in standard units
    # ...
    
    # Convert back for output
    return {
        'position_km': m_to_km(final_pos_m),
        'velocity_kmps': mps_to_kmps(final_vel_ms)
    }
```

#### 3. Time Handling
```python
def calculate_transfer_window(departure_dt, arrival_dt):
    # Convert to J2000 for celestial calculations
    dep_epoch = datetime_to_j2000(departure_dt)
    arr_epoch = datetime_to_j2000(arrival_dt)
    
    # Get states at epochs
    earth_state = CelestialBody.get_earth_state(dep_epoch)
    moon_state = CelestialBody.get_moon_state(arr_epoch)
```

### Common Pitfalls

1. **Mixing Units**
   ```python
   # DON'T: Mix units in calculations
   radius = altitude_km + PhysicalConstants.EARTH_RADIUS  # Wrong!
   
   # DO: Convert to common units first
   radius_m = km_to_m(altitude_km) + PhysicalConstants.EARTH_RADIUS
   ```

2. **Forgetting Time Zones**
   ```python
   # DON'T: Use naive datetimes
   dt = datetime(2024, 1, 1)  # Wrong!
   
   # DO: Always use timezone-aware datetimes
   dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
   ```

3. **Inconsistent Epoch References**
   ```python
   # DON'T: Mix epoch references
   epoch_j2000 = datetime_to_j2000(dt)
   epoch_pykep = epoch_j2000  # Wrong!
   
   # DO: Use appropriate conversion
   epoch_pykep = datetime_to_pykep_epoch(dt)
   ```

### Performance Tips

1. **Batch Conversions**
   ```python
   # More efficient for arrays
   positions_m = km_to_m(np.array(positions_km))
   ```

2. **Type Preservation**
   ```python
   # Preserves input type (list, tuple, ndarray)
   result = kmps_to_mps(velocity)
   ```

3. **Minimize Conversions**
   ```python
   # Convert once at boundaries
   def process_trajectory(states_km):
      states_m = km_to_m(states_km)
      # Do all processing in meters
      result_m = perform_calculations(states_m)
      return m_to_km(result_m)
   ``` 

## Elements Module (`elements.py`)

### Overview
The Elements module provides fundamental orbital mechanics calculations for both circular and elliptical orbits. It handles core functions for orbital period, velocity components, and anomaly conversions while maintaining consistent unit handling throughout all calculations.

### Dependencies
- `numpy`: For numerical computations
- `pykep`: For gravitational parameters and constants
- `logging`: For operation tracking and debugging

### Units Convention
All functions use consistent units:
- Distances: kilometers (km)
- Velocities: kilometers per second (km/s)
- Time: seconds (s)
- Angles: degrees for input/output, radians for internal calculations

### Key Functions

#### `orbital_period(semi_major_axis: float, mu: float = pk.MU_EARTH) -> float`
Calculates orbital period using Kepler's Third Law.
- **Input**: 
  - `semi_major_axis`: Semi-major axis in kilometers
  - `mu`: Gravitational parameter (defaults to Earth's)
- **Returns**: Orbital period in seconds
- **Example Use**: Calculate ISS orbital period (~92.68 minutes)

#### `velocity_at_point(semi_major_axis: float, eccentricity: float, true_anomaly: float, mu: float = pk.MU_EARTH) -> Tuple[float, float]`
Calculates radial and tangential velocity components.
- **Inputs**:
  - `semi_major_axis`: Semi-major axis in kilometers
  - `eccentricity`: Orbit eccentricity (0 ≤ e < 1)
  - `true_anomaly`: True anomaly in degrees
  - `mu`: Gravitational parameter
- **Returns**: Tuple of (radial velocity, tangential velocity) in km/s
- **Key Features**:
  - Handles both circular and elliptical orbits
  - Provides velocity decomposition in orbital plane
  - Validates input parameters

#### `mean_to_true_anomaly(mean_anomaly: float, eccentricity: float) -> float`
Converts mean anomaly to true anomaly.
- **Inputs**:
  - `mean_anomaly`: Mean anomaly in degrees
  - `eccentricity`: Orbit eccentricity
- **Returns**: True anomaly in degrees
- **Implementation**:
  - Uses Newton-Raphson iteration for Kepler's equation
  - Handles multiple revolutions
  - Maintains quadrant consistency

#### `true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float`
Converts true anomaly to mean anomaly.
- **Inputs**:
  - `true_anomaly`: True anomaly in degrees
  - `eccentricity`: Orbit eccentricity
- **Returns**: Mean anomaly in degrees
- **Features**:
  - Direct analytical solution
  - Handles all quadrants correctly
  - Preserves angle range

### Implementation Notes

#### Unit Conversion Best Practices
```python
# Convert input angles to radians for calculations
nu_rad = np.radians(true_anomaly)

# Convert output angles back to degrees
mean_anomaly_deg = np.degrees(mean_anomaly_rad)
```

#### Error Handling Patterns
```python
def validate_orbital_elements(a: float, e: float) -> None:
    """Validate basic orbital elements."""
    if a <= 0:
        raise ValueError("Semi-major axis must be positive")
    if e < 0 or e >= 1:
        raise ValueError("Eccentricity must be in range [0, 1)")
```

#### Performance Considerations
1. Optimized Newton-Raphson iteration for Kepler's equation
2. Vectorized operations for array inputs
3. Caching of intermediate calculations where beneficial

### Testing Strategy

The module includes comprehensive tests verifying:
1. **Orbital Period Calculations**
   - LEO orbits (ISS-like)
   - Lunar orbit
   - Validation against known periods

2. **Velocity Components**
   - Circular orbit verification
   - Elliptical orbit at key points
   - Conservation of energy

3. **Anomaly Conversions**
   - Circular orbit edge case (e = 0)
   - Elliptical orbit conversions
   - Roundtrip conversion accuracy

### Usage Examples

#### Basic Orbital Period
```python
from trajectory.elements import orbital_period

# Calculate ISS orbital period
iss_sma = 6778.0  # km (Earth radius + 400 km)
period = orbital_period(iss_sma)
print(f"ISS orbital period: {period/60:.2f} minutes")
```

#### Velocity Components
```python
from trajectory.elements import velocity_at_point

# Calculate velocity components in GTO
gto_sma = 24396.0  # km
gto_ecc = 0.7306
v_r, v_t = velocity_at_point(gto_sma, gto_ecc, true_anomaly=0.0)
print(f"Perigee velocity: {(v_r**2 + v_t**2)**0.5:.2f} km/s")
```

#### Anomaly Conversion
```python
from trajectory.elements import mean_to_true_anomaly

# Convert mean anomaly to true anomaly
true_anom = mean_to_true_anomaly(mean_anomaly=45.0, eccentricity=0.1)
print(f"True anomaly: {true_anom:.2f} degrees")
```

### Future Improvements
1. Add support for hyperbolic orbits (e > 1)
2. Implement additional orbital element conversions
3. Add perturbation effects calculations
4. Optimize performance for large-scale simulations
5. Extend test coverage for edge cases

### Common Pitfalls

1. **Angle Units**
   ```python
   # DON'T: Use radians for input
   true_anom = mean_to_true_anomaly(0.785, 0.1)  # Wrong!
   
   # DO: Use degrees for input
   true_anom = mean_to_true_anomaly(45.0, 0.1)
   ```

2. **Eccentricity Range**
   ```python
   # DON'T: Use e ≥ 1
   v_r, v_t = velocity_at_point(a, e=1.0, nu=0)  # Wrong!
   
   # DO: Use 0 ≤ e < 1
   v_r, v_t = velocity_at_point(a, e=0.7, nu=0)
   ```

3. **Semi-major Axis Units**
   ```python
   # DON'T: Use meters
   period = orbital_period(6778000)  # Wrong!
   
   # DO: Use kilometers
   period = orbital_period(6778.0)
   ```

### Physical Constants Used
- Earth's gravitational parameter (μ): 398600.4418 km³/s²
- Earth's radius: 6378.137 km
- Standard gravitational parameter conversions
- Mathematical constants (π, etc.) 

## Trajectory Models Architecture

### Overview
The trajectory calculation system is built on a modular architecture with several key classes split into dedicated files for better organization and maintainability. The original monolithic `models.py` has been refactored into specialized modules, with `models.py` now serving as a compatibility layer.

### Core Modules

#### Orbit State Module (`orbit_state.py`)
Provides the `OrbitState` class for representing orbital states using classical orbital elements.

##### Key Features
- Orbital element validation and conversion
- Position and velocity vector calculations
- Reference frame transformations
- PyKEP integration for propagation
- State vector conversion utilities

##### Example Usage
```python
from trajectory.orbit_state import OrbitState

# Create orbital state from elements
state = OrbitState(
    semi_major_axis=6778.0,  # km
    eccentricity=0.001,
    inclination=51.6,        # degrees
    raan=45.0,              # degrees
    arg_periapsis=90.0,     # degrees
    true_anomaly=180.0,     # degrees
    epoch=datetime.now(timezone.utc)
)

# Get state vectors
position = state.position  # km
velocity = state.velocity(mu=pk.MU_EARTH)  # km/s
```

#### Maneuver Module (`maneuver.py`)
Implements the `Maneuver` class for representing impulsive orbital maneuvers.

##### Key Features
- Delta-v vector representation
- Maneuver timing and sequencing
- Unit validation and conversion
- Velocity change application
- Maneuver scaling and reversal

##### Example Usage
```python
from trajectory.maneuver import Maneuver

# Create a maneuver
maneuver = Maneuver(
    delta_v=(0.1, 0.2, -0.05),  # km/s
    epoch=datetime.now(timezone.utc),
    description="Orbit correction burn"
)

# Apply maneuver to velocity
new_velocity = maneuver.apply_to_velocity(initial_velocity)
```

#### Base Trajectory Module (`trajectory_base.py`)
Defines the abstract `Trajectory` base class for implementing different types of trajectories.

##### Key Features
- State propagation framework
- Maneuver application logic
- State caching for performance
- Trajectory validation interface
- Delta-v calculation utilities

##### Example Usage
```python
from trajectory.trajectory_base import Trajectory

class CustomTrajectory(Trajectory):
    def validate_trajectory(self) -> bool:
        # Implement custom validation logic
        return True

# Create and propagate trajectory
trajectory = CustomTrajectory(
    initial_state=initial_orbit,
    maneuvers=[correction_burn],
    end_epoch=final_time
)

# Get state at specific time
state = trajectory.get_state_at(target_time)
```

#### Lunar Transfer Module (`lunar_transfer.py`)
Implements the `LunarTrajectory` class for Earth-Moon transfer trajectories.

##### Key Features
- Trans-lunar injection (TLI) maneuver handling
- Lunar transfer validation
- Flight time constraints
- Delta-v optimization
- State propagation in Earth-Moon system

##### Example Usage
```python
from trajectory.lunar_transfer import LunarTrajectory

# Create lunar transfer trajectory
transfer = LunarTrajectory(
    initial_state=parking_orbit,
    final_state=lunar_orbit,
    tli_dv=3200.0,  # m/s
    tof_days=4.5
)

# Validate transfer parameters
transfer.validate()
```

### Legacy Support
The `models.py` module now serves as a compatibility layer, re-exporting the core classes:
```python
from trajectory.models import OrbitState, Maneuver, Trajectory, LunarTrajectory
```
New code should import directly from the specific modules rather than using this compatibility layer.

### Implementation Notes

#### Unit Conventions
- Distances: kilometers (km)
- Velocities: kilometers per second (km/s)
- Times: datetime objects (timezone-aware) for epochs
- Angles: degrees for input/output
- Internal calculations: SI units (m, m/s, radians)

#### Validation Patterns
1. Input validation in `__post_init__` methods
2. Physics-based validation in dedicated methods
3. Unit conversion validation
4. Temporal sequence validation
5. State vector validation

#### State Management
1. Caching of propagated states
2. Clear cache on maneuver additions
3. Lazy propagation for performance
4. Consistent epoch handling
5. Error handling for invalid states

### Testing Strategy

#### Unit Tests
- Orbital element validation
- Maneuver application accuracy
- Trajectory propagation consistency
- State vector conversions
- Unit conversion correctness

#### Integration Tests
- End-to-end trajectory generation
- Multi-maneuver sequences
- State propagation with maneuvers
- Lunar transfer validation
- Reference frame transformations

### Future Improvements
1. Add support for continuous thrust maneuvers
2. Implement trajectory optimization methods
3. Add perturbation models
4. Enhance state visualization capabilities
5. Add trajectory comparison utilities
6. Implement trajectory export/import functionality

### Common Pitfalls

1. **Epoch Handling**
   ```python
   # DON'T: Use naive datetime
   epoch = datetime.now()  # Wrong!
   
   # DO: Use timezone-aware datetime
   epoch = datetime.now(timezone.utc)
   ```

2. **State Vector Units**
   ```python
   # DON'T: Mix units
   state = OrbitState.from_state_vectors(
       position_m,  # Wrong!
       velocity_kms
   )
   
   # DO: Use consistent units
   state = OrbitState.from_state_vectors(
       position_km,
       velocity_kms
   )
   ```

3. **Maneuver Timing**
   ```python
   # DON'T: Add maneuvers outside trajectory timespan
   trajectory.add_maneuver(late_maneuver)  # Will raise error
   
   # DO: Validate maneuver timing
   if maneuver.epoch <= trajectory.end_epoch:
       trajectory.add_maneuver(maneuver)
   ```

### Physical Constants Used
- Earth's gravitational parameter (μ): 398600.4418 km³/s²
- Moon's gravitational parameter: 4902.8001 km³/s²
- Earth radius: 6378.137 km
- Moon radius: 1737.4 km
- Moon's orbital parameters (distance, period) 