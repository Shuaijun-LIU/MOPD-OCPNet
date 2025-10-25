# PollutionModel3D: A Comprehensive Framework for Ocean Pollution Simulation

## Overview
PollutionModel3D is a three-dimensional ocean pollution simulation framework developed as part of the "A Large-Scale Oceanographic Dataset and Prediction Framework for Ocean Currents and Pollution Dispersion" project. The framework integrates physical, chemical, and biological processes to provide comprehensive ocean pollution prediction capabilities with modular design, data-driven approaches, and extensibility for both research and engineering applications.

## Project Structure
```
PollutionModel3D/
├── src/
│   ├── model.py                    # Core 3D pollution model implementation
│   ├── grid3d.py                   # 3D Eulerian grid system
│   ├── pollution_field.py          # Multi-pollutant concentration field management
│   ├── data_select.py              # Real data selection and preprocessing
│   ├── simulation_1.py             # Complete simulation scenario
│   ├── test_model.py               # Comprehensive test cases
│   ├── __init__.py                 # Package initialization
│   └── modules/                    # Process modules
│       ├── advection_module.py           # Advection process (upwind scheme)
│       ├── diffusion_module.py           # Diffusion process (environment-dependent)
│       ├── decay_module.py               # Decay process (environment-dependent)
│       ├── coupling_reaction_module.py   # Chemical reactions (mass action kinetics)
│       ├── bio_uptake_module.py         # Biological uptake (Michaelis-Menten)
│       ├── precipitation_module.py       # Precipitation reactions
│       ├── source_sink_module.py        # Source and sink terms
│       ├── boundary_conditions_module.py # Boundary conditions handling
│       └── output_module.py             # Output and visualization
├── data/                           # Data directory for input/output
├── setup.py                        # Package installation configuration
└── Readme_PollutionModel3D.md      # This documentation
```

## Module Architecture

### Core Modules

#### Grid3D (`grid3d.py`)
**Purpose**: 3D Eulerian grid system for spatial discretization
**Key Features**:
- Customizable grid resolution and spacing
- 3D coordinate arrays (X, Y, Z) and 1D vectors (x, y, z)
- Cell volume calculations
- Boundary cell identification
- Spatial query methods

**API**:
```python
grid = Grid3D(nx=50, ny=50, nz=25, dx=20, dy=20, dz=4)
center = grid.get_cell_center(i, j, k)
volume = grid.get_cell_volume(i, j, k)
is_boundary = grid.is_boundary_cell(i, j, k)
```

#### PollutionField (`pollution_field.py`)
**Purpose**: Multi-pollutant concentration field management
**Key Features**:
- Dynamic pollutant addition/removal
- Unit system support
- Background concentration values
- Boundary condition application
- Mass balance calculations
- Spatial interpolation (trilinear)

**API**:
```python
field = PollutionField(grid)
field.add_pollutant("NH4", initial_value=0.1, unit="kg/m^3")
conc = field.get_concentration("NH4")
mass = field.calculate_total_mass("NH4")
```

#### PollutionModel3D (`model.py`)
**Purpose**: Main model class integrating all modules
**Key Features**:
- Module integration and orchestration
- Explicit Euler time integration
- CFL stability checking
- Progress monitoring and reporting
- Mass balance tracking
- Unified API interface

**API**:
```python
model = PollutionModel3D(domain_size, grid_resolution, time_step, output_dir)
model.add_pollutant("NH4", initial_concentration=0.1, decay_rate=1e-6)
model.set_velocity_field(u, v, w)
model.run(end_time=86400.0)
```

### Physical Process Modules

#### AdvectionModule (`advection_module.py`)
**Purpose**: Handles pollutant advection using velocity fields
**Key Features**:
- Upwind scheme for numerical stability
- CFL condition checking
- Zero-gradient boundary conditions
- 3D advection in x, y, z directions

**Mathematical Implementation**:
```
∂C/∂t + u·∇C = 0
```
- Upwind differencing based on velocity direction
- CFL condition: max(|u|Δt/Δx, |v|Δt/Δy, |w|Δt/Δz) ≤ 1

**API**:
```python
adv_term = advection.compute_advection_term(concentration, u, v, w)
cfl_x, cfl_y, cfl_z = advection.compute_cfl_numbers(u, v, w, dt)
is_stable = advection.check_stability(u, v, w, dt)
```

#### DiffusionModule (`diffusion_module.py`)
**Purpose**: Handles pollutant diffusion with environment-dependent coefficients
**Key Features**:
- Environment-dependent diffusion coefficients
- Central difference scheme
- Temperature, wave, and salinity effects
- Stability criterion checking

**Mathematical Implementation**:
```
∂C/∂t = ∇·(K∇C)
K = K₀ × f(T) × f(W) × f(S)
f(T) = 1 + α_T(T - T_ref)
f(W) = 1 + βW
f(S) = 1 + α_s(1 - S/35)
```

**API**:
```python
K = diffusion.compute_diffusion_coefficient(temperature, wave_velocity, salinity)
diff_term = diffusion.compute_diffusion_term(concentration, K)
max_dt = diffusion.compute_stability_criterion(K, dt)
```

### Chemical Process Modules

#### DecayModule (`decay_module.py`)
**Purpose**: Handles pollutant decay processes with environment-dependent rates
**Key Features**:
- Environment-dependent decay rates
- Arrhenius temperature dependence
- pH and dissolved oxygen effects
- Half-life calculations
- First-order decay kinetics

**Mathematical Implementation**:
```
dC/dt = -λ(T,pH,DO) × C
λ = λ₀ × f(T) × f(pH) × f(DO)
f(T) = exp(-Ea/(RT))
f(pH) = 1 + a(pH-7) + b(pH-7)²
f(DO) = 1 + c×DO + d×DO²
```

**API**:
```python
decay.add_pollutant("NH4", base_decay_rate=1e-6, activation_energy=50000)
decay_rate = decay.compute_decay_rate("NH4", temperature, ph, do)
decay_term = decay.compute_decay_term("NH4", concentration, decay_rate)
half_life = decay.compute_half_life("NH4", temperature=298.15, ph=7.0)
```

#### CouplingReactionModule (`coupling_reaction_module.py`)
**Purpose**: Handles coupled reactions between pollutants
**Key Features**:
- Multi-component chemical reactions
- Mass action kinetics
- Environment-dependent reaction rates
- Flexible stoichiometry
- Predefined reaction types

**Mathematical Implementation**:
```
dCi/dt = Σ νij × rj
rj = kj × f(T) × f(pH) × Π Ci^αij
f(T) = exp(αT(T - Tref))
f(pH) = 1 + a(pH-7) + b(pH-7)²
```

**API**:
```python
reaction.add_reaction("nitrification", ["NH4", "O2"], ["NO3", "H2O"], 
                     {"NH4": -1, "O2": -2, "NO3": 1, "H2O": 1}, rate=1e-5)
reaction_terms = reaction.compute_reaction_terms(temperature, ph, concentrations)
reaction.add_mercury_methylation(base_rate=1e-7)
```

#### PrecipitationModule (`precipitation_module.py`)
**Purpose**: Handles precipitation reactions of metal ions and anions
**Key Features**:
- Solubility product-based precipitation
- Saturation index calculations
- Environment-dependent precipitation rates
- Settling velocity modeling
- Multiple precipitation types

**Mathematical Implementation**:
```
R = k × f(T) × f(pH) × max(0, SI-1)
SI = [M^n+][A^m-]/Ksp
Settling = w × ∂C/∂z
```

**API**:
```python
precip.add_precipitation_reaction("Fe_PO4", "Fe", "PO4", Ksp=1e-20, rate=1e-6)
SI = precip.compute_saturation_index("Fe_PO4", fe_conc, po4_conc)
precip_terms = precip.compute_precipitation_terms(concentrations, temperature, ph)
```

### Biological Process Modules

#### BioUptakeModule (`bio_uptake_module.py`)
**Purpose**: Handles biological uptake of pollutants by phytoplankton
**Key Features**:
- Michaelis-Menten kinetics
- Temperature and light dependencies
- Phytoplankton biomass dynamics
- Multiple nutrient types support

**Mathematical Implementation**:
```
R = V_max × f(T) × f(L) × C/(K_s + C) × B
f(T) = exp(α_T(T - T_ref))
f(L) = 1 + α_L × L
dB/dt = (μ - m) × B
```

**API**:
```python
bio.add_pollutant("PO4", max_uptake_rate=1e-5, half_saturation=0.01)
bio.set_phytoplankton_parameters(growth_rate=1e-5, mortality_rate=1e-6)
uptake_terms = bio.compute_uptake_terms(concentrations, temperature, light, biomass)
```

### Source and Sink Modules

#### SourceSinkModule (`source_sink_module.py`)
**Purpose**: Handles pollution sources and sinks
**Key Features**:
- Point, area, and line sources
- Time-dependent emission functions
- Multiple sink types (deposition, degradation, reaction)
- Gaussian distribution for point sources

**API**:
```python
source.add_point_source("NH4", position=(500, 500, 0), emission_rate=1.0)
source.add_area_source("PO4", area=(0, 1000, 0, 1000), emission_rate=0.1)
source.add_sink_term("NH4", "deposition", rate=1e-6)
source_terms = source.compute_source_sink_terms(concentrations, time, dependencies)
```

#### BoundaryConditionsModule (`boundary_conditions_module.py`)
**Purpose**: Handles boundary conditions for the model
**Key Features**:
- Dirichlet (fixed value) boundaries
- Neumann (fixed gradient) boundaries
- Periodic boundaries
- Open (radiation) boundaries
- Time-dependent boundary functions

**API**:
```python
bc.set_dirichlet_boundary("NH4", "bottom", value=0.0)
bc.set_neumann_boundary("PO4", "top", gradient=0.0)
bc.set_periodic_boundary("velocity", "x")
bc.apply_boundary_conditions("NH4", data, time)
```

### Output and Visualization Modules

#### OutputModule (`output_module.py`)
**Purpose**: Handles model output, visualization, and statistics
**Key Features**:
- NetCDF data export
- 2D/3D visualization (XY, XZ, YZ planes)
- Statistical analysis (mean, std, min, max)
- Configurable output intervals
- Time series tracking

**API**:
```python
output.set_output_fields(["NH4", "PO4", "Hg"])
output.set_visualization_fields(["NH4", "PO4"])
output.save_data(fields, time)
output.create_visualization(fields, time)
statistics = output.compute_statistics(fields, time)
```

## Key Features

### Physical Processes
- **Advection**: Simulates pollutant transport with water flow using upwind scheme
- **Diffusion**: Models molecular and turbulent diffusion with environment-dependent coefficients
- **Boundary Conditions**: Supports Dirichlet, Neumann, periodic, and open boundaries

### Chemical Processes
- **Reactions**: Handles multi-component chemical reactions with mass action kinetics
- **Precipitation**: Models precipitation-dissolution processes with solubility products
- **Environmental Response**: Considers temperature, pH, dissolved oxygen, and salinity effects

### Biological Processes
- **Phytoplankton Uptake**: Models biological absorption using Michaelis-Menten kinetics
- **Decay**: Simulates natural degradation processes with environment-dependent rates
- **Environmental Factors**: Incorporates light, temperature, and nutrient effects

### Source and Sink Terms
- **Point Sources**: Handles discrete emission sources with time-dependent functions
- **Area Sources**: Manages distributed pollution inputs
- **Line Sources**: Supports linear pollution sources
- **Sink Processes**: Models sedimentation, degradation, and absorption

### Data Integration
- **Real Data Support**: Integrates with real oceanographic datasets
- **Unit Conversion**: Handles different units between data and model
- **Geographic Data**: Supports lat/lon to Cartesian coordinate conversion
- **Data Visualization**: Provides data selection and visualization tools

## Mathematical Framework

### Advection Equation
\[
\frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} + w\frac{\partial C}{\partial z} = 0
\]

### Diffusion Equation
\[
\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}(D_x\frac{\partial C}{\partial x}) + \frac{\partial}{\partial y}(D_y\frac{\partial C}{\partial y}) + \frac{\partial}{\partial z}(D_z\frac{\partial C}{\partial z})
\]

### Environment-Dependent Diffusion Coefficient
\[
D = D_0 \cdot f(T) \cdot f(W) \cdot f(S)
\]
\[
f(T) = e^{-\frac{E_a}{R}(\frac{1}{T} - \frac{1}{T_{ref}})}
\]
\[
f(W) = 1 + \alpha W
\]
\[
f(S) = 1 + \beta S
\]

### Chemical Reactions
\[
\frac{dC_i}{dt} = \sum_{j=1}^{N} \nu_{ij} r_j
\]
\[
r_j = k_j \prod_{i=1}^{M} C_i^{\alpha_{ij}} \cdot f(T) \cdot f(pH) \cdot f(DO)
\]

### Biological Processes
\[
R_b = k_b B f(T) f(L) \frac{C}{K_s + C}
\]
\[
f(T) = e^{-\frac{E_a}{R}(\frac{1}{T} - \frac{1}{T_{opt}})}
\]
\[
f(L) = \frac{L}{K_L + L}
\]

### Precipitation Reactions
\[
R_p = k_p \max(0, \log(\frac{Q}{K_{sp}})) \cdot f(T) \cdot f(pH)
\]
\[
Q = [M^{n+}]^m [A^{m-}]^n
\]

## Usage

### Installation
1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib netcdf4 scipy
   ```
3. Install the package:
   ```bash
   python setup.py install
   ```

### Running the Model

#### Basic Usage
```python
from PollutionModel3D.src.model import PollutionModel3D

# Initialize model
model = PollutionModel3D(
    domain_size=(1000.0, 1000.0, 100.0),
    grid_resolution=(50, 50, 25),
    time_step=60.0,
    output_dir="output"
)

# Add pollutants
model.add_pollutant("NH4", initial_concentration=0.1, decay_rate=1e-6)
model.add_pollutant("PO4", initial_concentration=0.05, decay_rate=5e-7)

# Set velocity field
u, v, w = create_velocity_field()
model.set_velocity_field(u, v, w)

# Set environmental fields
model.set_environmental_field("temperature", temp_field)
model.set_environmental_field("pH", ph_field)

# Add reactions
model.add_reaction(
    name="nitrification",
    reactants=["NH4", "O2"],
    products=["NO3", "H2O"],
    stoichiometry={"NH4": -1, "O2": -2, "NO3": 1, "H2O": 1},
    rate=1e-5
)

# Run simulation
model.run(end_time=86400.0, progress_interval=3600.0)
```

#### Complete Simulation
```python
# Run the complete simulation scenario
from PollutionModel3D.src.simulation_1 import run_simulation

model = run_simulation()
```

### Data Requirements
The model requires the following input data:
- **Grid parameters**: Domain size and resolution
- **Velocity field**: u, v, w components (3D arrays)
- **Environmental fields**: Temperature, pH, DO, salinity, light intensity, etc.
- **Pollutant parameters**: Initial concentrations, decay rates, diffusion coefficients
- **Source terms**: Emission rates, locations, time functions
- **Boundary conditions**: Boundary types and values

## Output
The model generates:
- **Concentration fields**: 3D concentration distributions for each pollutant
- **Time series**: Temporal evolution of pollutant distributions
- **Statistical analysis**: Mean, max, min, standard deviation of concentrations
- **Visualization**: 2D/3D plots of pollutant spread
- **Data files**: NetCDF and CSV format output

## Testing
Run test cases using:
```bash
# Run comprehensive test
python src/test_model.py

# Run specific simulation
python src/simulation_1.py
```

## Implementation Status

### ✅ Completed Modules
- [x] 3D Grid System (`grid3d.py`)
- [x] Pollution Field Management (`pollution_field.py`)
- [x] Advection Module (`advection_module.py`)
- [x] Diffusion Module (`diffusion_module.py`)
- [x] Decay Module (`decay_module.py`)
- [x] Chemical Reactions (`coupling_reaction_module.py`)
- [x] Biological Uptake (`bio_uptake_module.py`)
- [x] Precipitation (`precipitation_module.py`)
- [x] Source/Sink Terms (`source_sink_module.py`)
- [x] Boundary Conditions (`boundary_conditions_module.py`)
- [x] Output Module (`output_module.py`)
- [x] Main Model Class (`model.py`)
- [x] Test Framework (`test_model.py`)
- [x] Complete Simulation (`simulation_1.py`)
- [x] Data Integration (`data_select.py`)

### 🔧 Key Features Implemented
- [x] Modular design for extensibility
- [x] Environment-dependent process rates
- [x] Multiple boundary condition types
- [x] Real data integration
- [x] Comprehensive output system
- [x] Visualization capabilities
- [x] Statistical analysis
- [x] Unit conversion support
- [x] Geographic data handling

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or support, please contact the project maintainers. 