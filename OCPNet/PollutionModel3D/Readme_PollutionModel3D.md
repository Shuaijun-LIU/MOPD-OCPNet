# PollutionModel3D: A Comprehensive Framework for Ocean Pollution Simulation

## Overview
PollutionModel3D is a three-dimensional ocean pollution simulation framework developed as part of the "A Large-Scale Oceanographic Dataset and Prediction Framework for Ocean Currents and Pollution Dispersion" project. The framework integrates physical, chemical, and biological processes to provide comprehensive ocean pollution prediction capabilities.

## Project Structure
```
PollutionModel3D/
├── src/
│   ├── main.py              # Main entry point for the model
│   ├── model.py             # Core model implementation
│   ├── grid3d.py            # 3D grid system implementation
│   ├── pollution_field.py   # Pollution field implementation
│   ├── data_select.py       # Data selection and preprocessing
│   ├── test_model.py        # Test cases for model validation
│   ├── __init__.py          # Package initialization
│   └── modules/             # Process modules
│       ├── advection_module.py      # Advection process module
│       ├── diffusion_module.py      # Diffusion process module
│       ├── boundary_conditions_module.py  # Boundary conditions handling
│       ├── coupling_reaction_module.py    # Chemical reactions module
│       ├── precipitation_module.py        # Precipitation process module
│       ├── bio_uptake_module.py          # Biological uptake module
│       ├── decay_module.py               # Decay process module
│       ├── source_sink_module.py         # Source and sink terms module
│       └── output_module.py              # Output and visualization module
├── data/                   # Data directory for input/output
└── setup.py               # Package installation configuration
```

## Key Features

### Physical Processes
- **Advection**: Simulates pollutant transport with water flow
- **Diffusion**: Models molecular and turbulent diffusion
- **Boundary Conditions**: Supports various boundary types (Dirichlet, Neumann, periodic, open)

### Chemical Processes
- **Reactions**: Handles multi-component chemical reactions
- **Precipitation**: Models precipitation-dissolution processes
- **Environmental Response**: Considers temperature, pH, and dissolved oxygen effects

### Biological Processes
- **Phytoplankton Uptake**: Models biological absorption
- **Decay**: Simulates natural degradation processes
- **Environmental Factors**: Incorporates light, temperature, and nutrient effects

### Source and Sink Terms
- **Point Sources**: Handles discrete emission sources
- **Area Sources**: Manages distributed pollution inputs
- **Sink Processes**: Models sedimentation, degradation, and absorption

## Mathematical Framework

### Advection Equation
\[
\frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} + w\frac{\partial C}{\partial z} = 0
\]

### Diffusion Equation
\[
\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}(D_x\frac{\partial C}{\partial x}) + \frac{\partial}{\partial y}(D_y\frac{\partial C}{\partial y}) + \frac{\partial}{\partial z}(D_z\frac{\partial C}{\partial z})
\]

### Chemical Reactions
\[
\frac{dC_i}{dt} = \sum_{j=1}^{N} \nu_{ij} r_j
\]
\[
r_j = k_j \prod_{i=1}^{M} C_i^{\alpha_{ij}}
\]

### Biological Processes
\[
R_b = k_b B f(T) f(L) C
\]
\[
f(T) = e^{-\frac{E_a}{R}(\frac{1}{T} - \frac{1}{T_{opt}})}
\]
\[
f(L) = \frac{L}{K_L + L}
\]

## Usage

### Installation
1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. Install the package:
   ```bash
   python setup.py install
   ```

### Running the Model
1. Prepare input data in the specified format
2. Configure model parameters in `main.py`
3. Run the model:
   ```bash
   python src/main.py
   ```

### Data Requirements
The model requires the following input data:
- Grid parameters (domain size, resolution)
- Velocity field (u, v, w components)
- Environmental fields (temperature, salinity, etc.)
- Pollutant parameters (initial concentrations, decay rates)
- Source terms (emission rates, locations)

## Output
The model generates:
- Concentration fields for each pollutant
- Time series of pollutant distributions
- Statistical analysis of results
- Visualization of pollutant spread

## Testing
Run test cases using:
```bash
python src/test_model.py
```

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