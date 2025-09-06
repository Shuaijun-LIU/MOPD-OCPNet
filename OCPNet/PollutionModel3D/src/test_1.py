import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .model import PollutionModel3D

def create_test_velocity_field(grid_shape):
    """
    Create a test velocity field with a simple circulation pattern.
    """
    nx, ny, nz = grid_shape
    
    # Create coordinate arrays
    x = np.linspace(0, 1000, nx)
    y = np.linspace(0, 1000, ny)
    z = np.linspace(0, 100, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create velocity components
    u = np.zeros(grid_shape)
    v = np.zeros(grid_shape)
    w = np.zeros(grid_shape)
    
    # Add a simple circulation pattern
    u = 0.1 * np.sin(2*np.pi*X/1000) * np.cos(2*np.pi*Y/1000)
    v = -0.1 * np.cos(2*np.pi*X/1000) * np.sin(2*np.pi*Y/1000)
    w = 0.01 * np.sin(2*np.pi*Z/100)
    
    return u, v, w

def create_test_environmental_fields(grid_shape):
    """
    Create test environmental fields (temperature, pH, DO, light_intensity, wave_velocity, salinity).
    """
    nx, ny, nz = grid_shape
    
    # Create coordinate arrays
    z = np.linspace(0, 100, nz)
    Z = np.tile(z, (nx, ny, 1))
    
    # Create environmental fields
    temperature = 20.0 + 5.0 * np.exp(-Z/20.0)  # Temperature decreases with depth
    ph = 7.0 + 0.5 * np.exp(-Z/30.0)  # pH decreases with depth
    do = 8.0 * np.exp(-Z/25.0)  # DO decreases with depth
    light_intensity = 1000.0 * np.exp(-Z/10.0)  # Light intensity decreases with depth
    wave_velocity = 0.1 * np.exp(-Z/15.0)  # Wave velocity decreases with depth
    salinity = 35.0 + 0.5 * np.exp(-Z/40.0)  # Salinity slightly increases with depth
    
    return {
        'temperature': temperature,
        'pH': ph,
        'DO': do,
        'light_intensity': light_intensity,
        'wave_velocity': wave_velocity,
        'salinity': salinity
    }

def test_model():
    """
    Test the pollution model with a complete simulation scenario.
    """
    # Create output directory
    output_dir = Path("../../output/test_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize model
    model = PollutionModel3D(
        domain_size=(1000.0, 1000.0, 100.0),  # 1000m x 1000m x 100m
        grid_resolution=(50, 50, 25),  # 50 x 50 x 25 grid
        time_step=60.0,  # 60s time step
        output_dir=output_dir
    )
    
    # Create and set velocity field
    u, v, w = create_test_velocity_field((50, 50, 25))
    model.set_velocity_field(u, v, w)
    
    # Create and set environmental fields
    environmental_fields = create_test_environmental_fields((50, 50, 25))
    for name, value in environmental_fields.items():
        model.set_environmental_field(name, value)
    
    # Add pollutants
    model.add_pollutant(
        name="NH4",
        initial_concentration=0.1,
        molecular_weight=18.0,
        decay_rate=1e-6,
        diffusion_coefficient=1e-9
    )
    
    model.add_pollutant(
        name="PO4",
        initial_concentration=0.05,
        molecular_weight=95.0,
        decay_rate=5e-7,
        diffusion_coefficient=8e-10
    )
    
    model.add_pollutant(
        name="Hg",
        initial_concentration=1e-6,
        molecular_weight=200.6,
        decay_rate=1e-7,
        diffusion_coefficient=5e-10
    )
    
    # Add reactions
    model.add_reaction(
        name="nitrification",
        reactants=["NH4", "O2"],
        products=["NO3", "H2O"],
        stoichiometry={"NH4": -1, "O2": -2, "NO3": 1, "H2O": 1},
        rate=1e-5
    )
    
    model.add_reaction(
        name="mercury_methylation",
        reactants=["Hg", "CH3"],
        products=["CH3Hg"],
        stoichiometry={"Hg": -1, "CH3": -1, "CH3Hg": 1},
        rate=1e-6
    )
    
    # Set phytoplankton parameters
    model.set_phytoplankton_parameters(
        growth_rate=1e-5,  # 1/s
        mortality_rate=1e-6,  # 1/s
        initial_biomass=0.1  # kg/m^3
    )
    
    # Add bio uptake
    model.add_bio_uptake(
        pollutant="PO4",
        max_uptake_rate=1e-5,
        half_saturation=0.01
    )
    
    # Add precipitation
    model.add_precipitation(
        cation="Fe",
        anion="PO4",
        solubility_product=1e-20,
        rate=1e-6
    )
    
    # Add sources
    model.add_source(
        type="point",
        pollutant="NH4",
        position=(500.0, 500.0, 0.0),
        emission_rate=1.0,
        time_function=lambda t: np.sin(2*np.pi*t/86400)  # Daily variation
    )
    
    model.add_source(
        type="area",
        pollutant="PO4",
        area=(0.0, 1000.0, 0.0, 1000.0),
        emission_rate=0.1,
        height=0.0
    )
    
    # Set boundary conditions
    model.set_boundary_condition(
        type="dirichlet",
        field="NH4",
        boundary="bottom",
        value=0.0
    )
    
    model.set_boundary_condition(
        type="neumann",
        field="PO4",
        boundary="top",
        gradient=0.0
    )
    
    model.set_boundary_condition(
        type="periodic",
        field="velocity",
        boundary="x",
        axis="x"
    )
    
    # Set output parameters
    model.set_output_parameters(
        output_fields=["NH4", "PO4", "Hg", "NO3", "CH3Hg"],
        output_interval=3600.0,  # Hourly output
        visualization_fields=["NH4", "PO4", "Hg"],
        visualization_interval=7200.0,  # Every 2 hours
        statistics_fields=["NH4", "PO4", "Hg"],
        statistics_interval=3600.0  # Hourly statistics
    )
    
    # Run simulation
    print("Starting simulation...")
    model.run(
        end_time=86400.0,  # 24 hours
        progress_interval=3600.0  # Report progress every hour
    )
    print("Simulation completed.")
    
    # Plot final concentrations
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, name in enumerate(["NH4", "PO4", "Hg"]):
        data = model.get_field(name)
        
        # XY plane at mid-depth
        axes[i, 0].imshow(data[:, :, 12], origin='lower')
        axes[i, 0].set_title(f"{name} (XY plane)")
        
        # XZ plane at mid-width
        axes[i, 1].imshow(data[:, 25, :], origin='lower')
        axes[i, 1].set_title(f"{name} (XZ plane)")
        
        # YZ plane at mid-length
        axes[i, 2].imshow(data[25, :, :], origin='lower')
        axes[i, 2].set_title(f"{name} (YZ plane)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_concentrations.png")
    plt.close()
    
    # Plot environmental fields
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, name in enumerate(["temperature", "pH", "DO"]):
        data = model.get_field(name)
        
        # XY plane at mid-depth
        axes[i, 0].imshow(data[:, :, 12], origin='lower')
        axes[i, 0].set_title(f"{name} (XY plane)")
        
        # XZ plane at mid-width
        axes[i, 1].imshow(data[:, 25, :], origin='lower')
        axes[i, 1].set_title(f"{name} (XZ plane)")
        
        # YZ plane at mid-length
        axes[i, 2].imshow(data[25, :, :], origin='lower')
        axes[i, 2].set_title(f"{name} (YZ plane)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "environmental_fields.png")
    plt.close()
    
    print(f"Test completed. Results saved to {output_dir}")

if __name__ == "__main__":
    test_model() 