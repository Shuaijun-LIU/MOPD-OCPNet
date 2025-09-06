import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
from grid3d import Grid3D
from pollution_field import PollutionField
from modules.advection_module import AdvectionModule
from modules.diffusion_module import DiffusionModule
from modules.decay_module import DecayModule
from modules.coupling_reaction_module import CouplingReactionModule
from modules.bio_uptake_module import BioUptakeModule
from modules.precipitation_module import PrecipitationModule
from modules.source_sink_module import SourceSinkModule
from modules.boundary_conditions_module import BoundaryConditionsModule
from modules.output_module import OutputModule

class PollutionModel3D:
    """
    Main 3D pollution model class that integrates all modules.
    """
    
    def __init__(self,
                 domain_size: Tuple[float, float, float],
                 grid_resolution: Tuple[int, int, int],
                 time_step: float,
                 output_dir: Union[str, Path]):
        """
        Initialize the pollution model.
        
        Args:
            domain_size: (Lx, Ly, Lz) domain size in meters
            grid_resolution: (nx, ny, nz) grid resolution
            time_step: Time step in seconds
            output_dir: Output directory path
        """
        # Calculate grid spacing
        dx = domain_size[0] / (grid_resolution[0] - 1)
        dy = domain_size[1] / (grid_resolution[1] - 1)
        dz = domain_size[2] / (grid_resolution[2] - 1)
        
        # Initialize grid
        self.grid = Grid3D(
            nx=grid_resolution[0],
            ny=grid_resolution[1],
            nz=grid_resolution[2],
            dx=dx,
            dy=dy,
            dz=dz
        )
        
        # Initialize other components
        self.time_step = time_step
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize modules
        self.advection = AdvectionModule(self.grid)
        self.diffusion = DiffusionModule(self.grid)
        self.decay = DecayModule(self.grid)
        self.coupling_reaction = CouplingReactionModule(self.grid)
        self.bio_uptake = BioUptakeModule(self.grid)
        self.precipitation = PrecipitationModule(self.grid)
        self.source_sink = SourceSinkModule(self.grid)
        self.boundary_conditions = BoundaryConditionsModule(self.grid)
        self.output = OutputModule(self.grid, self.output_dir)
        
        # Initialize fields
        self.velocity_field = None
        self.environmental_fields = {}
        self.pollutant_fields = {}
        
        # Initialize time
        self.current_time = 0.0
        
    def add_pollutant(self,
                     name: str,
                     initial_concentration: Union[float, np.ndarray],
                     molecular_weight: float,
                     decay_rate: Optional[float] = None,
                     diffusion_coefficient: Optional[float] = None) -> None:
        """
        Add a pollutant to the model.
        
        Args:
            name: Pollutant name
            initial_concentration: Initial concentration field
            molecular_weight: Molecular weight (g/mol)
            decay_rate: Optional decay rate (1/s)
            diffusion_coefficient: Optional diffusion coefficient (m^2/s)
        """
        # Create pollution field
        field = PollutionField(self.grid)
        field.add_pollutant(
            name=name,
            initial_value=initial_concentration,
            unit="kg/m^3"
        )
        
        # Add to pollutant fields dictionary
        self.pollutant_fields[name] = field
        
        # Add to decay module if decay rate is provided
        if decay_rate is not None:
            self.decay.add_pollutant(
                name=name,
                base_decay_rate=decay_rate,
                activation_energy=50000.0,  # Default activation energy (J/mol)
                ph_dependence=(0.3, 0.02),  # Default pH dependence
                do_dependence=(0.1, 0.01)   # Default DO dependence
            )
            
        # Store diffusion coefficient if provided
        if diffusion_coefficient is not None:
            if not hasattr(self, 'diffusion_coefficients'):
                self.diffusion_coefficients = {}
            self.diffusion_coefficients[name] = diffusion_coefficient
        
    def set_environmental_field(self,
                              name: str,
                              value: Union[float, np.ndarray]) -> None:
        """
        Set environmental field (temperature, pH, DO, etc.).
        
        Args:
            name: Field name
            value: Field value
        """
        self.environmental_fields[name] = value
        
    def set_velocity_field(self,
                          u: np.ndarray,
                          v: np.ndarray,
                          w: np.ndarray) -> None:
        """
        Set velocity field.
        
        Args:
            u: x-component of velocity
            v: y-component of velocity
            w: z-component of velocity
        """
        # Store velocity field
        self.velocity_field = (u, v, w)
        
        # Check CFL condition for stability
        cfl_x, cfl_y, cfl_z = self.advection.compute_cfl_numbers(u, v, w, self.time_step)
        if not self.advection.check_stability(u, v, w, self.time_step):
            raise ValueError(
                f"CFL condition violated: CFL_x={cfl_x:.3f}, CFL_y={cfl_y:.3f}, CFL_z={cfl_z:.3f}. "
                "Reduce time step or velocity magnitude."
            )
        
    def add_reaction(self,
                    name: str,
                    reactants: List[str],
                    products: List[str],
                    stoichiometry: List[float],
                    rate: float) -> None:
        """
        Add a chemical reaction.
        
        Args:
            name: Reaction name
            reactants: List of reactant names
            products: List of product names
            stoichiometry: List of stoichiometric coefficients
            rate: Reaction rate constant
        """
        self.coupling_reaction.add_reaction(
            name,
            reactants,
            products,
            stoichiometry,
            rate
        )
        
    def add_bio_uptake(self,
                      pollutant: str,
                      max_uptake_rate: float,
                      half_saturation: float,
                      temperature_coeff: float = 0.0,
                      light_coeff: float = 0.0) -> None:
        """
        Add a pollutant to the biological uptake module.
        
        Args:
            pollutant: Pollutant name
            max_uptake_rate: Maximum uptake rate (1/s)
            half_saturation: Half-saturation constant (kg/m^3)
            temperature_coeff: Temperature coefficient
            light_coeff: Light coefficient
        """
        self.bio_uptake.add_pollutant(
            pollutant,
            max_uptake_rate,
            half_saturation,
            temperature_coeff,
            light_coeff
        )
        
    def set_phytoplankton_parameters(self,
                                   growth_rate: float,
                                   mortality_rate: float,
                                   initial_biomass: float = 0.0) -> None:
        """
        Set phytoplankton growth parameters.
        
        Args:
            growth_rate: Phytoplankton growth rate (1/s)
            mortality_rate: Phytoplankton mortality rate (1/s)
            initial_biomass: Initial phytoplankton biomass (kg/m^3)
        """
        self.bio_uptake.set_phytoplankton_parameters(
            growth_rate,
            mortality_rate,
            initial_biomass
        )
        
    def add_precipitation(self,
                         cation: str,
                         anion: str,
                         solubility_product: float,
                         rate: float,
                         precipitate_density: float = 2000.0,
                         settling_velocity: float = 1e-4,
                         temperature_coeff: float = 0.0,
                         ph_coeff: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Add precipitation reaction.
        
        Args:
            cation: Cation name
            anion: Anion name
            solubility_product: Solubility product constant
            rate: Precipitation rate
            precipitate_density: Precipitate density (kg/m^3)
            settling_velocity: Settling velocity (m/s)
            temperature_coeff: Temperature coefficient
            ph_coeff: pH coefficient
        """
        self.precipitation.add_precipitation_reaction(
            name=f"{cation}_{anion}",
            cation=cation,
            anion=anion,
            solubility_product=solubility_product,
            reaction_rate=rate,
            precipitate_density=precipitate_density,
            settling_velocity=settling_velocity,
            temperature_coeff=temperature_coeff,
            ph_coeff=ph_coeff
        )
        
    def add_source(self,
                  type: str,
                  pollutant: str,
                  **kwargs) -> None:
        """
        Add a pollution source.
        
        Args:
            type: Source type ('point', 'area', 'line')
            pollutant: Pollutant name
            **kwargs: Source parameters
        """
        if type == 'point':
            self.source_sink.add_point_source(pollutant, **kwargs)
        elif type == 'area':
            self.source_sink.add_area_source(pollutant, **kwargs)
        elif type == 'line':
            self.source_sink.add_line_source(pollutant, **kwargs)
            
    def set_boundary_condition(self,
                              type: str,
                              field: str,
                              boundary: str,
                              **kwargs) -> None:
        """
        Set boundary condition.
        
        Args:
            type: Boundary condition type ('dirichlet', 'neumann', 'periodic', 'open')
            field: Field name
            boundary: Boundary name
            **kwargs: Boundary condition parameters
        """
        if type == 'dirichlet':
            self.boundary_conditions.set_dirichlet_boundary(field, boundary, **kwargs)
        elif type == 'neumann':
            self.boundary_conditions.set_neumann_boundary(field, boundary, **kwargs)
        elif type == 'periodic':
            self.boundary_conditions.set_periodic_boundary(field, boundary)
        elif type == 'open':
            self.boundary_conditions.set_open_boundary(field, boundary, **kwargs)
            
    def set_output_parameters(self,
                            output_fields: List[str],
                            output_interval: float,
                            visualization_fields: List[str],
                            visualization_interval: float,
                            statistics_fields: List[str],
                            statistics_interval: float) -> None:
        """
        Set output parameters.
        
        Args:
            output_fields: List of fields to output
            output_interval: Output interval in seconds
            visualization_fields: List of fields to visualize
            visualization_interval: Visualization interval in seconds
            statistics_fields: List of fields for statistics
            statistics_interval: Statistics interval in seconds
        """
        self.output.set_output_fields(output_fields)
        self.output.set_output_interval(output_interval)
        self.output.set_visualization_fields(visualization_fields)
        self.output.set_visualization_interval(visualization_interval)
        self.output.set_statistics_fields(statistics_fields)
        self.output.set_statistics_interval(statistics_interval)
        
    def compute_time_step(self) -> None:
        """
        Compute one time step of the model.
        """
        # Get current concentrations
        concentrations = {name: field.get_concentration(name) for name, field in self.pollutant_fields.items()}
        
        # Get velocity field
        u, v, w = self.velocity_field
        
        # Compute advection terms
        advection_terms = {}
        for name, concentration in concentrations.items():
            advection_terms[name] = self.advection.compute_advection_term(concentration, u, v, w)
        
        # Compute diffusion terms
        diffusion_terms = {}
        for name, concentration in concentrations.items():
            # Compute diffusion coefficient
            diffusion_coefficient = self.diffusion.compute_diffusion_coefficient(
                self.environmental_fields["temperature"],
                self.environmental_fields["wave_velocity"],
                self.environmental_fields["salinity"],
                self.diffusion_coefficients.get(name, 1e-6)
            )
            diffusion_terms[name] = self.diffusion.compute_diffusion_term(concentration, diffusion_coefficient)
        
        # Compute decay terms
        decay_terms = {}
        for name, concentration in concentrations.items():
            # Compute decay rate
            decay_rate = self.decay.compute_decay_rate(
                name,
                self.environmental_fields["temperature"],
                self.environmental_fields["pH"],
                self.environmental_fields["DO"]
            )
            decay_terms[name] = self.decay.compute_decay_term(name, concentration, decay_rate)
        
        # Compute reaction terms
        reaction_terms = self.coupling_reaction.compute_reaction_terms(
            self.environmental_fields["temperature"],
            self.environmental_fields["pH"],
            concentrations
        )
        
        # Compute bio uptake terms
        bio_terms = self.bio_uptake.compute_uptake_terms(
            concentrations,
            self.environmental_fields["temperature"],
            self.environmental_fields["light_intensity"],
            self.bio_uptake.phytoplankton_biomass
        )
        
        # Compute precipitation terms
        precipitation_terms = self.precipitation.compute_precipitation_terms(
            concentrations,
            self.environmental_fields["temperature"],
            self.environmental_fields["pH"]
        )
        
        # Compute source-sink terms
        source_sink_terms = self.source_sink.compute_source_sink_terms(
            concentrations,
            self.current_time,
            self.environmental_fields
        )
        
        # Update concentrations
        dt = self.time_step
        for name, field in self.pollutant_fields.items():
            # Combine all terms
            total_change = (
                advection_terms[name] +
                diffusion_terms[name] +
                decay_terms[name] +
                reaction_terms.get(name, 0) +
                bio_terms.get(name, 0) +
                precipitation_terms.get(name, 0) +
                source_sink_terms.get(name, 0)
            )
            
            # Update concentration
            new_concentration = field.get_concentration(name) + dt * total_change
            field.set_concentration(name, new_concentration)
            
            # Apply boundary conditions
            self.boundary_conditions.apply_boundary_conditions(
                field=name,
                data=new_concentration,
                time=self.current_time
            )
            
        # Update time
        self.current_time += dt
        
    def run(self, end_time, progress_interval=3600.0):
        """
        Run the simulation until the specified end time
        
        Args:
            end_time (float): Time to end the simulation (seconds)
            progress_interval (float): Interval for progress reporting (seconds)
        """
        print("\n=== Simulation Progress ===")
        print(f"Starting time: {self.current_time:.1f} seconds")
        print(f"Target end time: {end_time:.1f} seconds")
        print(f"Progress reporting every {progress_interval:.1f} seconds")
        
        next_progress_time = self.current_time + progress_interval
        next_output_time = self.current_time + self.output.output_interval
        next_statistics_time = self.current_time + self.output.statistics_interval
        
        while self.current_time < end_time:
            # Compute one time step
            self.compute_time_step()
            
            # Check if it's time to report progress
            if self.current_time >= next_progress_time:
                elapsed_time = self.current_time - (next_progress_time - progress_interval)
                total_time = end_time - (next_progress_time - progress_interval)
                progress = (elapsed_time / total_time) * 100
                
                print(f"\nProgress at {self.current_time:.1f} seconds ({progress:.1f}%):")
                print(f"Elapsed time: {self.current_time/3600:.1f} hours")
                print(f"Remaining time: {(end_time - self.current_time)/3600:.1f} hours")
                
                # Print current concentrations
                print("\nCurrent concentrations:")
                for pollutant in self.pollutant_fields:
                    conc = self.pollutant_fields[pollutant].get_concentration(pollutant)
                    print(f"{pollutant}: min={np.min(conc):.3e}, max={np.max(conc):.3e}, mean={np.mean(conc):.3e} mg/L")
                
                # Print reaction rates
                print("\nCurrent reaction rates:")
                for reaction in self.coupling_reaction.reactions:
                    name = reaction['name']
                    rate = self.coupling_reaction.reaction_rates[name]
                    print(f"{name}: {rate:.3e} 1/s")
                
                # Print source emissions
                print("\nCurrent source emissions:")
                # Point sources
                for source in self.source_sink.point_sources:
                    rate = source['emission_rate']
                    if source['time_function'] is not None:
                        rate *= source['time_function'](self.current_time)
                    print(f"Point source ({source['pollutant']}): {rate:.3e} mg/s")
                
                # Area sources
                for source in self.source_sink.area_sources:
                    rate = source['emission_rate']
                    if source['time_function'] is not None:
                        rate *= source['time_function'](self.current_time)
                    print(f"Area source ({source['pollutant']}): {rate:.3e} mg/s/m^2")
                
                # Line sources
                for source in self.source_sink.line_sources:
                    rate = source['emission_rate']
                    if source['time_function'] is not None:
                        rate *= source['time_function'](self.current_time)
                    print(f"Line source ({source['pollutant']}): {rate:.3e} mg/s/m")
                
                next_progress_time += progress_interval
            
            # Check if it's time to save output
            if self.current_time >= next_output_time:
                print(f"\nSaving output at {self.current_time:.1f} seconds...")
                concentrations = {name: field.get_concentration(name) for name, field in self.pollutant_fields.items()}
                self.output.process_output(concentrations, self.current_time)
                next_output_time += self.output.output_interval
            
            # Check if it's time to compute statistics
            if self.current_time >= next_statistics_time:
                print(f"\nComputing statistics at {self.current_time:.1f} seconds...")
                concentrations = {name: field.get_concentration(name) for name, field in self.pollutant_fields.items()}
                self.output.compute_statistics(concentrations, self.current_time)
                next_statistics_time += self.output.statistics_interval
        
        print("\n=== Simulation Completed ===")
        print(f"Final time: {self.current_time:.1f} seconds")
        print(f"Total simulation time: {self.current_time/3600:.1f} hours")
        
        # Print final concentrations
        print("\nFinal concentrations:")
        for pollutant in self.pollutant_fields:
            conc = self.pollutant_fields[pollutant].get_concentration(pollutant)
            print(f"{pollutant}: min={np.min(conc):.3e}, max={np.max(conc):.3e}, mean={np.mean(conc):.3e} mg/L")
        
        # Print total mass balance
        print("\nTotal mass balance:")
        for pollutant in self.pollutant_fields:
            initial_mass = self.pollutant_fields[pollutant].initial_mass
            final_mass = np.sum(self.pollutant_fields[pollutant].get_concentration(pollutant) * self.grid.volume)
            print(f"{pollutant}: Initial={initial_mass:.3e} mg, Final={final_mass:.3e} mg, Change={(final_mass-initial_mass)/initial_mass*100:.1f}%")
        
        # Save final output
        print("\nSaving final output...")
        concentrations = {name: field.get_concentration(name) for name, field in self.pollutant_fields.items()}
        self.output.process_output(concentrations, self.current_time)
        self.output.compute_statistics(concentrations, self.current_time)
        
    def get_field(self, name: str) -> np.ndarray:
        """
        Get field data.
        
        Args:
            name: Field name
            
        Returns:
            Field data array
        """
        if name in self.pollutant_fields:
            return self.pollutant_fields[name].get_concentration(name)
        elif name in self.environmental_fields:
            return self.environmental_fields[name]
        else:
            raise ValueError(f"Field {name} not found")
            
    def get_time(self) -> float:
        """
        Get current simulation time.
        
        Returns:
            Current time in seconds
        """
        return self.current_time 