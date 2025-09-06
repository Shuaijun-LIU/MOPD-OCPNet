import numpy as np
from typing import Dict, List, Tuple, Optional
from grid3d import Grid3D
from pollution_field import PollutionField

class BioUptakeModule:
    """
    Handles biological uptake of pollutants by phytoplankton.
    Supports multiple nutrient types and Michaelis-Menten kinetics.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the biological uptake module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        
        # Uptake parameters
        self.max_uptake_rates: Dict[str, float] = {}  # Maximum uptake rates
        self.half_saturation_constants: Dict[str, float] = {}  # Half-saturation constants
        self.temperature_dependencies: Dict[str, float] = {}  # Temperature coefficients
        self.light_dependencies: Dict[str, float] = {}  # Light coefficients
        
        # Phytoplankton parameters
        self.phytoplankton_growth_rate: float = 0.0
        self.phytoplankton_mortality_rate: float = 0.0
        self.phytoplankton_biomass: Optional[np.ndarray] = None
        
    def add_pollutant(self,
                     name: str,
                     max_uptake_rate: float,
                     half_saturation: float,
                     temperature_coeff: float = 0.0,
                     light_coeff: float = 0.0) -> None:
        """
        Add a pollutant with its uptake parameters.
        
        Args:
            name: Pollutant name
            max_uptake_rate: Maximum uptake rate (1/s)
            half_saturation: Half-saturation constant (kg/m^3)
            temperature_coeff: Temperature coefficient
            light_coeff: Light coefficient
        """
        self.max_uptake_rates[name] = max_uptake_rate
        self.half_saturation_constants[name] = half_saturation
        self.temperature_dependencies[name] = temperature_coeff
        self.light_dependencies[name] = light_coeff
        
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
        self.phytoplankton_growth_rate = growth_rate
        self.phytoplankton_mortality_rate = mortality_rate
        self.phytoplankton_biomass = np.full((self.nx, self.ny, self.nz), initial_biomass)
        
    def compute_uptake_rate(self,
                          name: str,
                          concentration: np.ndarray,
                          temperature: np.ndarray,
                          light_intensity: np.ndarray,
                          phytoplankton_biomass: np.ndarray) -> np.ndarray:
        """
        Compute uptake rate using Michaelis-Menten kinetics.
        
        Args:
            name: Pollutant name
            concentration: Pollutant concentration field
            temperature: Temperature field (K)
            light_intensity: Light intensity field (W/m^2)
            phytoplankton_biomass: Phytoplankton biomass field
            
        Returns:
            Uptake rate field
        """
        if name not in self.max_uptake_rates:
            raise ValueError(f"Pollutant {name} not found in uptake parameters")
            
        # Get parameters
        V_max = self.max_uptake_rates[name]
        K_s = self.half_saturation_constants[name]
        alpha_T = self.temperature_dependencies[name]
        alpha_L = self.light_dependencies[name]
        
        # Compute temperature effect
        T_effect = np.exp(alpha_T * (temperature - 298.15))
        
        # Compute light effect
        L_effect = 1.0 + alpha_L * light_intensity
        
        # Compute Michaelis-Menten term
        mm_term = concentration / (K_s + concentration)
        
        # Compute total uptake rate
        uptake_rate = V_max * T_effect * L_effect * mm_term * phytoplankton_biomass
        
        return uptake_rate
    
    def compute_uptake_terms(self,
                           concentrations: Dict[str, np.ndarray],
                           temperature: np.ndarray,
                           light_intensity: np.ndarray,
                           phytoplankton_biomass: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute uptake terms for all pollutants.
        
        Args:
            concentrations: Dictionary of concentration fields
            temperature: Temperature field (K)
            light_intensity: Light intensity field (W/m^2)
            phytoplankton_biomass: Phytoplankton biomass field
            
        Returns:
            Dictionary of uptake terms for each pollutant
        """
        uptake_terms = {name: np.zeros_like(temperature) for name in concentrations.keys()}
        
        for name, concentration in concentrations.items():
            if name in self.max_uptake_rates:
                uptake_rate = self.compute_uptake_rate(
                    name,
                    concentration,
                    temperature,
                    light_intensity,
                    phytoplankton_biomass
                )
                uptake_terms[name] = -uptake_rate  # Negative because it's a sink
        
        return uptake_terms
    
    def update_phytoplankton_biomass(self,
                                   dt: float,
                                   temperature: np.ndarray,
                                   light_intensity: np.ndarray,
                                   nutrient_concentrations: Dict[str, np.ndarray]) -> None:
        """
        Update phytoplankton biomass based on growth and mortality.
        
        Args:
            dt: Time step
            temperature: Temperature field (K)
            light_intensity: Light intensity field (W/m^2)
            nutrient_concentrations: Dictionary of nutrient concentration fields
        """
        if self.phytoplankton_biomass is None:
            raise ValueError("Phytoplankton parameters not set")
            
        # Compute growth rate
        growth = self.phytoplankton_growth_rate * np.exp(0.1 * (temperature - 298.15))
        growth *= (1.0 + 0.05 * light_intensity)  # Light effect
        
        # Compute mortality
        mortality = self.phytoplankton_mortality_rate
        
        # Update biomass
        self.phytoplankton_biomass *= (1.0 + dt * (growth - mortality))
        
        # Ensure non-negative biomass
        self.phytoplankton_biomass = np.maximum(self.phytoplankton_biomass, 0.0)
    
    def apply_boundary_conditions(self,
                                concentrations: Dict[str, np.ndarray]) -> None:
        """
        Apply boundary conditions for all concentration fields.
        
        Args:
            concentrations: Dictionary of concentration fields
        """
        for concentration in concentrations.values():
            # Apply zero-gradient boundary conditions
            concentration[0, :, :] = concentration[1, :, :]
            concentration[-1, :, :] = concentration[-2, :, :]
            concentration[:, 0, :] = concentration[:, 1, :]
            concentration[:, -1, :] = concentration[:, -2, :]
            concentration[:, :, 0] = concentration[:, :, 1]
            concentration[:, :, -1] = concentration[:, :, -2]
        
        # Apply boundary conditions to phytoplankton biomass
        if self.phytoplankton_biomass is not None:
            self.phytoplankton_biomass[0, :, :] = self.phytoplankton_biomass[1, :, :]
            self.phytoplankton_biomass[-1, :, :] = self.phytoplankton_biomass[-2, :, :]
            self.phytoplankton_biomass[:, 0, :] = self.phytoplankton_biomass[:, 1, :]
            self.phytoplankton_biomass[:, -1, :] = self.phytoplankton_biomass[:, -2, :]
            self.phytoplankton_biomass[:, :, 0] = self.phytoplankton_biomass[:, :, 1]
            self.phytoplankton_biomass[:, :, -1] = self.phytoplankton_biomass[:, :, -2]
