import numpy as np
from typing import Dict, List, Tuple, Optional
from grid3d import Grid3D
from pollution_field import PollutionField

class PrecipitationModule:
    """
    Handles precipitation reactions of metal ions and anions.
    Supports multiple precipitation types and solubility products.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the precipitation module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        
        # Precipitation parameters
        self.solubility_products: Dict[str, float] = {}  # Ksp values
        self.reaction_rates: Dict[str, float] = {}  # Precipitation rates
        self.temperature_dependencies: Dict[str, float] = {}  # Temperature coefficients
        self.ph_dependencies: Dict[str, Tuple[float, float]] = {}  # pH coefficients
        
        # Precipitate properties
        self.precipitate_densities: Dict[str, float] = {}  # Density of precipitates
        self.settling_velocities: Dict[str, float] = {}  # Settling velocities
        
    def add_precipitation_reaction(self,
                                 name: str,
                                 cation: str,
                                 anion: str,
                                 solubility_product: float,
                                 reaction_rate: float,
                                 precipitate_density: float,
                                 settling_velocity: float,
                                 temperature_coeff: float = 0.0,
                                 ph_coeff: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Add a precipitation reaction.
        
        Args:
            name: Reaction name
            cation: Cation name
            anion: Anion name
            solubility_product: Solubility product constant
            reaction_rate: Precipitation rate constant
            precipitate_density: Density of precipitate (kg/m^3)
            settling_velocity: Settling velocity (m/s)
            temperature_coeff: Temperature coefficient
            ph_coeff: (a, b) parameters for pH dependence: rate = rate * (1 + a*(pH-7) + b*(pH-7)^2)
        """
        self.solubility_products[name] = solubility_product
        self.reaction_rates[name] = reaction_rate
        self.temperature_dependencies[name] = temperature_coeff
        self.ph_dependencies[name] = ph_coeff
        self.precipitate_densities[name] = precipitate_density
        self.settling_velocities[name] = settling_velocity
        
    def compute_saturation_index(self,
                               name: str,
                               cation_concentration: np.ndarray,
                               anion_concentration: np.ndarray) -> np.ndarray:
        """
        Compute saturation index (SI) for a precipitation reaction.
        
        Args:
            name: Reaction name
            cation_concentration: Cation concentration field
            anion_concentration: Anion concentration field
            
        Returns:
            Saturation index field
        """
        if name not in self.solubility_products:
            raise ValueError(f"Reaction {name} not found")
            
        Ksp = self.solubility_products[name]
        ion_product = cation_concentration * anion_concentration
        return ion_product / Ksp
    
    def compute_precipitation_rate(self,
                                 name: str,
                                 saturation_index: np.ndarray,
                                 temperature: np.ndarray,
                                 ph: np.ndarray) -> np.ndarray:
        """
        Compute precipitation rate considering environmental factors.
        
        Args:
            name: Reaction name
            saturation_index: Saturation index field
            temperature: Temperature field (K)
            ph: pH field
            
        Returns:
            Precipitation rate field
        """
        if name not in self.reaction_rates:
            raise ValueError(f"Reaction {name} not found")
            
        # Get base reaction rate
        k0 = self.reaction_rates[name]
        
        # Initialize precipitation rate
        rate = np.full_like(temperature, k0)
        
        # Temperature effect
        alpha_T = self.temperature_dependencies[name]
        if alpha_T != 0:
            rate *= np.exp(alpha_T * (temperature - 298.15))
        
        # pH effect
        a, b = self.ph_dependencies[name]
        if a != 0 or b != 0:
            ph_deviation = ph - 7.0
            ph_factor = 1.0 + a * ph_deviation + b * ph_deviation**2
            rate *= np.maximum(ph_factor, 0.0)
        
        # Saturation effect (only precipitate when SI > 1)
        rate *= np.maximum(saturation_index - 1.0, 0.0)
        
        return rate
    
    def compute_settling_term(self,
                            name: str,
                            precipitate_concentration: np.ndarray) -> np.ndarray:
        """
        Compute settling term for precipitate.
        
        Args:
            name: Reaction name
            precipitate_concentration: Precipitate concentration field
            
        Returns:
            Settling term field
        """
        if name not in self.settling_velocities:
            raise ValueError(f"Reaction {name} not found")
            
        w = self.settling_velocities[name]
        dz = self.grid.get_grid_spacing()[2]
        
        # Initialize settling term
        settling_term = np.zeros_like(precipitate_concentration)
        
        # Compute vertical settling
        settling_term[:, :, 1:] = w * (precipitate_concentration[:, :, 1:] - precipitate_concentration[:, :, :-1]) / dz
        settling_term[:, :, 0] = w * precipitate_concentration[:, :, 0] / dz
        
        return settling_term
    
    def compute_precipitation_terms(self,
                                  concentrations: Dict[str, np.ndarray],
                                  temperature: np.ndarray,
                                  ph: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute precipitation terms for all species.
        
        Args:
            concentrations: Dictionary of concentration fields
            temperature: Temperature field (K)
            ph: pH field
            
        Returns:
            Dictionary of precipitation terms for each species
        """
        precipitation_terms = {name: np.zeros_like(temperature) for name in concentrations.keys()}
        
        for name in self.solubility_products.keys():
            # Get cation and anion names from reaction name
            cation, anion = name.split('_')
            
            if cation in concentrations and anion in concentrations:
                # Compute saturation index
                SI = self.compute_saturation_index(
                    name,
                    concentrations[cation],
                    concentrations[anion]
                )
                
                # Compute precipitation rate
                rate = self.compute_precipitation_rate(
                    name,
                    SI,
                    temperature,
                    ph
                )
                
                # Update terms for reactants
                precipitation_terms[cation] -= rate
                precipitation_terms[anion] -= rate
                
                # Update term for precipitate
                precipitate_name = f"{name}_precipitate"
                if precipitate_name in concentrations:
                    precipitation_terms[precipitate_name] += rate
                    
                    # Add settling term
                    settling_term = self.compute_settling_term(
                        name,
                        concentrations[precipitate_name]
                    )
                    precipitation_terms[precipitate_name] -= settling_term
        
        return precipitation_terms
    
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
    
    def add_phosphate_precipitation(self,
                                  base_rate: float = 1e-6,
                                  temperature_coeff: float = 0.05,
                                  ph_coeff: Tuple[float, float] = (0.3, 0.02)) -> None:
        """
        Add phosphate precipitation reaction: Me + PO4(3-) -> MePO4
        
        Args:
            base_rate: Base precipitation rate
            temperature_coeff: Temperature coefficient
            ph_coeff: pH coefficients
        """
        self.add_precipitation_reaction(
            name="Me_PO4",
            cation="Me",
            anion="PO4(3-)",
            solubility_product=1e-20,
            reaction_rate=base_rate,
            precipitate_density=2500.0,
            settling_velocity=1e-4,
            temperature_coeff=temperature_coeff,
            ph_coeff=ph_coeff
        )
    
    def add_metal_hydroxide_precipitation(self,
                                        metal: str,
                                        base_rate: float = 1e-5,
                                        temperature_coeff: float = 0.1,
                                        ph_coeff: Tuple[float, float] = (0.4, 0.03)) -> None:
        """
        Add metal hydroxide precipitation reaction: Me + OH- -> Me(OH)
        
        Args:
            metal: Metal name (e.g., "Fe", "Al")
            base_rate: Base precipitation rate
            temperature_coeff: Temperature coefficient
            ph_coeff: pH coefficients
        """
        self.add_precipitation_reaction(
            name=f"{metal}_OH",
            cation=metal,
            anion="OH-",
            solubility_product=1e-15,
            reaction_rate=base_rate,
            precipitate_density=3000.0,
            settling_velocity=2e-4,
            temperature_coeff=temperature_coeff,
            ph_coeff=ph_coeff
        )
