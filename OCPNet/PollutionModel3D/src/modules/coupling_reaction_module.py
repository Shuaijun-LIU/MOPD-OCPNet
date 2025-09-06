import numpy as np
from typing import Dict, List, Tuple, Optional
from grid3d import Grid3D
from pollution_field import PollutionField

class CouplingReactionModule:
    """
    Handles coupled reactions between pollutants.
    Supports multiple reaction types and environment-dependent reaction rates.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the coupling reaction module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        
        # Reaction definitions
        self.reactions: List[Dict] = []  # List of reaction dictionaries
        self.reaction_rates: Dict[str, float] = {}  # Base reaction rates
        self.temperature_dependencies: Dict[str, float] = {}  # Temperature coefficients
        self.ph_dependencies: Dict[str, Tuple[float, float]] = {}  # pH coefficients
        
    def add_reaction(self,
                    name: str,
                    reactants: List[str],
                    products: List[str],
                    stoichiometry: Dict[str, float],
                    base_rate: float,
                    temperature_coeff: float = 0.0,
                    ph_coeff: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Add a new chemical reaction.
        
        Args:
            name: Reaction name
            reactants: List of reactant names
            products: List of product names
            stoichiometry: Dictionary mapping species to stoichiometric coefficients
            base_rate: Base reaction rate constant
            temperature_coeff: Temperature coefficient for rate adjustment
            ph_coeff: (a, b) parameters for pH dependence: rate = rate * (1 + a*(pH-7) + b*(pH-7)^2)
        """
        reaction = {
            'name': name,
            'reactants': reactants,
            'products': products,
            'stoichiometry': stoichiometry
        }
        self.reactions.append(reaction)
        self.reaction_rates[name] = base_rate
        self.temperature_dependencies[name] = temperature_coeff
        self.ph_dependencies[name] = ph_coeff
        
    def compute_reaction_rate(self,
                            name: str,
                            temperature: np.ndarray,
                            ph: np.ndarray,
                            concentrations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the reaction rate considering environmental factors and concentrations.
        
        Args:
            name: Reaction name
            temperature: Temperature field (K)
            ph: pH field
            concentrations: Dictionary of concentration fields
            
        Returns:
            Reaction rate field
        """
        if name not in self.reaction_rates:
            raise ValueError(f"Reaction {name} not found")
            
        # Get base reaction rate
        k0 = self.reaction_rates[name]
        
        # Initialize reaction rate
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
        
        # Concentration effect (mass action kinetics)
        reaction = next(r for r in self.reactions if r['name'] == name)
        for reactant in reaction['reactants']:
            if reactant in concentrations:
                rate *= concentrations[reactant]
        
        return rate
    
    def compute_reaction_terms(self,
                             temperature: np.ndarray,
                             ph: np.ndarray,
                             concentrations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute reaction terms for all species.
        
        Args:
            temperature: Temperature field (K)
            ph: pH field
            concentrations: Dictionary of concentration fields
            
        Returns:
            Dictionary of reaction terms for each species
        """
        reaction_terms = {name: np.zeros_like(temperature) for name in concentrations.keys()}
        
        for reaction in self.reactions:
            # Compute reaction rate
            rate = self.compute_reaction_rate(
                reaction['name'],
                temperature,
                ph,
                concentrations
            )
            
            # Update reaction terms for each species
            for species, coeff in reaction['stoichiometry'].items():
                if species in concentrations:
                    reaction_terms[species] += coeff * rate
        
        return reaction_terms
    
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
    
    def add_mercury_methylation(self,
                              base_rate: float = 1e-7,
                              temperature_coeff: float = 0.1,
                              ph_coeff: Tuple[float, float] = (0.2, 0.01)) -> None:
        """
        Add mercury methylation reaction: Hg2+ -> CH3Hg+
        
        Args:
            base_rate: Base methylation rate
            temperature_coeff: Temperature coefficient
            ph_coeff: pH coefficients
        """
        self.add_reaction(
            name="mercury_methylation",
            reactants=["Hg2+"],
            products=["CH3Hg+"],
            stoichiometry={"Hg2+": -1, "CH3Hg+": 1},
            base_rate=base_rate,
            temperature_coeff=temperature_coeff,
            ph_coeff=ph_coeff
        )
    
    def add_phosphate_precipitation(self,
                                  base_rate: float = 1e-6,
                                  temperature_coeff: float = 0.05,
                                  ph_coeff: Tuple[float, float] = (0.3, 0.02)) -> None:
        """
        Add phosphate precipitation reaction: PO4(3-) + Me -> MePO4
        
        Args:
            base_rate: Base precipitation rate
            temperature_coeff: Temperature coefficient
            ph_coeff: pH coefficients
        """
        self.add_reaction(
            name="phosphate_precipitation",
            reactants=["PO4(3-)"],
            products=["MePO4"],
            stoichiometry={"PO4(3-)": -1, "MePO4": 1},
            base_rate=base_rate,
            temperature_coeff=temperature_coeff,
            ph_coeff=ph_coeff
        )
