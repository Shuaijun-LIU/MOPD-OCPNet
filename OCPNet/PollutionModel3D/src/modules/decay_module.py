import numpy as np
from typing import Dict, Optional, Tuple
from grid3d import Grid3D
from pollution_field import PollutionField

class DecayModule:
    """
    Handles pollutant decay processes with environment-dependent decay rates.
    Supports first-order decay and environment-dependent decay rates.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the decay module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        
        # Default decay parameters
        self.base_decay_rates: Dict[str, float] = {}  # Base decay rates for each pollutant
        self.activation_energies: Dict[str, float] = {}  # Activation energies for temperature dependence
        self.ph_dependencies: Dict[str, Tuple[float, float]] = {}  # pH dependence parameters
        self.do_dependencies: Dict[str, Tuple[float, float]] = {}  # DO dependence parameters
        
    def add_pollutant(self,
                     name: str,
                     base_decay_rate: float,
                     activation_energy: float = 0.0,
                     ph_dependence: Tuple[float, float] = (0.0, 0.0),
                     do_dependence: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Add a pollutant with its decay parameters.
        
        Args:
            name: Pollutant name
            base_decay_rate: Base decay rate (1/s)
            activation_energy: Activation energy for temperature dependence (J/mol)
            ph_dependence: (a, b) parameters for pH dependence: rate = rate * (1 + a*(pH-7) + b*(pH-7)^2)
            do_dependence: (c, d) parameters for DO dependence: rate = rate * (1 + c*DO + d*DO^2)
        """
        self.base_decay_rates[name] = base_decay_rate
        self.activation_energies[name] = activation_energy
        self.ph_dependencies[name] = ph_dependence
        self.do_dependencies[name] = do_dependence
        
    def compute_decay_rate(self,
                         name: str,
                         temperature: np.ndarray,
                         ph: np.ndarray,
                         dissolved_oxygen: np.ndarray) -> np.ndarray:
        """
        Compute the decay rate considering environmental factors.
        
        Args:
            name: Pollutant name
            temperature: Temperature field (K)
            ph: pH field
            dissolved_oxygen: Dissolved oxygen field (mg/L)
            
        Returns:
            Decay rate field (1/s)
        """
        if name not in self.base_decay_rates:
            raise ValueError(f"Pollutant {name} not found in decay parameters")
            
        # Get base decay rate
        lambda_0 = self.base_decay_rates[name]
        
        # Initialize decay rate field
        decay_rate = np.full_like(temperature, lambda_0)
        
        # Temperature effect (Arrhenius equation)
        if self.activation_energies[name] > 0:
            R = 8.314  # Universal gas constant (J/mol/K)
            E_a = self.activation_energies[name]
            decay_rate *= np.exp(-E_a / (R * temperature))
        
        # pH effect
        a, b = self.ph_dependencies[name]
        if a != 0 or b != 0:
            ph_deviation = ph - 7.0
            ph_factor = 1.0 + a * ph_deviation + b * ph_deviation**2
            decay_rate *= np.maximum(ph_factor, 0.0)  # Ensure non-negative rates
        
        # Dissolved oxygen effect
        c, d = self.do_dependencies[name]
        if c != 0 or d != 0:
            do_factor = 1.0 + c * dissolved_oxygen + d * dissolved_oxygen**2
            decay_rate *= np.maximum(do_factor, 0.0)  # Ensure non-negative rates
        
        return decay_rate
    
    def compute_decay_term(self,
                         name: str,
                         concentration: np.ndarray,
                         decay_rate: np.ndarray) -> np.ndarray:
        """
        Compute the decay term.
        
        Args:
            name: Pollutant name
            concentration: Pollutant concentration field
            decay_rate: Decay rate field
            
        Returns:
            Decay term array
        """
        return -decay_rate * concentration
    
    def apply_boundary_conditions(self,
                                concentration: np.ndarray,
                                decay_rate: np.ndarray) -> None:
        """
        Apply boundary conditions for decay.
        
        Args:
            concentration: Pollutant concentration field
            decay_rate: Decay rate field
        """
        # Apply zero-gradient boundary conditions
        concentration[0, :, :] = concentration[1, :, :]
        concentration[-1, :, :] = concentration[-2, :, :]
        concentration[:, 0, :] = concentration[:, 1, :]
        concentration[:, -1, :] = concentration[:, -2, :]
        concentration[:, :, 0] = concentration[:, :, 1]
        concentration[:, :, -1] = concentration[:, :, -2]
        
        # Apply zero-gradient to decay rate
        decay_rate[0, :, :] = decay_rate[1, :, :]
        decay_rate[-1, :, :] = decay_rate[-2, :, :]
        decay_rate[:, 0, :] = decay_rate[:, 1, :]
        decay_rate[:, -1, :] = decay_rate[:, -2, :]
        decay_rate[:, :, 0] = decay_rate[:, :, 1]
        decay_rate[:, :, -1] = decay_rate[:, :, -2]
    
    def compute_half_life(self,
                        name: str,
                        temperature: float = 298.15,
                        ph: float = 7.0,
                        dissolved_oxygen: float = 8.0) -> float:
        """
        Compute the half-life of a pollutant under given conditions.
        
        Args:
            name: Pollutant name
            temperature: Temperature (K)
            ph: pH
            dissolved_oxygen: Dissolved oxygen (mg/L)
            
        Returns:
            Half-life in seconds
        """
        if name not in self.base_decay_rates:
            raise ValueError(f"Pollutant {name} not found in decay parameters")
            
        # Compute decay rate
        decay_rate = self.compute_decay_rate(
            name,
            np.array([[temperature]]),
            np.array([[ph]]),
            np.array([[dissolved_oxygen]])
        )[0, 0]
        
        # Compute half-life
        return np.log(2) / decay_rate if decay_rate > 0 else float('inf')
