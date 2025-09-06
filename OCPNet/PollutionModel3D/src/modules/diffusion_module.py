import numpy as np
from typing import Tuple, Optional
from grid3d import Grid3D
from pollution_field import PollutionField

class DiffusionModule:
    """
    Handles pollutant diffusion with environment-dependent coefficients.
    Implements central difference scheme for diffusion terms.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the diffusion module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        self.dx, self.dy, self.dz = grid.get_grid_spacing()
        
    def compute_diffusion_coefficient(self,
                                    temperature: np.ndarray,
                                    wave_velocity: np.ndarray,
                                    salinity: np.ndarray,
                                    base_diffusion: float = 1e-6) -> np.ndarray:
        """
        Compute diffusion coefficient based on environmental variables.
        
        Args:
            temperature: Temperature field (K)
            wave_velocity: Wave velocity field (m/s)
            salinity: Salinity field (PSU)
            base_diffusion: Base diffusion coefficient (m^2/s)
            
        Returns:
            Diffusion coefficient field
        """
        # Temperature effect
        alpha_T = 0.02  # Temperature coefficient
        K_T = base_diffusion * (1 + alpha_T * (temperature - 273.15))
        
        # Wave effect
        beta = 0.1  # Wave coefficient
        K_wave = K_T + beta * wave_velocity
        
        # Salinity effect
        alpha_s = 0.05  # Salinity coefficient
        K = K_wave * (1 + alpha_s * (1 - salinity/35))
        
        return K
    
    def compute_diffusion_term(self,
                             concentration: np.ndarray,
                             diffusion_coefficient: np.ndarray) -> np.ndarray:
        """
        Compute the diffusion term using central differences.
        
        Args:
            concentration: Pollutant concentration field
            diffusion_coefficient: Diffusion coefficient field
            
        Returns:
            Diffusion term array
        """
        # Initialize diffusion term
        diff_term = np.zeros_like(concentration)
        
        # Compute x-direction diffusion
        diff_term[1:-1, :, :] += (
            (diffusion_coefficient[1:-1, :, :] * (concentration[2:, :, :] - concentration[1:-1, :, :]) / self.dx
             - diffusion_coefficient[:-2, :, :] * (concentration[1:-1, :, :] - concentration[:-2, :, :]) / self.dx)
            / self.dx
        )
        
        # Compute y-direction diffusion
        diff_term[:, 1:-1, :] += (
            (diffusion_coefficient[:, 1:-1, :] * (concentration[:, 2:, :] - concentration[:, 1:-1, :]) / self.dy
             - diffusion_coefficient[:, :-2, :] * (concentration[:, 1:-1, :] - concentration[:, :-2, :]) / self.dy)
            / self.dy
        )
        
        # Compute z-direction diffusion
        diff_term[:, :, 1:-1] += (
            (diffusion_coefficient[:, :, 1:-1] * (concentration[:, :, 2:] - concentration[:, :, 1:-1]) / self.dz
             - diffusion_coefficient[:, :, :-2] * (concentration[:, :, 1:-1] - concentration[:, :, :-2]) / self.dz)
            / self.dz
        )
        
        return diff_term
    
    def apply_boundary_conditions(self,
                                concentration: np.ndarray,
                                diffusion_coefficient: np.ndarray) -> None:
        """
        Apply boundary conditions for diffusion.
        
        Args:
            concentration: Pollutant concentration field
            diffusion_coefficient: Diffusion coefficient field
        """
        # Apply zero-gradient boundary conditions
        concentration[0, :, :] = concentration[1, :, :]
        concentration[-1, :, :] = concentration[-2, :, :]
        concentration[:, 0, :] = concentration[:, 1, :]
        concentration[:, -1, :] = concentration[:, -2, :]
        concentration[:, :, 0] = concentration[:, :, 1]
        concentration[:, :, -1] = concentration[:, :, -2]
        
        # Apply zero-gradient to diffusion coefficient
        diffusion_coefficient[0, :, :] = diffusion_coefficient[1, :, :]
        diffusion_coefficient[-1, :, :] = diffusion_coefficient[-2, :, :]
        diffusion_coefficient[:, 0, :] = diffusion_coefficient[:, 1, :]
        diffusion_coefficient[:, -1, :] = diffusion_coefficient[:, -2, :]
        diffusion_coefficient[:, :, 0] = diffusion_coefficient[:, :, 1]
        diffusion_coefficient[:, :, -1] = diffusion_coefficient[:, :, -2]
    
    def compute_stability_criterion(self,
                                 diffusion_coefficient: np.ndarray,
                                 dt: float) -> float:
        """
        Compute stability criterion for diffusion.
        
        Args:
            diffusion_coefficient: Diffusion coefficient field
            dt: Time step
            
        Returns:
            Maximum stable time step
        """
        max_K = np.max(diffusion_coefficient)
        min_dx = min(self.dx, self.dy, self.dz)
        return 0.5 * min_dx**2 / max_K
