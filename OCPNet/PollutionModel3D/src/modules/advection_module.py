import numpy as np
from typing import Tuple, Optional
from grid3d import Grid3D
from pollution_field import PollutionField

class AdvectionModule:
    """
    Handles pollutant advection using velocity fields.
    Implements upwind scheme for stability.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the advection module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        self.dx, self.dy, self.dz = grid.get_grid_spacing()
        
    def compute_advection_term(self,
                             concentration: np.ndarray,
                             u: np.ndarray,
                             v: np.ndarray,
                             w: np.ndarray) -> np.ndarray:
        """
        Compute the advection term using upwind scheme.
        
        Args:
            concentration: Pollutant concentration field
            u: x-direction velocity field
            v: y-direction velocity field
            w: z-direction velocity field
            
        Returns:
            Advection term array
        """
        # Initialize advection term
        adv_term = np.zeros_like(concentration)
        
        # Compute x-direction advection
        u_plus = np.maximum(u, 0)
        u_minus = np.minimum(u, 0)
        adv_term[1:-1, :, :] += (
            -u_plus[1:-1, :, :] * (concentration[1:-1, :, :] - concentration[:-2, :, :]) / self.dx
            -u_minus[1:-1, :, :] * (concentration[2:, :, :] - concentration[1:-1, :, :]) / self.dx
        )
        
        # Compute y-direction advection
        v_plus = np.maximum(v, 0)
        v_minus = np.minimum(v, 0)
        adv_term[:, 1:-1, :] += (
            -v_plus[:, 1:-1, :] * (concentration[:, 1:-1, :] - concentration[:, :-2, :]) / self.dy
            -v_minus[:, 1:-1, :] * (concentration[:, 2:, :] - concentration[:, 1:-1, :]) / self.dy
        )
        
        # Compute z-direction advection
        w_plus = np.maximum(w, 0)
        w_minus = np.minimum(w, 0)
        adv_term[:, :, 1:-1] += (
            -w_plus[:, :, 1:-1] * (concentration[:, :, 1:-1] - concentration[:, :, :-2]) / self.dz
            -w_minus[:, :, 1:-1] * (concentration[:, :, 2:] - concentration[:, :, 1:-1]) / self.dz
        )
        
        return adv_term
    
    def apply_boundary_conditions(self, 
                                concentration: np.ndarray,
                                u: np.ndarray,
                                v: np.ndarray,
                                w: np.ndarray) -> None:
        """
        Apply boundary conditions for advection.
        
        Args:
            concentration: Pollutant concentration field
            u, v, w: Velocity fields
        """
        # Apply zero-gradient boundary conditions
        concentration[0, :, :] = concentration[1, :, :]
        concentration[-1, :, :] = concentration[-2, :, :]
        concentration[:, 0, :] = concentration[:, 1, :]
        concentration[:, -1, :] = concentration[:, -2, :]
        concentration[:, :, 0] = concentration[:, :, 1]
        concentration[:, :, -1] = concentration[:, :, -2]
        
    def compute_cfl_numbers(self,
                          u: np.ndarray,
                          v: np.ndarray,
                          w: np.ndarray,
                          dt: float) -> Tuple[float, float, float]:
        """
        Compute CFL numbers for stability check.
        
        Args:
            u, v, w: Velocity fields
            dt: Time step
            
        Returns:
            CFL numbers for x, y, z directions
        """
        cfl_x = np.max(np.abs(u)) * dt / self.dx
        cfl_y = np.max(np.abs(v)) * dt / self.dy
        cfl_z = np.max(np.abs(w)) * dt / self.dz
        
        return cfl_x, cfl_y, cfl_z
    
    def check_stability(self,
                       u: np.ndarray,
                       v: np.ndarray,
                       w: np.ndarray,
                       dt: float) -> bool:
        """
        Check if the simulation is stable based on CFL condition.
        
        Args:
            u, v, w: Velocity fields
            dt: Time step
            
        Returns:
            True if stable, False otherwise
        """
        cfl_x, cfl_y, cfl_z = self.compute_cfl_numbers(u, v, w, dt)
        return max(cfl_x, cfl_y, cfl_z) <= 1.0
