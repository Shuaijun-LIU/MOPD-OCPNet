import numpy as np
from typing import Dict, List, Optional, Tuple
from grid3d import Grid3D

class PollutionField:
    """
    Manages multiple pollutant concentration fields in a 3D grid.
    Supports dynamic addition and removal of pollutants.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize a pollution field manager.
        
        Args:
            grid: Grid3D instance defining the spatial domain
        """
        self.grid = grid
        self.concentrations: Dict[str, np.ndarray] = {}
        self.units: Dict[str, str] = {}
        self.background_values: Dict[str, float] = {}
        
    def add_pollutant(self, 
                     name: str, 
                     initial_value: float = 0.0,
                     unit: str = "kg/m^3",
                     background_value: float = 0.0) -> None:
        """
        Add a new pollutant to the system.
        
        Args:
            name: Unique identifier for the pollutant
            initial_value: Initial concentration value
            unit: Unit of measurement
            background_value: Background concentration level
        """
        if name in self.concentrations:
            raise ValueError(f"Pollutant {name} already exists")
            
        nx, ny, nz = self.grid.get_grid_shape()
        self.concentrations[name] = np.full((nx, ny, nz), initial_value)
        self.units[name] = unit
        self.background_values[name] = background_value
        
    def remove_pollutant(self, name: str) -> None:
        """Remove a pollutant from the system."""
        if name not in self.concentrations:
            raise ValueError(f"Pollutant {name} does not exist")
            
        del self.concentrations[name]
        del self.units[name]
        del self.background_values[name]
        
    def get_concentration(self, name: str) -> np.ndarray:
        """Get the concentration field for a specific pollutant."""
        if name not in self.concentrations:
            raise ValueError(f"Pollutant {name} does not exist")
        return self.concentrations[name]
    
    def set_concentration(self, name: str, value: np.ndarray) -> None:
        """Set the concentration field for a specific pollutant."""
        if name not in self.concentrations:
            raise ValueError(f"Pollutant {name} does not exist")
        if value.shape != self.grid.get_grid_shape():
            raise ValueError("Concentration array shape mismatch with grid")
        self.concentrations[name] = value
        
    def get_pollutant_names(self) -> List[str]:
        """Get list of all pollutant names."""
        return list(self.concentrations.keys())
    
    def get_unit(self, name: str) -> str:
        """Get the unit of measurement for a specific pollutant."""
        if name not in self.units:
            raise ValueError(f"Pollutant {name} does not exist")
        return self.units[name]
    
    def get_background_value(self, name: str) -> float:
        """Get the background concentration for a specific pollutant."""
        if name not in self.background_values:
            raise ValueError(f"Pollutant {name} does not exist")
        return self.background_values[name]
    
    def apply_boundary_conditions(self, name: str, boundary_value: float) -> None:
        """
        Apply constant boundary conditions to a pollutant field.
        
        Args:
            name: Pollutant name
            boundary_value: Value to set at boundaries
        """
        if name not in self.concentrations:
            raise ValueError(f"Pollutant {name} does not exist")
            
        concentration = self.concentrations[name]
        boundary_mask = self.grid.boundary_mask
        concentration[boundary_mask] = boundary_value
        
    def calculate_total_mass(self, name: str) -> float:
        """
        Calculate total mass of a pollutant in the domain.
        
        Args:
            name: Pollutant name
            
        Returns:
            Total mass in kg
        """
        if name not in self.concentrations:
            raise ValueError(f"Pollutant {name} does not exist")
            
        concentration = self.concentrations[name]
        volumes = self.grid.volumes
        return np.sum(concentration * volumes)
    
    def get_concentration_at_point(self, 
                                 name: str, 
                                 x: float, y: float, z: float) -> float:
        """
        Get concentration at a specific point using trilinear interpolation.
        
        Args:
            name: Pollutant name
            x, y, z: Coordinates
            
        Returns:
            Interpolated concentration value
        """
        if name not in self.concentrations:
            raise ValueError(f"Pollutant {name} does not exist")
            
        # Get grid coordinates
        x_coords = self.grid.x
        y_coords = self.grid.y
        z_coords = self.grid.z
        
        # Find surrounding grid points
        i = np.searchsorted(x_coords, x) - 1
        j = np.searchsorted(y_coords, y) - 1
        k = np.searchsorted(z_coords, z) - 1
        
        # Ensure indices are within bounds
        i = max(0, min(i, len(x_coords)-2))
        j = max(0, min(j, len(y_coords)-2))
        k = max(0, min(k, len(z_coords)-2))
        
        # Get concentration values at surrounding points
        c000 = self.concentrations[name][i, j, k]
        c001 = self.concentrations[name][i, j, k+1]
        c010 = self.concentrations[name][i, j+1, k]
        c011 = self.concentrations[name][i, j+1, k+1]
        c100 = self.concentrations[name][i+1, j, k]
        c101 = self.concentrations[name][i+1, j, k+1]
        c110 = self.concentrations[name][i+1, j+1, k]
        c111 = self.concentrations[name][i+1, j+1, k+1]
        
        # Calculate interpolation weights
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[j+1] - y_coords[j]
        dz = z_coords[k+1] - z_coords[k]
        
        wx = (x - x_coords[i]) / dx
        wy = (y - y_coords[j]) / dy
        wz = (z - z_coords[k]) / dz
        
        # Perform trilinear interpolation
        c00 = c000 * (1-wx) + c100 * wx
        c01 = c001 * (1-wx) + c101 * wx
        c10 = c010 * (1-wx) + c110 * wx
        c11 = c011 * (1-wx) + c111 * wx
        
        c0 = c00 * (1-wy) + c10 * wy
        c1 = c01 * (1-wy) + c11 * wy
        
        return c0 * (1-wz) + c1 * wz
