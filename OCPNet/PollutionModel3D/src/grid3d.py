import numpy as np
from typing import Tuple, Dict, Optional

class Grid3D:
    """
    A 3D Eulerian grid for ocean pollution modeling.
    Handles grid geometry, spatial coordinates, and volume calculations.
    """
    
    def __init__(self, 
                 nx: int, ny: int, nz: int,
                 dx: float, dy: float, dz: float,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Initialize a 3D grid.
        
        Args:
            nx, ny, nz: Number of grid points in x, y, z directions
            dx, dy, dz: Grid spacing in x, y, z directions (meters)
            origin: Origin point coordinates (x0, y0, z0)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.origin = origin
        
        # Initialize coordinate arrays
        self.x = np.linspace(origin[0], origin[0] + (nx-1)*dx, nx)
        self.y = np.linspace(origin[1], origin[1] + (ny-1)*dy, ny)
        self.z = np.linspace(origin[2], origin[2] + (nz-1)*dz, nz)
        
        # Create meshgrid for 3D coordinates
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Calculate cell volumes
        self.volumes = np.ones((nx, ny, nz)) * dx * dy * dz
        
        # Initialize boundary masks
        self.boundary_mask = np.zeros((nx, ny, nz), dtype=bool)
        self.boundary_mask[0, :, :] = True  # x=0 boundary
        self.boundary_mask[-1, :, :] = True  # x=nx-1 boundary
        self.boundary_mask[:, 0, :] = True  # y=0 boundary
        self.boundary_mask[:, -1, :] = True  # y=ny-1 boundary
        self.boundary_mask[:, :, 0] = True  # z=0 boundary
        self.boundary_mask[:, :, -1] = True  # z=nz-1 boundary
        
    def get_cell_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Get the coordinates of cell center at index (i,j,k)."""
        return (self.x[i], self.y[j], self.z[k])
    
    def get_cell_volume(self, i: int, j: int, k: int) -> float:
        """Get the volume of cell at index (i,j,k)."""
        return self.volumes[i, j, k]
    
    def is_boundary_cell(self, i: int, j: int, k: int) -> bool:
        """Check if cell at index (i,j,k) is a boundary cell."""
        return self.boundary_mask[i, j, k]
    
    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Return the shape of the grid (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)
    
    def get_grid_spacing(self) -> Tuple[float, float, float]:
        """Return the grid spacing (dx, dy, dz)."""
        return (self.dx, self.dy, self.dz)
    
    def get_domain_size(self) -> Tuple[float, float, float]:
        """Return the total domain size in each direction."""
        return ((self.nx-1)*self.dx, (self.ny-1)*self.dy, (self.nz-1)*self.dz)
    
    def get_coordinate_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the coordinate arrays (X, Y, Z)."""
        return (self.X, self.Y, self.Z)
