import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from grid3d import Grid3D
from pollution_field import PollutionField

class BoundaryConditionsModule:
    """
    Handles boundary conditions for the model.
    Supports Dirichlet, Neumann, periodic, and open boundary conditions.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the boundary conditions module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        
        # Boundary conditions for each field
        self.boundary_conditions: Dict[str, Dict[str, Dict]] = {}
        
        # Time-dependent boundary functions
        self.time_functions: Dict[str, Dict[str, Callable]] = {}
        
    def set_dirichlet_boundary(self,
                             field: str,
                             boundary: str,
                             value: Union[float, np.ndarray, Callable],
                             time_function: Optional[Callable] = None) -> None:
        """
        Set Dirichlet boundary condition (fixed value).
        
        Args:
            field: Field name
            boundary: Boundary name ('left', 'right', 'front', 'back', 'bottom', 'top')
            value: Boundary value (constant, array, or function)
            time_function: Optional time-dependent function
        """
        if field not in self.boundary_conditions:
            self.boundary_conditions[field] = {}
            
        self.boundary_conditions[field][boundary] = {
            'type': 'dirichlet',
            'value': value,
            'time_function': time_function
        }
        
    def set_neumann_boundary(self,
                           field: str,
                           boundary: str,
                           gradient: Union[float, np.ndarray, Callable],
                           time_function: Optional[Callable] = None) -> None:
        """
        Set Neumann boundary condition (fixed gradient).
        
        Args:
            field: Field name
            boundary: Boundary name ('left', 'right', 'front', 'back', 'bottom', 'top')
            gradient: Boundary gradient (constant, array, or function)
            time_function: Optional time-dependent function
        """
        if field not in self.boundary_conditions:
            self.boundary_conditions[field] = {}
            
        self.boundary_conditions[field][boundary] = {
            'type': 'neumann',
            'gradient': gradient,
            'time_function': time_function
        }
        
    def set_periodic_boundary(self,
                            field: str,
                            axis: str) -> None:
        """
        Set periodic boundary condition.
        
        Args:
            field: Field name
            axis: Axis name ('x', 'y', 'z')
        """
        if field not in self.boundary_conditions:
            self.boundary_conditions[field] = {}
            
        if axis == 'x':
            self.boundary_conditions[field]['left'] = {'type': 'periodic', 'axis': 'x'}
            self.boundary_conditions[field]['right'] = {'type': 'periodic', 'axis': 'x'}
        elif axis == 'y':
            self.boundary_conditions[field]['front'] = {'type': 'periodic', 'axis': 'y'}
            self.boundary_conditions[field]['back'] = {'type': 'periodic', 'axis': 'y'}
        elif axis == 'z':
            self.boundary_conditions[field]['bottom'] = {'type': 'periodic', 'axis': 'z'}
            self.boundary_conditions[field]['top'] = {'type': 'periodic', 'axis': 'z'}
            
    def set_open_boundary(self,
                        field: str,
                        boundary: str,
                        advection_velocity: Union[float, np.ndarray, Callable],
                        time_function: Optional[Callable] = None) -> None:
        """
        Set open boundary condition (radiation condition).
        
        Args:
            field: Field name
            boundary: Boundary name ('left', 'right', 'front', 'back', 'bottom', 'top')
            advection_velocity: Advection velocity at boundary
            time_function: Optional time-dependent function
        """
        if field not in self.boundary_conditions:
            self.boundary_conditions[field] = {}
            
        self.boundary_conditions[field][boundary] = {
            'type': 'open',
            'advection_velocity': advection_velocity,
            'time_function': time_function
        }
        
    def apply_boundary_conditions(self,
                                field: str,
                                data: np.ndarray,
                                time: float = 0.0) -> None:
        """
        Apply boundary conditions to a field.
        
        Args:
            field: Field name
            data: Field data
            time: Current time
        """
        if field not in self.boundary_conditions:
            return
            
        for boundary, condition in self.boundary_conditions[field].items():
            if condition['type'] == 'dirichlet':
                self._apply_dirichlet(data, boundary, condition, time)
            elif condition['type'] == 'neumann':
                self._apply_neumann(data, boundary, condition, time)
            elif condition['type'] == 'periodic':
                self._apply_periodic(data, boundary, condition)
            elif condition['type'] == 'open':
                self._apply_open(data, boundary, condition, time)
                
    def _apply_dirichlet(self,
                        data: np.ndarray,
                        boundary: str,
                        condition: Dict,
                        time: float) -> None:
        """
        Apply Dirichlet boundary condition.
        
        Args:
            data: Field data
            boundary: Boundary name
            condition: Boundary condition parameters
            time: Current time
        """
        value = condition['value']
        if callable(value):
            value = value(time)
        elif condition['time_function'] is not None:
            value *= condition['time_function'](time)
            
        if boundary == 'left':
            data[0, :, :] = value
        elif boundary == 'right':
            data[-1, :, :] = value
        elif boundary == 'front':
            data[:, 0, :] = value
        elif boundary == 'back':
            data[:, -1, :] = value
        elif boundary == 'bottom':
            data[:, :, 0] = value
        elif boundary == 'top':
            data[:, :, -1] = value
            
    def _apply_neumann(self,
                      data: np.ndarray,
                      boundary: str,
                      condition: Dict,
                      time: float) -> None:
        """
        Apply Neumann boundary condition.
        
        Args:
            data: Field data
            boundary: Boundary name
            condition: Boundary condition parameters
            time: Current time
        """
        gradient = condition['gradient']
        if callable(gradient):
            gradient = gradient(time)
        elif condition['time_function'] is not None:
            gradient *= condition['time_function'](time)
            
        dx, dy, dz = self.grid.get_grid_spacing()
        
        if boundary == 'left':
            data[0, :, :] = data[1, :, :] - gradient * dx
        elif boundary == 'right':
            data[-1, :, :] = data[-2, :, :] + gradient * dx
        elif boundary == 'front':
            data[:, 0, :] = data[:, 1, :] - gradient * dy
        elif boundary == 'back':
            data[:, -1, :] = data[:, -2, :] + gradient * dy
        elif boundary == 'bottom':
            data[:, :, 0] = data[:, :, 1] - gradient * dz
        elif boundary == 'top':
            data[:, :, -1] = data[:, :, -2] + gradient * dz
            
    def _apply_periodic(self,
                       data: np.ndarray,
                       boundary: str,
                       condition: Dict) -> None:
        """
        Apply periodic boundary condition.
        
        Args:
            data: Field data
            boundary: Boundary name
            condition: Boundary condition parameters
        """
        axis = condition['axis']
        
        if axis == 'x':
            if boundary == 'left':
                data[0, :, :] = data[-2, :, :]
            elif boundary == 'right':
                data[-1, :, :] = data[1, :, :]
        elif axis == 'y':
            if boundary == 'front':
                data[:, 0, :] = data[:, -2, :]
            elif boundary == 'back':
                data[:, -1, :] = data[:, 1, :]
        elif axis == 'z':
            if boundary == 'bottom':
                data[:, :, 0] = data[:, :, -2]
            elif boundary == 'top':
                data[:, :, -1] = data[:, :, 1]
                
    def _apply_open(self,
                   data: np.ndarray,
                   boundary: str,
                   condition: Dict,
                   time: float) -> None:
        """
        Apply open boundary condition (radiation condition).
        
        Args:
            data: Field data
            boundary: Boundary name
            condition: Boundary condition parameters
            time: Current time
        """
        velocity = condition['advection_velocity']
        if callable(velocity):
            velocity = velocity(time)
        elif condition['time_function'] is not None:
            velocity *= condition['time_function'](time)
            
        dx, dy, dz = self.grid.get_grid_spacing()
        
        if boundary == 'left':
            if velocity > 0:
                data[0, :, :] = data[1, :, :]
        elif boundary == 'right':
            if velocity < 0:
                data[-1, :, :] = data[-2, :, :]
        elif boundary == 'front':
            if velocity > 0:
                data[:, 0, :] = data[:, 1, :]
        elif boundary == 'back':
            if velocity < 0:
                data[:, -1, :] = data[:, -2, :]
        elif boundary == 'bottom':
            if velocity > 0:
                data[:, :, 0] = data[:, :, 1]
        elif boundary == 'top':
            if velocity < 0:
                data[:, :, -1] = data[:, :, -2] 