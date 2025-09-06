import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from grid3d import Grid3D
from pollution_field import PollutionField

class SourceSinkModule:
    """
    Handles pollution sources and sinks.
    Supports point sources, area sources, line sources, and various sink terms.
    """
    
    def __init__(self, grid: Grid3D):
        """
        Initialize the source-sink module.
        
        Args:
            grid: Grid3D instance
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        
        # Source parameters
        self.point_sources: List[Dict] = []  # Point sources
        self.area_sources: List[Dict] = []  # Area sources
        self.line_sources: List[Dict] = []  # Line sources
        
        # Sink parameters
        self.sink_terms: Dict[str, Dict] = {}  # Sink terms for each pollutant
        
        # Time-dependent source functions
        self.time_functions: Dict[str, Callable] = {}
        
    def add_point_source(self,
                        pollutant: str,
                        position: Tuple[float, float, float],
                        emission_rate: float,
                        time_function: Optional[Callable] = None) -> None:
        """
        Add a point source.
        
        Args:
            pollutant: Pollutant name
            position: (x, y, z) position of the source
            emission_rate: Emission rate (kg/s)
            time_function: Optional time-dependent function for emission rate
        """
        source = {
            'type': 'point',
            'pollutant': pollutant,
            'position': position,
            'emission_rate': emission_rate,
            'time_function': time_function
        }
        self.point_sources.append(source)
        
    def add_area_source(self,
                       pollutant: str,
                       area: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
                       emission_rate: float,
                       height: float = 0.0,
                       time_function: Optional[Callable] = None) -> None:
        """
        Add an area source.
        
        Args:
            pollutant: Pollutant name
            area: Area boundaries (x_min, x_max, y_min, y_max)
            emission_rate: Emission rate per unit area (kg/m^2/s)
            height: Height of the source (m)
            time_function: Optional time-dependent function for emission rate
        """
        source = {
            'type': 'area',
            'pollutant': pollutant,
            'area': area,
            'emission_rate': emission_rate,
            'height': height,
            'time_function': time_function
        }
        self.area_sources.append(source)
        
    def add_line_source(self,
                       pollutant: str,
                       points: List[Tuple[float, float, float]],
                       emission_rate: float,
                       time_function: Optional[Callable] = None) -> None:
        """
        Add a line source.
        
        Args:
            pollutant: Pollutant name
            points: List of (x, y, z) points defining the line
            emission_rate: Emission rate per unit length (kg/m/s)
            time_function: Optional time-dependent function for emission rate
        """
        source = {
            'type': 'line',
            'pollutant': pollutant,
            'points': points,
            'emission_rate': emission_rate,
            'time_function': time_function
        }
        self.line_sources.append(source)
        
    def add_sink_term(self,
                     pollutant: str,
                     sink_type: str,
                     rate: float,
                     dependencies: Optional[Dict[str, float]] = None) -> None:
        """
        Add a sink term for a pollutant.
        
        Args:
            pollutant: Pollutant name
            sink_type: Type of sink ('deposition', 'degradation', etc.)
            rate: Sink rate constant
            dependencies: Optional dependencies on other variables
        """
        if pollutant not in self.sink_terms:
            self.sink_terms[pollutant] = {}
            
        self.sink_terms[pollutant][sink_type] = {
            'rate': rate,
            'dependencies': dependencies or {}
        }
        
    def compute_point_source_term(self,
                                source: Dict,
                                time: float) -> np.ndarray:
        """
        Compute point source term.
        
        Args:
            source: Source parameters
            time: Current time
            
        Returns:
            Source term field
        """
        # Get grid coordinates
        X, Y, Z = self.grid.get_coordinate_arrays()
        
        # Get source position
        x0, y0, z0 = source['position']
        
        # Compute emission rate
        emission_rate = source['emission_rate']
        if source['time_function'] is not None:
            emission_rate *= source['time_function'](time)
            
        # Compute source term using Gaussian distribution
        dx = self.grid.get_grid_spacing()[0]
        dy = self.grid.get_grid_spacing()[1]
        dz = self.grid.get_grid_spacing()[2]
        
        sigma_x = dx / 2
        sigma_y = dy / 2
        sigma_z = dz / 2
        
        term = emission_rate * np.exp(-((X-x0)**2/(2*sigma_x**2) +
                                       (Y-y0)**2/(2*sigma_y**2) +
                                       (Z-z0)**2/(2*sigma_z**2)))
        
        return term
        
    def compute_area_source_term(self,
                               source: Dict,
                               time: float) -> np.ndarray:
        """
        Compute area source term.
        
        Args:
            source: Source parameters
            time: Current time
            
        Returns:
            Source term field
        """
        # Get grid coordinates
        X, Y, Z = self.grid.get_coordinate_arrays()
        
        # Get area boundaries
        x_min, x_max, y_min, y_max = source['area']
        height = source['height']
        
        # Compute emission rate
        emission_rate = source['emission_rate']
        if source['time_function'] is not None:
            emission_rate *= source['time_function'](time)
            
        # Initialize source term
        term = np.zeros_like(X)
        
        # Set source term in area
        mask = ((X >= x_min) & (X <= x_max) &
                (Y >= y_min) & (Y <= y_max) &
                (Z >= height - 0.5) & (Z <= height + 0.5))
        term[mask] = emission_rate
        
        return term
        
    def compute_line_source_term(self,
                               source: Dict,
                               time: float) -> np.ndarray:
        """
        Compute line source term.
        
        Args:
            source: Source parameters
            time: Current time
            
        Returns:
            Source term field
        """
        # Get grid coordinates
        x, y, z = self.grid.get_grid_coordinates()
        
        # Get line points
        points = source['points']
        
        # Compute emission rate
        emission_rate = source['emission_rate']
        if source['time_function'] is not None:
            emission_rate *= source['time_function'](time)
            
        # Initialize source term
        term = np.zeros_like(x)
        
        # Compute distance to line segments
        for i in range(len(points)-1):
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            
            # Vector from p1 to p2
            v = p2 - p1
            
            # Vector from p1 to grid points
            w = np.stack([x-p1[0], y-p1[1], z-p1[2]], axis=-1)
            
            # Projection of w onto v
            c1 = np.sum(w * v, axis=-1) / np.sum(v**2)
            c1 = np.clip(c1, 0, 1)
            
            # Distance to line segment
            distance = np.sqrt(np.sum((w - c1[..., np.newaxis] * v)**2, axis=-1))
            
            # Add source term
            sigma = self.grid.get_grid_spacing()[0] / 2
            term += emission_rate * np.exp(-distance**2/(2*sigma**2))
            
        return term
        
    def compute_sink_term(self,
                        pollutant: str,
                        concentration: np.ndarray,
                        dependencies: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute sink term for a pollutant.
        
        Args:
            pollutant: Pollutant name
            concentration: Concentration field
            dependencies: Dictionary of dependent variables
            
        Returns:
            Sink term field
        """
        if pollutant not in self.sink_terms:
            return np.zeros_like(concentration)
            
        sink_term = np.zeros_like(concentration)
        
        for sink_type, params in self.sink_terms[pollutant].items():
            rate = params['rate']
            
            # Apply dependencies
            for dep_name, dep_coeff in params['dependencies'].items():
                if dep_name in dependencies:
                    rate *= dependencies[dep_name]**dep_coeff
                    
            # Add sink term
            if sink_type == 'deposition':
                sink_term += rate * concentration
            elif sink_type == 'degradation':
                sink_term += rate * concentration
            elif sink_type == 'reaction':
                sink_term += rate * concentration
                
        return sink_term
        
    def compute_source_sink_terms(self,
                                concentrations: Dict[str, np.ndarray],
                                time: float,
                                dependencies: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute source and sink terms for all pollutants.
        
        Args:
            concentrations: Dictionary of concentration fields
            time: Current time
            dependencies: Dictionary of dependent variables
            
        Returns:
            Dictionary of source-sink terms for each pollutant
        """
        terms = {name: np.zeros_like(conc) for name, conc in concentrations.items()}
        
        # Add source terms
        for source in self.point_sources:
            if source['pollutant'] in terms:
                terms[source['pollutant']] += self.compute_point_source_term(source, time)
                
        for source in self.area_sources:
            if source['pollutant'] in terms:
                terms[source['pollutant']] += self.compute_area_source_term(source, time)
                
        for source in self.line_sources:
            if source['pollutant'] in terms:
                terms[source['pollutant']] += self.compute_line_source_term(source, time)
                
        # Add sink terms
        for pollutant, concentration in concentrations.items():
            terms[pollutant] -= self.compute_sink_term(pollutant, concentration, dependencies)
            
        return terms
        
    def apply_boundary_conditions(self,
                                terms: Dict[str, np.ndarray]) -> None:
        """
        Apply boundary conditions for all source-sink terms.
        
        Args:
            terms: Dictionary of source-sink terms
        """
        for term in terms.values():
            # Apply zero-gradient boundary conditions
            term[0, :, :] = term[1, :, :]
            term[-1, :, :] = term[-2, :, :]
            term[:, 0, :] = term[:, 1, :]
            term[:, -1, :] = term[:, -2, :]
            term[:, :, 0] = term[:, :, 1]
            term[:, :, -1] = term[:, :, -2]
