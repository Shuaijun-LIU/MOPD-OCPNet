import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
from grid3d import Grid3D
from pollution_field import PollutionField

class OutputModule:
    """
    Handles model output, including data saving, visualization, and statistics.
    """
    
    def __init__(self, grid: Grid3D, output_dir: Union[str, Path]):
        """
        Initialize the output module.
        
        Args:
            grid: Grid3D instance
            output_dir: Output directory path
        """
        self.grid = grid
        self.nx, self.ny, self.nz = grid.get_grid_shape()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Output parameters
        self.output_fields: List[str] = []  # Fields to output
        self.output_times: List[float] = []  # Output times
        self.output_format: str = 'netcdf'  # Output format
        self.output_interval: float = 3600.0  # Output interval (s)
        self.time_step: float = 60.0  # Time step (s)
        
        # Visualization parameters
        self.visualization_fields: List[str] = []  # Fields to visualize
        self.visualization_times: List[float] = []  # Visualization times
        self.visualization_interval: float = 3600.0  # Visualization interval (s)
        
        # Statistics parameters
        self.statistics_fields: List[str] = []  # Fields for statistics
        self.statistics_times: List[float] = []  # Statistics times
        self.statistics_interval: float = 3600.0  # Statistics interval (s)
        
    def set_output_fields(self, fields: List[str]) -> None:
        """
        Set fields to output.
        
        Args:
            fields: List of field names
        """
        self.output_fields = fields
        
    def set_output_interval(self, interval: float) -> None:
        """
        Set output interval.
        
        Args:
            interval: Output interval in seconds
        """
        self.output_interval = interval
        
    def set_visualization_fields(self, fields: List[str]) -> None:
        """
        Set fields to visualize.
        
        Args:
            fields: List of field names
        """
        self.visualization_fields = fields
        
    def set_visualization_interval(self, interval: float) -> None:
        """
        Set visualization interval.
        
        Args:
            interval: Visualization interval in seconds
        """
        self.visualization_interval = interval
        
    def set_statistics_fields(self, fields: List[str]) -> None:
        """
        Set fields for statistics.
        
        Args:
            fields: List of field names
        """
        self.statistics_fields = fields
        
    def set_statistics_interval(self, interval: float) -> None:
        """
        Set statistics interval.
        
        Args:
            interval: Statistics interval in seconds
        """
        self.statistics_interval = interval
        
    def set_time_step(self, time_step: float) -> None:
        """
        Set time step.
        
        Args:
            time_step: Time step in seconds
        """
        self.time_step = time_step
        
    def save_data(self,
                 fields: Dict[str, np.ndarray],
                 time: float,
                 filename: Optional[str] = None) -> None:
        """
        Save field data to file.
        
        Args:
            fields: Dictionary of field data
            time: Current time
            filename: Optional output filename
        """
        if filename is None:
            filename = f"output_{time:.0f}.nc"
            
        output_file = self.output_dir / filename
        
        # Create NetCDF file
        with nc.Dataset(output_file, 'w') as ds:
            # Create dimensions
            ds.createDimension('x', self.nx)
            ds.createDimension('y', self.ny)
            ds.createDimension('z', self.nz)
            ds.createDimension('time', None)
            
            # Create coordinate variables
            x = ds.createVariable('x', 'f4', ('x',))
            y = ds.createVariable('y', 'f4', ('y',))
            z = ds.createVariable('z', 'f4', ('z',))
            t = ds.createVariable('time', 'f8', ('time',))
            
            # Set coordinate values
            x[:] = self.grid.x
            y[:] = self.grid.y
            z[:] = self.grid.z
            t[0] = time
            
            # Create field variables
            for name, data in fields.items():
                if name in self.output_fields:
                    var = ds.createVariable(name, 'f4', ('time', 'z', 'y', 'x'))
                    var[0, :, :, :] = data
                    
            # Add metadata
            ds.description = "PollutionModel3D output"
            ds.history = f"Created {datetime.now().isoformat()}"
            
    def create_visualization(self,
                           fields: Dict[str, np.ndarray],
                           time: float,
                           filename: Optional[str] = None) -> None:
        """
        Create visualization of field data.
        
        Args:
            fields: Dictionary of field data
            time: Current time
            filename: Optional output filename
        """
        if filename is None:
            filename = f"visualization_{time:.0f}.png"
            
        output_file = self.output_dir / filename
        
        # Create figure
        n_fields = len(self.visualization_fields)
        fig, axes = plt.subplots(n_fields, 3, figsize=(15, 5*n_fields))
        
        if n_fields == 1:
            axes = axes[np.newaxis, :]
            
        # Plot each field
        for i, name in enumerate(self.visualization_fields):
            if name in fields:
                data = fields[name]
                
                # XY plane at mid-depth
                im = axes[i, 0].imshow(data[:, :, self.nz//2], origin='lower')
                axes[i, 0].set_title(f"{name} (XY plane)")
                plt.colorbar(im, ax=axes[i, 0])
                
                # XZ plane at mid-width
                im = axes[i, 1].imshow(data[:, self.ny//2, :], origin='lower')
                axes[i, 1].set_title(f"{name} (XZ plane)")
                plt.colorbar(im, ax=axes[i, 1])
                
                # YZ plane at mid-length
                im = axes[i, 2].imshow(data[self.nx//2, :, :], origin='lower')
                axes[i, 2].set_title(f"{name} (YZ plane)")
                plt.colorbar(im, ax=axes[i, 2])
                
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
    def compute_statistics(self,
                         fields: Dict[str, np.ndarray],
                         time: float) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for fields.
        
        Args:
            fields: Dictionary of field data
            time: Current time
            
        Returns:
            Dictionary of statistics for each field
        """
        statistics = {}
        
        for name in self.statistics_fields:
            if name in fields:
                data = fields[name]
                statistics[name] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'time': time
                }
                
        return statistics
        
    def save_statistics(self,
                       statistics: Dict[str, Dict[str, float]],
                       filename: Optional[str] = None) -> None:
        """
        Save statistics to file.
        
        Args:
            statistics: Dictionary of statistics
            filename: Optional output filename
        """
        if filename is None:
            filename = "statistics.txt"
            
        output_file = self.output_dir / filename
        
        with open(output_file, 'a') as f:
            for name, stats in statistics.items():
                f.write(f"Time: {stats['time']:.1f} s\n")
                f.write(f"Field: {name}\n")
                f.write(f"Mean: {stats['mean']:.6f}\n")
                f.write(f"Std: {stats['std']:.6f}\n")
                f.write(f"Min: {stats['min']:.6f}\n")
                f.write(f"Max: {stats['max']:.6f}\n")
                f.write("\n")
                
    def should_output(self, time: float) -> bool:
        """
        Check if output should be performed at current time.
        
        Args:
            time: Current time
            
        Returns:
            True if output should be performed
        """
        return time in self.output_times or \
               (self.output_interval > 0 and
               time % self.output_interval < self.time_step)
                
    def should_visualize(self, time: float) -> bool:
        """
        Check if visualization should be performed at current time.
        
        Args:
            time: Current time
            
        Returns:
            True if visualization should be performed
        """
        return time in self.visualization_times or \
               (self.visualization_interval > 0 and
                time % self.visualization_interval < self.time_step)
                
    def should_compute_statistics(self, time: float) -> bool:
        """
        Check if statistics should be computed at current time.
        
        Args:
            time: Current time
            
        Returns:
            True if statistics should be computed
        """
        return time in self.statistics_times or \
               (self.statistics_interval > 0 and
                time % self.statistics_interval < self.time_step)
                
    def process_output(self,
                      fields: Dict[str, np.ndarray],
                      time: float) -> None:
        """
        Process all output tasks.
        
        Args:
            fields: Dictionary of field data
            time: Current time
        """
        if self.should_output(time):
            self.save_data(fields, time)
            
        if self.should_visualize(time):
            self.create_visualization(fields, time)
            
        if self.should_compute_statistics(time):
            statistics = self.compute_statistics(fields, time)
            self.save_statistics(statistics) 