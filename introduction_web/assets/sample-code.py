#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code for working with the Combined Ocean Environment Dataset
Author: Dataset Team
Version: 1.0.0
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

def load_dataset(file_path):
    """
    Load the Combined Ocean Environment Dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the NetCDF dataset file
        
    Returns:
    --------
    xarray.Dataset
        The loaded dataset
    """
    print(f"Loading dataset from {file_path}...")
    ds = xr.open_dataset(file_path)
    print("Dataset loaded successfully.")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    return ds

def plot_bathymetry(ds, figsize=(12, 8)):
    """
    Plot bathymetry from the dataset
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing elevation data
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Create figure with projection
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines and features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    # Create custom colormap for bathymetry
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
    colors = np.vstack((colors_undersea, colors_land))
    cut = 56
    terrain_map = LinearSegmentedColormap.from_list('terrain_map', colors)
    
    # Plot bathymetry
    elevation = ds.elevation.isel(time=0) if 'time' in ds.elevation.dims else ds.elevation
    im = elevation.plot(ax=ax, cmap=terrain_map, transform=ccrs.PlateCarree(),
                       cbar_kwargs={'shrink': 0.8, 'label': 'Elevation (m)'})
    
    # Set title and labels
    plt.title('Bathymetry from Combined Ocean Environment Dataset')
    
    return fig, ax

def plot_ocean_currents(ds, depth_idx=0, time_idx=0, figsize=(12, 8)):
    """
    Plot ocean currents from the dataset
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing current data
    depth_idx : int
        Index of depth level to plot
    time_idx : int
        Index of time step to plot
    figsize : tuple
        Figure size (width, height) in inches
    """
    # Create figure with projection
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines and features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    # Extract current components
    u = ds.uo.isel(depth=depth_idx, time=time_idx)
    v = ds.vo.isel(depth=depth_idx, time=time_idx)
    
    # Calculate current speed
    speed = np.sqrt(u**2 + v**2)
    
    # Plot current speed as background
    speed.plot(ax=ax, cmap='viridis', transform=ccrs.PlateCarree(),
              cbar_kwargs={'shrink': 0.8, 'label': 'Current Speed (m/s)'})
    
    # Plot current vectors
    # Subsample for clarity
    step = 5
    q = ax.quiver(ds.longitude[::step], ds.latitude[::step],
                 u.values[::step, ::step], v.values[::step, ::step],
                 transform=ccrs.PlateCarree(), scale=20, width=0.002)
    
    # Add quiver key
    plt.quiverkey(q, 0.9, 0.9, 0.5, '0.5 m/s', labelpos='E', coordinates='figure')
    
    # Set title
    depth_value = ds.depth.values[depth_idx]
    time_value = ds.time.values[time_idx]
    plt.title(f'Ocean Currents at {depth_value}m, {np.datetime_as_string(time_value, unit="D")}')
    
    return fig, ax

def plot_temperature_salinity_profile(ds, lon_idx, lat_idx, time_idx=0):
    """
    Plot temperature and salinity profiles at a specific location
    
    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing temperature and salinity data
    lon_idx : int
        Index of longitude point
    lat_idx : int
        Index of latitude point
    time_idx : int
        Index of time step to plot
    """
    # Extract profiles
    temp = ds.thetao.isel(time=time_idx, longitude=lon_idx, latitude=lat_idx)
    salt = ds.so.isel(time=time_idx, longitude=lon_idx, latitude=lat_idx)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(8, 10))
    
    # Plot temperature
    ax1.plot(temp, ds.depth, 'r-', label='Temperature')
    ax1.set_xlabel('Temperature (°C)', color='r')
    ax1.set_ylabel('Depth (m)')
    ax1.tick_params(axis='x', labelcolor='r')
    ax1.invert_yaxis()  # Depth increases downward
    
    # Create second y-axis for salinity
    ax2 = ax1.twiny()
    ax2.plot(salt, ds.depth, 'b-', label='Salinity')
    ax2.set_xlabel('Salinity (PSU)', color='b')
    ax2.tick_params(axis='x', labelcolor='b')
    
    # Add location information to title
    lon = ds.longitude.values[lon_idx]
    lat = ds.latitude.values[lat_idx]
    time = ds.time.values[time_idx]
    plt.title(f'Temperature and Salinity Profiles at ({lon:.2f}°E, {lat:.2f}°N)\n{np.datetime_as_string(time, unit="D")}')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def main():
    """Main function demonstrating dataset usage"""
    # Replace with actual path to your dataset
    dataset_path = "path/to/combined_environment.nc"
    
    try:
        # Load dataset
        ds = load_dataset(dataset_path)
        
        # Plot bathymetry
        fig_bath, ax_bath = plot_bathymetry(ds)
        plt.savefig('bathymetry_plot.png', dpi=300, bbox_inches='tight')
        
        # Plot ocean currents
        fig_curr, ax_curr = plot_ocean_currents(ds)
        plt.savefig('currents_plot.png', dpi=300, bbox_inches='tight')
        
        # Plot temperature-salinity profile
        # Use middle indices for demonstration
        lon_idx = len(ds.longitude) // 2
        lat_idx = len(ds.latitude) // 2
        fig_prof, axes_prof = plot_temperature_salinity_profile(ds, lon_idx, lat_idx)
        plt.savefig('ts_profile_plot.png', dpi=300, bbox_inches='tight')
        
        print("All plots generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Close all figures
        plt.close('all')

if __name__ == "__main__":
    main()
