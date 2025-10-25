#!/usr/bin/env python3
"""
Simplified Pollutant Diffusion Visualization
Fixed version with realistic diffusion behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def plot_pollutant_diffusion(lon_grid, lat_grid, pollutant_data, days, pollutant_name, pollutant_data_all_days=None):
    """
    Plot pollutant diffusion for multiple days with improved visualization
    """
    # Create output directory
    os.makedirs('./output', exist_ok=True)
    
    n_days = len(days)
    n_cols = 4
    n_rows = (n_days + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, (day, data) in enumerate(zip(days, pollutant_data)):
        ax = axes_flat[i]
        
        # Create contour plot
        contour = ax.contourf(lon_grid, lat_grid, data, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(lon_grid, lat_grid, data, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Concentration', fontsize=10)
        
        # Formatting
        ax.set_title(f'Day {day}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        max_conc = np.max(data)
        mean_conc = np.mean(data)
        ax.text(0.02, 0.98, f'Max: {max_conc:.3f}\nMean: {mean_conc:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_days, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle(pollutant_name, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./output/pollutant_diffusion_static.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create animation if all data is provided
    if pollutant_data_all_days is not None:
        print("Creating animation...")
        fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax_anim.clear()
            data = pollutant_data_all_days[frame]
            contour = ax_anim.contourf(lon_grid, lat_grid, data, levels=20, cmap='viridis', alpha=0.8)
            ax_anim.contour(lon_grid, lat_grid, data, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            ax_anim.set_title(f'Day {frame + 1}', fontsize=14, fontweight='bold')
            ax_anim.set_xlabel('Longitude', fontsize=12)
            ax_anim.set_ylabel('Latitude', fontsize=12)
            ax_anim.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax_anim)
            cbar.set_label('Concentration', fontsize=12)
            
            return contour,
        
        anim = FuncAnimation(fig_anim, animate, frames=len(pollutant_data_all_days), 
                           interval=200, blit=False, repeat=True)
        
        # Save animation
        anim.save('./output/pollutant_diffusion.gif', writer='pillow', fps=5)
        print("GIF saved as 'pollutant_diffusion.gif'")

def main():
    """
    Main function with improved diffusion model
    """
    # Set up grid
    lon_min, lon_max = 130.0, 140.0
    lat_min, lat_max = 30.0, 40.0
    resolution = 0.2  # Reduced resolution to avoid memory issues
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Modified: Output every few days, customizable interval
    total_days = 120
    interval = 10  # Every 10 days, can be modified to other values like 5, 15, 20, etc.
    days = list(range(interval, total_days + 1, interval))  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    
    # If interval is too large, ensure at least 6 time points are shown
    if len(days) < 6:
        days = [10, 25, 40, 55, 70, 85, 100, 115]  # Fixed display of 8 time points
    
    print(f"Will output results for the following days: {days}")
    
    pollutant_data = []
    pollutant_data_all_days = []
    
    # Add debugging information
    print("\n=== Diffusion Parameter Analysis ===")
    print("Day\tCenter_Lon\tCenter_Lat\tScale\t\tMax_Concentration")
    print("-" * 60)

    # Pre-calculate global maximum concentration to avoid per-frame normalization issues
    print("Pre-calculating global maximum concentration...")
    max_concentration = 0
    all_fields = []
    
    for day in range(1, total_days + 1):
        # More realistic diffusion: continuous growth with slight random variation
        center_lon = 135.0 + 0.1 * np.sin(day / 20.0)  # Very slow oscillation
        center_lat = 35.0 + 0.1 * np.cos(day / 20.0)   # Very slow oscillation
        scale = 1.0 + 0.8 * (day / total_days) + 0.2 * np.sin(day / 30.0)  # Growing with slight variation
        
        # Ensure center position is within grid bounds
        center_lon = np.clip(center_lon, lon_min + 2, lon_max - 2)
        center_lat = np.clip(center_lat, lat_min + 2, lat_max - 2)
        
        dist_lon = lon_grid - center_lon
        dist_lat = lat_grid - center_lat
        field = np.exp(-((dist_lon ** 2 + dist_lat ** 2) / (2 * scale ** 2)))
        field += 0.1 * np.random.rand(*lon_grid.shape)
        
        all_fields.append(field)
        max_concentration = max(max_concentration, field.max())
    
    print(f"Global maximum concentration: {max_concentration:.6f}")
    
    # Use global maximum concentration for normalization
    for day in range(1, total_days + 1):
        field_normalized = all_fields[day-1] / max_concentration
        pollutant_data_all_days.append(field_normalized)
        
        # Output debugging information
        if day in days:
            center_lon = 135.0 + 0.1 * np.sin(day / 20.0)
            center_lat = 35.0 + 0.1 * np.cos(day / 20.0)
            scale = 1.0 + 0.8 * (day / total_days) + 0.2 * np.sin(day / 30.0)
            field = all_fields[day-1]
            print(f"{day}\t{center_lon:.3f}\t\t{center_lat:.3f}\t\t{scale:.3f}\t\t{field.max():.6f}")

    # Select data for specified days
    for d in days:
        pollutant_data.append(pollutant_data_all_days[d - 1])

    pollutant_name = f"Synthetic Pollutant Diffusion (Every {interval} days)"
    plot_pollutant_diffusion(lon_grid, lat_grid, pollutant_data, days, pollutant_name, pollutant_data_all_days)
    
    # Add diffusion range analysis
    print("\n=== Diffusion Range Analysis ===")
    spread_areas = []
    max_concentrations = []
    
    for i, day in enumerate(days):
        data = pollutant_data[i]
        # Calculate number of grid points with concentration > 0.2 as diffusion range indicator
        high_conc_mask = data > 0.2
        spread_area = np.sum(high_conc_mask)
        max_conc = np.max(data)
        spread_areas.append(spread_area)
        max_concentrations.append(max_conc)
        print(f"Day {day}: Spread Area={spread_area} grid points, Max Concentration={max_conc:.4f}")
    
    # Plot diffusion range change over time
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(days, spread_areas, 'bo-', linewidth=2, markersize=6)
    plt.title('Spread Area Over Time')
    plt.xlabel('Days')
    plt.ylabel('Spread Area (grid points)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(days, max_concentrations, 'ro-', linewidth=2, markersize=6)
    plt.title('Maximum Concentration Over Time')
    plt.xlabel('Days')
    plt.ylabel('Maximum Concentration')
    plt.grid(True, alpha=0.3)
    
    # Analyze changes in diffusion scale and center position
    scales = [1.0 + 0.8 * (day / total_days) + 0.2 * np.sin(day / 30.0) for day in days]
    center_lons = [135.0 + 0.1 * np.sin(day / 20.0) for day in days]
    center_lats = [35.0 + 0.1 * np.cos(day / 20.0) for day in days]
    
    plt.subplot(1, 3, 3)
    plt.plot(days, scales, 'go-', linewidth=2, markersize=6)
    plt.title('Diffusion Scale Over Time')
    plt.xlabel('Days')
    plt.ylabel('Diffusion Scale')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/diffusion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nDiffusion analysis plot saved to: ./output/diffusion_analysis.png")

if __name__ == "__main__":
    main()
