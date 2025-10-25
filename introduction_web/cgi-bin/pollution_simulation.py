#!/usr/bin/env python3
"""
CGI script for pollution simulation web interface
This script processes simulation requests and runs the OCPNet/simu1.py simulation
"""

import sys
import os
import json
import cgi
import cgitb
import subprocess
import tempfile
import shutil
from pathlib import Path
import time

# Enable CGI error reporting
cgitb.enable()

# Add the OCPNet directory to Python path
ocpnet_path = Path(__file__).parent.parent.parent / "OCPNet"
sys.path.insert(0, str(ocpnet_path))

def run_pollution_simulation(params):
    """Run the pollution simulation with given parameters"""
    try:
        # Import the simulation module
        from simu1 import PollutantDiffusionPlotter, simulate_gaussian_fields
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Extract parameters
        pollutant_type = params.get('pollutantType', 'oil')
        initial_dose = float(params.get('initialDose', 1000))
        longitude = float(params.get('longitude', 135))
        latitude = float(params.get('latitude', 35))
        simulation_days = int(params.get('simulationDays', 30))
        water_depth = float(params.get('waterDepth', 50))
        reactions = params.getlist('reactions') if isinstance(params.getlist('reactions'), list) else [params.get('reactions', 'biodegradation')]
        environmental_conditions = params.get('environmentalConditions', 'moderate')
        
        # Create domain around the specified location
        lon_min, lon_max = longitude - 5, longitude + 5
        lat_min, lat_max = latitude - 5, latitude + 5
        resolution = 0.1
        
        lons = np.arange(lon_min, lon_max + resolution, resolution)
        lats = np.arange(lat_min, lat_max + resolution, resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Create timeline based on simulation days
        days = list(range(1, simulation_days + 1, max(1, simulation_days // 6)))
        if days[-1] != simulation_days:
            days.append(simulation_days)
        
        # Generate pollutant data based on parameters
        pollutant_stack = simulate_gaussian_fields(lon_grid, lat_grid, days)
        
        # Adjust simulation based on environmental conditions
        if environmental_conditions == 'calm':
            # Slower dispersion
            for i, data in enumerate(pollutant_stack):
                pollutant_stack[i] = data * 0.7
        elif environmental_conditions == 'stormy':
            # Faster dispersion
            for i, data in enumerate(pollutant_stack):
                pollutant_stack[i] = data * 1.3
        
        # Create plotter
        plotter = PollutantDiffusionPlotter(
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            pollutant_stack=pollutant_stack,
            days=days,
            pollutant_name=f"{pollutant_type.title()} Dispersion Simulation",
            threshold=0.2,
            contour_levels=np.linspace(0.2, 1.0, 9)
        )
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "output" / "pollution_simulation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate animation
        anim = plotter.animate(interval_ms=700, figsize=(10, 8), blit=False)
        
        # Save outputs
        gif_path = output_dir / "pollutant_diffusion.gif"
        mp4_path = output_dir / "pollutant_diffusion.mp4"
        png_path = output_dir / "pollutant_panels.png"
        
        plotter.save_gif(anim, str(gif_path), fps=1)
        plotter.save_mp4(anim, str(mp4_path), fps=2)
        plotter.save_grid_panels(str(png_path).replace('.png', ''), ncols=3, dpi=300)
        
        # Generate statistics
        max_concentrations = [np.nanmax(data) for data in pollutant_stack]
        total_dispersion = [np.nansum(data) for data in pollutant_stack]
        
        statistics = {
            'max_concentrations': max_concentrations,
            'total_dispersion': total_dispersion,
            'days': days,
            'simulation_parameters': {
                'pollutant_type': pollutant_type,
                'initial_dose': initial_dose,
                'location': [longitude, latitude],
                'simulation_days': simulation_days,
                'water_depth': water_depth,
                'reactions': reactions,
                'environmental_conditions': environmental_conditions
            }
        }
        
        # Save statistics
        stats_path = output_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        return {
            'success': True,
            'output_files': {
                'gif': str(gif_path.relative_to(Path(__file__).parent.parent)),
                'mp4': str(mp4_path.relative_to(Path(__file__).parent.parent)),
                'png': str(png_path.relative_to(Path(__file__).parent.parent)),
                'statistics': str(stats_path.relative_to(Path(__file__).parent.parent))
            },
            'statistics': statistics
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main CGI function"""
    # Set content type
    print("Content-Type: application/json")
    print()
    
    try:
        # Parse form data
        form = cgi.FieldStorage()
        
        # Convert form data to dictionary
        params = {}
        for key in form.keys():
            if key == 'reactions':
                params[key] = form.getlist(key)
            else:
                params[key] = form.getvalue(key)
        
        # Run simulation
        result = run_pollution_simulation(params)
        
        # Return JSON response
        print(json.dumps(result))
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': f'CGI Error: {str(e)}'
        }
        print(json.dumps(error_response))

if __name__ == '__main__':
    main()
