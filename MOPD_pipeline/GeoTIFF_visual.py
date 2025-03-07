import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_elevation_map(cropped_data, view_angles, lat_min, lat_max, lon_min, lon_max, save_path):
    # Create latitude and longitude grids
    latitudes = np.linspace(lat_min, lat_max, cropped_data.shape[0])
    longitudes = np.linspace(lon_min, lon_max, cropped_data.shape[1])
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Iterate through the list of view angles and plot the map from different perspectives
    for idx, (elev, azim) in enumerate(view_angles):
        fig = plt.figure(figsize=(10, 7), dpi=300)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(lon_grid, lat_grid, cropped_data, cmap='cividis', edgecolor='none')

        # color bar and labels
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation (m)')
        ax.set_title(f'3D Elevation Map (Elev: {elev}, Azim: {azim})')

        # view angle
        ax.view_init(elev=elev, azim=azim)

        # Save figure to the specified path
        plt.savefig(f'{save_path}/3D_Elevation_Map_Elev_{elev}_Azim_{azim}.png')

        plt.show()