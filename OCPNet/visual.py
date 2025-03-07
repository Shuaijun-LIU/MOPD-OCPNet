import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import os

def plot_3d_currents(dataset, output_dir, skip=2, arrow_size=0.05, arrow_height_offset=5, arrow_alpha=0.6, arrow_head_size=10):
    os.makedirs(output_dir, exist_ok=True)

    elevation = dataset["elevation"].values
    lats = dataset["latitude"].values
    lons = dataset["longitude"].values

    uo = dataset["uo"].isel(time=0, depth=0).values
    vo = dataset["vo"].isel(time=0, depth=0).values
    utotal = dataset["utotal"].isel(time=0, depth=0).values
    vtotal = dataset["vtotal"].isel(time=0, depth=0).values

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    base_speed = np.sqrt(uo**2 + vo**2)
    total_speed = np.sqrt(utotal**2 + vtotal**2)

    norm_base = mcolors.Normalize(vmin=np.min(base_speed), vmax=np.max(base_speed))
    norm_total = mcolors.Normalize(vmin=np.min(total_speed), vmax=np.max(total_speed))
    cmap = plt.cm.viridis

    arrow_height = elevation + arrow_height_offset

    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')

    surf1 = ax1.plot_surface(lon_grid, lat_grid, elevation, facecolors=cmap(norm_base(base_speed)),
                             edgecolor='none', alpha=1)

    for i in range(0, uo.shape[0], skip):
        for j in range(0, uo.shape[1], skip):
            norm_factor = np.sqrt(uo[i, j]**2 + vo[i, j]**2) + 1e-6
            ax1.quiver(lon_grid[i, j], lat_grid[i, j], arrow_height,
                       uo[i, j] / norm_factor, vo[i, j] / norm_factor, 0,
                       color='black', alpha=arrow_alpha, length=arrow_size, linewidth=0.8, arrow_length_ratio=0.8)
            ax1.scatter(lon_grid[i, j] + uo[i, j] / norm_factor * arrow_size,
                        lat_grid[i, j] + vo[i, j] / norm_factor * arrow_size,
                        arrow_height,
                        color='black', alpha=arrow_alpha, s=arrow_head_size, marker='^')

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_base)
    sm1.set_array([])
    fig1.colorbar(sm1, ax=ax1, shrink=0.5, aspect=10, label="Base Current Speed (m/s)")

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_zlabel("Elevation (m)")
    ax1.set_title("3D Terrain Colored by Base Current Speed (uo, vo) with Transparent Arrows")

    base_output_path = os.path.join(output_dir, "base_current_3d.png")
    plt.savefig(base_output_path, dpi=300)
    plt.show()
    plt.close(fig1)

    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')

    surf2 = ax2.plot_surface(lon_grid, lat_grid, elevation, facecolors=cmap(norm_total(total_speed)),
                             edgecolor='none', alpha=1)

    for i in range(0, utotal.shape[0], skip):
        for j in range(0, utotal.shape[1], skip):
            norm_factor = np.sqrt(utotal[i, j]**2 + vtotal[i, j]**2) + 1e-6
            ax2.quiver(lon_grid[i, j], lat_grid[i, j], arrow_height,
                       utotal[i, j] / norm_factor, vtotal[i, j] / norm_factor, 0,
                       color='black', alpha=arrow_alpha, length=arrow_size, linewidth=0.8, arrow_length_ratio=0.8)
            ax2.scatter(lon_grid[i, j] + utotal[i, j] / norm_factor * arrow_size,
                        lat_grid[i, j] + vtotal[i, j] / norm_factor * arrow_size,
                        arrow_height,
                        color='black', alpha=arrow_alpha, s=arrow_head_size, marker='^')

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_total)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax2, shrink=0.5, aspect=10, label="Total Current Speed (m/s)")

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_zlabel("Elevation (m)")
    ax2.set_title("3D Terrain Colored by Total Current Speed (utotal, vtotal) with Transparent Arrows")

    total_output_path = os.path.join(output_dir, "total_current_3d.png")
    plt.savefig(total_output_path, dpi=300)
    plt.show()
    plt.close(fig2)
