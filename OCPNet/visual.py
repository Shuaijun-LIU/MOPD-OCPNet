import xarray as xr
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy import feature as cfeature
from shapely.vectorized import contains
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import math

import xarray as xr
import numpy as np
import os


def analyze_nc_file(nc_file_path):
    if not os.path.exists(nc_file_path):
        raise FileNotFoundError(f"File not found: {nc_file_path}")

    dataset = xr.open_dataset(nc_file_path)

    print("Variables in the dataset:")
    print(dataset.variables.keys())

    def compute_stats(var):
        if var in ["time", "depth", "latitude", "longitude"]:
            return None

        data = dataset[var].values.flatten()
        data = data[~np.isnan(data)]
        stats = {
            "Min": np.min(data),
            "Max": np.max(data),
            "Mean": np.mean(data),
            "Median": np.median(data),
            "Std": np.std(data)
        }
        return stats

    stats_dict = {}
    for var in dataset.variables.keys():
        stats = compute_stats(var)
        if stats:
            stats_dict[var] = stats

    print("\nStatistical Information:")
    for var, stats in stats_dict.items():
        print(f"\nVariable: {var}")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")

    time_info = None
    if "time" in dataset.variables:
        time_values = dataset["time"].values
        start_time = time_values[0]
        end_time = time_values[-1]
        time_span = end_time - start_time

        time_info = (start_time, end_time, time_span)

        print("\nTime Range:")
        print(f"  Start Time: {start_time}")
        print(f"  End Time: {end_time}")
        print(f"  Time Span: {time_span}")

    dataset.close()
    return stats_dict, time_info

def plot_3d_currents(dataset, output_dir, skip=2, arrow_size=0.05, arrow_height_offset=5, arrow_alpha=0.6,
                     arrow_head_size=10):
    os.makedirs(output_dir, exist_ok=True)

    elevation = dataset["elevation"].values
    lats = dataset["latitude"].values
    lons = dataset["longitude"].values

    uo = dataset["uo"].isel(time=0, depth=0).values
    vo = dataset["vo"].isel(time=0, depth=0).values
    utotal = dataset["utotal"].isel(time=0, depth=0).values
    vtotal = dataset["vtotal"].isel(time=0, depth=0).values

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    base_speed = np.sqrt(uo ** 2 + vo ** 2)
    total_speed = np.sqrt(utotal ** 2 + vtotal ** 2)

    norm_base = mcolors.Normalize(vmin=np.min(base_speed), vmax=np.max(base_speed))
    norm_total = mcolors.Normalize(vmin=np.min(total_speed), vmax=np.max(total_speed))
    cmap = plt.cm.viridis

    arrow_height = np.max(elevation) + arrow_height_offset

    fig1, ax1 = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
    
    surf1 = ax1.plot_surface(lon_grid, lat_grid, elevation, facecolors=cmap(norm_base(base_speed)),
                             edgecolor='none', alpha=1)

    lon_sample = lon_grid[::skip, ::skip]
    lat_sample = lat_grid[::skip, ::skip]
    uo_sample = uo[::skip, ::skip]
    vo_sample = vo[::skip, ::skip]
    
    speed = np.sqrt(uo_sample**2 + vo_sample**2)
    mask = speed > 1e-6
    
    for i in range(lon_sample.shape[0]):
        for j in range(lon_sample.shape[1]):
            if mask[i, j]:
                ax1.quiver(lon_sample[i, j], lat_sample[i, j], arrow_height,
                          uo_sample[i, j]/speed[i, j], vo_sample[i, j]/speed[i, j], 0,
                          color='black', alpha=arrow_alpha, length=arrow_size,
                          normalize=True)

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_base)
    sm1.set_array([])
    fig1.colorbar(sm1, ax=ax1, shrink=0.5, aspect=10, label="Base Current Speed (m/s)")

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_zlabel("Elevation (m)")
    ax1.set_title("3D Terrain Colored by Base Current Speed (uo, vo) with Transparent Arrows")

    base_output_path = os.path.join(output_dir, "base_current_3d.png")
    plt.show()
    plt.savefig(base_output_path, dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
    
    surf2 = ax2.plot_surface(lon_grid, lat_grid, elevation, facecolors=cmap(norm_total(total_speed)),
                             edgecolor='none', alpha=1)

    utotal_sample = utotal[::skip, ::skip]
    vtotal_sample = vtotal[::skip, ::skip]
    
    speed = np.sqrt(utotal_sample**2 + vtotal_sample**2)
    mask = speed > 1e-6
    
    for i in range(lon_sample.shape[0]):
        for j in range(lon_sample.shape[1]):
            if mask[i, j]:
                ax2.quiver(lon_sample[i, j], lat_sample[i, j], arrow_height,
                          utotal_sample[i, j]/speed[i, j], vtotal_sample[i, j]/speed[i, j], 0,
                          color='black', alpha=arrow_alpha, length=arrow_size,
                          normalize=True)

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_total)
    sm2.set_array([])
    fig2.colorbar(sm2, ax=ax2, shrink=0.5, aspect=10, label="Total Current Speed (m/s)")

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_zlabel("Elevation (m)")
    ax2.set_title("3D Terrain Colored by Total Current Speed (utotal, vtotal) with Transparent Arrows")

    total_output_path = os.path.join(output_dir, "total_current_3d.png")
    plt.show()
    plt.savefig(total_output_path, dpi=300)
    plt.close(fig2)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
from cartopy import feature as cfeature
from shapely.vectorized import contains
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import math


def plot_pollutant_diffusion(lon_grid, lat_grid, pollutant_data, days, pollutant_name, pollutant_data_all_days=None):
    land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    land_geoms = list(land_feature.geometries())
    ocean_mask = np.ones(lon_grid.shape, dtype=bool)
    for geom in land_geoms:
        ocean_mask &= ~contains(geom, lon_grid, lat_grid)

    elev_noise = np.random.rand(*lon_grid.shape)
    land_elev = gaussian_filter(elev_noise, sigma=3) * 1000
    land_elev[ocean_mask] = np.nan

    land_colors = [(0.0, "#2e4536"), (0.5, "#4e5e3c"), (1.0, "#a69176")]
    land_cmap = mcolors.LinearSegmentedColormap.from_list("satellite_land", land_colors)
    pollutant_colors = [(0.0, "#b3e6ff"), (0.2, "#9ad1f0"), (0.4, "#80bccc"),
                        (0.6, "#f0e68c"), (0.8, "#ff9999"), (1.0, "#ff0000")]
    pollutant_cmap = mcolors.LinearSegmentedColormap.from_list("pollutant_custom", pollutant_colors)
    contour_levels = np.linspace(0.2, 1.0, 9)

    n = len(days)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4 * ncols, 5 * nrows),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.array(axes).flatten()
    fig.suptitle(pollutant_name, fontsize=16)

    for i, day in enumerate(days):
        ax = axes[i]
        ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())
        ax.set_facecolor('#000c3f')
        gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.pcolormesh(lon_grid, lat_grid, land_elev, cmap=land_cmap, shading='auto', transform=ccrs.PlateCarree())
        pollutant = np.copy(pollutant_data[i])
        pollutant[pollutant < 0.2] = np.nan
        pollutant[~ocean_mask] = np.nan
        cf = ax.contourf(lon_grid, lat_grid, pollutant, levels=contour_levels,
                         cmap=pollutant_cmap, alpha=1, transform=ccrs.PlateCarree())
        cl = ax.contour(lon_grid, lat_grid, pollutant, levels=contour_levels,
                        colors='#aa66f5', linewidths=0.5, transform=ccrs.PlateCarree())
        ax.clabel(cl, inline=True, fmt="%.2f", fontsize=8, colors="#001f7f")
        ax.coastlines(resolution='10m', color='black', linewidth=0.2)
        ax.set_title(f"Day {day}", fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(cf, cax=cbar_ax)
    cb.set_label('Pollutant Concentration')
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.9)
    dpi_value = 300

    # fig.savefig("pollutant_diffusion.eps", format="eps", dpi=dpi_value)
    # print("Saved as pollutant_diffusion.eps")
    # fig.savefig("pollutant_diffusion.pdf", format="pdf", dpi=dpi_value)
    # print("Saved as pollutant_diffusion.pdf")
    # fig.savefig("pollutant_diffusion.png", format="png", dpi=dpi_value)
    # print("Saved as pollutant_diffusion.png")
    plt.show()

    if pollutant_data_all_days is not None:
        fig_anim, ax_anim = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax_anim.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()])
        ax_anim.set_facecolor('#000c3f')
        gl = ax_anim.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax_anim.pcolormesh(lon_grid, lat_grid, land_elev, cmap=land_cmap,
                           shading='auto', transform=ccrs.PlateCarree())
        ax_anim.coastlines(resolution='10m', color='black', linewidth=0.2)
        title = ax_anim.set_title("")

        img = [ax_anim.contourf(lon_grid, lat_grid, np.full_like(lon_grid, np.nan),
                                levels=contour_levels, cmap=pollutant_cmap, transform=ccrs.PlateCarree())]

        def update(frame):
            for coll in img[0].collections:
                coll.remove()
            data = np.copy(pollutant_data_all_days[frame])
            data[data < 0.2] = np.nan
            data[~ocean_mask] = np.nan
            img[0] = ax_anim.contourf(lon_grid, lat_grid, data,
                                      levels=contour_levels, cmap=pollutant_cmap, transform=ccrs.PlateCarree())
            title.set_text(f"Day {frame + 1}")
            return img[0].collections

        ani = animation.FuncAnimation(fig_anim, update, frames=len(pollutant_data_all_days), interval=200, blit=False)
        ani.save("pollutant_diffusion.gif", writer='pillow', fps=5)
        print("GIF saved as 'pollutant_diffusion.gif'")


