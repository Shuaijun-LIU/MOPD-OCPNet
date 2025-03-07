import os
import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.interpolate import griddata


def interpolate_and_merge(terrain_file, water_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read terrain file
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)
        bounds = src.bounds
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

    # Read water data
    ds = xr.open_dataset(water_file)
    original_lats = ds['latitude'].values
    original_lons = ds['longitude'].values
    original_depths = ds['depth'].values
    original_times = ds['time'].values

    lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)
    target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

    variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
    interpolated_variables = {var: [] for var in variables_to_interpolate}

    for var in variables_to_interpolate:
        print(f"Interpolating variable: {var}")

        for t_idx, time in enumerate(original_times):
            for d_idx, depth in enumerate(original_depths):
                print(f"  Time: {time}, Depth: {depth}")
                data = ds[var].isel(time=t_idx, depth=d_idx).values

                valid_mask = np.isfinite(data)
                if not valid_mask.any():
                    print(
                        f"  Warning: All values are NaN or Inf for variable {var} at time {time}, depth {depth}. Skipping.")
                    interpolated_variables[var].append(np.full((len(lats), len(lons)), np.nan))
                    continue

                valid_lon = lon_grid[valid_mask]
                valid_lat = lat_grid[valid_mask]
                valid_data = data[valid_mask]

                rbf_interpolator = Rbf(valid_lon, valid_lat, valid_data, function='linear')
                interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
                interpolated_variables[var].append(interpolated_data)

    combined_variables = {}
    for var, interpolated_list in interpolated_variables.items():
        stacked_data = np.stack(interpolated_list, axis=0)
        stacked_data = stacked_data.reshape((len(original_times), len(original_depths), len(lats), len(lons)))
        combined_variables[var] = (['time', 'depth', 'latitude', 'longitude'], stacked_data)

    combined_variables["elevation"] = (["latitude", "longitude"], elevation)

    combined_ds = xr.Dataset(combined_variables,
                             coords={'time': original_times, 'depth': original_depths, 'latitude': lats,
                                     'longitude': lons})
    combined_output_path = os.path.join(output_dir, "combined_environment.nc")
    combined_ds.to_netcdf(combined_output_path)

    print('===================================')
    print(f"All interpolated and combined data have been saved to {output_dir}")
    print("Combined dataset:")
    print(combined_ds)
    print('===================================')


def visualize_combined_data(combined_file, selected_time=None, selected_depth=None, time_index=None, depth_index=None):
    combined_ds = xr.open_dataset(combined_file)

    if time_index is not None:
        selected_time = combined_ds['time'].values[time_index]
    if depth_index is not None:
        selected_depth = combined_ds['depth'].values[depth_index]

    for var in combined_ds.data_vars:
        data = combined_ds[var]
        print(f"Data characteristics for variable {var}:")
        print(f"  Mean: {data.mean().item()} ")
        print(f"  Std Dev: {data.std().item()} ")
        print(f"  Min: {data.min().item()} ")
        print(f"  Max: {data.max().item()} ")

    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cool', 'hot', 'spring', 'summer']
    for i, var in enumerate(combined_ds.data_vars):
        if selected_time is not None and selected_depth is not None:
            plt.figure(figsize=(10, 6))
            combined_ds[var].sel(time=selected_time, depth=selected_depth).plot(cmap=colormaps[i % len(colormaps)])
            plt.title(f"Combined Data: {var} (Time: {selected_time}, Depth: {selected_depth})")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.show()
        else:
            for t_idx in range(len(combined_ds['time'])):
                for d_idx in range(len(combined_ds['depth'])):
                    plt.figure(figsize=(10, 6))
                    combined_ds[var].isel(time=t_idx, depth=d_idx).plot(cmap=colormaps[i % len(colormaps)])
                    plt.title(
                        f"Combined Data: {var} (Time: {combined_ds['time'].values[t_idx]}, Depth: {combined_ds['depth'].values[d_idx]})")
                    plt.xlabel("Longitude")
                    plt.ylabel("Latitude")
                    plt.show()


def print_structure(combined_file):
    combined_ds = xr.open_dataset(combined_file)
    print("Combined dataset structure:")
    print(combined_ds)


def interpolate_geotiff(file_path, new_resolution, save_path):
    try:
        with rasterio.open(file_path) as dataset:

            data = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
            profile = dataset.profile

            rows, cols = data.shape
            x = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
            y = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
            xv, yv = np.meshgrid(x, y)

            valid_mask = ~np.isnan(data)
            valid_points = np.array([xv[valid_mask], yv[valid_mask]]).T
            valid_values = data[valid_mask]

            new_x = np.linspace(x.min(), x.max(), new_resolution[1])
            new_y = np.linspace(y.min(), y.max(), new_resolution[0])
            new_xv, new_yv = np.meshgrid(new_x, new_y)

            interpolated_data = griddata(valid_points, valid_values, (new_xv, new_yv), method='linear')

            new_transform = rasterio.transform.from_bounds(new_x.min(), new_y.min(), new_x.max(), new_y.max(),
                                                           new_resolution[1], new_resolution[0])
            profile.update({
                'height': new_resolution[0],
                'width': new_resolution[1],
                'transform': new_transform,
                'dtype': 'float32'
            })

            with rasterio.open(save_path, 'w', **profile) as dst:
                dst.write(interpolated_data.astype(rasterio.float32), 1)

            print(f"* tif saved as: {save_path}")
            return interpolated_data

    except Exception as e:
        print(f"!!! error: {e}")
        return None