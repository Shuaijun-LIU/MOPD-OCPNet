import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import re
import os

# Folder path for GeoTIFF files
folder_path = './Data/gebco_2024_sub_ice_topo_geotiff/'

file_paths = glob.glob(folder_path + '/*.tif')

def save_as_geotiff(data, dataset, save_path, window_bounds):
    profile = dataset.profile
    left, bottom, right, top = window_bounds
    new_transform = from_bounds(left, bottom, right, top, data.shape[1], data.shape[0])

    profile.update({
        'dtype': rasterio.float32,
        'count': 1,
        'height': data.shape[0],
        'width': data.shape[1],
        'transform': new_transform,
        'crs': dataset.crs
    })

    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.float32), 1)

def Get_GeoTIFF_Data(lat_min, lat_max, lon_min, lon_max, elev_min, elev_max, save_path):
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]

        match = re.search(r'n(-?\d+(\.\d+)?)_s(-?\d+(\.\d+)?)_w(-?\d+(\.\d+)?)_e(-?\d+(\.\d+)?)', file_name)
        print(f"match: {match.group()}")

        if match:
            n = float(match.group(1))  # Northern boundary
            s = float(match.group(3))  # Southern boundary
            w = float(match.group(5))  # Western boundary
            e = float(match.group(7))  # Eastern boundary

            if (s <= lat_min <= n or s <= lat_max <= n) and (w <= lon_min <= e or w <= lon_max <= e):
                print(f"Loading file: {file_path}")
                with rasterio.open(file_path) as dataset:
                    full_elevation_data = dataset.read(1)

                    # Print sample data from the top-left 5x5 pixels in the full file
                    print("Sample elevation data from the top-left 5x5 pixels in the full file:")
                    print(full_elevation_data[:5, :5])

                    print("\n" + "-" * 50 + "\n")

                    window = dataset.window(lon_min, lat_min, lon_max, lat_max)

                    elevation_data = dataset.read(1, window=window)

                    filtered_data = np.where((elevation_data >= elev_min) & (elevation_data <= elev_max),
                                             elevation_data, np.nan)

                    left, bottom, right, top = dataset.window_bounds(window)
                    extent = [left, right, bottom, top]

                    # cropped area statistics
                    print("Cropped elevation data statistics:")
                    print(f"Minimum value (lowest point): {np.nanmin(filtered_data)} m")
                    print(f"Maximum value (highest point): {np.nanmax(filtered_data)} m")
                    print(f"Mean value: {np.nanmean(filtered_data):.2f} m")
                    print(f"Data type: {filtered_data.dtype}")
                    print("\n" + "-" * 50 + "\n")

                    plt.imshow(filtered_data, cmap='terrain', extent=extent)
                    plt.colorbar(label='Elevation (m)')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.title(f'Underwater Terrain 2024 ({lat_min}° to {lat_max}°N, {lon_min}° to {lon_max}°E, {elev_min}m to {elev_max}m)')

                    # Save the fig
                    filename = 'GeoTIFF_Terrain.png'
                    full_save_path = os.path.join(save_path, filename)
                    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    save_as_geotiff(filtered_data, dataset, os.path.join(save_path, 'filtered_data.tif'), (left, bottom, right, top))

                return filtered_data

    print("No matching file found for the specified coordinates.")
    return None


def analyze_geotiff(file_path):
    try:
        with rasterio.open(file_path) as dataset:
            data = dataset.read(1)
            stats = {
                "shape": data.shape,
                "min_val": np.nanmin(data),
                "max_val": np.nanmax(data),
                "mean_val": np.nanmean(data),
                "data_type": data.dtype
            }
            return stats
    except FileNotFoundError:
        return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}


def tif_to_csv(file_path):
    with rasterio.open(file_path) as dataset:
        print("Reading data...")
        data = dataset.read(1)
        print("Converting data to DataFrame...")
        df = pd.DataFrame(data)
        print("Saving data to CSV...")
        csv_output_path = os.path.splitext(file_path)[0] + ".csv"
        df.to_csv(csv_output_path, index=False, header=False)
        print("CSV file saved successfully.")
