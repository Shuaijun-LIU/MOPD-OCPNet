'''
Created on Nov 2024
@author: <Shuaijun LIU, Qifu WEN>
'''

''' ------------- STEP1. GeoTIFF 1.1 Get Terrain Data and Visual --------------- '''
from GeoTIFF_Data import Get_GeoTIFF_Data
from GeoTIFF_Data import analyze_geotiff
from GeoTIFF_Data import tif_to_csv
from GeoTIFF_visual import visualize_elevation_map
import rasterio
import numpy as np
import os

# Input values for latitude, longitude, and elevation ranges
# Boston
# lat_min = 32.0  # Minimum latitude
# lat_max = 33.0  # Maximum latitude
# lon_min = -66.5  # Minimum longitude
# lon_max = -65.5  # Maximum longitude
# elev_min = -10000  # Minimum elevation
# elev_max = 0      # Maximum elevation

# US-NE
# lat_min = 32.0  # Minimum latitude
# lat_max = 42.0  # Maximum latitude
# lon_min = -70.5  # Minimum longitude
# lon_max = -65.5  # Maximum longitude
# elev_min = -10000  # Minimum elevation
# elev_max = 0      # Maximum elevation

# JP
lat_min = 24.0  # Minimum latitude
lat_max = 34.0  # Maximum latitude
lon_min = 132.0  # Minimum longitude
lon_max = 142.0  # Maximum longitude
elev_min = -11000  # Minimum elevation
elev_max = 0      # Maximum elevation

save_path = "./output"

# Call the function and retrieve the cropped data
cropped_data = Get_GeoTIFF_Data(lat_min, lat_max, lon_min, lon_max, elev_min, elev_max, save_path)

if cropped_data is not None:
    print("Shape of cropped data:", cropped_data.shape)
    # view angles
    view_angles = [(30, 45), (60, 45), (90, 45), (90, 135), (60, 135), (30, 135)]
    visualize_elevation_map(cropped_data, view_angles, lat_min, lat_max, lon_min, lon_max, save_path)
    print(f"Fig saved to {save_path}")
else:
    print("No data returned for the specified region.")

''' ------------- STEP 1. GeoTIFF 1.2 Get Terrain Data Inform / Trans to CAV (Optional) --------------- '''
cropped_file_path = "./output/filtered_data.tif"
result = analyze_geotiff(cropped_file_path)
if "error" in result:
    print(result["error"])
else:
    print(f"Shape: {result['shape']}")
    print(f"Min: {result['min_val']}, Max: {result['max_val']}, Mean: {result['mean_val']:.2f}")
    print(f"Data type: {result['data_type']}")

tif_to_csv
tif_to_csv(cropped_file_path)

''' ------------- STEP 2. GCPAF: Get Currents Forecast Data --------------- '''
from GCPAF_Data import fetch_and_merge_copernicus_data
if __name__ == "__main__":

    fetch_and_merge_copernicus_data(
        username='sliu42',  # CMEMS username
        password='Lsj200106013519',  # CMEMS password

#         # Bos
#         minimum_longitude=32.0,
#         maximum_longitude=33.0,
#         minimum_latitude=-66.5,
#         maximum_latitude=-65.5,

        # JP
        minimum_latitude = 24.0,
        maximum_latitude = 34.0,
        minimum_longitude = 132.0,
        maximum_longitude = 142.0,

        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-06-30T23:00:00",
        minimum_depth=0.4,  # Depth range is from 0.5 m to 5727.9 m
        maximum_depth=0.5,
        output_filename="combined_gcpaf_data.nc"
    )
print('===================================')

from GCPAF_Data import get_shapes
if __name__ == "__main__":
    file_path = "./Data/GCPAF/combined_gcpaf_data.nc"
    get_shapes(file_path)

''' ------------- STEP 3. Combine Terrain and Currents Data 3.1 Build Env Small-------------- '''
from Combine import interpolate_and_merge
from Combine import visualize_combined_data

interpolate_and_merge('./output/filtered_data.tif', './Data/GCPAF/combined_gcpaf_data.nc', './Data/Combined')

# visualize
visualize_combined_data('./Data/Combined/combined_environment.nc', time_index=0, depth_index=0)

''' ------------- STEP 3. Combine Terrain and Currents Data 3.2 Build Env Large (Optional)-------------- '''
# from Combine import interpolate_geotiff
#
# new_resolution = (1000, 1000)
# interpolated_data = interpolate_geotiff("./output/filtered_data.tif", new_resolution, "./output/interpolated_data.tif")
#
# if interpolated_data is not None:
#     print("New Data Shape:", interpolated_data.shape)
# else:
#     print("Failed")
#
# interpolate_and_merge('./output/interpolated_data.tif', './Data/GCPAF/combined_gcpaf_data.nc', './Data/Combined/large')


''' ------------- From NOAA Get Currents Ture Data (Optional) --------------- '''
# from get_UW_data import fetch_noaa_data
# from get_station_id import get_nearest_station_id
# from get_station_id import find_nearest_current_station
#
# if __name__ == "__main__":
#     latitude = 32.7749
#     longitude = -65.4194
#     station_id = get_nearest_station_id(latitude, longitude)
#     print(f"Using NOAA station ID: {station_id}")
#     fetch_noaa_data(station_id)
#
#     current_station_id = find_nearest_current_station(latitude, longitude)
#     if current_station_id:
#         print(f"Use Station ID: {current_station_id} for current data retrieval.")
#         fetch_noaa_data(current_station_id)

