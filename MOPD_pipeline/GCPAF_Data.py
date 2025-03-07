import copernicusmarine
import xarray as xr
import os
import numpy as np

'''
Parameters:
    - so: 海水盐度 (Sea water salinity)
    - thetao: 海水潜在温度 (Sea water potential temperature)
    - uo: 东向海水流速 (Eastward sea water velocity)
    - vo: 北向海水流速 (Northward sea water velocity)
    - zos: 海面高度 (Sea surface height above geoid)
    - utide: 东向潮流速度 (Eastward tidal velocity)
    - utotal: 总东向海水流速 (Total eastward sea water velocity)
    #- vsdx: 海浪斯托克斯漂移速度 - x 方向 (Sea surface wave stokes drift x velocity)
    #- vsdy: 海浪斯托克斯漂移速度 - y 方向 (Sea surface wave stokes drift y velocity)
    - vtide: 北向潮流速度 (Northward tidal velocity)
    - vtotal: 总北向海水流速 (Total northward sea water velocity)
'''

def fetch_and_merge_copernicus_data(username, password,
                                    minimum_longitude, maximum_longitude,
                                    minimum_latitude, maximum_latitude,
                                    start_datetime, end_datetime,
                                    minimum_depth, maximum_depth,
                                    output_filename):

    copernicusmarine.login(
        username=username,
        password=password
    )

    output_directory = "./Data/GCPAF"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        variables=["so", "thetao", "uo", "vo", "zos"],
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        output_filename=f"{output_directory}/cmems_basic_phy_output_data.nc"
    )

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
        variables=["utide", "utotal", "vtide", "vtotal"],
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        output_filename=f"{output_directory}/cmems_detailed_uv_output_data.nc"
    )

    basic_phy_data = xr.open_dataset(f"{output_directory}/cmems_basic_phy_output_data.nc")
    detailed_uv_data = xr.open_dataset(f"{output_directory}/cmems_detailed_uv_output_data.nc")

    # Merge datasets
    combined_data = xr.merge([basic_phy_data, detailed_uv_data])

    # Step 1: Fill missing values using rolling mean interpolation
    for var in combined_data.data_vars:
        if combined_data[var].isnull().any():
            combined_data[var] = combined_data[var].where(
                ~np.isnan(combined_data[var]),
                combined_data[var].rolling(latitude=3, longitude=3, center=True, min_periods=1).mean()
            )

    # Step 2: Interpolate along the time dimension using linear interpolation
    combined_data = combined_data.interpolate_na(dim="time", method="linear")


    '''Check the nan'''
    # data_vars_nan = {var: combined_data[var].isnull().sum().values for var in combined_data.data_vars}
    #
    #
    # coords_nan = {coord: combined_data[coord].isnull().sum().values for coord in combined_data.coords if
    #               combined_data[coord].isnull().any()}
    #
    # dims_nan = {}
    # for dim in combined_data.dims:
    #     try:
    #         dims_nan[dim] = combined_data[dim].isnull().sum().values
    #     except AttributeError:
    #         dims_nan[dim] = 'Not Applicable'
    #
    # print("\n--- Detailed NaN Summary ---")
    # print("Data Variables (data_vars):")
    # for var, nan_count in data_vars_nan.items():
    #     print(f"  {var}: {nan_count} NaNs")
    # print("\nCoordinates (coords):")
    # for coord, nan_count in coords_nan.items():
    #     print(f"  {coord}: {nan_count} NaNs")
    # print("\nDimensions (dims):")
    # for dim, nan_info in dims_nan.items():
    #     print(f"  {dim}: {nan_info}")
    #
    # if any(count > 0 for count in data_vars_nan.values()) or coords_nan:
    #     raise ValueError("!!! NaN values still exist in the dataset after interpolation")

    combined_data.to_netcdf(f"{output_directory}/{output_filename}")

    print(combined_data)


def get_shapes(file_path):
    ds = xr.open_dataset(file_path)

    for var_name in ds.data_vars:
        data = ds[var_name]
        shape_details = ', '.join([f"{dim_name}: {dim_size}" for dim_name, dim_size in zip(data.dims, data.shape)])
        print(f"Variable '{var_name}' shape: {data.shape} ({shape_details})")
