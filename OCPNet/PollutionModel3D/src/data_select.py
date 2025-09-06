import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

def query_data_by_geo_range(lat_min, lat_max, lon_min, lon_max, data_folder):
    print(f"Searching for data in range: Latitude [{lat_min}, {lat_max}], Longitude [{lon_min}, {lon_max}]")

    region_file_map = {
        'N_W1': 'n90.0_s0.0_w-180.0_e-90.0',
        'N_W2': 'n90.0_s0.0_w-90.0_e0.0',
        'N_E1': 'n90.0_s0.0_w0.0_e90.0',
        'N_E2': 'n90.0_s0.0_w90.0_e180.0',
        'S_W1': 'n0.0_s-90.0_w-180.0_e-90.0',
        'S_W2': 'n0.0_s-90.0_w-90.0_e0.0',
        'S_E1': 'n0.0_s-90.0_w0.0_e90.0',
        'S_E2': 'n0.0_s-90.0_w90.0_e180.0'
    }

    value_cols = [
        'temperature', 'salinity', 'ugo', 'vgo', 'waveVelocity', 'chlorophyll',
        'nitrate', 'phosphate', 'silicate', 'dissolvedMolecularOxygen', 'nppv',
        'dissolvedIron', 'spCO2', 'ph', 'phytoplanktonExpressedAsCarbon'
    ]

    def region_intersects(region_code):
        parts = region_code.split('_')
        lat_zone = parts[0]
        lon_zone = parts[1]

        if lat_zone == 'N':
            region_lat_min, region_lat_max = 0, 90
        else:
            region_lat_min, region_lat_max = -90, 0

        if lon_zone == 'W1':
            region_lon_min, region_lon_max = -180, -90
        elif lon_zone == 'W2':
            region_lon_min, region_lon_max = -90, 0
        elif lon_zone == 'E1':
            region_lon_min, region_lon_max = 0, 90
        else:
            region_lon_min, region_lon_max = 90, 180

        return not (region_lat_max < lat_min or region_lat_min > lat_max or
                    region_lon_max < lon_min or region_lon_min > lon_max)

    all_dfs = []

    for region_code, filename in region_file_map.items():
        if region_intersects(region_code):
            file_path = os.path.join(data_folder, f"{filename}.parquet")
            if not os.path.exists(file_path):
                print(f"Warning: file not found: {file_path}")
                continue
            print(f"Reading: {file_path}")
            df = pd.read_parquet(file_path)
            df_filtered = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
                             (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]
            if not df_filtered.empty:
                all_dfs.append(df_filtered)

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / 'data' / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization to file
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min - 10, lon_max + 10, lat_min - 10, lat_max + 10], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True)

    # plot bounding box
    lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(lons, lats, color='red', linewidth=2, label='Selected Region')

    plt.title('Selected Geographic Region')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to file
    output_file = output_dir / f'selected_region_{lat_min}_{lat_max}_{lon_min}_{lon_max}.png'
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory
    print(f"Visualization saved to: {output_file}")

    if all_dfs:
        result_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal records found: {result_df.shape[0]}")

        print("\nMissing value counts:")
        print(result_df[value_cols].isnull().sum())

        print("\nDescriptive statistics:")
        print(result_df[value_cols].describe())

        print("\nExtreme values (row with max and min for each column):")
        for col in value_cols:
            if col in result_df.columns:
                max_idx = result_df[col].idxmax()
                min_idx = result_df[col].idxmin()
                print(f"\n{col}:")
                print("Max value at:")
                print(result_df.loc[max_idx, ['latitude', 'longitude', col]])
                print("Min value at:")
                print(result_df.loc[min_idx, ['latitude', 'longitude', col]])
        return result_df
    else:
        print("No data found in the specified range.")
        return pd.DataFrame()