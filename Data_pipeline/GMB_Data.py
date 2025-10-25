# import dask.dataframe as dd
#
# print("Step 1: Reading CSV with selected columns and dtype handling...")
# file_path = 'Data/GMB/Global_Marine_Biodiversity_Records_All.csv'
#
# keep_columns = [
#     'eventDate', 'year', 'month', 'day', 'latitude', 'longitude', 'depth',
#     'country', 'countryCode', 'locality', 'continent', 'waterBody',
#     'temperature', 'salinity', 'ugo', 'vgo', 'waveVelocity', 'chlorophyll',
#     'nitrate', 'phosphate', 'silicate', 'dissolvedMolecularOxygen', 'nppv',
#     'dissolvedIron', 'spCO2', 'ph', 'phytoplanktonExpressedAsCarbon'
# ]
#
# dtypes = {col: 'object' for col in keep_columns}
# df = dd.read_csv(file_path, usecols=keep_columns, dtype=dtypes)
#
# print("Step 2: Converting numeric columns...")
# numeric_cols = [
#     'latitude', 'longitude', 'depth', 'temperature', 'salinity',
#     'ugo', 'vgo', 'waveVelocity', 'chlorophyll', 'nitrate', 'phosphate',
#     'silicate', 'dissolvedMolecularOxygen', 'nppv', 'dissolvedIron',
#     'spCO2', 'ph', 'phytoplanktonExpressedAsCarbon'
# ]
# for col in numeric_cols:
#     df[col] = dd.to_numeric(df[col], errors='coerce')
#
# print("Step 3: Dropping duplicates...")
# df = df.drop_duplicates()
#
# print("Step 4: Assigning region ID based on latitude and longitude...")
#
# def assign_region(lat, lon):
#     # Divide the globe into 8 regions: 2 lat zones × 4 lon quadrants
#     if lat is None or lon is None:
#         return 'unknown'
#     try:
#         lat = float(lat)
#         lon = float(lon)
#         if lat >= 0:
#             if -180 <= lon < -90:
#                 return 'N_W1'
#             elif -90 <= lon < 0:
#                 return 'N_W2'
#             elif 0 <= lon < 90:
#                 return 'N_E1'
#             else:
#                 return 'N_E2'
#         else:
#             if -180 <= lon < -90:
#                 return 'S_W1'
#             elif -90 <= lon < 0:
#                 return 'S_W2'
#             elif 0 <= lon < 90:
#                 return 'S_E1'
#             else:
#                 return 'S_E2'
#     except:
#         return 'unknown'
#
# df['region'] = df.map_partitions(lambda d: d.apply(lambda row: assign_region(row['latitude'], row['longitude']), axis=1), meta=('region', 'object'))
#
# print("Step 5: Saving to 8 Parquet partitions by region...")
#
# schema = {
#     'eventDate': 'string',
#     'year': 'string',
#     'month': 'string',
#     'day': 'string',
#     'latitude': 'float64',
#     'longitude': 'float64',
#     'depth': 'float64',
#     'country': 'string',
#     'countryCode': 'string',
#     'locality': 'string',
#     'continent': 'string',
#     'waterBody': 'string',
#     'temperature': 'float64',
#     'salinity': 'float64',
#     'ugo': 'float64',
#     'vgo': 'float64',
#     'waveVelocity': 'float64',
#     'chlorophyll': 'float64',
#     'nitrate': 'float64',
#     'phosphate': 'float64',
#     'silicate': 'float64',
#     'dissolvedMolecularOxygen': 'float64',
#     'nppv': 'float64',
#     'dissolvedIron': 'float64',
#     'spCO2': 'float64',
#     'ph': 'float64',
#     'phytoplanktonExpressedAsCarbon': 'float64',
#     'region': 'string'
# }
#
# df.to_parquet(
#     'Data/GMB/cleaned_marine_dataset_by_region/',
#     engine='pyarrow',
#     write_index=False,
#     schema=schema,
#     partition_on=['region']
# )
#
# print("Finished: Output saved in 8 regional folders under 'cleaned_marine_dataset_by_region/'.")
#
#
#
# import pandas as pd
# import glob
# import os
#
# print("Starting per-region analysis...")
#
# base_path = 'Data/GMB/cleaned_marine_dataset_by_region/'
#
# numeric_cols = [
#     'latitude', 'longitude', 'depth', 'temperature', 'salinity',
#     'ugo', 'vgo', 'waveVelocity', 'chlorophyll', 'nitrate', 'phosphate',
#     'silicate', 'dissolvedMolecularOxygen', 'nppv', 'dissolvedIron',
#     'spCO2', 'ph', 'phytoplanktonExpressedAsCarbon'
# ]
#
# region_folders = glob.glob(os.path.join(base_path, 'region=*'))
#
# for region_path in region_folders:
#     region_name = os.path.basename(region_path).split('=')[1]
#     print(f"\nAnalyzing region: {region_name}")
#
#     df = pd.read_parquet(region_path)
#
#     print("Missing values:")
#     print(df[numeric_cols].isnull().sum())
#
#     print("\nDescriptive statistics:")
#     print(df[numeric_cols].describe())
#
#     print("\nOutlier detection (values beyond ±3 std):")
#     for col in numeric_cols:
#         if df[col].dtype in ['float64', 'float32']:
#             mean = df[col].mean()
#             std = df[col].std()
#             outliers = df[(df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)]
#             print(f"{col}: {len(outliers)} outliers")
#
# print("\nAll regions analyzed.")


import pandas as pd
import glob
import os
import numpy as np
from scipy.spatial import cKDTree

print("Starting data cleaning and saving by region...")

base_path = 'Data/GMB/cleaned_marine_dataset_by_region/'
save_path = 'Data/GMB/cleaned_marine_dataset_final/'

essential_cols = [
    'year', 'month', 'day', 'latitude', 'longitude', 'depth'
]

region_cols_to_drop = [
    'country', 'countryCode', 'waterBody', 'eventDate', 'continent', 'locality'
]

value_cols = [
    'temperature', 'salinity', 'ugo', 'vgo', 'waveVelocity', 'chlorophyll',
    'nitrate', 'phosphate', 'silicate', 'dissolvedMolecularOxygen', 'nppv',
    'dissolvedIron', 'spCO2', 'ph', 'phytoplanktonExpressedAsCarbon'
]

region_suffix_map = {
    'N_W1': 'n90.0_s0.0_w-180.0_e-90.0',
    'N_W2': 'n90.0_s0.0_w-90.0_e0.0',
    'N_E1': 'n90.0_s0.0_w0.0_e90.0',
    'N_E2': 'n90.0_s0.0_w90.0_e180.0',
    'S_W1': 'n0.0_s-90.0_w-180.0_e-90.0',
    'S_W2': 'n0.0_s-90.0_w-90.0_e0.0',
    'S_E1': 'n0.0_s-90.0_w0.0_e90.0',
    'S_E2': 'n0.0_s-90.0_w90.0_e180.0'
}

region_folders = glob.glob(os.path.join(base_path, 'region=*'))

for region_path in region_folders:
    region_name = os.path.basename(region_path).split('=')[1]
    if region_name not in region_suffix_map:
        print(f"Skipping region: {region_name}")
        continue

    print(f"\nCleaning region: {region_name}")

    df = pd.read_parquet(region_path)

    print("Step 1: Drop region-related and unused columns")
    df = df.drop(columns=[col for col in region_cols_to_drop if col in df.columns], errors='ignore')

    print("Step 2: Drop records without latitude or longitude")
    df = df.dropna(subset=['latitude', 'longitude'])

    print("Step 3: Keep only the latest record for each (lat, lon, depth)")
    df[['year', 'month', 'day']] = df[['year', 'month', 'day']].apply(pd.to_numeric, errors='coerce')
    df['ymd'] = df['year'].astype(str).str.zfill(4) + df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2)
    df['ymd'] = pd.to_numeric(df['ymd'], errors='coerce')
    df = df.sort_values('ymd')
    df = df.dropna(subset=['ymd'])
    df = df.groupby(['latitude', 'longitude', 'depth'], as_index=False).last()
    df = df.drop(columns='ymd')

    print("Step 4: Drop rows with > 2/3 missing in value fields")
    threshold = int(len(value_cols) * (2 / 3))
    df['missing_count'] = df[value_cols].isnull().sum(axis=1)
    df = df[df['missing_count'] <= threshold]
    df = df.drop(columns='missing_count')

    print("Step 5: Fill remaining missing values using spatial average with fallback to median")
    valid_coords = df[['latitude', 'longitude']].dropna().values
    tree = cKDTree(valid_coords)

    for col in value_cols:
        if col not in df.columns:
            continue
        missing_idx = df[df[col].isnull()].index
        if len(missing_idx) == 0:
            continue
        for idx in missing_idx:
            lat = df.at[idx, 'latitude']
            lon = df.at[idx, 'longitude']
            if pd.isna(lat) or pd.isna(lon):
                continue
            dist, ind = tree.query([lat, lon], k=50, distance_upper_bound=1.0)
            neighbors = df.iloc[ind[np.isfinite(dist)]] if np.any(np.isfinite(dist)) else pd.DataFrame()
            local_values = neighbors[col].dropna()
            if not local_values.empty:
                df.at[idx, col] = local_values.mean()
            else:
                df.at[idx, col] = df[col].median()

    print(f"Saving cleaned region {region_name}, shape: {df.shape}")
    os.makedirs(save_path, exist_ok=True)
    filename = f"{region_suffix_map[region_name]}.parquet"
    df.to_parquet(os.path.join(save_path, filename), index=False)

print("\nAll regions cleaned and saved.")