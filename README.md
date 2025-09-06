# A Large-Scale Oceanographic Dataset and Prediction Framework for Ocean Currents and Pollution Dispersion
## Overview
MOPD-OCPNet consists of a large Marine Oceanic Pollution and Dynamics Dataset (MOPD) that integrates topography, currents, and pollution, and a machine learning model (OCPNet) for accurate prediction.

### Required Packages
Ensure the following Python packages are installed to execute the steps:
```bash
pip install numpy scipy pandas rasterio xarray matplotlib pillow copernicusmarine
```

### File Structure
```plaintext
env_config/
│
├── Data/
│   ├── Combined/
│   │   ├── combined_environment.nc
│   │   ├── large/
│   ├── GOPAF/
│   ├── gebco_2024_sub_ice_topo_geotiff/
│   ├── Visualizations/
│
├── other_code/
│   ├── buildmap.ipynb
│   ├── gopaf_original.ipynb
│   ├── GeoTIFF_original.ipynb
│   ├── geotifftest2.ipynb
│
├── main.py # Calling the modules
├── Combine.py # [Data Integration Module] Fusion of terrain (GeoTIFF) and current data (NetCDF), harmonization of grids by RBF interpolation, interpolation of current data and storage as NetCDF (combined_environment.nc).
├── GOPAF_Data.py # [Current Data-processing Module] Acquisition, merging and missing value interpolation of Copernicus marine data
├── GOPAF_visual.py  
├── GeoTIFF_Data.py # [Terrain Data-processing Module] Filter, crop, filter and convert GeoTIFF terrain data. 
├── GeoTIFF_visual.py 
├── get_station_id.py # [NOAA Site Acquisition Module] Finds the closest site ID from NOAA to the specified location
├── get_UW_data.py  # [NOAA Data Acquisition Module] Obtain meteorological and oceanographic data from the NOAA Tides and Currents API for specified site ID
│
├── README.md
├── requirements.txt
```
---

## MOPD (Marine Oceanic Pollution and Dynamics Dataset)
### Pipeline
#### Step 1: Get Terrain Data and Visualize
1. **Dataset Description: Global Ocean & Land Terrain Models**  
   The terrain data used in this step is based on **GEBCO’s gridded bathymetric dataset**, the **[GEBCO_2024 Grid](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)**. This dataset provides a global terrain model for both ocean and land. 
   - **Resolution**: 15 arc-second interval grid  
   - **Elevation Units**: Meters  
   - **Accompanying Data**: Includes a Type Identifier (TID) Grid that specifies the source data types used for the GEBCO_2024 Grid.  
 
2. **Crop Terrain Data**  
   - **File:** `GeoTIFF_Data.py`  
   - **Function:** `Get_GeoTIFF_Data`  
   Extract terrain data based on specified latitude, longitude, and elevation ranges.

3. **Visualize Elevation Map**  
   - **File:** `GeoTIFF_visual.py`  
   - **Function:** `visualize_elevation_map`  
   Visualize the cropped terrain data using predefined view angles.  

4. **Analyze GeoTIFF File (Optional)**  
   - **File:** `GeoTIFF_Data.py`  
   - **Function:** `analyze_geotiff`  
   Analyze terrain data for shape, min/max values, mean, and data type.

5. **Convert GeoTIFF to CSV (Optional)**  
   - **File:** `GeoTIFF_Data.py`  
   - **Function:** `tif_to_csv`  
   Convert the terrain data from `.tif` to `.csv` for further processing.

---
### Step 2: Get Currents Forecast Data (GOPAF)
1. **Dataset Description:**  
   The marine data is sourced from the *Global Ocean Physics Analysis and Forecast (GOPAF)* datasets provided by CMEMS. We fetch two separate datasets from GOPAF and merge them into a single NetCDF file named `combined_gopaf_data.nc`.
   - **Variables**:  
     - **Basic Physical Variables**:  
       - `so`: Sea Water Salinity  
       - `thetao`: Sea Water Potential Temperature  
       - `uo`: Eastward Velocity  
       - `vo`: Northward Velocity  
       - `zos`: Sea Surface Height Above Geoid  
     - **Detailed Velocity Variables**:  
       - `utide`: Eastward Tidal Velocity  
       - `utotal`: Total Eastward Velocity  
       - `vsdx`: Stokes Drift X Velocity  
       - `vsdy`: Stokes Drift Y Velocity  
       - `vtide`: Northward Tidal Velocity  
       - `vtotal`: Total Northward Velocity  
   - **Attributes**:  
     - **Conventions**: CF-1.8  
     - **Area**: Global  
     - **Credit**: E.U. Copernicus Marine Service Information (CMEMS)  
     - **Producer**: CMEMS - Global Monitoring and Forecasting Centre  
     - **References**: [CMEMS Reference](http://marine.copernicus.eu)
    

2. **Fetch and Merge Copernicus Data**  
   - **File:** `GOPAF_Data.py`  
   - **Function:** `fetch_and_merge_copernicus_data`  
  
3. **Complete and Inspect Dataset**  
Fill missing values (`NaN`) using linear and nearest-neighbor interpolation, then verify the structure and attributes for data quality.  

### Dataset



---

### Step 3: Combine Terrain and Currents Data

#### 3.1 Build Small Environment  
- **Interpolate and Merge Data**  
  - **File:** `Combine.py`  
  - **Function:** `interpolate_and_merge`  
  Combine terrain (`GeoTIFF`) and currents (`GOPAF`) data into a unified dataset.

- **Visualize Combined Data**   
  - **Function:** `visualize_combined_data`  
  Visualize the combined dataset by selecting specific time and depth indexes.

#### 3.2 Build Large Environment  
- **Interpolate Terrain Data**
  - **Function:** `interpolate_geotiff`  
  Increase the resolution of the terrain data for higher accuracy.

- **Interpolate and Merge**
  - **Function:** `interpolate_and_merge`  
  Merge the high-resolution terrain data with currents data.

---

### (Optional) Fetch Currents Ture Data from NOAA

1. **Get Station ID**  
   - **File:** `get_station_id.py`  
   - **Function:** `get_nearest_station_id`  
   Identify the nearest NOAA station for the specified geographic coordinates.

2. **Fetch NOAA Data**  
   - **File:** `get_UW_data.py`  
   - **Function:** `fetch_noaa_data`  
   Retrieve the current data using the station ID.

3. **Find and Use Current Station**  
   - **File:** `get_station_id.py`  
   - **Function:** `find_nearest_current_station`  
   Determine the closest current station and fetch the corresponding data.

---

## OCPNet (Ocean Current and Pollution Prediction Network)


---
### Support
E.U. Copernicus Marine Service Information (2024). Global Ocean Physics Analysis and Forecast (GOPAF). doi:https://doi.org/10.48670/moi-00016. 
GEBCO Compilation Group (2024). GEBCO 2024 Grid. doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f.

