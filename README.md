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




GEBCO 2025 https://www.gebco.net/data_and_products/gridded_bathymetry_data/ ， 
The terrain data used in this step is based on **GEBCO's gridded bathymetric dataset**, the **[GEBCO_2025 Grid](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)**. 
- **DOI**: 10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f  

GOPAF https://marine.copernicus.eu/，
- **Full Name**: Global Ocean Physics Analysis and Forecast
- **Abbreviation**: GOPAF
- **Provider**: Copernicus Marine Service (CMEMS)
- **DOI**: 10.48670/moi-00016

GMB # Global Marine Biodiversity Dataset with Environmental Elements 
**Data Producer:** Dou Fangkun  
**Dataset Index Number:** DATA2024000001  
**DOI:** [10.12157/IOC.AS.20240110.001](http://dx.doi.org/10.12157/IOC.AS.20240110.001)  
**CSTR:** [CSTR:33685.11.IOCAS.20240110.002](https://datapid.cn/CSTR:33685.11.IOCAS.20240110.002)  
**Data Sharing Mode:** Fully Open  
**Open License:** Applicable  

NOAA NCEI Microplastics ，用于验证的也是NOAA NCEI的数据
# Marine Microplastics Database – National Centers for Environmental Information (NCEI)
**Organization:** NOAA National Centers for Environmental Information (NCEI)  
**Product Name:** Marine Microplastics  
**Access Link:** [NCEI Marine Microplastics](https://www.ncei.noaa.gov/products/microplastics)  
**Data Formats:** CSV, JSON, GeoJSON  
**Availability:** Public and Open Access  
**Last Update:** Sep 2025 (Product Flier)







## MOPD (Marine Oceanic Pollution and Dynamics Dataset)
### Pipeline
#### Step 1: Get Terrain Data and Visualize
1. **Dataset Description: Global Ocean & Land Terrain Models**  
   The terrain data used in this step is based on **GEBCO's gridded bathymetric dataset**, the **[GEBCO_2024 Grid](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)**. This dataset provides a global terrain model for both ocean and land. 
   - **Full Name**: General Bathymetric Chart of the Oceans 2024 Grid
   - **Resolution**: 15 arc-second interval grid (approximately 450m at the equator)
   - **Elevation Units**: Meters  
   - **Coverage**: Global ocean and land terrain
   - **Data Format**: GeoTIFF files (8 regional files)
   - **Data Size**: ~15 GB total (8 regional files)
   - **Accompanying Data**: Includes a Type Identifier (TID) Grid that specifies the source data types used for the GEBCO_2024 Grid
   - **Access URL**: [https://www.gebco.net/data_and_products/gridded_bathymetry_data/](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)
   - **DOI**: 10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f  
 
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
   - **Full Name**: Global Ocean Physics Analysis and Forecast
   - **Abbreviation**: GOPAF
   - **Provider**: Copernicus Marine Service (CMEMS)
   - **Time Range**: June 2025 (720 time steps, hourly data)
   - **Spatial Coverage**: 32°N-33°N, 65.5°W-66.5°W (Boston Harbor area)
   - **Depth**: 0.494 meters (surface layer)
   - **Data Format**: NetCDF (CF-1.8 conventions)
   - **Variables**:  
     - **Basic Physical Variables**:  
       - `so`: Sea Water Salinity (PSU)
       - `thetao`: Sea Water Potential Temperature (K)
       - `uo`: Eastward Velocity (m/s)
       - `vo`: Northward Velocity (m/s)
       - `zos`: Sea Surface Height Above Geoid (m)
     - **Detailed Velocity Variables**:  
       - `utide`: Eastward Tidal Velocity (m/s)
       - `utotal`: Total Eastward Velocity (m/s)
       - `vsdx`: Stokes Drift X Velocity (m/s)
       - `vsdy`: Stokes Drift Y Velocity (m/s)
       - `vtide`: Northward Tidal Velocity (m/s)
       - `vtotal`: Total Northward Velocity (m/s)
   - **Access URL**: [https://marine.copernicus.eu/](https://marine.copernicus.eu/)
   - **DOI**: 10.48670/moi-00016
   - **References**: [CMEMS GOPAF Product](https://marine.copernicus.eu/access-data/access-to-ocean-data)
    

2. **Fetch and Merge Copernicus Data**  
   - **File:** `GOPAF_Data.py`  
   - **Function:** `fetch_and_merge_copernicus_data`  
  
3. **Complete and Inspect Dataset**  
Fill missing values (`NaN`) using linear and nearest-neighbor interpolation, then verify the structure and attributes for data quality.  

### Additional Datasets

#### Global Marine Biodiversity (GMB)
- **Full Name**: Global Marine Biodiversity Dataset
- **Abbreviation**: GMB
- **Sources**: OBIS (Ocean Biodiversity Information System) + GBIF (Global Biodiversity Information Facility)
- **Data Scale**: 200M+ species occurrence records, 122M sampling points
- **Time Span**: 1600-2025
- **Environmental Variables** (13 key parameters):
  - Temperature, Salinity, Sea Currents (ugo, vgo)
  - Chlorophyll, Nitrate, Phosphate, Silicate
  - Dissolved Molecular Oxygen, Net Primary Productivity
  - Dissolved Iron, Carbon Dioxide Partial Pressure, pH Value
  - Phytoplankton Carbon Concentration
- **Regional Classification**: 8 regions (N_E1, N_E2, N_W1, N_W2, S_E1, S_E2, S_W1, S_W2)
- **Access URLs**: 
  - [OBIS](https://obis.org/)
  - [GBIF](https://www.gbif.org/)

#### NOAA Microplastics Data
- **Full Name**: NOAA National Centers for Environmental Information Microplastics Data
- **Provider**: NOAA National Centers for Environmental Information (NCEI)
- **Data Type**: Global ocean microplastics concentration data
- **Format**: CSV files
- **Fields**:
  - Latitude/Longitude coordinates
  - Density ranges (1-2, 2-40, 40-200)
  - Density classes (Low, Medium, High)
  - Ocean regions (Atlantic Ocean, etc.)
  - Sampling dates
- **Applications**: Water quality monitoring, ecosystem protection, remote sensing validation
- **Access URL**: [NOAA NCEI Microplastics](https://www.ncei.noaa.gov/data/oceans/ncei/microplastics/)

### Dataset Integration
- **Combined Dataset**: `combined_environment.nc`
- **Spatial Resolution**: 240×240 grid points
- **Time Resolution**: 720 hourly time steps
- **Variables**: 9 ocean physics + 1 terrain variable
- **Total Size**: >4.2GB integrated dataset

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
### Support and References

#### Data Sources
- **E.U. Copernicus Marine Service Information (2024)**. Global Ocean Physics Analysis and Forecast (GOPAF). doi:https://doi.org/10.48670/moi-00016
- **GEBCO Compilation Group (2024)**. GEBCO 2024 Grid. doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f
- **NOAA National Centers for Environmental Information**. Microplastics and oceanographic data. Available at: https://www.ncei.noaa.gov/data/oceans/ncei/microplastics/
- **OBIS/GBIF**. Global Marine Biodiversity Information System. Available at: https://obis.org/ and https://www.gbif.org/

#### Dataset URLs
- **GEBCO 2024**: https://www.gebco.net/data_and_products/gridded_bathymetry_data/
- **Copernicus Marine**: https://marine.copernicus.eu/
- **OBIS**: https://obis.org/
- **GBIF**: https://www.gbif.org/
- **NOAA NCEI**: https://www.ncei.noaa.gov/

#### Contact
For questions, support, or collaboration opportunities, please contact the project maintainers.

#### License
This project is licensed under the MIT License - see the LICENSE file for details.

