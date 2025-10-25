# Pipeline: Marine Oceanographic Pollution and Dynamics Data Processing Pipeline

## Overview

The **MOPD Pipeline** is a comprehensive data processing framework designed to integrate, process, and analyze multi-source marine oceanographic datasets for pollution prediction and ocean current modeling. This pipeline serves as the foundation for the larger **OCPNet (Ocean Current and Pollution Prediction Network)** project, providing clean, standardized, and spatially-temporally aligned datasets.

## 🎯 Key Achievements

### 1. **Multi-Source Data Integration**
- **GEBCO 2025 Terrain Data**: Global bathymetric data processing and regional extraction
- **GOPAF Ocean Current Data**: Copernicus Marine Service ocean physics analysis and forecast data
- **GMB Biodiversity Data**: Global marine biodiversity records with 13 environmental parameters
- **NOAA Microplastics Data**: Marine microplastics concentration and density data

### 2. **Advanced Data Processing Capabilities**
- **Spatial Interpolation**: RBF and griddata interpolation for data alignment
- **Temporal Synchronization**: Time-series data harmonization across different sources
- **Quality Control**: Missing value imputation, outlier detection, and data validation
- **Regional Partitioning**: Intelligent geographic data segmentation for efficient processing

### 3. **Comprehensive Data Pipeline**
- **Automated Data Fetching**: Direct API integration with Copernicus Marine Service
- **Multi-Format Support**: NetCDF, GeoTIFF, CSV, and Parquet data handling
- **Scalable Processing**: Dask-based processing for large-scale datasets
- **Visualization Tools**: 2D/3D terrain mapping and data visualization

## 📊 Supported Datasets

### 1. **GEBCO 2025 Terrain Data**
- **Source**: General Bathymetric Chart of the Oceans 2025 Grid
- **Coverage**: Global ocean and land terrain (8 regional GeoTIFF files)
- **Resolution**: 15 arc-second interval grid (~450m at equator)
- **Data Size**: ~15 GB total
- **Processing**: Regional extraction, elevation filtering, 3D visualization

### 2. **GOPAF Ocean Current Data**
- **Source**: Copernicus Marine Service - Global Ocean Physics Analysis and Forecast
- **Variables**: 9 ocean physics parameters (salinity, temperature, currents, sea level, tides)
- **Temporal Coverage**: 2025年6月 (720 hourly time steps)
- **Spatial Coverage**: Configurable regions (Boston Harbor, Japan, US-NE)
- **Processing**: Data merging, interpolation, quality control

### 3. **GMB Biodiversity Data**
- **Source**: OBIS + GBIF (Ocean Biodiversity Information System + Global Biodiversity Information Facility)
- **Scale**: 200M+ species occurrence records, 122M sampling points
- **Temporal Span**: 1600-2025
- **Data Size**: 35.2 GB raw data
- **Environmental Variables**: 13 key parameters (temperature, salinity, nutrients, etc.)
- **Processing**: Regional partitioning, data cleaning, spatial interpolation

### 4. **NOAA Microplastics Data**
- **Source**: NOAA National Centers for Environmental Information
- **Data Type**: Marine microplastics concentration and density classification
- **Coverage**: Global
- **Processing**: Data cleaning, deduplication, temporal sorting

## 🛠️ Pipeline Components

### **Core Processing Modules**

#### 1. **GeoTIFF_Data.py** - Terrain Data Processing
```python
# Key Functions:
- Get_GeoTIFF_Data(): Regional terrain data extraction
- analyze_geotiff(): Data quality analysis
- visualize_elevation_map(): 3D terrain visualization
- tif_to_csv(): Format conversion
```

#### 2. **GOPAF_Data.py** - Ocean Current Data Processing
```python
# Key Functions:
- fetch_and_merge_copernicus_data(): API data fetching and merging
- get_shapes(): Data structure analysis
- visualize_each_variable(): Multi-variable visualization
- create_combined_image(): Composite visualization
```

#### 3. **GMB_Data.py** - Biodiversity Data Processing
```python
# Key Functions:
- Regional data partitioning (8 global regions)
- Data cleaning and quality control
- Spatial interpolation for missing values
- Temporal data deduplication
```

#### 4. **Combine.py** - Data Integration
```python
# Key Functions:
- interpolate_and_merge(): Multi-source data integration
- visualize_combined_data(): Integrated data visualization
- interpolate_geotiff(): High-resolution terrain interpolation
```

#### 5. **Microplastics_Data.py** - Microplastics Processing
```python
# Key Functions:
- Data cleaning and deduplication
- Temporal sorting and latest record selection
- Format standardization
```

#### 6. **NOAA Data Integration** (Optional)
- **get_station_id.py**: Nearest NOAA station identification
- **get_UW_data.py**: Real-time NOAA data fetching

## 📈 Data Processing Workflow

### **Step 1: Terrain Data Extraction**
1. Load GEBCO 2025 GeoTIFF files
2. Extract regional terrain data based on coordinates
3. Apply elevation filtering
4. Generate 3D visualizations from multiple viewing angles
5. Save processed terrain data

### **Step 2: Ocean Current Data Fetching**
1. Authenticate with Copernicus Marine Service
2. Fetch basic physics data (salinity, temperature, currents, sea level)
3. Fetch detailed current data (tidal velocities, total velocities)
4. Merge datasets and perform quality control
5. Apply spatial and temporal interpolation

### **Step 3: Data Integration**
1. Interpolate ocean data to terrain grid resolution
2. Merge terrain and ocean physics data
3. Create unified coordinate system
4. Generate combined visualization
5. Export integrated dataset (NetCDF format)

### **Step 4: Biodiversity Data Processing**
1. Load and clean GMB dataset
2. Partition data into 8 global regions
3. Apply spatial interpolation for missing values
4. Perform temporal deduplication
5. Export cleaned regional datasets

## 🎨 Visualization Capabilities

### **Terrain Visualization**
- 2D elevation maps with terrain colormap
- 3D surface plots from multiple viewing angles
- Interactive elevation analysis

### **Ocean Data Visualization**
- Multi-variable time series plots
- Spatial distribution maps
- Combined variable visualization

### **Data Quality Visualization**
- Missing value analysis
- Outlier detection plots
- Data distribution histograms

## 📁 Output Structure

```
Data/
├── Combined/
│   └── combined_environment.nc          # Integrated dataset
├── GEBCO_2025/
│   └── filtered_data.tif               # Processed terrain data
├── GOPAF/
│   ├── cmems_basic_phy_output_data.nc  # Basic physics data
│   ├── cmems_detailed_uv_output_data.nc # Detailed current data
│   └── combined_gopaf_data.nc          # Merged ocean data
├── GMB/
│   ├── cleaned_marine_dataset_by_region/ # Regional partitions
│   └── cleaned_marine_dataset_final/    # Final cleaned data
├── NOAA_Microplastics/
│   └── Marine_Microplastics_cleaned.csv # Cleaned microplastics data
└── Visualizations/                      # Generated plots and maps
```

## 🚀 Usage Examples

### **Basic Terrain Processing**
```python
from GeoTIFF_Data import Get_GeoTIFF_Data, visualize_elevation_map

# Extract terrain data for Boston Harbor
cropped_data = Get_GeoTIFF_Data(
    lat_min=32.0, lat_max=33.0,
    lon_min=-66.5, lon_max=-65.5,
    elev_min=-10000, elev_max=0,
    save_path="./output"
)

# Generate 3D visualizations
view_angles = [(30, 45), (60, 45), (90, 45)]
visualize_elevation_map(cropped_data, view_angles, 32.0, 33.0, -66.5, -65.5, "./output")
```

### **Ocean Current Data Fetching**
```python
from GOPAF_Data import fetch_and_merge_copernicus_data

# Fetch and merge ocean data
fetch_and_merge_copernicus_data(
    username='your_username',
    password='your_password',
    minimum_longitude=32.0, maximum_longitude=33.0,
    minimum_latitude=-66.5, maximum_latitude=-65.5,
    start_datetime="2025-01-01T00:00:00",
    end_datetime="2025-06-30T23:00:00",
    minimum_depth=0.4, maximum_depth=0.5,
    output_filename="combined_gopaf_data.nc"
)
```

### **Data Integration**
```python
from Combine import interpolate_and_merge, visualize_combined_data

# Integrate terrain and ocean data
interpolate_and_merge(
    './output/filtered_data.tif',
    './Data/GOPAF/combined_gopaf_data.nc',
    './Data/Combined'
)

# Visualize integrated data
visualize_combined_data(
    './Data/Combined/combined_environment.nc',
    time_index=0, depth_index=0
)
```

## 🔧 Technical Requirements

### **Dependencies**
```python
# Core libraries
rasterio>=1.3.0
xarray>=0.20.0
netCDF4>=1.5.0
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0

# Visualization
matplotlib>=3.5.0
mpl_toolkits

# Data processing
dask>=2021.0.0
pyarrow>=5.0.0

# API integration
copernicusmarine>=0.6.0
requests>=2.25.0
```

### **System Requirements**
- **Memory**: Minimum 16GB RAM (recommended 32GB+ for large datasets)
- **Storage**: 50GB+ free space for data processing
- **Network**: Stable internet connection for API data fetching

## 📊 Data Quality Metrics

### **Processing Statistics**
- **Terrain Data**: 15 arc-second resolution, 8 global regions
- **Ocean Data**: 720 time steps, 9 variables, interpolated to terrain grid
- **Biodiversity Data**: 200M+ records processed, 8 regional partitions
- **Integration**: 240×240 spatial grid, unified coordinate system

### **Quality Control**
- **Missing Value Handling**: Spatial interpolation with fallback to median
- **Outlier Detection**: ±3 standard deviation filtering
- **Temporal Consistency**: Latest record selection for duplicate coordinates
- **Spatial Alignment**: RBF interpolation for coordinate system unification

## 🌟 Key Features

### **1. Scalable Processing**
- Dask-based parallel processing for large datasets
- Memory-efficient data handling
- Configurable processing parameters

### **2. Multi-Format Support**
- NetCDF for scientific data
- GeoTIFF for geospatial data
- Parquet for efficient storage
- CSV for compatibility

### **3. Advanced Interpolation**
- RBF (Radial Basis Function) interpolation
- Linear interpolation for temporal data
- Spatial averaging for missing values

### **4. Comprehensive Visualization**
- 2D/3D terrain mapping
- Multi-variable time series
- Interactive data exploration

## 🔗 Integration with OCPNet

The MOPD Pipeline serves as the data foundation for the OCPNet project:

1. **Data Preparation**: Provides clean, standardized datasets
2. **Feature Engineering**: Creates spatial-temporal features for ML models
3. **Model Training**: Supplies training data for LSTM networks
4. **Simulation Support**: Provides environmental data for pollution modeling

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mopd_pipeline_2025,
  title={MOPD Pipeline: Marine Oceanographic Pollution and Dynamics Data Processing Pipeline},
  author={Liu, Shuaijun and Wen, Qifu},
  year={2025},
  url={https://github.com/your-repo/MOPD_pipeline}
}
```

## 📞 Support

For questions, issues, or contributions, please contact:
- **Email**: [your-email@domain.com]
- **GitHub Issues**: [project-repository/issues]

---

**Note**: This pipeline is part of the larger OCPNet project for ocean current and pollution prediction. For the complete system, see the main project repository.
