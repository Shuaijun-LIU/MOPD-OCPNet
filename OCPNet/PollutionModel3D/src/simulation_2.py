import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from data_select import query_data_by_geo_range
from model import PollutionModel3D

def aggregate_data_by_grid(data, grid_size=0.1):
    """
    按经纬度网格聚合数据
    grid_size: 网格大小（度）
    """
    # 创建网格边界
    lat_bins = np.arange(data['latitude'].min(), data['latitude'].max() + grid_size, grid_size)
    lon_bins = np.arange(data['longitude'].min(), data['longitude'].max() + grid_size, grid_size)
    
    # 为每个点分配网格ID
    data['lat_grid'] = pd.cut(data['latitude'], bins=lat_bins, labels=False)
    data['lon_grid'] = pd.cut(data['longitude'], bins=lon_bins, labels=False)
    
    # 按网格聚合数据
    aggregated = data.groupby(['lat_grid', 'lon_grid']).agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'depth': 'mean',
        'temperature': 'mean',
        'salinity': 'mean',
        'ugo': 'mean',
        'vgo': 'mean',
        'waveVelocity': 'mean',
        'chlorophyll': 'mean',
        'nitrate': 'mean',
        'phosphate': 'mean',
        'silicate': 'mean',
        'dissolvedMolecularOxygen': 'mean',
        'nppv': 'mean',
        'dissolvedIron': 'mean',
        'spCO2': 'mean',
        'ph': 'mean',
        'phytoplanktonExpressedAsCarbon': 'mean'
    }).reset_index()
    
    return aggregated

def find_pollution_sources(data, pollutant, max_sources=5, min_sources=1, initial_threshold=2.0):
    """
    找出污染物浓度显著高于平均值的点作为污染源
    pollutant: 污染物名称
    max_sources: 最大污染源数量
    min_sources: 最小污染源数量
    initial_threshold: 初始浓度阈值（相对于平均值的倍数）
    """
    # 计算平均浓度
    mean_concentration = data[pollutant].mean()
    
    # 动态调整阈值，直到找到合适数量的污染源
    threshold = initial_threshold
    while True:
        # 找出浓度显著高于平均值的点
        high_concentration_points = data[data[pollutant] > mean_concentration * threshold]
        
        # 按浓度排序
        sorted_sources = high_concentration_points.sort_values(by=pollutant, ascending=False)
        
        # 如果找到的点太多，只取前max_sources个
        if len(sorted_sources) > max_sources:
            sorted_sources = sorted_sources.head(max_sources)
            print(f"找到 {len(sorted_sources)} 个{pollutant}污染源（超过最大限制{max_sources}个，只取浓度最高的{max_sources}个）")
            break
        
        # 如果找到的点太少，降低阈值
        if len(sorted_sources) < min_sources:
            if threshold <= 1.1:  # 如果阈值已经很低了，就使用所有点
                sorted_sources = data.sort_values(by=pollutant, ascending=False).head(min_sources)
                print(f"找到 {len(sorted_sources)} 个{pollutant}污染源（阈值已降至最低，使用浓度最高的{min_sources}个点）")
                break
            threshold *= 0.9  # 降低阈值
            continue
        
        print(f"找到 {len(sorted_sources)} 个{pollutant}污染源，浓度阈值为平均值的{threshold:.1f}倍")
        break
    
    # 计算排放速率（与浓度成正比）
    sorted_sources['emission_rate'] = sorted_sources[pollutant] / mean_concentration
    
    return sorted_sources

def setup_pollution_sources(model, data):
    """
    设置污染源
    """
    # 设置NH4污染源
    nh4_sources = find_pollution_sources(data, 'nitrate', max_sources=5, min_sources=1, initial_threshold=2.0)
    for _, source in nh4_sources.iterrows():
        model.source_sink.add_point_source(
            pollutant="NH4",
            position=(source['longitude'], source['latitude'], source['depth']),
            emission_rate=source['emission_rate'] * 0.1,  # 使用硝酸盐作为参考，乘以系数
            time_function=lambda t: np.sin(2*np.pi*t/86400)  # 日变化
        )
    
    # 设置PO4污染源
    po4_sources = find_pollution_sources(data, 'phosphate', max_sources=5, min_sources=1, initial_threshold=2.0)
    for _, source in po4_sources.iterrows():
        model.source_sink.add_point_source(
            pollutant="PO4",
            position=(source['longitude'], source['latitude'], source['depth']),
            emission_rate=source['emission_rate'],
            time_function=lambda t: np.sin(2*np.pi*t/86400)
        )
    
    # 设置Hg污染源
    hg_sources = find_pollution_sources(data, 'dissolvedIron', max_sources=5, min_sources=1, initial_threshold=2.0)
    for _, source in hg_sources.iterrows():
        model.source_sink.add_point_source(
            pollutant="Hg",
            position=(source['longitude'], source['latitude'], source['depth']),
            emission_rate=source['emission_rate'] * 1e-6,  # 使用溶解铁作为参考，乘以系数
            time_function=lambda t: np.sin(2*np.pi*t/86400)
        )

def convert_units(data):
    """
    Convert various environmental parameters to appropriate units
    """
    # Convert temperature from Celsius to Kelvin
    data['temperature'] = data['temperature'] + 273.15
    
    # Convert salinity from PSU to dimensionless
    data['salinity'] = data['salinity'] / 1000
    
    # Convert velocity from m/s to m/s (no conversion needed)
    data['ugo'] = data['ugo']
    data['vgo'] = data['vgo']
    
    # Convert wave velocity from m/s to m/s (no conversion needed)
    data['waveVelocity'] = data['waveVelocity']
    
    # Convert chlorophyll from mg/m³ to mg/L
    data['chlorophyll'] = data['chlorophyll'] / 1000
    
    # Convert nitrate from μmol/L to mg/L
    data['nitrate'] = data['nitrate'] * 14.0067 / 1000
    
    # Convert phosphate from μmol/L to mg/L
    data['phosphate'] = data['phosphate'] * 30.9738 / 1000
    
    # Convert silicate from μmol/L to mg/L
    data['silicate'] = data['silicate'] * 28.0855 / 1000
    
    # Convert dissolved oxygen from μmol/L to mg/L
    data['dissolvedMolecularOxygen'] = data['dissolvedMolecularOxygen'] * 31.9988 / 1000
    
    # Convert NPP from mg C/m³/d to kg/m³/s
    data['nppv'] = data['nppv'] * 0.001 / 86400
    
    # Convert dissolved iron from μmol/L to mg/L
    data['dissolvedIron'] = data['dissolvedIron'] * 55.845 / 1000
    
    # Convert CO2 partial pressure from μatm to atm
    data['spCO2'] = data['spCO2'] / 1e6
    
    # Convert pH (no conversion needed)
    data['ph'] = data['ph']
    
    # Convert phytoplankton from mg C/m³ to kg/m³
    data['phytoplanktonExpressedAsCarbon'] = data['phytoplanktonExpressedAsCarbon'] * 0.001
    
    return data

def calculate_time_difference(data):
    """
    Calculate time difference from observation date to current date
    """
    current_date = datetime.now()
    data['time_diff'] = data.apply(lambda row: 
        (current_date - datetime(int(row['year']), int(row['month']), int(row['day']))).total_seconds(), 
        axis=1)
    return data

def setup_model(data):
    """
    设置污染模型参数
    """
    # 按经纬度网格聚合数据
    aggregated_data = aggregate_data_by_grid(data, grid_size=0.1)
    print(f"数据聚合后剩余 {len(aggregated_data)} 个网格点")
    
    # 确定模型域大小（使用经纬度范围）
    lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
    lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
    depth_max = data['depth'].max()
    
    # 计算网格分辨率（增加分辨率）
    nx, ny, nz = 100, 100, 50  # 增加网格分辨率
    
    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent / 'output' / 'main_output'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 初始化模型，使用更小的时间步长
    model = PollutionModel3D(
        domain_size=(lon_max - lon_min, lat_max - lat_min, depth_max),
        grid_resolution=(nx, ny, nz),
        time_step=0.1,  # 使用更小的时间步长
        output_dir=output_dir
    )
    
    # 设置流速场
    u = np.zeros((nx, ny, nz))
    v = np.zeros((nx, ny, nz))
    w = np.zeros((nx, ny, nz))
    
    # 使用聚合后的数据设置流速场，保持真实流速
    for _, point in aggregated_data.iterrows():
        # 将经纬度转换为网格坐标
        i = int((point['longitude'] - lon_min) / (lon_max - lon_min) * (nx - 1))
        j = int((point['latitude'] - lat_min) / (lat_max - lat_min) * (ny - 1))
        k = int(point['depth'] / depth_max * (nz - 1))
        
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            u[i,j,k] = point['ugo']  # 保持真实流速
            v[i,j,k] = point['vgo']  # 保持真实流速
            w[i,j,k] = 0.0  # 假设垂直速度为0
    
    model.set_velocity_field(u, v, w)
    
    # 设置环境场
    environmental_fields = {
        'temperature': np.ones((nx, ny, nz)) * aggregated_data['temperature'].mean(),
        'salinity': np.ones((nx, ny, nz)) * aggregated_data['salinity'].mean(),
        'wave_velocity': np.ones((nx, ny, nz)) * aggregated_data['waveVelocity'].mean(),
        'pH': np.ones((nx, ny, nz)) * aggregated_data['ph'].mean(),
        'DO': np.ones((nx, ny, nz)) * aggregated_data['dissolvedMolecularOxygen'].mean(),
        'light_intensity': np.ones((nx, ny, nz)) * 1000.0,  # 使用固定值
        'chlorophyll': np.ones((nx, ny, nz)) * aggregated_data['chlorophyll'].mean(),
        'nitrate': np.ones((nx, ny, nz)) * aggregated_data['nitrate'].mean(),
        'phosphate': np.ones((nx, ny, nz)) * aggregated_data['phosphate'].mean(),
        'silicate': np.ones((nx, ny, nz)) * aggregated_data['silicate'].mean(),
        'nppv': np.ones((nx, ny, nz)) * aggregated_data['nppv'].mean(),
        'dissolved_iron': np.ones((nx, ny, nz)) * aggregated_data['dissolvedIron'].mean(),
        'spco2': np.ones((nx, ny, nz)) * aggregated_data['spCO2'].mean(),
        'phytoplankton': np.ones((nx, ny, nz)) * aggregated_data['phytoplanktonExpressedAsCarbon'].mean()
    }
    
    for name, value in environmental_fields.items():
        model.set_environmental_field(name, value)
    
    # 添加污染物
    model.add_pollutant(
        name="NH4",
        initial_concentration=np.zeros((nx, ny, nz)),  # 初始浓度设为0，因为我们将使用污染源
        molecular_weight=18.0,
        decay_rate=1e-6,
        diffusion_coefficient=np.ones((nx, ny, nz)) * 1e-9  # 改为数组形式
    )
    
    model.add_pollutant(
        name="PO4",
        initial_concentration=np.zeros((nx, ny, nz)),
        molecular_weight=95.0,
        decay_rate=5e-7,
        diffusion_coefficient=np.ones((nx, ny, nz)) * 8e-10  # 改为数组形式
    )
    
    model.add_pollutant(
        name="Hg",
        initial_concentration=np.zeros((nx, ny, nz)),
        molecular_weight=200.6,
        decay_rate=1e-7,
        diffusion_coefficient=np.ones((nx, ny, nz)) * 5e-10  # 改为数组形式
    )
    
    # 设置污染源
    setup_pollution_sources(model, aggregated_data)
    
    # 设置边界条件
    model.set_boundary_condition(
        type="dirichlet",
        field="NH4",
        boundary="bottom",
        value=0.0
    )
    
    model.set_boundary_condition(
        type="neumann",
        field="PO4",
        boundary="top",
        gradient=0.0
    )
    
    model.set_boundary_condition(
        type="periodic",
        field="velocity",
        boundary="x",
        axis="x"
    )
    
    # 设置输出参数
    model.set_output_parameters(
        output_fields=["NH4", "PO4", "Hg"],
        output_interval=3600.0,
        visualization_fields=["NH4", "PO4", "Hg"],
        visualization_interval=7200.0,
        statistics_fields=["NH4", "PO4", "Hg"],
        statistics_interval=3600.0
    )
    
    return model

def predict_current_distribution(data):
    """
    Run model and predict current pollutant distribution
    """
    # Convert units
    data = convert_units(data)
    
    # Calculate time differences
    data = calculate_time_difference(data)
    
    # Set up model
    model = setup_model(data)
    
    # Run model for each time difference
    for time_diff in data['time_diff'].unique():
        model.run(
            end_time=time_diff,
            progress_interval=86400 * 30  # Report progress every month
        )
    
    # Get current concentrations using get_field method
    nh4_conc = model.get_field('NH4')
    po4_conc = model.get_field('PO4')
    hg_conc = model.get_field('Hg')
    
    # Analyze results
    print("\nCurrent Pollutant Distribution Analysis:")
    print(f"NH4 average concentration: {np.mean(nh4_conc):.2e} mg/L")
    print(f"PO4 average concentration: {np.mean(po4_conc):.2e} mg/L")
    print(f"Hg average concentration: {np.mean(hg_conc):.2e} mg/L")
    
    # Calculate areas exceeding threshold (e.g., 0.1 mg/L)
    threshold = 0.1
    nh4_above = np.sum(nh4_conc > threshold) / nh4_conc.size * 100
    po4_above = np.sum(po4_conc > threshold) / po4_conc.size * 100
    hg_above = np.sum(hg_conc > threshold) / hg_conc.size * 100
    
    print(f"\nPercentage of area above {threshold} mg/L:")
    print(f"NH4: {nh4_above:.2f}%")
    print(f"PO4: {po4_above:.2f}%")
    print(f"Hg: {hg_above:.2f}%")
    
    return nh4_conc, po4_conc, hg_conc

def main():
    # Set data path
    workspace_root = Path(__file__).parent.parent.parent.parent
    data_folder = workspace_root / 'MOPD_pipeline' / 'Data' / 'GMB' / 'cleaned_marine_dataset_final'
    
    print(f"Searching for data in: {data_folder}")
    
    # Load data
    data = query_data_by_geo_range(
        lat_min=30,
        lat_max=45,
        lon_min=130,
        lon_max=145,
        data_folder=str(data_folder)
    )
    
    if data is None or len(data) == 0:
        print("Error: No data found in the specified range")
        return
        
    # Check data fields
    required_fields = ['temperature', 'salinity', 'nitrate', 'phosphate', 'silicate', 
                      'dissolvedIron', 'chlorophyll', 'phytoplanktonExpressedAsCarbon', 'spCO2']
    missing_fields = [field for field in required_fields if field not in data.columns]
    if missing_fields:
        print(f"Error: Missing required fields: {missing_fields}")
        print("Available fields:", list(data.columns))
        return
        
    # Predict current pollutant distribution
    nh4_conc, po4_conc, hg_conc = predict_current_distribution(data)
    
    # Analyze results
    print("\nPrediction Results Analysis:")
    print(f"NH4 average concentration: {np.mean(nh4_conc):.2e} kg/m³")
    print(f"PO4 average concentration: {np.mean(po4_conc):.2e} kg/m³")
    print(f"Hg average concentration: {np.mean(hg_conc):.2e} kg/m³")

if __name__ == '__main__':
    main() 