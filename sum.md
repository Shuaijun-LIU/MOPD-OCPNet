# 项目模块总结

## 项目概述
本项目是一个大规模海洋学数据集和机器学习框架，用于海洋洋流和污染预测。项目包含两个主要模块：**MOPD_pipeline**（海洋污染和动力学数据集管道）和**OCPNet**（海洋洋流预测网络）。

---

## 1. MOPD_pipeline 模块 (版本 2.1.0)

### 功能概述
MOPD_pipeline 是一个综合的海洋数据收集、处理和整合管道，用于构建大规模海洋学数据集。

### 主要功能

#### 1.1 数据源集成
- **GeoTIFF 地形数据**：从 GEBCO 2024 获取海底地形数据
- **GOPAF 海洋预报数据**：从 Copernicus Marine 获取海洋环流预报数据
- **GMB 全球海洋生物多样性数据**：处理海洋生物多样性记录
- **NOAA 微塑料数据**：整合 NOAA 微塑料污染数据

#### 1.2 数据处理功能
- **地形数据处理**：`GeoTIFF_Data.py` - 处理海底地形数据，支持区域裁剪和可视化
- **海洋环流数据**：`GOPAF_Data.py` - 获取海洋流速、温度、盐度等物理参数
- **生物数据清理**：`GMB_Data.py` - 处理全球海洋生物多样性数据，按区域分类
- **微塑料数据**：`Microplastics_Data.py` - 处理 NOAA 微塑料污染数据
- **数据整合**：`Combine.py` - 将地形数据与海洋环流数据进行空间插值和合并

#### 1.3 数据输出
- 生成综合环境数据集 (`combined_environment.nc`)
- 包含变量：盐度(so)、温度(thetao)、流速(uo,vo)、海面高度(zos)、潮流(utide,vtide)等
- 支持多种可视化输出格式

---

## 2. OCPNet 模块 (版本 1.3.0)

### 功能概述
OCPNet 是一个基于机器学习的海洋洋流预测和污染扩散模拟网络。

### 主要功能

#### 2.1 海洋洋流预测
- **LSTM 神经网络**：用于时间序列的海洋洋流预测
- **3D 可视化**：`visual.py` - 提供海洋洋流的3D可视化功能
- **数据统计分析**：对海洋数据进行统计分析和趋势预测

#### 2.2 污染扩散模拟 (PollutionModel3D)
- **三维污染扩散模型**：完整的3D海洋污染模拟框架
- **物理过程**：
  - 平流传输 (Advection)
  - 分子和湍流扩散 (Diffusion)
  - 边界条件处理
- **化学过程**：
  - 多组分化学反应
  - 沉淀-溶解过程
  - 环境响应（温度、pH、溶解氧）
- **生物过程**：
  - 浮游植物吸收
  - 自然降解过程
  - 环境因子影响

#### 2.3 数学模型
- **平流方程**：$\frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} + w\frac{\partial C}{\partial z} = 0$
- **扩散方程**：$\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}(D_x\frac{\partial C}{\partial x}) + \frac{\partial}{\partial y}(D_y\frac{\partial C}{\partial y}) + \frac{\partial}{\partial z}(D_z\frac{\partial C}{\partial z})$
- **化学反应**：$\frac{dC_i}{dt} = \sum_{j=1}^{N} \nu_{ij} r_j$

---

## 3. 详细数据

### 3.1 数据源概览
项目整合了**5大类数据源**，涵盖物理海洋学、生物海洋学、地形学和污染监测等多个领域：

#### 3.1.1 地形数据 (GEBCO 2024)
- **数据来源**：GEBCO (General Bathymetric Chart of the Oceans) 2024
- **数据类型**：海底地形高程数据
- **空间覆盖**：全球海洋（8个GeoTIFF文件）
- **分辨率**：高分辨率海底地形网格
- **用途**：提供海底地形信息，影响海洋环流和污染扩散

#### 3.1.2 海洋环流预报数据 (GOPAF - Copernicus Marine)
- **数据来源**：Copernicus Marine Service - Global Ocean Physics Analysis and Forecast
- **时间范围**：2024年6月全月（720个时间步，每小时数据）
- **空间范围**：32°N-33°N, 65.5°W-66.5°W（波士顿海域）
- **深度**：0.494米（表层）
- **包含变量**：
  - `so`: 海水盐度 (Sea water salinity)
  - `thetao`: 海水潜在温度 (Sea water potential temperature)
  - `uo`: 东向海水流速 (Eastward sea water velocity)
  - `vo`: 北向海水流速 (Northward sea water velocity)
  - `zos`: 海面高度 (Sea surface height above geoid)
  - `utide`: 东向潮流速度 (Eastward tidal velocity)
  - `utotal`: 总东向海水流速 (Total eastward sea water velocity)
  - `vtide`: 北向潮流速度 (Northward tidal velocity)
  - `vtotal`: 总北向海水流速 (Total northward sea water velocity)

#### 3.1.3 全球海洋生物多样性数据 (GMB)
- **数据来源**：OBIS (Ocean Biodiversity Information System) + GBIF (Global Biodiversity Information Facility)
- **数据规模**：超过2亿条物种出现记录，1.22亿个采样点记录
- **时间跨度**：1600-2021年
- **数据量**：28.89 GB原始数据
- **环境变量**（13个关键参数）：
  - 温度 (Temperature)
  - 盐度 (Salinity)
  - 海流 (Sea Currents: ugo, vgo)
  - 叶绿素 (Chlorophyll)
  - 硝酸盐 (Nitrate)
  - 磷酸盐 (Phosphate)
  - 硅酸盐 (Silicate)
  - 溶解分子氧 (Dissolved Molecular Oxygen)
  - 净初级生产力 (Net Primary Productivity)
  - 溶解铁 (Dissolved Iron)
  - 二氧化碳分压 (Carbon Dioxide Partial Pressure)
  - pH值 (pH Value)
  - 浮游植物碳浓度 (Phytoplankton Carbon Concentration)
- **区域分类**：按8个区域分类存储（N_E1, N_E2, N_W1, N_W2, S_E1, S_E2, S_W1, S_W2）

#### 3.1.4 NOAA微塑料污染数据
- **数据来源**：NOAA National Centers for Environmental Information (NCEI)
- **数据类型**：全球海洋微塑料浓度数据
- **数据格式**：CSV格式
- **包含字段**：
  - 纬度/经度坐标
  - 密度范围 (Density Range: 1-2, 2-40, 40-200)
  - 密度等级 (Density Class: Low, Medium, High)
  - 海洋区域 (Oceans: Atlantic Ocean等)
  - 采样日期 (Date)
- **应用**：支持水质监测、生态系统保护和遥感验证

#### 3.1.5 综合环境数据集 (Combined Dataset)
- **数据规模**：720个时间步 × 240×240空间网格 × 1个深度层
- **总数据量**：每个变量约332MB，总计超过3GB
- **整合变量**：
  - 9个海洋物理变量（来自GOPAF）
  - 1个地形变量（来自GEBCO）
- **空间分辨率**：高分辨率网格插值
- **时间分辨率**：小时级数据

### 3.2 数据整合技术
- **空间插值**：使用科学插值方法将不同分辨率的数据统一到240×240网格
- **时间同步**：统一时间坐标系，确保时间序列数据一致性
- **质量控制**：处理缺失值、异常值和数据验证
- **格式标准化**：统一为NetCDF格式，便于科学计算和可视化

### 3.3 数据应用价值
- **海洋环流预测**：提供高精度海洋流速和温度数据
- **污染扩散模拟**：结合地形和环流数据，模拟污染物传播
- **生态系统研究**：整合生物多样性数据，研究海洋生态
- **环境监测**：微塑料数据支持海洋污染监测
- **气候变化研究**：长期时间序列数据支持气候分析

## 4. 项目特色

### 4.1 数据规模
- 时间跨度：2024年6月全月（720个时间步）
- 空间分辨率：240×240网格点
- 数据量：每个变量约332MB，总计超过3GB
- 多源数据：5大类数据源，超过2亿条记录

### 4.2 技术特点
- **多源数据融合**：整合地形、环流、生物、污染等多种数据源
- **高精度插值**：使用科学插值方法确保数据空间一致性
- **3D可视化**：提供丰富的3D可视化功能
- **模块化设计**：各功能模块独立，便于维护和扩展

### 4.3 应用领域
- 海洋环流预测
- 海洋污染扩散模拟
- 海洋环境保护
- 海洋科学研究
- 海洋工程应用

---

## 5. 技术栈

- **数据处理**：xarray, numpy, pandas, rasterio
- **机器学习**：LSTM神经网络
- **可视化**：matplotlib, plotly, cartopy
- **海洋数据**：Copernicus Marine, NOAA, GEBCO
- **科学计算**：scipy, scikit-learn

---

## 6. 项目状态

项目已完成核心功能开发，包括：
- ✅ 多源海洋数据收集和预处理
- ✅ 数据融合和空间插值
- ✅ 3D海洋洋流可视化
- ✅ 污染扩散模拟框架
- ✅ 机器学习预测模型
- ✅ 完整的文档和示例代码

项目为海洋科学研究和环境保护提供了强大的数据基础和预测工具。

---

## 7. 3D污染物扩散模型详细分析 (PollutionModel3D)

### 7.1 模型架构概述
PollutionModel3D 是一个高度模块化的三维海洋污染物扩散模拟框架，采用面向对象设计，集成了物理、化学、生物等多种过程。

#### 7.1.1 核心组件
- **Grid3D**: 三维欧拉网格系统，处理空间几何和坐标计算
- **PollutionField**: 多污染物浓度场管理器，支持动态添加/移除污染物
- **PollutionModel3D**: 主模型类，集成所有功能模块

#### 7.1.2 模块化设计
模型包含**10个专业模块**，每个模块负责特定的物理/化学/生物过程：

### 7.2 物理过程模块

#### 7.2.1 平流模块 (AdvectionModule)
- **功能**：处理污染物随水流传输
- **数值方法**：迎风格式 (Upwind Scheme) 确保数值稳定性
- **输入**：浓度场 + 三维速度场 (u, v, w)
- **数学方程**：$\frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} + w\frac{\partial C}{\partial z} = 0$

#### 7.2.2 扩散模块 (DiffusionModule)
- **功能**：处理分子扩散和湍流扩散
- **环境依赖**：扩散系数随温度、波浪速度、盐度变化
- **数值方法**：中心差分格式
- **数学方程**：$\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}(D_x\frac{\partial C}{\partial x}) + \frac{\partial}{\partial y}(D_y\frac{\partial C}{\partial y}) + \frac{\partial}{\partial z}(D_z\frac{\partial C}{\partial z})$

#### 7.2.3 边界条件模块 (BoundaryConditionsModule)
- **支持类型**：Dirichlet、Neumann、周期性、开放边界
- **功能**：处理各种边界条件，确保数值稳定性

### 7.3 化学过程模块

#### 7.3.1 耦合反应模块 (CouplingReactionModule)
- **功能**：处理多组分化学反应
- **反应类型**：支持复杂化学反应网络
- **环境响应**：反应速率随温度、pH变化
- **数学方程**：$\frac{dC_i}{dt} = \sum_{j=1}^{N} \nu_{ij} r_j$，$r_j = k_j \prod_{i=1}^{M} C_i^{\alpha_{ij}}$

#### 7.3.2 沉淀模块 (PrecipitationModule)
- **功能**：模拟沉淀-溶解过程
- **环境因子**：考虑温度、pH、溶解氧影响
- **应用**：重金属、营养盐等污染物的沉淀过程

### 7.4 生物过程模块

#### 7.4.1 生物吸收模块 (BioUptakeModule)
- **功能**：模拟浮游植物对污染物的吸收
- **动力学模型**：Michaelis-Menten动力学
- **环境因子**：温度、光照强度影响
- **数学方程**：$R_b = k_b B f(T) f(L) C$
- **温度依赖**：$f(T) = e^{-\frac{E_a}{R}(\frac{1}{T} - \frac{1}{T_{opt}})}$
- **光照依赖**：$f(L) = \frac{L}{K_L + L}$

#### 7.4.2 衰减模块 (DecayModule)
- **功能**：模拟污染物的自然降解
- **衰减类型**：一级衰减、二级衰减
- **环境依赖**：温度、pH、溶解氧影响

### 7.5 源汇模块

#### 7.5.1 源汇模块 (SourceSinkModule)
- **点源**：离散排放源，支持时间变化
- **面源**：分布式污染输入
- **线源**：线性污染源
- **汇项**：沉降、降解、吸收等去除过程

### 7.6 输出与可视化模块

#### 7.6.1 输出模块 (OutputModule)
- **数据格式**：NetCDF格式，便于科学计算
- **可视化**：2D/3D浓度场可视化
- **统计功能**：浓度统计、质量平衡分析
- **时间序列**：支持时间序列数据输出

### 7.7 模型特色功能

#### 7.7.1 多污染物支持
- 支持同时模拟多种污染物
- 污染物间相互作用和耦合反应
- 独立的物理化学参数设置

#### 7.7.2 环境响应
- **温度效应**：影响扩散系数、反应速率、生物活动
- **pH效应**：影响化学反应和生物过程
- **光照效应**：影响生物吸收和光合作用
- **波浪效应**：影响湍流扩散

#### 7.7.3 数值稳定性
- 迎风格式确保平流稳定性
- 自适应时间步长
- 质量守恒检查

### 7.8 应用场景

#### 7.8.1 海洋污染模拟
- 石油泄漏扩散
- 化学污染物传播
- 微塑料分布

#### 7.8.2 环境影响评估
- 污染物对海洋生态的影响
- 长期环境风险评估
- 污染源追踪

#### 7.8.3 管理决策支持
- 应急响应规划
- 污染控制策略
- 环境容量评估

### 7.9 技术优势

#### 7.9.1 模块化设计
- 各模块独立，便于维护和扩展
- 支持选择性启用/禁用特定过程
- 易于添加新的物理/化学过程

#### 7.9.2 高精度数值方法
- 采用成熟的数值方法
- 保证数值稳定性和精度
- 支持复杂几何和边界条件

#### 7.9.3 环境耦合
- 充分考虑环境因子的影响
- 支持实时环境数据输入
- 模拟真实海洋环境条件

### 7.10 模型验证与测试
- 包含完整的测试用例
- 支持多种验证场景
- 提供标准化的输出格式
- 支持与其他海洋模型的耦合
