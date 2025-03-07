import xarray as xr
from visual import plot_3d_currents

nc_file = "../env_config/Data/Combined/combined_environment.nc"
dataset = xr.open_dataset(nc_file)

plot_3d_currents(dataset, output_dir="output", skip=20, arrow_size=0.05, arrow_height_offset=5, arrow_alpha=0.4, arrow_head_size=10)
