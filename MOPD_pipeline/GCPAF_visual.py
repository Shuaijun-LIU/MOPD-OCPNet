import xarray as xr
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize_each_variable(dataset, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for var_name in dataset.data_vars:
        data = dataset[var_name]
        plt.figure(figsize=(16, 10), dpi=200)  # Larger figure size and higher DPI for better clarity
        plt.pcolormesh(data.longitude, data.latitude, data.isel(time=0, depth=0), shading='auto')
        plt.colorbar(label=f'{var_name} ({data.attrs.get("units", "unknown unit")})')
        plt.title(f'{data.attrs.get("long_name", var_name)}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        file_path = os.path.join(output_directory, f"{var_name}.png")
        plt.savefig(file_path)
        plt.close()

def create_combined_image(image_directory, output_path):
    image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.png')]
    images = [Image.open(file) for file in image_files]

    widths, heights = zip(*(img.size for img in images))
    total_width = max(widths) * 2
    total_height = (len(images) + 1) // 2 * max(heights)

    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        combined_image.paste(img, (x_offset, y_offset))
        if i % 2 == 1:
            x_offset = 0
            y_offset += img.height
        else:
            x_offset += img.width

    combined_image.save(output_path)

if __name__ == "__main__":
    input_file_path = "./Data/GCPAF/combined_gcpaf_data.nc"
    output_images_directory = "./output"
    combined_image_output_path = "./output/combined_visualization.png"

    ds = xr.open_dataset(input_file_path)
    visualize_each_variable(ds, output_images_directory)
    create_combined_image(output_images_directory, combined_image_output_path)

    print(f"Combined visualization saved at: {combined_image_output_path}")
