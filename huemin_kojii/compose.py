import os
import random
from PIL import Image

def create_grid_image(image_folder, grid_size, output_path):
    n, m = grid_size  # n is the number of rows, m is the number of columns
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) < n * m:
        raise ValueError(f"Not enough images to create a {n}x{m} grid. Only {len(images)} images found.")
    
    selected_images = random.sample(images, n * m)
    imgs = [Image.open(img).convert('RGB') for img in selected_images]
    
    # Get the size of the first image to set the size of the grid cells
    img_width, img_height = imgs[0].size
    grid_img_width = img_width * m
    grid_img_height = img_height * n
    grid_img = Image.new('RGB', (grid_img_width, grid_img_height))
    
    for i, img in enumerate(imgs):
        # Resize image to match the first one's size if necessary
        if img.size != imgs[0].size:
            img = img.resize(imgs[0].size, Image.ANTIALIAS)
        # Calculate the position of the current image
        x = (i % m) * img_width
        y = (i // m) * img_height
        grid_img.paste(img, (x, y))
    
    grid_img.save(output_path, quality=90)

# Example usage
image_folder = '/data/xander/Projects/cog/eden-sd-pipelines/huemin_kojii/test_imgs'  # Update this path
grid_size = (3, 4)  # Update this grid size (rows, columns)

for i in range(10):
    output_path = f'grid_image_{i}.jpg'  # Update this path
    create_grid_image(image_folder, grid_size, output_path)
