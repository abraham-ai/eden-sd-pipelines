import random
import numpy as np
import cv2
from PIL import Image
from noise import snoise2

def gen(seed):

    random.seed(seed)
    np.random.seed(seed)

    # Example usage
    hue_dict = {
        'red': 0,
        'yellow': 30,
        'green': 60,
        'cyan': 90,
        'blue': 120,
        'magenta': 150,
    }

    color_list = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red', 'yellow', 'blue']
    color_name = random.choice(color_list)
    hue_val = hue_dict[color_name]
    spread = random.randint(0,20)

    def random_hsv(spread=10):
        # Predefined colors with corresponding hue value
        # Generate color based on the selected hue value in the HSV color space
        hue = (hue_val + random.randint(-spread, spread)) % 180
        saturation = random.randint(0, 100)
        value = random.randint(80, 220)
        color_hsv = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2RGB)
        color = color_hsv[0][0]
        color = (int(color[0]), int(color[1]), int(color[2]))
        return color

    def random_dark_hsv():
        # Generate random pastel color in the HSV color space
        hue = (hue_val + random.randint(-spread, spread)) % 180
        saturation = random.randint(20, 60)
        value = random.randint(0, 30)
        pastel = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2RGB)
        color = pastel[0][0]
        color = (int(color[0]),int(color[1]),int(color[2]))
        return color

    def add_gradient_background(canvas, start_color, end_color):
        height, width = canvas.shape[:2]
        start_color = np.array(start_color, dtype=np.uint8)
        end_color = np.array(end_color, dtype=np.uint8)
        gradient = np.linspace(start_color, end_color, width).astype(np.uint8)
        canvas[:] = np.repeat(gradient[np.newaxis, :, :], height, axis=0)
        return canvas

    def add_random_gradient_background(canvas):
        start_color = random_dark_hsv()
        end_color = random_dark_hsv()
        canvas = add_gradient_background(canvas, start_color, end_color)
        return canvas

    def create_canvas(height, width):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        return canvas

    def add_noise(canvas, noise_type, noise_param):
        if noise_type == "gaussian":
            # Add Gaussian noise to the canvas
            canvas = canvas + np.random.normal(0, noise_param, canvas.shape)
        elif noise_type == "salt_and_pepper":
            # Add salt and pepper noise to the canvas
            canvas = np.copy(canvas)
            canvas[np.random.randint(0, canvas.shape[0], int(canvas.size * noise_param * 0.004))] = 255
            canvas[np.random.randint(0, canvas.shape[0], int(canvas.size * (1 - noise_param) * 0.004))] = 0
        else:
            raise ValueError("Invalid noise type. Please specify 'gaussian' or 'salt_and_pepper'.")
        return canvas

    def liquid_distortion(canvas, strength=0.1, scale=0.0):
        strength = random.randint(0,100)/10
        height, width = canvas.shape[:2]
        dx = strength * np.random.randn(height, width)
        dy = strength * np.random.randn(height, width)
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map1 = (x + dx).astype(np.float32)
        map2 = (y + dy).astype(np.float32)
        canvas = cv2.remap(canvas, map1, map2, cv2.INTER_LINEAR)
        return canvas

    def add_random_blur(canvas):
        kernel_size = random.randint(3, 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = random.uniform(1, 2)
        canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), sigma)
        return canvas

    def zoom_in(canvas, zoom_percent):
        height, width = canvas.shape[:2]
        zoom_factor = 1 + zoom_percent / 100
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
        canvas = cv2.warpAffine(canvas, M, (new_width, new_height))
        canvas = cv2.getRectSubPix(canvas, (width, height), center)
        return canvas

    def draw_random_filled_rectangles(canvas):
        # Get the height and width of the canvas
        height, width = canvas.shape[:2]
        n_rectangles = random.randint(20, 40)
        w_scale = np.random.randint(50, 80)/1000
        h_scale = np.random.randint(100, 180)/1000
        for i in range(n_rectangles):
            # Generate random rectangle properties
            color = random_hsv()
            W = np.random.randint(100, 150)
            H = np.random.randint(25, 50)
            center = (np.random.normal(loc=width*0.5,scale=width*w_scale), np.random.normal(loc=height*0.5,scale=height*h_scale))
            angle = np.random.choice([0, 45, 90])

            # Generate rectangle points
            points = np.array([[center[0]-W, center[1]-H],
                            [center[0]+W, center[1]-H],
                            [center[0]+W, center[1]+H],
                            [center[0]-W, center[1]+H]], dtype=np.float32)

            # Rotate the rectangle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            points = cv2.transform(points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)

            # Draw the filled rectangle on the canvas
            cv2.fillConvexPoly(canvas, points.astype(int), color, 16)

        return canvas

    canvas = create_canvas(1024, 576)
    canvas = add_random_gradient_background(canvas)
    canvas = draw_random_filled_rectangles(canvas)
    canvas = liquid_distortion(canvas, strength=5, scale=5)
    canvas = add_random_blur(canvas)
    canvas = zoom_in(canvas, 10)
    canvas = add_noise(canvas, "gaussian", 5)
    cv2.imwrite("init.png", canvas)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    return image


def add_lines_to_image(seed, image, threshold=20, offset_margin=-0.1):

    random.seed(seed)
    np.random.seed(seed)

    width, height = image.size

    # Convert PIL image to NumPy array
    img = np.array(image)

    # Parameters for lines
    min_num_lines = 10
    max_num_lines = 50
    min_line_thickness = 1
    max_line_thickness = 1
    min_noise_scale = 0.005
    max_noise_scale = 0.010
    min_noise_strength = 10
    max_noise_strength = 200

    alpha = random.uniform(0, 0.1)

    # Randomize the number of lines and line thickness
    num_lines = random.randint(min_num_lines, max_num_lines)
    line_thickness = random.randint(min_line_thickness, max_line_thickness)

    # Randomize noise_scale and noise_strength
    noise_scale = random.uniform(min_noise_scale, max_noise_scale)
    noise_strength = random.uniform(min_noise_strength, max_noise_strength)

    # Calculate top and bottom margins
    top_margin = int(height * offset_margin)
    bottom_margin = int(height * (1 - offset_margin))

    # Generate lines
    for i in range(num_lines):
        y_offset = top_margin + int((bottom_margin - top_margin) * i / num_lines)

        # Define the line path points
        points = []
        for x in range(0, width):
            y = y_offset + int(noise_strength * snoise2(x * noise_scale, i * noise_scale))
            points.append((x, y))

        # Draw the line
        for j in range(len(points) - 1):
            x1, y1 = points[j]
            x2, y2 = points[j + 1]

            # Check if the points are within the image boundaries
            if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                # Check if both points have pixel values below the threshold
                if img[y1, x1].mean() < threshold and img[y2, x2].mean() < threshold:
                    color = img[y1, x1] * (1 - alpha) + np.array([255, 255, 255]) * alpha
                    img = cv2.line(img, (x1, y1), (x2, y2), tuple(map(int, color)), line_thickness)

    # Convert NumPy array back to PIL image
    return Image.fromarray(img)

def selective_sharpen(input_image, low_threshold=50, high_threshold=150, blur_size=(5, 5), blur_sigma=1.5, alpha=0.5):
    # Convert the PIL Image to an OpenCV image (numpy array)
    image = np.array(input_image)

    # Convert the image to BGR format (OpenCV uses BGR instead of RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply the unsharp mask to the entire image
    blurred_image = cv2.GaussianBlur(image, blur_size, blur_sigma)
    sharpened_image = cv2.addWeighted(image, 1 + alpha, blurred_image, -alpha, 0)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detection
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    # Dilate the edges to create a mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)

    # Convert the mask to a 3-channel image
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the sharpened image
    masked_sharpened_image = cv2.bitwise_and(sharpened_image, mask_colored)

    # Invert the mask (edge pixels have a value of 0, other pixels have a value of 255)
    inverted_mask = cv2.bitwise_not(mask_colored)

    # Apply the inverted mask to the original image
    masked_original_image = cv2.bitwise_and(image, inverted_mask)

    # Combine the masked original image with the masked sharpened image
    result_image = cv2.add(masked_original_image, masked_sharpened_image)

    # Convert the result back to RGB format
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Convert the result back to a PIL Image
    result_image_pil = Image.fromarray(result_image)

    return result_image_pil

def unsharp_mask(input_image, blur_size=(5, 5), blur_sigma=1.5, alpha=0.5):
    # Convert the PIL Image to an OpenCV image (numpy array)
    image = np.array(input_image)

    # Convert the image to BGR format (OpenCV uses BGR instead of RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blur the image using a Gaussian blur
    blurred_image = cv2.GaussianBlur(image, blur_size, blur_sigma)

    # Subtract the blurred image from the original image
    sharpened_image = cv2.addWeighted(image, 1 + alpha, blurred_image, -alpha, 0)

    # Convert the result back to RGB format
    result_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

    # Convert the result back to a PIL Image
    result_image_pil = Image.fromarray(result_image)

    return result_image_pil

def add_uniform_monochromatic_noise(seed, image, noise_intensity):
    random.seed(seed)
    np.random.seed(seed)
    
    # Load the pixel data
    pixels = image.load()

    # Add noise to each pixel
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            noise_value = random.randint(-noise_intensity, noise_intensity)
            new_pixel_values = tuple(
                max(0, min(value + noise_value, 255)) for value in pixels[i, j]
            )

            # Set the new pixel value
            pixels[i, j] = new_pixel_values

    return image