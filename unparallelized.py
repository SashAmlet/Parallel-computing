import cv2
import numpy as np
import time

def median_filter_manual(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Implementation of a median filter manually without using library functions.
    :param image: Input image in NumPy format (BGR)
    :param kernel_size: Kernel size (must be odd)
    :return: Filtered image
    """
    pad = kernel_size // 2
    height, width, channels = image.shape
    
    # Output and padding image
    output = np.zeros_like(image)
    padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)
    
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                # Extract the window
                window = padded_image[y:y + kernel_size, x:x + kernel_size, c].flatten()
                # Compute the median
                output[y, x, c] = np.median(window)
    
    return output

def apply_median_filter(input_image_path: str, output_image_path: str, kernel_size: int = 3):
    """
    Reads an image, processes it with a median filter, and saves the result.
    :param input_image_path: Path to the input image
    :param output_image_path: Path to save the output image
    :param kernel_size: Kernel size of the filter (must be odd)
    """
    # Load the image
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load the image: {input_image_path}")
    
    # Process the image
    start_time = time.time()
    filtered_image = median_filter_manual(image, kernel_size)
    end_time = time.time()
    
    # Save the result
    cv2.imwrite(output_image_path, filtered_image)
    print(f"Processed image saved to: {output_image_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds", flush=True)



# Example call
apply_median_filter("data\\rawdata\\1.jpg", "data\\1_1.jpg", kernel_size=5)