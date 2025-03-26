import cv2
import time

from median_filter import median_filter_manual


def apply_median_filter(input_image_path: str, output_image_path: str, kernel_size: int = 3):
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load the image: {input_image_path}")
    
    start_time = time.time()
    filtered_image = median_filter_manual(image, kernel_size)
    end_time = time.time()
    
    cv2.imwrite(output_image_path, filtered_image)
    print(f"Processed image saved to: {output_image_path}")
    print(f"Execution time: {end_time - start_time:.2f} seconds", flush=True)

# Виклик
apply_median_filter("data/rawdata/1.jpg", "data/1_1.jpg", kernel_size=5)
