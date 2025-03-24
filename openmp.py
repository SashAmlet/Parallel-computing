import cv2
import numpy as np
import sys

from median_filter import median_filter_manual


def apply_median_filter(input_image_path: str, output_image_path: str, kernel_size: int = 3):
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to load the image: {input_image_path}")
    
    filtered_image = median_filter_manual(image, kernel_size)
    
    cv2.imwrite(output_image_path, filtered_image)
    print(f"Processed image saved to: {output_image_path}")

# Виклик
apply_median_filter("data/rawdata/1.jpg", "data/1_1.jpg", kernel_size=5)
