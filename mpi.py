from mpi4py import MPI
import cv2
import numpy as np
import time

def median_filter_manual(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Manual implementation of the median filter without using library functions.
    :param image: Input image in NumPy format (BGR)
    :param kernel_size: Kernel size (must be odd)
    :return: Filtered image
    """
    pad = kernel_size // 2
    height, width, channels = image.shape
    
    # Output image and padding
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

def parallel_median_filter(input_image_path: str, output_image_path: str, kernel_size: int = 3):
    """
    MPI implementation of the median filter.
    """
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello world {rank}", flush=True)

    if rank == 0:
        start_time = time.time()  # Start time measurement
        # Load the image on the main process
        image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Error loading image: {input_image_path}")

        height, _, _ = image.shape

        # Determine the number of rows per process
        chunk_size = height // size
        remainder = height % size
        overlap = kernel_size // 2  # Overlap size

        # Send data to processes
        # Since remainder is guaranteed to be smaller than size, 
        # the first remaining processes will get 1 row more
        for i in range(1, size):
            start_row = i * chunk_size + min(i, remainder) - overlap
            end_row = start_row + chunk_size + (1 if i < remainder else 0) + overlap
            start_row = max(0, start_row)  # Ensure we don't go out of bounds
            end_row = min(height, end_row)  # Ensure we don't go out of bounds
            comm.send(image[start_row:end_row], dest=i)
            print(f"Sent data to process {i}", flush=True)

        # Process own chunk
        start_row = 0
        end_row = chunk_size + (1 if 0 < remainder else 0) + overlap
        filtered_chunk = median_filter_manual(image[start_row:end_row], kernel_size)

    else:
        # Receive own chunk of the image
        sub_image = comm.recv(source=0)
        print(f"Received data from process {comm.Get_rank()}", flush=True)
        filtered_chunk = median_filter_manual(sub_image, kernel_size)

    # Gather all chunks on rank=0
    gathered_chunks = comm.gather(filtered_chunk, root=0)
    print(f"Gather all chunks", flush=True)

    if rank == 0:
        # Combine chunks, removing overlap
        final_image = gathered_chunks[0][:-overlap]  # Remove bottom overlap of the first chunk
        for i in range(1, len(gathered_chunks) - 1):
            final_image = np.vstack((final_image, gathered_chunks[i][overlap:-overlap]))  # Remove top and bottom overlap
        final_image = np.vstack((final_image, gathered_chunks[-1][overlap:]))  # Remove top overlap of the last chunk

        cv2.imwrite(output_image_path, final_image)
        print(f"Processed image saved to: {output_image_path}", flush=True)
        end_time = time.time()  # End time measurement
        print(f"Execution time: {end_time - start_time:.2f} seconds", flush=True)


parallel_median_filter("data/rawdata/1.jpg", "data/1_1.jpg", kernel_size=5)
