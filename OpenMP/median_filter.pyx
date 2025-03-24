# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, qsort
from cython.parallel import prange

# Comparison function for qsort
cdef int compare(const void* a, const void* b) noexcept nogil:
    return (<int*>a)[0] - (<int*>b)[0]

def median_filter_manual(np.ndarray[np.uint8_t, ndim=3] image, int kernel_size):
    """
    Fast median filter with OpenMP parallelization.
    """
    cdef int height = image.shape[0], width = image.shape[1], channels = image.shape[2]
    cdef int pad = kernel_size // 2, window_size = kernel_size * kernel_size

    # Output and padding image
    cdef np.ndarray[np.uint8_t, ndim=3] output = np.zeros_like(image)
    cdef np.ndarray[np.uint8_t, ndim=3] padded_image = np.pad(
        image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0
    )

    cdef int y, x, c, i, j
    cdef int* window = <int*> malloc(window_size * sizeof(int))
    if not window:
        raise MemoryError("Failed to allocate memory for window buffer")

    try:
        # OpenMP parallelized loop
        with nogil:
            for y in prange(height):
                for x in range(width):
                    for c in range(channels):
                        
                        # Extract window
                        for i in range(kernel_size):
                            for j in range(kernel_size):
                                window[i * kernel_size + j] = padded_image[y + i, x + j, c]
                    
                        # Sort window using C qsort (avoiding GIL)
                        qsort(window, window_size, sizeof(int), compare)

                        # Assign median value
                        output[y, x, c] = window[window_size // 2]
    finally:
        free(window)  # Free memory

    return output
