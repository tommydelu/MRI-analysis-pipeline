import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import scipy.fft as fft
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------- #
# This function performs a low pass filtering in the frequcency     #
# domain with different options (ideal, gaussian, butterworth)      #
# ----------------------------------------------------------------- #
def freqLowPassFilter(slice: np.ndarray, threshold: float, order: int = 2) -> None:

    """
    Input
    - slice: 2D data structure
    - threshold: distance in pixels from the center frequency that we want to erase and put to 0

    Output:
    – filtered_ideal, filtered_gaussian, filtered_butterworth: the slice cleaned from the noise for each of the methods
    """
    x, y = np.indices(slice.shape[0:2])
    distance_map = np.sqrt((x - slice.shape[0]//2)**2 + (y - slice.shape[1]//2)**2) # Distance of each point from the center

    freq_slice = fft.fftshift(fft.fft2(slice)) # Apply DFT and centering
    magnitude_slice = np.abs(freq_slice)

    # 1) ideal low pass filter
    ideal_mask = np.where(distance_map <= threshold, 1.0, 0.0)
    magnitude_slice_clean = magnitude_slice * ideal_mask # stretching operation
    freq_slice_clean = freq_slice * ideal_mask
    filtered_ideal = np.abs(fft.ifft2(fft.ifftshift(freq_slice_clean)))

    print(f"Before contrast stretching - Maximum value: {np.max(magnitude_slice_clean)}")
    magnitude_slice_clean = np.log(1+(magnitude_slice * ideal_mask)) # stretching operation
    print(f"After contrast stretching - Maximum value: {np.max(magnitude_slice_clean)}")

    # 2) Gaussian low pass filter
    gaussian_mask = np.exp(-distance_map**2 / (2 * (threshold**2)))
    gaussian_magnitude_clean = np.log(1 + (magnitude_slice * gaussian_mask))
    freq_gaussian_slice = freq_slice * gaussian_mask
    filtered_gaussian = np.abs(fft.ifft2(fft.ifftshift(freq_gaussian_slice)))

    # 3) Butterworth low pass filter
    butterworth_mask = 1 / (1 + (distance_map / threshold)**(2*order))
    butterworth_magnitude_clean = np.log(1 + (magnitude_slice * butterworth_mask))
    freq_butterworth_slice = freq_slice * butterworth_mask
    filtered_butterworth = np.abs(fft.ifft2(fft.ifftshift(freq_butterworth_slice)))

    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    axs[0,0].imshow(magnitude_slice_clean, cmap='gray')
    axs[0,0].set_title('Ideal Low Pass Filtered Magnitude')
    axs[0,0].axis('off')

    axs[0,1].imshow(gaussian_magnitude_clean, cmap='gray')
    axs[0,1].set_title('Gaussian Low Pass Filtered Magnitude')
    axs[0,1].axis('off')

    axs[0,2].imshow(butterworth_magnitude_clean, cmap='gray')
    axs[0,2].set_title('Butterworth Low Pass Filtered Magnitude')
    axs[0,2].axis('off')

    axs[1,0].imshow(filtered_ideal, cmap='gray')
    axs[1,0].set_title('Original Image - Ideal filter')
    axs[1,0].axis('off')

    axs[1,1].imshow(filtered_gaussian, cmap='gray')
    axs[1,1].set_title('Original Image - Gaussian filter')
    axs[1,1].axis('off')

    axs[1,2].imshow(filtered_butterworth, cmap='gray')
    axs[1,2].set_title('Original Image - Butterworth filter')
    axs[1,2].axis('off')

    plt.tight_layout()
    plt.show()

    return filtered_ideal, filtered_gaussian, filtered_butterworth

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def firstDerivativeEdgeDetection(slice):
    # Convert to float64 to prevent overflow/truncation during convolutions
    slice_float = slice.astype(np.float64)

    # 1) Convolution with Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float64)
    
    prewitt_y = np.array([[-1, -1, -1],
                          [ 0,  0,  0],
                          [ 1,  1,  1]], dtype=np.float64)

    prewitt_edges_x = cv.filter2D(slice_float, -1, prewitt_x)
    prewitt_edges_y = cv.filter2D(slice_float, -1, prewitt_y)
    
    # Compute gradient magnitude
    prewitt_edges = np.sqrt(prewitt_edges_x**2 + prewitt_edges_y**2)
    
    # 2) Convolution with Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float64)

    sobel_edges_x = cv.filter2D(slice_float, -1, sobel_x)
    sobel_edges_y = cv.filter2D(slice_float, -1, sobel_y)
    
    sobel_edges = np.sqrt(sobel_edges_x**2 + sobel_edges_y**2)

    # Plot original side-by-side with edges for better comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].imshow(slice, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(prewitt_edges, cmap='gray')
    axs[1].set_title('Prewitt Edge Detection')
    axs[1].axis('off')

    axs[2].imshow(sobel_edges, cmap='gray')
    axs[2].set_title('Sobel Edge Detection')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    
    return prewitt_edges, sobel_edges

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def secondDerivativeEdgeDetection(slice):
    slice_float = slice.astype(np.float64)

    # 4-neighborhood Laplacian (horizontal and vertical only)
    laplacian_mask_90 = np.array([[ 0,  1,  0],
                                  [ 1, -4,  1],
                                  [ 0,  1,  0]], dtype=np.float64)

    # 8-neighborhood Laplacian (includes diagonals)
    laplacian_mask_45 = np.array([[ 1,  1,  1],
                                  [ 1, -8,  1],
                                  [ 1,  1,  1]], dtype=np.float64)

    # Taking the absolute value makes zero-crossings and edges pop out visually
    laplacian_edges_90 = np.abs(cv.filter2D(slice_float, -1, laplacian_mask_90))
    laplacian_edges_45 = np.abs(cv.filter2D(slice_float, -1, laplacian_mask_45))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].imshow(slice, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(laplacian_edges_90, cmap='gray')
    axs[1].set_title('Laplacian (4-neighborhood)')
    axs[1].axis('off')

    axs[2].imshow(laplacian_edges_45, cmap='gray')
    axs[2].set_title('Laplacian (8-neighborhood)')
    axs[2].axis('off')  

    plt.tight_layout()
    plt.show()
    
    return laplacian_edges_90, laplacian_edges_45

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def cannyEdgeDetection(slice, t1: int = 100, t2: int = 200):
    # Canny requires uint8 format, so we normalize the input if necessary
    if slice.dtype != np.uint8:
        slice_normalized = cv.normalize(slice, None, 0, 255, cv.NORM_MINMAX)
        slice_uint8 = slice_normalized.astype(np.uint8)
    else:
        slice_uint8 = slice.copy()

    # Apply Canny using the provided thresholds
    edges = cv.Canny(slice_uint8, t1, t2)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(slice, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(edges, cmap='gray')
    axs[1].set_title(f'Canny Edge Detection (t1={t1}, t2={t2})')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return edges

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def fmriVolumesSmoothing(data_4d: np.ndarray, sigma: float = 1.0) -> np.ndarray:

    smoothed_data = np.zeros_like(data_4d)
    n_volumes = data_4d.shape[3]

    # For each volume at time t, apply smoothing
    for t in range(n_volumes):
        volume3d = data_4d[:, :, :, t]
        smoothed_data[:, :, :, t] = gaussian_filter(volume3d, sigma=sigma)  
    return smoothed_data

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def logEdgeDetection(slice, sigmas=[1.0, 2.0, 3.0]):

    slice_float = slice.astype(np.float64)
    laplacian_mask = np.array([[ 1,  1,  1],
                               [ 1, -8,  1],
                               [ 1,  1,  1]], dtype=np.float64)

    fig, axs = plt.subplots(1, len(sigmas) + 1, figsize=(5 * (len(sigmas) + 1), 5))
    
    axs[0].imshow(slice, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    log_edges_list = []

    for i, sigma in enumerate(sigmas):
        # 1. Smoothing Gaussiano (compute the kernel size according to the sigma)
        ksize = int(2 * round(3 * sigma) + 1) 
        blurred_slice = cv.GaussianBlur(slice_float, (ksize, ksize), sigmaX=sigma)
        log_edges = np.abs(cv.filter2D(blurred_slice, -1, laplacian_mask))
        log_edges_list.append(log_edges)

        axs[i+1].imshow(log_edges, cmap='gray')
        axs[i+1].set_title(f'LoG Edge Detection (Sigma={sigma})')
        axs[i+1].axis('off')

    plt.tight_layout()
    plt.show()
    
    return log_edges_list