import cv2 as cv
import math
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt


def low_pass_filter(slice_2d, threshold):

    """
    With this function we can perform a low pass filtering with different options
    - Ideal low pass filter
    - Gaussian low pass filter
    - Butterworth low pass filter

    Input: a 2D slice, and the threshold (distance in pixels from the center frequency that we want to erase and put to 0)
    Output: the same slice but cleaned from noise at low frequency
    """

    x, y = np.indices(slice_2d.shape[0:2])
    distance_map = np.sqrt((x - slice_2d.shape[0]//2)**2 + (y - slice_2d.shape[1]//2)**2) # to apply the thresholding this map will be used

    freq_img = fft.fft2(slice_2d) # apply fast fourier transform
    center_freq_img = fft.fftshift(freq_img) # center the freq img to translate the mean value at coordinates (h/2, w/2)
    magnitude_img = np.abs(center_freq_img)

    # Now let's implement the 3 different kinds of low pass filters
    # 1) ideal low pass filter

    ideal_mask = np.where(distance_map <= threshold, 1.0, 0.0)
    filtered_magnitude = np.log(1 + (magnitude_img * ideal_mask))
    filtered_freq = center_freq_img * ideal_mask
    original = np.abs(fft.ifft2(fft.ifftshift(filtered_freq)))

    # 2) Gaussian low pass filter
    gaussian_mask = np.exp(-distance_map**2 / (2 * (threshold**2)))
    filtered_magnitude_gaussian = np.log(1 + (magnitude_img * gaussian_mask))
    filtered_freq_gaussian = center_freq_img * gaussian_mask
    original_gaussian = np.abs(fft.ifft2(fft.ifftshift(filtered_freq_gaussian)))

    # 3) Butterworth low pass filter
    n = 3 # order of the filter
    butterworth_mask = 1 / (1 + (distance_map / threshold)**(2*n))
    filtered_magnitude_butterworth = np.log(1 + (magnitude_img * butterworth_mask))
    filtered_freq_butterworth = center_freq_img * butterworth_mask
    original_butterworth = np.abs(fft.ifft2(fft.ifftshift(filtered_freq_butterworth)))

    fig, axs = plt.subplots(2, 3, figsize=(15, 5))
    axs[0,0].imshow(filtered_magnitude, cmap='gray')
    axs[0,0].set_title('Ideal Low Pass Filtered Magnitude')
    axs[0,0].axis('off')

    axs[0,1].imshow(filtered_magnitude_gaussian, cmap='gray')
    axs[0,1].set_title('Gaussian Low Pass Filtered Magnitude')
    axs[0,1].axis('off')

    axs[0,2].imshow(filtered_magnitude_butterworth, cmap='gray')
    axs[0,2].set_title('Butterworth Low Pass Filtered Magnitude')
    axs[0,2].axis('off')

    axs[1,0].imshow(original, cmap='gray')
    axs[1,0].set_title('Original Image - Ideal filter')
    axs[1,0].axis('off')

    axs[1,1].imshow(original_gaussian, cmap='gray')
    axs[1,1].set_title('Original Image - Gaussian filter')
    axs[1,1].axis('off')

    axs[1,2].imshow(original_butterworth, cmap='gray')
    axs[1,2].set_title('Original Image - Butterworth filter')
    axs[1,2].axis('off')



def first_derivative_edge_detection(slice):

    slice = slice.astype(np.float64)

    # 1) Convolution with Prewitt and Sobel kernels

    prewitt_x = np.array([ [-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]
                          ], dtype=np.float64)
    
    prewitt_y = np.array([ [-1, -1, -1],
                           [0,  0, 0],
                           [1, 1, 1]
                          ], dtype=np.float64)

    prewitt_edges_x = cv.filter2D(slice, -1, prewitt_x)
    prewitt_edges_y = cv.filter2D(slice, -1, prewitt_y)
    prewitt_edges = np.sqrt(prewitt_edges_x**2 + prewitt_edges_y**2)
    
    sobel_x = np.array([ [-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]
                          ], dtype=np.float64)
    
    sobel_y = np.array([ [-1, -2, -1],
                           [0,  0, 0],
                           [1, 2, 1]
                          ], dtype=np.float64)

    sobel_edges_x = cv.filter2D(slice, -1, sobel_x)
    sobel_edges_y = cv.filter2D(slice, -1, sobel_y)
    sobel_edges = np.sqrt(sobel_edges_x**2 + sobel_edges_y**2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(prewitt_edges, cmap='gray')
    axs[0].set_title('Prewitt Edge Detection')
    axs[0].axis('off')

    axs[1].imshow(sobel_edges, cmap='gray')
    axs[1].set_title('Sobel Edge Detection')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()



def second_derivative_edge_detection(slice):

    slice = slice.astype(np.float64)

    laplacian_mask_45 = np.array([ [0, 1, 0],
                           [1,  -4, 1],
                           [0, 1, 0]
                          ], dtype=np.float64)

    laplacian_mask_90 = np.array([ [1, 1, 1],
                           [1,  -8, 1],
                           [1, 1, 1]
                          ], dtype=np.float64)

    laplacian_edges_45 = cv.filter2D(slice, -1, laplacian_mask_45)
    laplacian_edges_90 = cv.filter2D(slice, -1, laplacian_mask_90)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].imshow(laplacian_edges_45, cmap='gray')
    axs[0].set_title('Laplacian Edge Detection (45°)')
    axs[0].axis('off')

    axs[1].imshow(laplacian_edges_90, cmap='gray')
    axs[1].set_title('Laplacian Edge Detection (90°)')
    axs[1].axis('off')  

    plt.tight_layout()
    plt.show()



def canny_edge_detection(slice):

    # Canny works with uint8 images, so we need to ensure the input slice is in the correct format
    if slice.dtype != np.uint8:
        slice_normalized = cv.normalize(slice, None, 0, 255, cv.NORM_MINMAX)
        slice_uint8 = slice_normalized.astype(np.uint8)
    else:
        slice_uint8 = slice

    # 2. Applicazione di Canny (richiede uint8)
    edges = cv.Canny(slice_uint8, 100, 200)

    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    plt.show()






# def laplacian_of_gaussian():

#     pass

# def difference_of_gaussian():

#     img1 = cv.GaussianBlur(img, (0,0), sigmaX=1, sigmaY=1)
#     img2 = cv.GaussianBlur(img, (0,0), sigmaX=10, sigmaY=10)

#     result = img2 - img1 # differenze tra due immagini blurrate con due diversi valori di sigma

