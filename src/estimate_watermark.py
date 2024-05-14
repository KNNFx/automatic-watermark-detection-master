import sys, os
import cv2
import numpy as np
import warnings
from matplotlib import pyplot as plt
import glob
import math
import numpy
import scipy, scipy.fftpack

# Variables
KERNEL_SIZE = 3

def estimate_watermark_gradients(directory):
    """
    Given a directory of images, estimate the watermark gradients.
    Returns the median gradients along with the list of gradients for further processing.
    
    Parameters:
    directory (str): The path to the directory containing the images.
    
    Returns:
    tuple: Median gradients in x and y directions, list of x gradients, list of y gradients.
    """
    if not os.path.exists(directory):
        warnings.warn("Directory does not exist.", UserWarning)
        return None

    image_files = glob.glob(os.path.join(directory, '*'))
    images = [cv2.imread(file) for file in image_files if cv2.imread(file) is not None]

    if not images:
        warnings.warn("No valid images found in directory.", UserWarning)
        return None

    # Compute gradients
    print("Computing gradients.")
    gradients_x, gradients_y = zip(*[(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE), 
                                      cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)) 
                                     for img in images])

    # Compute median of gradients
    print("Computing median gradients.")
    median_gradient_x = np.median(np.array(gradients_x), axis=0)
    median_gradient_y = np.median(np.array(gradients_y), axis=0)

    return (median_gradient_x, median_gradient_y, gradients_x, gradients_y)

def normalize_image(image):
    """ 
    Normalize an image matrix to [0, 1] range for visualization.
    
    Parameters:
    image (numpy.ndarray): Input image.
    
    Returns:
    numpy.ndarray: Normalized image.
    """
    return np.clip((image - np.min(image)) / (np.max(image) - np.min(image)), 0, 1)

def poisson_image_reconstruction(gradient_x, gradient_y, kernel_size=KERNEL_SIZE, iterations=100, step_size=0.1, 
                                 boundary_img=None, boundary_zero=True):
    """
    Perform Poisson reconstruction to reconstruct an image from its gradients.
    
    Parameters:
    gradient_x (numpy.ndarray): Gradient in x direction.
    gradient_y (numpy.ndarray): Gradient in y direction.
    kernel_size (int): Size of the Sobel kernel.
    iterations (int): Number of iterations for convergence.
    step_size (float): Step size for convergence.
    boundary_img (numpy.ndarray): Boundary image for reconstruction.
    boundary_zero (bool): Flag to indicate if the boundary should be zero.
    
    Returns:
    numpy.ndarray: Reconstructed image.
    """
    laplacian_xx = cv2.Sobel(gradient_x, cv2.CV_64F, 1, 0, ksize=kernel_size)
    laplacian_yy = cv2.Sobel(gradient_y, cv2.CV_64F, 0, 1, ksize=kernel_size)
    laplacian = laplacian_xx + laplacian_yy
    rows, cols, channels = laplacian.shape

    estimate = np.zeros(laplacian.shape) if boundary_zero else boundary_img.copy()
    estimate[1:-1, 1:-1, :] = np.random.random((rows-2, cols-2, channels))
    error_list = []

    for _ in range(iterations):
        old_estimate = estimate.copy()
        estimate[1:-1, 1:-1, :] = 0.25 * (estimate[0:-2, 1:-1, :] + estimate[1:-1, 0:-2, :] + 
                                          estimate[2:, 1:-1, :] + estimate[1:-1, 2:, :] - step_size**2 * laplacian[1:-1, 1:-1, :])
        error = np.sum(np.square(estimate - old_estimate))
        error_list.append(error)

    return estimate

def threshold_image(image, threshold=0.5):
    """
    Threshold an image such that all elements greater than threshold*MAX are set to 1, others to 0.
    
    Parameters:
    image (numpy.ndarray): Input image.
    threshold (float): Threshold value.
    
    Returns:
    numpy.ndarray: Thresholded image.
    """
    norm_image = normalize_image(image)
    return np.where(norm_image >= threshold, 1, 0)

def crop_watermark_area(gradient_x, gradient_y, threshold=0.4, boundary_margin=2):
    """
    Crop the watermark area by thresholding the magnitude of gradient.
    
    Parameters:
    gradient_x (numpy.ndarray): Gradient in x direction.
    gradient_y (numpy.ndarray): Gradient in y direction.
    threshold (float): Threshold value.
    boundary_margin (int): Boundary size around the cropped area.
    
    Returns:
    tuple: Cropped gradients in x and y directions.
    """
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    average_magnitude = np.average(normalize_image(gradient_magnitude), axis=2)
    binary_mask = threshold_image(average_magnitude, threshold=threshold)
    x_coords, y_coords = np.where(binary_mask == 1)

    x_min, x_max = np.min(x_coords) - boundary_margin - 1, np.max(x_coords) + boundary_margin + 1
    y_min, y_max = np.min(y_coords) - boundary_margin - 1, np.max(y_coords) + boundary_margin + 1

    return gradient_x[x_min:x_max, y_min:y_max, :], gradient_y[x_min:x_max, y_min:y_max, :]
