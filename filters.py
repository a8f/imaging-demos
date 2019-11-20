#!/usr/bin/python3

import numpy as np
import cv2
from util import read_image


def correlation(I: np.ndarray, h: np.ndarray, mode: str) -> np.ndarray:
    """
    The correlation operation
    :param I: image to run correlation on
    :param h: filter to correlate with
    :param mode: 'full', 'same', or 'valid'
    :return: the processed image as a numpy array
    """
    if len(h.shape) > 2 or h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
        raise ValueError('Invalid filter')
    # To simplify computation, get the range of filter rows and columns with respect to the center of the filter
    # e.g. for 3 row filter we want row range of range(-1, 2)
    frow_range = range(-(h.shape[0] // 2), (h.shape[0] // 2) + 1)
    fcol_range = range(-(h.shape[1] // 2), (h.shape[1] // 2) + 1)
    # Set up the output array depending on the mode
    if mode == 'full':
        # For 'full' mode, insert a buffer of 0 pixels with size = 1/2 filter size around the image
        # Then the operations are all the same as in 'same' mode
        # Insert (filter height // 2) rows of 0s at top and bottom
        I = np.insert(I, I.shape[0], np.zeros((h.shape[0] // 2, I.shape[1])), 0)
        I = np.insert(I, 0, np.zeros((h.shape[0] // 2, I.shape[1])), 0)
        # Insert (filter width // 2) cols of 0s at the left and right
        I = np.insert(I, I.shape[1], np.zeros((h.shape[1] // 2, I.shape[0])), 1)
        I = np.insert(I, 0, np.zeros((h.shape[1] // 2, I.shape[0])), 1)
        result = np.zeros(I.shape)
    elif mode == 'same':
        result = np.zeros(I.shape)
    elif mode == 'valid':
        # For 'valid' mode, we don't process the outer 1/2 filter size pixels around the image
        result = np.zeros((I.shape[0] - 2 * (h.shape[0] // 2), I.shape[1] - 2 * (h.shape[1] // 2)))
    else:
        raise ValueError('Invalid mode')
    # Iterate over filter centers
    for row in range(result.shape[0]):  # Image row
        for col in range(result.shape[1]):  # Image col
            val = 0  # Running total for the value of pixel at [y, x]
            # Iterate over filter pixels
            for filter_row in frow_range:  # Filter row
                if row + filter_row < 0:
                    continue
                if row + filter_row >= result.shape[0]:
                    break
                for filter_col in fcol_range:  # Filter col
                    if col + filter_col < 0:
                        continue
                    if col + filter_col >= result.shape[1]:
                        break
                    # If we are in 'valid' mode then the first row of the output is actually row (filter height // 2) of
                    # the image (similar for cols). So we need to offset the index into I by that amount.
                    if mode == 'valid':
                        val += h[filter_row + frow_range[-1], filter_col + fcol_range[-1]] * I[
                            row + filter_row + (h.shape[0] // 2), col + filter_col + (h.shape[1] // 2)]
                    else:
                        val += h[filter_row + frow_range[-1], filter_col + fcol_range[-1]] * I[
                            row + filter_row, col + filter_col]
            result[row, col] = val
    return result


def run_correlation():
    image = read_image('lenna.png')
    h = np.asarray([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]])
    cv2.imwrite('full.png', correlation(image, h, 'full'))
    cv2.imwrite('same.png', correlation(image, h, 'same'))
    cv2.imwrite('valid.png', correlation(image, h, 'valid'))


def portrait_mode(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a portrait mode version of the image where the background is blurred
    :param image: input image
    :param mask: mask where background is white and foreground is black
    :return: portrait mode version of image
    """
    # Gaussian blur
    f = 1 / 256 * np.asarray(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    # Set up the output array
    result = np.zeros(image.shape)
    filter_dim_range = range(-f.shape[0] // 2, f.shape[0] // 2 + 1)
    # Iterate over filter centers
    for row in range(result.shape[0]):  # Image row
        for col in range(result.shape[1]):  # Image col
            # If this is the foreground just copy the pixel value
            if mask[row, col] == 255:
                result[row, col] = image[row, col]
                continue
            # Otherwise apply filter
            val = np.zeros((3,))
            # Iterate over filter pixels
            for filter_row in filter_dim_range:  # Filter row
                if row + filter_row < 0:
                    continue
                if row + filter_row >= result.shape[0]:
                    break
                for filter_col in filter_dim_range:  # Filter col
                    if col + filter_col < 0:
                        continue
                    if col + filter_col >= result.shape[1]:
                        break
                    val[0] += f[filter_row + filter_dim_range[-1], filter_col + filter_dim_range[-1]] * image[
                        row + filter_row, col + filter_col, 0]
                    val[1] += f[filter_row + filter_dim_range[-1], filter_col + filter_dim_range[-1]] * image[
                        row + filter_row, col + filter_col, 1]
                    val[2] += f[filter_row + filter_dim_range[-1], filter_col + filter_dim_range[-1]] * image[
                        row + filter_row, col + filter_col, 2]
            result[row, col] = val
    return result


def run_portrait_mode():
    image = read_image('kanye.png')
    mask = read_image('kanye_mask.png')
    cv2.imwrite('portrait.png', portrait_mode(image, mask))


def is_separable_filter(f: np.ndarray) -> bool:
    """
    Returns true iff filter is separable
    :param f: the filter to check for separability
    :return: true iff the filter is separable
    """
    u, sigma, vt = np.linalg.svd(f)
    # Get the elements of sigma which are near 0
    # (just using nonzero will give false positives for #s near machine epsilon)
    # Could also just use np.linalg.matrix_rank but then we are calculating SVD twice
    # which is usually more work than this way
    zero = np.isclose(sigma, np.zeros_like(sigma))
    # If there is not exactly one nonzero sigma then it is not separable
    if len(zero) - zero.sum() != 1:
        print('Filter {} is not separable'.format(f))
        return False
    # Get the horizontal and vertical filters
    vert = np.abs(u[:, 0]) * np.sqrt(sigma[0])
    horz = np.abs(vt[0]) * np.sqrt(sigma[0])
    print("Filter {} is separable into:\nu = {}\nv = {}^T".format(f, vert, horz))
    return True


def run_is_separable_filter():
    is_separable_filter(np.asarray([[0, 1, 0], [1, 4, 1], [0, 1, 0]]))
    print()
    is_separable_filter(np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    print()
    is_separable_filter(np.asarray([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]))
