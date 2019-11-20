#!/usr/bin/python3

import numpy as np
import cv2
from util import read_image


def add_salt_and_pepper_noise(I: np.ndarray, d: float) -> np.ndarray:
    """
    Return copy of I with salt and pepper noise added
    :param I: Grayscale image
    :param d: density of salt and pepper noise in output image
    :return: np.ndarray of same dimensions as I with salt and pepper noise added
    """
    noisy_pixel_count = int(round(I.shape[0] * I.shape[1] * d))
    indices = np.stack((np.random.randint(0, I.shape[0], noisy_pixel_count),
                        np.random.randint(0, I.shape[1], noisy_pixel_count)), axis=1)
    noisy = np.copy(I)
    for i in indices:
        noisy[i[0], i[1]] = I.max() if np.random.random() < 0.5 else I.min()
    return noisy


def filter_salt_and_pepper_noise(I: np.ndarray) -> np.ndarray:
    h = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    if len(h.shape) > 2 or h.shape[0] % 2 == 0 or h.shape[1] % 2 == 0:
        raise ValueError('Invalid filter')
    # To simplify computation, get the range of filter rows and columns with respect to the center of the filter
    # e.g. for 3 row filter we want row range of range(-1, 2)
    frow_range = range(-(h.shape[0] // 2), (h.shape[0] // 2) + 1)
    fcol_range = range(-(h.shape[1] // 2), (h.shape[1] // 2) + 1)
    result = np.zeros(I.shape)
    # Iterate over filter centers
    for row in range(result.shape[0]):  # Image row
        for col in range(result.shape[1]):  # Image col
            if I[row, col] != 0 and I[row, col] != 255:
                result[row, col] = I[row, col]
                continue
            val = 0  # Running total for the value of pixel at [y, x]
            vals_added = 0
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
                    if I[row + filter_row, col + filter_col] == 255 or I[row + filter_row, col + filter_col] == 0:
                        continue
                    vals_added += 1
                    val += h[filter_row + frow_range[-1], filter_col + fcol_range[-1]] * I[
                        row + filter_row, col + filter_col]
            # All surrounding pixels are 0 or 255
            if vals_added == 0:
                # Just pick the value of the pixel up or to the left, or leave it noisy if those don't exist
                if col > 0:
                    result[row, col] = I[row, col - 1]
                elif row > 0:
                    result[row, col] = I[row - 1, col]
            # Got some non-noisy pixels, so average them to get the final value of this pixel
            else:
                result[row, col] = val / vals_added
    return result


def run_filter_salt_and_pepper_noise():
    image = read_image('image.png')
    noisy = add_salt_and_pepper_noise(image)
    result = filter_salt_and_pepper_noise(noisy)
    cv2.imwrite('denoised.png', result)
