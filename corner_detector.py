#!/usr/bin/python3

import numpy as np
import cv2
from util import read_image


def corner_detection(image: np.ndarray):
    """
    Runs corner detection on image using Harris algorithm and Brown algorithm with and without using determinant/trace
    :param image: image to run corner detection on
    """
    blur = cv2.GaussianBlur(image, (5, 5), 7)
    x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    xy = np.multiply(x, y)
    x_squared = np.multiply(x, x)
    y_squared = np.multiply(y, y)
    x_squared_blur = cv2.GaussianBlur(x_squared, (7, 7), 10)
    y_squared_blur = cv2.GaussianBlur(y_squared, (7, 7), 10)
    xy_blur = cv2.GaussianBlur(xy, (7, 7), 10)

    alpha = 0.05  # Good value determined by experimentation
    """
    Corner detection using determinant/trace
    """
    det = np.multiply(x_squared_blur, y_squared_blur) - np.multiply(xy_blur, xy_blur)
    trace = x_squared_blur + y_squared_blur
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = np.divide(det, trace)
    mean[np.isnan(mean)] = 0
    cv2.imwrite("brown_hmean_dettrace.png", mean * (255. / np.max(mean)))
    R_det = det - alpha * np.multiply(trace, trace)
    cv2.imwrite("harris_{}_dettrace.png".format(alpha), R_det * (255. / np.max(R_det)))

    """
    Corner detection without using determinant/trace
    """
    mean = np.zeros_like(image)
    R = np.zeros_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            for ch in range(image.shape[2]):
                M = np.asarray(
                    [[x_squared_blur[r, c, ch], xy_blur[r, c, ch]], [xy_blur[r, c, ch], y_squared_blur[r, c, ch]]])
                lambdas = np.linalg.eigh(M)[0]
                R[r, c, ch] = lambdas[0] * lambdas[1] + alpha * (lambdas[0] + lambdas[1]) ** 2
                h_mean = (lambdas[0] * lambdas[1]) / (lambdas[0] + lambdas[1])
                mean[r, c, ch] = 0 if np.isnan(h_mean) else h_mean
    cv2.imwrite("brown_hmean_eig.png", mean * (255. / np.max(mean)))
    R[R < 1] = 0
    cv2.imwrite("harris_{}_eig.png".format(alpha), R * (255. / np.max(R)))


def run_corner_detection():
    image = read_image('building.jpg')
    corner_detection(image)
