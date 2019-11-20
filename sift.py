#!/usr/bin/python3

import cv2
import matplotlib.pyplot as plt
import numpy as np

from util import read_image


def sample_sift_features(image1: str = "sample1.jpg", image2: str = "sample2.jpg") -> tuple:
    """
    Returns keypoints and features of image1 and image2
    :return: (image1 keypoints, image2 keypoints), (image1 features, image2 features)
    """
    sift = cv2.xfeatures2d.SIFT_create()
    sample1 = cv2.cvtColor(read_image(image1), cv2.COLOR_BGR2GRAY)
    sample2 = cv2.cvtColor(read_image(image2), cv2.COLOR_BGR2GRAY)
    k1, f1 = sift.detectAndCompute(sample1, None)
    k2, f2 = sift.detectAndCompute(sample2, None)
    # NOTE keypoints are (x, y) format
    return (k1, k2), (f1, f2)


def sift_match(features: tuple, max_ratio: float, dist_function) -> np.ndarray:
    """
    Returns ratios and distances for SIFT keypoints/features
    :param features: sift features of images
    :param max_ratio: maximum ratio between images to count as a match
    :param dist_function: distance function
    """
    ratio_and_dist = np.zeros((len(features[0]), 2)) * np.nan
    indices = np.zeros((len(features[0]), 2), dtype=int) * np.nan
    map_idx = 0
    for i in range(len(features[0])):
        diff = features[0][i] - features[1]
        distances = np.asarray([abs(dist_function(diff[j])) for j in range(diff.shape[0])])
        best = np.partition(distances, 1)
        ratio = best[0] / best[1]
        if ratio < max_ratio:
            ratio_and_dist[map_idx] = [ratio, best[0]]
            indices[map_idx] = [i, np.argmin(distances)]
            map_idx += 1
        if i % 100 == 0:
            print('{}/{}'.format(i, len(features[0])))
    return ratio_and_dist[~np.isnan(ratio_and_dist).all(1)], indices[~np.isnan(indices).all(1)].astype(int)


def test_sift_ratios(ratio_and_dist: np.ndarray, ratios=np.arange(0.05, 1, 0.05), outfile: str = "ratios.png",
                     dpi: int = 1000, dist_fn_name: str = "L2 Norm"):
    counts = [np.count_nonzero(np.where(ratio_and_dist[:, 0] <= i)) for i in ratios]
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.plot(ratios, counts)
    plt.xlabel("Threshold ratio")
    plt.ylabel("Feature matches")
    plt.title("# feature matches for different threshold ratios ({})".format(dist_fn_name))
    plt.savefig(outfile, dpi=dpi)


def visualize_top_n_features(n: int, ratio_and_dist: np.ndarray, indices: np.ndarray, keypoints) -> np.ndarray:
    """
    Returns an image with lines between sample1 and sample2 for the top n keypoints according to ratio
    :param n: number of keypoints
    :param ratio_and_dist: ndarray where each row is [ratio, dist]
    :param indices: ndarray where row order is same as ratio_and_dist and
                    each row is [coords in sample1, coords in sample2]
    :param keypoints: tuple of (keypoints in sample1, keypoints in sample2)
    :return: ndarray of image with lines between sample1 and sample2 for the top n keypoints according to ratio
    """
    sample1 = cv2.cvtColor(read_image("sample1.jpg"), cv2.COLOR_BGR2GRAY)
    sample2 = cv2.cvtColor(read_image("sample2.jpg"), cv2.COLOR_BGR2GRAY)
    image = np.concatenate((sample1, sample2), axis=1)

    top_indices = ratio_and_dist[:, 0].argsort()
    for i in range(n):
        pt1 = keypoints[0][indices[top_indices[i], 0]].pt
        pt1 = (round(pt1[0]), round(pt1[1]))
        pt2 = keypoints[1][indices[top_indices[i], 1]].pt
        pt2 = (round(pt2[0]) + sample1.shape[1], round(pt2[1]))  # Keypoints are (x, y)
        cv2.line(image, pt1, pt2, (255, 255, 255), thickness=3)
    return image


if __name__ == '__main__':
    sift_features = sample_sift_features()
    norm1ratio, norm1index = sift_match(sift_features[1], 1, lambda x: np.linalg.norm(x, 1))
    norm2ratio, norm2index = sift_match(sift_features[1], 1, np.linalg.norm)
    norm3ratio, norm3index = sift_match(sift_features[1], 1, lambda x: np.linalg.norm(x, 3))
    test_sift_ratios(norm2ratio)
    cv2.imwrite('matches.png', visualize_top_n_features(10, norm2ratio, norm2index, sift_features[0]))
