#!/usr/bin/env python3
"""
    Data Agumentation project
"""
import numpy as np
import tensorflow as tf


def pca_color(image, alphas):
    """ perform PCA color change as in AlexNet """
    orig_image = image.numpy().astype(float).copy()

    image = image.numpy().astype(float) / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    image_rs = image.reshape(-1, 3)
    # image_rs shape (640000, 3)

    # center mean
    image_centered = image_rs - np.mean(image_rs, axis=0)

    # paper says 3x3 covariance matrix
    image_cov = np.cov(image_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(image_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation
    # (not once per channel)

    # broad cast to speed things up
    m2[:, 0] = alphas * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_image[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_image /= 255.0
    orig_image = np.clip(orig_image, 0.0, 255.0)

    # orig_image *= 255
    orig_image = orig_image.astype(np.uint8)

    return orig_image
