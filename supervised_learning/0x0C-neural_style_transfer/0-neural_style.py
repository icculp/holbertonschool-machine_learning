#!/usr/bin/env python3
"""
    Neural Style Transfer
    partially completed after solutions available
"""
import numpy as np
import tensorflow as tf


class NST():
    """ Neural style transfer class """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ class constructor """
        if type(style_image) is not np.ndarray or
                style_image.shape[2] != 3 or
                style_image.ndmin != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if type(content_image) is not np.ndarray or
                style_image.shape[2] != 3 or
                style_image.ndmin != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if type(alpha) is not in [int, float] or
                alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if type(beta) is not in [int, float] or
                beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ scales the image to 