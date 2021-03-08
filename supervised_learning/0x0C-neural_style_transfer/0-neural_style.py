#!/usr/bin/env python3
"""
    Neural Style Transfer
    partially completed after solutions available
"""
import numpy as np
import tensorflow as tf


class NST:
    """ Neural style transfer class """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ class constructor """
        if type(style_image) is not np.ndarray or\
                style_image.shape[2] != 3 or\
                style_image.ndim != 3:
            raise TypeError('style_image must be a' +
                            ' numpy.ndarray with shape (h, w, 3)')
        if type(content_image) is not np.ndarray or\
                content_image.shape[2] != 3 or\
                content_image.ndim != 3:
            raise TypeError('content_image must be a' +
                            ' numpy.ndarray with shape (h, w, 3)')
        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape
        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if type(alpha) not in [int, float] or\
                alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if type(beta) not in [int, float] or\
                beta < 0:
            raise TypeError('beta must be a non-negative number')
        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ scales image dimensions and values to 0-1 """
        if type(image) is not np.ndarray or\
                image.shape[2] != 3 or\
                image.ndim != 3:
            raise TypeError('image must be a ' +
                            'numpy.ndarray with shape (h, w, 3)')
        h, w, c = image.shape
        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))
        new_shape = (h_new, w_new)
        image = np.expand_dims(image, axis=0)
        scaled_image = tf.image.resize_bicubic(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255, 0, 1)

        return scaled_image
