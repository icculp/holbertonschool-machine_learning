#!/usr/bin/env python3
"""
    Neural Style Transfer
    partially completed after solutions available
"""
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


class NST():
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
                style_image.shape[2] != 3 or\
                style_image.ndim != 3:
            raise TypeError('content_image must be a' +
                            ' numpy.ndarray with shape (h, w, 3)')
        if type(alpha) not in [int, float] or\
                alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if type(beta) not in [int, float] or\
                beta < 0:
            raise TypeError('beta must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """ scales image dimensions and values to 0-1 """
        if type(image) is not np.ndarray or\
                image.shape[2] != 3 or\
                image.ndim != 3:
            raise TypeError('image must be a ' +
                            'numpy.ndarray with shape (h, w, 3)')
        h, w, _ = image.shape
        max_dim = 512
        maximum = max(h, w)
        scale = max_dim / maximum
        new_shape = (int(h * scale), int(w * scale))
        image = np.expand_dims(image, axis=0)
        scaled_image = tf.image.resize_bicubic(image, new_shape)
        scaled_image = tf.clip_by_value(scaled_image / 255, 0, 1)

        return scaled_image

    def load_model(self):
        """ loads keras model """
        vgg = tf.keras.applications.VGG19(include_top=False)
        x = vgg.input
        style_outputs = []
        content_output = None
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                ps = layer.pool_size
                layer = tf.keras.layers.AveragePooling2D(pool_size=ps,
                                                         strides=layer.strides,
                                                         padding=layer.padding,
                                                         name=layer.name)
                x = layer(x)
            else:
                x = layer(x)
                if layer.name in self.style_layers:
                    style_outputs.append(layer.output)
                if layer.name == self.content_layer:
                    content_output = layer.output
            layer.trainable = False

        outputs = style_outputs + [content_output]
        model = tf.keras.models.Model(vgg.input, outputs)
        self.model = model
