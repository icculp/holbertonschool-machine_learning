#!/usr/bin/env python3
"""
    Neural Style Transfer
    partially completed after solutions available
"""
import numpy as np
import tensorflow as tf


Class NST():
    """ Neural style transfer class """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ class constructor """
        if type(style_image) is not ndarray
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
        
