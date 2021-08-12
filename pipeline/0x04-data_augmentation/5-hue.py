#!/usr/bin/env python3
"""
    Data Agumentation project
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def change_hue(image, delta):
    """ change hue of the image """
    return tf.image.adjust_hue(image, delta)
