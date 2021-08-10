#!/usr/bin/env python3
"""
    Data Agumentation project
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def change_brightness(image, max_delta):
    """ change brightness of the image """
    return tf.image.adjust_brightness(image, max_delta)
