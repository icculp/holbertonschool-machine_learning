#!/usr/bin/env python3
"""
    Data Agumentation project
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def crop_image(image, size):
    """ flips the image """
    return tf.image.flip_left_right(image)
