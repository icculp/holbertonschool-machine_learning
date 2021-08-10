#!/usr/bin/env python3
"""
    Data Agumentation project
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def rotate_image(image):
    """ rotate the image """
    return tf.image.rot90(image, k=1, name=None)
