#!/usr/bin/env python3
"""
    Data Agumentation project
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def shear_image(image, intensity):
    """ sheers the image """
    return tf.compat.v1.keras.preprocessing.image.apply_affine_transform(
        image.numpy(), shear=intensity)
