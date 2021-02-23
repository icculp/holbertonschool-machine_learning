#!/usr/bin/env python3
"""
    Object detection
"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Yolo v3 for object detection """

    def __init__(self, model_path, classes_path, class_t,
                 nms_t, anchors):
        """ Yolo constructor
            model_path path where Darknet model stored
            classes_path path for list of class names
            class_t float representing box score threshold for initial filter
            nms_t float IOS threshold for non-max suppression
            anchors ndarray (outputs, anchor_boxes, 2) (2> anchor dimensions)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path) as f:
            cn = f.read().split('\n')
            cn.pop()
        self.class_names = cn
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
