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

    def process_outputs(self, outputs, image_size):
        """ Process predictions from darknet
            outputs is list of ndarrays containing predictions for single image
                each output shape (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
                    4 => (t_x, t_y, t_w, t_h) 1=> box confidence
                    classes =? class probabilities for all classes
            image_size ndarray image original size (img_h, img_w)
            Returns: tuple (boxes, box_confidences, box_class_probs)
                boxes (grid_height, grid_width, anchor_boxes, 4)
                    4=> (x1, y1, x2, y2) (boundary box relative to original)
                box_confidences ndarray (grid_height, grid_width, anchor_boxes, 1)
                box_class_probs ndarray (grid_height, grid_width, anchor_boxes, classes)
        """
        print(outputs)
        for output in outputs:
            for detection in output:
                scores = detection