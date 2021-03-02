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
            nms_t float IOU threshold for non-max suppression
            anchors ndarray (outputs, anchor_boxes, 2) 2=> anchor dimensions
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
                each output shape (grid_height, grid_width,
                anchor_boxes, 4 + 1 + classes)
                    4 => (t_x, t_y, t_w, t_h) 1=> box confidence
                    classes => class probabilities for all classes
            image_size ndarray image original size (img_h, img_w)
            Returns: tuple (boxes, box_confidences, box_class_probs)
                boxes (grid_height, grid_width, anchor_boxes, 4)
                    4=> (x1, y1, x2, y2) (boundary box relative to original)
                box_confidences ndarray (grid_height, grid_width,
                                         anchor_boxes, 1)
                box_class_probs ndarray (grid_height,
                                         grid_width, anchor_boxes, classes)
        """
        height, width = image_size
        '''print(outputs)'''
        box_confidences = []
        boxes = []
        box_class_probs = []

        def sigmoid(x):
            ''' sigmoid function '''
            return 1 / (1 + np.exp(-x))

        for output in range(len(outputs)):
            anchors = self.anchors[output]
            grid_h, grid_w = outputs[output].shape[0:2]
            t_xy = outputs[output][..., :2]
            t_wh = outputs[output][..., 2:4]
            box_confidence = np.expand_dims(sigmoid(outputs[output][..., 4]),
                                            axis=-1)
            box_class_prob = sigmoid(outputs[output][..., 5:])
            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.inputs[0].shape.as_list()[1:3]
            ''' h/w flipped'''
            grid = np.tile(np.indices((grid_w, grid_h)).T,
                           anchors.shape[0]).reshape((grid_h, grid_w) +
                                                     anchors.shape)
            ''' bx & by => b_xy '''
            b_xy = (sigmoid(t_xy) + grid) / [grid_w, grid_h]

            ''' x1, y1, x2, y2 '''
            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)

            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)
        '''
        boxes = np.asarray(boxes)  # , dtype=np.float32)
        box_confidences = np.asarray(box_confidences)  # , dtype=np.float32)
        box_class_probs = np.asarray(box_class_probs)  # , dtype=np.float32)
        '''
        return (boxes, box_confidences, box_class_probs)

        '''
        for output in range(len(outputs)):
            anchors = self.anchors[output]
            for col in range(output):
                for row in range(col):
                    for box in range(col):
                        a2 += 1
                        print(col.shape)
                        print(output.shape)
                        tx = box[0]
                        ty = box[1]
                        tw = box[2]
                        th = box[3]
                        bx = sigmoid(tx) + row
                        by = sigmoid(ty) + col
                        bw = pw * exp(tw)
                        bh = ph * exp(th)
                        scores = output[5:]
                        class_id = np.argmax(scores)
                        confidence = sigmoid(output[4])
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        box_confidences.append(confidence)
            '''
        '''
        #output = output.ravel
        #for detection in output[:, :, , :]:
        #print(detection.shape)
        scores = output[:, :, :, 5:]
        #class_id = np.argmax(scores)
        confidence = sigmoid(output[:, :, : 4])#scores[class_id]
        if confidence > self.nms_t:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w / 2
            y = center_y - h / 2
            box_confidences.append(confidence)'''
