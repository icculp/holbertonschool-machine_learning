#!/usr/bin/env python3
"""
    Object detection
"""
import cv2
import glob
import numpy as np
import tensorflow.keras as K


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
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Filtering outputs
            boxes list of ndarray (grid_height, grid_width, anchor_boxes, 4)
            box_confidences list of ndarray
                (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs list of ndarray
                (grid_height, grid_width, anchor_boxes, classes)

            Returns tuple (filtered_boxes, box_classes, box_scores)
                filtered_boxes: a numpy.ndarray of shape (?, 4)
                box_classes: a numpy.ndarray of shape (?,)
                box_scores: a numpy.ndarray of shape (?)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, box in enumerate(boxes):
            bc = box_confidences[i]
            bcp = box_class_probs[i]
            bs = bc * bcp
            bcl = np.argmax(bs, axis=-1)
            bcs = np.max(bs, axis=-1)
            idx = np.where(bcs > self.class_t)

            filtered_boxes.append(box[idx])
            box_classes.append(bcl[idx])
            box_scores.append(bcs[idx])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_confidences = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)
        return (filtered_boxes, box_confidences, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ selects the best bounding boxes for each potential object
            filtered_boxes: a numpy.ndarray of shape (?, 4)
            box_classes: a numpy.ndarray of shape (?,)
                class number for class filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?)
                box scores for each box in filtered_boxes, respectively
            Returns a tuple of (box_predictions,
                                predicted_box_classes, predicted_box_scores):
                box_predictions: ndarray (?, 4) predicted bounding boxes
                    ordered by class and box score
                predicted_box_classes: ndarray (?,)  class number for
                    box_predictions ordered by class and box score
                predicted_box_scores: ndarray (?) box scores for
                    box_predictions ordered by class and box score
        """
        '''
        print("filtered_boxes", filtered_boxes.shape)
        print("box_classes", box_classes.shape)
        print("box_scores", box_scores.shape)
        '''
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []
        pick = []

        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            '''print(idxs)'''
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
            '''print(152)'''
            for pos in range(last):
                j = idxs[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                overlap = float(w * h) / area[j]
            if overlap > self.nms_t:
                suppress.append(pos)
            idxs = np.delete(idxs, suppress)
        '''print('picckkkkkk', pick)'''
        box_predictions = filtered_boxes[pick]
        predicted_box_classes = box_classes[pick]
        predicted_box_scores = box_scores[pick]

        box_predictions.sort()
        predicted_box_classes.sort()
        '''print(type(predicted_box_classes))
        print("shape is", predicted_box_classes.shape)'''
        predicted_box_scores.shape
        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """ Loads images from folder_path """
        images = []
        image_paths = glob.glob(folder_path + "/*")
        images = [cv2.imread(img) for img in image_paths]
        return (images, image_paths)

    def preprocess_images(self, images):
        """ Resize the images with inter-cubic interpolation
            Rescale all images to have pixel values in the range [0, 1]
            Returns a tuple of (pimages, image_shapes)
        """
        pimages = []
        image_shapes = []
        input_w, input_h = self.model.inputs[0].shape.as_list()[1:3]
        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(image, dsize=(input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255)
        pimages = np.asarray(pimages)
        image_shapes = np.asarray(image_shapes)
        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """ Show the boxes with cv2
            image: a numpy.ndarray unprocessed image
            boxes: a numpy.ndarray boundary boxes for the image
            box_classes: a numpy.ndarray  class indices for each box
            box_scores: a numpy.ndarray  box scores for each box
            file_name: the file path where the original image is stored
            Displays the image with all boundary boxes,
                class names, and box scores (see example below)
            Boxes should be drawn as with a blue line of thickness 2
            Class names and box scores drawn above each box in red
            Box scores should be rounded to 2 decimal places
            Text should be written 5 pixels above the top left corner
            Text should be written in FONT_HERSHEY_SIMPLEX
            Font scale should be 0.5
            Line thickness should be 1
            You should use LINE_AA as the line type
            The window name should be the same as file_name
            If the s key is pressed:
            The image should be saved in the directory detections,
                located in the current directory
            If detections does not exist, create it
            The saved image should have the file name file_name
            The image window should be closed
            If any key besides s is pressed,
                the image window should be closed without saving
        """
        #print(type(image))
        #image, _ = self.preprocess_images([image])
        #cv2.imshow("thereee", image)
        """
        def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
        """
        print(boxes)
        for i in range(len(boxes)):
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            #color = [int(c) for c in colors[classIDs[i]]]
            #cv2.rectangle()
            cv2.rectangle(img=image, rec=(x, y, x + w, y + h), color='blue', thickness=2)
            text = "{} {:.2f}".format(box_classes[i], box_scores[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, "red", 1, 'LINE_AA')

        cv2.imshow(file_name, image)