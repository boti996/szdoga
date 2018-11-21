# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow
from cv2 import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from model.yolo3.model import yolo_eval, yolo_body
from model.yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        # "model_path": '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/mod_trained_weights_stage_1.h5',
        "model_path": '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/trained_weights_stage_1.h5',
        # "model_path": '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/trained_weights_final.h5',
        "anchors_path": '/home/boti/Workspace/PyCharmWorkspace/training_data/yolo_anchors.txt',
        "classes_path": '/home/boti/Workspace/PyCharmWorkspace/training_data/yolov3_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (608, 608),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, mod_mask=(False, False, False, False, False), pruning_mtx=(-1, -1, -1, -1, -1), **kwargs):
        # TODO: delete this line
        self.pruning_mtx = pruning_mtx
        self.mod_mask = mod_mask

        self.model_image_size = (608, 608)
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        image_input = Input(shape=(None, None, 3))
        self.yolo_model = yolo_body(image_input, num_anchors // 3, num_classes, mod=False,
                                    pruning_mtx=self.pruning_mtx, mod_mask=self.mod_mask)  # load_model(model_path, compile=False)

        self.yolo_model.summary()

        self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True)  # make sure model, anchors and classes match

        assert self.yolo_model.layers[-1].output_shape[-1] == \
               num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, gt_boxes):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(self.model_image_size))
        else:
            h, w, _ = image.shape
            new_image_size = (w - (w % 32),
                              h - (h % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        start = timer()

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [boxed_image.shape[1], boxed_image.shape[0]],
                K.learning_phase(): 0
            })

        end = timer()
        duration = end - start
        # print(duration)

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # TODO: relative path
        h, w, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 0

        thickness = 4

        for box in gt_boxes:
            # draw ground truth boxes
            cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), (255, 255, 255), thickness)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            label_size = np.asarray(cv2.getTextSize(label, font, font_scale, font_thickness)[0])

            top, left, bottom, right = box

            h, w, _ = image.shape

            # rescale
            scale_x = w / self.model_image_size[0]
            scale_y = h / self.model_image_size[1]
            left, right = left * scale_x, right * scale_x
            top, bottom = top * scale_y, bottom * scale_y

            # should stay in image
            left = w if left > w else left
            right = w if right > w else right
            top = h if top > h else top
            bottom = h if bottom > h else bottom

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
            right = min(w, np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # save changes
            out_boxes[i, :4] = top, left, bottom, right

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # draw detection boxes & class name texts
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)
            cv2.rectangle(image, tuple(text_origin - [0, int(label_size[1] * 1.5)]), tuple(text_origin + label_size), self.colors[c], thickness=cv2.FILLED)
            cv2.putText(image, label, tuple(text_origin), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        return duration, image, out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = frame
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
