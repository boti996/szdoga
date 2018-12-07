import glob
import os
from enum import Enum
from timeit import default_timer as timer

import numpy as np
from cv2 import cv2
from numpy import random

from data.train_model import ModelType
from data.yolov3_load_dataset import YoloV3DataLoader
from model.yolo3.utils import yolov3_classes
from model.yolo3.yolo_eval import YOLO
import keras.backend as K
import sys

def transform_box_format_pred(box):
    """y1,x1,y2,x2 to x1, y1, w, h"""
    y1, x1, y2, x2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def transform_box_format_gt(box):
    """x1,y1,x2,y2 to x1, y1, w, h"""
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    return [x1, y1, x2 - x1, y2 - y1]


def get_iou(pred_box, gt_box):
    b1_xy = np.array(pred_box[:2])
    b1_wh = np.array(pred_box[2:4])
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2_xy = np.array(gt_box[:2])
    b2_wh = np.array(gt_box[2:4])
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = [max(b1_mins[0], b2_mins[0]), max(b1_mins[1], b2_mins[1])]
    intersect_maxes = [min(b1_maxes[0], b2_maxes[0]), min(b1_maxes[1], b2_maxes[1])]

    intersect_wh = [max(intersect_maxes[0] - intersect_mins[0], 0.), max(intersect_maxes[1] - intersect_mins[1], 0.)]
    intersect_area = intersect_wh[0] * intersect_wh[1]
    b1_area = b1_wh[0] * b1_wh[1]
    b2_area = b2_wh[0] * b2_wh[1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def get_TPs(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5, printing=True):
    n_tp = 0
    for i in range(0, len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_class = pred_classes[i]

        for j in range(0, len(gt_boxes)):
            gt_box = gt_boxes[j]
            gt_class = gt_classes[j]

            if get_iou(pred_box, gt_box) > iou_threshold:

                if pred_class == gt_class:
                    n_tp += 1
                    # if printing:
                        # print('iou: ' + str(get_iou(pred_box, gt_box)))
                    break
    return n_tp


def get_FNs(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold=0.5):
    return len(gt_boxes) - get_TPs(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold, printing=False)


def load_dataset(dataset_path, n_images, input_shape=(608, 608), model_type=ModelType.YOLO_V3.value):
    my_random_seed = 1337

    # choose dataset loader according to the current network type
    dataset_loader = None
    if ModelType(model_type) == ModelType.YOLO_V3:
        dataset_loader = YoloV3DataLoader()
    assert dataset_loader is not None

    # get dataset
    validation = dataset_loader.load_dataset(dataset_path, input_shape, random_seed=my_random_seed)['val']

    # reformat into one array
    val_dataset = []
    for dataset in validation:
        val_dataset.extend(dataset)

    # shuffle with fix seed
    random.seed(my_random_seed)
    random.shuffle(val_dataset)

    # get n_images number of samples
    return val_dataset[:n_images]


def evaluate(val_dataset, dataset_path, out_path, model_type=ModelType.YOLO_V3.value,
             pruning=None, mod_mask=(0, 0, 0, 0, 0), model_path=None):
    """Select n_images number of images from validation dataset and return evaluation statistics"""
    model = None
    if ModelType(model_type) == ModelType.YOLO_V3:
        model = YOLO(mod_mask=mod_mask, pruning=pruning, model_path=model_path)
    assert model is not None

    st = timer()

    idx = 0
    avg_precision = 0
    avg_recall = 0
    avg_duration = 0

    for data in val_dataset:
        image = None
        # get the random image by name
        for filename in glob.iglob(dataset_path + '/**/' + data.name, recursive=True):
            image = cv2.imread(filename)
            break
        assert image is not None

        gt_boxes = []
        gt_classes = []
        for label in data.labels:
            gt_boxes.append(label.box)
            gt_classes.append(yolov3_classes[label.category])

        duration, pred_image, pred_boxes, pred_scores, pred_classes = model.detect_image(image, gt_boxes)

        for i in range(0, len(pred_boxes)):
            pred_boxes[i] = transform_box_format_pred(pred_boxes[i])

        for i in range(0, len(gt_boxes)):
            gt_boxes[i] = transform_box_format_gt(gt_boxes[i])

        curr_tp = get_TPs(pred_boxes, pred_classes, gt_boxes, gt_classes)
        avg_precision += curr_tp / len(pred_boxes) if len(pred_boxes) > 0 else 1
        curr_fn = get_FNs(gt_boxes, gt_classes, pred_boxes, pred_classes)
        avg_recall += curr_tp / (curr_tp + curr_fn) if curr_fn > 0 else 1

        avg_duration += duration

        # save every 100. images
        if idx % 10 == 0:
            # cv2.imshow('evaluation', pred_image)
            cv2.imwrite(os.path.join(out_path, data.name), pred_image)

        idx += 1

    length = len(val_dataset)
    mean_avg_precision = avg_precision / length
    mean_avg_recall = avg_recall / length
    avg_duration /= length

    print(timer() - st)

    return mean_avg_precision, mean_avg_recall, avg_duration


def normal(dataset, model_path=None):
    mean_avg_precision, mean_avg_recall, avg_duration = evaluate(dataset, '/media/boti/Adatok/Datasets-pc/',
                                                                 '/media/boti/Adatok/Datasets-pc/evaluation',
                                                                 model_path=model_path)
    print('mean_avg_precision: ' + str(mean_avg_precision))
    print('mean_avg_recall: ' + str(mean_avg_recall))
    print('avg_duration: ' + str(avg_duration))

    K.clear_session()


def with_modification(dataset, model_mask, model_path=None):
    mean_avg_precision, mean_avg_recall, avg_duration = evaluate(dataset, '/media/boti/Adatok/Datasets-pc/',
                                                                 '/media/boti/Adatok/Datasets-pc/evaluation',
                                                                 mod_mask=model_mask, model_path=model_path)
    print('mean_avg_precision: ' + str(mean_avg_precision))
    print('mean_avg_recall: ' + str(mean_avg_recall))
    print('avg_duration: ' + str(avg_duration))

    K.clear_session()


def with_pruning(dataset, model_mask, pruning, model_path=None):
    mean_avg_precision, mean_avg_recall, avg_duration = evaluate(dataset, '/media/boti/Adatok/Datasets-pc/',
                                                                 '/media/boti/Adatok/Datasets-pc/evaluation',
                                                                 mod_mask=model_mask, pruning=pruning,
                                                                 model_path=model_path)
    print('mean_avg_precision: ' + str(mean_avg_precision))
    print('mean_avg_recall: ' + str(mean_avg_recall))
    print('avg_duration: ' + str(avg_duration))

    K.clear_session()


def pruning_one_layer(dataset, model_mask, model_path=None):
    n_blocks = 1 + 2 + 8 + 8 + 4

    for i in range(0, n_blocks + 1):
        print(str(i) + '. block was pruned:')
        with_pruning(dataset, model_mask, [i], model_path)


if __name__ == "__main__":

    # LOAD DATASET
    val_dataset = load_dataset('/media/boti/Adatok/Datasets-pc/', 101)

    # sys.stdout = open('../logs/evaluation_log.txt', 'w')

    # normal(val_dataset, '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/trained_weights_stage_1.h5')
    # normal(val_dataset, '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/trained_weights_final.h5')
    # normal(val_dataset, '/home/boti/Workspace/PyCharmWorkspace/szdoga/logs/train_stage1/trained_weights_stage_1.h5')
    normal(val_dataset, '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/trained_weights_stage_11543915433.6909509.h5')
    # with_modification(val_dataset, (0, 0, 0, 0, 1))
    # pruning_one_layer(val_dataset, (0, 0, 0, 0, 0))
    # with_modification(val_dataset, (0, 0, 0, 1, 4), '/home/boti/Workspace/PyCharmWorkspace/szdoga/trained_weights/mod_trained_weights_stage_3_0.h5')
