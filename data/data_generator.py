import random
import numpy as np
from cv2 import cv2

from model.yolo3.utils import get_random_data


def get_n(batch_size):
    if batch_size % 8 > 0:
        remainder = 1
    else:
        remainder = 0
    return batch_size // 8 + remainder


def n_randoms_from_dataset(images_path, n, dataset, input_shape):
    image_data = []
    box_data = []
    length = len(dataset)
    for i in range(0, n):
        idx = random.randint(0, length - 1)
        image, boxes = get_random_data(images_path, dataset[idx], input_shape)
        image_data.append(image)
        box_data.append(boxes)
        # TODO delete
        # for box in boxes:
        #    cv2.rectangle(image, ( int(box[0]), int(box[1]) ), ( int(box[2]), int(box[3]) ), (0, 255, 0), 3)
        # import time
        # cv2.imwrite('/media/boti/Adatok/szemet/' + dataset[idx].name, image * 255)
    return image_data, box_data


def yolov3_generator(images_path, datasets, batch_size, input_shape):

    # TODO: currently mini-batch size must be multipl. of 8
    n = get_n(batch_size)

    # on_rails, two_wheeler, person, truck, bdd, others
    ns = [n * 1, n * 1, n * 1, n * 1, n * 1, n * 3]

    # validation dataset detected
    if len(datasets) == 1:
        ns = [batch_size]

    image_batch = []
    box_batch = []
    # get n images into a mini-batch
    for i in range(0, len(datasets)):
        n_image_batch, n_box_batch = n_randoms_from_dataset(images_path, ns[i], datasets[i], input_shape)
        image_batch.extend(n_image_batch)
        box_batch.extend(n_box_batch)

    random.shuffle(image_batch)
    random.shuffle(box_batch)
    return image_batch[:batch_size], box_batch[:batch_size]


