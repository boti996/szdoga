import random

import numpy as np
import cv2
import os.path

def get_n(batch_size):
    if batch_size % 8 > 0:
        remainder = 1
    else:
        remainder = 0
    return batch_size // 8 + remainder


def get_next_i(i, dataset):
    length = len(dataset)
    return i + 1 if i < length - 1 else 0


def n_randoms_from_dataset(path, n, dataset):
    indexes = []
    length = len(dataset)
    for i in range(0, n):
        idx = random.randint(0, length - 1)
        indexes.append(dataset[idx])

    images = []
    for i in indexes:
        img = cv2.imread(os.path.join(path, dataset[i])) / 255
        images.append(img)

    # resize bboxes
    height, width, _ = images[0].shape








def yolov3_generator(path, datasets, batch_size):

    on_rails, two_wheeler, person, truck, bdd, others = datasets

    n = get_n(batch_size)

    # TODO: összeg jó legyen nem 8-cal osztható batch_size-ra is
    n_on_rails =    n * 1
    n_two_wheeler = n * 1
    n_person =      n * 1
    n_others =      n * 2
    n_bdd =         n * 3

    while True:
        batch = []
        for i in range(0, n_on_rails):


        for i in range(0, n_two_wheeler):
        for i in range(0, n_person):
        for i in range(0, n_others):
        for i in range(0, n_bdd):





