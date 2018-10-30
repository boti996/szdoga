import random

import numpy

from model.yolo3.utils import get_random_data


def get_n(batch_size):
    if batch_size % 8 > 0:
        remainder = 1
    else:
        remainder = 0
    return batch_size // 8 + remainder


def n_randoms_from_dataset(path, n, dataset,input_shape):
    image_data = []
    box_data = []
    length = len(dataset)
    for i in range(0, n):
        idx = random.randint(0, length - 1)
        image, box = get_random_data(path, dataset[idx], input_shape, random=True)
        image_data.append(image)
        box_data.append(box)
    return image_data, box_data


def yolov3_generator(path, datasets, batch_size, input_shape):

    # TODO: összeg jó legyen nem 8-cal osztható batch_size-ra is
    n = get_n(batch_size)
    # on_rails, two_wheeler, person, truck, bdd, others
    ns = [n * 1, n * 1, n * 1, n * 1, n * 1, n * 3]

    # validation dataset detected
    if len(datasets) == 1:
        ns = [batch_size]

    image_batch = []
    box_batch = []
    print(numpy.array(datasets).shape)
    for i in range(0, len(datasets)):
        n_image_batch, n_box_batch = n_randoms_from_dataset(path, ns[i], datasets[i], input_shape)
        image_batch.extend(n_image_batch)
        box_batch.extend(n_box_batch)

    return image_batch, box_batch


