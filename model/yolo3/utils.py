"""Miscellaneous utility functions."""
import glob
from functools import reduce

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    """Resize image"""

    # target size
    h, w = size

    # resize image
    image = image.resize((w, h), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image)
    return new_image


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


# list of class-names
yolov3_classes = {'person': 0, 'person_group': 1, 'two_wheeler': 2, 'on_rails': 3, 'car': 4, 'truck': 5}


def get_random_data(images_path, annotation, input_shape, max_boxes=20, hue=.1, sat=1.5, val=1.5):
    """random pre-processing for real-time data augmentation"""

    # TODO: PIL vs cv2
    image = None
    # get the random image by name
    for filename in glob.iglob(images_path + '/**/' + annotation.name, recursive=True):
        image = Image.open(filename)
        break
    assert image is not None

    # annotation of the image
    boxes = np.array([np.array(list(map(int, [label.box.x1, label.box.y1, label.box.x2, label.box.y2,
                                              yolov3_classes.get(label.category)]))) for label in annotation.labels])

    # original size
    iw, ih = image.size
    # target size
    h, w = input_shape
    # scale for target size
    scale_x = w / iw
    scale_y = h / ih

    # resize image
    image = image.resize((w, h), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image)
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))

    if len(boxes) > 0:
        np.random.shuffle(boxes)

        # only use first 'max_boxes' number of boxes
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]

        # rescale boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        # flip boxes
        if flip:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # x and y coordinates should be between [0..w] and [0..h]
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h

        # boxes should be at least >1 pixel wide and long
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]

        box_data[:len(boxes)] = boxes

    return image_data, box_data
