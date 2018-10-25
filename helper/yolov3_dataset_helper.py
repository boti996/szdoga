import glob
import os
import json
import sys

import jsonpickle


# Helper classes for serialization
class CommonList(object):

    def __init__(self, image_array):
        self.images = image_array

    def add_image(self, image):
        self.images.append(image)


class CommonImage(object):
    def __init__(self, name, labels):
        self.name = name
        self.labels = labels

    def __str__(self):
        stri = 'name: ' + self.name + ' labels: \n'
        for label in self.labels:
            stri += '\t' + str(label) + '\n'
        return stri

    def encode_json(self):
        labels_json = []
        for label in self.labels:
            labels_json.append(label.encode_json())
        return {'name': self.name, 'labels': labels_json}


class CommonLabel(object):
    def __init__(self, category, box):
        self.category = category
        self.box = box

    def __str__(self):
        return 'category: ' + str(self.category) + ' box: ' + str(self.box)

    def encode_json(self):
        return {'category': self.category, 'box': self.box.encode_json()}



class CommonBox(object):
    def __init__(self, p1, p2):
        self.x1, self.y1 = p1
        self.x2, self.y2 = p2

    def __str__(self):
        return 'p1: ' + str(self.x1) + ', ' + str(self.y1) + ' p2: ' + str(self.x2) + ', ' + str(self.y2)

    def encode_json(self):
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}


def encode_image_array(images, filename):
    images_json = []
    for image in images:
        images_json.append(image.encode_json())
    index = filename.find('.json')
    with open(filename[:index] + '_common.json', 'w') as outfile:
        json.dump(images_json, outfile, indent=4, separators=(',', ': '))


def bdd100k_to_common(path):
    """ Read original json files and reformat them into the common format"""
    labels_path = os.path.join(path, "labels")

    # JSON FILES
    for filename in glob.iglob(labels_path + '\**\*.json', recursive=True):
        with open(filename) as json_file:

            j_images = json.load(json_file)
            images = []
            # IMAGES
            for j_image in j_images:

                labels = []
                # LABELS
                for j_label in j_image['labels']:
                    try:
                        j_box = j_label['box2d']
                    except KeyError as e:
                        continue;

                    box = CommonBox((j_box['x1'], j_box['y1']), (j_box['x2'], j_box['y2']))

                    label = CommonLabel(j_label['category'], box)
                    labels.append(label)

                image = CommonImage(j_image['name'], labels)
                images.append(image)

            # print(images[0])
            encode_image_array(images, filename)


def wd_to_common(path):
    """ Read original json files and reformat them into the common format"""
    labels_path = os.path.join(path, "labels")

    images = []
    # IMAGES
    for filename in glob.iglob(labels_path + '\**\*.json', recursive=True):
        with open(filename) as json_file:
            j_image = json.load(json_file)

            labels = []
            # LABELS
            for j_label in j_image['objects']:
                # POLYGON's XYs
                xmin = sys.maxsize
                ymin = sys.maxsize
                xmax = -sys.maxsize - 1
                ymax = -sys.maxsize - 1
                for j_xy in j_label['polygon']:
                    x, y = j_xy[0], j_xy[1]
                    if x > xmax:
                        xmax = x
                    elif x < xmin:
                        xmin = x
                    if y > ymax:
                        ymax = y
                    elif y < ymin:
                        ymin = y

                box = CommonBox((xmin, ymin), (xmax, ymax))

                label = CommonLabel(j_label['label'], box)
                labels.append(label)

        base = os.path.basename(filename)
        index = base.find('_polygons.json')
        image = CommonImage(os.path.splitext(base)[0][:index] + '.jpg', labels)
        images.append(image)

    # print(images[0])
    encode_image_array(images, os.path.join(labels_path, 'wd.json'))


def cp_to_common(path, labels_folder=True):
    """ Read original json files and reformat them into the common format"""
    if labels_folder:
        labels_path = os.path.join(path, "labels")
    else:
        labels_path = path

    images = []
    # IMAGES
    for filename in glob.iglob(labels_path + '\**\*.json', recursive=True):
        with open(filename) as json_file:
            j_image = json.load(json_file)

            labels = []
            # LABELS
            for j_label in j_image['objects']:

                j_box = j_label['bbox']
                box = CommonBox((j_box[0], j_box[1]), (int(j_box[0]) + int(j_box[2]), int(j_box[1]) + int(j_box[3])))

                label = CommonLabel(j_label['label'], box)
                labels.append(label)

        base = os.path.basename(filename)
        index = base.find('_gtBboxCityPersons.json')
        image = CommonImage(os.path.splitext(base)[0][:index] + '.png', labels)
        images.append(image)

    # print(images[0])
    encode_image_array(images, os.path.join(labels_path, 'cp.json'))


def cs_to_common(path, labels_folder=True):
    """ Read original json files and reformat them into the common format"""
    if labels_folder:
        labels_path = os.path.join(path, "labels")
    else:
        labels_path = path

    images = []
    # IMAGES
    for filename in glob.iglob(labels_path + '\**\*.json', recursive=True):
        with open(filename) as json_file:
            j_image = json.load(json_file)

            labels = []
            # LABELS
            for j_label in j_image['objects']:
                # POLYGON's XYs
                xmin = sys.maxsize
                ymin = sys.maxsize
                xmax = -sys.maxsize - 1
                ymax = -sys.maxsize - 1
                for j_xy in j_label['polygon']:
                    x, y = j_xy[0], j_xy[1]
                    if x > xmax:
                        xmax = x
                    elif x < xmin:
                        xmin = x
                    if y > ymax:
                        ymax = y
                    elif y < ymin:
                        ymin = y

                box = CommonBox((xmin, ymin), (xmax, ymax))

                label = CommonLabel(j_label['label'], box)
                labels.append(label)

        base = os.path.basename(filename)
        index = base.find('_gtFine_polygons.json')
        image = CommonImage(os.path.splitext(base)[0][:index] + '.png', labels)
        images.append(image)

    # print(images[0])
    encode_image_array(images, os.path.join(labels_path, 'cs.json'))



if __name__ == '__main__':
    # bdd100k_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\bdd')

    # wd_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\wd')

    cp_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons')
    cp_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons\\labels\\train', labels_folder=False)
    cp_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons\\labels\\val', labels_folder=False)

    # cs_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes')
    # cs_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes\\labels\\train', labels_folder=False)
    # cs_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes\\labels\\val', labels_folder=False)





# path: should be the bdd100k root folder
# labels: bdd100k/labels
# images: bdd100k/images
def load_bdd100k(path):
    # TODO
    raise NotImplementedError
