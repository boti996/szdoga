"""Helper classes for serialization & serialization, deserialization"""
import json


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


def encode_annot_array(images, filename, postfix=''):
    images_json = []
    for image in images:
        images_json.append(image.encode_json())
    index = filename.find('.json')
    with open(filename[:index] + postfix + '.json', 'w') as outfile:
        json.dump(images_json, outfile, indent=4, separators=(',', ': '))


def decode_annot_array(filename):
    with open(filename) as json_file:
        j_images = json.load(json_file)

    images = []
    # IMAGES
    for j_image in j_images:

        labels = []
        # LABELS
        for j_label in j_image['labels']:
            # LABEL
            j_box = j_label['box']
            box = CommonBox((j_box['x1'], j_box['y1']), (j_box['x2'], j_box['y2']))

            label = CommonLabel(j_label['category'], box)
            labels.append(label)

        image = CommonImage(j_image['name'], labels)
        images.append(image)

    return images