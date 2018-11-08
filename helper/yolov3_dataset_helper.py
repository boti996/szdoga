import glob
import os
import json
import sys
from math import sqrt
import cv2
from data.serialization import CommonImage, CommonLabel, CommonBox, encode_annot_array, decode_annot_array


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
                    except KeyError:
                        continue

                    box = CommonBox((j_box['x1'], j_box['y1']), (j_box['x2'], j_box['y2']))

                    label = CommonLabel(j_label['category'], box)
                    labels.append(label)

                image = CommonImage(j_image['name'], labels)
                images.append(image)

            # print(images[0])
            encode_annot_array(images, filename, '_common')


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
    encode_annot_array(images, os.path.join(labels_path, 'wd_common.json'))


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
    encode_annot_array(images, os.path.join(labels_path, 'cp_common.json'))


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
    encode_annot_array(images, os.path.join(labels_path, 'cs_common.json'))


def kitti_to_comon(path):
    labels_path = os.path.join(path, "labels")

    images = []
    # IMAGES
    for filename in glob.iglob(labels_path + '\**\*.txt', recursive=True):
        with open(filename, "r") as f:

            labels = []
            # LABELS
            for line in f:
                label = line.split(' ')
                box = CommonBox((label[4], label[5]), (label[6], label[7]))

                label = CommonLabel(label[0], box)
                labels.append(label)

        base = os.path.basename(filename)
        index = base.find('.txt')
        image = CommonImage(os.path.splitext(base)[0][:index] + '.png', labels)
        images.append(image)

    # print(images[0])
    encode_annot_array(images, os.path.join(labels_path, 'kitti_common.json'))


def _calc_center(box):
    return (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0


def _calc_dist(box1, box2):
    box1_x, box1_y = _calc_center(box1)
    box2_x, box2_y = _calc_center(box2)
    return sqrt((box1_x - box2_x) ** 2 + (box1_y - box2_y) ** 2)


def _find_nearest_cycle(image, rider, paired_cycles):
    nearest_idx = -1
    nearest_dist = sys.maxsize
    for i in range(0, len(image.labels)):
        if i in paired_cycles:
            continue

        label = image.labels[i]

        cycle = {'motorcycle', 'bicycle', 'motor', 'bike'}
        if not (label.category in cycle):
            continue

        curr_dist = _calc_dist(rider.box, label.box)
        if nearest_dist > curr_dist:
            nearest_idx = i
            nearest_dist = curr_dist

    return nearest_idx


def _get_common_box(box1, box2):
    p1 = min([box1.x1, box1.x2, box2.x1, box2.x2]), min([box1.y1, box1.y2, box2.y1, box2.y2])
    p2 = max([box1.x1, box1.x2, box2.x1, box2.x2]), max([box1.y1, box1.y2, box2.y1, box2.y2])
    return CommonBox(p1, p2)


def rider_reformat(path):
    """Combine riders' bboxes with the nearest motorcycle/bicycle"""
    for filename in glob.iglob(path + '\**\*.json', recursive=True):
        images = decode_annot_array(filename)
        for i in range(0, len(images)):
            image = images[i]

            to_delete = []
            for j in range(0, len(image.labels)):
                label = image.labels[j]
                if not (label.category == 'rider'):
                    continue

                rider = label
                nearest_idx = _find_nearest_cycle(image, label, to_delete)
                if nearest_idx < 0:
                    continue

                to_delete.append(nearest_idx)
                cycle = image.labels[nearest_idx]
                rider.box = _get_common_box(rider.box, cycle.box)

            to_delete.sort(reverse=True)
            for idx in to_delete:
                del image.labels[idx]

        encode_annot_array(images, filename, '_rider')


def keep_only_images_w_categories(path):
    """Remove images where there were no trains/trams annotated"""
    for filename in glob.iglob(path + '/**/*.json', recursive=True):

        categories = [['person', 'person_group'], ['two_wheeler'], ['on_rails'], ['car'], ['truck']]
        for cat in categories:
            cats = set([])
            for category in cat:
                print(category)
                cats = cats.union(get_category_set(category))

            images = decode_annot_array(filename)

            images_w_cats = []
            for image in images:
                for label in image.labels:
                    if label.category in cats:
                        images_w_cats.append(image)
                        break

            postfix = ''
            for category in cat:
                postfix += '_' + category
            encode_annot_array(images_w_cats, filename, postfix)


category_sets = [
    {'person', 'person (other)', 'pedestrian', 'sitting person', 'Pedestrian', 'Person_sitting'},
    {'person_group', 'person group'},
    {'two_wheeler', 'rider', 'motor', 'bike', 'motorcycle', 'bicycle', 'Cyclist'},
    {'on_rails', 'train', 'on rails', 'Tram'},
    {'car', 'Car'},
    {'truck', 'bus', 'Truck', 'caravan', 'trailer', 'Van'}
]


def get_category_sets_dict():
    cat_dict = {}
    cat_keys = ['person', 'person_group', 'two_wheeler', 'on_rails', 'car', 'truck']

    for i in range(0, len(category_sets)):
        cat_dict.update({frozenset(category_sets[i]): cat_keys[i]})
    return cat_dict


def get_category_set(category):
    """Return the correct category set (synonyms) or None if category was not found"""
    for category_set in category_sets:
        if category in category_set:
            return category_set
    return None


def keep_only_needed_categories(path, categories):
    cats = set([])
    for category in categories:
        print(category)
        cats = cats.union(get_category_set(category))

    for filename in glob.iglob(path + '\**\*.json', recursive=True):

        images = decode_annot_array(filename)

        for i in range(0, len(images)):
            image = images[i]

            labels_to_delete = []
            for j in range(0, len(image.labels)):
                label = image.labels[j]
                if not (label.category in cats):
                    labels_to_delete.append(j)

            labels_to_delete.sort(reverse=True)
            for idx in labels_to_delete:
                del image.labels[idx]

        encode_annot_array(images, filename, '_needed')


def cityscapes_citypersons_union(path):
    subfolders = ['both', 'train', 'val']

    for subfolder in subfolders:
        sub_path = os.path.join(path, subfolder)

        images_both = []
        for filename in glob.iglob(sub_path + '\**\*.json', recursive=True):
            images_both.append(decode_annot_array(filename))

        images_first = {}
        for image in images_both[0]:
            images_first.update({image.name: image})

        for i in range(0, len(images_both[1])):
            images_both[1][i].labels.extend(images_first.get(images_both[1][i].name).labels)

        encode_annot_array(images_both[1], os.path.join(sub_path, 'cs_2.json'))


def common_category_names(path):
    cat_dict = get_category_sets_dict()

    for filename in glob.iglob(path + '\**\*.json', recursive=True):
        images = decode_annot_array(filename)
        for i in range(0, len(images)):
            image = images[i]

            for j in range(0, len(image.labels)):
                label = image.labels[j]

                common_category = cat_dict.get(frozenset(get_category_set(label.category)))
                label.category = common_category

        encode_annot_array(images, filename, '_common')


def crop_dataset(path, x_offset, width, height, only_json=False):

    images_path = os.path.join(path, "images")

    if not only_json:
        for filename in glob.iglob(images_path + '/**/*.png', recursive=True):
            print(filename)
            img = cv2.imread(filename)
            crop_img = img[0:height, x_offset:x_offset + width]
            splitted = os.path.splitext(filename)
            cv2.imwrite(splitted[0] + '_cropped' + splitted[1], crop_img)

    for filename in glob.iglob(path + '/**/*.json', recursive=True):
        print(filename)
        images = decode_annot_array(filename)

        for i in range(0, len(images)):
            image = images[i]
            print(image.name)

            to_delete = []
            for j in range(0, len(image.labels)):
                box = image.labels[j].box

                if (float(box.x1) < x_offset and float(box.x2) < x_offset) or (float(box.x1) > x_offset + width and float(box.x2) > x_offset + width):
                    to_delete.append(j)
                    continue

                box.x1 = max(x_offset, float(box.x1))
                box.x2 = max(x_offset, float(box.x2))

                box.x1 = min(x_offset + width, float(box.x1))
                box.x2 = min(x_offset + width, float(box.x2))

                box.y1 = min(height, float(box.y1))
                box.y2 = min(height, float(box.y2))

            to_delete.sort(reverse=True)
            for idx in to_delete:
                del image.labels[idx]

        encode_annot_array(images, filename)


def crop_bbox_correction(path, x_offset):
    for filename in glob.iglob(path + '/**/*.json', recursive=True):
        print(filename)
        images = decode_annot_array(filename)

        for i in range(0, len(images)):
            image = images[i]
            print(image.name)

            for j in range(0, len(image.labels)):
                box = image.labels[j].box

                box.x1 -= x_offset
                box.x2 -= x_offset

        encode_annot_array(images, filename)


def resize_images(path, out_path, size):
    _resize_images(path, out_path, size, 'jpg')
    _resize_images(path, out_path, size, 'png')


def _resize_images(path, out_path, size, extension):
    for filename in glob.iglob(path + '/**/*.' + extension, recursive=True):
        img = cv2.imread(filename)
        resized_img = cv2.resize(img, size)
        base = os.path.basename(filename)
        cv2.imwrite(os.path.join(out_path, base), resized_img)


if __name__ == '__main__':
    # bdd100k_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\bdd')

    # wd_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\wd')

    # cp_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons')
    # cp_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons\\labels\\train', labels_folder=False)
    # cp_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons\\labels\\val', labels_folder=False)

    # cs_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes')
    # cs_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes\\labels\\train', labels_folder=False)
    # cs_to_common('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes\\labels\\val', labels_folder=False)

    # kitti_to_comon('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\kitti')

    # rider_reformat('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes')
    # rider_reformat('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\bdd100k')

    # keep_only_needed_categories('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\kitti',
    #                             ['Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Car', 'Truck', 'Van'])

    # keep_only_needed_categories('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\wd',
    #                      ['person', 'rider', 'motorcycle', 'bicycle', 'on rails', 'bus', 'car', 'truck', 'caravan'])

    # keep_only_needed_categories('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes',
    #                     ['rider', 'motorcycle', 'bicycle', 'on rails', 'bus', 'car', 'truck', 'caravan', 'trailer'])

    # keep_only_needed_categories('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\citypersons',
    #                             ['person (other)', 'pedestrian', 'sitting person', 'person group'])

    # keep_only_needed_categories('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\bdd100k',
    #                             ['person', 'rider', 'motor', 'bike', 'train', 'bus', 'car', 'truck'])

    # cityscapes_citypersons_union('C:\\Users\\ext-dobaib\\Desktop\\Datasets\\cityscapes_v2')

    # common_category_names('C:\\Users\\ext-dobaib\\Desktop\\Datasets')

    # keep_only_images_w_categories('/home/boti/Workspace/data')

    # crop_dataset('/media/boti/Adatok/Datasets-pc/cityscapes', x_offset=120, width=1808, height=1017)

    # crop_dataset('/media/boti/Adatok/Datasets-pc/kitti', x_offset=294, width=656, height=369, only_json=True)

    # resize_images('/media/boti/Adatok/Datasets-pc', '/media/boti/Adatok/Datasets-pc/resized', (608, 608))

    # crop_bbox_correction('/media/boti/Adatok/Datasets-pc/cityscapes', x_offset=120)
    # crop_bbox_correction('/media/boti/Adatok/Datasets-pc/kitti', x_offset=294)
    pass

# path: should be the bdd100k root folder
# labels: bdd100k/labels
# images: bdd100k/images
def load_bdd100k(path):
    # TODO
    raise NotImplementedError
