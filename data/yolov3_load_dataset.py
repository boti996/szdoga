import glob
from cv2 import cv2
from data.load_dataset import DataLoader
from data.serialization import decode_annot_array
import random
from os.path import join


class YoloV3DataLoader(DataLoader):

    # TODO: implementation
    def load_nth_batch(self, n):
        raise NotImplementedError


    def resize_bboxes(self, images, size, size_new):
        #TODO: SORREND FONTOS
        y_max, x_max = size
        y_max_new, x_max_new = size_new

        for i in range(0, len(images)):
            image = images[i]

            for j in range(0, len(image.labels)):
                label = image.labels[j]
                box = label.box

                box.x1 = float(box.x1) / x_max * x_max_new
                box.x2 = float(box.x2) / x_max * x_max_new

                box.y1 = float(box.y1) / y_max * y_max_new
                box.y2 = float(box.y2) / y_max * y_max_new

        return images


    def load_folder(self, path, size_new):
        images = []
        print(path)
        for filename in glob.iglob(path + '/**/*.json', recursive=True):
            images_new = decode_annot_array(filename)
            print(len(images_new))

            # # resize bounding boxes
            # print(path + ' ' + images_new[0].name)
            # img = next(glob.iglob(path + '/**/' + images_new[0].name, recursive=True))
            # height, width, _ = img.shape

            # i# mages_new = self.resize_bboxes(images_new, (height, width), size_new)

            images.extend(images_new)
        return images


    def resize_img(self, img, size):
        return cv2.resize(img, size, img)


    # TODO: implementation
    def load_dataset(self, path):

        size = (608, 608)

        on_rails = self.load_folder(join(path, '_on_rails'), size)
        two_wheeler = self.load_folder(join(path, '_two_wheeler'), size)
        person = self.load_folder(join(path, '_person'), size)
        truck = self.load_folder(join(path, '_truck'), size)

        bdd = self.load_folder(join(path, '_bdd-only'), size)
        others = self.load_folder(join(path, '_others'), size)

        # validation dataset
        validation = self.load_folder(join(path, '_validation'), size)

        # training datasets
        datasets = [on_rails, two_wheeler, person, truck, bdd, others]

        for i in range(0, len(datasets)):
            random.shuffle(datasets[i])

        random.shuffle(validation)

        return {'val': [validation], 'train': datasets}


    # TODO: implementation
    def create_dataset_batch(self, n_batch_size):
        raise NotImplementedError
