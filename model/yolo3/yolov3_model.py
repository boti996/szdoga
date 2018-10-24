from keras import Model
from keras.layers import Lambda
from keras import Input

from model.networkmodel import NetworkModel

import keras.backend as keras

from model.yolo3.model import yolo_body, yolo_loss


class YoloV3Model(NetworkModel):

    def __init__(self, input_shape, anchors, num_classes, weights=None, freeze_body=2):
        self.h, self.w = input_shape
        self.anchors = anchors
        self.n_classes = num_classes
        self.weights = weights
        self.freeze_body = freeze_body

    def get_model(self):
        keras.clear_session()  # new model session

        n_anchors = len(self.anchors)

        image_input = Input(shape=(None, None, 3))
        model_body = yolo_body(image_input, n_anchors // 3, self.n_classes)

        print('Create YOLOv3 model with {} anchors and {} classes.'.format(n_anchors, self.n_classes))

        if self.weights:
            model_body.load_weights(self.weights, by_name=True, skip_mismatch=True)

            print('Load weights {}.'.format(self.weights))

            # Freeze darknet53 body or freeze all but 3 output layers.
            if self.freeze_body in [1, 2]:
                num = (185, len(model_body.layers) - 3)[self.freeze_body - 1]

                for i in range(num):
                    model_body.layers[i].trainable = False

                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        y_true = [Input(shape=(self.h // {0: 32, 1: 16, 2: 8}[l], self.w // {0: 32, 1: 16, 2: 8}[l],
                               n_anchors // 3, self.n_classes + 5)) for l in range(3)]

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={
            'anchors': self.anchors, 'num_classes': self.n_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], model_loss)

        return model


