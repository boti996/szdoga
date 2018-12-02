from enum import Enum

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

    def _pruning(self, model_body):
        # step backwards in layers and rename
        class PruningState(Enum):
            SEARCH_ADD = 0
            SEARCH_CONV = 1
            DONE = 2

        n_layers = len(self.yolo_model.layers)
        print('n_layers: ' + str(n_layers))
        curr_state = PruningState.SEARCH_ADD

        if self.pruning is not None and self.pruning[0] > 0:

            for to_prone in self.pruning:

                for i in range(n_layers - 1, -1, -1):

                    ''' rename last conv2d layer of the block to be pruned
                        this way the zeros set by their initializer 
                        will stay untouched after loading the weights '''
                    if curr_state == PruningState.DONE:
                        break
                    if curr_state == PruningState.SEARCH_ADD:
                        if to_prone <= 0:
                            curr_state = PruningState.DONE
                            continue
                        elif self.yolo_model.layers[i].name.startswith('add_'):
                            curr_state = PruningState.SEARCH_CONV
                            # print(self.pruning)
                            continue
                    if curr_state == PruningState.SEARCH_CONV:
                        if self.yolo_model.layers[i].name.startswith('conv2d_'):
                            to_prone -= 1
                            if to_prone == 0:
                                self.yolo_model.layers[i].name = self.yolo_model.layers[i].name + '_pruned'
                            # print(self.pruning)
                            curr_state = PruningState.SEARCH_ADD
                            continue

        # self.yolo_model.summary()
        # input("Press Enter to continue...")

    def get_model(self, mod_mask=(False, False, False, False, False)):
        print('n_classe: ' + str(self.n_classes))

        n_anchors = len(self.anchors)

        keras.clear_session()  # new model session

        image_input = Input(shape=(None, None, 3))
        model_body = yolo_body(image_input, n_anchors // 3, self.n_classes, mod_mask=mod_mask)

        print('Create YOLOv3 model with {} anchors and {} classes.'.format(n_anchors, self.n_classes))

        if self.weights:
            model_body.load_weights(self.weights, by_name=True, skip_mismatch=True)

            print('Load weights {}.'.format(self.weights))

            # Freeze darknet53 body or freeze all but 3 output layers.
            if self.freeze_body in [1, 2]:
                num = (185, len(model_body.layers) - 3)[self.freeze_body - 1]

                for i in range(num):
                    model_body.layers[i].trainable = False

                print('Unfreezed layers: ')
                for layer in model_body.layers:
                    if layer.trainable:
                        print(layer.name)
                        print(layer.get_weights())

                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        y_true = [Input(shape=(self.h // {0: 32, 1: 16, 2: 8}[l], self.w // {0: 32, 1: 16, 2: 8}[l],
                               n_anchors // 3, self.n_classes + 5)) for l in range(3)]

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={
            'anchors': self.anchors, 'num_classes': self.n_classes, 'ignore_thresh': 0.5, 'print_loss': True})([*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], model_loss)

        return model
