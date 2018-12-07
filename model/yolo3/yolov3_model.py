from enum import Enum

from keras import Model
from keras.layers import Lambda
from keras import Input

from model.networkmodel import NetworkModel

import keras.backend as keras

from model.yolo3.model import yolo_body, yolo_body_2, yolo_loss, yolo_loss_2


class YoloV3Model(NetworkModel):

    def __init__(self, input_shape, anchors, num_classes, weights=None, freeze_body=2):
        self.h, self.w = input_shape
        self.anchors = anchors
        self.n_classes = num_classes
        self.weights = weights
        self.freeze_body = freeze_body


    def _pruning(self, model, pruning):
        # step backwards in layers and rename
        class PruningState(Enum):
            SEARCH_ADD = 0
            SEARCH_CONV = 1
            DONE = 2

        n_layers = len(model.layers)
        print('n_layers: ' + str(n_layers))
        curr_state = PruningState.SEARCH_ADD

        if pruning is not None and pruning[0] > 0:

            for to_prone in pruning:

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
                        elif model.layers[i].name.startswith('add_'):
                            curr_state = PruningState.SEARCH_CONV
                            continue
                    if curr_state == PruningState.SEARCH_CONV:
                        if model.layers[i].name.startswith('conv2d_'):
                            to_prone -= 1
                            if to_prone == 0:
                                # overwrite name to prevent loading weights
                                model.layers[i].name = model.layers[i].name + '_pruned'
                                # make it non-trainable to keep the weights zeros during the training
                                model.layers[i].trainable = False
                            curr_state = PruningState.SEARCH_ADD
                            continue

        # model.summary()
        # input("Press Enter to continue...")

        return model


    def get_model(self, mod_mask=(0, 0, 0, 0, 0), pruning=None, is_bottom_up=False, n_blocks=24):
        print('n_classe: ' + str(self.n_classes))

        n_anchors = len(self.anchors)

        keras.clear_session()  # new model session

        image_input = Input(shape=(None, None, 3))

        if is_bottom_up:
            model_body = yolo_body_2(image_input, n_anchors // 3, self.n_classes, n_blocks=n_blocks)
        else:
            model_body = yolo_body(image_input, n_anchors // 3, self.n_classes, mod_mask=mod_mask)

        print('Create YOLOv3 model with {} anchors and {} classes.'.format(n_anchors, self.n_classes))

        # pruning step
        model_body = self._pruning(model_body, pruning)

        if self.weights:
            model_body.load_weights(self.weights, by_name=True, skip_mismatch=True)

            print('Load weights {}.'.format(self.weights))

            # Freeze darknet53 body or freeze all but 3 output layers.
            if self.freeze_body in [1, 2]:
                num = (185, len(model_body.layers) - 3)[self.freeze_body - 1]

                for i in range(num):
                    model_body.layers[i].trainable = False

                ''' print('Unfrozen layers: ')
                for layer in model_body.layers:
                    if layer.trainable:
                        print(layer.name)
                        # print(layer.get_weights()) '''

                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        y_true = [Input(shape=(self.h // {0: 32, 1: 16, 2: 8}[l], self.w // {0: 32, 1: 16, 2: 8}[l],
                               n_anchors // 3, self.n_classes + 5)) for l in range(3)]

        if is_bottom_up:
            model_loss = Lambda(yolo_loss_2, output_shape=(1,), name='yolo_loss_2')([*model_body.output])
        else:
            model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={
                'anchors': self.anchors, 'num_classes': self.n_classes, 'ignore_thresh': 0.5, 'print_loss': True})(
                [*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], model_loss)

        return model
