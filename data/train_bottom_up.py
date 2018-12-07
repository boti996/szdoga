import time
from enum import Enum

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

import data.train_model
from data.yolov3_load_dataset import YoloV3DataLoader
from model.yolo3.yolov3_model import YoloV3Model


def _unfreeze_block(model, i_unfreeze, i_block_block):
    # TODO
    pass


# TODO: refactor to prevent code redundancy with train_model.py
def main():
    """Load model, then train it with training data"""
    # read command line arguments
    model_type, input_shape, anchors, class_names, init_weights_path, freeze_body, dataset_path = \
        data.train_model._read_args()
    n_classes = len(class_names)

    # choose loaders according to the current network type
    model_loader, dataset_loader = None, None
    if model_type == data.train_model.ModelType.YOLO_V3:
        model_loader = YoloV3Model(input_shape, anchors, n_classes, init_weights_path, freeze_body)
        dataset_loader = YoloV3DataLoader()
    assert None not in (model_loader, dataset_loader)

    # load dataset
    dataset_loaded = dataset_loader.load_dataset(dataset_path, input_shape)
    datasets = dataset_loaded['train']
    validation = dataset_loaded['val']
    print('Dataset loaded.')

    # logging with tensorboard
    log_dir = '../logs'
    logging = TensorBoard(log_dir=log_dir)

    # save weights into folder
    weights_dir = '../trained_weights/'

    # save model weights periodically
    checkpoint = ModelCheckpoint(weights_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

    # reduce learning rate according to the loss measured on validation set
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    # stop training after 'patience' number of epochs if there were no improvements
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # number of training samples
    num_train = 0
    for dataset in datasets:
        num_train += len(dataset)

    # number of validation samples
    num_val = 0
    for dataset in validation:
        num_val += len(dataset)

    for i in range(1, 25):

        # load model
        model = model_loader.get_model(pruning=None, n_blocks=i)
        print('Model loaded.')
        # model.summary()

        # freeze all layers
        for l in range(len(model.layers)):
            model.layers[l].trainable = False

        # unfreeze block to train
        # TODO model = _unfreeze_block(model, i_block_to_freeze, i_block_block)

        # model.summary()

        for l in model.layers:
            if l.trainable:
                print(l.name)
        print()

        # mini-batch size
        batch_size = 8

        # Train with frozen layers first, to get a stable loss.
        model.compile(optimizer=Adam(lr=0.001), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.fit_generator(
            generator=data.train_model.data_generator_wrapper(datasets, batch_size, input_shape, anchors, n_classes,
                                                              dataset_path),
            steps_per_epoch=max(1, num_train // batch_size // 20),
            validation_data=data.train_model.data_generator_wrapper(validation, batch_size, input_shape, anchors,
                                                                    n_classes, dataset_path),
            validation_steps=max(1, min(50, num_val // batch_size)),
            epochs=10,
            initial_epoch=0,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])

        # serialize weights
        model.save_weights(weights_dir + 'mod_trained_weights_stage_bottomup_' + str(i) +'.h5')
        model.save_weights(weights_dir + 'mod_trained_weights_stage_bottomup_' + str(i) +str(time.time()) + '.h5')

        model_loader = YoloV3Model(input_shape, anchors, n_classes, weights_dir + 'mod_trained_weights_stage_bottomup_' +
                                   str(i) + '.h5', freeze_body)


if __name__ == "__main__":
    main()
