import argparse
from enum import Enum
import time

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from data.data_generator import yolov3_generator
from data.yolov3_load_dataset import YoloV3DataLoader
from model.yolo3.model import preprocess_true_boxes
from model.yolo3.yolov3_model import YoloV3Model


class ModelType(Enum):
    """Type of used model during training"""
    YOLO_V3 = 'yolo_v3'


def _read_args():
    """Read command-line arguments"""
    parser = argparse.ArgumentParser(description='Parameters for creating and training a model.')
    parser.add_argument('-t', '--type', nargs=1, help='Type of trained model.', required=True)
    parser.add_argument('-s', '--shape', nargs=2, help='Input shape (width height).', required=True)
    parser.add_argument('-a', '--anchors_path', nargs=1, help='File path of anchors.', required=True)
    parser.add_argument('-c', '--classes_path', nargs=1, help='File path of classes.', required=True)
    parser.add_argument('-w', '--weights_path', nargs=1,
                        help='File path of weights (serialized in Keras\' .h5 format).', required=True)
    parser.add_argument('-f', '--freeze_body', nargs=1, help='Layer-freezing parameter (0, 1 or 2).', default=0)
    parser.add_argument('-d', '--dataset_path', nargs=1, help='Dataset\'s path.', required=True)

    args = parser.parse_args()

    model_type = ModelType(args.type[0])
    input_shape = (int(args.shape[0]), int(args.shape[1]))
    anchors = _get_anchors(args.anchors_path[0])
    class_names = _get_classes(args.classes_path[0])
    weights_path = args.weights_path[0]
    freeze_body = int(args.freeze_body[0])
    dataset_path = args.dataset_path[0]

    return model_type, input_shape, anchors, class_names, weights_path, freeze_body, dataset_path


def _get_classes(classes_path):
    """Read class-names into an array"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def _get_anchors(anchors_path):
    """Read anchors into an array in [ [x_1, y_1], ..., [x_n, y_n] ] format"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def _data_generator(dataset, batch_size, input_shape, anchors, num_classes, images_path):
    """Data generator for training the model on augmented mini-batches"""
    while True:
        # get images and bboxes
        image_data, box_data = yolov3_generator(images_path, dataset, batch_size, input_shape)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # convert bounding boxes into yolo format
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # return data in the needed format
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(dataset, batch_size, input_shape, anchors, num_classes, images_path):
    """Data generator wrapper"""
    n = len(dataset)
    if n == 0 or batch_size <= 0:
        return None
    return _data_generator(dataset, batch_size, input_shape, anchors, num_classes, images_path)


def main():
    """Load model, then train it with training data"""
    # read command line arguments
    model_type, input_shape, anchors, class_names, init_weights_path, freeze_body, dataset_path = _read_args()
    n_classes = len(class_names)

    # choose loaders according to the current network type
    model_loader, dataset_loader = None, None
    if model_type == ModelType.YOLO_V3:
        model_loader = YoloV3Model(input_shape, anchors, n_classes, init_weights_path, freeze_body)
        dataset_loader = YoloV3DataLoader()
    assert None not in (model_loader, dataset_loader)

    # load model
    model = model_loader.get_model()
    print('Model loaded.')
    model.summary()

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

    fine_tuning_only = True
    if not fine_tuning_only:

        # mini-batch size
        # TODO: larger batch size possibilities
        batch_size = 8

        # Train with frozen layers first, to get a stable loss.
        model.compile(optimizer=Adam(lr=0.001), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.fit_generator(
            generator=data_generator_wrapper(datasets, batch_size, input_shape, anchors, n_classes, dataset_path),
            steps_per_epoch=max(1, num_train // batch_size // 20),
            validation_data=data_generator_wrapper(validation, batch_size, input_shape, anchors,
                                                   n_classes, dataset_path),
            validation_steps=max(1, min(50, num_val // batch_size)),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])

        # serialize weights
        model.save_weights(weights_dir + 'trained_weights_stage_1.h5')
        model.save_weights(weights_dir + 'trained_weights_stage_1' + str(time.time()) + '.h5')

    print('fine-tuning')

    # mini-batch size
    batch_size = 8  # note that more GPU memory is required after unfreezing the body

    # Unfreeze and continue training, to fine-tune.
    model.load_weights(weights_dir + 'trained_weights_stage_1.h5')

    # freeze2 = True
    # if not freeze2:
    #    for i in range(len(model.layers)):
    #        model.layers[i].trainable = True

    model.compile(optimizer=Adam(lr=0.0001), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    print('All layers are unfrozen now.')

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    model.fit_generator(
        generator=data_generator_wrapper(datasets, batch_size, input_shape, anchors, n_classes, dataset_path),
        steps_per_epoch=max(1, num_train // batch_size // 20),
        validation_data=data_generator_wrapper(validation, batch_size, input_shape, anchors,
                                               n_classes, dataset_path),
        validation_steps=max(1, min(50, num_val // batch_size)),
        epochs=100,
        initial_epoch=50,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    # serialize weights
    model.save_weights(weights_dir + 'trained_weights_final.h5')
    model.save_weights(weights_dir + 'trained_weights_final' + str(time.time()) + '.h5')


if __name__ == "__main__":
    main()
