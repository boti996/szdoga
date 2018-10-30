from enum import Enum
import argparse
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

from data.yolov3_load_dataset import YoloV3DataLoader
from model.yolo3.model import preprocess_true_boxes
from model.yolo3.yolov3_model import YoloV3Model
from model.yolo3.utils import get_random_data


class ModelType(Enum):
    YOLO_V3 = 'yolo_v3'


def main():
    """load model, then train it with training data"""
    # read command line arguments
    model_type, input_shape, anchors, class_names, weights_path, freeze_body, dataset_path = _read_args()

    model_loader = None
    dataset_loader = None
    if model_type == ModelType.YOLO_V3:
        model_loader = YoloV3Model(input_shape, anchors, len(class_names), weights_path, freeze_body)
        dataset_loader = YoloV3DataLoader()
    assert model_loader is not None
    assert dataset_loader is not None

    # load model
    model = model_loader.get_model()
    print('Model loaded.')
    # model.summary()

    # load dataset
    dataset_loaded = dataset_loader.load_dataset(dataset_path)
    datasets = dataset_loaded['train']
    validation = dataset_loaded['val']
    print('Dataset loaded.')

    # TODO: correct bounding-boxes after resize!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # train model
    log_dir = 'logs'
    logging = TensorBoard(log_dir=log_dir)

    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        # TODO: set batch size param
        batch_size = 32

        num_train = 0
        for dataset in datasets:
            num_train += len(dataset)
        num_val = len(validation)

        model.fit_generator(
            data_generator_wrapper(datasets, batch_size, input_shape, anchors, len(class_names)),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(validation, batch_size, input_shape, anchors,
                                                   len(class_names)),
            validation_steps=max(1, num_val // batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

        # Unfreeze and continue training, to fine-tune.
        # Train longer if the result is not good.
        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-4),
                          loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = 32  # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            model.save_weights(log_dir + 'trained_weights_final.h5')


def _read_args():
    """read the call parameters"""
    parser = argparse.ArgumentParser(description='Parameters for creating and training a model.')
    parser.add_argument('-t', '--type', nargs=1, help='Type of trained model.', required=True)
    parser.add_argument('-s', '--shape', nargs=2, help='Input shape (height width).', required=True)
    parser.add_argument('-a', '--anchors_path', nargs=1, help='File path of anchors.', required=True)
    parser.add_argument('-c', '--classes_path', nargs=1, help='File path of classes.', required=True)
    parser.add_argument('-w', '--weights_path', nargs=1,
                        help='File path of weights (serialized in Keras\' .h5 format).', required=True)
    parser.add_argument('-f', '--freeze_body', nargs=1, help='Layer-freezing parameter (0, 1 or 2).', default=0)
    parser.add_argument('-d', '--dataset_path', nargs=1, help='Dataset\'s path.', required=True)

    args = parser.parse_args()

    type = ModelType(args.type[0])
    input_shape = (int(args.shape[0]), int(args.shape[1]))
    anchors = _get_anchors(args.anchors_path[0])
    class_names = _get_classes(args.classes_path[0])
    weights_path = args.weights_path[0]
    freeze_body = int(args.freeze_body[0])
    dataset_path = args.dataset_path[0]

    return type, input_shape, anchors, class_names, weights_path, freeze_body, dataset_path


def _get_classes(classes_path):
    """load the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def _get_anchors(anchors_path):
    """load the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == "__main__":
    main()
