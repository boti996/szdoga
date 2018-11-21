import time

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

import data.train_model
from data.yolov3_load_dataset import YoloV3DataLoader
from model.yolo3.yolov3_model import YoloV3Model

# TODO: refactor to prevent code redundancy with train_model.py
def main():
    """Load model, then train it with training data"""
    # read command line arguments
    model_type, input_shape, anchors, class_names, init_weights_path, freeze_body, dataset_path = data.train_model._read_args()
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

    # number of training samples
    num_train = 0
    for dataset in datasets:
        num_train += len(dataset)

    # number of validation samples
    num_val = 0
    for dataset in validation:
        num_val += len(dataset)

    curr_model_mask = [False, False, False, False, False]

    for i in range(0, 4):

        curr_model_mask[4 - i] = True

        # load model
        model = model_loader.get_model(mod_mask=tuple(curr_model_mask))
        print('Model loaded.')
        # model.summary()

        # mini-batch size
        # TODO: larger batch size possibilities
        batch_size = 1

        # Train with frozen layers first, to get a stable loss.
        model.compile(optimizer=Adam(lr=0.001), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.fit_generator(
            generator=data.train_model.data_generator_wrapper(datasets, batch_size, input_shape, anchors, n_classes, dataset_path),
            steps_per_epoch=max(1, num_train // batch_size // 20),
            validation_data=data.train_model.data_generator_wrapper(validation, batch_size, input_shape, anchors,
                                                        n_classes, dataset_path),
            validation_steps=max(1, min(50, num_val // batch_size)),
            epochs=3,
            initial_epoch=0,
            callbacks=[logging, checkpoint])

        # serialize weights
        model.save_weights(weights_dir + 'mod_trained_weights_stage_' + str(i) + '.h5')
        model.save_weights(weights_dir + 'mod_trained_weights_stage_' + str(i) + '_' + str(time.time()) + '.h5')


if __name__ == "__main__":
    main()
