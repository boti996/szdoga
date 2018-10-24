from enum import Enum
import argparse
import numpy as np
from model.yolo3.yolov3_model import YoloV3Model


class ModelType(Enum):
    YOLO_V3 = 'yolo_v3'


def main():
    """load model, then train it with training data"""
    # read command line arguments
    type, input_shape, anchors, class_names, weights_path, freeze_body = _read_args()

    model_loader = None
    if type == ModelType.YOLO_V3:
        model_loader = YoloV3Model(input_shape, anchors, len(class_names), weights_path, freeze_body)
    assert model_loader is not None

    # load model
    model = model_loader.get_model()
    print('Model loaded.')
    # model.summary()




def _read_args():
    """read the call parameters"""
    parser = argparse.ArgumentParser(description='Parameters for creating and training a model.')
    parser.add_argument('-t', '--type', nargs=1, help='Type of trained model.', required=True)
    parser.add_argument('-s', '--shape', nargs=2, help='Input shape (height width).', required=True)
    parser.add_argument('-a', '--anchors_path', nargs=1, help='File path of anchors.', required=True)
    parser.add_argument('-c', '--classes_path', nargs=1, help='File path of classes.', required=True)
    parser.add_argument('-w', '--weights_path', nargs=1, help='File path of weights (serialized in Keras\' .h5 format).', required=True)
    parser.add_argument('-f', '--freeze_body', nargs=1, help='Layer-freezing parameter (0, 1 or 2).', default=0)

    args = parser.parse_args()

    type = ModelType(args.type[0])
    input_shape = (int(args.shape[0]), int(args.shape[1]))
    anchors = _get_anchors(args.anchors_path[0])
    class_names = _get_classes(args.classes_path[0])
    weights_path = args.weights_path[0]
    freeze_body = int(args.freeze_body[0])

    return type, input_shape, anchors, class_names, weights_path, freeze_body


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


if __name__ == "__main__":
    main()