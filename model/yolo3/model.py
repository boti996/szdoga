"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.activations import linear
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, SeparableConv2D, ReLU, \
    Activation, DepthwiseConv2D
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from model.yolo3.utils import compose


# pruning vs initialization
init_string = 'zeros'


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'kernel_initializer': init_string,
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


@wraps(SeparableConv2D)
def MobilenetSeparableConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for SeparableConvolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'kernel_initializer': init_string,
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return SeparableConv2D(*args, **darknet_conv_kwargs)



def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def MobilenetConv2D_BN_ReLU(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        ReLU(max_value=6))


def MobilenetConv2D_BN_Linear(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization())


def MobilenetSeparableConv2D_BN_ReLU(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    global i_name
    i_name += 1
    return compose(
        MobilenetSeparableConv2D(*args, **no_bias_kwargs),
        BatchNormalization(name='batch_normalization_extra_' + str(i_name)),   # TODO delete name
        ReLU(max_value=6))


cutting_layer_out, cutting_layer_in = None, None


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    if num_blocks == 0:
        return x

    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)

    for i in range(0, num_blocks):

        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])

    return x


def inverted_resblock_body(x, num_filters, num_blocks):
    if num_blocks == 0:
        return x

    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = MobilenetSeparableConv2D_BN_ReLU(num_filters // 2, (3, 3), strides=(2, 2))(x)

    for i in range(0, num_blocks):

        y = compose(MobilenetConv2D_BN_ReLU(num_filters // 2, (1, 1)),
                    MobilenetSeparableConv2D_BN_ReLU(num_filters // 2 * 6, (3, 3), strides=(1, 1)),
                    MobilenetConv2D_BN_Linear(num_filters // 2, (1, 1)))(x)
        x = Add()([x, y])

    return x


def mixed_resblock_body(x, num_filters, num_blocks, n_inverted=0):
    n_multipl = 1   # unit of changing blocks at once

    # there are both inverted- and normal residual blocks
    if n_multipl * n_inverted < num_blocks:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    # only inverted residual blocks
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = MobilenetSeparableConv2D_BN_ReLU(num_filters // 2, (3, 3), strides=(2, 2))(x)
        x = Conv2D(num_filters // 2, (1, 1))(x) # TODO nem lesz jó

    # add normal residual blocks
    for i in range(0, num_blocks - n_multipl * n_inverted):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])

    global i_name
    i_name += 1

    # TODO: what to do with this layer?
    if n_inverted > 0:
        x = Conv2D(num_filters // 2, (1, 1), name='conv2d_adapter_' + str(i_name))(x)

    # add inverted residual blocks
    # todo freeze all except last iteration: only when (n_multipl * n_inverted == num_blocks)
    for i in range(0, n_multipl * n_inverted):
        y = compose(
            MobilenetConv2D_BN_ReLU(num_filters // 2, (1, 1)),
            MobilenetSeparableConv2D_BN_ReLU(num_filters // 2 * 6, (3, 3), strides=(1, 1)),
            MobilenetConv2D_BN_Linear(num_filters // 2, (1, 1)))(x)
        x = Add()([x, y])

    if n_inverted > 0:
        x = Conv2D(num_filters, (1, 1), name='conv2d_adapter2_' + str(i_name))(x)

    return x


out1, out2 = None, None

i_name = 0


def darknet_body(x, pruning, mod_mask=(0, 0, 0, 0, 0)):
    """Darknent body having 52 Convolution2D layers"""

    global cutting_layer_in

    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = mixed_resblock_body(x, 64, 1, mod_mask[0])
    x = mixed_resblock_body(x, 128, 2, mod_mask[1])
    x = mixed_resblock_body(x, 256, 8, mod_mask[2])
    global out1
    out1 = x
    x = mixed_resblock_body(x, 512, 8, mod_mask[3])
    global out2
    out2 = x
    x = mixed_resblock_body(x, 1024, 4, mod_mask[4])

    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def _get_n_blocks_array(n_blocks):
    # number of blocks: 1+2+8+8+4
    n_array = [0, 0, 0, 0, 0]
    n_array[0] = min(1, max(0, n_blocks - 0))
    n_array[1] = min(2, max(0, n_blocks - 1))
    n_array[2] = min(8, max(0, n_blocks - 2 - 1))
    n_array[3] = min(8, max(0, n_blocks - 8 - 2 - 1))
    n_array[4] = min(4, max(0, n_blocks - 8 - 8 - 2 - 1))
    return n_array


out1_orig, out2_orig = None, None


# used for bottom-up knowledge distillation
def darknet_body_2(x, n_blocks, is_inverted):

    n_array = _get_n_blocks_array(n_blocks)

    blocks = inverted_resblock_body if is_inverted else resblock_body

    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = blocks(x, 64, n_array[0])
    x = blocks(x, 128, n_array[1])
    x = blocks(x, 256, n_array[2])
    global out1, out1_orig
    out1 = x if is_inverted else out1_orig = x
    x = blocks(x, 512, n_array[3])
    global out2, out2_orig
    out2 = x if is_inverted else out2_orig = x
    x = blocks(x, 1024, n_array[4])

    return x


# used for bottom-up knowledge distillation
def yolo_body_2(inputs, num_anchors, num_classes, pruning=None, n_blocks=23):

    global init_string
    init_string = 'glorot_uniform'

    darknet = Model(inputs, darknet_body_2(inputs, n_blocks, True))
    darknet_orig = Model(inputs, darknet_body_2(inputs, n_blocks, False))

    # no second part
    if n_blocks <= 23:
        return Model(inputs, [darknet, darknet_orig])

    # non-orig
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, out2])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, out1])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    # orig
    x_orig, y1_orig = make_last_layers(darknet_orig.output, 512, num_anchors * (num_classes + 5))

    x_orig = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x_orig)
    x_orig = Concatenate()([x_orig, out2_orig])
    x_orig, y2_orig = make_last_layers(x_orig, 256, num_anchors * (num_classes + 5))

    x_orig = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x_orig)
    x_orig = Concatenate()([x_orig, out1_orig])
    x_orig, y3_orig = make_last_layers(x_orig, 128, num_anchors * (num_classes + 5))

    init_string = 'zeros'

    return Model(inputs, [y1, y2, y3, y1_orig, y2_orig, y3_orig])


def yolo_body(inputs, num_anchors, num_classes, pruning=None,
              mod_mask=(0, 0, 0, 0, 0)):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs, pruning, mod_mask))

    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, out2])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, out1])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    # model_1st_part = Model(input=inputs, output=cutting_layer_out)
    # model_2nd_part = Model(input=model_1st_part.output, output=cutting_layer_in)

    return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    print(K.shape(yolo_outputs))
    num_layers = 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss_2(self_out, trainer_out):
    # TODO: L1 loss sum(|gt - pred|)
    y_pred = self_out
    y_true = trainer_out
    return K.sum(K.abs(y_true - y_pred))


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """Return yolo_loss tensor

    Parameters
    ----------
    args:
        yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body,
        y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    print_loss: print the loss after evaluation

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    # Area of smallest and largest anchor boxes
    area_0 = anchors[0][0] * anchors[0][1]
    area_2 = anchors[6][0] * anchors[6][1]
    output_weight_multipl = 2.0 / (np.log2(area_2) - np.log2(area_0))
    area_0_log2 = np.log2(area_0)


    # there are 3 layers in default, first one has the lowest resolution
    for l in range(0, num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, _ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            _ignore_mask = _ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, _ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *_args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # Find the best output layer according to the size of an object
        area_pred = raw_pred[..., 2] * raw_pred[..., 3]

        area_pred = K.map_fn(lambda x: K.maximum(x, area_0), area_pred)
        area_pred = K.map_fn(lambda x: K.minimum(x, area_2), area_pred)

        output_weight = K.map_fn(lambda x: x - area_0_log2,
                                 K.log(area_pred)) * output_weight_multipl

        # It will be a float between 0 and 2
        output_weight = K.abs(K.map_fn(lambda x: x - l, output_weight))

        output_weight = K.expand_dims(output_weight, -1)

        obj_scale = 5
        xywh_scale = 0.5
        output_weight_scale = 1

        xy_loss = xywh_scale * object_mask * box_loss_scale * K.square(raw_true_xy - raw_pred[..., 0:2])

        wh_loss = xywh_scale * object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        confidence_loss = obj_scale * object_mask * K.binary_crossentropy(
            object_mask, raw_pred[..., 4:5], from_logits=True)

        confidence_loss += (1 - object_mask) * K.binary_crossentropy(
            object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        output_loss = object_mask * output_weight_scale * output_weight * ignore_mask

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        output_loss = K.sum(output_loss) / mf

        loss += xy_loss + wh_loss + confidence_loss + class_loss + output_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, output_loss,
                                   K.sum(ignore_mask)], message=str(l) + '. loss: ')

    return loss
