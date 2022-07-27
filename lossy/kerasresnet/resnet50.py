"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras.applications import get_submodules_from_kwargs
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf

preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

backend = None
layers = None
models = None
keras_utils = None


class IdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv2a = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.relu2a = layers.Activation('relu')
        self.conv2b = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.relu2b = layers.Activation('relu')
        self.conv2c = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
        self.relu2c = layers.Activation('relu')

    def __call__(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = self.bn2a (x)
        x = self.relu2a(x)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu2b(x)
        x = self.conv2c(x)
        x = self.bn2c(x)
        x = layers.add([x, input_tensor])
        x = self.relu2c(x)
        return x

class ConvBlock(tf.keras.Model):
    def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv2a = layers.Conv2D(filters1, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.relu2a = layers.Activation('relu')
        self.conv2b = layers.Conv2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.relu2b = layers.Activation('relu')

        self.conv2c = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

        self.conv1 = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')
        self.bn1 = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')
        self.relu2c = layers.Activation('relu')

    def __call__(self, input_tensor):

        x = self.conv2a(input_tensor)
        x = self.bn2a(x)
        x = self.relu2a(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu2b(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        shortcut = self.conv1 (input_tensor)
        shortcut = self.bn1(shortcut)

        x = layers.add([x, shortcut])
        x = self.relu2c(x)
        return x


class ResNet50(tf.keras.Model):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    def __init__(self,
        include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
        global backend, layers, models, keras_utils
        backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `imagenet` '
                             '(pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                             ' as true, `classes` should be 1000')
        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape,
                                          default_size=224,
                                          min_size=32,
                                          data_format=backend.image_data_format(),
                                          require_flatten=include_top,
                                          weights=weights)

        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        self.conv1_pad = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')
        self.conv1 = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')
        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.relu1 = layers.Activation('relu')
        self.pool1_pad = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')
        self.pool1 = layers.MaxPooling2D((3, 3), strides=(2, 2))
        self.block2a = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.block2b = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self.block2c = IdentityBlock(3, [64, 64, 256], stage=2, block='c')

        self.block3a = ConvBlock(3, [128, 128, 512], stage=3, block='a')
        self.block3b = IdentityBlock(3, [128, 128, 512], stage=3, block='b')
        self.block3c = IdentityBlock(3, [128, 128, 512], stage=3, block='c')
        self.block3d = IdentityBlock(3, [128, 128, 512], stage=3, block='d')

        self.block4a = ConvBlock(3, [256, 256, 1024], stage=4, block='a')
        self.block4b = IdentityBlock(3, [256, 256, 1024], stage=4, block='b')
        self.block4c = IdentityBlock(3, [256, 256, 1024], stage=4, block='c')
        self.block4d = IdentityBlock(3, [256, 256, 1024], stage=4, block='d')
        self.block4e = IdentityBlock(3, [256, 256, 1024], stage=4, block='e')
        self.block4f = IdentityBlock(3, [256, 256, 1024], stage=4, block='f')

        self.block5a = ConvBlock(3, [512, 512, 2048], stage=5, block='a')
        self.block5b = IdentityBlock(3, [512, 512, 2048], stage=5, block='b')
        self.block5c = IdentityBlock(3, [512, 512, 2048], stage=5, block='c')

        if include_top:
            self.global_pool = layers.GlobalAveragePooling2D(name='avg_pool')
            self.fc1000 = layers.Dense(classes, activation='softmax', name='fc1000')
        else:
            if pooling == 'avg':
                self.global_pool = layers.GlobalAveragePooling2D()
            elif pooling == 'max':
                self.global_pool = layers.GlobalMaxPooling2D()

        # Load weights.
        if weights == 'imagenet':
            if include_top:
                weights_path = keras_utils.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = keras_utils.get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path)
            if backend.backend() == 'theano':
                keras_utils.convert_all_kernels_in_model(model)
        elif weights is not None:
            self.load_weights(weights)

    def __call__(self, input_tensor):

        x = self.conv1_pad(input_tensor)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.relu1(x)
        x = self.pool1_pad(x)
        x = self.pool1(x)

        x = self.block2a(x)
        x = self.block2b(x)
        x = self.block2c(x)

        x = self.block3a(x)
        x = self.block3b(x)
        x = self.block3c(x)
        x = self.block3d(x)

        x = self.block4a(x)
        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block4d(x)
        x = self.block4e(x)
        x = self.block4f(x)

        x = self.block5a(x)
        x = self.block5b(x)
        x = self.block5c(x)

        if self.include_top:
            x = self.global_pool(x)
            x = self.fc1000(x)
        else:
            x = self.global_pool(x)
        return x


