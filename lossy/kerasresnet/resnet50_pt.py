from collections import OrderedDict
from torch import nn


class ResNet(nn.Module):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet and ResNeXt).
      use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
      model_name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    """

    def __init__(
            self,
            stack_fn,
            preact,
            use_bias,
            model_name="resnet",
            include_top=True,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
            **kwargs):
        super().__init__()
        self.stack_fn = stack_fn
        self.conv1_conv = nn.Conv2d(3, 64, 7, padding=3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64, eps=1.001e-5)
        self.conv1_relu = nn.ReLU()
        self.pool1_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.preact = preact
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1_conv(x)

        assert not self.preact
        if not self.preact:
            x = self.conv1_bn(x)
            x = self.conv1_relu(x)

        x = self.pool1_pool(x)
        x = self.stack_fn(x)

        # if preact:
        #    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="post_bn")(x)
        #    x = layers.Activation("relu", name="post_relu")(x)

        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        return x


def resnet50(
        include_top=True,
        pooling=None,
        classes=1000,
        **kwargs
):
    """Instantiates the ResNet50 architecture."""

    stack_fn = nn.Sequential(
        stack1(64, 64, 3, stride1=1, name="conv2"),
        stack1(256, 128, 4, name="conv3"),
        stack1(512, 256, 6, name="conv4"),
        stack1(1024, 512, 3, name="conv5"),
    )

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet50",
        include_top,
        pooling,
        classes,
        **kwargs
    )


class Block1(nn.Module):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """

    def __init__(self, in_filters, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        super().__init__()
        in_filters
        if conv_shortcut:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_filters, 4 * filters, 1, stride=stride, )),
                ('bn', nn.BatchNorm2d(4 * filters, eps=1.001e-5),)
            ]))
        else:
            self.shortcut = nn.Identity()

        self.conv1x1 = nn.Conv2d(in_filters, filters, 1, stride=stride)
        self.bn1 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.relu1 = nn.ReLU()
        self.conv_normal = nn.Conv2d(filters, filters, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(filters, eps=1.001e-5)
        self.relu2 = nn.ReLU()
        self.conv_unbottle = nn.Conv2d(filters, 4 * filters, 1, )
        self.bn3 = nn.BatchNorm2d(4 * filters, eps=1.001e-5)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        shortcutted = self.shortcut(x)
        x = self.relu1(self.bn1(self.conv1x1(x)))
        x = self.relu2(self.bn2(self.conv_normal(x)))
        x = self.bn3(self.conv_unbottle(x))
        x = x + shortcutted
        x = self.relu3(x)
        return x


def stack1(in_filters, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Args:
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    modules = [Block1(in_filters, filters, stride=stride1, name=name + "_block1")]

    for i in range(2, blocks + 1):
        modules.append(Block1(4 * filters, filters, conv_shortcut=False, name=name + "_block" + str(i)))
    return nn.Sequential(*modules)
