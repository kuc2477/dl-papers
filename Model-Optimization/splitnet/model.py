import abc
import functools
import operator
from torch import nn


class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class SequentialModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = None

    @abc.abstractproperty
    def layers(self):
        raise NotImplementedError

    def build(self):
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.sequential(x)


# ================
# Concrete Modules
# ================

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()

        # 1
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 2
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # transformation
        self.need_transform = input_channels != output_channels
        self.conv_transform = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

    def forward(self, x):
        x_nonlinearity_applied = self.relu1(self.bn1(x))
        y = self.conv1(x_nonlinearity_applied)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y.add_(self.conv_transform(x) if self.need_transform else x)


class ResidualBlockGroup(SequentialModule):
    def __init__(self, block_number, input_channels, output_channels, stride):
        super().__init__()
        self.block_number = block_number
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.build()

    @property
    def layers(self):
        return [
            ResidualBlock(
                self.input_channels if i == 0 else self.output_channels,
                self.output_channels,
                self.stride if i == 0 else 1
            ) for i in range(self.block_number)
        ]


class WideResNet(SequentialModule):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None):
        super().__init__()

        # model name label.
        self.label = label

        # data specific hyperparameters.
        self.input_size = input_size
        self.input_channels = input_channels
        self.classes = classes

        # model hyperparameters.
        self.total_block_number = total_block_number
        self.widen_factor = widen_factor
        self.baseline_strides = baseline_strides or [1, 1, 2, 2]
        self.baseline_channels = baseline_channels or [16, 16, 32, 64]
        self.widened_channels = [
            w*widen_factor if i != 0 else w for i, w in
            enumerate(self.baseline_channels)
        ]
        self.group_number = len(self.widened_channels) - 1

        # validate total block number.
        assert len(self.baseline_channels) == len(self.baseline_strides)
        assert (
            self.total_block_number % (2*self.group_number) == 0 and
            self.total_block_number // (2*self.group_number) >= 1
        ), 'Total number of residual blocks should be multiples of 2 x N.'

        # build the sequential model.
        self.build()

    @property
    def name(self):
        return (
            'WRN-{depth}-{widen_factor}-'
            '{label}-{size}x{size}x{channels}'
        ).format(
            depth=(self.total_block_number+4),
            widen_factor=self.widen_factor,
            size=self.input_size,
            channels=self.input_channels,
            label=self.label,
        )

    @property
    def layers(self):
        # define group configurations.
        blocks_per_group = self.total_block_number // self.group_number
        zipped_group_channels_and_strides = zip(
            self.widened_channels[:-1],
            self.widened_channels[1:],
            self.baseline_strides[1:]
        )

        # convolution layer.
        conv = nn.Conv2d(
            self.input_channels, self.widened_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # residual block groups.
        residual_block_groups = [
            ResidualBlockGroup(blocks_per_group, i, o, s) for
            i, o, s in zipped_group_channels_and_strides
        ]

        # batchnorm & nonlinearity & pooling.
        bn = nn.BatchNorm2d(self.widened_channels[self.group_number])
        relu = nn.ReLU(inplace=True)
        pool = nn.AvgPool2d(
            self.input_size //
            functools.reduce(operator.mul, self.baseline_strides)
        )

        # classification scores from linear combinations of features.
        view = LambdaModule(lambda x: x.view(-1, self.widened_channels[-1]))
        fc = nn.Linear(self.widened_channels[self.group_number], self.classes)

        # the final model structure.
        return [conv, *residual_block_groups, pool, bn, relu, view, fc]
