from collections import namedtuple
from functools import reduce
from torch import nn


HyperParams = namedtuple('HyperParams', (
    'input_size',
    'input_channel',
    'depth'
))


class SplitNet(nn.Module):
    def __init__(self, hparams):
        super().__init__(self)

        # Hyperparameters
        output_size = hparams.input_size // (2 ** 4)

        # Layers
        self.conv1 = self._conv(hparams.input_channel, 16)
        self.conv2 = self._conv(16, 32)
        self.conv3 = self._conv(32, 64)
        self.conv4 = self._conv(64, 128)
        self.conv5 = self._conv(128, 1, stride=1)
        self.fc1 = self._fc(self.output_size, 256)
        self.fc2 = self._fc(256, 256)

    def forward(self, x):
        return reduce(lambda x, f: f(x), [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.fc1,
            self.fc2
        ], x)

    def _res(self):
        """Residual block factory"""
        # TODO: NOT IMPLEMENTED YET
        pass

    def _conv(
            self,
            input_channels,
            output_channels,
            kernel_size=3, padding=1, stride=2,
            relu=True, bn=True, pool=True):
        """Convolutional layer factory"""
        return nn.Sequential(*filter(bool, [
            nn.Conv2d(
                input_channels, output_channels, kernel_size,
                padding=padding, stride=stride
            ),
            nn.ReLU() if relu else None,
            nn.BatchNorm2d(output_channels) if bn else None,
            nn.MaxPool2d(2) if pool else None,
        ]))

    def _fc(self, input_features, output_features, relu=True):
        """Affine layer factory"""
        return nn.Sequential(*filter(bool, [
            nn.Linear(input_features, output_features),
            nn.ReLU() if relu else None
        ]))
