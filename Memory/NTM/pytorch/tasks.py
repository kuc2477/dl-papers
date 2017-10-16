import abc
import time
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Task(metaclass=abc.ABCMeta):
    name = NotImplemented

    @abc.abstractmethod
    def data_loader(self, batch_size):
        raise NotImplementedError

    @abc.abstractproperty
    def model_input_size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def model_output_size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def model_output_activation(self):
        raise NotImplementedError

    @abc.abstractproperty
    def criterion(self):
        raise NotImplementedError


class Copy(Task):
    name = 'copy'

    def __init__(self,
                 sequence_width=8,
                 sequence_length_min=1,
                 sequence_length_max=20):
        self.sequence_width = sequence_width
        self.sequence_length_min = sequence_length_min
        self.sequence_length_max = sequence_length_max

    def data_loader(self, batch_size):
        random.seed(time.time())
        while True:
            sequence_length = random.randint(
                self.sequence_length_min,
                self.sequence_length_max,
            )

            # sequences.
            sequences = torch.from_numpy(np.random.binomial(1, 0.5, (
                batch_size, sequence_length, self.sequence_width
            )))

            # sequences with delimiters and a separate channel for them.
            x = torch.zeros(
                batch_size, sequence_length+1, self.model_input_size
            )
            x[:, :sequence_length, :self.sequence_width] = sequences
            x[:, sequence_length, self.sequence_width] = 1.

            yield x.float(), sequences.float()

    @property
    def model_input_size(self):
        return self.sequence_width + 1

    @property
    def model_output_size(self):
        return self.sequence_width

    @property
    def model_output_activation(self):
        return F.sigmoid

    @property
    def criterion(self):
        return nn.BCELoss()


class RepeatCopy(Task):
    name = 'repeat_copy'

    def __init__(self,
                 sequence_width=8,
                 sequence_length_min=1,
                 sequence_length_max=10,
                 repeat_min=1,
                 repeat_max=10):
        self.sequence_width = sequence_width
        self.sequence_length_min = sequence_length_min
        self.sequence_length_max = sequence_length_max
        self.repeat_min = repeat_min
        self.repeat_max = repeat_max

    def data_loader(self, batch_size):
        random.seed(time.time())
        while True:
            sequence_length = random.randint(
                self.sequence_length_min,
                self.sequence_length_max,
            )
            repeat = random.randint(self.repeat_min, self.repeat_max)
            repeat_mean = (self.repeat_max + self.repeat_min) / 2
            repeat_var = ((self.repeat_max-self.repeat_min+1)**2 - 1) / 12
            repeat_normalized = (repeat - repeat_mean) / np.sqrt(repeat_var)

            # sequences.
            sequences = torch.from_numpy(np.random.binomial(1, 0.5, (
                batch_size, sequence_length, self.sequence_width
            )))

            # sequences with delimiters, repeats, and separate channels for
            # them.
            x = torch.zeros(
                batch_size, sequence_length+2, self.model_input_size
            )
            x[:, :sequence_length, :self.sequence_width] = sequences
            x[:, sequence_length, self.sequence_width] = 1.
            x[:, sequence_length+1, self.sequence_width+1] = repeat_normalized

            # repeated sequences.
            y = torch.zeros(
                batch_size,
                sequence_length*repeat+1,
                self.model_output_size
            )
            y[:, :sequence_length*repeat, :self.sequence_width] = \
                sequences.repeat(1, repeat, 1)
            y[:, sequence_length*repeat, self.sequence_width] = 1.

            yield x.float(), y.float()

    @property
    def model_input_size(self):
        return self.sequence_width + 2

    @property
    def model_output_size(self):
        return self.sequence_width + 1

    @property
    def model_output_activation(self):
        return F.sigmoid

    @property
    def criterion(self):
        return nn.BCELoss()


TASKS = {
    Copy.name: Copy,
    RepeatCopy.name: RepeatCopy
}
