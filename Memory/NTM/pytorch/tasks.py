import abc
from torch import nn


class Task(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def data_loader(self, batch_size):
        raise NotImplementedError

    @abc.abstractproperty
    def criterion(self):
        raise NotImplementedError

    @abc.abstractproperty
    def name(self):
        raise NotImplementedError


class Copy(Task):
    def __init__(self,
                 sequence_width=8,
                 sequence_length_min=1,
                 sequence_length_max=20):
        self.sequence_width = sequence_width
        self.sequence_length_min = sequence_length_min
        self.sequence_length_max = sequence_length_max

    def data_loader(self, batch_size):
        # TODO: NOT IMPLEMENTED YET
        pass

    @property
    def criterion(self):
        return nn.BCELoss()

    @property
    def name(self):
        return 'copy'


class RepeatCopy(Task):
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
        # TODO: NOT IMPLEMENTED YET
        pass

    @property
    def criterion(self):
        return nn.BCELoss()

    @property
    def name(self):
        return 'repeat_copy'


TASKS = {
    Copy.name: Copy,
    RepeatCopy.name: RepeatCopy
}
