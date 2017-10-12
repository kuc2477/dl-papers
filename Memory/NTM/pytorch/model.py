import abc
import operator as op
from functools import reduce
import torch
from torch import Tensor
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from const import EPSILON


class StatefulComponent(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self._expected_batch_size = None

    @abc.abstractmethod
    def reset(self, batch_size):
        """
        Resets the state of the component. Note that all initialized
        state variables must fit into the given batch size.
        """
        self._expected_batch_size = batch_size

    @property
    def expected_batch_size(self):
        return self._expected_batch_size


# ==============
# NTM Components
# ==============

class Controller(StatefulComponent):
    def __init__(self, dictionary_size, embedding_size, hidden_size):
        # Configurations.
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Layers.
        self.embedding = nn.Embedding(
            self.dictionary_size,
            self.embedding_size
        )
        self.cell = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.h = None
        self.c = None

    def forward(self, x):
        self.h, self.c = self.cell(self.embedding(x), (self.h, self.c))
        return self.h

    def reset(self, batch_size):
        super().reset(batch_size)
        self.h = Variable(Tensor(batch_size, self.hidden_size))
        self.c = Variable(Tensor(batch_size, self.hidden_size))


class Head(StatefulComponent):
    def __init__(self,
                 hidden_size,
                 memory_size,
                 memory_feature_size,
                 max_shift_size):
        # Configurations
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_feature_size = memory_feature_size
        self.max_shift_size = max_shift_size

        # Interprets a hidden state passed from a controller
        self.interpreter = nn.Linear(
            self.hidden_size,
            sum([s for s, _ in self.interpreter_splitting_scheme()])
        )

        # Current head position
        self.w = None

    @abc.abstractmethod
    def forward(self, h, m):
        raise NotImplementedError

    @abc.abstractmethod
    def interpreter_splitting_scheme(self):
        raise NotImplementedError

    def interpret(self, h):
        interpreted = self.interpreter(h)
        splits, activations = zip(*self.interpreter_splitting_scheme())
        splits = [
            reduce(op.add, [0, *splits][:i+1]) for i in
            range(len(splits)+1)
        ]
        return tuple(
            activation(interpreted[:, s:e]) for s, e, activation in
            zip(splits[:-1], splits[1:], activations)
        )

    def move(self, m, k, b, g, s, r):
        w = self._find_by_content_addressing(m, k, b)
        w = self._interpolate_with_current_position(w, g)
        w = self._shift_by_location_addressing(w, s)
        w = self.w = self._sharpen(w, r)
        return w

    def reset(self, batch_size):
        super().reset(batch_size)
        self.w = Variable(Tensor(batch_size, self.memory_size))

    def _interpolate_with_current_position(self, w, g):
        return w + (1-g)*self.w

    def _find_by_content_addressing(self, m, k, b):
        entire_memory_size = self.expected_batch_size * self.memory_size
        m_unrolled = m.view(entire_memory_size, -1)
        k_unrolled = k.unsqueeze(1).expand_as(m).view(entire_memory_size, -1)

        similarity_between_key_and_memories = F\
            .cosine_similarity(m_unrolled + EPSILON, k_unrolled + EPSILON)\
            .view(-1, self.memory_size)

        return F.softmax(b * similarity_between_key_and_memories)

    def _shift_by_location_addressing(self, w, s):
        shifted = Variable(Tensor(*w.size()))
        for b in range(self.expected_batch_size):
            wb, sb = w[b], s[b]
            # unroll the head positions so that we can convolve with.
            wb_modulo_unrolled = torch.cat(
                wb[-self.max_shift_size:],
                wb,
                wb[:self.max_shift_size]
            )
            # convolve head positions with the shifts.
            shifted[b, :] = F.conv1d(
                wb_modulo_unrolled.view(1, 1, -1),
                sb.view(1, 1, -1)
            ).view(-1)
        return shifted

    def _sharpen(self, w, r):
        return (w**r) / ((w**r).sum(1).view(-1, 1) + EPSILON)


class ReadHead(Head):
    def forward(self, h, m):
        k, b, g, s, r = self.interpret(h)
        return self.move(m, k, b, g, s, r)

    def interpreter_splitting_scheme(self):
        return [
            (self.memory_feature_size, lambda x: x),     # k
            (1, F.softplus),                        # b
            (1, F.sigmoid),                         # g
            (2*self.max_shift_size+1, F.softmax),   # s
            (1, lambda x: 1 + F.softplus(x))        # r
        ]


class WriteHead(Head):
    def forward(self, h, m):
        k, b, g, s, r, e, a = self.interpret(h)
        return self.move(m, k, b, g, s, r), e, a

    def interpreter_splitting_scheme(self):
        return [
            (self.memory_feature_size, lambda x: x),     # k
            (1, F.softplus),                        # b
            (1, F.sigmoid),                         # g
            (2*self.max_shift_size+1, F.softmax),   # s
            (1, lambda x: 1 + F.softplus(x))        # r,
            (self.memory_feature_size, F.sigmoid)        # e
            (self.memory_feature_size, lambda x: x)      # a
        ]


class Memory(StatefulComponent):
    def __init__(self, memory_size, memory_feature_size):
        # Configurations.
        super().__init__()
        self.memory_size = memory_size
        self.memory_feature_size = memory_feature_size

        # Memory bank.
        self.bank = None

    def read(self, w):
        return (w * self.bank).sum(1)

    def write(self, w, e, a):
        self.bank = self.bank * (1 - w*e)
        self.bank = self.bank + w*a

    def reset(self, batch_size):
        super().reset(batch_size)
        self.bank = Variable(Tensor(
            batch_size,
            self.memory_size,
            self.memory_feature_size,
        ))


class NTM(StatefulComponent):
    def __init__(self,
                 dictionary_size, embedding_size, hidden_size,
                 memory_size, memory_feature_size, max_shift_size):
        # Configurations.
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_feature_size = memory_feature_size
        self.max_shift_size = max_shift_size

        # Components.
        self.controller = Controller(
            self.dictionary_size,
            self.embedding_size,
            self.hidden_size,
        )
        self.read_head = ReadHead(
            self.hidden_size,
            self.memory_size,
            self.memory_feature_size,
            self.max_shift_size,
        )
        self.write_head = WriteHead(
            self.hidden_size,
            self.memory_size,
            self.memory_feature_size,
            self.max_shift_size,
        )
        self.memory = Memory(self.memory_size, self.memory_feature_size)
        self.linear = nn.Linear(self.memory_feature_size, self.dictionary_size)

    def forward(self, x=None, return_read_memory=False):
        if self.expected_batch_size is None:
            raise RuntimeError(
                'NTM tape is not ready. '
                'You need to initialize the NTM tape before running it.'
            )

        if x is not None and x.size(0) != self.expected_batch_size:
            raise RuntimeError((
                'Input batch size and NTM tape size does not match. '
                'Given batch is size of {}. Batch size {} is expected.'
            ).format(x.size(0), self.expected_batch_size))

        h = self.controller(x)
        self.memory.write(*self.write_head(h, self.memory.bank))
        r = self.memory.read(*self.read_head(h, self.memory.bank))
        return r if return_read_memory else self.linear(r)

    def reset(self, batch_size):
        super().reset(batch_size)
        self.controller.reset(batch_size)
        self.read_head.reset(batch_size)
        self.write_head.reset(batch_size)
        self.memory.reset(batch_size)
