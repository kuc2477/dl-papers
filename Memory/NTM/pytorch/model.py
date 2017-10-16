import abc
import operator as op
from functools import reduce
import torch
from torch import Tensor
from torch.cuda import FloatTensor as CudaTensor
from torch.autograd import Variable
from torch import nn
from torch.nn import (
    functional as F,
    Parameter
)
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
    def __init__(self, embedding_size, hidden_size, dictionary_size=None):
        # Configurations
        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Embedding layer (optional)
        self.embedding = nn.Embedding(
            self.dictionary_size,
            self.embedding_size
        ) if dictionary_size else None

        # LSTM cell to extract features from input
        self.cell = nn.LSTMCell(self.embedding_size, self.hidden_size)

        # Learnable LSTM hidden state biases
        self.h_bias = Parameter(Tensor(self.hidden_size).normal_())
        self.c_bias = Parameter(Tensor(self.hidden_size).normal_())

        # States
        self.h = None
        self.c = None

    def forward(self, x):
        # supply the lstm cell with zeros in the embedding space. (no input)
        if x is None:
            e = Variable(torch.zeros(
                self.expected_batch_size,
                self.embedding_size,
            ).type_as(self.h.data))
        # supply the lstm cell with an embedded input.
        elif self.embedding:
            e = self.embedding(x)
        # supply the lstm cell with the input as-is. (assuming that the input
        # is already in the embedding space)
        else:
            assert x.size() == (
                self.expected_batch_size,
                self.embedding_size
            ), 'Input should have size of {b}x{e}, while given {s}.'.format(
                b=self.expected_batch_size,
                e=self.embedding_size,
                s=x.size(),
            )
            e = x

        # run an lstm cell and update the states.
        self.h, self.c = self.cell(e, (self.h, self.c))
        return self.h

    def reset(self, batch_size, cuda=False):
        super().reset(batch_size)
        dt = CudaTensor if cuda else Tensor
        self.h = self.h_bias.clone().repeat(batch_size, 1).type(dt)
        self.c = self.c_bias.clone().repeat(batch_size, 1).type(dt)


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

        # Projects a hidden state passed from the controller
        self.linear = nn.Linear(
            self.hidden_size,
            sum([s for s, _ in self.hidden_state_splitting_scheme()])
        )

        # Current head position
        self.w = None

    @abc.abstractmethod
    def forward(self, h, m):
        raise NotImplementedError

    @abc.abstractmethod
    def hidden_state_splitting_scheme(self):
        raise NotImplementedError

    def split_hidden_state(self, h):
        projected = self.linear(h)
        splits, activations = zip(*self.hidden_state_splitting_scheme())
        splits = [
            reduce(op.add, [0, *splits][:i+1]) for i in
            range(len(splits)+1)
        ]
        return tuple(
            activation(projected[:, s:e]) for s, e, activation in
            zip(splits[:-1], splits[1:], activations)
        )

    def move(self, m, k, b, g, s, r):
        w = self._find_by_content_addressing(m, k, b)
        w = self._interpolate_with_current_position(w, g)
        w = self._shift_by_location_addressing(w, s)
        w = self.w = self._sharpen(w, r)
        return w

    def reset(self, batch_size, cuda=False):
        super().reset(batch_size)
        self.w = Variable(torch.zeros(batch_size, self.memory_size)).type(
            CudaTensor if cuda else Tensor
        )

    def _interpolate_with_current_position(self, w, g):
        return w + (1-g)*self.w

    def _find_by_content_addressing(self, m, k, b):
        return F.softmax(b * F.cosine_similarity(
            m, k.unsqueeze(1).expand_as(m), 2
        ))

    def _shift_by_location_addressing(self, w, s):
        w_modulo_unrolled = torch.cat([
            w[:, -self.max_shift_size:], w,
            w[:, :self.max_shift_size]
        ], 1)
        return F.conv1d(
            w_modulo_unrolled.view(self.expected_batch_size, 1, -1),
            s.view(self.expected_batch_size, 1, -1)
        )[range(self.expected_batch_size),
          range(self.expected_batch_size), :]

    def _sharpen(self, w, r):
        return (w**r) / ((w**r).sum(1).view(-1, 1) + EPSILON)


class ReadHead(Head):
    def forward(self, h, m):
        k, b, g, s, r = self.split_hidden_state(h)
        return self.move(m, k, b, g, s, r)

    def hidden_state_splitting_scheme(self):
        return [
            (self.memory_feature_size, lambda x: x),    # k
            (1, F.softplus),                            # b
            (1, F.sigmoid),                             # g
            (2*self.max_shift_size+1, F.softmax),       # s
            (1, lambda x: 1 + F.softplus(x)),           # r
        ]


class WriteHead(Head):
    def forward(self, h, m):
        k, b, g, s, r, e, a = self.split_hidden_state(h)
        return self.move(m, k, b, g, s, r), e, a

    def hidden_state_splitting_scheme(self):
        return [
            (self.memory_feature_size, lambda x: x),    # k
            (1, F.softplus),                            # b
            (1, F.sigmoid),                             # g
            (2*self.max_shift_size+1, F.softmax),       # s
            (1, lambda x: 1 + F.softplus(x)),           # r
            (self.memory_feature_size, F.sigmoid),      # e
            (self.memory_feature_size, lambda x: x),    # a
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
        return (w.unsqueeze(2) * self.bank).sum(1)

    def write(self, w, e, a):
        self.bank = self.bank * (1 - w.unsqueeze(2)*e.unsqueeze(1))
        self.bank = self.bank + w.unsqueeze(2)*a.unsqueeze(1)
        return self.bank

    def reset(self, batch_size, cuda=False):
        super().reset(batch_size)
        self.bank = Variable(Tensor(
            batch_size,
            self.memory_size,
            self.memory_feature_size,
        )).type(CudaTensor if cuda else Tensor)
        nn.init.xavier_uniform(self.bank)


class NTM(StatefulComponent):
    def __init__(self, label,
                 embedding_size, hidden_size,
                 memory_size, memory_feature_size,
                 output_size=None, head_num=3, max_shift_size=1,
                 dictionary_size=None, dictionary_hash=None):
        # Validation.
        assert dictionary_size is not None or output_size is not None
        assert ((dictionary_size is None) == (dictionary_hash is None))

        # Configurations.
        super().__init__()
        self.label = label
        self.dictionary_hash = dictionary_hash
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_feature_size = memory_feature_size
        self.output_size = output_size
        self.head_num = head_num
        self.max_shift_size = max_shift_size

        # Components.
        self.controller = Controller(
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            dictionary_size=self.dictionary_size,
        )

        self.read_heads = nn.ModuleList([ReadHead(
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            memory_feature_size=self.memory_feature_size,
            max_shift_size=self.max_shift_size,
        ) for _ in range(self.head_num)])

        self.write_heads = nn.ModuleList([WriteHead(
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            memory_feature_size=self.memory_feature_size,
            max_shift_size=self.max_shift_size,
        ) for _ in range(self.head_num)])

        self.memory = Memory(self.memory_size, self.memory_feature_size)
        self.linear = nn.Linear(
            len(self.read_heads)*self.memory_feature_size + self.hidden_size,
            self.dictionary_size if self.dictionary_size else self.output_size
        )

    @property
    def name(self):
        return ''.join([
            'NTM-lstm-{label}',
            '-dict{dict_size}/{dict_hash}' if self.dictionary_size else '',
            '-embed{embedding_size}',
            '-out{output_size}',
            '-mem{memory_size}x{memory_feature_size}',
            '-head{head_num}',
            '-maxshift{max_shift_size}',
        ]).format(
            label=self.label,
            dict_size=self.dictionary_size,
            dict_hash=self.dictionary_hash,
            embedding_size=self.embedding_size,
            output_size=self.output_size,
            memory_size=self.memory_size,
            memory_feature_size=self.memory_feature_size,
            head_num=self.head_num,
            max_shift_size=self.max_shift_size
        )

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
        r = []
        for read_head, write_head in zip(self.read_heads, self.write_heads):
            self.memory.write(*write_head(h, self.memory.bank))
            r.append(self.memory.read(read_head(h, self.memory.bank)))
        return r if return_read_memory else self.linear(torch.cat([*r, h], 1))

    def reset(self, batch_size, cuda=False):
        super().reset(batch_size)
        self.controller.reset(batch_size, cuda=cuda)
        self.memory.reset(batch_size, cuda=cuda)
        for read_head, write_head in zip(self.read_heads, self.write_heads):
            read_head.reset(batch_size, cuda=cuda)
            write_head.reset(batch_size, cuda=cuda)
