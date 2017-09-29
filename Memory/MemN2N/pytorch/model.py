from torch import nn


class MemN2N(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, memory_size, hops):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.hops = hops

    def forward(x):
        pass
