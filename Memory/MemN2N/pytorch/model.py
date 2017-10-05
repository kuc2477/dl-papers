from torch import Tensor
from torch import nn
from torch.autograd import Variable


class AbstractMemory(nn.Module):
    def __init__(self,
                 vocabulary_size, sentence_size,
                 embedding_size, memory_size,
                 embedding=None):
        self.embedding_size = embedding_size
        self.memory_size = memory_size

        self.softmax = nn.Softmax()
        self.embedding = embedding or nn.Embedding(
            vocabulary_size, embedding_size
        )
        self.memory = Variable(Tensor(
            memory_size, embedding_size
        ).normal_(std=.1))


class InputMemory(AbstractMemory):
    def forward(self, x, query):
        embedded_x = self.embedding(x)
        embedded_query = self.embedding(query)
        return self.softmax(embedded_query.dot(embedded_x))


class OutputMemory(AbstractMemory):
    def forward(self, x, p):
        embedded_x = self.embedding(x)
        return (p * embedded_x).sum()


class MemN2N(nn.Module):
    def __init__(self,
                 vocabulary_size,
                 sentence_size,
                 embedding_size,
                 memory_size, hops):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.sentence_size = sentence_size
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.hops = hops

        self.linear = nn.Linear(self.vocabulary_size, self.embedding_size)
        self.softmax = nn.Softmax()
        self.A = InputMemory(
            self.vocabulary_size,
            self.sentence_size,
            self.embedding_size,
            self.memory_size
        )

        self.C = OutputMemory(
            self.vocabulary_size,
            self.sentence_size,
            self.embedding_size,
            self.memory_size
        )

    def forward(self, x, query):
        u = self.A.embedding(query)
        p = self.A(x, u)
        o = self.C(x, p)
        return self.softmax(self.linear(o + u))
