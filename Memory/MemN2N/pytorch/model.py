from torch import nn, Tensor, LongTensor
from torch.autograd import Variable


class Memory(nn.Module):
    def __init__(self,
                 vocabulary_size, sentence_size,
                 memory_size, embedding_size,
                 embedding=None, embedding_temporal=None):
        super().__init__()
        # Memory configurations.
        self.vocabulary_size = vocabulary_size
        self.sentence_size = sentence_size
        self.memory_size = memory_size
        self.embedding_size = embedding_size

        # Memory Embeddings.
        self.embedding = (
            embedding or
            nn.Embedding(vocabulary_size, embedding_size)
        )
        self.temporal_embedding = (
            embedding_temporal or
            nn.Embedding(memory_size, embedding_size)
        )

    def _embedded(self, x):
        return self\
            .embedding(x.view(-1, self.sentence_size))\
            .view(
                -1,
                self.memory_size,
                self.sentence_size,
                self.embedding_size
            )

    def _position_encoding(self):
        encoding = Variable(Tensor(self.embedding_size, self.sentence_size))
        for i, j in [(i, j) for
                     i in range(self.embedding_size) for
                     j in range(self.sentence_size)]:
            encoding[i, j] = (
                ((i+1) - (self.embedding_size+1)/2) *
                ((j+1) - (self.sentence_size+1)/2)
            )
        encoding *= 4 / (self.embedding_size * self.sentence_size)
        encoding[:, -1] = 1.
        return encoding.t()

    def _temporal_encoding(self):
        time = Variable(LongTensor(range(self.memory_size)))
        return self.temporal_embedding(time)

    def forward(self, x):
        return (
            (self._position_encoding() * self._embedded(x)).sum(2) +
            self._temporal_encoding()
        )


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
        self.A = Memory(
            self.vocabulary_size,
            self.sentence_size,
            self.embedding_size,
            self.memory_size
        )

        self.C = Memory(
            self.vocabulary_size,
            self.sentence_size,
            self.embedding_size,
            self.memory_size
        )

    def forward(self, x, q):
        u = self.A.embedding(q)
        m = self.A(x)
        c = self.C(x)
        p = self.softmax(u.dot(m))
        o = p.dot(c)
        return self.softmax(self.linear(o + u))
