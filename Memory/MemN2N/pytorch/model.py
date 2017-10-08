from torch import nn


class Memory(nn.Module):
    def __init__(self,
                 vocabulary_size, embedding_size,
                 sentence_size, memory_size,
                 embedding=None,
                 embedding_temporal=None):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.embedding = embedding or nn.Embedding(
            vocabulary_size, embedding_size)
        self.embedding_temporal = embedding_temporal or nn.Embedding(
            memory_size, embedding_size
        )

    def forward(self, x):
        return (
            self.position_encoding(x) * self.embedding(x) +
            self.temporal_encoding(x)
        )

    def position_encoding(self, x):
        # TODO: NOT IMPLEMENTED YET
        pass

    def temporal_encoding(self, x):
        # TODO: NOT IMPLEMENTED YET
        pass


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

    def forward(self, x, query):
        u = self.A.embedding(query)
        m = self.A(x)
        c = self.C(x)
        p = self.softmax(u.dot(m))
        o = p.dot(c)
        return self.softmax(self.linear(o + u))
