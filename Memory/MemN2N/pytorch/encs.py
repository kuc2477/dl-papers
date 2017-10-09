from torch.autograd import Variable
from torch import Tensor, LongTensor


def position_encoding(embedding_size, sentence_size):
        encoding = Variable(Tensor(embedding_size, sentence_size))
        for i, j in [(i, j) for
                     i in range(embedding_size) for
                     j in range(sentence_size)]:
            encoding[i, j] = (
                ((i+1) - (embedding_size+1)/2) *
                ((j+1) - (sentence_size+1)/2)
            )
        encoding *= 4 / (embedding_size * sentence_size)
        encoding[:, -1] = 1.
        return encoding.t()


def temporal_encoding(memory_size, embedding):
    time = Variable(LongTensor(range(memory_size)))
    return embedding(time)
