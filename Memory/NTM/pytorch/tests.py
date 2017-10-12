from random import randint
import pytest
import torch
from torch.autograd import Variable
from model import Controller, ReadHead, WriteHead, Memory, NTM


# ============================
# Model Configuration Fixtures
# ============================

@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def dictionary_size():
    return 30


@pytest.fixture
def input_length():
    return 20


@pytest.fixture
def embedding_size():
    return 10


@pytest.fixture
def memory_size():
    return 30


@pytest.fixture
def memory_feature_size():
    return 15


@pytest.fixture
def hidden_size():
    return 15


@pytest.fixture
def max_shift_size():
    return 2


@pytest.fixture
def batch(batch_size, dictionary_size, input_length):
    samples = []
    for _ in range(batch_size):
        sequence = [randint(0, dictionary_size-1) for _ in range(input_length)]
        samples.append(sequence)
    return Variable(torch.Tensor(samples).long())


# ========================
# Model Component Fixtures
# ========================

@pytest.fixture
def controller(dictionary_size, embedding_size, hidden_size):
    return Controller(dictionary_size, embedding_size, hidden_size)


@pytest.fixture
def read_head(hidden_size, memory_size, memory_feature_size, max_shift_size):
    return ReadHead(
        hidden_size,
        memory_size,
        memory_feature_size,
        max_shift_size
    )


@pytest.fixture
def write_head(hidden_size, memory_size, memory_feature_size, max_shift_size):
    return WriteHead(
        hidden_size,
        memory_size,
        memory_feature_size,
        max_shift_size,
    )


@pytest.fixture
def memory(memory_size, memory_feature_size):
    return Memory(memory_size, memory_feature_size)


# =====================
# Model Component Tests
# =====================

def test_controller(controller, input_length, hidden_size, batch_size, batch):
    assert batch.size() == (batch_size, input_length)
    assert controller.h is None and controller.c is None
    controller.reset(batch_size)
    assert controller.expected_batch_size == batch_size
    assert controller.h is not None and controller.c is not None
    assert controller(batch[:, 0]).size() == (batch_size, hidden_size)


def test_read_head(controller, read_head, memory, batch_size, batch):
    assert read_head.w is None
    controller.reset(batch_size)
    read_head.reset(batch_size)
    memory.reset(batch_size)
    assert read_head.expected_batch_size == batch_size
    assert read_head.w is not None

    h = controller(batch[:, 0])
    assert len(read_head.interpret(h)) == 5
    assert read_head.move(memory.bank, *read_head.interpret(h)).size() == (
        batch_size, memory.memory_size
    )


def test_write_head():
    pass


def test_memory():
    pass
