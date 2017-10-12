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


@pytest.fixture
def ntm(dictionary_size, embedding_size, hidden_size,
        memory_size, memory_feature_size, max_shift_size):
    return NTM(
        dictionary_size,
        embedding_size,
        hidden_size,
        memory_size,
        memory_feature_size,
        max_shift_size,
    )


# =====================
# Model Component Tests
# =====================

def test_controller(controller, input_length, hidden_size, batch_size, batch):
    # test reset
    assert batch.size() == (batch_size, input_length)
    assert controller.h is None and controller.c is None
    controller.reset(batch_size)
    assert controller.expected_batch_size == batch_size
    assert controller.h is not None and controller.c is not None
    # test forward
    assert controller(batch[:, 0]).size() == (batch_size, hidden_size)


def test_read_head(controller, read_head, memory, batch_size, batch):
    # test reset
    assert read_head.w is None
    controller.reset(batch_size)
    read_head.reset(batch_size)
    memory.reset(batch_size)
    assert read_head.expected_batch_size == batch_size
    assert read_head.w is not None
    # test interpret, move and forward
    h = controller(batch[:, 0])
    assert len(read_head.interpret(h)) == 5
    assert read_head(h, memory.bank).size() == (batch_size, memory.memory_size)


def test_write_head(controller, write_head, memory, batch_size, batch):
    # test reset
    assert write_head.w is None
    controller.reset(batch_size)
    write_head.reset(batch_size)
    memory.reset(batch_size)
    assert write_head.expected_batch_size == batch_size
    assert write_head.w is not None
    # test interpret, move and forward
    h = controller(batch[:, 0])
    assert len(write_head.interpret(h)) == 7
    w, e, a = write_head(h, memory.bank)
    assert w.size() == (batch_size, memory.memory_size)
    assert e.size() == (batch_size, memory.memory_feature_size)
    assert a.size() == (batch_size, memory.memory_feature_size)


def test_memory(controller, read_head, write_head, memory,
                batch_size, batch):
    # test reset
    assert memory.bank is None
    assert memory.expected_batch_size is None
    controller.reset(batch_size)
    write_head.reset(batch_size)
    read_head.reset(batch_size)
    memory.reset(batch_size)
    assert memory.expected_batch_size == batch_size
    assert memory.bank.size() == (
        memory.expected_batch_size,
        memory.memory_size,
        memory.memory_feature_size
    )
    # test read and write
    h = controller(batch[:, 0])
    assert memory.read(read_head(h, memory.bank)).size() == (
        batch_size, memory.memory_feature_size
    )
    memory.write(*write_head(h, memory.bank))


def test_ntm(ntm,
             memory_feature_size,
             dictionary_size,
             batch_size, batch):
    # test reset
    assert ntm.expected_batch_size is None
    ntm.reset(batch_size)
    assert ntm.expected_batch_size == batch_size
    # test forward (with input)
    assert ntm(batch[:, 0], return_read_memory=True).size() == (
        batch_size, memory_feature_size
    )
    assert ntm(batch[:, 0], return_read_memory=False).size() == (
        batch_size, dictionary_size
    )
    # test forward (without input)
    assert ntm(return_read_memory=True).size() == (
        batch_size, memory_feature_size
    )
    assert ntm(return_read_memory=False).size() == (
        batch_size, dictionary_size
    )
