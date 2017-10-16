from functools import reduce
import operator
import numpy as np
import pytest
import torch
from torch.autograd import Variable
from model import Controller, ReadHead, WriteHead, Memory, NTM
import tasks


# ============================
# Model Configuration Fixtures
# ============================

@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def input_length():
    return 20


@pytest.fixture
def embedding_size():
    return 10


@pytest.fixture
def hidden_size():
    return 15


@pytest.fixture
def output_size():
    return 10


@pytest.fixture
def memory_size():
    return 30


@pytest.fixture
def memory_feature_size():
    return 15


@pytest.fixture
def dictionary_size():
    return 20


@pytest.fixture
def max_shift_size():
    return 2


@pytest.fixture
def batch(batch_size, embedding_size, input_length):
    samples = []
    for _ in range(batch_size):
        samples.append([
            np.random.binomial(1, 0.5, embedding_size) for
            _ in range(input_length)
        ])
    return Variable(torch.from_numpy(np.array(samples)).float())


# ========================
# Model Component Fixtures
# ========================

@pytest.fixture
def controller(embedding_size, hidden_size):
    return Controller(embedding_size, hidden_size)


@pytest.fixture
def read_head(hidden_size, memory_size, memory_feature_size, max_shift_size):
    return ReadHead(
        hidden_size=hidden_size,
        memory_size=memory_size,
        memory_feature_size=memory_feature_size,
        max_shift_size=max_shift_size
    )


@pytest.fixture
def write_head(hidden_size, memory_size, memory_feature_size, max_shift_size):
    return WriteHead(
        hidden_size=hidden_size,
        memory_size=memory_size,
        memory_feature_size=memory_feature_size,
        max_shift_size=max_shift_size,
    )


@pytest.fixture
def memory(memory_size, memory_feature_size):
    return Memory(memory_size, memory_feature_size)


@pytest.fixture
def ntm(embedding_size, hidden_size, output_size,
        memory_size, memory_feature_size, max_shift_size):
    return NTM(
        label='test',
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        output_size=output_size,
        memory_size=memory_size,
        memory_feature_size=memory_feature_size,
        max_shift_size=max_shift_size,
    )


# ======================
# Task Specific Fixtures
# ======================

@pytest.fixture
def copy_task():
    return tasks.Copy()


@pytest.fixture
def ntm_copy(copy_task, hidden_size,
             memory_size, memory_feature_size,
             max_shift_size):
    return NTM(
        label=copy_task.name,
        embedding_size=copy_task.model_input_size,
        hidden_size=hidden_size,
        output_size=copy_task.model_output_size,
        memory_size=memory_size,
        memory_feature_size=memory_feature_size,
        max_shift_size=max_shift_size,
    )


# =====================
# Model Component Tests
# =====================

def test_controller(controller, input_length, embedding_size, hidden_size,
                    batch_size, batch):
    # test reset
    assert batch.size() == (batch_size, input_length, embedding_size)
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
    assert len(read_head.split_hidden_state(h)) == 5
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
    assert len(write_head.split_hidden_state(h)) == 7
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


def test_ntm(ntm, output_size, batch_size, batch):
    # test reset
    assert ntm.expected_batch_size is None
    ntm.reset(batch_size)
    assert ntm.expected_batch_size == batch_size
    # test forward
    assert ntm(batch[:, 0]).size() == (batch_size, output_size)
    assert ntm().size() == (batch_size, output_size)
    # check if gradient is flowing
    assert all([p.grad is None for p in ntm.parameters()])
    ntm().sum().backward()
    assert all([p.grad is not None for p in ntm.parameters()])


def test_copy_task(ntm_copy, copy_task, batch_size):
    # prepare the data loader and reset NTM state variables.
    data_loader = copy_task.data_loader(batch_size)
    x, y = next(data_loader)
    x, y = Variable(x), Variable(y)
    ntm_copy.reset(batch_size)

    # run the model to take input sequences.
    for index in range(x.size(1)):
        ntm_copy(x[:, index, :])

    # run the model to output sequences.
    predictions = []
    for index in range(y.size(1)):
        activation = copy_task.model_output_activation(ntm_copy())
        predictions.append(activation.round())

    predictions = torch.stack(predictions, 1).long()
    precision = (
        (predictions == y).sum().data[0] /
        reduce(operator.mul, predictions.size())
    )

    assert isinstance(precision, float)
    assert 0 <= precision <= 1
