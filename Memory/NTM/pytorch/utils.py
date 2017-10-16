from functools import reduce
import operator
import os
import os.path
import shutil
import torch
from torch.autograd import Variable
from torch.nn import init


def save_checkpoint(model, model_dir, iteration, precision, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'state': model.state_dict(),
        'iteration': iteration,
        'precision': precision,
    }, path)

    # override the best model if it's the best.
    if best:
        shutil.copy(path, path_best)
        print('=> updated the best model of {name} at {path}'.format(
            name=model.name, path=path_best
        ))

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir, best=True):
    path = os.path.join(model_dir, model.name)
    path_best = os.path.join(model_dir, '{}-best'.format(model.name))

    # load the checkpoint.
    checkpoint = torch.load(path_best if best else path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path_best if best else path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    iteration = checkpoint['iteration']
    precision = checkpoint['precision']
    return iteration, precision


def validate(model, task,
             test_size=256, batch_size=32,
             cuda=False, verbose=True):
    data_loader = task.data_loader(batch_size)
    total_tested = 0
    total_bits = 0
    wrong_bits = 0

    for x, y in data_loader:
        # break on test size.
        if total_tested >= test_size:
            break

        # prepare the batch and reset the model's state variables.
        model.reset(x.size(0), cuda=cuda)
        x = Variable(x).cuda() if cuda else Variable(x)
        y = Variable(y).cuda() if cuda else Variable(y)

        # run the model through the sequences.
        for index in range(x.size(1)):
            model(x[:, index, :])

        # run the model to output sequences.
        predictions = []
        for index in range(y.size(1)):
            activation = task.model_output_activation(model())
            predictions.append(activation.round())
        predictions = torch.stack(predictions, 1).long()

        # calculate the wrong bits per sequence.
        total_tested += x.size(0)
        total_bits += reduce(operator.mul, y.size())
        wrong_bits += torch.abs(predictions-y.long()).sum().data[0]

    precision = 1 - wrong_bits/total_bits
    if verbose:
        print('=> precision: {prec:.4}'.format(prec=precision))
    return precision


def xavier_initialize(model, uniform=False):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p) if uniform else init.xavier_normal(p)


def gaussian_intiailize(model, std=.1):
    for p in model.parameters():
        init.normal(p, std=std)
