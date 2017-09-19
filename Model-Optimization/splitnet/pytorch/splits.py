import torch
from torch import nn
from torch.autograd import Variable


def split_loss(w, p, q):
    pass


def split_indicator(split_groups, dimension, cuda=True):
    alpha = Variable(
        torch.Tensor(split_groups, dimension).normal_(std=0.01),
        requires_grad=True
    )
    return nn.Softmax()(alpha)


def merge_split_indicator(split_indicator, merge_groups, cuda=True):
    pass
