from functools import reduce
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=400,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        # Layers.
        self.layers = nn.ModuleList([
            # input
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
            # output
            nn.Linear(self.hidden_size, self.output_size)
        ])

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

    def estimate_fisher(self, x, y):
        loglikelihood = F.log_softmax(self(x))[:, y].mean()
        loglikelihood_grads = autograd.grad(loglikelihood, self.parameters())
        parameter_names = [n for n, p in self.named_parameters()]
        return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            self.register_buffer('{}_estimated_mean'.format(n), p.data)
            self.register_buffer('{}_estimated_cramer_rao_lower_bound'
                                 .format(n), fisher[n].data)

    def ewc_loss(self):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and crlb.
                mean = getattr(self, '{}_estimated_mean'.format(n))
                crlb = getattr(self, '{}_estimated_cramer_rao_lower_bound'
                               .format(n))
                # wrap mean and crlb in variables.
                mean = Variable(mean)
                crlb = Variable(crlb)
                # calculate a ewc loss.
                losses.append((crlb * (p-mean)**2).sum())
            return sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return 0
