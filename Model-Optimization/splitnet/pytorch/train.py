import os.path
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


def _get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def train(model, dataset, lr=1e-04, batch_size=32, epochs=5,
          cuda=False, log_interval=100, model_dir='models'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # prepare the model and statistics.
    model.train()
    running_loss = 0

    for epoch in range(1, epochs + 1):
        # prepare the data stream.
        data_loader = _get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (data, labels) in data_stream:
            # clear the gradients.
            optimizer.zero_grad()

            # run & update the network.
            x = Variable(data).cuda() if cuda else Variable(data)
            labels = Variable(labels).cuda() if cuda else Variable(labels)
            scores = model(x)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            # update & display statistics.
            running_loss += loss.data[0]
            if batch_index % log_interval == 0:
                data_stream.set_description((
                    'epoch: {epoch} | '
                    'progress: [{trained}/{dataset}] ({progress:.0f}%) | '
                    'loss: {loss}'
                ).format(
                    epoch=epoch,
                    trained=batch_index * len(data),
                    dataset=len(data_loader.dataset),
                    progress=(100. * batch_index / len(data_loader)),
                    loss=(loss.data[0] / len(data))
                ))

    # save the trained model
    path = os.path.join(model_dir, model.name)
    print('saving the model to the path {}'.format(path))
    torch.save(model.state_dict(), path)
