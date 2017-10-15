import os.path
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import DATASETS


MODEL_PATH = 'models'


def get_data_loader(config):
    return DataLoader(
        DATASETS[config.dataset],
        batch_size=config.batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if config.cuda else {})
    )


def train_model(model, config):
    # prepare optimizer and model
    model.train()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr, momentum=config.momentum
    )

    for epoch in range(1, config.epoch + 1):
        train_loss = 0
        data_loader = get_data_loader(config)
        stream = tqdm(enumerate(data_loader))

        for batch_index, (data, _) in stream:
            # prepare data on gpu if needed
            data = Variable(data).cuda() if config.cuda else Variable(data)

            # flush gradients and run the model forward
            optimizer.zero_grad()
            x_reconstructed, loss = model(data)

            # accumulate train loss
            train_loss += loss.data[0]

            # backprop gradients from the loss
            loss.backward()
            optimizer.step()

            # update progress
            stream.set_description((
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
    path = os.path.join(MODEL_PATH, model.name)
    print('saving model to the path {}'.format(path))
    torch.save(model.state_dict(), path)
