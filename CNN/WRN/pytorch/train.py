from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm
import utils


def train(model, dataset, dataset_test=None, model_dir='models',
          lr=1e-04, batch_size=32, test_size=16, epochs=5,
          checkpoint_interval=500, resume=False, cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # prepare the model and statistics.
    model.train()
    epoch_start = 1
    best_precision = 0
    running_loss = 0

    # load checkpoint if needed.
    if resume:
        epoch_start, best_precision = utils.load_checkpoint(
            model, model_dir, best=True
        )

    for epoch in range(epoch_start, epochs + 1):
        # prepare a data stream for the epoch.
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
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

            if batch_index % checkpoint_interval == 0:
                # test the model.
                model_precision = utils.validate(
                    model, dataset_test or dataset,
                    batch_size=test_size, verbose=False
                )

                # update best precision if needed.
                is_best = model_precision > best_precision
                best_precision = max(model_precision, best_precision)

                # save the checkpoint.
                utils.save_checkpoint(
                    model, model_dir, epoch,
                    model_precision, best_precision, best=is_best
                )
