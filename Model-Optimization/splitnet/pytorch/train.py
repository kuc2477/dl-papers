from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm
import utils


def train(model, train_dataset, test_dataset=None, model_dir='models',
          lr=1e-04, batch_size=32, test_size=256, epochs=5,
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
        data_loader = utils.get_data_loader(
            train_dataset, batch_size, cuda=cuda
        )
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (data, labels) in data_stream:
            # clear the gradients.
            optimizer.zero_grad()

            # run & update the network.
            x = Variable(data).cuda() if cuda else Variable(data)
            labels = Variable(labels).cuda() if cuda else Variable(labels)
            scores = model(x)
            cross_entropy_loss = criterion(scores, labels)
            regularization_loss = model.split_loss()
            total_loss = cross_entropy_loss + regularization_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # update & display statistics.
            running_loss += total_loss.data[0]
            data_stream.set_description((
                'epoch: {epoch}/{epochs} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | loss => '
                'ce {ce_loss:.4} / reg {reg_loss:.6} / total {total_loss:.4}'
            ).format(
                epoch=epoch,
                epochs=epochs,
                trained=batch_index * len(data),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                ce_loss=(cross_entropy_loss.data[0] / len(data)),
                reg_loss=(regularization_loss.data[0] / len(data)),
                total_loss=(total_loss.data[0] / len(data)),
            ))

            if batch_index % checkpoint_interval == 0:
                # notify that we've reached to a new checkpoint.
                print()
                print()
                print('#############')
                print('# checkpoint!')
                print('#############')
                print()

                # test the model.
                model_precision = utils.validate(
                    model, test_dataset or train_dataset,
                    test_size=test_size, cuda=cuda, verbose=True
                )

                # update best precision if needed.
                is_best = model_precision > best_precision
                best_precision = max(model_precision, best_precision)

                # save the checkpoint.
                utils.save_checkpoint(
                    model, model_dir, epoch,
                    model_precision, best=is_best
                )
                print()
