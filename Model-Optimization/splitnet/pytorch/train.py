from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from tqdm import tqdm
import visual
import utils


def train(model, train_dataset, test_dataset=None, model_dir='models',
          lr=1e-04, lr_decay=.1, lr_decay_epochs=None, weight_decay=1e-04,
          gamma1=1., gamma2=1., gamma3=10.,
          batch_size=32, test_size=256, epochs=5,
          loss_log_interval=30,
          weight_log_interval=500,
          checkpoint_interval=500,
          resume=False, cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay
    )
    scheduler = MultiStepLR(optimizer, lr_decay_epochs, gamma=lr_decay)

    # prepare the model and statistics.
    model.train()
    epoch_start = 1
    best_precision = 0

    # load checkpoint if needed.
    if resume:
        epoch_start, best_precision = utils.load_checkpoint(
            model, model_dir, best=True
        )

    for epoch in range(epoch_start, epochs + 1):
        # adjust learning rate if needed.
        scheduler.step(epoch-1)

        # prepare a data stream for the epoch.
        data_loader = utils.get_data_loader(
            train_dataset, batch_size, cuda=cuda
        )
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (data, labels) in data_stream:
            # where are we?
            data_size = len(data)
            dataset_size = len(data_loader.dataset)
            dataset_batches = len(data_loader)
            iteration = (
                (epoch-1)*(len(data_loader.dataset) // batch_size) +
                batch_index + 1
            )

            # clear the gradients.
            optimizer.zero_grad()

            # run the network.
            x = Variable(data).cuda() if cuda else Variable(data)
            labels = Variable(labels).cuda() if cuda else Variable(labels)
            scores = model(x)

            # update the network.
            cross_entropy_loss = criterion(scores, labels)
            split_loss = model.split_loss(gamma1, gamma2, gamma3)
            total_loss = cross_entropy_loss + split_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # update & display statistics.
            data_stream.set_description((
                'epoch: {epoch}/{epochs} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'ce: {ce_loss:.4} / split: {split_loss:.5} / '
                'total: {total_loss:.4}'
            ).format(
                epoch=epoch,
                epochs=epochs,
                iteration=iteration,
                trained=(batch_index+1)*batch_size,
                total=dataset_size,
                progress=(100.*(batch_index+1)/dataset_batches),
                ce_loss=(cross_entropy_loss.data[0] / data_size),
                split_loss=(split_loss.data[0] / data_size),
                total_loss=(total_loss.data[0] / data_size),
            ))

            # Send losses to the visdom server.
            if iteration % loss_log_interval == 0:
                visual.visualize_scalar(
                    cross_entropy_loss.data / data_size,
                    'cross entropy loss', iteration, env=model.name
                )
                visual.visualize_scalar(
                    split_loss.data / data_size,
                    'split loss', iteration, env=model.name
                )
                visual.visualize_scalar(
                    total_loss.data / data_size,
                    'total loss', iteration, env=model.name
                )

            # Send weights to the visdom server.
            if iteration % weight_log_interval == 0:
                weights = [
                    w.data if w is not None else None for
                    g in model.residual_block_groups for
                    b in g.residual_blocks for
                    w in (b.w1, b.w2, b.w3)
                ] + [model.fc.linear.weight.data]

                weight_names = [
                    'g{i}-b{j}-w{k}'.format(i=i+1, j=j+1, k=k+1) for
                    i, g in enumerate(model.residual_block_groups) for
                    j, b in enumerate(g.residual_blocks) for
                    k, w in enumerate((b.w1, b.w2, b.w3))
                ] + ['fc-w']

                for weight, name in zip(weights, weight_names):
                    visual.visualize_kernel(
                        weight, name,
                        'epoch{}-{}'.format(epoch, batch_index+1),
                        env=model.name
                    )

            if iteration % checkpoint_interval == 0:
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
