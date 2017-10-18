import numpy as np
from torch import optim
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual


def train(self, model, train_datasets, test_datasets, epochs_per_tasks=10,
          batch_size=64, test_size=1024, fisher_estimation_sample_size=1024,
          lr=1e-3, weight_decay=1e-5, cuda=False):
    # prepare the loss criteriton and the optimizer.
    criteriton = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # set the model's mode to training mode.
    model.train()

    for task, (train_dataset, test_dataset) in enumerate(zip(train_datasets,
                                                             test_datasets)):
        # prepare the data loader.
        data_loader = utils.get_data_loader(
            train_dataset, batch_size=batch_size,
            cuda=cuda
        )
        data_stream = tqdm(enumerate(data_loader, 1))

        for epoch in range(epochs_per_tasks):
            for batch_index, (x, y) in data_stream:
                # where are we?
                data_size = len(x)
                dataset_size = len(data_loader.dataset)
                dataset_batches = len(data_loader)
                # TODO: NOT IMPLEMENTED YET
                previous_task_iteration = None
                current_task_iteration = None
                iteration = (
                    previous_task_iteration +
                    current_task_iteration
                )

                # prepare the data.
                x = x.view(batch_size, -1)
                x = Variable(x).cuda() if cuda else Variable(x)
                y = Variable(y).cuda() if cuda else Variable(y)

                # run the model and backpropagate the errors.
                optimizer.zero_grad()
                scores = model(x)
                ce_loss = criteriton(scores, y)
                ewc_loss = model.ewc_loss()
                loss = ce_loss + ewc_loss
                loss.backward()
                optimizer.step()

                # calculate the training precision.
                _, predicted = scores.max(1)
                precision = (predicted == y).sum().data[0] / len(x)

                data_stream.set_description((
                    'task: {task}/{tasks} |'
                    'epoch: {epoch} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) |'
                    'prec: {prec:.4}'
                    'loss => '
                    'ce: {ce_loss:.4} / '
                    'ewc: {ewc_loss:.4} / '
                    'total: {loss:.4} |'
                ).format(
                    task=task+1,
                    tasks=len(train_datasets),
                    epoch=epoch,
                    trained=(batch_index+1)*batch_size,
                    total=dataset_size,
                    progress=(100.*(batch_index+1)/dataset_batches),
                    prec=precision,
                    ce_loss=ce_loss.data[0],
                    ewc_loss=ewc_loss.data[0],
                    loss=loss.data[0],
                ))

                # TODO: NOT IMPLEMENTED YET
                # Send test precision to the visdom server.
                if iteration % eval_log_interval == 0:
                    visual.visualize_scalar(utils.validate(
                        model, test_dataset, test_size=test_size,
                        cuda=cuda, verbose=False
                    ), 'precision', iteration, env=model.name)

                # TODO: NOT IMPLEMENTED YET
                # Send losses to the visdom server.
                if iteration % loss_log_interval == 0:
                    visual.visualize_scalar(
                        loss.data / data_size,
                        'loss', iteration, env=model.name
                    )

        # TODO: NOT IMPLEMENTED YET
        # estimate the fisher information for each parameters and consolidate
        # them in the network.
        fisher = model.estimate_fisher()
        model.consolidate(fisher)
