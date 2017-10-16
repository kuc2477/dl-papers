import operator
from functools import reduce
import torch
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import visual
import utils


def train(
        model, task, model_dir='checkpoints',
        lr=1e-3, lr_decay=.1, lr_decay_iterations=None, weight_decay=1e-04,
        batch_size=32, test_size=512, iterations=50000,
        checkpoint_interval=500,
        eval_log_interval=30,
        gradient_log_interval=50,
        loss_log_interval=30,
        resume_best=False,
        resume_latest=False,
        cuda=False):
    # define an optimizer and a training scheduler.
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = MultiStepLR(optimizer, lr_decay_iterations, gamma=lr_decay)

    # prepare the model for the training.
    model.train()
    iteration_start = 1
    best_precision = 0

    # load from the checkpoint if any resume flag is given.
    if resume_best or resume_latest:
        iteration_start, best_precision = utils.load_checkpoint(
            model, model_dir, best=resume_best
        )

    # the train loop.
    data_loader = task.data_loader(batch_size)
    progress = tqdm(range(iteration_start, iterations+1))
    for iteration in progress:
        optimizer.zero_grad()
        scheduler.step(iteration-1)

        # prepare the batch and reset the model's state variables.
        x, y = next(data_loader)
        x, y = Variable(x), Variable(y)
        x, y = (x.cuda(), y.cuda()) if cuda else (x, y)
        model.reset(x.size(0), cuda=cuda)

        # run the model to take input sequences.
        for index in range(x.size(1)):
            model(x[:, index, :])

        # run the model to output sequences.
        activations = []
        predictions = []
        for index in range(y.size(1)):
            activation = task.model_output_activation(model())
            activations.append(activation)
            predictions.append(activation.round())
        activations = torch.stack(activations, 1)
        predictions = torch.stack(predictions, 1).long()

        # back propagate the error and update the network.
        loss = task.criterion(activations, y)
        loss.backward()
        optimizer.step()

        # calculate the wrong bits per sequence.
        total_bits = reduce(operator.mul, y.size()) // batch_size
        wrong_bits = torch.abs(predictions-y.long()).sum().data[0] / batch_size
        precision = 1 - wrong_bits/total_bits

        # update the progress.
        progress.set_description((
            'progress: [{trained}/{total}] ({percentage:.0f}%) | '
            'prec: {prec:.4} | '
            'loss: {loss:.4} '
        ).format(
            trained=iteration*batch_size,
            total=iterations*batch_size,
            percentage=(100.*iteration/iterations),
            prec=precision,
            loss=loss.data[0],
        ))

        # Send gradient norms to the visdom server.
        if iteration % gradient_log_interval == 0:
            for name, gradient_norm in [
                (n, p.grad.norm().data) for
                n, p in model.named_parameters()
            ]:
                visual.visualize_scalar(
                    gradient_norm, name+' gradient l2 norm',
                    iteration, env=model.name
                )

        # Send test precision to the visdom server.
        if iteration % eval_log_interval == 0:
            visual.visualize_scalar(utils.validate(
                model, task,
                test_size=test_size, cuda=cuda, verbose=False
            ), 'precision', iteration, env=model.name)

        # Send losses to the visdom server.
        if iteration % loss_log_interval == 0:
            visual.visualize_scalar(
                loss.data, 'loss', iteration,
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
                model, task, test_size=test_size,
                cuda=cuda, verbose=True
            )

            # update best precision if needed.
            is_best = model_precision > best_precision
            best_precision = max(model_precision, best_precision)

            # save the checkpoint.
            utils.save_checkpoint(
                model, model_dir, iteration,
                model_precision, best=is_best
            )
            print()
