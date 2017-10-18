#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch
from data import get_dataset, DATASET_CONFIGS
from train import train
from model import MLP


parser = ArgumentParser('EWC PyTorch Implementation')
parser.add_argument('--task-number')
parser.add_argument('--epochs-per-task')
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--input-dropout-prob', type=float, default=.2)
parser.add_argument('--hidden-dropout-prob', type=float, default=.5)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')


if __name__ == '__main__':
    args = parser.parse_args()

    # generate permutations for the tasks.
    np.random.seed(args.random_seed)
    permutations = [
        np.random.permutation(DATASET_CONFIGS['mnist']['size']**2) for
        _ in range(args.task_number)
    ]

    # prepare mnist datasets.
    train_datasets = [
        get_dataset('mnist', permutation=p) for p in permutations
    ]
    test_datasets = [
        get_dataset('mnist', train=False, permutation=p) for p in permutations
    ]

    # prepare the model.
    mlp = MLP(
        DATASET_CONFIGS['mnist']['size']**2,
        DATASET_CONFIGS['mnist']['classes'],
        hidden_size=args.hidden_size,
        hidden_layer_num=args.hidden_layer_num,
        hidden_dropout_prob=args.hidden_dropout_prob,
        input_dropout_prob=args.input_dropout_prob,
    )

    # prepare the cuda if needed.
    if torch.cuda.is_available() and args.cuda:
        mlp.cuda()

    # run the experiment.
    train(mlp)
