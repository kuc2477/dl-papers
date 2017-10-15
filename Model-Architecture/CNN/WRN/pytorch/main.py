#! /usr/bin/env python3
from argparse import ArgumentParser
import torch
from data import TRAIN_DATASETS, TEST_DATASETS, DATASET_CONFIGS
from model import WideResNet
from train import train
import utils


parser = ArgumentParser('WRN torch implementation')
parser.add_argument(
    '--dataset', default='cifar10', type=str,
    choices=list(TRAIN_DATASETS.keys())
)
parser.add_argument('--total-block-number', type=int, default=36)
parser.add_argument('--widen-factor', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--test-size', type=int, default=1000)
parser.add_argument('--checkpoint-interval', type=int, default=500)
parser.add_argument('--model-dir', type=str, default='models')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
command = parser.add_mutually_exclusive_group(required=True)
command.add_argument('--test', action='store_true', dest='test')
command.add_argument('--train', action='store_false', dest='test')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    train_dataset = TRAIN_DATASETS[args.dataset]
    test_dataset = TEST_DATASETS[args.dataset]
    dataset_config = DATASET_CONFIGS[args.dataset]

    # instantiate the model instance.
    wrn = WideResNet(
        args.dataset,
        dataset_config['size'],
        dataset_config['channels'],
        dataset_config['classes'],
        total_block_number=args.total_block_number,
        widen_factor=args.widen_factor,
    )

    # prepare cuda if needed.
    if cuda:
        wrn.cuda()

    # run the given command.
    if args.test:
        utils.load_checkpoint(wrn, args.model_dir, best=True)
        utils.validate(
            wrn, test_dataset, test_size=args.test_size,
            cuda=cuda, verbose=True
        )
    else:
        train(
            wrn, train_dataset, test_dataset=test_dataset,
            model_dir=args.model_dir,
            lr=args.lr,
            batch_size=args.batch_size,
            test_size=args.test_size,
            epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
            cuda=cuda,
        )
