from argparse import ArgumentParser
import torch
from data import DATASETS, DATASET_CONFIGS
from model import WideResNet
from train import train


parser = ArgumentParser('WRN torch implementation')
parser.add_argument(
    '--dataset', default='cifar10', type=str,
    choices=list(DATASETS.keys())
)
parser.add_argument('--total-block-number', type=int, default=60)
parser.add_argument('--widen-factor', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-06)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--no-gpus', action='store_false')
parser.add_argument('--test', action='store_true')
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--model-dir', type=str, default='models')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and not args.no_gpus
    dataset = DATASETS[args.dataset]
    dataset_config = DATASET_CONFIGS[args.dataset]
    wrn = WideResNet(
        dataset_config['size'],
        dataset_config['channels'],
        dataset_config['classes'],
        total_block_number=args.total_block_number,
        widen_factor=args.widen_factor,
    )

    if cuda:
        wrn.cuda()

    if args.test:
        # TODO: NOT IMPLEMENTED YET
        pass
    else:
        train(
            wrn, dataset,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            cuda=cuda,
            log_interval=args.log_interval,
            model_dir=args.model_dir,
        )
