import argparse
import torch
from model import VAE
from data import DATASETS, DATASET_CONFIGS
from train import train_model


parser = argparse.ArgumentParser('VAE pytorch implementation')
parser.add_argument('--no-gpu', action='store_false')
parser.add_argument('--test', action='store_true')
parser.add_argument(
    '--dataset', default='mnist',
    choices=list(DATASETS.keys())
)
parser.add_argument('--image-size', type=int, default=32)
parser.add_argument('--channel-num', type=int, default=1)
parser.add_argument('--kernel-num', type=int, default=128)
parser.add_argument('--kernel-size', type=int, default=4)
parser.add_argument('--z-size', type=int, default=128)
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-l', '--lr', type=float, default=3e-6)
parser.add_argument('-m', '--momentum', type=float, default=0.2)
parser.add_argument('--log-interval', type=int, default=10)


def patch_dataset_specific_configs(config):
    dataset_specific = DATASET_CONFIGS[config.dataset]
    for k, v in dataset_specific.items():
        setattr(config, k, v)


config = parser.parse_args()
config.cuda = not config.no_gpu and torch.cuda.is_available()
patch_dataset_specific_configs(config)


if __name__ == '__main__':
    vae = VAE(
        dataset=config.dataset,
        image_size=config.image_size,
        channel_num=config.channel_num,
        kernel_num=config.kernel_num,
        kernel_size=config.kernel_size,
        z_size=config.z_size,
        use_cuda=config.cuda,
    )

    # configure cuda if needed
    if config.cuda:
        vae.cuda()

    # run test or training
    if config.test:
        vae.sample()
    else:
        train_model(vae, config)
