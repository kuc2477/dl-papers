import argparse
import torch
from model import VAE


parser = argparse.ArgumentParser('VAE pytorch implementation')
parser.add_argument('--no-gpu', action='store_false')
parser.add_argument('--image-size', type=int, default=64)
parser.add_argument('--channel-num', type=int, default=3)
parser.add_argument('--kernel-num', type=int, default=128)
parser.add_argument('--kernel-size', type=int, default=4)
parser.add_argument('--latent-code-size', type=int, default=128)
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-l', '--lr', type=float, default=0.01)
parser.add_argument('--log-interval', type=int, default=10)


args = parser.parse_args()
args.cuda = not args.no_gpu and torch.cuda.is_available()


if __name__ == '__main__':
    model = VAE(
        image_size=args.image_size,
        channel_num=args.channel_num,
        kernel_num=args.kernel_num,
        kernel_size=args.kernel_size,
        latent_code_size=args.latent_code_size,
        use_cuda=args.cuda,
    )
