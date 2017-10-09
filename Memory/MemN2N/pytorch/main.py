# !/usr/bin/env python3
import argparse
from model import MemN2N
from train import train
import utils


parser = argparse.ArgumentParser(
    'End-to-End Memory Network PyTorch Implementation'
)
parser.add_argument('--vocabulary-size', type=int, default=200)
parser.add_argument('--embedding-size', type=int, default=50)
parser.add_argument('--sentence-size', type=int, default=20)
parser.add_argument('--memory-size', type=int, default=30)
parser.add_argument('--hops', type=int, default=3)
parser.add_argument(
    '--weight-tying-scheme',
    choices=MemN2N.WEIGHT_TYING_SCHEMES,
    default=MemN2N.ADJACENT
)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--test-size', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--lr-decay', type=float, deefault=.1)

parser.add_argument('--checkpoint-interval')
parser.add_argument('--eval-interval')
parser.add_argument('--loss-interval')
parser.add_argument('--model-dir')

resume_command = parser.add_mutually_exclusive_group()
resume_command.add_argument('--resume-best', action='store_true')
resume_command.add_argument('--resume-latest', action='store_true')
parser.add_argument('--best', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_true', dest='train')
main_command.add_argument('--test', action='store_false', dest='train')


if __name__ == "__main__":
    args = parser.parse()

    memn2n = MemN2N(
        vocabulary_size=args.vocabulary_size,
        embedding_size=args.embedding_size,
        sentence_size=args.sentence_size,
        memory_size=args.memory_size,
        hops=args.hops,
        weight_tying_scheme=args.weight_tying_scheme,
    )

    if args.train:
        train(memn2n)
    else:
        utils.validate(memn2n)
