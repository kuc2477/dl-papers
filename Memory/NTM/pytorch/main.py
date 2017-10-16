#!/usr/bin/env python3
import argparse
import torch
from tasks import TASKS, Copy
from model import NTM
from train import train
import utils


parser = argparse.ArgumentParser(
    'Neural Turing Machine PyTorch Implementation'
)

parser.add_argument('--task', choices=TASKS.keys(), default=Copy.name)
parser.add_argument('--hidden-size', type=int, default=100)
parser.add_argument('--memory-size', type=int, default=20)
parser.add_argument('--memory-feature-size', type=int, default=128)
parser.add_argument('--head-num', type=int, default=1)
parser.add_argument('--max-shift-size', type=int, default=1)

parser.add_argument('--iterations', type=int, default=5000)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--test-size', type=int, default=1024)
parser.add_argument('--weight-decay', type=float, default=1e-05)
parser.add_argument('--lr', type=float, default=5e-03)
parser.add_argument('--lr-decay', type=float, default=.1)
parser.add_argument('--lr-decay-iterations', type=int,
                    default=[1000, 3000, 4000], nargs='+')

parser.add_argument('--checkpoint-interval', type=int, default=300)
parser.add_argument('--eval-log-interval', type=int, default=50)
parser.add_argument('--gradient-log-interval', type=int, default=50)
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--model-dir', default='./checkpoints')

resume_command = parser.add_mutually_exclusive_group()
resume_command.add_argument('--resume-best', action='store_true')
resume_command.add_argument('--resume-latest', action='store_true')
parser.add_argument('--best', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--train', action='store_true', dest='train')
main_command.add_argument('--test', action='store_false', dest='train')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    task = TASKS[args.task]()

    ntm = NTM(
        label=task.name,
        embedding_size=task.model_input_size,
        hidden_size=args.hidden_size,
        memory_size=args.memory_size,
        memory_feature_size=args.memory_feature_size,
        output_size=task.model_output_size,
        head_num=args.head_num,
        max_shift_size=args.max_shift_size,
    )

    # initialize the model weights.
    utils.xavier_initialize(ntm)

    # prepare cuda if needed.
    if cuda:
        ntm.cuda()

    if args.train:
        train(
            ntm, task,
            model_dir=args.model_dir,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_decay_iterations=args.lr_decay_iterations,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            test_size=args.test_size,
            iterations=args.iterations,
            checkpoint_interval=args.checkpoint_interval,
            eval_log_interval=args.eval_log_interval,
            gradient_log_interval=args.gradient_log_interval,
            loss_log_interval=args.loss_log_interval,
            resume_best=args.resume_best,
            resume_latest=args.resume_latest,
            cuda=cuda,
        )
    else:
        utils.load_checkpoint(ntm, args.model_dir, best=True)
        utils.validate(
            ntm, task, test_size=args.test_size,
            cuda=cuda, verbose=True
        )
