#!/usr/bin/env python3
import argparse
import functools
import struct
import os
import os.path
import random
import tempfile
import numpy as np
import lmdb
from PIL import Image
import utils


# =================
# Dataset Utilities
# =================

def _export_mdb_images(db_path, out_dir=None, flat=True, limit=-1, size=256):
    out_dir = out_dir or db_path
    env = lmdb.open(
        db_path, map_size=1099511627776,
        max_readers=1000, readonly=True
    )
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            key = str(key, 'utf-8')
            # decide image out directory
            if not flat:
                image_out_dir = os.path.join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir

            # create the directory if an image out directory doesn't exist
            if not os.path.exists(image_out_dir):
                os.makedirs(image_out_dir)

            with tempfile.NamedTemporaryFile('wb') as temp:
                temp.write(val)
                temp.flush()
                temp.seek(0)
                image_out_path = os.path.join(image_out_dir, key + '.jpg')
                Image.open(temp.name).resize((size, size)).save(image_out_path)
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')


def _dataset(name, image_size=None, channel_size=None):
    def decorator(dataset_generator):
        @functools.wraps(dataset_generator)
        def wrapper(*args, **kwargs):
            return dataset_generator(*args, **kwargs)
        wrapper.name = name
        wrapper.image_size = image_size
        wrapper.channel_size = channel_size
        return wrapper
    return decorator


# ========
# Datasets
# ========

@_dataset('images')
def image_dataset(batch_size, dirpath,
                  resize_height=None, resize_width=None,
                  use_crop=True, is_grayscale=False):
    paths = [
        os.path.join(dirpath, name) for name in os.listdir(dirpath) if
        name.endswith('.jpg')
    ]

    # shuffle and yield images
    random.shuffle(paths)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_images = np.array([
            utils.get_image(
                p,
                resize_height=resize_height,
                resize_width=resize_width,
                use_crop=use_crop, is_grayscale=is_grayscale
            ) for p in batch_paths
        ])
        yield batch_images


def image_dataset_length(dirpath):
    paths = [
        os.path.join(dirpath, name) for name in os.listdir(dirpath) if
        name.endswith('.jpg')
    ]
    return len(paths)


@_dataset('mnist', image_size=32, channel_size=1)
def mnist_dataset(batch_size, test=False):
    if test:
        fname_img = './data/mnist/val/t10k-images-idx3-ubyte'
    else:
        fname_img = './data/mnist/train/train-images-idx3-ubyte'

    with open(fname_img, 'rb') as fd:
        magic, num, rows, cols = struct.unpack('>IIII', fd.read(16))
        images = np.fromfile(fd, dtype=np.uint8)\
            .reshape((num, rows, cols, 1))\
            .astype(np.float)

        images /= 255.
        images = (images - 0.5) * 2.
        images = np.lib.pad(
            images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
            constant_values=(-1, -1)
        )

    # shuffle and yield images
    np.random.shuffle(images)
    for i in range(0, num, batch_size):
        yield images[i:i+batch_size]


def mnist_dataset_length(test=False):
    if test:
        fname_img = './data/mnist/val/t10k-images-idx3-ubyte'
    else:
        fname_img = './data/mnist/train/train-images-idx3-ubyte'

    with open(fname_img, 'rb') as fd:
        magic, num, rows, cols = struct.unpack('>IIII', fd.read(16))
        return num


@_dataset('lsun', image_size=64, channel_size=3)
def lsun_dataset(batch_size,
                 test=False, resize=True, use_crop=False):
    path = './data/lsun/val' if test else './data/lsun/train'
    if resize:
        return image_dataset(
            batch_size, path,
            resize_width=64,
            resize_height=64,
            use_crop=use_crop
        )
    else:
        return image_dataset(batch_size, path)


def lsun_dataset_length(test=False):
    path = './data/lsun/val' if test else './data/lsun/train'
    return image_dataset_length(path)


# datasets available out-of-the-box
DATASETS = {
    mnist_dataset.name: mnist_dataset,
    lsun_dataset.name: lsun_dataset,
    image_dataset.name: image_dataset,
}


DATASET_LENGTH_GETTERS = {
    mnist_dataset.name: mnist_dataset_length,
    lsun_dataset.name: lsun_dataset_length,
    image_dataset.name: image_dataset_length
}


# =============
# Script Parser
# =============

def export_lsun(args):
    with utils.log(
            'export lsun images from mdb file',
            'exported lsun images from mdb file'):
        _export_mdb_images('./data/lsun/train', size=args.size)
        _export_mdb_images('./data/lsun/val', size=args.size)


parser = argparse.ArgumentParser(description='Data pre/post processing CLI')
subparsers = parser.add_subparsers(dest='command')
parser_export_lsun = subparsers.add_parser('export_lsun')
parser_export_lsun.set_defaults(func=export_lsun)
parser_export_lsun.add_argument(
    '--format', type=str, default='jpg', choices=['jpg'],
    help='format to export'
)
parser_export_lsun.add_argument(
    '--size', type=int, default=256, help='image size to be exported'
)


if __name__ == '__main__':
    try:
        args = parser.parse_args()
        args.func(args)
    except AttributeError:
        parser.print_help()
