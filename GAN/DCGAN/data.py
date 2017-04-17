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


# ===========================
# Dataset export / Utilities
# ===========================

def export_mdb_images(db_path, out_dir=None, flat=True, limit=-1, size=256):
    out_dir = out_dir or db_path
    env = lmdb.open(
        db_path, map_size=1099511627776,
        max_readers=100, readonly=True
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


def dataset(name, image_size=None, channel_size=None):
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

@dataset('images')
def image_dataset(batch_size, dirpath,
                  resize_height, resize_width,
                  is_crop=True, is_grayscale=False):
    paths = [
        os.path.join(dirpath, name) for name in os.listdir(dirpath) if
        name.endswith('.jpg')
    ]
    random.shuffle(paths)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        batch_images = np.array([
            utils.get_image(
                p,
                resize_height=resize_height,
                resize_width=resize_width,
                is_crop=is_crop,
                is_grayscale=is_grayscale
            ) for p in batch_paths
        ])
        yield batch_images


@dataset('mnist', image_size=32, channel_size=1)
def mnist_dataset(batch_size, test=False):
    if test:
        fname_img = './data/mnist/val/t10k-images-idx3-ubyte'
        fname_lbl = './data/mnist/val/t10k-labels-idx1-ubyte'
    else:
        fname_img = './data/mnist/train/train-images-idx3-ubyte'
        fname_lbl = './data/mnist/train/train-labels-idx1-ubyte'

    with open(fname_lbl, 'rb') as fd:
        magic, num = struct.unpack('>II', fd.read(8))
        labels = np.fromfile(fd, dtype=np.int8)

    with open(fname_img, 'rb') as fd:
        magic, num, rows, cols = struct.unpack('>IIII', fd.read(16))
        images = np.fromfile(fd, dtype=np.uint8).reshape(
            len(labels), rows, cols
        )

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(images)

    for i in range(0, len(labels), batch_size):
        yield images[i:i+batch_size]


@dataset('lsun', image_size=256, channel_size=3)
def lsun_dataset(batch_size, test=False):
    path = './data/lsun/val' if test else './data/lsun/train'
    return image_dataset(batch_size, path, 256, 256)


DATASETS = {
    mnist_dataset.name: mnist_dataset,
    lsun_dataset.name: lsun_dataset,
    image_dataset.name: image_dataset,
}
