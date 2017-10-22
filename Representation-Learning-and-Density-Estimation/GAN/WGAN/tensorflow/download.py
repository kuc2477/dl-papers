#!/usr/bin/env python3
import os
import os.path
import argparse
import subprocess
import zipfile
from tqdm import tqdm
import requests
from utils import log, c


# NOTE: Dataset downloading script. Referenced carpedm20's DCGAN.


# =============
# Script Parser
# =============

parser = argparse.ArgumentParser(description='CLI for dataset downloading')
parser.add_argument(
    'datasets', metavar='NAME', type=str, nargs='+',
    choices=['lsun', 'mnist'],
    help='name of dataset to download [lsun, mnist]'
)


# ================
# Helper Functions
# ================


def _download(url, filename=None):
    local_filename = filename or url.split('/')[-1]
    temp_filename = '.{}'.format(local_filename)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(temp_filename, 'wb') as f:
        for chunk in tqdm(
                response.iter_content(1024 * 32),
                total=total_size // (1024 * 32),
                unit='KiB', unit_scale=True,
        ):
            if chunk:
                f.write(chunk)
    response.close()
    os.rename(temp_filename, local_filename)
    return local_filename


def _extract_zip(zipfile_path, extraction_path='.'):
    with zipfile.ZipFile(zipfile_path) as zf:
        extracted_dirname = zf.namelist()[0]
        zf.extractall(extraction_path)
    return extracted_dirname


def _extract_gz(gzfile_path, extraction_path='.'):
    cmd = ['gzip', '-d', gzfile_path]
    subprocess.call(cmd)
    return '.'.join(gzfile_path.split('.')[:-1])


def _download_zip_dataset(
        url, dataset_dirpath, dataset_dirname, download_path=None):
    download_path = _download(url)
    download_dirpath = os.path.dirname(download_path)
    extracted_dirname = _extract_zip(download_path)

    os.remove(download_path)
    os.renames(os.path.join(download_dirpath, extracted_dirname),
               os.path.join(dataset_dirpath, dataset_dirname))


def _download_gz_dataset(
        url, dataset_dirpath, dataset_dirname, download_path=None):
    download_path = _download(url)
    download_dirpath = os.path.dirname(download_path)
    extracted_filename = _extract_gz(download_path)

    os.renames(os.path.join(download_dirpath, extracted_filename),
               os.path.join(dataset_dirpath, dataset_dirname,
                            extracted_filename))


# ====
# Main
# ====

def maybe_download_lsun(dataset_dirpath, dataset_dirname,
                        category, set_name, tag='latest'):
    dataset_path = os.path.join(dataset_dirpath, dataset_dirname)
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
        '&category={category}&set={set_name}'.format(**locals())

    # check existance
    if os.path.exists(dataset_path):
        print(c(
            'lsun dataset already exists: {}'
            .format(dataset_path), 'red'
        ))
        return

    # start downloading lsun dataset
    with log(
            'download lsun dataset from {}'.format(url),
            'downloaded lsun dataset to {}'.format(dataset_path)):
        _download_zip_dataset(url, dataset_dirpath, dataset_dirname)


def maybe_download_mnist(dataset_dirpath, dataset_dirname, set_name):
    dataset_path = os.path.join(dataset_dirpath, dataset_dirname)
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    train_filenames = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
    ]
    val_filenames = [
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    assert set_name in ['train', 'val']
    filenames = train_filenames if set_name == 'train' else val_filenames

    # check existance
    if os.path.exists(dataset_path):
        print(c(
            'mnist dataset already exists: {}'
            .format(dataset_path), 'red'
        ))
        return

    # start downloading mnist dataset
    for filename in filenames:
        url = base_url + filename
        with log(
                'download mnist dataset from {}'.format(url),
                'downloaded mnist dataset {} to {}'
                .format(filename, dataset_path)):
            _download_gz_dataset(url, dataset_dirpath, dataset_dirname)


if __name__ == '__main__':
    args = parser.parse_args()

    if 'lsun' in args.datasets:
        maybe_download_lsun(
            './data/lsun', 'train', category='living_room', set_name='train'
        )
        maybe_download_lsun(
            './data/lsun', 'val', category='living_room', set_name='val'
        )
    if 'mnist' in args.datasets:
        maybe_download_mnist('./data/mnist', 'train', set_name='train')
        maybe_download_mnist('./data/mnist', 'val', set_name='val')
