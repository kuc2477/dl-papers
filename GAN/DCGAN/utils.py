# Below functions are taken from carpdem20's implementation
# https://github.com/carpedm20/DCGAN-tensorflow
import os
import os.path
import tempfile
import numpy as np
import scipy.misc
import scipy
import lmdb
from PIL import Image


# ===========
# Image Utils
# ===========

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
    image = imread(image_path, is_grayscale)
    return transform(
        image, input_height, input_width,
        resize_height, resize_width, is_crop
    )


def imread(path, is_grayscale=False):
    return scipy.misc.imread(path, flatten=is_grayscale).astype(np.float)


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width
        )
    else:
        cropped_image = scipy.misc.imresize(image, [
            resize_height, resize_width
        ])
        return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h)/2.))
        i = int(round((w - crop_w)/2.))
        return scipy.misc.imresize(
            x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.)/2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img


# =============
# Dataset Utils
# =============

def export_mdb_images(db_path, out_dir, flat=True, limit=-1, size=256):
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
