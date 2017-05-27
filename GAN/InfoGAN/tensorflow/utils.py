# Below functions are taken from carpdem20's implementation
# https://github.com/carpedm20/DCGAN-tensorflow
from contextlib import contextmanager
from colorama import Fore
import numpy as np
import scipy.misc
import scipy


def c(string, color):
    return '{}{}{}'.format(getattr(Fore, color.upper()), string, Fore.RESET)


@contextmanager
def log(start, end, start_color='yellow', end_color='cyan'):
    print(c('>> ' + start, start_color))
    yield
    print(c('>> ' + end, end_color) + '\n')


def get_image(image_path,
              input_height=None, input_width=None,
              resize_height=None, resize_width=None,
              use_crop=True, is_grayscale=False):
    image = imread(image_path, is_grayscale)
    return image if (resize_height is None and resize_width is None) else \
        transform(
            image, input_height, input_width,
            resize_height, resize_width, use_crop
        )


def imread(path, is_grayscale=False):
    return scipy.misc.imread(path, flatten=is_grayscale).astype(np.float)


def transform(image,
              resize_height, resize_width,
              input_height=None, input_width=None, use_crop=True):
    input_height = input_height or image.shape[0]
    input_width = input_width or image.shape[1]
    if use_crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width
        )
    else:
        cropped_image = scipy.misc.imresize(image, [
            resize_height, resize_width
        ])
        return np.array(cropped_image)/127.5 - 1.


def center_crop(x, crop_h=64, crop_w=64, resize_h=32, resize_w=32):
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
