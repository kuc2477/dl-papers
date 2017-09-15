import os
import os.path
from contextlib import contextmanager
from colorama import Fore
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy


# =========
# Log Utils
# =========

def c(string, color):
    return '{}{}{}'.format(getattr(Fore, color.upper()), string, Fore.RESET)


@contextmanager
def log(start, end, start_color='yellow', end_color='cyan'):
    print(c('>> ' + start, start_color))
    yield
    print(c('>> ' + end, end_color) + '\n')


# ===========
# Image Utils
# ===========

# Below functions are taken from carpdem20's implementation
# https://github.com/carpedm20/DCGAN-tensorflow

def get_image(image_path,
              input_height=None, input_width=None,
              resize_height=None, resize_width=None,
              use_crop=True, is_grayscale=False):
    image = imread(image_path, is_grayscale)
    return image if (resize_height is None and resize_width is None) else \
        transform(
            image, resize_height, resize_width,
            input_height, input_width, use_crop
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
    # multi-channels
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img
    # single-channels
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w] = image[:, :, 0]
        return img
    else:
        raise ValueError(
            'in merge(images,size) images parameter '
            'must have dimensions: HxW or HxWx3 or HxWx4'
        )


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


# ========================
# Training / Testing Utils
# ========================

def save_checkpoint(sess, model, epoch, config):
    print()
    print()
    print('#############')
    print('# checkpoint!')
    print('#############')
    print()

    checkpoint_dir = os.path.join(config.checkpoint_dir, model.name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = 'epoch{epoch}.ckpt'.format(epoch=epoch)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    saver = tf.train.Saver()
    saver.save(sess, checkpoint_path)
    print('=> saved model to {}'.format(checkpoint_path))
    print()


def load_checkpoint(sess, model, config):
    print('=> searching for a checkpoint')

    checkpoint_dir = os.path.join(config.checkpoint_dir, model.name)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

    if checkpoint and checkpoint.model_checkpoint_path:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        raise FileNotFoundError

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print('=> loaded model {model_name} from {checkpoint_path}'.format(
        model_name=model.name,
        checkpoint_path=checkpoint_path
    ))

    return int(checkpoint_name.strip('epoch').rstrip('.ckpt'))


def test_samples(sess, model, name, config):
    # sample z from which to generate images
    z_sampled = np.random.uniform(
        -1., 1., size=[config.sample_size, config.z_size]
    ).astype(np.float32)

    # generate images from the sampled z
    x_generated = sess.run(
        model.G, feed_dict={model.z_in: z_sampled}
    )

    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir, exist_ok=True)

    save_images(
        np.reshape(
            x_generated[:config.sample_size],
            [config.sample_size,
             config.image_size,
             config.image_size,
             config.channel_size]
        ),
        image_manifold_size(config.sample_size),
        '{}/{}.png'.format(config.sample_dir, name)
    )
