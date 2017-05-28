import pprint
import tensorflow as tf
from data import DATASETS
from model import DCGAN
from train import train


flags = tf.app.flags
flags.DEFINE_integer('z_size', 100, 'size of latent code z [100]')
flags.DEFINE_integer('image_size', 32, 'size of image [32]')
flags.DEFINE_integer('channel_size', 1, 'size of channel [1]')
flags.DEFINE_integer(
    'g_filter_number', 64,
    'number of generator\'s filters at the last transposed conv layer'
)
flags.DEFINE_integer(
    'd_filter_number', 64,
    'number of discriminator\'s filters at the first conv layer'
)
flags.DEFINE_integer('g_filter_size', 5, 'generator\'s filter size')
flags.DEFINE_integer('d_filter_size', 4, 'discriminator\'s filter size')
flags.DEFINE_float('learning_rate', 0.00002,
                   'learning rate for Adam [0.00002]')
flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam [0.5]')
flags.DEFINE_string('dataset', 'mnist', 'dataset to use {}'.format(
    DATASETS.keys()
))
flags.DEFINE_bool('resize', True, 'whether to resize images on the fly or not')
flags.DEFINE_bool(
    'crop', True,
    'whether to use crop for image resizing or not'
)
flags.DEFINE_integer('iterations', 5000, 'training iteration number')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('sample_size', 36, 'generator sample size')
flags.DEFINE_integer('log_for_every', 100, 'number of batches per logging')
flags.DEFINE_integer(
    'save_for_every', 500, 'number of batches per saving the model'
)
flags.DEFINE_integer(
    'generator_update_ratio', 2,
    'number of updates for generator parameters per discriminator\'s updates'
)
flags.DEFINE_bool('test', False, 'flag defining whether it is in test mode')
flags.DEFINE_string('sample_dir', 'figures', 'directory of generated figures')
flags.DEFINE_string('model_dir', 'checkpoints', 'directory of trained models')
FLAGS = flags.FLAGS


def _patch_flags_with_dataset(flags_):
    flags_.image_size = DATASETS[flags_.dataset].image_size or \
        flags_.image_size
    flags_.channel_size = DATASETS[flags_.dataset].channel_size or \
        flags_.channel_size
    return flags_


def main(_):
    global FLAGS

    # patch and display flags with dataset's width and height
    FLAGS = _patch_flags_with_dataset(FLAGS)
    pprint.PrettyPrinter().pprint(FLAGS.__flags)

    # compile the model
    dcgan = DCGAN(
        z_size=FLAGS.z_size,
        image_size=FLAGS.image_size,
        channel_size=FLAGS.channel_size,
        g_filter_number=FLAGS.g_filter_number,
        d_filter_number=FLAGS.d_filter_number,
        g_filter_size=FLAGS.g_filter_size,
        d_filter_size=FLAGS.d_filter_size,
    )

    # test / train the model
    if FLAGS.test:
        # TODO: NOT IMPLEMENTED YET
        print('TEST MODE NOT IMPLEMENTED YET')
    else:
        train(dcgan, FLAGS)


if __name__ == '__main__':
    tf.app.run()
