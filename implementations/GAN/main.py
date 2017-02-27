import os.path
import numpy as np
import tensorflow as tf
from models import DCGAN
import pprint


flags = tf.app.flags
flags.DEFINE_integer('z_size', 100, 'size of latent code z [100]')
flags.DEFINE_integer('image_size', 32, 'size of image [32]')
flags.DEFINE_float('learning_rate', 0.0002, 'learning rate for Adam [0.0002]')
flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam [0.5]')
flags.DEFINE_integer('iterations', 5000, 'training iteration number')
flags.DEFINE_integer('batch_size', 128, 'training batch size')
flags.DEFINE_integer('sample_size', 36, 'generator sample size')
flags.DEFINE_integer(
    'generator_update_ratio', 2, 
    'number of updates for generator parameters per discriminator\'s updates'
)
flags.DEFINE_integer('log_for_every', 10, 'number of batches per logging')
flags.DEFINE_integer(
    'save_for_every', 1000, 'number of batches per saving the model'
)
flags.DEFINE_string('sample_dir', 'figures', 'directory of generated figures')
flags.DEFINE_string('model_dir', 'models', 'directory of trained models')
FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(FLAGS.__flags)
    dcgan = DCGAN(z_size=FLAGS.z_size, image_size=FLAGS.image_size)
    dcgan.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
