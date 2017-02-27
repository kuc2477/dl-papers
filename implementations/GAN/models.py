import os.path
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from activations import lrelu
import utils


class DCGAN(object):
    def __init__(self, z_size, image_size, learning_rate=0.0002, beta1=0.5,
                 iterations=5000, batch_size=128, generator_update_ratio=2,
                 log_for_every=10, save_for_every=10,
                 sample_dir='./figures', model_dir='./models'):
        # training related variables and hyperparameters
        self._initialzier = tf.truncated_normal_initializer(stddev=0.02)
        self._iterations = iterations
        self._batch_size = batch_size
        self._generator_update_ratio = generator_update_ratio
        self._log_for_every = log_for_every
        self._save_for_every = save_for_every
        self._sample_dir = sample_dir
        self._model_dir = model_dir

        # model hyperparameters
        self._z_size = z_size
        self._image_size = image_size
        self._learning_rate = learning_rate
        self._beta1 = beta1

        # placeholders
        self.z_in = z_in = tf.placeholder(
            shape=[None, z_size], dtype=tf.float32
        )
        self.image_in = image_in = tf.placeholder(
            shape=[None, image_size, image_size, 1], dtype=tf.float32
        )

        # build graph using a generator and a discriminator.
        self.G = G = self.generator(z_in)
        self.D_x = D_x = self.discriminator(image_in)
        self.D_g = D_g = self.discriminator(G, reuse=True)
        self.update_D, self.update_G, self.d_loss, self.g_loss = \
            self.build(G, D_x, D_g)

    def generator(self, z):
        z_projected = slim.fully_connected(
            z, 4 * 4 * 256,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_projection',
        )

        z_reshaped = tf.reshape(z_projected, [-1, 4, 4, 256])

        g_conv1 = slim.convolution2d_transpose(
            z_reshaped, num_outputs=64,
            kernel_size=[5, 5], stride=[2, 2], padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv1',
        )

        g_conv2 = slim.convolution2d_transpose(
            g_conv1, num_outputs=32,
            kernel_size=[5, 5], stride=[2, 2], padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv2',
        )

        g_conv3 = slim.convolution2d_transpose(
            g_conv2, num_outputs=16,
            kernel_size=[5, 5], stride=[2, 2], padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv3',
        )

        g_out = slim.convolution2d_transpose(
            g_conv3, num_outputs=1,
            kernel_size=[32, 32], padding='SAME',
            biases_initializer=None,
            activation_fn=tf.nn.tanh,
            weights_initializer=self._initialzier,
            scope='g_out'
        )

        return g_out

    def discriminator(self, bottom, reuse=False):
        d_conv1 = slim.convolution2d(
            bottom, 16, kernel_size=[4, 4], stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv1'
        )

        d_conv2 = slim.convolution2d(
            d_conv1, 32, kernel_size=[4, 4], stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv2'
        )

        d_conv3 = slim.convolution2d(
            d_conv2, 64, kernel_size=[4, 4], stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv3'
        )

        d_out = slim.fully_connected(
            slim.flatten(d_conv3), 1,
            weights_initializer=self._initialzier,
            activation_fn=tf.nn.sigmoid,
            reuse=reuse,
            scope='d_out'
        )

        return d_out

    def build(self, G, D_x, D_g):
        d_loss = -tf.reduce_mean(tf.log(D_x) + tf.log(1. - D_g))
        g_loss = -tf.reduce_mean(tf.log(D_g))

        D_trainer = tf.train.AdamOptimizer(
            learning_rate=self._learning_rate,
            beta1=self._beta1
        )
        G_trainer = tf.train.AdamOptimizer(
            learning_rate=self._learning_rate,
            beta1=self._beta1,
        )

        trainables = tf.trainable_variables()
        d_grads = D_trainer.compute_gradients(d_loss, var_list=trainables[9:])
        g_grads = G_trainer.compute_gradients(g_loss, var_list=trainables[:9])
        update_D = D_trainer.apply_gradients(d_grads)
        update_G = G_trainer.apply_gradients(g_grads)
        return update_D, update_G, d_loss, g_loss


    def train(self, config, sess=None):
        mnist = input_data.read_data_sets('MNIST/', one_hot=False)
        saver = tf.train.Saver()
        with sess or tf.Session() as sess:
            try:
                sess.run(tf.initialize_all_varaibles())
            except:
                sess.run(tf.global_variables_initializer())

            for i in range(config.iterations):
                # sample z from uniform distribution and prepare 
                # fake/real images
                zs = np.random.uniform(
                    -1., 1., size=[config.batch_size, self._z_size]
                ).astype(np.float32)
                xs, _ = mnist.train.next_batch(config.batch_size)
                xs = (np.reshape(xs, [config.batch_size, 28, 28, 1]) - 0.5) * 2.
                xs = np.lib.pad(
                    xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', 
                    constant_values=(-1, -1)
                )

                # run discriminator trainer
                _, d_loss = sess.run(
                    [self.update_D, self.d_loss], 
                    feed_dict={
                        self.z_in: zs, 
                        self.image_in: xs
                    }
                )
                # run generator trainer
                for _ in range(config.generator_update_ratio):
                    _, g_loss = sess.run(
                        [self.update_G, self.g_loss], 
                        feed_dict={self.z_in: zs}
                    )

                if i % config.log_for_every == 0:
                    # log current training process status
                    print('Generator Loss: {} / Discriminator Loss: {}'.format(
                        g_loss, d_loss
                    ))
                    if not os.path.exists(config.sample_dir):
                        os.makedirs(config.sample_dir, exist_ok=True)

                    # generate images from the current generator
                    z_sampled = np.random.uniform(
                        -1., 1., size=[config.batch_size, self._z_size]
                    ).astype(np.float32)

                    x_generated = sess.run(
                        self.G, feed_dict={self.z_in: z_sampled}
                    )

                    utils.save_images(
                        np.reshape(
                            x_generated[:config.sample_size], 
                            [config.sample_size, 
                             self._image_size, 
                             self._image_size]
                        ), 
                        [config.sample_size // 6, 6], 
                        '{}/fig{}.png'.format(config.sample_dir, i)
                    )

                # save the model
                if i % config.save_for_every == 0 and i != 0:
                    if not os.path.exists(config.model_dir):
                        os.makedirs(config.model_dir, exist_ok=True)
                    path = '{}/model-{}.cptk'.format(config.model_dir, i)
                    saver.save(sess, path)
                    print('saved model to {}'.format(path))
