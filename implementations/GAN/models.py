import tensorflow as tf
from tensorflow.contrib import slim
from activations import lrelu


class DCGAN(object):
    def __init__(self, z_size, image_size, lr=0.0002, beta1=0.5, sess=None,
                 sample_dir='./figures', model_dir='./models'):
        self._initialzier = tf.truncated_normal_initializer(stddev=0.02)
        self._z_size = z_size
        self._image_size = image_size
        self._lr = lr
        self._beta1 = beta1

        # placeholders
        self.z_in = z_in = tf.placeholder(
            shape=[None, z_size], dtype=tf.float32,
        )
        self.image_in = image_in = tf.placeholder(
            shape=[None, image_size, image_size, 1],
            dtype=tf.float32,
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
            learning_rate=self._lr,
            beta1=self._beta1
        )
        G_trainer = tf.train.AdamOptimizer(
            learning_rate=self._lr,
            beta1=self._beta1,
        )

        trainables = tf.trainable_variables()
        d_grads = D_trainer.compute_gradients(d_loss, var_list=trainables[9:])
        g_grads = G_trainer.compute_gradients(g_loss, var_list=trainables[:9])
        update_D = D_trainer.apply_gradients(d_grads)
        update_G = G_trainer.apply_gradients(g_grads)

        return update_D, update_G, d_loss, g_loss


    def train(self):
        # TODO: NOT IMPLEMENTED YET
        pass
