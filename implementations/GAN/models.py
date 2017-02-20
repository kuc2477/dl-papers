import tensorflow as tf
from tensorflow.contrib import slim
from ..activations import lrelu


class DCGAN(object):
    def __init__(self, z_size, image_size, lr, beta1,
                 sample_dir='./figures',
                 model_dir='./models'):
        self._session = tf.Session()
        self._initialzier = tf.truncated_normal_initializer(stddev=0.2)
        self._z_size = z_size
        self._image_size = image_size
        self._lr = lr
        self._beta1 = beta1
        self._sample_dir = sample_dir
        self._model_dir = model_dir

        # placeholders
        z_in = tf.placeholder(shape=[None, z_size], dtype=tf.float32,)
        image_in = tf.placeholder(
            shape=[None, image_size, image_size],
            dtype=tf.float32,
        )

        # build graph using a generator and a discriminator.
        tf.reset_default_graph()
        G = self.generator(z_in)
        D_x = self.discriminator(image_in)
        D_g = self.discriminator(G, reuse=True)
        D_trainer, G_trainer, update_D, update_G = \
            self.build(z_in, image_in, G, D_x, D_g)

        # set trainers and update ops
        self.D_trainer = D_trainer
        self.G_trainer = G_trainer
        self.update_D = update_D
        self.update_G = update_G

        # initialize variables
        self._session.run(tf.initialize_all_variables())


    def generator(self, z):
        z_projected = slim.fully_connected(
            z, 4 * 4 * 256,
            activation_fn=tf.nn.relu,
            normalizer=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_projection',
        )

        z_reshaped = tf.reshape(z_projected, [-1, 4, 4, 256])

        g_conv1 = slim.conv2d_transpose(
            z_reshaped, num_outputs=64,
            kernel_size=[5, 5], stride=[2, 2], padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv1',
        )

        g_conv2 = slim.conv2d_transpose(
            g_conv1, num_outputs=32,
            kernel_size=[5, 5], stride=[2, 2], padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv2',
        )

        g_conv3 = slim.conv2d_transpose(
            g_conv2, num_outputs=16,
            kernel_size=[5, 5], stride=[2, 2], padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv3',
        )

        g_out = slim.conv2d_transpose(
            g_conv3, num_outputs=1,
            kernel_size=[32, 32], stride=[2, 2], padding='SAME',
            biases_initializer=None,
            activation_fn=tf.nn.tanh,
            weights_initializer=self._initialzier,
            scope='g_out'
        )

        return g_out

    def discriminator(self, bottom, initialize, reuse=False):
        d_conv1 = slim.conv2d(
            bottom, 16, kernel_size=[4, 4], stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv1'
        )

        d_conv2 = slim.conv2d(
            d_conv1, 32, kernel_size=[4, 4], stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv2'
        )

        d_conv3 = slim.conv2d(
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

    def build(self, z_in, image_in, G, D_x, D_g):
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

        return D_trainer, G_trainer, update_D, update_G
