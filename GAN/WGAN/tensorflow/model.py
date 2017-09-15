import tensorflow as tf
from tensorflow.contrib import slim
from activations import lrelu


class WGAN(object):
    def __init__(
        self, label,
        z_size, image_size, channel_size,
        g_filter_number, c_filter_number,
        g_filter_size, c_filter_size,
    ):
        # model's label.
        self.label = label

        # model-wise initializer
        self._initialzier = tf.truncated_normal_initializer(stddev=0.002)

        # basic hyperparameters
        self.z_size = z_size
        self.image_size = image_size
        self.channel_size = channel_size
        self.g_filter_number = g_filter_number
        self.c_filter_number = c_filter_number
        self.g_filter_size = g_filter_size
        self.c_filter_size = c_filter_size

        # basic placeholders
        self.z_in = tf.placeholder(
            shape=[None, z_size], dtype=tf.float32
        )
        self.image_in = tf.placeholder(
            shape=[None, image_size, image_size, channel_size],
            dtype=tf.float32,
        )

        # build graph using a generator and a critic.
        self.G = self.generator(self.z_in)
        self.C_x, self.C_x_logits = self.critic(self.image_in)
        self.C_g, self.C_g_logits = self.critic(self.G, reuse=True)

        # build objective function
        self.c_expected_logits_real = tf.reduce_mean(self.C_x_logits)
        self.c_expected_logits_fake = tf.reduce_mean(self.C_g_logits)
        self.c_loss = -(
            self.c_expected_logits_real -
            self.c_expected_logits_fake
        )
        self.g_loss = -self.c_expected_logits_fake
        self.g_vars = [v for v in tf.trainable_variables() if 'g_' in v.name]
        self.c_vars = [v for v in tf.trainable_variables() if 'c_' in v.name]

    @property
    def name(self):
        return (
            'WGAN'
            '-{g_filter_number}g'
            '-{c_filter_number}c'
            '-{label}-{size}x{size}x{channels}'
        ).format(
            g_filter_number=self.g_filter_number,
            c_filter_number=self.c_filter_number,
            label=self.label,
            size=self.image_size,
            channels=self.channel_size
        )

    def generator(self, z):
        # project z
        z_filter_number = self.g_filter_number * 8
        z_projection_size = self.image_size // 8
        z_projected = slim.fully_connected(
            z,
            z_projection_size * z_projection_size * z_filter_number,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_projection',
        )
        z_reshaped = tf.reshape(
            z_projected,
            [-1, z_projection_size, z_projection_size, z_filter_number]
        )

        # transposed conv1
        g_conv1_filter_number = z_filter_number // 2
        g_conv1 = slim.convolution2d_transpose(
            z_reshaped,
            num_outputs=g_conv1_filter_number,
            kernel_size=[self.g_filter_size, self.g_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv1',
        )

        # transposed conv2
        g_conv2_filter_number = g_conv1_filter_number // 2
        g_conv2 = slim.convolution2d_transpose(
            g_conv1,
            num_outputs=g_conv2_filter_number,
            kernel_size=[self.g_filter_size, self.g_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv2',
        )

        # transposed conv3
        g_conv3_filter_number = g_conv2_filter_number // 2
        g_conv3 = slim.convolution2d_transpose(
            g_conv2,
            num_outputs=g_conv3_filter_number,
            kernel_size=[self.g_filter_size, self.g_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_conv3',
        )

        # out
        g_out = slim.convolution2d_transpose(
            g_conv3,
            num_outputs=self.channel_size,
            kernel_size=[self.image_size, self.image_size], padding='SAME',
            biases_initializer=None,
            activation_fn=tf.nn.tanh,
            weights_initializer=self._initialzier,
            scope='g_out'
        )

        return g_out

    def critic(self, bottom, reuse=False):
        c_conv1_filter_number = self.c_filter_number
        c_conv1 = slim.convolution2d(
            bottom,
            c_conv1_filter_number,
            kernel_size=[self.c_filter_size, self.c_filter_size],
            stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='c_conv1'
        )

        c_conv2_filter_number = c_conv1_filter_number * 2
        c_conv2 = slim.convolution2d(
            c_conv1,
            c_conv2_filter_number,
            kernel_size=[self.c_filter_size, self.c_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='c_conv2'
        )

        c_conv3_filter_number = c_conv2_filter_number * 2
        c_conv3 = slim.convolution2d(
            c_conv2,
            c_conv3_filter_number,
            kernel_size=[self.c_filter_size, self.c_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='c_conv3'
        )

        c_out = slim.fully_connected(
            slim.flatten(c_conv3), 1,
            weights_initializer=self._initialzier,
            reuse=reuse,
            scope='c_out'
        )

        return tf.nn.sigmoid(c_out), c_out
