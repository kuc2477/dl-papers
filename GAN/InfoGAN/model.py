import tensorflow as tf
from tensorflow.contrib import slim
from activations import lrelu
from distributions import Product


class InfoGAN(object):
    def __init__(
            self, z_distributions, c_distributions,
            batch_size, reg_rate,
            image_size, channel_size, q_hidden_size,
            g_filter_number, d_filter_number,
            g_filter_size, d_filter_size):
        # model-wise initializer
        self._initialzier = tf.truncated_normal_initializer(stddev=0.002)

        # basic hyperparameters
        self.z_distribution = Product([d for d in z_distributions])
        self.c_distribution = Product([d for d in c_distributions])
        self.latent_distribution = Product([
            d for d in z_distributions + c_distributions
        ])
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.image_size = image_size
        self.channel_size = channel_size
        self.q_hidden_size = q_hidden_size
        self.g_filter_number = g_filter_number
        self.d_filter_number = d_filter_number
        self.g_filter_size = g_filter_size
        self.d_filter_size = d_filter_size

        # basic placeholders
        self.z_in = tf.placeholder(
            shape=[self.batch_size, self.z_distribution.dim], dtype=tf.float32
        )
        self.c_in = tf.placeholder(
            shape=[self.batch_size, self.c_distribution.dim], dtype=tf.float32
        )
        self.image_in = tf.placeholder(
            shape=[self.batch_size, image_size, image_size, channel_size],
            dtype=tf.float32,
        )

        # build a generator and a discriminator
        self.g = self.generator(self.z_in, self.c_in)
        self.d_x, self.d_x_logits = self.discriminator(self.image_in)
        self.d_g, self.d_g_logits = self.discriminator(self.g, reuse=True)
        self.encoded_g = self.discriminator(self.g, reuse=True, encode=True)

        # estimate parameters of the distribution c given g, using the features
        # extracted from the samples of g.
        self.estimated_parameters_of_distribution_c_given_g = \
            self.estimate_parameters_of_distribution_c_given_g(
                self.encoded_g
            )

        # estimate mutual information between c and c given g, using the
        # variational posterior distribution that we estimated above.
        self.estimated_mutual_information_between_c_and_c_given_g = \
            self.estimate_mutual_information_between_c_and_c_given_g(
                self.c_in,
                self.estimated_parameters_of_distribution_c_given_g
            )

        # build objective functions
        self.d_loss_real = self._sigmoid_cross_entropy_loss(
            logits=self.d_x_logits, labels=tf.ones_like(self.d_x)
        )
        self.d_loss_fake = self._sigmoid_cross_entropy_loss(
            logits=self.d_g_logits, labels=tf.zeros_like(self.d_g)
        )
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = self._sigmoid_cross_entropy_loss(
            logits=self.d_g_logits, labels=tf.ones_like(self.d_g)
        )
        # regularize objectives to maximize mutual information between
        # original code and reconstructed latent code.
        self.d_loss -= \
            self.estimated_mutual_information_between_c_and_c_given_g
        self.g_loss -= \
            self.estimated_mutual_information_between_c_and_c_given_g

        # variables
        self.g_vars = [v for v in tf.trainable_variables() if 'g_' in v.name]
        self.d_vars = [v for v in tf.trainable_variables() if 'd_' in v.name]
        self.q_vars = [v for v in tf.trainable_variables() if 'q_' in v.name]

    def generator(self, z, c):
        # project zc
        zc = tf.concat([z, c], axis=1)
        zc_filter_number = self.g_filter_number * 8
        zc_projection_size = self.image_size // 8
        zc_projected = slim.fully_connected(
            zc,
            zc_projection_size * zc_projection_size * zc_filter_number,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            weights_initializer=self._initialzier,
            scope='g_projection',
        )
        zc_reshaped = tf.reshape(
            zc_projected,
            [-1, zc_projection_size, zc_projection_size, zc_filter_number]
        )

        # transposed conv1
        g_conv1_filter_number = zc_filter_number // 2
        g_conv1 = slim.convolution2d_transpose(
            zc_reshaped,
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

    def discriminator(self, bottom, reuse=False, encode=False):
        d_conv1_filter_number = self.d_filter_number
        d_conv1 = slim.convolution2d(
            bottom,
            d_conv1_filter_number,
            kernel_size=[self.d_filter_size, self.d_filter_size],
            stride=[2, 2], padding='SAME',
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv1'
        )

        d_conv2_filter_number = d_conv1_filter_number * 2
        d_conv2 = slim.convolution2d(
            d_conv1,
            d_conv2_filter_number,
            kernel_size=[self.d_filter_size, self.d_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv2'
        )

        d_conv3_filter_number = d_conv2_filter_number * 2
        d_conv3 = slim.convolution2d(
            d_conv2,
            d_conv3_filter_number,
            kernel_size=[self.d_filter_size, self.d_filter_size],
            stride=[2, 2],
            padding='SAME',
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            biases_initializer=None,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='d_conv3'
        )

        d_conv3_flattened = slim.flatten(d_conv3)
        d_out = slim.fully_connected(
            d_conv3_flattened, 1,
            weights_initializer=self._initialzier,
            reuse=reuse,
            scope='d_out'
        )

        # return the encoded features or the discriminator's judgement.
        if encode:
            return d_conv3_flattened
        else:
            return tf.nn.sigmoid(d_out), d_out

    def estimate_parameters_of_distribution_c_given_g(
            self, encoded_g, reuse=False):
        projected = slim.fully_connected(
            encoded_g, self.q_hidden_size,
            weights_initializer=self._initialzier,
            activation_fn=lrelu,
            normalizer_fn=slim.batch_norm,
            reuse=reuse, scope='q_projection'
        )

        latent_code_approximated = slim.fully_connected(
            projected, self.c_distribution.dist_flat_dim,
            weights_initializer=self._initialzier,
            reuse=reuse, scope='q_out'
        )

        return latent_code_approximated

    def estimate_mutual_information_between_c_and_c_given_g(
            self, c, estimated_parameters_of_distribution_c_given_g):
        # Wrap flat estimated parameters into a dictionary.
        wrapped_parameters = self.c_distribution.activate_dist(
            estimated_parameters_of_distribution_c_given_g
        )

        # Calculate log likelihood of the distribution c and the estimated
        # distribution c given g for the sampled c.
        c_log_likelihoods = self.c_distribution.logli_prior(c)
        cg_log_likelihoods = self.c_distribution.logli(
            c, wrapped_parameters
        )

        # Estimate entropy of the distribution c and the distribution
        # c given g.
        estimated_c_entropy = -tf.reduce_mean(
            c_log_likelihoods, name='estimated_c_entropy'
        )
        estimated_cg_entropy = -tf.reduce_mean(
            cg_log_likelihoods, name='estimated_cg_entropy'
        )

        # Estimate mutual information between the distribution c and the
        # distribution c given g.
        return estimated_c_entropy - estimated_cg_entropy

    @staticmethod
    def _sigmoid_cross_entropy_loss(logits, labels):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels
        ))
