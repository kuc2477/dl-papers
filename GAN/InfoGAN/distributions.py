import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import (
    RelaxedOneHotCategorical,
    MultivariateNormalDiag,
    Uniform,
)


# ===============
# Hyperparameters
# ===============
_TEMPERATURE = 0.25


# =============
# Distributions
# =============

def uniform(size):
    return Uniform(low=.0, high=tf.ones(size))


def normal(size):
    return MultivariateNormalDiag(
        loc=tf.zeros(size),
        scale_diag=tf.ones(size)
    )


def normal_for_given_uniform_size(uniform_dist):
    return normal(uniform_dist.batch_shape[0])


def categorical(size):
    return RelaxedOneHotCategorical(_TEMPERATURE, tf.ones(size) / size)


# distributions allowed
DISTRIBUTIONS = {
    'uniform': uniform,
    'normal': normal,
    'categorical': categorical,
}


# ========
# Sampling
# ========

# z sampling function
def sample_z(sess, cfg):
    uniform_distribution = uniform(cfg.z_size)
    return sess.run(uniform_distribution.sample(cfg.batch_size, 1))


# latent code sampling function
def sample_c(sess, cfg):
    distribution_classes = [DISTRIBUTIONS[n] for n in cfg.c_distributions]
    distribution_sizes = cfg.c_sizes
    distributions = [
        distribution_class(size) for
        distribution_class, size in zip(
            distribution_classes, distribution_sizes
        )
    ]
    sampled = np.array([d.sample(cfg.batch_size) for d in distributions])
    return np.c_(sampled)
