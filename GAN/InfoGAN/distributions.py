import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import (
    RelaxedOneHotCategorical,
    Uniform,
)


# distributions allowed
_TEMPERATURE = 0.08
DISTRIBUTIONS = {
    'categorical': lambda size: RelaxedOneHotCategorical(
        _TEMPERATURE, tf.ones(size) / size
    ),
    'uniform': lambda size: Uniform(
        low=.0, high=tf.ones(size)
    )
}


# z sampling function
def sample_z(sess, cfg):
    return np.random.uniform(
        -1., 1., size=[cfg.batch_size, cfg.z_size]
    ).astype(np.float32)


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
