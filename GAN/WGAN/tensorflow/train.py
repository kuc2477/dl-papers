import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
from data import DATASETS, DATASET_LENGTH_GETTERS


def _sample_z(cfg):
    return np.random.uniform(
        -1., 1., size=[cfg.batch_size, cfg.z_size]
    ).astype(np.float32)


def train(model, config, sess=None):
    # define optimizers and a model saver.
    C_traner = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1
    )
    G_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
    )

    # define parameter update tasks
    c_grads = C_traner.compute_gradients(model.c_loss, var_list=model.c_vars)
    g_grads = G_trainer.compute_gradients(model.g_loss, var_list=model.g_vars)
    update_C = C_traner.apply_gradients(c_grads)
    update_G = G_trainer.apply_gradients(g_grads)
    clip_C = [
        v.assign(tf.clip_by_value(v, -config.clip_size, config.clip_size))
        for v in model.c_vars
    ]

    # main training session context
    with sess or tf.Session() as sess:
        if config.resume:
            epoch_start = utils.load_checkpoint(sess, model, config) + 1
        else:
            epoch_start = 1
            sess.run(tf.global_variables_initializer())

        for epoch in range(epoch_start, config.epochs+1):
            dataset = DATASETS[config.dataset](config.batch_size)
            dataset_length = DATASET_LENGTH_GETTERS[config.dataset]()
            dataset_stream = tqdm(enumerate(dataset, 1))

            for batch_index, xs in dataset_stream:
                # place more weight on ciritic in the begining of the training.
                critic_update_ratio = (
                    30 if (batch_index < 25 or batch_index % 500 == 0) else
                    config.critic_update_ratio
                )

                # train the critic against the current generator and the data.
                for _ in range(critic_update_ratio):
                    zs = _sample_z(config)
                    _, c_loss = sess.run(
                        [update_C, model.c_loss],
                        feed_dict={
                            model.z_in: zs,
                            model.image_in: xs
                        }
                    )
                    sess.run(clip_C)

                # train the generator against the current critic.
                zs = _sample_z(config)
                _, g_loss = sess.run(
                    [update_G, model.g_loss],
                    feed_dict={model.z_in: zs}
                )

                # display current training process status
                dataset_stream.set_description((
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'g loss: {g_loss:.4f} | '
                    'w distance: {w_dist:.4f}'
                ).format(
                    epoch=epoch,
                    epochs=config.epochs,
                    trained=batch_index*config.batch_size,
                    total=dataset_length,
                    progress=(
                        100.
                        * batch_index
                        * config.batch_size
                        / dataset_length
                    ),
                    g_loss=g_loss,
                    w_dist=-c_loss,
                ))

                # test the samples
                if batch_index % config.log_for_every == 0:
                    name = 'epoch{}-fig{}'.format(epoch, batch_index)
                    utils.test_samples(sess, model, name, config)

            # save the model at the every end of the epochs.
            utils.save_checkpoint(sess, model, epoch, config)
