import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
from data import DATASETS, DATASET_LENGTH_GETTERS


def _sample_z(sample_size, z_size):
    return np.random.uniform(
        -1., 1., size=[sample_size, z_size]
    ).astype(np.float32)


def train(model, config, session=None):
    # define a session if needed.
    session = session or tf.Session()

    # define summaries.
    summary_writer = tf.summary.FileWriter(config.log_dir, session.graph)
    image_summary = tf.summary.image('generated images', model.G)
    loss_summaries = tf.summary.merge([
        tf.summary.scalar('wasserstein distance', -model.c_loss),
        tf.summary.scalar('generator loss', model.g_loss),
    ])

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
    with session:
        if config.resume:
            epoch_start = utils.load_checkpoint(session, model, config) + 1
        else:
            epoch_start = 1
            session.run(tf.global_variables_initializer())

        for epoch in range(epoch_start, config.epochs+1):
            dataset = DATASETS[config.dataset](config.batch_size)
            dataset_length = DATASET_LENGTH_GETTERS[config.dataset]()
            dataset_stream = tqdm(enumerate(dataset, 1))

            for batch_index, xs in dataset_stream:
                # where are we?
                iteration = (epoch-1)*dataset_length + batch_index

                # place more weight on ciritic in the begining of the training.
                critic_update_ratio = (
                    30 if (batch_index < 25 or batch_index % 500 == 0) else
                    config.critic_update_ratio
                )

                # train the critic against the current generator and the data.
                for _ in range(critic_update_ratio):
                    zs = _sample_z(config)
                    _, c_loss = session.run(
                        [update_C, model.c_loss],
                        feed_dict={
                            model.z_in: zs,
                            model.image_in: xs
                        }
                    )
                    session.run(clip_C)

                # train the generator against the current critic.
                zs = _sample_z(config)
                _, g_loss = session.run(
                    [update_G, model.g_loss],
                    feed_dict={model.z_in: zs}
                )

                # display current training process status
                dataset_stream.set_description((
                    'epoch: {epoch}/{epochs} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'g loss: {g_loss:.3f} | '
                    'w distance: {w_dist:.3f}'
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

                # log the generated samples
                if iteration % config.image_log_interval == 0:
                    zs = _sample_z(config.sample_size, model.z_size)
                    summary_writer.add_summary(session.run(
                        image_summary, feed_dict={
                            model.z_in: zs
                        }
                    ), iteration)

                # log the losses
                if iteration % config.loss_log_interval == 0:
                    zs = _sample_z(config.batch_size, model.z_size)
                    summary_writer.add_summary(session.run(
                        loss_summaries, feed_dict={
                            model.z_in: zs,
                            model.image_in: xs
                        }
                    ), iteration)

            # save the model at the every end of the epochs.
            utils.save_checkpoint(session, model, epoch, config)
