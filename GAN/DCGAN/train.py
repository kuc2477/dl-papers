import os.path
import numpy as np
import tensorflow as tf
import utils
from data import DATASETS


def train(model, config, sess=None):
    # define optimizers
    D_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1
    )
    G_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
    )

    # get parameter update tasks
    d_grads = D_trainer.compute_gradients(model.d_loss, var_list=model.d_vars)
    g_grads = G_trainer.compute_gradients(model.g_loss, var_list=model.g_vars)
    update_D = D_trainer.apply_gradients(d_grads)
    update_G = G_trainer.apply_gradients(g_grads)

    # prepare training data and saver
    dataset = DATASETS[config.dataset](config.batch_size)
    saver = tf.train.Saver()

    with sess or tf.Session() as sess:
        try:
            sess.run(tf.initialize_all_varaibles())
        except:
            sess.run(tf.global_variables_initializer())

        for i in range(config.iterations):
            # sample z from uniform distribution and prepare fake/real
            # images
            zs = np.random.uniform(
                -1., 1., size=[config.batch_size, config.z_size]
            ).astype(np.float32)
            xs = next(dataset)

            # we need to pad a little bit for mnist dataset.
            if config.dataset == 'mnist':
                xs = np.reshape(xs, [config.batch_size, 28, 28, 1]) - 0.5
                xs *= 2.
                xs = np.lib.pad(
                    xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                    constant_values=(-1, -1)
                )

            # run discriminator trainer
            _, d_loss = sess.run(
                [update_D, model.d_loss],
                feed_dict={
                    model.z_in: zs,
                    model.image_in: xs
                }
            )

            # run generator trainer
            for _ in range(config.generator_update_ratio):
                _, g_loss = sess.run(
                    [update_G, model.g_loss],
                    feed_dict={model.z_in: zs}
                )

            if i % config.log_for_every == 0:
                # log current training process status
                print('Generator Loss: {} / Discriminator Loss: {}'.format(
                    g_loss, d_loss
                ))
                if not os.path.exists(config.sample_dir):
                    os.makedirs(config.sample_dir, exist_ok=True)

                # sample z from which to generate images
                z_sampled = np.random.uniform(
                    -1., 1., size=[config.batch_size, config.z_size]
                ).astype(np.float32)

                # generate images from the sampled z
                x_generated = sess.run(
                    model.G, feed_dict={model.z_in: z_sampled}
                )

                utils.save_images(
                    np.reshape(
                        x_generated[:config.sample_size],
                        [config.sample_size,
                         config.image_size,
                         config.image_size]
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
