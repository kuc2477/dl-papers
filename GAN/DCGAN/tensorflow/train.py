import os.path
import numpy as np
import tensorflow as tf
import utils
from .data import DATASETS


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

    # z sampling function
    def _sample_z(cfg):
        return np.random.uniform(
            -1., 1., size=[cfg.batch_size, cfg.z_size]
        ).astype(np.float32)

    # main training session context
    with sess or tf.Session() as sess:
        try:
            sess.run(tf.initialize_all_varaibles())
        except:
            sess.run(tf.global_variables_initializer())

        for i in range(config.iterations):
            # sample z prepare real images
            zs = _sample_z(config)
            xs = next(dataset)
            # run discriminator trainer
            _, d_loss = sess.run(
                [update_D, model.d_loss],
                feed_dict={
                    model.z_in: zs,
                    model.image_in: xs
                }
            )

            for _ in range(config.generator_update_ratio):
                # run generator trainer
                zs = _sample_z(config)
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

                # generate images from the sampled z
                z_sampled = _sample_z(config)
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
