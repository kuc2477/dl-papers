import os.path
import numpy as np
import tensorflow as tf
import utils
from data import DATASETS


def train(session, model, config):
    # define optimizers
    d_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1
    )
    g_trainer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
    )

    # define parameter update tasks
    d_grads = d_trainer.compute_gradients(model.d_loss, var_list=(
        model.d_vars + model.q_vars
    ))
    g_grads = g_trainer.compute_gradients(model.g_loss, var_list=(
        model.g_vars + model.q_vars
    ))
    update_d = d_trainer.apply_gradients(d_grads)
    update_g = g_trainer.apply_gradients(g_grads)

    # prepare training data and saver
    dataset = DATASETS[config.dataset](config.batch_size)
    saver = tf.train.Saver()

    # prepare summaries
    tf.summary.scalar('d_loss_fake', model.d_loss_fake)
    tf.summary.histogram('encoded_g', model.encoded_g)
    tf.summary.histogram(
        'estimated_parameters_of_distribution_c_given_g',
        model.estimated_parameters_of_distribution_c_given_g
    )
    tf.summary.scalar(
        'estimated_mutual_information_between_c_and_c_given_g',
        model.estimated_mutual_information_between_c_and_c_given_g
    )
    summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./log', graph=session.graph)

    # main training session context
    with session:
        try:
            session.run(tf.initialize_all_varaibles())
        except:
            session.run(tf.global_variables_initializer())

        for i in range(config.iterations):
            # sample z prepare real images
            zs_d, cs_d = session.run([
                model.z_distribution.sample_prior(config.batch_size),
                model.c_distribution.sample_prior(config.batch_size)
            ])
            xs = next(dataset)
            # run discriminator trainer
            _, d_loss = session.run(
                [update_d, model.d_loss],
                feed_dict={
                    model.z_in: zs_d,
                    model.c_in: cs_d,
                    model.image_in: xs
                }
            )

            for _ in range(config.generator_update_ratio):
                # run generator trainer
                zs_g, cs_g = session.run([
                    model.z_distribution.sample_prior(config.batch_size),
                    model.c_distribution.sample_prior(config.batch_size)
                ])
                _, g_loss = session.run(
                    [update_g, model.g_loss], feed_dict={
                        model.z_in: zs_g,
                        model.c_in: cs_g
                    }
                )

            if i % config.log_for_every == 0:
                # log current training process status
                print('Generator Loss: {} / Discriminator Loss: {}'.format(
                    g_loss, d_loss
                ))
                if not os.path.exists(config.sample_dir):
                    os.makedirs(config.sample_dir, exist_ok=True)

                # generate images and summaries from the sampled z
                x_generated, summaries_to_be_written = session.run(
                    [model.g, summaries], feed_dict={
                        model.z_in: zs_d,
                        model.c_in: cs_d,
                        model.image_in: xs
                    }
                )

                # write generated images
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

                # write summaries
                summary_writer.add_summary(summaries_to_be_written, i)

            # save the model
            if i % config.save_for_every == 0 and i != 0:
                if not os.path.exists(config.model_dir):
                    os.makedirs(config.model_dir, exist_ok=True)
                path = '{}/model-{}.cptk'.format(config.model_dir, i)
                saver.save(session, path)
                print('saved model to {}'.format(path))
