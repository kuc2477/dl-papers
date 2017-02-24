import os.path
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from models import DCGAN
import utils



# ===============================
# Model / Training Configurations
# ===============================

z_size = 100
image_size = 32
lr = 0.0002
beta1 = 0.5

iterations = 5000
batch_size = 128
sample_size = 36
generator_loop_ratio = 2
log_for_every = 10
save_for_every = 1000

sample_dir = './figures'
model_dir = './models'


def train(model, iterations=5000, batch_size=128,
          sample_size = 36, generator_loop_ratio=2, 
          log_for_every=10, save_for_every=1000):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        try:
            sess.run(tf.initialize_all_varaibles())
        except:
            sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            # sample z from uniform distribution and prepare fake/real images
            zs = np.random.uniform(
                -1., 1., size=[batch_size, z_size]
            ).astype(np.float32)
            xs, _ = mnist.train.next_batch(batch_size)
            xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.
            xs = np.lib.pad(
                xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', 
                constant_values=(-1, -1)
            )

            # run discriminator trainer
            _, d_loss = sess.run(
                [model.update_D, model.d_loss], 
                feed_dict={
                    model.z_in: zs, 
                    model.image_in: xs
                }
            )

            # run generator trainer
            for _ in range(generator_loop_ratio):
                _, g_loss = sess.run(
                    [model.update_G, model.g_loss], 
                    feed_dict={model.z_in: zs}
                )

            # log current training process status
            if i % log_for_every == 0:
                print('Generator Loss: {} / Discriminator Loss: {}'.format(
                    g_loss, d_loss
                ))

                z_sampled = \
                    np.random.uniform(-1., 1., size=[batch_size, z_size])\
                    .astype(np.float32)
                x_generated = sess.run(model.G, feed_dict={model.z_in: z_sampled})

                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir, exist_ok=True)

                utils.save_images(np.reshape(
                    x_generated[:sample_size], 
                    [sample_size, image_size, image_size]
                ), [sample_size // 6, 6], '{}/fig{}.png'.format(sample_dir, i))

            # save the model
            if i % save_for_every == 0 and i != 0:
                if not os.path.exists(model_dir):
                    os.mkdirs(model_dir, exist_ok=True)
                saver.save(sess, '{}/model-{}.cptk'.format(model_dir, i))
                print('saved model')
