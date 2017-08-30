import tensorflow as tf
from model import SplitNet
from train import train


flags = tf.app.flags
FLAGS = flags.FLAGS


def main(_):
    global FLAGS
    # TODO: NOT IMPLEMENTED YET
    splitnet = SplitNet(FLAGS)

    if FLAGS.test:
        # TODO: NOT IMPLEMENTED YET
        pass
    else:
        train(splitnet)


if __name__ == "__main__":
    tf.app.run()
