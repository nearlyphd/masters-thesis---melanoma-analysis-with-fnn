import tensorflow as tf
import tensorflow_addons as tfa


class Clusterize(tf.keras.layers.Layer):
    def __init__(self):
        super(Clusterize, self).__init__()

    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size = tf.unstack(tf.shape(inputs))

        outputs = tf.reshape(inputs, shape=(-1, patch_size, patch_size))
        outputs = tfa.image.connected_components(outputs)
        outputs = tf.reshape(outputs, shape=(-1, patch_number, patch_size, patch_size))

        return outputs
