import tensorflow as tf


class PercolationM(tf.keras.layers.Layer):
    def __init__(self):
        super(PercolationM, self).__init__()

    def call(self, inputs):
        batch_size, segment_number, segment_size, segment_size = tf.unstack(tf.shape(inputs))

        outputs = tf.reshape(inputs, shape=(-1, segment_number, segment_size ** 2))
        outputs = tf.map_fn(lambda image: tf.math.reduce_max(tf.math.bincount(image)), outputs)
        outputs = tf.cast(outputs, dtype=tf.float32)

        return outputs
