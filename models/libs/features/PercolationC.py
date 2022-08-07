import tensorflow as tf


class PercolationC(tf.keras.layers.Layer):
    def __init__(self):
        super(PercolationC, self).__init__()

    def call(self, inputs):
        outputs = tf.cast(inputs, dtype=tf.float32)
        outputs = tf.math.reduce_max(outputs, axis=(2, 3))
        outputs = tf.math.reduce_mean(outputs, axis=1)

        return outputs
