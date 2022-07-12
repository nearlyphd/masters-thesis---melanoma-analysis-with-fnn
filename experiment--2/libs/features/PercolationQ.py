import tensorflow as tf


class PercolationQ(tf.keras.layers.Layer):
    def __init__(self, threshold=0.59275):
        super(PercolationQ, self).__init__()

        self.threshold = threshold

    def call(self, inputs):
        batch_size, segment_number, segment_size, segment_size = tf.unstack(tf.shape(inputs))

        outputs = tf.math.reduce_sum(inputs, axis=(2, 3))
        outputs = tf.math.divide(outputs, segment_size ** 2)
        outputs = tf.math.greater_equal(outputs, self.threshold)
        outputs = tf.cast(outputs, dtype=tf.float32)
        outputs = tf.math.reduce_mean(outputs, axis=1)

        return outputs
