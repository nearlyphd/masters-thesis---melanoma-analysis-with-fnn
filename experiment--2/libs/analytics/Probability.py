import tensorflow as tf


class Probability(tf.keras.layers.Layer):
    def __init__(self):
        super(Probability, self).__init__()

    def call(self, inputs):
        batch_size, segment_number, segment_size, segment_size = tf.unstack(tf.shape(inputs))

        outputs = tf.math.reduce_sum(inputs, axis=(2, 3))
        outputs = tf.vectorized_map(lambda image: tf.math.bincount(image, minlength=segment_size ** 2 + 1), outputs)
        outputs = tf.math.divide(outputs, segment_number)

        return outputs

