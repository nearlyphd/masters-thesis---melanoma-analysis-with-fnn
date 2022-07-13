import tensorflow as tf


class Probability(tf.keras.layers.Layer):
    def __init__(self):
        super(Probability, self).__init__()

    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size = tf.unstack(tf.shape(inputs))

        outputs = tf.math.reduce_sum(inputs, axis=(2, 3))
        outputs = tf.vectorized_map(lambda image: tf.math.bincount(image, minlength=patch_size ** 2 + 1), outputs)
        outputs = tf.math.divide(outputs, patch_number)

        return outputs
