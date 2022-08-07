import tensorflow as tf


class FractalDimension(tf.keras.layers.Layer):
    def __init__(self):
        super(FractalDimension, self).__init__()

    def call(self, inputs):
        batch_size, _len = tf.unstack(tf.shape(inputs))
        numbers = tf.reshape(
            tf.concat(
                [tf.constant([1], dtype=tf.float32), tf.range(1, _len, dtype=tf.float32)],
                axis=0
            ),
            shape=(1, -1)
        )

        outputs = tf.math.divide(inputs, numbers)
        outputs = tf.math.reduce_sum(outputs, axis=1)

        return outputs
