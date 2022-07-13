import tensorflow as tf


class Lacunarity(tf.keras.layers.Layer):
    def __init__(self):
        super(Lacunarity, self).__init__()

    def call(self, inputs):
        batch_size, _len = tf.unstack(tf.shape(inputs))
        numbers = tf.reshape(
            tf.concat(
                [tf.constant([1], dtype=tf.float32), tf.range(1, _len, dtype=tf.float32)],
                axis=0
            ),
            shape=(1, -1)
        )

        mu_first_2 = tf.math.multiply(inputs, numbers)
        mu_first_2 = tf.math.reduce_sum(mu_first_2, axis=1)
        mu_first_2 = tf.math.pow(mu_first_2, 2)

        mu_second = tf.math.pow(numbers, 2)
        mu_second = tf.math.multiply(inputs, mu_second)
        mu_second = tf.math.reduce_sum(mu_second, axis=1)

        outputs = tf.math.divide(
            tf.math.subtract(mu_second, mu_first_2),
            mu_first_2
        )

        return outputs
