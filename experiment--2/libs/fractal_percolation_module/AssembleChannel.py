import tensorflow as tf


class AssembleChannel(tf.keras.layers.Layer):
    def __init__(self):
        super(AssembleChannel, self).__init__()

    def call(self, inputs):
        fractal_dimension = tf.convert_to_tensor(inputs[0])
        fractal_dimension = tf.transpose(fractal_dimension, perm=(1, 0))

        lacunarity = tf.convert_to_tensor(inputs[1])
        lacunarity = tf.transpose(lacunarity, perm=(1, 0))

        percolation_q = tf.convert_to_tensor(inputs[2])
        percolation_q = tf.transpose(percolation_q, perm=(1, 0))

        percolation_c = tf.convert_to_tensor(inputs[3])
        percolation_c = tf.transpose(percolation_c, perm=(1, 0))

        percolation_m = tf.convert_to_tensor(inputs[4])
        percolation_m = tf.transpose(percolation_m, perm=(1, 0))

        outputs = tf.concat([
            percolation_c,
            percolation_q,
            percolation_m,
            lacunarity,
            fractal_dimension
        ], axis=1)
        outputs = tf.reshape(outputs, shape=(-1, 10, 10))

        return outputs
