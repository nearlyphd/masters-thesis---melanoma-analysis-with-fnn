import tensorflow as tf


class AssembleChannel(tf.keras.layers.Layer):
    def __init__(self, fractal_width, fractal_height):
        super(AssembleChannel, self).__init__()

        self.fractal_width = fractal_width
        self.fractal_height = fractal_height

    def call(self, inputs):
        fractal_dimension = tf.convert_to_tensor(inputs[0])
        fractal_dimension = tf.transpose(fractal_dimension, perm=(1, 0))

        lacunarity = tf.convert_to_tensor(inputs[1])
        lacunarity = tf.transpose(lacunarity, perm=(1, 0))

        outputs = tf.concat([
            lacunarity,
            fractal_dimension
        ], axis=1)
        outputs = tf.reshape(outputs, shape=(-1, self.fractal_width, self.fractal_height))

        return outputs
