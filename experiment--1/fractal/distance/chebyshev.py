import tensorflow as tf


class Chebyshev(tf.keras.layers.Layer):
    def __init__(self):
        super(Chebyshev, self).__init__()

    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size, channels = tf.unstack(tf.shape(inputs))
        outputs = tf.reshape(inputs, shape=(-1, patch_number, patch_size, channels))

        centers = tf.image.resize_with_crop_or_pad(outputs, 1, 1)

        outputs = tf.math.subtract(outputs, centers)
        outputs = tf.math.abs(outputs)
        outputs = tf.math.reduce_max(outputs, axis=3)
        outputs = tf.math.less_equal(outputs, tf.cast(patch_size, dtype=tf.float32))
        outputs = tf.cast(outputs, dtype=tf.int32)
        outputs = tf.reshape(outputs, shape=(-1, patch_number, patch_size, patch_size))

        return outputs
