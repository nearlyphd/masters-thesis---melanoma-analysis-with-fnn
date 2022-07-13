import tensorflow as tf


class Manhattan(tf.keras.layers.Layer):
    def __init__(self):
        super(Manhattan, self).__init__()

    def call(self, inputs):
        batch_size, segment_number, segment_size, segment_size, channels = tf.unstack(tf.shape(inputs))
        outputs = tf.reshape(inputs, shape=(-1, segment_number, segment_size, channels))

        centers = tf.image.resize_with_crop_or_pad(outputs, 1, 1)

        outputs = tf.math.subtract(outputs, centers)
        outputs = tf.math.abs(outputs)
        outputs = tf.math.reduce_sum(outputs, axis=3)
        outputs = tf.math.less_equal(outputs, tf.cast(segment_size, dtype=tf.float32))
        outputs = tf.cast(outputs, dtype=tf.int32)
        outputs = tf.reshape(outputs, shape=(-1, segment_number, segment_size, segment_size))

        return outputs
