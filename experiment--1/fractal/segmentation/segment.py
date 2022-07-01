import tensorflow as tf


class Segment(tf.keras.layers.Layer):
    def __init__(self, segment_size):
        super(Segment, self).__init__()

        self.segment_size = segment_size

    def call(self, inputs):
        outputs = tf.image.extract_patches(
            inputs,
            sizes=(1, self.segment_size, self.segment_size, 1),
            strides=(1, self.segment_size, self.segment_size, 1),
            rates=(1, 1, 1, 1),
            padding='SAME'
        )

        _, rows, cols, _ = tf.unstack(tf.shape(outputs))
        outputs = tf.reshape(outputs, shape=(-1, rows * cols, self.segment_size, self.segment_size, 3))

        return outputs
