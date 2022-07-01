import tensorflow as tf


class AssembleImage(tf.keras.layers.Layer):
    def __init__(self):
        super(AssembleImage, self).__init__()

    def call(self, inputs):
        outputs = tf.stack(inputs)
        outputs = tf.transpose(outputs, perm=(1, 2, 3, 0))

        return outputs
