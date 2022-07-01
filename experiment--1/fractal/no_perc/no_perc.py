import tensorflow as tf

from fractal.distance import Manhattan, Euclidean, Chebyshev
from fractal.metrics import Probability, FractalDimension, Lacunarity
from fractal.segmentation import Segment
from fractal.utils import AssembleImage


class ChebyshevFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(ChebyshevFeatures, self).__init__()

        self.distance = Chebyshev()
        self.probability = Probability()
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.assemble_channel = AssembleChannel()

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]
        probability = [self.probability(d) for d in distances]
        fractal_dimension = [self.fractal_dimension(p) for p in probability]
        lacunarity = [self.lacunarity(p) for p in probability]

        features = self.assemble_channel([
            fractal_dimension,
            lacunarity
        ])

        return features


class EuclideanFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(EuclideanFeatures, self).__init__()

        self.distance = Euclidean()
        self.probability = Probability()
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.assemble_channel = AssembleChannel()

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]
        probability = [self.probability(d) for d in distances]
        fractal_dimension = [self.fractal_dimension(p) for p in probability]
        lacunarity = [self.lacunarity(p) for p in probability]

        features = self.assemble_channel([
            fractal_dimension,
            lacunarity
        ])

        return features


class ManhattanFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(ManhattanFeatures, self).__init__()

        self.distance = Manhattan()
        self.probability = Probability()
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.assemble_channel = AssembleChannel()

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]
        probability = [self.probability(d) for d in distances]
        fractal_dimension = [self.fractal_dimension(p) for p in probability]
        lacunarity = [self.lacunarity(p) for p in probability]

        features = self.assemble_channel([
            fractal_dimension,
            lacunarity
        ])

        return features


class AssembleChannel(tf.keras.layers.Layer):
    def __init__(self):
        super(AssembleChannel, self).__init__()

    def call(self, inputs):
        fractal_dimension = tf.convert_to_tensor(inputs[0])
        fractal_dimension = tf.transpose(fractal_dimension, perm=(1, 0))

        lacunarity = tf.convert_to_tensor(inputs[1])
        lacunarity = tf.transpose(lacunarity, perm=(1, 0))

        outputs = tf.concat([
            lacunarity,
            fractal_dimension
        ], axis=1)
        outputs = tf.reshape(outputs, shape=(-1, 10, 10))

        return outputs


class FractalImage(tf.keras.layers.Layer):
    def __init__(self):
        super(FractalImage, self).__init__()

        self.segments = [Segment(segment_size) for segment_size in range(3, 101 + 1, 2)]

        self.chebyshev_features = ChebyshevFeatures()
        self.euclidean_features = EuclideanFeatures()
        self.manhattan_features = ManhattanFeatures()

        self.assemble_image = AssembleImage()

    def call(self, inputs):
        segments = [segment(inputs) for segment in self.segments]

        chebyshev_features = self.chebyshev_features(segments)
        euclidean_features = self.euclidean_features(segments)
        manhattan_features = self.manhattan_features(segments)

        outputs = self.assemble_image([
            chebyshev_features,
            euclidean_features,
            manhattan_features
        ])

        return outputs
