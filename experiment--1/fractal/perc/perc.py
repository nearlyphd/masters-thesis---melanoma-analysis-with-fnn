import tensorflow as tf

from fractal.distance import Manhattan, Euclidean, Chebyshev
from fractal.metrics import Probability, Clusterize, FractalDimension, Lacunarity, PercolationQ, PercolationC, \
    PercolationM
from fractal.segmentation import Segment
from fractal.utils import AssembleImage


class ChebyshevFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(ChebyshevFeatures, self).__init__()

        self.distance = Chebyshev()
        self.probability = Probability()
        self.clusterize = Clusterize()

        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()

        self.assemble_channel = AssembleChannel()

    def call(self, inputs):
        distance = [self.distance(i) for i in inputs]

        probability = [self.probability(d) for d in distance]
        cluster = [self.clusterize(d) for d in distance]

        fractal_dimension = [self.fractal_dimension(p) for p in probability]
        lacunarity = [self.lacunarity(p) for p in probability]
        percolation_q = [self.percolation_q(d) for d in distance]
        percolation_c = [self.percolation_c(c) for c in cluster]
        percolation_m = [self.percolation_m(c) for c in cluster]

        features = self.assemble_channel([
            fractal_dimension,
            lacunarity,
            percolation_q,
            percolation_c,
            percolation_m
        ])

        return features


class EuclideanFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(EuclideanFeatures, self).__init__()

        self.distance = Euclidean()
        self.probability = Probability()
        self.clusterize = Clusterize()

        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()

        self.assemble_channel = AssembleChannel()

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]

        probability = [self.probability(d) for d in distances]
        cluster = [self.clusterize(d) for d in distances]

        fractal_dimension = [self.fractal_dimension(p) for p in probability]
        lacunarity = [self.lacunarity(p) for p in probability]
        percolation_q = [self.percolation_q(d) for d in distances]
        percolation_c = [self.percolation_c(c) for c in cluster]
        percolation_m = [self.percolation_m(c) for c in cluster]

        features = self.assemble_channel([
            fractal_dimension,
            lacunarity,
            percolation_q,
            percolation_c,
            percolation_m
        ])

        return features


class ManhattanFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(ManhattanFeatures, self).__init__()

        self.distance = Manhattan()
        self.probability = Probability()
        self.clusterize = Clusterize()

        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()

        self.assemble_channel = AssembleChannel()

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]

        probability = [self.probability(d) for d in distances]
        cluster = [self.clusterize(d) for d in distances]

        fractal_dimension = [self.fractal_dimension(p) for p in probability]
        lacunarity = [self.lacunarity(p) for p in probability]
        percolation_q = [self.percolation_q(d) for d in distances]
        percolation_c = [self.percolation_c(c) for c in cluster]
        percolation_m = [self.percolation_m(c) for c in cluster]

        features = self.assemble_channel([
            fractal_dimension,
            lacunarity,
            percolation_q,
            percolation_c,
            percolation_m
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


class FractalImage(tf.keras.layers.Layer):
    def __init__(self):
        super(FractalImage, self).__init__()

        self.segments = [Segment(segment_size) for segment_size in range(3, 41 + 1, 2)]

        self.chebyshev_features = ChebyshevFeatures()
        self.euclidean_features = EuclideanFeatures()
        self.manhattan_features = ManhattanFeatures()

        self.assemble_image = AssembleImage()

    def call(self, inputs):
        patchifies = [segment(inputs) for segment in self.segments]

        chebyshev_features = self.chebyshev_features(patchifies)
        euclidean_features = self.euclidean_features(patchifies)
        manhattan_features = self.manhattan_features(patchifies)

        outputs = self.assemble_image([
            chebyshev_features,
            euclidean_features,
            manhattan_features
        ])

        return outputs