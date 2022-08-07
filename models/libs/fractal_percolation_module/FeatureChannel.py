import tensorflow as tf

from libs.analytics import Probability, Clusterize
from libs.features import FractalDimension, Lacunarity, PercolationQ, PercolationC, PercolationM
from libs.fractal_percolation_module import AssembleChannel


class FeatureChannel(tf.keras.layers.Layer):
    def __init__(self, distance, fractal_width, fractal_height):
        super(FeatureChannel, self).__init__()

        self.distance = distance()
        self.probability = Probability()
        self.clusterize = Clusterize()

        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()

        self.assemble_channel = AssembleChannel(fractal_width, fractal_height)

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]

        probabilities = [self.probability(d) for d in distances]
        clusters = [self.clusterize(d) for d in distances]

        fractal_dimensions = [self.fractal_dimension(p) for p in probabilities]
        lacunarities = [self.lacunarity(p) for p in probabilities]
        percolations_q = [self.percolation_q(d) for d in distances]
        percolations_c = [self.percolation_c(c) for c in clusters]
        percolations_m = [self.percolation_m(c) for c in clusters]

        features = self.assemble_channel([
            fractal_dimensions,
            lacunarities,
            percolations_q,
            percolations_c,
            percolations_m
        ])

        return features
