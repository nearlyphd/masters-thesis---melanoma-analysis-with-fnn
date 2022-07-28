import tensorflow as tf

from libs.analytics import Probability
from libs.features import FractalDimension, Lacunarity
from libs.fractal_percolation_module import AssembleChannel


class FeatureChannel(tf.keras.layers.Layer):
    def __init__(self, distance, fractal_width, fractal_height):
        super(FeatureChannel, self).__init__()

        self.distance = distance()
        self.probability = Probability()

        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()

        self.assemble_channel = AssembleChannel(fractal_width, fractal_height)

    def call(self, inputs):
        distances = [self.distance(i) for i in inputs]

        probabilities = [self.probability(d) for d in distances]

        fractal_dimensions = [self.fractal_dimension(p) for p in probabilities]
        lacunarities = [self.lacunarity(p) for p in probabilities]

        features = self.assemble_channel([
            fractal_dimensions,
            lacunarities,
        ])

        return features
