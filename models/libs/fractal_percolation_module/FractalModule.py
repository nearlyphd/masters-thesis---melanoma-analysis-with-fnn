import tensorflow as tf

from libs.assemble_image.AssembleImage import AssembleImage
from libs.colour_distance.Chebyshev import Chebyshev
from libs.colour_distance.Euclidean import Euclidean
from libs.colour_distance.Manhattan import Manhattan
from libs.fractal_percolation_module.FeatureChannel import FeatureChannel
from libs.segmentation.Segmentation import Segmentation


class FractalModule(tf.keras.layers.Layer):
    FRACTAL_FEATURE_NUMBER = 5

    def __init__(self, fractal_width, fractal_height):
        super(FractalModule, self).__init__()

        self.segments = [
            Segmentation(segment_size)
            for segment_size in range(3, self._max_scale(fractal_width, fractal_height), 2)
        ]

        self.chebyshev_feature_channel = FeatureChannel(distance=Chebyshev,
                                                        fractal_width=fractal_width,
                                                        fractal_height=fractal_height)
        self.euclidean_feature_channel = FeatureChannel(distance=Euclidean,
                                                        fractal_width=fractal_width,
                                                        fractal_height=fractal_height)
        self.manhattan_feature_channel = FeatureChannel(distance=Manhattan,
                                                        fractal_width=fractal_width,
                                                        fractal_height=fractal_height)

        self.assemble_image = AssembleImage()

    def call(self, inputs):
        segments = [segment(inputs) for segment in self.segments]

        chebyshev_feature_channel = self.chebyshev_feature_channel(segments)
        euclidean_feature_channel = self.euclidean_feature_channel(segments)
        manhattan_feature_channel = self.manhattan_feature_channel(segments)

        outputs = self.assemble_image([
            chebyshev_feature_channel,
            euclidean_feature_channel,
            manhattan_feature_channel
        ])

        return outputs

    def _max_scale(self, fractal_width, fractal_height):
        scales_number = fractal_width * fractal_height / self.FRACTAL_FEATURE_NUMBER
        max_scale = 2 * scales_number + 2
        return int(max_scale)
