import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from random import shuffle
import tensorflow_hub as hub
import tensorflow_addons as tfa
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


tf.get_logger().setLevel('ERROR')


class Patchify(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patchify, self).__init__()
        
        self.patch_size = patch_size
        
    def call(self, inputs):
        outputs = tf.image.extract_patches(
            inputs,
            sizes=(1, self.patch_size, self.patch_size, 1),
            strides=(1, self.patch_size, self.patch_size, 1),
            rates=(1, 1, 1, 1),
            padding='SAME'
        )
        
        _, rows, cols, _ = tf.unstack(tf.shape(outputs))
        outputs = tf.reshape(outputs, shape=(-1, rows * cols, self.patch_size, self.patch_size, 3))
        
        return outputs

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


class Euclidean(tf.keras.layers.Layer):
    def __init__(self):
        super(Euclidean, self).__init__()
        
    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size, channels = tf.unstack(tf.shape(inputs))
        outputs = tf.reshape(inputs, shape=(-1, patch_number, patch_size, channels))
        
        centers = tf.image.resize_with_crop_or_pad(outputs, 1, 1)

        outputs = tf.math.subtract(outputs, centers)
        outputs = tf.math.pow(outputs, 2)
        outputs = tf.math.reduce_sum(outputs, axis=3)
        outputs = tf.math.pow(outputs, 0.5)
        outputs = tf.math.less_equal(outputs, tf.cast(patch_size, dtype=tf.float32))
        outputs = tf.cast(outputs, dtype=tf.int32)
        outputs = tf.reshape(outputs, shape=(-1, patch_number, patch_size, patch_size))
        
        return outputs


class Manhattan(tf.keras.layers.Layer):
    def __init__(self):
        super(Manhattan, self).__init__()
        
    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size, channels = tf.unstack(tf.shape(inputs))
        outputs = tf.reshape(inputs, shape=(-1, patch_number, patch_size, channels))
        
        centers = tf.image.resize_with_crop_or_pad(outputs, 1, 1)

        outputs = tf.math.subtract(outputs, centers)
        outputs = tf.math.abs(outputs)
        outputs = tf.math.reduce_sum(outputs, axis=3)
        outputs = tf.math.less_equal(outputs, tf.cast(patch_size, dtype=tf.float32))
        outputs = tf.cast(outputs, dtype=tf.int32)
        outputs = tf.reshape(outputs, shape=(-1, patch_number, patch_size, patch_size))
            
        
        return outputs

class Probability(tf.keras.layers.Layer):
    def __init__(self):
        super(Probability, self).__init__()
        
    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size = tf.unstack(tf.shape(inputs))
        
        outputs = tf.math.reduce_sum(inputs, axis=(2, 3))
        outputs = tf.vectorized_map(lambda image: tf.math.bincount(image, minlength=patch_size ** 2 + 1), outputs)
        outputs = tf.math.divide(outputs, patch_number)        
        
        return outputs

class FractalDimension(tf.keras.layers.Layer):
    def __init__(self):
        super(FractalDimension, self).__init__()
        
    def call(self, inputs):
        batch_size, _len = tf.unstack(tf.shape(inputs))
        numbers = tf.reshape(
            tf.concat(
                [tf.constant([1], dtype=tf.float32), tf.range(1, _len, dtype=tf.float32)], 
                axis=0
            ), 
            shape=(1, -1)
        )
        
        outputs = tf.math.divide(inputs, numbers)
        outputs = tf.math.reduce_sum(outputs, axis=1)
        
        return outputs

class Lacunarity(tf.keras.layers.Layer):
    def __init__(self):
        super(Lacunarity, self).__init__()
        
    def call(self, inputs):
        batch_size, _len = tf.unstack(tf.shape(inputs))
        numbers = tf.reshape(
            tf.concat(
                [tf.constant([1], dtype=tf.float32), tf.range(1, _len, dtype=tf.float32)], 
                axis=0
            ), 
            shape=(1, -1)
        )
                
        mu_first_2 = tf.math.multiply(inputs, numbers)
        mu_first_2 = tf.math.reduce_sum(mu_first_2, axis=1)
        mu_first_2 = tf.math.pow(mu_first_2, 2)

        mu_second = tf.math.pow(numbers, 2)
        mu_second = tf.math.multiply(inputs, mu_second)
        mu_second = tf.math.reduce_sum(mu_second, axis=1)

        outputs = tf.math.divide(
            tf.math.subtract(mu_second, mu_first_2),
            mu_first_2
        )
        
        return outputs


class PercolationQ(tf.keras.layers.Layer):
    def __init__(self, threshold=0.59275):
        super(PercolationQ, self).__init__()
        
        self.threshold = threshold
        
    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size = tf.unstack(tf.shape(inputs))
        
        outputs = tf.math.reduce_sum(inputs, axis=(2, 3))
        outputs = tf.math.divide(outputs, patch_size ** 2)
        outputs = tf.math.greater_equal(outputs, self.threshold)
        outputs = tf.cast(outputs, dtype=tf.float32)
        outputs = tf.math.reduce_mean(outputs, axis=1)
        
        return outputs


class Clusterize(tf.keras.layers.Layer):
    def __init__(self):
        super(Clusterize, self).__init__()
        
    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size = tf.unstack(tf.shape(inputs))
        
        outputs = tf.reshape(inputs, shape=(-1, patch_size, patch_size))
        outputs = tfa.image.connected_components(outputs)
        outputs = tf.reshape(outputs, shape=(-1, patch_number, patch_size, patch_size))
        
        return outputs

class PercolationC(tf.keras.layers.Layer):
    def __init__(self):
        super(PercolationC, self).__init__()
        
    def call(self, inputs):
        outputs = tf.cast(inputs, dtype=tf.float32)
        outputs = tf.math.reduce_max(outputs, axis=(2, 3))
        outputs = tf.math.reduce_mean(outputs, axis=1)
        
        return outputs

class PercolationM(tf.keras.layers.Layer):
    def __init__(self):
        super(PercolationM, self).__init__()
        
    def call(self, inputs):
        batch_size, patch_number, patch_size, patch_size = tf.unstack(tf.shape(inputs))
        
        outputs = tf.reshape(inputs, shape=(-1, patch_number, patch_size ** 2))
        outputs = tf.map_fn(lambda image: tf.math.reduce_max(tf.math.bincount(image)), outputs)
        outputs = tf.cast(outputs, dtype=tf.float32)
        
        return outputs


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


class ChebyshevFeatures(tf.keras.layers.Layer):
    def __init__(self):
        super(ChebyshevFeatures, self).__init__()
        
        self.chebyshev = Chebyshev()
        self.probability = Probability()
        self.clusterize = Clusterize()
        
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()
        
        self.assemble_channel = AssembleChannel()
        
    def call(self, inputs):
        chebyshevs = [self.chebyshev(i) for i in inputs]
        
        probability = [self.probability(ch) for ch in chebyshevs]
        cluster = [self.clusterize(ch) for ch in chebyshevs]
        
        fractal_dimension = [self.fractal_dimension(ch) for ch in probability]
        lacunarity = [self.lacunarity(ch) for ch in probability]        
        percolation_q = [self.percolation_q(ch) for ch in chebyshevs]
        percolation_c = [self.percolation_c(ch) for ch in cluster]
        percolation_m = [self.percolation_m(ch) for ch in cluster]
        
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
        
        self.euclidean = Euclidean()
        self.probability = Probability()
        self.clusterize = Clusterize()
        
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()
        
        self.assemble_channel = AssembleChannel()
        
    def call(self, inputs):
        euclideans = [self.euclidean(i) for i in inputs]
        
        probability = [self.probability(eu) for eu in euclideans]
        cluster = [self.clusterize(eu) for eu in euclideans]
        
        fractal_dimension = [self.fractal_dimension(eu) for eu in probability]
        lacunarity = [self.lacunarity(eu) for eu in probability]        
        percolation_q = [self.percolation_q(eu) for eu in euclideans]
        percolation_c = [self.percolation_c(eu) for eu in cluster]
        percolation_m = [self.percolation_m(eu) for eu in cluster]
        
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
        
        self.manhattan = Manhattan()
        self.probability = Probability()
        self.clusterize = Clusterize()
        
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        self.percolation_q = PercolationQ()
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()
        
        self.assemble_channel = AssembleChannel()
        
    def call(self, inputs):
        manhattans = [self.manhattan(i) for i in inputs]
        
        probability = [self.probability(mh) for mh in manhattans]
        cluster = [self.clusterize(mh) for mh in manhattans]
        
        fractal_dimension = [self.fractal_dimension(mh) for mh in probability]
        lacunarity = [self.lacunarity(mh) for mh in probability]        
        percolation_q = [self.percolation_q(mh) for mh in manhattans]
        percolation_c = [self.percolation_c(mh) for mh in cluster]
        percolation_m = [self.percolation_m(mh) for mh in cluster]
        
        features = self.assemble_channel([
            fractal_dimension,
            lacunarity,
            percolation_q,
            percolation_c,
            percolation_m
        ])
        
        return features


class AssembleImage(tf.keras.layers.Layer):
    def __init__(self):
        super(AssembleImage, self).__init__()
        
    def call(self, inputs):
        outputs = tf.stack(inputs)
        outputs = tf.transpose(outputs, perm=(1, 2, 3, 0))
        
        return outputs


class FractalImage(tf.keras.layers.Layer):
    def __init__(self):
        super(FractalImage, self).__init__()
        
        self.patchifies = [Patchify(patch_size) for patch_size in range(3, 41 + 1, 2)]
        
        self.chebyshev_features = ChebyshevFeatures()
        self.euclidean_features = EuclideanFeatures()
        self.manhattan_features = ManhattanFeatures()
        
        self.assemble_image = AssembleImage()
        
    def call(self, inputs):
        patchifies = [patchify(inputs) for patchify in self.patchifies]
        
        chebyshev_features = self.chebyshev_features(patchifies)
        euclidean_features = self.euclidean_features(patchifies)
        manhattan_features = self.manhattan_features(patchifies)
        
        outputs = self.assemble_image([
            chebyshev_features,
            euclidean_features,
            manhattan_features
        ])
        
        return outputs



class FractalNeuralNetwork(tf.keras.Model):
    TARGET_WIDTH, TARGET_HEIGHT = 299, 299
    
    def __init__(self, class_number):
        super(FractalNeuralNetwork, self).__init__()
        
        self.fractal_image = FractalImage()
        self.resize = tf.keras.layers.Resizing(
            width=self.TARGET_WIDTH,
            height=self.TARGET_HEIGHT
        )
        self.rescale = tf.keras.layers.Rescaling(scale=1./255)
        
        self.original_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(299, 299, 3),
            pooling='avg'
        )
        for layer in self.original_model.layers:
            if layer.name == 'conv_7b':
                layer.trainable = True
            else:
                layer.trainable = False
                
        self.fractal_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(299, 299, 3),
            pooling='avg'
        )
        for layer in self.fractal_model.layers:
            if layer.name == 'conv_7b':
                layer.trainable = True
            else:
                layer.trainable = False
                
        self.combine = tf.keras.layers.Concatenate()
        self.score = tf.keras.layers.Dense(class_number, activation='softmax')
        
    def call(self, inputs):
        fractal_outputs = self.fractal_image(inputs)
        fractal_outputs = self.resize(fractal_outputs)
        fractal_outputs = self.rescale(fractal_outputs)
        fractal_outputs = self.fractal_model(fractal_outputs)
        
        original_outputs = self.rescale(inputs)
        original_outputs = self.original_model(original_outputs)
        
        outputs = self.combine([fractal_outputs, original_outputs])
        outputs = self.score(outputs)
        
        return outputs



