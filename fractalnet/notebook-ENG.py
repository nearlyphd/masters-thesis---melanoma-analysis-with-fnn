#!/usr/bin/env python
# coding: utf-8

# # Melanoma analysis with fractal neural networks

# This notebook shows how good is [Fractal neural network](#Fractal-neural-network) for [melanoma](#Melanoma) analysis.

# In[1]:


import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Check if a GPU is available.

# In[2]:


tf.config.list_physical_devices('GPU')


# # Melanoma

# __Melanoma__, also redundantly known as __malignant melanoma__, is a type of skin cancer that develops from the pigment-producing cells known as melanocytes. Melanomas typically occur in the skin, but may rarely occur in the mouth, intestines, or eye (uveal melanoma). In women, they most commonly occur on the legs, while in men, they most commonly occur on the back. About 25% of melanomas develop from moles. Changes in a mole that can indicate melanoma include an increase in size, irregular edges, change in color, itchiness, or skin breakdown.

# ![melanoma image](../assets/melanoma.jpg)

# <div style="text-align: center; font-weight: bold">Pic.1. A melanoma of approximately 2.5 cm (1 in) by 1.5 cm (0.6 in)</div>

# The primary cause of melanoma is ultraviolet light (UV) exposure in those with low levels of the skin pigment melanin. The UV light may be from the sun or other sources, such as tanning devices. Those with many moles, a history of affected family members, and poor immune function are at greater risk. A number of rare genetic conditions, such as xeroderma pigmentosum, also increase the risk. Diagnosis is by biopsy and analysis of any skin lesion that has signs of being potentially cancerous.

# Melanoma is the most dangerous type of skin cancer. Globally, in 2012, it newly occurred in 232,000 people. In 2015, 3.1 million people had active disease, which resulted in 59,800 deaths. Australia and New Zealand have the highest rates of melanoma in the world. High rates also occur in Northern Europe and North America, while it is less common in Asia, Africa, and Latin America. In the United States, melanoma occurs about 1.6 times more often in men than women. Melanoma has become more common since the 1960s in areas mostly populated by people of European descent.

# # Fractal neural network

# We propose an ensemble model based on handcrafted fractal features and deep learning that consists of combining the classification of two CNNs by applying the sum rule. We apply feature extraction to obtain 300 fractal features from different
# dermoscopy datasets. These features are reshaped into a 10 × 10 × 3 matrix to compose an artificial image that
# is given as input to the first CNN. The second CNN model receives as input the correspondent original image.

# ![CNN image](../assets/fnn.png)

# <div style="text-align: center; font-weight: bold">Pic.2. Overview of the proposed FNN model.</div>

# If you want to learn more about fractal neural networks, read [here](https://www.sciencedirect.com/science/article/abs/pii/S0957417420308563).

# ## Developing the layer that divides images into patches.

# According to the acticle:
# > One of the approaches available in the literature for multiscale
# analysis is the gliding-box algorithm (Ivanovici & Richard, 2011). The
# main advantage of this approach is that it can be applied on datasets
# containing images with different resolutions since the output features
# are given in relation to the scale instead of being absolute values.
# This algorithm consists in placing a box 𝛽𝑖
# sized 𝐿 × 𝐿 on the left
# superior corner of the image, wherein 𝐿 is given in pixels. This box
# glides through the image, one column and then one row at a time. After
# reaching the end of the image, the box is repositioned at the starting
# point and the value of 𝐿 is increased by 2.

# The gliding-box method will not be used since it consumes too much RAM. We'll employ a box-counting approach, which basically means we'll partition the images into non-overlapping chunks.

# In[3]:


class BoxCountingPatch(tf.keras.layers.Layer):
    def __init__(self, box_size):
        super(BoxCountingPatch, self).__init__()
        
        self.box_size = box_size
    
    def call(self, inputs):
        patched_inputs = tf.image.extract_patches(
            inputs,
            sizes=(1, self.box_size, self.box_size, 1),
            strides=(1, self.box_size, self.box_size, 1),
            rates=(1, 1, 1, 1),
            padding='SAME'
        )
        
        patched_inputs_shape = tf.shape(patched_inputs)
        _, rows, cols, _ = tf.unstack(patched_inputs_shape)
        
        patched_inputs = tf.reshape(patched_inputs, shape=(-1, rows * cols, self.box_size, self.box_size, 3))
        
        return patched_inputs


# ## Developing the layer that creates an array of binary values from image patches using the Chebyshev colour distance function applied to the patch centre and each pixel.

# According to the article:
# > For each time the box 𝛽<sub>𝑖</sub> is moved, a multidimensional analysis of colour similarity is performed for every pixel inside it. This is done by assigning the centre pixel to a vector 𝑓<sub>𝑐</sub> = 𝑟<sub>𝑐</sub>, 𝑔<sub>𝑐</sub>, 𝑏<sub>𝑐</sub>, where 𝑟<sub>𝑐</sub>, 𝑔<sub>𝑐</sub> and 𝑏<sub>𝑐</sub> correspond to the colour intensities for each of the RGB colour channels of given pixel. The other pixels in the box are assigned to a vector 𝑓<sub>𝑖</sub> = 𝑟<sub>𝑖</sub>, 𝑔<sub>𝑖</sub>, 𝑏<sub>𝑖</sub> and compared to the centre pixel by calculating a colour distance 𝛥. On the proposed approach, the Chebyshev (𝛥<sub>ℎ</sub>) ...

# The following equation is used to compute the Chebyshev distance.

# $$
# \Delta_{h} = max(|f_{i}(k_{i}) - f_{c}(k_{c})|), k \in r, g, b. 
# $$ 

# In[4]:


class ChebyshevBinaryPatch(tf.keras.layers.Layer):
    def __init__(self):
        super(ChebyshevBinaryPatch, self).__init__()

    def call(self, inputs):
        def helper(_input_):
            _input__shape = tf.shape(_input_)
            _, number_of_patches, box_size, _, channels = tf.unstack(_input__shape)
            _input_ = tf.reshape(_input_, shape=(-1, box_size, box_size, channels))
            
            centers = tf.image.resize_with_crop_or_pad(_input_, 1, 1)
            
            binary = tf.math.subtract(_input_, centers)
            binary = tf.math.abs(binary)
            binary = tf.math.reduce_max(binary, axis=3)
            binary = tf.math.less_equal(binary, tf.cast(box_size, dtype=tf.float32))
            binary = tf.cast(binary, dtype=tf.int32)
            binary = tf.reshape(binary, shape=(-1, number_of_patches, box_size, box_size))
            
            return binary
        
        return [helper(_input_) for _input_ in inputs]


# ## Developing the layer that creates an array of binary values from image patches using the Euclidean colour distance function applied to the patch centre and each pixel.

# According to the article:
# > For each time the box 𝛽<sub>𝑖</sub> is moved, a multidimensional analysis of colour similarity is performed for every pixel inside it. This is done by assigning the centre pixel to a vector 𝑓<sub>𝑐</sub> = 𝑟<sub>𝑐</sub>, 𝑔<sub>𝑐</sub>, 𝑏<sub>𝑐</sub>, where 𝑟<sub>𝑐</sub>, 𝑔<sub>𝑐</sub> and 𝑏<sub>𝑐</sub> correspond to the colour intensities for each of the RGB colour channels of given pixel. The other pixels in the box are assigned to a vector 𝑓<sub>𝑖</sub> = 𝑟<sub>𝑖</sub>, 𝑔<sub>𝑖</sub>, 𝑏<sub>𝑖</sub> and compared to the centre pixel by calculating a colour distance 𝛥. On the proposed approach, ... Euclidean (𝛥<sub>e</sub>) ..

# $$
# \Delta_{e} = \sqrt{\sum_{k} (f_{i}(k_{i}) - f_{c}(k_{c}))^2}, k \in r, g, b
# $$

# In[5]:


class EuclideanBinaryPatch(tf.keras.layers.Layer):
    def __init__(self):
        super(EuclideanBinaryPatch, self).__init__()

    def call(self, inputs):
        def helper(_input_):
            _input__shape = tf.shape(_input_)
            _, number_of_patches, box_size, _, channels = tf.unstack(_input__shape)
            _input_ = tf.reshape(_input_, shape=(-1, box_size, box_size, channels))
            
            centers = tf.image.resize_with_crop_or_pad(_input_, 1, 1)
            
            binary = tf.math.subtract(_input_, centers)
            binary = tf.math.pow(_input_, 2)
            binary = tf.math.reduce_sum(binary, axis=3)
            binary = tf.math.pow(binary, 0.5)
            binary = tf.math.less_equal(binary, tf.cast(box_size, dtype=tf.float32))
            binary = tf.cast(binary, dtype=tf.int32)
            binary = tf.reshape(binary, shape=(-1, number_of_patches, box_size, box_size))
            
            return binary
        
        return [helper(_input_) for _input_ in inputs]


# ## Developing the layer that creates an array of binary values from image patches using the Manhattan colour distance function applied to the patch centre and each pixel.

# According to the article:
# > For each time the box 𝛽<sub>𝑖</sub> is moved, a multidimensional analysis of colour similarity is performed for every pixel inside it. This is done by assigning the centre pixel to a vector 𝑓<sub>𝑐</sub> = 𝑟<sub>𝑐</sub>, 𝑔<sub>𝑐</sub>, 𝑏<sub>𝑐</sub>, where 𝑟<sub>𝑐</sub>, 𝑔<sub>𝑐</sub> and 𝑏<sub>𝑐</sub> correspond to the colour intensities for each of the RGB colour channels of given pixel. The other pixels in the box are assigned to a vector 𝑓<sub>𝑖</sub> = 𝑟<sub>𝑖</sub>, 𝑔<sub>𝑖</sub>, 𝑏<sub>𝑖</sub> and compared to the centre pixel by calculating a colour distance 𝛥. On the proposed approach, ... Manhattan (𝛥<sub>m</sub>) .. 

# $$
# \Delta_{m} = \sum_{k} |f_{i}(k_{i}) - f_{c}(k_{c})|, k \in r, g, b
# $$

# In[6]:


class ManhattanBinaryPatch(tf.keras.layers.Layer):
    def __init__(self):
        super(ManhattanBinaryPatch, self).__init__()

    def call(self, inputs):
        def helper(_input_):
            _input__shape = tf.shape(_input_)
            _, number_of_patches, box_size, _, channels = tf.unstack(_input__shape)
            _input_ = tf.reshape(_input_, shape=(-1, box_size, box_size, channels))
            
            centers = tf.image.resize_with_crop_or_pad(_input_, 1, 1)
            
            binary = tf.math.subtract(_input_, centers)
            binary = tf.math.abs(binary)
            binary = tf.math.reduce_sum(binary, axis=3)
            binary = tf.math.less_equal(binary, tf.cast(box_size, dtype=tf.float32))
            binary = tf.cast(binary, dtype=tf.int32)
            binary = tf.reshape(binary, shape=(-1, number_of_patches, box_size, box_size))
            
            return binary
        
        return [helper(_input_) for _input_ in inputs]


# ## Developing the layer that calculates probability matrices

# According to the article:
# > After performing this conversion for every box of every given 𝐿 scale, a structure known as probability matrix is generated. Each element of the matrix corresponds to the probability 𝑃 that 𝑚 pixels on a scale 𝐿 are labelled as 1 on each box. ... The matrix is normalized in a way that the sum of the elements in a column is equal to 1, as showed here:

# $$
# \sum_{m=1}^{L^2} P(m, L) = 1, \forall L
# $$

# In[7]:


class ProbabilityMatrix(tf.keras.layers.Layer):
    def __init__(self):
        super(ProbabilityMatrix, self).__init__()

    def call(self, inputs):
        color_distance_inputs = []
        
        for color_distance_input in inputs:
            box_outputs = []
            
            for box_input in color_distance_input:
                number_of_ones_for_every_patch = tf.map_fn(
                    lambda batch: tf.map_fn(
                        lambda patch: tf.math.reduce_sum(patch),
                        batch
                    ),
                    box_input
                )
                
                box_input_shape = tf.shape(box_input)
                _, number_of_patches, box_size, _ = tf.unstack(box_input_shape)
                
                probabilities = tf.math.bincount(
                    number_of_ones_for_every_patch,
                    minlength=1, 
                    maxlength=box_size ** 2, 
                    axis=-1
                )
                probabilities = tf.math.divide(probabilities, number_of_patches)
                
                probabilities = tf.map_fn(
                    lambda x: x[0] / x[1], 
                    elems=(probabilities, tf.math.reduce_sum(probabilities, axis=1)),
                    fn_output_signature=tf.float64
                )
                
                
                box_outputs.append(probabilities)
                
            color_distance_inputs.append(box_outputs)
            
        return color_distance_inputs
                
            


# ## Developing the layer that calculates fractal dimensions

# According to the article:
# > FD is the most common technique to evaluate the fractal properties of an image. This is a measure for evaluating the irregularity and the complexity of a fractal. To obtain local FD features from the probability
# matrix, for each value of 𝐿, the FD denominated 𝐷(𝐿) is calculated according to

# $$
# D(L) = \sum_{m=1}^{L^2} \frac{P(m, L)}{m}
# $$

# In[8]:


class FractalDimension(tf.keras.layers.Layer):
    def __init__(self):
        super(FractalDimension, self).__init__()

    def call(self, inputs):
        color_distance_outputs = []
        
        for color_distance_input in inputs:
            box_outputs = []
            
            for box_input in color_distance_input:
                box_input_shape = tf.shape(box_input)
                _, box_input_len = tf.unstack(box_input_shape)
                
                probability_numbers = tf.range(1, box_input_len + 1, dtype=tf.float32)

                
                fractal_dimension = tf.map_fn(
                    lambda probability_input: tf.map_fn(
                        lambda x: x[0] / x[1],
                        elems=(probability_input, probability_numbers),
                        fn_output_signature=tf.float32
                    ),
                    box_input,
                    fn_output_signature=tf.float32
                )
                fractal_dimension = tf.math.reduce_sum(fractal_dimension, axis=1)
                
                box_outputs.append(fractal_dimension)
                
            color_distance_outputs.append(box_outputs)
            
        return color_distance_outputs


# ## Developing the layer that calculates lacunarity

# According to the article:
# > LAC is a measure complementary to FD and allows to evaluate how the space of a fractal is filled (Ivanovici & Richard, 2009). From the probability matrix, first and second-order moments are calculated with

# $$
# \mu(L) = \sum_{m=1}^{L^2} mP(m, L)
# $$

# $$
# \mu^2(L) = \sum_{m=1}^{L^2} m^{2}P(m, L)
# $$

# > The LAC value for a scale 𝐿 is given by 𝛬(𝐿), which is obtained according to

# $$
# \Lambda(L) = \frac{\mu^{2}(L) - (\mu(L))^{2}}{(\mu(L))^{2}}
# $$

# In[9]:


class Lacunarity(tf.keras.layers.Layer):
    def __init__(self):
        super(Lacunarity, self).__init__()
        
    def call(self, inputs):
        color_distance_outputs = []
        
        for color_distance_input in inputs:
            box_outputs = []
            
            for box_input in color_distance_input:
                probability_numbers = tf.range(1, len(box_input) + 1, dtype=tf.float32)
                
                mu_first_2 = tf.map_fn(
                    lambda x: x[0] * x[1], 
                    elems=(box_input, probability_numbers),
                    fn_output_signature=tf.float32
                )
                mu_first_2 = tf.math.reduce_sum(mu_first_2, axis=1)
                mu_first_2 = tf.math.pow(mu_first_2, 2)
                
                mu_second = tf.math.pow(probability_numbers, 2)
                mu_second = tf.map_fn(
                    lambda x: x[0] * x[1], 
                    elems=(box_input, mu_second),
                    fn_output_signature=tf.float32
                )
                mu_second = tf.math.reduce_sum(mu_second, axis=1)
                
                lacunarity = tf.math.divide(
                    tf.math.subtract(mu_second, mu_first_2),
                    mu_first_2
                )
                
                box_outputs.append(lacunarity)
                
            color_distance_outputs.append(box_outputs)
            
        return color_distance_outputs


# ## Developing the layer that calculates percolation C - the average number of clusters per box on a scale L

# According to the article:
# > Let 𝑐<sub>𝑖</sub> be the number of clusters on a box 𝛽<sub>𝑖</sub>, the feature 𝐶(𝐿) that represents the average number of clusters per box on a scale 𝐿 is given by

# $$
# C(L) = \frac{\sum_{i=1}^{T(L)} c_{i}}{T(L)}
# $$

# In[10]:


class PercolationC(tf.keras.layers.Layer):
    def __init__(self):
        super(PercolationC, self).__init__()

    def call(self, inputs):
        color_distance_outputs = []
        
        for color_distance_input in inputs:
            box_outputs = []
            
            for box_input in color_distance_input:
                percolation_c = tf.math.reduce_mean(
                    tf.map_fn(
                        lambda batch: tf.map_fn(
                            lambda patch: tf.math.reduce_max(tfa.image.connected_components(patch)), 
                            batch
                        ),
                        box_input
                    ),
                    axis=1
                )
                percolation_c = tf.cast(percolation_c, dtype=tf.float32)
                box_outputs.append(percolation_c)
                
            color_distance_outputs.append(box_outputs)
            
        return color_distance_outputs


# ## Developing the layer that calculates percolation M - the average coverage area of the largest cluster on a scale L

# According to the article:
# >Another feature that can be obtained is the average coverage area of the largest cluster in a box and is given by 𝑀(𝐿). Let 𝑚<sub>𝑖</sub> be the size in pixels of the largest cluster of the box 𝛽<sub>𝑖</sub>. The feature 𝑀(𝐿) is givenaccording to

# $$
# M(L) = \frac{\sum_{i=1}^{T(L)} \frac{m_{i}}{L^2}}{T(L)}
# $$

# In[11]:


class PercolationM(tf.keras.layers.Layer):
    def __init__(self):
        super(PercolationM, self).__init__()

    def call(self, inputs):
        color_distance_outputs = []
        
        for color_distance_input in inputs:
            box_outputs = []
            
            for box_input in color_distance_input:
                percolation_m = tf.math.reduce_mean(
                    tf.map_fn(
                        lambda batch: tf.map_fn(
                            lambda patch: self.most_common(tf.reshape(tfa.image.connected_components(patch), shape=(-1,))), 
                            batch
                        ),
                        box_input
                    ),
                    axis=1
                )
                percolation_m = tf.cast(percolation_m, dtype=tf.float32)
                box_outputs.append(percolation_m)
                
            color_distance_outputs.append(box_outputs)
            
        return color_distance_outputs
    
    def most_common(self, array):
        _, _, counts = tf.unique_with_counts(array)
        return tf.math.reduce_max(counts)


# ## Developing the layer that calculates percolation Q - the average occurrence of percolation on a scale L

# According to the article:
# > We can also verify whether a box 𝛽<sub>𝑖</sub> is percolating. This can be achieved due to a property that states a percolation threshold for different types of structures. In squared matrices (digital images), this threshold has the value of 𝑝 = 0.59275, which means that if the ratio between pixels labelled as 1 and pixels labelled as 0 is greater or equal than 𝑝, the matrix is considered as percolating. Let 𝛺<sub>𝑖</sub> be the number of pixels labelled as 1 in a box 𝛽<sub>𝑖</sub> with size 𝐿 × 𝐿, we determine whether such box is percolating according to

# $$
# q_{i} = 
# \begin{cases}
# 1, & \frac{\Omega_{i}}{L^2} \ge 0.59275 \\
# 0, & \frac{\Omega_{i}}{L^2} < 0.59275
# \end{cases}
# $$

# > This results in a binary value for 𝑞<sub>𝑖</sub>, wherein 1 indicates that thebox is percolating. The feature 𝑄(𝐿) regards the average occurrence of percolation on a scale 𝐿 and can be obtained by

# $$
# Q(L) = \frac{\sum_{i=1}^{T(L)} q_{i}}{T(L)}
# $$

# In[12]:


class PercolationQ(tf.keras.layers.Layer):
    def __init__(self, threshold=0.59275):
        super(PercolationQ, self).__init__()
        
        self.threshold = threshold

    def call(self, inputs):
        color_distance_outputs = []
        
        for color_distance_input in inputs:
            box_outputs = []
            
            for box_input in color_distance_input:
                number_of_ones_for_every_patch = tf.map_fn(
                    lambda batch: tf.map_fn(
                        lambda patch: tf.math.reduce_sum(patch),
                        batch
                    ),
                    box_input
                )
                
                box_input_shape = tf.shape(box_input)
                _, number_of_patches, box_size, _ = tf.unstack(box_input_shape)
                
                percolation_q = tf.math.divide(number_of_ones_for_every_patch, box_size ** 2)
                percolation_q = tf.math.greater_equal(percolation_q, self.threshold)
                percolation_q = tf.cast(percolation_q, dtype=tf.float32)
                percolation_q = tf.math.reduce_mean(percolation_q, axis=1)
                
                box_outputs.append(percolation_q)
                
            color_distance_outputs.append(box_outputs)
            
        return color_distance_outputs
    
    def most_common(self, array):
        _, _, counts = tf.unique_with_counts(array)
        return tf.math.reduce_max(counts)


# ## Developing the layer that assembles fractal features into images

# According to the article:
# > To serve as input for the incoming CNN classification, the feature vectors generated on the previous layers of the network must be converted into feature matrices. To do so, the 100 features obtained by each distance 𝛥 are rearranged as a 10 × 10 matrix. The matrices generated by 𝛥<sub>ℎ</sub>, 𝛥<sub>𝑒</sub> and 𝛥<sub>𝑚</sub> correspond to the R, G and B colour channels, respectively. ... Since each of the functions 𝐶(𝐿), 𝑄(𝐿), 𝑀(𝐿), 𝛬(𝐿) and 𝐷(𝐿), obtained from a specific 𝛥, generate 20 features, each function is fit exactly into 2 columns of the matrix.
# 
# >Since each of the functions 𝐶(𝐿), 𝑄(𝐿), 𝑀(𝐿), 𝛬(𝐿) and 𝐷(𝐿), obtained from a specific 𝛥, generate 20 features, each function is fit exactly into 2 columns of the matrix.

# In[13]:


class AssembleFractalImage(tf.keras.layers.Layer):
    def __init__(self):
        super(AssembleFractalImage, self).__init__()

    def call(self, inputs):
        output = tf.convert_to_tensor(inputs)
        output = tf.transpose(output, perm=(3, 1, 0, 2))
        output = tf.reshape(output, shape=(-1, 5, 5, 3))
        return output


# ## Assembling the layers into fractal neural network

# In[14]:


class FractalNeuralNetwork(tf.keras.Model):
    def __init__(self, class_number, input_shape):
        super(FractalNeuralNetwork, self).__init__()
        
        self.input_shape_ = input_shape
        
        self.box_counting_patches = [BoxCountingPatch(box_size) for box_size in range(15, 23 + 1, 2)]
        
        self.chebyshev = ChebyshevBinaryPatch()
        self.euclidean = EuclideanBinaryPatch()
        self.manhattan = ManhattanBinaryPatch()
        
        self.percolation_c = PercolationC()
        self.percolation_m = PercolationM()
        self.percolation_q = PercolationQ()
        
        self.probability = ProbabilityMatrix()
        self.fractal_dimension = FractalDimension()
        self.lacunarity = Lacunarity()
        
        self.assemble = AssembleFractalImage()
        self.resize = tf.keras.layers.Resizing(width=self.input_shape_[1], height=self.input_shape_[2])
        self.rescale_original = tf.keras.layers.Rescaling(scale=1./255)
        self.rescale_fractal = tf.keras.layers.Lambda(lambda x: tf.math.divide(x, tf.math.reduce_max(x)))
        
        self.mobilenet_v2 = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/5")
        self.combine = tf.keras.layers.Add()
        self.score = tf.keras.layers.Dense(class_number, activation='softmax')
        
    def call(self, inputs):
        inputs = tf.ensure_shape(inputs, self.input_shape_)
        box_counting_patches = [box_counting_patch(inputs) for box_counting_patch in self.box_counting_patches]

        chebyshev = self.chebyshev(inputs=box_counting_patches)
        euclidean = self.euclidean(inputs=box_counting_patches)
        manhattan = self.manhattan(inputs=box_counting_patches)

        percolation_c = self.percolation_c(inputs=[chebyshev, euclidean, manhattan])
        percolation_m = self.percolation_m(inputs=[chebyshev, euclidean, manhattan])
        percolation_q = self.percolation_q(inputs=[chebyshev, euclidean, manhattan])

        probability = self.probability(inputs=[chebyshev, euclidean, manhattan])

        fractal_dimension = self.fractal_dimension(inputs=probability)
        lacunarity = self.lacunarity(inputs=probability)

        fractal_output = self.assemble(
            inputs=[
                fractal_dimension, 
                lacunarity, 
                percolation_c, 
                percolation_m, 
                percolation_q
            ]
        )
        fractal_output = self.resize(fractal_output)
        fractal_output = self.rescale_fractal(fractal_output)
        fractal_output = self.mobilenet_v2(fractal_output)

        combined_output = fractal_output
        output = self.score(combined_output)

        return output


# # Data loading

# In[15]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.2, 1.5),
    validation_split=0.2,
)


training_set = generator.flow_from_directory(
    './small-data',
    target_size=(96, 96),
    batch_size=32, 
    class_mode='categorical', 
    subset='training'
)
validation_set = generator.flow_from_directory(
    './small-data',
    target_size=(96, 96),
    batch_size=32, 
    class_mode='categorical', 
    subset='validation'
)


CLASS_NUMBER = len(training_set.class_indices)

model = FractalNeuralNetwork(
    class_number=CLASS_NUMBER,
    input_shape=(32, 96, 96, 3)
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    training_set,
    epochs=1
)
