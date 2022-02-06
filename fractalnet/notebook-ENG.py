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
# histological datasets. These features are reshaped into a 10 × 10 × 3 matrix to compose an artificial image that
# is given as input to the first CNN. The second CNN model receives as input the correspondent original image.

# ![CNN image](../assets/fnn.png)

# <div style="text-align: center; font-weight: bold">Pic.2. Overview of the proposed FNN model.</div>

# If you want to learn more about fractal neural networks, read [here](https://www.sciencedirect.com/science/article/abs/pii/S0957417420308563).

# # Data loading

# In[3]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.2, 1.5),
    validation_split=0.2,
)


training_set = generator.flow_from_directory(
    '/small-data', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training'
)
validation_set = generator.flow_from_directory(
    '/small-data', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical', 
    subset='validation'
)


# In[4]:


CLASS_NUMBER = len(training_set.class_indices)


# ### Data source

# As a data source, we use the ISIC Archive.

# The ISIC Archive is an open source platform with publicly available images of skin lesions under Creative Commons licenses. The images are associated with ground-truth diagnoses and other clinical metadata. Images can be queried using faceted search and downloaded individually or in batches. The initial focus of the archive has been on dermoscopy images of individual skin lesions, as these images are inherently standardized by the use of a specialized acquisition device and devoid of many of the privacy challenges associated with clinical images. To date, the images have been provided by specialized melanoma centers from around the world. The archive is designed to accept contributions from new sources under the Terms of Use and welcomes new contributors. There are ongoing efforts to supplement the dermoscopy images in the archive with close-up clinical images and a broader representation of skin types. The images in the Archive are used to support educational efforts through linkage with Dermoscopedia and are used for Grand Challenges and Live Challenges to engage the computer science community for the development of diagnostic AI.

# For more information, go to [ISIC Archive web site](https://www.isic-archive.com/)

# # Define fractal neural network

# ## Colour distance functions

# We define functions, which calculate colour distance between the centres of images to every pixel of the images.

# ### Chessboard distance

# In mathematics, __Chebyshev distance__ (or __Tchebychev distance__), __maximum metric__, or __L∞ metric__ is a metric defined on a vector space where the distance between two vectors is the greatest of their differences along any coordinate dimension. It is named after Pafnuty Chebyshev.

# It is also known as __chessboard distance__, since in the game of chess the minimum number of moves needed by a king to go from one square on a chessboard to another equals the Chebyshev distance between the centers of the squares, if the squares have side length one, as represented in 2-D spatial coordinates with axes aligned to the edges of the board. For example, the Chebyshev distance between f6 and e2 equals 4.

# The Chebyshev distance between two vectors or points x and y, with standard coordinates x<sub>i</sub> and y<sub>i</sub>, respectively, is
# ![chessboard distance](../assets/chessboard_distancesvg.svg)

# In[5]:


def chessboard_distance(inputs, kernel_size):
    centers = tf.image.resize_with_crop_or_pad(inputs, 1, 1)
    return tf.cast(
        tf.math.less_equal(
            tf.math.reduce_max(
                tf.math.abs(tf.math.subtract(inputs, centers)), 
                axis=3), 
            kernel_size), 
        dtype=tf.int32)


# ### Euclidean distance

# In mathematics, the __Euclidean distance__ between two points in Euclidean space is the length of a line segment between the two points. It can be calculated from the Cartesian coordinates of the points using the Pythagorean theorem, therefore occasionally being called the Pythagorean distance. These names come from the ancient Greek mathematicians Euclid and Pythagoras, although Euclid did not represent distances as numbers, and the connection from the Pythagorean theorem to distance calculation was not made until the 18th century.

# The distance between two objects that are not points is usually defined to be the smallest distance among pairs of points from the two objects. Formulas are known for computing distances between different types of objects, such as the distance from a point to a line. In advanced mathematics, the concept of distance has been generalized to abstract metric spaces, and other distances than Euclidean have been studied. In some applications in statistics and optimization, the square of the Euclidean distance is used instead of the distance itself.

# - formula for one dimension
# ![euclidian distance](../assets/eud_1.svg)
# - formula for two dimension
# ![euclidian distance](../assets/eud_2.svg)
# - formula for higher dimension
# ![euclidian distance](../assets/eud_n.svg)

# In[6]:


def euclidean_distance(inputs, kernel_size):
    centers = tf.image.resize_with_crop_or_pad(inputs, 1, 1)
    return tf.cast(
        tf.math.less_equal(
            tf.math.pow(
                tf.math.reduce_sum(
                    tf.math.pow(
                        tf.math.subtract(inputs, centers), 
                        2), 
                    axis=3), 
                0.5), 
            kernel_size), 
        dtype=tf.int32)


# ### Manhattan distance

# A taxicab geometry is a form of geometry in which the usual distance function or metric of Euclidean geometry is replaced by a new metric in which the distance between two points is the sum of the absolute differences of their Cartesian coordinates. The taxicab metric is also known as rectilinear distance, __Manhattan distance__ or Manhattan length, with corresponding variations in the name of the geometry. The latter names allude to the grid layout of most streets on the island of Manhattan, which causes the shortest path a car could take between two intersections in the borough to have length equal to the intersections' distance in taxicab geometry.

# The geometry has been used in regression analysis since the 18th century, and today is often referred to as LASSO. The geometric interpretation dates to non-Euclidean geometry of the 19th century and is due to Hermann Minkowski.

# ![manhatten distance](../assets/md.svg)

# In[7]:


def manhattan_distance(inputs, kernel_size):
    centers = tf.image.resize_with_crop_or_pad(inputs, 1, 1)
    return tf.cast(
        tf.math.less_equal(
            tf.math.reduce_sum(
                tf.math.abs(tf.math.subtract(inputs, centers)), 
                axis=3), 
            kernel_size), 
        dtype=tf.int32)


# ## Fractal layer

# We define a custom layer, which extracts fractal features from images and reshapes them into an artificial image.

# In[8]:


class Fractal2D(tf.keras.layers.Layer):
    def __init__(
        self, 
        fractal_output, 
        percolation_threshold, 
        distance_functions, 
        next_shape,
        verbose
    ):
        super(Fractal2D, self).__init__(name='fractal_2d')
        
        a, b = fractal_output
        self.kernel_size_start = 3
        self.kernel_size_end = int(0.4 * a * b + 1)
        self.kernel_size_step = 2
        
        self.percolation_threshold = percolation_threshold
        self.distance_functions = distance_functions
        self.next_shape = next_shape
        self.fractal_output = fractal_output
        self.verbose = verbose

        
    def calculate_binarized_patches(self, inputs, kernel_size, distance_function):
        patched_inputs = tf.image.extract_patches(
            inputs,
            sizes=(1, kernel_size, kernel_size, 1),
            strides=(1, kernel_size, kernel_size, 1),
            rates=(1, 1, 1, 1),
            padding='SAME'
        )
        _, rows, cols, _ = patched_inputs.shape
        patched_inputs = tf.reshape(patched_inputs, shape=(-1, kernel_size, kernel_size, 3))
        
        return tf.reshape(
            distance_function(patched_inputs, kernel_size), 
            shape=(-1, rows * cols, kernel_size, kernel_size)
        )
    
    
    def calculate_probability_matrices(self, binarized_inputs, kernel_size):
        number_of_ones = tf.map_fn(
            lambda binarized_input: tf.map_fn(lambda binary_patch: tf.math.reduce_sum(binary_patch), binarized_input), 
            binarized_inputs
        )
        _, patch_number = number_of_ones.shape
        return tf.math.bincount(number_of_ones,
                                minlength=1, 
                                maxlength=kernel_size ** 2, 
                                axis=-1) / patch_number
    
    
    def calculate_fractal_dimensions(self, probability_matrices):
        def fd_helper(matrix):
            return tf.math.reduce_sum(tf.math.divide(matrix, tf.range(1, len(matrix) + 1, dtype=tf.float64)))
        return tf.map_fn(lambda matrix: fd_helper(matrix), probability_matrices)
    
    
    def calculate_lacunarity(self, probability_matrices):
        def m_helper(matrix):
            return tf.math.reduce_sum(tf.math.multiply(matrix, tf.range(1, len(matrix) + 1, dtype=tf.float64)))
        
        def m2_helper(matrix):
            return tf.math.reduce_sum(tf.math.multiply(tf.math.pow(matrix, 2), tf.range(1, len(matrix) + 1, dtype=tf.float64)))
        
        return tf.map_fn(lambda probability_matrix: 
                         tf.math.divide(
                             tf.math.subtract(m2_helper(probability_matrix), 
                                               tf.math.pow(m_helper(probability_matrix), 2)), 
                             tf.math.pow(m_helper(probability_matrix), 2)), 
                         probability_matrices)
    
    
    def calculate_average_cluster_percolation(self, binarized_inputs, kernel_size):
        number_of_ones = tf.map_fn(lambda binarized_input: tf.map_fn(lambda binary_patch: tf.math.reduce_sum(binary_patch), 
                                                                  binarized_input), 
                                   binarized_inputs)
        
        return tf.math.reduce_mean(
                        tf.cast(
                            tf.math.greater_equal(
                                tf.math.divide(number_of_ones, kernel_size ** 2), 
                                self.percolation_threshold), 
                            dtype=tf.int32), 
                    axis=1)
    
    
    def calculate_average_cluster_number(self, binarized_inputs):
        return tf.math.reduce_mean(
            tf.map_fn(
                lambda binarized_input: tf.map_fn(
                    lambda patch: tf.math.reduce_max(tfa.image.connected_components(patch)), 
                    binarized_input), 
                binarized_inputs), 
            axis=1)
    
    def calculate_average_cluster_max_area(self, binarized_inputs):    
        def most_common(array):
            _, _, counts = tf.unique_with_counts(array)
            return tf.math.reduce_max(counts)
        
        return tf.math.reduce_mean(
                tf.map_fn(lambda binarized_input: 
                            tf.map_fn(lambda patch: 
                                        most_common(tf.reshape(tfa.image.connected_components(patch), shape=(-1,))), 
                                      binarized_input), 
                          binarized_inputs), axis=1)
    
    def calculate_components(self, inputs, kernel_size, distance_function):
        self.log(message=f'\t\tfractal2d: calculating binarized patches')
        binary_patches = self.calculate_binarized_patches(inputs, kernel_size, distance_function)
        
        self.log(message=f'\t\tfractal2d: calculating probability matrices')
        probability_matrices = self.calculate_probability_matrices(binary_patches, kernel_size)
        
        self.log(message=f'\t\tfractal2d: calculating fractal dimensions')
        fractal_dimensions = self.calculate_fractal_dimensions(probability_matrices)
        
        self.log(message=f'\t\tfractal2d: calculating lacunarity')
        lacunarity = self.calculate_lacunarity(probability_matrices)
        
        self.log(message=f'\t\tfractal2d: calculating average cluster percolation')
        average_cluster_percolation = self.calculate_average_cluster_percolation(binary_patches, kernel_size)
        
        self.log(message=f'\t\tfractal2d: calculating average cluster number')
        average_cluster_number = self.calculate_average_cluster_number(binary_patches)
        
        self.log(message=f'\t\tfractal2d: calculating average cluster max area')
        average_cluster_max_area = self.calculate_average_cluster_max_area(binary_patches)
        return tf.convert_to_tensor((average_cluster_number,
                                    average_cluster_percolation,
                                    average_cluster_max_area,
                                    lacunarity,
                                    fractal_dimensions), dtype=tf.float64)
    
    def rearrage_metrics(self, components):
        def helper(components_input):
            length, = components_input.shape
            
            rearranged_components = tf.concat([
                tf.boolean_mask(components_input, tf.range(length) % 5 == 0),
                tf.boolean_mask(components_input, tf.range(length) % 5 == 1),
                tf.boolean_mask(components_input, tf.range(length) % 5 == 2),
                tf.boolean_mask(components_input, tf.range(length) % 5 == 3),
                tf.boolean_mask(components_input, tf.range(length) % 5 == 4),
            ], axis=0)
            return rearranged_components
        return tf.map_fn(helper, components)
    
    
    def call(self, inputs):
        r_components, g_components, b_components = [], [], []
        for kernel_size in range(self.kernel_size_start, self.kernel_size_end + 1, self.kernel_size_step):
            self.log(message=f'\tfractal2d: kernel_size={kernel_size}')
            self.log(message=f'\tfractal2d: adding red components')
            r_components.append(
                tf.transpose(
                    self.calculate_components(inputs, kernel_size, distance_function=self.distance_functions['r'])
                )
            )
            
            self.log(message=f'\tfractal2d: adding green components')
            g_components.append(
                tf.transpose(
                    self.calculate_components(inputs, kernel_size, distance_function=self.distance_functions['g'])
                )
            )
            
            self.log(message=f'\tfractal2d: adding blue components')
            b_components.append(
                tf.transpose(
                    self.calculate_components(inputs, kernel_size, distance_function=self.distance_functions['b'])
                )
            )
            
        r_components = tf.reshape(self.rearrage_metrics(tf.concat(r_components, axis=1)), shape=(-1,) + self.fractal_output)
        g_components = tf.reshape(self.rearrage_metrics(tf.concat(g_components, axis=1)), shape=(-1,) + self.fractal_output)
        b_components = tf.reshape(self.rearrage_metrics(tf.concat(b_components, axis=1)), shape=(-1,) + self.fractal_output)
        
        outputs = tf.concat([
            tf.expand_dims(r_components, axis=3), 
            tf.expand_dims(g_components, axis=3),
            tf.expand_dims(b_components, axis=3)
        ], 
            axis=3)
        
        return tf.image.resize(outputs, size=self.next_shape)
    
    def log(self, message):
        if self.verbose:
            print(message)


# ## Combine function

# We define a function, combining results from the two sub models.

# In[9]:


combine_function = lambda fractal, ordinary: tf.math.add(fractal, ordinary)


# ## Fractal model

# We define a fractal model, which is an ensemble model, consisting of a convolutional neural network and a convolutional neural network with the fractal layer.

# In[10]:


class FractalModel(tf.keras.Model):
    def __init__(
        self, 
        input_shape, 
        fractal_output, 
        percolation_threshold, 
        distance_functions, 
        combine_function, 
        class_number,
        verbose=True
    ):
        super(FractalModel, self).__init__(self, name='fractal_model')
        
        self.input_shape_ = input_shape
        
        self.fractal2d = Fractal2D(
            fractal_output=fractal_output, 
            percolation_threshold=percolation_threshold,
            distance_functions=distance_functions,
            next_shape=input_shape[:-1],
            verbose=verbose
        )
        self.rescaling = tf.keras.layers.Rescaling(scale=1./255)
        self.mobilenet_v2 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", 
                                           output_shape=[1280],
                                           trainable=False)
        self.score = tf.keras.layers.Dense(class_number, activation='softmax')
        self.verbose = verbose
        
    def call(self, inputs):
        inputs = tf.ensure_shape(inputs, (None, ) + self.input_shape_)
        
        fractal = inputs
        self.log(message='fractal: fractal2d: start')
        fractal = self.fractal2d(fractal)
        
        self.log(message='fractal: rescaling: start')
        fractal = self.rescaling(fractal)
        
        self.log(message='fractal: mobilenet_v2: start')
        fractal = self.mobilenet_v2(fractal)
        
        ordinary = inputs
        self.log(message='ordinary: rescaling: start')
        ordinary = self.rescaling(ordinary)
        
        self.log(message='ordinary: mobilenet_v2: start')
        ordinary = self.mobilenet_v2(ordinary)
        
        self.log(message='fractal & ordinary: combine_function: start')
        combine = combine_function(fractal, ordinary)
        
        self.log(message='fractal & ordinary: score: start')
        score = self.score(combine)
        
        return score
    
    def log(self, message):
        if self.verbose:
            print(message)


# # Model training

# ### Building the model

# We take the model from TensorFlow Hub. [Look here](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4).

# In[11]:


model = FractalModel(
    input_shape=(224, 224, 3), 
    fractal_output=(10, 10), 
    percolation_threshold=0.59275, 
    distance_functions={
        'r': chessboard_distance,
        'g': euclidean_distance,
        'b': manhattan_distance
    }, 
    combine_function=combine_function, 
    class_number=CLASS_NUMBER
)


# In[12]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# ### Preparing TensorFlow callbacks

# For our convenience, we create a few TensorFlow callbacks.

# #### The TensorBoard callback

# We want to see how the training is going. We add the callback, which will log the metrics to TensorBoard.

# In[13]:


log_dir = '../logs/fit/' + datetime.datetime.now().strftime('fractalnet')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# #### The EarlyStopping callback

# This callback stops training when the metrics (e.g. validation loss) are not improving,

# In[14]:


early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    min_delta=0.01, 
    patience=10, 
    restore_best_weights=True
)


# #### The ModelCheckpoint callback

# This callback saves the model with the best metrics during training.

# In[15]:


checkpoint_path = 'checkpoints/fractalnet.ckpt'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch',
    mode='auto'
)


# ### Training the model

# In[ ]:


model.fit(
    training_set, 
    validation_data=validation_set, 
    epochs=1, 
    callbacks=[
                tensorboard_callback,
                checkpoint_callback,
                early_stop_callback
            ]
)


# # Model validation

# ### Loading the model

# We load the model with the best metrics (e.g. validation loss) from the checkpoint.

# In[ ]:


model = FractalModel(
    input_shape=(224, 224, 3), 
    fractal_output=(10, 10), 
    percolation_threshold=0.59275, 
    distance_functions={
        'r': chessboard_distance,
        'g': euclidean_distance,
        'b': manhattan_distance
    }, 
    combine_function=combine_function, 
    class_number=CLASS_NUMBER
)


# In[ ]:


model.load_weights('./checkpoints/fractalnet.ckpt')


# ### Loading the test data

# In[ ]:


testing_set = generator.flow_from_directory(
    '/small-data-test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# ### Making diagnoses

# In[ ]:


true_labels = np.concatenate([testing_set[i][1] for i in range(len(testing_set))], axis=0)


# In[ ]:


predicted_labels = model.predict(testing_set)


# ### Plot the ROC Curve

# In[ ]:


fpr = dict()
tpr = dict()
auc_metric = dict()

diagnosis_index_dict = {v: k for k, v in testing_set.class_indices.items()}

for i in range(CLASS_NUMBER):
    diagnosis = diagnosis_index_dict[i]
    fpr[diagnosis], tpr[diagnosis], _ = roc_curve(true_labels[:, i], predicted_labels[:, i])
    auc_metric[diagnosis] = auc(fpr[diagnosis], tpr[diagnosis])


# In[ ]:


for diagnosis in testing_set.class_indices:
    plt.plot(fpr[diagnosis], tpr[diagnosis], label=diagnosis)
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# ### Show AUC

# In[ ]:


auc_metric


# In[ ]:




