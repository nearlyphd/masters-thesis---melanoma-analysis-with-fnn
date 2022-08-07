import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load(sample_number=None):
    metadata = pd.read_csv(f"{os.environ['SCRATCH']}/isic-archive/metadata.csv")

    # remove invalid images
    metadata = metadata.drop(metadata[metadata['isic_id'].isin(['ISIC_0060052', 'ISIC_0029842'])].index)

    # fill out the benign_malignant column
    metadata.loc[metadata['diagnosis'] == 'basal cell carcinoma', 'benign_malignant'] = 'malignant'
    metadata.loc[metadata['diagnosis'] == 'pigmented benign keratosis', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'seborrheic keratosis', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'actinic keratosis', 'benign_malignant'] = 'malignant'
    metadata.loc[metadata['diagnosis'] == 'squamous cell carcinoma', 'benign_malignant'] = 'malignant'
    metadata.loc[metadata['diagnosis'] == 'vascular lesion', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'dermatofibroma', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'solar lentigo', 'benign_malignant'] = 'benign'

    # drop samples without diagnosis
    metadata = metadata.drop(metadata[pd.isnull(metadata['diagnosis'])].index)

    # drop samples with rare diagnosis
    diagnosis_numbers = metadata.groupby(['diagnosis'], dropna=False, as_index=False).size().sort_values(by='size')
    rare_diagnosis = diagnosis_numbers[:-10]['diagnosis']
    metadata = metadata.drop(metadata[metadata['diagnosis'].isin(rare_diagnosis)].index)

    # drop samples which are not benign or malignant
    metadata = metadata.drop(metadata[~metadata['benign_malignant'].isin(['benign', 'malignant'])].index)

    if sample_number:
        metadata = metadata.sample(sample_number)

    # calculate class weights
    class_weights = metadata.groupby(['diagnosis'], dropna=False, as_index=False).size()
    class_weights.loc[:, 'size'] = 1 / class_weights.loc[:, 'size']
    class_weights.loc[:, 'size'] *= (len(metadata) / 10)
    class_weights = dict(class_weights['size'])

    # add .jpg extension to the IDs
    metadata['filename'] = metadata['isic_id'] + '.jpg'

    rest_set, testing_set = train_test_split(metadata, test_size=0.1)

    # load the images
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.2, 1.5),
        validation_split=0.2
    )

    training_set = generator.flow_from_dataframe(
        rest_set,
        directory=f"{os.environ['SCRATCH']}/isic-archive",
        x_col='filename',
        y_col='diagnosis',

        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_set = generator.flow_from_dataframe(
        rest_set,
        directory=f"{os.environ['SCRATCH']}/isic-archive",
        x_col='filename',
        y_col='diagnosis',

        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    testing_set = generator.flow_from_dataframe(
        testing_set,
        directory=f"{os.environ['SCRATCH']}/isic-archive",
        x_col='filename',
        y_col='diagnosis',

        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'
    )

    return (training_set, validation_set, testing_set), class_weights, len(training_set.class_indices)
