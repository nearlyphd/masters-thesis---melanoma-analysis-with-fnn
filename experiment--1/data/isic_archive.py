import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def isic_archive_load(sample_size=None):
    metadata = load_metadata()
    if sample_size is not None:
        metadata = metadata.sample(sample_size)
    fix_benign_malignant(metadata)
    metadata = remove_with_nan(metadata)
    metadata = remove_rare(metadata)
    rest_set, testing_set = train_test_split(metadata, test_size=0.1)

    return load_images(rest_set, testing_set)


def load_metadata():
    metadata = pd.read_csv(f"{os.environ['SCRATCH']}/isic-archive/metadata.csv")
    metadata = metadata.drop(metadata[metadata['isic_id'].isin(['ISIC_0060052', 'ISIC_0029842'])].index)
    metadata['isic_id'] += '.jpg'
    return metadata


def fix_benign_malignant(metadata):
    metadata.loc[metadata['diagnosis'] == 'basal cell carcinoma', 'benign_malignant'] = 'malignant'
    metadata.loc[metadata['diagnosis'] == 'pigmented benign keratosis', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'seborrheic keratosis', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'actinic keratosis', 'benign_malignant'] = 'malignant'
    metadata.loc[metadata['diagnosis'] == 'squamous cell carcinoma', 'benign_malignant'] = 'malignant'
    metadata.loc[metadata['diagnosis'] == 'vascular lesion', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'dermatofibroma', 'benign_malignant'] = 'benign'
    metadata.loc[metadata['diagnosis'] == 'solar lentigo', 'benign_malignant'] = 'benign'


def remove_with_nan(metadata):
    metadata = metadata.drop(metadata[pd.isnull(metadata['diagnosis'])].index)
    return metadata


def load_images(rest_set, testing_set):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.2, 1.5),
        validation_split=0.2,
    )
    training_set = generator.flow_from_dataframe(
        rest_set,
        directory=f"{os.environ['SCRATCH']}/isic-archive",
        x_col='isic_id',
        y_col='diagnosis',

        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_set = generator.flow_from_dataframe(
        rest_set,
        directory=f"{os.environ['SCRATCH']}/isic-archive",
        x_col='isic_id',
        y_col='diagnosis',

        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    testing_set = generator.flow_from_dataframe(
        testing_set,
        directory=f"{os.environ['SCRATCH']}/isic-archive",
        x_col='isic_id',
        y_col='diagnosis',

        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'
    )
    return training_set, testing_set,  validation_set


def remove_rare(metadata):
    diagnosis_numbers = metadata.groupby(['diagnosis'], dropna=False, as_index=False).size().sort_values(by='size')
    rare_diagnosis = diagnosis_numbers[:-10]['diagnosis']
    metadata = metadata.drop(metadata[metadata['diagnosis'].isin(rare_diagnosis)].index)
    metadata = metadata.drop(metadata[~metadata['benign_malignant'].isin(['benign', 'malignant'])].index)
    return metadata
