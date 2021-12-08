"""
This file defines the logic of the yolo dataset creation with tensorflow.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os

def get_labels_from_csv(path, index='image', **kwargs):
    data = pd.read_csv(path, **kwargs)
    grouper = data.groupby(index)
    max_bounding_boxes = grouper.count().max().unique()

    def flatten_and_pad(x):
        coordinates = x.values[..., 1:].flatten()
        rest = int(max_bounding_boxes - x.shape[0])
        coordinates = np.lib.pad(coordinates, (0, rest*4), 'constant', constant_values=(0,))
        return pd.Series(coordinates)

    return grouper.apply(flatten_and_pad).sort_values(index).values


def yolo_generator(directory, labels_directory):
    nb_images = len(os.listdir(directory))
    dataset = tf.keras.utils.image_dataset_from_directory(
            directory, labels=None, batch_size=nb_images, image_size=(256,
            256), shuffle=False, seed=None, validation_split=None, subset='training',
            interpolation='bilinear', follow_links=False,
            crop_to_aspect_ratio=False)

    labels = get_labels_from_csv(labels_directory)
    labels = tf.convert_to_tensor(labels)

    def fill_images(x):
        return 

    dataset.map(lambda x: x, labels)
    dataset.map()

    return dataset