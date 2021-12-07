# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 14:42:46 2021

@author: Sebad
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
import glob
import tensorflow_addons as tfa
from tensorflow.keras import layers
import RandomLayers


class CarDetection():
    def __init__(self):
        self.images = []
        self.data = []
        self.files = []

    def load_images_and_data(self, images_path, data_path):
        self.files = os.listdir(images_path)
        self.images = np.array([cv2.imread(images_path + file) for file in self.files])
        self.data = pd.read_csv(data_path)  
        
    def get_data(self):
        
        return self.data
    
    def get_images(self):
        
        return self.images
    
    def show_images(self):
        
        for index, file in enumerate(self.files):
    
            rectangle = self.data.loc[self.data['image']==file].drop('image', 1).astype(int)
            
            if rectangle.empty:
                
                pass
            
            else:
                
                for rect in rectangle.values:
                    
                    pt1 = tuple(rect[:2])
                    
                    pt2 = tuple(rect[2:])
                    
                    cv2.rectangle(self.images[index], pt1, pt2 , (0, 0, 255), 2)
                 
            cv2.imshow('img', self.images[index])
            
            key = cv2.waitKey(100)
            
            if key == ord('q'):
                
                cv2.destroyAllWindows()
                
                break


class SizeImages():
    def __init__(self, w, h):
        self.w = w
        self.h = h


class ShapeGridCell():
    def __init__(self, x, y):
        self.x = x
        self.y = y



def get_bounding_boxes(training_path, path, size_images, shape_grid_cell):

    files = os.listdir(training_path)

    data = pd.read_csv(path)

    labels = []

    for index, file in enumerate(files):

        rectangle = data.loc[data['image']==file].drop('image', 1).astype(int)
            
        if rectangle.empty:
                
            labels.append([[0, 0, 0, 0, 0]])
            
        else:
                
            sub_labels = []


                sub_labels.append([x, y, w, h, 1])

                break
            
            labels.append(sub_labels)

    return labels
                    
def print_images(dataset, labels_for_printing) :

    plt.figure(figsize=(10, 10))

    for images, _ in dataset.take(1):

        for i in range(9):

            ax = plt.subplot(3, 3, i + 1)

            bounding_boxes = tf.cast(tf.expand_dims(labels_for_printing[i], axis=0), dtype="float32")

            image_drawn = tf.image.draw_bounding_boxes( tf.expand_dims(images[i].numpy().astype("float32"), axis=0),
                                                        bounding_boxes,
                                                        np.array([[1.0, 0.0, 0.0]])
                                                        )

            plt.imshow(tf.cast(image_drawn[0], dtype="uint8"))

            plt.axis("off")

    plt.show()

def data_augmentation(dataset) :

    fud_dataset = dataset.map(lambda x, y: (tf.image.flip_up_down(x),y))

    flr_dataset = dataset.map(lambda x, y: (tf.image.flip_left_right(x),y))

    fud_lr_dataset = fud_dataset.map(lambda x, y: (tf.image.flip_left_right(x),y))

    dataset = dataset.concatenate(fud_dataset.concatenate(flr_dataset.concatenate(fud_lr_dataset)))

    return dataset

def parse_function(filename, label):

    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string)
    image = tf.cast(image_decoded, tf.float32)
    return image, label










"""
            for rect in rectangle.values:

                w = rect[2] - rect[0]

                h = rect[3] - rect[1]
                
                x = w /2 + rect[0]

                x = x / (size_images.w/shape_grid_cell.x) % 1

                y = h /2 + rect[1]

                y = y / (size_images.h/shape_grid_cell.y) % 1

                h = tf.sqrt(h/size_images.h)

                w = tf.sqrt(w/size_images.w)
    
"""


        