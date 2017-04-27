##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     data_preprocessing.py
# Desc:     Defines functions for preprocessing
#           the Caltech-UCSD Birds-200 dataset.
##################################################

import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras.utils import to_categorical
from nudged import estimate

from utility import *


##############################
# DATA PREPROCESSING
##############################

def save_data(cropped=False, warped=False):
    """
    Save the Caltech-UCSD Birds-200 dataset.
    """

    # Load relevant files.
    image_paths = np.genfromtxt(DATA_DIR + "images.txt", dtype=None)
    train_test_split = np.genfromtxt(DATA_DIR + "train_test_split.txt", dtype=None)
    image_class_labels = np.genfromtxt(DATA_DIR + "image_class_labels.txt", dtype=None)

    if cropped:
        bounding_boxes = np.genfromtxt(DATA_DIR + "bounding_boxes.txt", dtype=None)
    if warped:
        part_locations = np.genfromtxt(PARTS_DIR + "part_locs.txt", dtype=None)

    X_train, Y_train, X_test, Y_test = [], [], [], []

    # Find the best positions to warp to.
    if warped:
        ref_id, ref_path = image_paths[2]
        ref_path = ref_path.decode("UTF-8")
        ref_image = cv2.imread(IMAGE_DIR + ref_path)

        x, y, n_channels = ref_image.shape
        scale = RESNET50_SIZE / max(x, y)
        new_dim = (int(scale * y), int(scale * x))
        
        # Determine padding.
        vpadding = (224 - new_dim[1]) / 2
        hpadding = (224 - new_dim[0]) / 2

        show_image(image)
        quit()

    # Load and modify images.
    for i, (image_id, image_path) in enumerate(image_paths):
        # Extract information.
        train_test = train_test_split[i][1]
        class_label = image_class_labels[i][1] - 1

        # Read and resize image.
        image_path = image_path.decode("UTF-8")
        image = cv2.imread(IMAGE_DIR + image_path)
        
        if cropped:
            box = bounding_boxes[i]
            box = (box[1], box[2], box[3], box[4])
            image = crop_box(image, box)
        if warped:
            parts = part_locations[i]
            show_image(image)
            quit()

        image = resize_to_square(image)

        # Add the image and label to the datasets.
        if train_test:
            X_train.append(image)
            Y_train.append(class_label)
        else:
            X_test.append(image)
            Y_test.append(class_label)

    # One-hot encode the labels.
    Y_train = to_categorical(Y_train, N_CLASSES)
    Y_test = to_categorical(Y_test, N_CLASSES)

    # Save images and labels.
    if cropped:
        file_suffix = "_cropped"
    elif warped:
        file_suffix = "_warped"
    else:
        file_suffix = ""

    np.save(PREPROCESSED_DIR + "X_train" + file_suffix, X_train)
    np.save(PREPROCESSED_DIR + "Y_train" + file_suffix, Y_train)
    np.save(PREPROCESSED_DIR + "X_test" + file_suffix, X_test)
    np.save(PREPROCESSED_DIR + "Y_test" + file_suffix, Y_test)

def load_data(cropped=False):
    """
    Load and return the Caltech-UCSD Birds-200 dataset.
    """

    if cropped:
        file_suffix = "_cropped"
    elif warped:
        file_suffix = "_warped"
    else:
        file_suffix = ""

    X_train = np.load(PREPROCESSED_DIR + "X_train" + file_suffix + ".npy")
    Y_train = np.load(PREPROCESSED_DIR + "Y_train" + file_suffix + ".npy")
    X_test = np.load(PREPROCESSED_DIR + "X_test" + file_suffix + ".npy")
    Y_test = np.load(PREPROCESSED_DIR + "Y_test" + file_suffix + ".npy")

    return (X_train, Y_train), (X_test, Y_test)

def crop_box(image, box):
    """
    Crop the given image to the given bounding box.
    """

    x = int(box[0])
    y = int(box[1])
    width = int(box[2])
    height = int(box[3])

    cropped = image[y:y+height, x:x+width]
    
    return cropped

def resize_to_square(image):
    """
    Resize the given image to a 224x224 square, corresponding to an
    input for ResNet50.
    """

    # Determine new dimensions.
    x, y, n_channels = image.shape
    scale = RESNET50_SIZE / max(x, y)
    new_dim = (int(scale * y), int(scale * x))
    
    # Determine padding.
    vpadding = (224 - new_dim[1]) / 2
    hpadding = (224 - new_dim[0]) / 2

    # Resize and pad image.
    image = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image,
                               math.floor(vpadding), math.ceil(vpadding),
                               math.floor(hpadding), math.ceil(hpadding),
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image


if __name__ == "__main__":
    save_data(warped=True)
    (X_train, Y_train), (X_test, Y_test) = load_data(warped=True)
