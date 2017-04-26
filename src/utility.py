##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     utility.py
# Desc:     Defines utility functions.
##################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Suppress compiler warnings.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


##############################
# PARAMETERS
##############################

# Directories
DATA_DIR = "../data/"
IMAGE_DIR = DATA_DIR + "images/"
MODEL_DIR = "../models/"
VISUALIZATION_DIR = "../images/"
PREPROCESSED_DIR = DATA_DIR + "preprocessed/"

# Model parameters
RESNET50_SIZE = 224
N_CLASSES = 200

# Model training parameters
OPTIMIZER = "rmsprop"
BATCH_SIZE = 32
EPOCHS = 4
VERBOSE = 1
SAVE = 0


##############################
# IMAGE FUNCTIONS
##############################

def show_image(image):
    """
    Show given image.
    """

    plt.xticks([]), plt.yticks([])
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()

##############################
# VISUALIZATION FUNCTIONS
##############################

def to_multiclass(lst):
    """
    Convert a one-hot encoded array to multiclass.
    """
    return np.argmax(lst, axis=1)

def visualize_cmatrix(model, X_test, Y_test, filename):
    """
    Visualize the confusion matrix for a model on a validation set.
    """

    # Predict on the test set.
    Y_true = to_multiclass(Y_test)
    Y_predict = to_multiclass(model.predict(X_test))

    # Calculate the confusion matrix.
    cmatrix = confusion_matrix(Y_true, Y_predict)
    show_image(cmatrix)

    plt.savefig(VISUALIZATION_DIR + filename)
