##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     utility.py
# Desc:     Defines utility functions.
##################################################

import matplotlib.pyplot as plt

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
PREPROCESSED_DIR = DATA_DIR + "preprocessed/"

# Model parameters
RESNET50_SIZE = 224
N_CLASSES = 200

# Model training parameters
OPTIMIZER = "adadelta"
BATCH_SIZE = 32
EPOCHS = 32
VERBOSE = 1


##############################
# IMAGE FUNCTIONS
##############################

def show_image(image):
    """
    Show given image.
    """

    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()


def visualize_cmatrix(model, X_test, Y_test):
    """
    Visualize the confusion matrix for a model on a validation set.
    """

    # Predict on the test set.
    Y_true = to_multiclass(Y_test)
    Y_predict = to_multiclass(model.predict(X_test))

    # Calculate the confusion matrix.
    cmatrix = confusion_matrix(Y_true, Y_predict)
    show_image(cmatrix)

    if AUGMENT:
        plt.savefig("../img/augment/confusion_matrix.png")
    else:
        plt.savefig("../img/no_augment/confusion_matrix.png")
