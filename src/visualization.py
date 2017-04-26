##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     visualization.py
# Desc:     Visualizes models trained to predict
#           the Caltech-UCSD Birds-200 dataset.
##################################################

import numpy as np

from keras.models import load_model

from utility import *


##############################
# VISUALIZATION
##############################

# Load the model and its history.
model = load_model(RESULTS_DIR + "weights.hdf5")
history = np.load(RESULTS_DIR + "image_classification_results.npy", fix_imports=True, encoding="bytes")

# Visualize the confusion matrix.
visualize_cmatrix(model, X_test, Y_test, VISUALIZATION_DIR + "cmatrix.png")

# Visualize the training and validation accuracy over time.
# plot_accuracy(history, "image_classification_accuracy.png")
