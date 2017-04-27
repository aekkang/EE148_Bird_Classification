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

from data_preprocessing import load_data
from utility import *


##############################
# VISUALIZATION
##############################

# Load the model and its history, along with the data.
model = load_model(RESULTS_DIR + MODE_KEY + "weights.hdf5")
history = np.load(RESULTS_DIR + MODE_KEY + "image_classification_results.npy", fix_imports=True, encoding="bytes")
(X_train, Y_train), (X_test, Y_test) = load_data()

# Visualize the confusion matrix.
cmatrix = visualize_cmatrix(model, X_test, Y_test, VISUALIZATION_DIR + MODE_KEY + "confusion_matrix.png")
np.save(RESULTS_DIR + MODE_KEY + "confusion_matrix", cmatrix)

# Visualize the training and validation accuracy over time.
# plot_accuracy(history, MODE_KEY + "image_classification_accuracy.png")
