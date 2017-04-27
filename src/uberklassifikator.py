##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     uberklassifikator.py
# Desc:     Trains a classifier that connects
#           several pre-trained networks to
#           predict on the Caltech-UCSD
#           Birds-200 dataset.
##################################################

import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from keras.models import Model

from data_preprocessing import *
from utility import *


##############################
# LAYER OUTPUTS
##############################

def get_first_dense_outputs(model, X):
    """
    Get the outputs of the first dense layer as features.
    """

    # The model truncated at the first dense layer.
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer("dense_1").output)

    # Find the latent representation of the dataset.
    output = intermediate_model.predict(X)

    return output

def uberklassifikator(models, X_trains, X_tests, Y_train, Y_test):
    """
    Train an SVM that concatenates the given models.
    """

    Y_train_m = to_multiclass(Y_train)
    Y_test_m = to_multiclass(Y_test)

    # Gather layer outputs as features.
    X_train_final = None
    X_test_final = None

    for i, model in enumerate(models):
        if X_train_final is None:
            X_train_final = get_first_dense_outputs(model, X_trains[i])
            X_test_final = get_first_dense_outputs(model, X_tests[i])
        else:
            X_train_final = np.concatenate((X_train_final, get_first_dense_outputs(model, X_trains[i])), axis=1)
            X_test_final = np.concatenate((X_test_final, get_first_dense_outputs(model, X_tests[i])), axis=1)

    # Train the SVC.
    uber_model = SVC()
    uber_model.fit(X_train_final, Y_train_m)
    Y_pred = uber_model.predict(X_test_final)

    # Score the SVC.
    print(uber_model.score(X_train_final, Y_train_m))
    print(uber_model.score(X_test_final, Y_test_m))

    # Visualize the confusion matrix.
    cmatrix = confusion_matrix(Y_test_m, Y_pred)
    np.save(RESULTS_DIR + "SVM_confusion_matrix", cmatrix)
