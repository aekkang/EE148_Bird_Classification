from uberklassifikator import *
from keras.models import load_model
from data_preprocessing import load_data

original = load_model("../results/weights.hdf5")
cropped = load_model("../results/cropped_weights.hdf5")
warped = load_model("../results/warped_weights.hdf5")

(X_train_o, Y_train), (X_test_o, Y_test) = load_data()
(X_train_c, _), (X_test_c, _) = load_data(cropped=True)
(X_train_w, _), (X_test_w, _) = load_data(warped=True)

uberklassifikator([original, cropped, warped], [X_train_o, X_train_c, X_train_w], [X_test_o, X_test_c, X_test_w], Y_train, Y_test)
