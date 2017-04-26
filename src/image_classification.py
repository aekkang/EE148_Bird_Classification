##################################################
# EE 148 Assignment 3
#
# Author:   Andrew Kang
# File:     image_classification.py
# Desc:     Modifies a pre-trained network
#           to predict on the Caltech-UCSD
#           Birds-200 dataset.
##################################################

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from data_preprocessing import *
from utility import *


##############################
# MODEL ARCHITECTURE
##############################

# Load the pre-trained ResNet50.
base_model = ResNet50(weights='imagenet', include_top=False)

# Add new layers in place of the last layer in the original model.
output = base_model.output
output = GlobalAveragePooling2D()(output)
output = Dense(1024, activation='relu')(output)
output = Dense(N_CLASSES, activation='softmax')(output)

# Create the final model.
model = Model(inputs=base_model.input, outputs=output)


##############################
# TRAINING
##############################

(X_train, Y_train), (X_test, Y_test) = load_data()

# Freeze original ResNet50 layers during training.
for layer in base_model.layers:
    layer.trainable = False

# Print summary and compile.
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Fit the model.
fit = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)





# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np

# model = ResNet50(weights='imagenet')

# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]