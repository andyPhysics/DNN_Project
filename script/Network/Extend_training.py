#!/usr/bin/env python 

import numpy as np
import pandas as pd
import sys,os
import math
import multiprocessing as mp
from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, Concatenate, Lambda
from keras.layers import SeparableConv2D, MaxPooling2D, GaussianNoise
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from keras import regularizers
from Data_generator_no import *

num_cpus = 3

output_file = '/data/user/amedina/test.h5'
output_best = '/data/user/amedina/test.h5'
input_cnn = '/data/user/amedina/DNN/models/cnn_model_simple.h5'
file_path_test = '/data/user/amedina/DNN/processed_simple/test/'
file_path_train = '/data/user/amedina/DNN/processed_simple/train/'
training_output = '/data/user/amedina/test.csv'

def loss_space_angle(y_true,y_pred):
    y_true1 = y_true*2.0-1.0                                
    y_pred1 = y_pred*2.0-1.0 
    subtraction = tf.math.subtract(y_true1,y_pred1)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')

best_model = keras.callbacks.ModelCheckpoint(output_best,
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1)

epochs=50

img_heights,img_rows = 60,86

kernel = 3
kernel2 = 2

feature_number = 9

#------------------------------------------------------------------------------------------

cnn_model = load_model(input_cnn,custom_objects={'loss_space_angle':loss_space_angle})

input_new = Input(shape=(feature_number,img_heights,img_rows))

output = Lambda(lambda x: cnn_model(x))(input_new)

model = Dense(512)(output)
model = LeakyReLU(alpha = 0.01)(model)
model = Dropout(0.5)(model)
model = Dense(512)(model)
model = LeakyReLU(alpha = 0.01)(model)

input_new_prime = Flatten()(input_new)
model = Concatenate(axis=-1)([model, input_new_prime])

predictions = Dense(3,activation='sigmoid')(model)

model = Model(inputs=input_new,outputs=predictions)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5, amsgrad=False)
model.compile(optimizer=opt , loss = loss_space_angle)

history = model.fit_generator(Data_generator(file_path_train,4,percent=0.01),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4,percent=0.01),
                              workers = num_cpus,
                              use_multiprocessing = True)

training = zip(history.history['loss'],history.history['val_loss'])


cnn_model.save(input_cnn)
model.save(output_file)
np.savetxt(training_output,training,delimiter=',')

