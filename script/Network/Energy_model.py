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
from Data_generator_energy import *

num_cpus = 3

output_file = '/data/user/amedina/Energy.h5'
cnn_model_name = '/data/user/amedina/cnn_energy.h5'
file_path_test = '/data/user/amedina/DNN/processed_simple/test/'
file_path_train = '/data/user/amedina/DNN/processed_simple/train/'
training_output = '/data/user/amedina/training_curve_energy.csv'

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')


epochs=50

img_heights,img_rows = 60,86

kernel = 3
kernel2 = 2

feature_number = 9

#------------------------------------------------------------------------------------------

model1_input = Input(shape=(feature_number,img_heights,img_rows))

model1 = LeakyReLU(alpha = 0.01)(model1_input)
output1 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(output1)

model1 = LeakyReLU(alpha = 0.01)(model1)
output2 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(output2)

model1 = LeakyReLU(alpha = 0.01)(model1)
output3 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(output3)

cnn_model1 = Flatten()(model1_input)
cnn_model2 = Flatten()(model1)
cnn_model3 = Flatten()(output1)
cnn_model4 = Flatten()(output2)
cnn_model5 = Flatten()(output3)
cnn_model = Concatenate(axis=-1)([cnn_model1,cnn_model2,cnn_model3,cnn_model4,cnn_model5])

cnn_model = Model(inputs=model1_input,outputs=cnn_model)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5, amsgrad=False)
cnn_model.compile(optimizer=opt , loss = 'mse')

#---------------------------------------------------------------------------------------------

input_new = Input(shape=(feature_number,img_heights,img_rows))

output = Lambda(lambda x: cnn_model(x))(input_new)

model = Dense(512)(output)
model = LeakyReLU(alpha = 0.01)(model)
model = Dropout(0.5)(model)
model = Dense(512)(model)
model = LeakyReLU(alpha = 0.01)(model)

input_new_prime = Flatten()(input_new)
model = Concatenate(axis=-1)([model, input_new_prime])

predictions = Dense(1)(model)

model = Model(inputs=input_new,outputs=predictions)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5, amsgrad=False)
model.compile(optimizer=opt , loss = 'mse')

history = model.fit_generator(Data_generator(file_path_train,4),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4),
                              workers = num_cpus,
                              use_multiprocessing = True)

training = zip(history.history['loss'],history.history['val_loss'])


cnn_model.save(cnn_model_name)
model.save(output_file)
np.savetxt(training_output,training,delimiter=',')

