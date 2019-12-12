#!/usr/bin/env python 

import numpy as np
import pandas as pd
import sys,os
import math
import multiprocessing as mp
from keras import backend as K
import tensorflow as tf
import keras
from keras import initializers
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, Concatenate, Lambda, ELU
from keras.layers import SeparableConv2D, MaxPooling2D, GaussianNoise
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from keras import regularizers
from Data_generator import *
import argparse


num_cpus = 3
epochs=100
Percent_files = 1.0
first_iter = False

parser = argparse.ArgumentParser(description='Process DNN')

parser.add_argument('-a',
                    dest = 'activation',
                    default='sigmoid',
                    help='Last layer activation')

parser.add_argument('-o',
                    dest='output_file',
                    default='model_all_extended.h5',
                    help='This is the output model at the end of training(.h5)')

parser.add_argument('-c',
                    dest='cnn_model',
                    default='cnn_model_all.h5',
                    help='This is the output of the CNN model(.h5)')

parser.add_argument('-t',
                    dest='training_output',
                    default = 'training_curve_extended.csv',
                    help='This is the output file with the training curve(.csv)')

parser.add_argument('-do',
                    dest='do_rate',
                    default=0.5,
                    help='This is the dropout rate')

args = parser.parse_args()

file_path_test = '/data/user/amedina/DNN/processed_simple/test/'
file_path_train = '/data/user/amedina/DNN/processed_simple/train/'


def loss_space_angle(y_true,y_pred):
    if args.activation=='sigmoid':
        y_true1 = y_true*2.0-1.0                     
        y_pred1 = y_pred*2.0-1.0
    else:
        y_true1 = y_true
        y_pred1 = y_pred
    subtraction = tf.math.subtract(y_true1,y_pred1)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')

opt = keras.optimizers.Adamax(lr=3e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5)

img_heights,img_rows = 60,86

kernel = 3
kernel2 = 2

feature_number = 9

#------------------------------------------------------------------------------------------

cnn_model = load_model(args.cnn_model,custom_objects = {'loss_space_angle':loss_space_angle})
for n in cnn_model.layers[0:len(cnn_model.layers)-4]:
    n.trainable=False

input_new = Input(shape=(feature_number,img_heights,img_rows))
cos_values_line = Input(shape=(3,))

output = Lambda(lambda x: cnn_model(x))(input_new)

model = Dropout(rate=args.do_rate)(output)
model = Concatenate(axis=-1)([model,cos_values_line])
model = Dense(512)(model)
model = ELU()(model)
model = Dropout(rate=args.do_rate)(model)
model = Dense(512)(model)
model = ELU()(model)

input_new_prime = Flatten()(input_new)
model = Concatenate(axis=-1)([model, input_new_prime])

predictions = Dense(3,activation=args.activation)(model)

model = Model(inputs=[input_new,cos_values_line],outputs=predictions)
model.compile(optimizer=opt , loss = loss_space_angle)

history = model.fit_generator(Data_generator(file_path_train,2,activation_function=args.activation,first_iter=first_iter,percent=Percent_files),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4,activation_function=args.activation),
                              workers = num_cpus,
                              use_multiprocessing = False)

training = zip(history.history['loss'],history.history['val_loss'])

cnn_model.save('cnn_model_all_extend.h5')
model.save(args.output_file)
np.save(args.training_output,training)
