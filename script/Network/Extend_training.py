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
                    default='linear',
                    help='Last layer activation')

parser.add_argument('-o',
                    dest='output_file',
                    default='model_all_extended.h5',
                    help='This is the output model at the end of training(.h5)')

parser.add_argument('-m',
                    dest='model',
                    default='model_best.h5',
                    help='This is the output of the previous model(.h5)')

parser.add_argument('-n',
                    dest='model_best',
                    default='model_best_new.h5',
                    help='This is the output of the model(.h5)')


parser.add_argument('-t',
                    dest='training_output',
                    default = 'training_curve_extended.csv',
                    help='This is the output file with the training curve(.csv)')

parser.add_argument('-do',
                    dest='do_rate',
                    default=0.5,
                    help='This is the dropout rate')

args = parser.parse_args()

file_path_test = '/data/user/amedina/DNN/processed_2D/test/'
file_path_train = '/data/user/amedina/DNN/processed_2D/train/'


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

best_model = keras.callbacks.ModelCheckpoint(args.model_best,
                                             monitor='val_loss',
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='auto')


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')

opt = keras.optimizers.RMSprop(decay=1e-5)

img_heights,img_rows = 60,86

#------------------------------------------------------------------------------------------

model = load_model(args.model,custom_objects = {'loss_space_angle':loss_space_angle})
for n in model.layers[0:10]:
    print(n)
    n.trainable=False

model.compile(optimizer=opt , loss = loss_space_angle)

history = model.fit_generator(Data_generator(file_path_train,2,activation_function=args.activation,first_iter=first_iter,percent=Percent_files),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4,activation_function=args.activation),
                              workers = num_cpus,
                              callbacks = [best_model,early_stop],
                              use_multiprocessing = False)

training = zip(history.history['loss'],history.history['val_loss'])

model.save(args.output_file)
np.save(args.training_output,training)
