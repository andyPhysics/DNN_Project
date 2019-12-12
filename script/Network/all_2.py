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
from Data_generator_new import *
import argparse


num_cpus = 3
epochs=30
Percent_files = 1.0
first_iter = False

parser = argparse.ArgumentParser(description='Process DNN')

parser.add_argument('-a',
                    dest = 'activation',
                    default='tanh',
                    help='Last layer activation')

parser.add_argument('-o',
                    dest='output_file',
                    default='/data/user/amedina/model.h5',
                    help='This is the output model at the end of training(.h5)')

parser.add_argument('-b',
                    dest='output_best',
                    default='/data/user/amedina/model_best.h5',
                    help='This is the best output model(.h5)')

parser.add_argument('-c',
                    dest='cnn_model',
                    default='/data/user/amedina/cnn_model.h5',
                    help='This is the output of the CNN model(.h5)')

parser.add_argument('-t',
                    dest='training_output',
                    default='/data/user/amedina/training_curve.csv',
                    help='This is the output file with the training curve(.csv)')

parser.add_argument('-do',
                    dest='do_rate',
                    default=0.5,
                    type = float,
                    help='This is the dropout rate')

parser.add_argument('-energy',
                    dest='filter_energy',
                    default=0,
                    help='Filter energy:[0] all,[1] is 4 to 5, [2] is 5 to 6 and [3] is 6 to 7')


args = parser.parse_args()

file_path_test = '/data/user/amedina/DNN/processed_simple/test/'
file_path_train = '/data/user/amedina/DNN/processed_simple/train/'
filter_energy = int(args.filter_energy)

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

#best_model = keras.callbacks.ModelCheckpoint(args.output_best,
#                                             monitor='val_loss',
#                                             save_best_only=True,
#                                             save_weights_only=False,
#                                             mode='auto')

opt = keras.optimizers.Adamax(lr=3e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5)
#opt = keras.optimizers.RMSprop(decay=1e-5)


img_heights,img_rows = 60,86

kernel = 3
kernel2 = 2

feature_number = 9

#------------------------------------------------------------------------------------------

model1_input = Input(shape=(feature_number,img_heights,img_rows))

model1 = ELU()(model1_input)
output1 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(output1)

model1 = ELU()(model1)
output2 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(output2)

model1 = ELU()(model1)
output3 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01),data_format='channels_first')(output3)
cnn_model = Flatten()(model1)

cnn_model = Model(inputs=model1_input,outputs=cnn_model)
cnn_model.compile(optimizer=opt , loss = loss_space_angle)

#---------------------------------------------------------------------------------------------

input_new = Input(shape=(feature_number,img_heights,img_rows))
cos_values_line = Input(shape=(3,))

output1 = Lambda(lambda x: cnn_model(x))(input_new)

input_new_prime = Flatten()(input_new)

model1 = Dropout(rate=args.do_rate)(output1)
model1 = Concatenate(axis=-1)([model1,cos_values_line])
model1 = Dense(256)(model1)
model1 = ELU()(model1)
model1 = Dropout(rate=args.do_rate)(model1)
model1 = Dense(256)(model1)
model1 = ELU()(model1)
model1 = Concatenate(axis=-1)([model1, input_new_prime,cos_values_line])

predictions = Dense(3)(model1)

model = Model(inputs=[input_new,cos_values_line],outputs=predictions)
model.compile(optimizer=opt , loss = loss_space_angle)

history = model.fit_generator(Data_generator(file_path_train,2,activation_function=args.activation,first_iter=first_iter,percent=Percent_files,up=filter_energy),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4,activation_function=args.activation,up=filter_energy),
                              workers = num_cpus,
                              #callbacks=[best_model],
                              use_multiprocessing = False)

training = zip(history.history['loss'],history.history['val_loss'])


cnn_model.save(args.cnn_model)
model.save(args.output_file)
np.save(args.training_output,training)
