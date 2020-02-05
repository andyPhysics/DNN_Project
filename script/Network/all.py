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
from keras.layers import MaxPooling2D, GaussianNoise, SeparableConv2D, Conv2D
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from keras import regularizers
from Data_generator import *
import argparse


num_cpus = 3
epochs=100
Percent_files = 0.1
first_iter = False

parser = argparse.ArgumentParser(description='Process DNN')

parser.add_argument('-a',
                    dest = 'activation',
                    default='linear',
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
                    help='This is the dropout rate')

parser.add_argument('-energy',
                    dest='filter_energy',
                    default=0,
                    help='Filter Energy:0 is all, 1 is below 10^5, 2 is above 10^5')


args = parser.parse_args()

file_path_test = '/data/user/amedina/DNN/processed_2D/test/'
file_path_train = '/data/user/amedina/DNN/processed_2D/train/'
filter_energy = int(args.filter_energy)

check = ['sigmoid']

def loss_space_angle(y_true,y_pred):
    if args.activation in check:
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

best_model = keras.callbacks.ModelCheckpoint(args.output_best,
                                             monitor='val_loss',
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='auto')

#opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#opt = keras.optimizers.Adam(lr=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5)
opt = keras.optimizers.RMSprop(decay=1e-5)

img_heights,img_rows = 60,86

kernel = 3
kernel2 = 2

feature_number = 9

#------------------------------------------------------------------------------------------
#model1_input = Input(shape=(feature_number,img_heights,img_rows))

input_new = Input(shape=(feature_number,img_heights,img_rows))

model = SeparableConv2D(32,kernel,padding='same',data_format='channels_first')(input_new)
model = ELU()(model)
output = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model)
model = SeparableConv2D(32,kernel,padding='same',data_format='channels_first')(output)

#model1 = LeakyReLU(alpha = 0.01)(model)
model1 = ELU()(model)
output1 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model1)
model1 = SeparableConv2D(32,kernel,padding='same',data_format='channels_first')(output1)

#model2 = LeakyReLU(alpha = 0.01)(model1)
model2 = ELU()(model1)
output2 = MaxPooling2D(kernel2,padding='same',data_format='channels_first')(model2)
model2 = SeparableConv2D(32,kernel,padding='same',data_format='channels_first')(output2)
model2 = ELU()(model2)

cnn_model = Flatten()(model2)

#cnn_model = Model(inputs=model1_input,outputs=cnn_model)
#cnn_model.compile(optimizer=opt , loss = loss_space_angle)

#---------------------------------------------------------------------------------------------

#input_new = Input(shape=(feature_number,img_heights,img_rows))

cos_values_line = Input(shape=(3,))

#output = Lambda(lambda x: cnn_model(x))(input_new)

model3 = Dropout(args.do_rate)(cnn_model)

model3 = Dense(32)(model3)
output3 = ELU()(model3)
#output1 = LeakyReLU(alpha = 0.01)(model1)

model3 = Dropout(args.do_rate)(output3)

model4 = Dense(16)(model3)
output4 = ELU()(model4)
#output2 = LeakyReLU(alpha = 0.01)(model2)

model5 = Dropout(args.do_rate)(output4)

predictions = Dense(3,activation=args.activation)(model5)

model = Model(inputs=[input_new,cos_values_line],outputs=predictions)

model.compile(optimizer=opt , loss = 'mse')

print(model.summary())

history = model.fit_generator(Data_generator(file_path_train,2,activation_function=args.activation,first_iter=first_iter,percent=Percent_files,up=filter_energy),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4,activation_function=args.activation,up=filter_energy),
                              workers = num_cpus,
                              callbacks = [early_stop,best_model],
                              use_multiprocessing = False)

training = zip(history.history['loss'],history.history['val_loss'])


#cnn_model.save(args.cnn_model)
model.save(args.output_file)
np.save(args.training_output,training)
