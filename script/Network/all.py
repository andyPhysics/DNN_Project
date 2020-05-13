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
from keras.layers import MaxPooling2D, GaussianNoise, SeparableConv2D, Conv2D, GlobalAveragePooling2D,LSTM
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
                    default='model.h5',
                    help='This is the output model at the end of training(.h5)')

parser.add_argument('-b',
                    dest='output_best',
                    default='model_best.h5',
                    help='This is the best output model(.h5)')

parser.add_argument('-c',
                    dest='cnn_model',
                    default='cnn_model.h5',
                    help='This is the output of the CNN model(.h5)')

parser.add_argument('-t',
                    dest='training_output',
                    default='training_curve',
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

def loss_space_angle(y_true,y_pred):
    y_true1 = y_true
    y_pred1 = y_pred
    
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

feature_number = 1

#------------------------------------------------------------------------------------------

input_new = Input(shape=(img_heights,img_rows))

cnn_model = LSTM(64)(input_new)



#------------------------------------
def output_DNN(cnn_model1,activation,shape):
    model5 = Dense(32)(cnn_model1)
    output5 = ELU()(model5)

    model6 = Dropout(args.do_rate)(output5)

    model6 = Dense(16)(model6)
    output6 = ELU()(model6)

    predictions = Dense(shape,activation=activation)(output6)
    
    return predictions

pred1 = output_DNN(cnn_model,args.activation,1)
pred2 = output_DNN(cnn_model,args.activation,1)
pred3 = output_DNN(cnn_model,'sigmoid',1)



model = Model(inputs=[input_new],outputs=[pred1,pred2,pred3])

model.compile(optimizer=opt , loss = ['mse','mse','binary_crossentropy'],metrics = ['mse','mse','acc'])

print(model.summary())

history = model.fit_generator(Data_generator(file_path_train,2,activation_function=args.activation,first_iter=first_iter,percent=Percent_files,up=filter_energy),
                              epochs = epochs,
                              validation_data=Data_generator(file_path_test,4,activation_function=args.activation,up=filter_energy),
                              workers = num_cpus,
                              callbacks = [early_stop,best_model],
                              use_multiprocessing = False)

training = zip(history.history['loss'],history.history['val_loss'])


model.save(args.output_file)
np.save(args.training_output,training)
