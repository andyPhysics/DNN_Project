import numpy as np
import pandas as pd
import sys,os
from create_images import entire_image

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

output_file = 'test6.h5'
output_best = 'test_best6.h5'
file_path = '/fs/scratch/PAS1495/amedina/DNN/'
y = os.listdir(file_path+'processed_new')

file_names = []

for i in y:
    file_names.append(file_path+'processed_new/'+i)

file_names_batched = list(np.array_split(file_names,1))

images = []
labels = []

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        print('Loading File: ' + i)
        x = np.load(i,allow_pickle=True).item()
        keys = x.keys()
        for key in keys:
            images.append(x[key][0])
            labels.append(x[key][1])
    return np.array(images),np.array(labels)

def get_feature(labels,feature):
    feature_values = []
    for i in labels:
        feature_values.append(i[1])
    feature_values = np.array(feature_values)
    return feature_values

images,labels = load_files(file_names_batched[0])

zenith_values = get_feature(labels,1)
azimuth_values = get_feature(labels,2)

cos_zenith_values = np.cos(zenith_values)

import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , SeparableConv2D , GlobalAveragePooling2D, BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split,KFold

def get_data(images,values):
    x_train, x_test , y_train , y_test = train_test_split(images,values,test_size = 0.2 , random_state=42)
    x_train = x_train.reshape([x_train.shape[0],x_train.shape[1],x_train.shape[2],1])
    x_test = x_test.reshape([x_test.shape[0],x_test.shape[1],x_test.shape[2],1])
    return x_train,x_test,y_train,y_test

import tensorflow as tf
tf.distribute.Strategy

def loss_space_angle(y_true,y_pred):
    subtraction = tf.math.subtract(y_true,y_pred)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

x_train,x_test,y_train,y_test = get_data(images,cos_zenith_values)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=1, mode='auto')

best_model = keras.callbacks.ModelCheckpoint(output_best,
                                             monitor='val_loss',
                                             verbose=0,
                                             save_best_only=True,
                                             save_weights_only=False,
                                             mode='auto',
                                             period=1)

batch_size = 128
epochs=5

img_rows, img_cols = 300,342
input_shape = (img_rows, img_cols)

kernel = 3
kernel2 = 2

from keras.layers import LeakyReLU
from keras import regularizers

batch_size = 128
epochs=100

img_rows, img_cols = 300,342
input_shape = (img_rows, img_cols)

kernel = 3
kernel2 = 2

model1_input = Input(shape= (img_rows,img_cols,1))

model1 = SeparableConv2D(32,kernel,
                          kernel_regularizer = regularizers.l2(0.01))(model1_input)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = MaxPooling2D(kernel2)(model1)

model1 = Dropout(0.25)(model1)

model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = MaxPooling2D(kernel2)(model1)

model1 = Dropout(0.25)(model1)

model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = SeparableConv2D(32,kernel,kernel_regularizer = regularizers.l2(0.01))(model1)
model1 = LeakyReLU(alpha = 0.01)(model1)
model1 = MaxPooling2D(kernel2)(model1)

model1 = Dropout(0.25)(model1)

model1 = Flatten()(model1)

cnn_model = Model(inputs=model1_input,outputs=model1)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6, amsgrad=False)
cnn_model.compile(optimizer=opt , loss = loss_space_angle)

input_new = Input(shape=(img_rows,img_cols,1))

model = Lambda(lambda x: cnn_model(x))(input_new)

model = Dense(16,activation='linear')(model)
model = LeakyReLU(alpha = 0.01)(model)
predictions = Dense(3)(model)

model = Model(inputs=input_new,outputs=predictions)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6, amsgrad=False)
model.compile(optimizer=opt , loss = loss_space_angle)


model.save(output_file)

#for i in file_names_batched[1:len(file_names_batched)-1]:
#    model = load_model(output_best)
#    images,labels = load_files(i)
#    zenith_values = get_feature(labels,1)
#    cos_zenith_values = np.cos(zenith_values)

#    x_train,x_test,y_train,y_test = get_data(images,cos_zenith_values)

#    model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test),
#          callbacks=[early_stop,best_model])

#    model.save(output_file)

