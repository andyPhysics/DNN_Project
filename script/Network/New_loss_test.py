import numpy as np
import pandas as pd
import sys,os

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

output_file = 'gpu_test.h5'
output_best = 'gpu_test_best.h5'
file_path = '/fs/scratch/PAS1495/amedina/processed_new/'
y = os.listdir(file_path)

file_names = []

for i in y:
    file_names.append(file_path+i)

file_names_batched = list(np.array_split(file_names,5))

images = []
labels = []

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        print('Loading File: ' + i)
        x = np.load(i,allow_pickle=True,encoding='latin1').item()
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

def loss_space_angle(y_true,y_pred):
    subtraction = tf.math.subtract(y_true,y_pred)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

x_train,x_test,y_train,y_test = get_data(images,cos_zenith_values)

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


from keras.layers import LeakyReLU
from keras import regularizers

model = Sequential()
batch_size = 128
epochs=10

img_rows, img_cols = 300,342
input_shape = (img_rows, img_cols)

kernel = 3
kernel2 = 2

model = Sequential()
model.add(SeparableConv2D(16,kernel,
                          input_shape = (img_rows,img_cols,1)))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(MaxPooling2D(kernel2))

model.add(Dropout(0.1))

model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(MaxPooling2D(kernel2))

model.add(Dropout(0.1))

model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(MaxPooling2D(kernel2))

model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(SeparableConv2D(16,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(MaxPooling2D(kernel2))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(32,activation='linear'))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dense(3))

opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=.09, nesterov=True)
model.compile(optimizer=opt , loss = loss_space_angle)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stop,best_model])

model.save(output_file)

for i in file_names_batched[1:len(file_names_batched)]:
	model = load_model(output_best,custom_objects={'loss_space_angle':loss_space_angle})
	images,labels = load_files(i)
	zenith_values = get_feature(labels,1)
	cos_zenith_values = np.cos(zenith_values)

	x_train,x_test,y_train,y_test = get_data(images,cos_zenith_values)

	model.fit(x_train,y_train,
		batch_size = batch_size,
		epochs = epochs,
		verbose = 1,
		validation_data = (x_test,y_test),
		callbacks = [early_stop,best_model])
	model.save(output_file)
