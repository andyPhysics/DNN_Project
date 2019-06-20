import numpy as np
import pandas as pd
import sys,os

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

output_file = 'gpu_test_3D.h5'
output_best = 'gpu_test_best_3D.h5'
file_path = '/fs/scratch/PAS1495/amedina/processed_3D/'
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
        feature_values.append(i[feature])
    feature_values = np.array(feature_values)
    return feature_values

def get_cos_values(zenith,azimuth):
    cos1 = []
    cos2 = []
    cos3 = []
    for i,j in zip(zenith,azimuth):
        cos1.append(np.sin(i) * np.cos(j))
        cos2.append(np.sin(i) * np.sin(j))
        cos3.append(np.cos(i))
    return np.array(cos1),np.array(cos2),np.array(cos3)

#images,labels = load_files(file_names_batched[0])

#zenith_values = get_feature(labels,1)
#azimuth_values = get_feature(labels,2)

#cos_zenith_values = np.cos(zenith_values)

images,labels = load_files(file_names_batched[0])

images = images[:,:,:,:,4:7]

zenith_values = get_feature(labels,1)
azimuth_values = get_feature(labels,2)

cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values)

cos_values = np.array(list(zip(cos1,cos2,cos3)))

import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D , BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split,KFold

def get_data(images,values):
    x_train, x_test , y_train , y_test = train_test_split(images,values,test_size = 0.2 , random_state=42)
    return x_train,x_test,y_train,y_test

import tensorflow as tf

def loss_space_angle(y_true,y_pred):
    subtraction = tf.math.subtract(y_true,y_pred)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

x_train,x_test,y_train,y_test = get_data(images,cos_values)

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
epochs=20

img_heights,img_rows, img_cols = 60,10,19
input_shape = (img_rows, img_cols)

kernel = 3
kernel2 = 2

model = Sequential()
model.add(Conv3D(16,kernel,
                 input_shape = (img_heights,img_rows,img_cols,3),
                 padding='same',
                 kernel_regularizer=regularizers.l2(0.01)))
                #activity_regularizer=regularizers.l1(0.01)))

model.add(LeakyReLU(alpha = 0.01))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))

model.add(Dropout(rate=1-0.5))

model.add(LeakyReLU(alpha = 0.01))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))
model.add(MaxPooling3D(kernel2,padding='same'))
model.add(Conv3D(32,kernel,padding='same'))

#model.add(GlobalAveragePooling3D())
model.add(Flatten())

model.add(Dense(16,activation='linear'))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dense(3))

#opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=.09, nesterov=True)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-6, amsgrad=False)
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
	cos_values = np.cos(zenith_values)

	x_train,x_test,y_train,y_test = get_data(images,cos_values)

	model.fit(x_train,y_train,
		batch_size = batch_size,
		epochs = epochs,
		verbose = 1,
		validation_data = (x_test,y_test),
		callbacks = [early_stop,best_model])
	model.save(output_file)
