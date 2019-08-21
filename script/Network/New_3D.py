import numpy as np
import pandas as pd
import sys,os

from keras import backend as K

output_file = 'SWNN_1.h5'
output_best = 'SWNN_best_1.h5'
file_path = '/fs/scratch/PAS1495/amedina/processed_3D/'
y = os.listdir(file_path)

file_names = []

for i in y:
    file_names.append(file_path+i)

file_names_batched = list(np.array_split(file_names,1))

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


images,labels = load_files(file_names_batched[0])

images = images[:,:,:,:,[0,7]]

zenith_values = get_feature(labels,1)
azimuth_values = get_feature(labels,2)

cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values)

cos_values = np.array(list(zip(cos1,cos2,cos3)))
cos_values1 = np.zeros([len(cos_values),len(cos_values[0])])
for i in range(len(cos_values)):
    for j in range(len(cos_values[0])):
        cos_values1[i][j]=(cos_values[i][j] + 1.0)/2.0

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate, UpSampling3D, Concatenate, SpatialDropout3D, Lambda
from keras.layers import Conv3D, MaxPooling3D, GaussianNoise
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from keras import regularizers

def get_data(images,values):
    x_train, x_test , y_train , y_test = train_test_split(images,values,test_size = 0.005 , random_state=42)
    return x_train,x_test,y_train,y_test

tf.distribute.Strategy

def loss_space_angle(y_true,y_pred):
    y_true1 = y_true*2.0-1.0
    y_pred1 = y_pred*2.0-1.0
    subtraction = tf.math.subtract(y_true1,y_pred1)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

x_train,x_test,y_train,y_test = get_data(images,cos_values1)

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


batch_size = 16
epochs=200

img_heights,img_rows, img_cols = 60,10,19
input_shape = (img_rows, img_cols)

kernel = 3
kernel2 = 2

feature_number = 2

#------------------------------------------------------------------------------------------

model1_input = Input(shape=(img_heights,img_rows,img_cols,feature_number))

model1 = LeakyReLU(alpha = 0.01)(model1_input)
output1 = MaxPooling3D(kernel2,padding='same')(model1)
model1 = Conv3D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01))(output1)

model1 = LeakyReLU(alpha = 0.01)(model1)
output2 = MaxPooling3D(kernel2,padding='same')(model1)
model1 = Conv3D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01))(output2)

model1 = LeakyReLU(alpha = 0.01)(model1)
output3 = MaxPooling3D(kernel2,padding='same')(model1)
model1 = Conv3D(32,kernel,padding='same',kernel_regularizer=regularizers.l2(0.01))(output3)

cnn_model1 = Flatten()(model1_input)
cnn_model2 = Flatten()(model1)
cnn_model3 = Flatten()(output1)
cnn_model4 = Flatten()(output2)
cnn_model5 = Flatten()(output3)
cnn_model = Concatenate(axis=-1)([cnn_model1,cnn_model2,cnn_model3,cnn_model4,cnn_model5])

cnn_model = Model(inputs=model1_input,outputs=cnn_model)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5, amsgrad=False)
cnn_model.compile(optimizer=opt , loss = loss_space_angle)

#---------------------------------------------------------------------------------------------

input_new = Input(shape=(img_heights,img_rows,img_cols,feature_number))

output = Lambda(lambda x: cnn_model(x))(input_new)

model = Dropout(0.5)(output)
model = Dense(512)(model)
model = LeakyReLU(alpha = 0.01)(model)
model = Dropout(0.5)(model)
model = Dense(512)(model)
model = LeakyReLU(alpha = 0.01)(model)

input_new_prime = Flatten()(input_new)
model = Concatenate(axis=-1)([model, input_new_prime])

predictions = Dense(3)(model)

model = Model(inputs=input_new,outputs=predictions)
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-5, amsgrad=False)
model.compile(optimizer=opt , loss = loss_space_angle)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[best_model])

cnn_model.save('cnn_model_1.h5')
model.save(output_file)

#cnn_model = load_model('/users/PAS1495/amedina/work/DNN_Project/script/Network/cnn_model_1.h5',custom_objects={'loss_space_angle':loss_space_angle})


#for i in file_names_batched[1:len(file_names_batched)]:
#	model = load_model(output_best,custom_objects={'cnn_model':cnn_model,'loss_space_angle':loss_space_angle})
#	images,labels = load_files(i)
#        images = images[:,:,:,:,[0,7]]

#        zenith_values = get_feature(labels,1)
#        azimuth_values = get_feature(labels,2)

#        cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values)

#        cos_values = np.array(list(zip(cos1,cos2,cos3)))
#	x_train,x_test,y_train,y_test = get_data(images,cos_values)

#	model.fit(x_train,y_train,
#		batch_size = batch_size,
#		epochs = epochs,
#		verbose = 1,
#		validation_data = (x_test,y_test),
#		callbacks = [best_model])
#	model.save(output_file)
