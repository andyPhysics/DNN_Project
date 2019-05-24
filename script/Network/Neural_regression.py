Bimport numpy as np
import pandas as pd
import sys,os
from create_images import entire_image

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

output_file = 'all_new2.h5'
output_best = 'all_best_new2.h5'
file_path = '/fs/scratch/PAS1495/amedina/'
y = os.listdir(file_path+'processed_new')

file_names = []

for i in y:
    file_names.append(file_path+'processed_new/'+i)

file_names_batched = list(np.array_split(file_names,5))

images = []
labels = []

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        print('Loading File: ' + i)
        x = np.load(i).item()
        keys = x.keys()
        for key in keys:
            images.append(x[key][0])
            labels.append(x[key][1])
    return np.array(images),np.array(labels)

def get_zenith(labels):
    zenith_values = []
    for i in labels:
        zenith_values.append(i[1])
    zenith_values = np.array(zenith_values)
    return zenith_values

images,labels = load_files(file_names_batched[0])

zenith_values = get_zenith(labels)

hist,bin_edges = np.hist(zenith_values,bins=1000)
hist_inverse = [1.0/i for i in hist]
hist_inverse_norm = [i/max(hist_inverse) for i in hist_inverse]

weight_comparison = []
for i in range(hist):
    weight_comparison.append([hist_inverse_norm[i],[bin_edges[i],bin_edges[i+1]]])

def weight_output(data):
    data_weights = []
    for i in data:
        for j in weight_comparison:
            if i > j[1][0] & i < j[1][1]:
                data_weights.append(j[0])
    return np.array(data_weights)

import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , SeparableConv2D , GlobalAveragePooling2D, BatchNormalization
from keras import backend as K
from sklearn.model_selection import train_test_split,KFold
from sklearn.model_selection import train_test_split

def get_data(images,zenith_values):
    x_train, x_test , y_train , y_test = train_test_split(images,zenith_values,test_size = 0.2 , random_state=42)
    x_train = x_train.reshape([x_train.shape[0],x_train.shape[1],x_train.shape[2],1])
    x_test = x_test.reshape([x_test.shape[0],x_test.shape[1],x_test.shape[2],1])
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = get_data(images,zenith_values)

weight = weight_output(y_train)

early_stop = keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                              min_delta=0,
                              patience=2,
                              verbose=1, mode='auto')

best_model = keras.callbacks.ModelCheckpoint(output_best,
                                             monitor='mean_squared_error',
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

model = Sequential()
model.add(SeparableConv2D(32,kernel,
                          input_shape = (img_rows,img_cols,1)))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(SeparableConv2D(64,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(kernel2))
model.add(Dropout(0.5))

model.add(SeparableConv2D(64,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(SeparableConv2D(128,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D(kernel2))
model.add(Dropout(0.5))

model.add(SeparableConv2D(64,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(SeparableConv2D(128,kernel))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(32,activation='linear'))
model.add(LeakyReLU(alpha = 0.01))
model.add(Dense(1))


model.compile(optimizer='adam' , loss = 'mse' , metrics = ['mse'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stop,best_model],
          sample_weight = weight)

score = model.evaluate(x_test, y_test, verbose=0)
total_score = score[0]
total_accuracy = score[1]

print('Test loss:', total_score)
print('Test accuracy:', total_accuracy)
model.save(output_file)

for i in file_names_batched[1:len(file_names_batched)]:
    model = load_model(output_best)
    images,labels = load_files(i)
    zenith_values = get_zenith(labels)

    x_train,x_test,y_train,y_test = get_data(images,zenith_values)

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stop,best_model])

    model.save(output_file)

