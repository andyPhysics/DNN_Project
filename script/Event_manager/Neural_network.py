import numpy as np
import pandas as pd
import sys,os


file_path = '/data/user/amedina/DNN/images/'
y = os.listdir(file_path)

file_list = []

for i in y:
    file_list.append(file_path + i)

print(len(file_list))
print(file_list[0])


images = []
labels = []
count = 0
for i in file_list:
    x = pd.read_pickle(i)
    print(count)
    count+=1
    for key in x.keys():
        images.append(np.array(x[key][0]))
        labels.append(np.array(x[key][1]))
    if count==167:
        break


images = np.array(images)
labels = np.array(labels)

labels_zipped = zip(*labels)


zenith_label = []

for i in labels_zipped[1]:
    if i < (2.0 * np.pi/180.0):
        zenith_label.append(np.array([1,0]))
    elif i > (88.0 * np.pi/180.0):
        zenith_label.append(np.array([1,0]))
    else:
        zenith_label.append(np.array([0,1]))

zenith_label = np.array(zenith_label)

#Plotting histogram of data to see what is available
#import matplotlib
#matplotlib.use('AGG')
#import matplotlib.pyplot as plt

#plt.hist(labels_zipped[2],100)
#plt.savefig('Zenith.png')


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , SeparableConv2D , GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

x_train, x_test , y_train , y_test = train_test_split(images,zenith_label,test_size = 0.2 , random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape([x_train.shape[0],x_train.shape[1],x_train.shape[2],1])
x_test = x_test.reshape([x_test.shape[0],x_test.shape[1],x_test.shape[2],1])
#y_train = y_train.reshape((1,) + y_train.shape)
#y_test = y_test.reshape((1,) + y_test.shape)


batch_size = 128
epochs=12

img_rows, img_cols = 300,342
input_shape = (img_rows, img_cols)

model = Sequential()
model.add(SeparableConv2D(32,(3,3),activation='relu',
                          input_shape = (img_rows,img_cols,1)))
model.add(SeparableConv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(SeparableConv2D(64,(3,3),activation='relu'))
model.add(SeparableConv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.5))

model.add(SeparableConv2D(64,(3,3),activation='relu'))
model.add(SeparableConv2D(128,(3,3),activation='relu'))
model.add(GlobalAveragePooling2D())

model.add(Dense(32,activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('My_model.h5')


