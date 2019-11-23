import numpy as np
import pandas as pd
import sys,os

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

cnn_best = '/home/amedina/DNN_Project.git/trunk/script/Network/cnn_model_all.h5'  
output_best='/home/amedina/DNN_Project.git/trunk/script/Network/model_all_best.h5'
output_file = 'output_new'


file_path = '/data/user/amedina/DNN/processed_simple/validation/'
y = os.listdir(file_path)
file_names = []

for i in y:
    file_names.append(file_path+i)

#file_names_batched = list(np.array_split(file_names,50))

images,labels = load_files(file_names)

images1 = images[:,:,:,:]

zenith_values = get_feature(labels,1)
azimuth_values = get_feature(labels,2)

zenith_check = list(zip(images1,zenith_values,azimuth_values))
new_images = []
new_zenith_values = []
new_azimuth_values = []

for i in zenith_check:
    if i[1] != None:
        new_images.append(i[0])
        new_zenith_values.append(i[1])
        new_azimuth_values.append(i[2])

new_images = np.array(new_images)

cos1,cos2,cos3=get_cos_values(new_zenith_values,new_azimuth_values)

cos_values = np.array(list(zip(cos1,cos2,cos3)))

from keras.models import load_model
import tensorflow as tf
from keras import backend as K

def predict_images(model_name,images):
    model = load_model(model_name)
    predicted_cos_values = model.predict(images)
    return predicted_cos_values

def loss_space_angle(y_true,y_pred):
    subtraction = tf.math.subtract(y_true,y_pred)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

cnn_model = load_model(cnn_best)
model = load_model(output_best,custom_objects={'cnn_model':cnn_model})
cos_values_pred = model.predict(new_images)

cos_values_1_1 = [(i*np.pi + np.pi) for i in list(zip(*cos_values_pred)[0])]
cos_values_1_2 = [ (i*np.pi + np.pi) for i in list(zip(*cos_values_pred)[1])]
cos_values_1_3 = [(i*np.pi + np.pi) for i in list(zip(*cos_values_pred)[2])]

cos_values_1=zip(*cos_values)[0]
cos_values_2=zip(*cos_values)[1]
cos_values_3=zip(*cos_values)[2]

def azimuth(sincos,sinsin):
    values = []
    for i,j in zip(sinsin,sincos):
        if i > 0:
            if j > 0:
                values.append(np.arctan(i/j))
            if j < 0:
                values.append(np.arctan(i/j)+2.0*np.pi)
        if i < 0:
            if j > 0:
                values.append(np.arctan(i/j)+np.pi)
            if j < 0:
                values.append(np.arctan(i/j)+np.pi)
    return values

azimuth_predicted = azimuth(list(cos_values_1_1),list(cos_values_1_2))

zenith_predicted = np.arccos(cos_values_1_3)  

def space_angle_error(variable1,variable2):
    x = []
    for i,j in zip(variable1,variable2):
        magnitude1 = (i[0]**2.0+i[1]**2.0+i[2]**2.0)**0.5
        magnitude2 = (j[0]**2.0+j[1]**2.0+j[2]**2.0)**0.5
        dot_product = (i[0]*j[0]+i[1]*j[1]+i[2]*j[2])
        error = np.arccos(dot_product/(magnitude1*magnitude2))
        x.append(error)
    return x,magnitude1,magnitude2

value1 = zip(cos_values_1,cos_values_2,cos_values_3)
value2 = zip(cos_values_1_1,cos_values_1_2,cos_values_1_3)
value1_predicted=value2
error,mag1,mag2 = space_angle_error(value1,value2)

#all_values = zip(new_zenith_values,new_azimuth_values,azimuth_predicted,zenith_predicted,error)
all_values = (value1,value1_predicted,error)
np.savez(output_file,all_values,delimiter=',')
