import numpy as np
import pandas as pd
import sys,os

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        print('Loading File: ' + i)
        x = np.load(i,allow_pickle=True)['arr_0'].item()
        keys = x.keys()
        for key in keys:
            images.append([x[key][0][0]])
            labels.append(x[key][1])
    return np.array(images),np.array(labels)

def get_feature(labels,feature):
    feature_values = []
    for i in labels:
        feature_values.append(i[feature])
    feature_values = np.array(feature_values)
    return feature_values

def get_cuts(labels):
    feature_values = []
    for i in labels:
        try:
            feature_values.append(i[10])
        except:
            feature_values.append(0)
    feature_values=np.array(feature_values)
    return feature_values


def get_cos_values(zenith,azimuth,activation):
    cos1 = []
    cos2 = []
    cos3 = []
    if activation == 'tanh':
        for i,j in zip(zenith,azimuth):
            cos1.append(np.sin(i) * np.cos(j))
            cos2.append(np.sin(i) * np.sin(j))
            cos3.append(np.cos(i))
    elif activation == 'sigmoid':
        for i,j in zip(zenith,azimuth):
            cos1.append((np.sin(i) * np.cos(j)+1.0)/2.0)
            cos2.append((np.sin(i) * np.sin(j)+1.0)/2.0)
            cos3.append((np.cos(i)+1.0)/2.0)

    return np.array(cos1),np.array(cos2),np.array(cos3)


cnn = '/data/user/amedina/cnn_model_high.h5'
model = '/data/user/amedina/model_high.h5'

output_file = 'output_high'
up = 2

file_path = '/data/user/amedina/DNN/processed_simple/validation/'
y = os.listdir(file_path)
file_names = []

for i in y:
    file_names.append(file_path+i)

#file_names_batched = list(np.array_split(file_names,50))

images,labels = load_files(file_names)
energy = get_feature(labels,0)
pre_zenith_values = get_feature(labels,1)
pre_azimuth_values = get_feature(labels,2)
pre_line_fit_az = get_feature(labels,8)
pre_line_fit_zen = get_feature(labels,9)
line_fit_status = get_cuts(labels)

check_zip = list(zip(images,pre_zenith_values,pre_azimuth_values,pre_line_fit_az,pre_line_fit_zen,line_fit_status,energy))

def get_values(check_zip,up):
    zenith_values = []
    azimuth_values = []
    line_fit_az = []
    line_fit_zen = []
    energy = []
    new_images = []
    status = []
    for i in check_zip:
        if up ==0:
            new_images.append(i[0])
            zenith_values.append(i[1])
            azimuth_values.append(i[2])
            line_fit_az.append(i[3])
            line_fit_zen.append(i[4])
            status.append(i[5])
            energy.append(i[6])
        if up ==1:
            if np.log10(i[6]) < 5.5:
                new_images.append(i[0])
                zenith_values.append(i[1])
                azimuth_values.append(i[2])
                line_fit_az.append(i[3])
                line_fit_zen.append(i[4])
                status.append(i[5])
                energy.append(i[6])
        if up ==2:
            if np.log10(i[6]) >= 5.5:
                new_images.append(i[0])
                zenith_values.append(i[1])
                azimuth_values.append(i[2])
                line_fit_az.append(i[3])
                line_fit_zen.append(i[4])
                status.append(i[5])
                energy.append(i[6])
    new_images = np.array(new_images)
    zenith_values = np.array(zenith_values)
    azimuth_values = np.array(azimuth_values)
    line_fit_az = np.array(line_fit_az)
    line_fit_zen = np.array(line_fit_zen)
    energy = np.array(energy)
    status = np.array(status)

    cos1_line,cos2_line,cos3_line = get_cos_values(line_fit_zen,line_fit_az,'tanh')
    cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values,'tanh')
    cos_values = np.array(list(zip(cos1,cos2,cos3)))
    cos_values_line = np.array(list(zip(cos1_line,cos2_line)))

    return new_images,cos_values,cos_values_line,cos3_line,energy,status


new_images,cos_values,cos_values_line,cos3_line,energy,status= get_values(check_zip,up)



from keras.models import load_model
import tensorflow as tf
from keras import backend as K

def predict_images(model_name,images,cos_values_line,cos3_line):
    model = load_model(model_name)
    predicted_cos_values = model.predict([images,cos_values_line,cos3_line])
    return predicted_cos_values

def loss_space_angle(y_true,y_pred):
    subtraction = tf.math.subtract(y_true,y_pred)
    y = tf.matrix_diag_part(K.dot(subtraction,K.transpose(subtraction)))
    loss = tf.math.reduce_mean(y)
    return loss

cnn_model = load_model(cnn)
model = load_model(model,custom_objects={'cnn_model':cnn_model,'loss_space_angle'=loss_space_angle})
cos_values_pred = model.predict([new_images,cos_values_line,cos3_line])

all_values = (cos_values,cos_values_pred,energy,status)



np.savez(output_file,all_values,delimiter=',')
