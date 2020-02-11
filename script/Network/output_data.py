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
            images.append(x[key][0])
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
    for i,j in zip(zenith,azimuth):
        cos1.append(np.sin(i) * np.cos(j))
        cos2.append(np.sin(i) * np.sin(j))
        cos3.append(np.cos(i))
#        cos1.append((np.sin(i) * np.cos(j)+1.0)/2.0)
#        cos2.append((np.sin(i) * np.sin(j)+1.0)/2.0)
#        cos3.append((np.cos(i)+1.0)/2.0)
    
    return np.array(cos1),np.array(cos2),np.array(cos3)


#cnn = '/home/amedina/DNN_Project.git/trunk/script/Network/cnn_model.h5'
model = '/home/amedina/DNN_Project.git/trunk/script/Network/model_best.h5'

output_file = 'output'
up = 0

file_path = '/data/user/amedina/DNN/processed_2D/validation/'
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
            if np.log10(i[6]) < 5:
                new_images.append(i[0])
                zenith_values.append(i[1])
                azimuth_values.append(i[2])
                line_fit_az.append(i[3])
                line_fit_zen.append(i[4])
                status.append(i[5])
                energy.append(i[6])
        if up ==2:
            if np.log10(i[6]) >= 5:
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

    cos1_line,cos2_line,cos3_line = get_cos_values(line_fit_zen,line_fit_az,'linear')
    cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values,'linear')
    cos_values = np.array(list(zip(cos1,cos2,cos3)))
    cos_values_line = np.array(list(zip(cos1_line,cos2_line,cos3_line)))

    return new_images,cos_values,cos1_line,cos2_line,cos3_line,energy,status

new_images,cos_values,cos1_line,cos2_line,cos3_line,energy,status= get_values(check_zip,up)

activation = 'linear'

def loss_space_angle(y_true,y_pred):
    #y_true1 = y_true*2.0-1.0
    #y_pred1 = y_pred*2.0-1.0
    y_true1 = y_true
    y_pred1 = y_pred
    y = tf.math.squared_difference(y_true1,y_pred1)
    loss = tf.math.reduce_mean(y)
    return loss


from keras.models import load_model
import tensorflow as tf
from keras import backend as K

model = load_model(model,custom_objects={'loss_space_angle':loss_space_angle})
cos_values_pred = model.predict([new_images,cos1_line,cos2_line,cos3_line])

all_values = {'cos_values':cos_values,'cos_values_pred':cos_values_pred,'energy':energy,'status':status}



np.savez(output_file,all_values,delimiter=',')
