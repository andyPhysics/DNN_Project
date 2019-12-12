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


#cnn_best = '/home/amedina/DNN_Project.git/trunk/script/Network/cnn_model_all.h5'  
#output_best='/home/amedina/DNN_Project.git/trunk/script/Network/model_all.h5'
cnn_up = '/home/amedina/DNN_Project.git/trunk/script/Network/cnn_model_up.h5'
model_up = '/home/amedina/DNN_Project.git/trunk/script/Network/model_up.h5'
cnn_down = '/home/amedina/DNN_Project.git/trunk/script/Network/cnn_model_down.h5'
model_down = '/home/amedina/DNN_Project.git/trunk/script/Network/model_down.h5'

output_file = 'output_new'


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
            if i[4] >= np.pi/2:
                new_images.append(i[0])
                zenith_values.append(i[1])
                azimuth_values.append(i[2])
                line_fit_az.append(i[3])
                line_fit_zen.append(i[4])
                status.append(i[5])
                energy.append(i[6])
        if up ==2:
            if i[4] < np.pi/2:
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


new_images_up,cos_values_up,cos_values_line_up,cos3_line_up,energy_up,status_up= get_values(check_zip,1)
new_images_down,cos_values_down,cos_values_line_down,cos3_line_down,energy_down,status_down = get_values(check_zip,2)



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

cnn_model = load_model(cnn_up)
model = load_model(model_up,custom_objects={'cnn_model':cnn_model})
cos_values_pred_up = model.predict([new_images_up,cos_values_line_up,cos3_line_up])

cnn_model = load_model(cnn_down)
model = load_model(model_down,custom_objects={'cnn_model':cnn_model})
cos_values_pred_down = model.predict([new_images_down,cos_values_line_down,cos3_line_down])


all_values = (cos_values_up,cos_values_down,cos_values_pred_up,cos_values_pred_down,energy_up,energy_down,status_up,status_down)



np.savez(output_file,all_values,delimiter=',')
