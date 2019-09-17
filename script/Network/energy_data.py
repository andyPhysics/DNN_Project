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

cnn_best = '/home/amedina/DNN_Project/script/Network/cnn_energy.h5'  
output_best='/home/amedina/DNN_Project/script/Network/energy_model.h5'

file_path = '/data/user/amedina/DNN/'
y = os.listdir(file_path+'processed_simple')
file_names = []

for i in y:
    file_names.append(file_path+'processed_simple/'+i)

file_names_batched = list(np.array_split(file_names,50))

images,labels = load_files(file_names_batched[30])

images1 = images[:,:,:,:]


energy = get_feature(labels,0)
log_energy = np.log10(energy)

from keras.models import load_model
import tensorflow as tf
from keras import backend as K

def predict_images(model_name,images):
    model = load_model(model_name)
    predicted_log10 = model.predict(images)
    return predicted_log10

cnn_model = load_model(cnn_best)
model = load_model(output_best,custom_objects={'cnn_model':cnn_model})
energy_pred = model.predict(images1)

all_values = list(zip(log_energy,energy_pred))

np.savetxt('output_energy.csv',all_values,delimiter=',')
