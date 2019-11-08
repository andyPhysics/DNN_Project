#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys,os
from I3Hexagon import I3Hexagon
from scipy import stats

def log_function(data):
    len1 = data.shape[0]
    len2 = data.shape[1]
    result = np.zeros([len1,len2])
    for i in range(len1):
        for j in range(len2):
            if data[i][j] != 0:
                result[i][j] = abs(np.log(data[i][j]))
    return result

def min_max(data):
    len1 = data.shape[0]
    len2 = data.shape[1]
    result = np.zeros([len1,len2])
    for i in range(len1):
        for j in range(len2):
            if data[i][j] != 0:
                result[i][j] = (data[i][j] - min(data.flatten()))/(max(data.flatten())-min(data.flatten()))
    return result

def tile(data,event_number,feature):
    x = I3Hexagon()
    x.initiate_array()
    y = []
    input_data = np.array(data.loc[event_number,feature])
    if feature in [0,1,2]:
        my_data = min_max(log_function(input_data))
    else:
        my_data = min_max(input_data)
    count = 0
    for data in my_data:
        x.fill_array(data[0:78])
        y.append(x.end_array)
        x.reset_array()
    new_y = np.array(y)
    count = 0
    count_end = 6
    image = []
    while count <= len(new_y)-1:
        image.append(np.hstack(new_y[count:count_end]))
        count += 6
        count_end += 6
    image = np.array(image)
    image = np.vstack(image)
    return image

def tile3D(data,event_number,feature):
    x = I3Hexagon()
    x.initiate_array()
    y = []
    input_data = np.array(data.loc[event_number,feature])
    if feature in [0,1,2]:
        my_data = min_max(log_function(input_data))
    else:
        my_data = min_max(input_data)
    count = 0
    for data in my_data:
        x.fill_array(data[0:78])
        y.append(x.end_array)
        x.reset_array()
    new_y = np.array(y)
    image = np.array(new_y)
    return image

def tile_simple(data,event_number_feature):
    y = []
    input_data = np.array(data.loc[event_number,feature])
    if feature in [0,1,2]:
        my_data = min_max(log_function(input_data))
    else:
        my_data = min_max(input_data)
    for data in my_data:
        y.append(data)
    new_y = np.array(y)
    image = np.array(new_y)
    return image


def entire_image(data,event_number):
    x = range(0,9)
    image = []
    for i in x:
        tile_preprocessed = tile(data,event_number,i)
        tile_done = tile_preprocessed
        image.append(tile_done)
    image = np.array(image)
    new_image = []
    count = 0
    count_end = 3
    while count <= len(image)-1:
        new_image.append(np.hstack(image[count:count_end]))
        count += 3
        count_end += 3
    new_image = np.array(new_image)
    new_image = np.vstack(new_image)
    return np.array(new_image)

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects
        
def entire_image_3D(data,event_number):
    x = range(0,9)
    image = []
    for i in x:
        tile_preprocessed = tile3D(data,event_number,i)
        tile_done = tile_preprocessed
        image.append(tile_done)
    new_image = init_list_of_objects(60)
    for i in new_image:
        i+=init_list_of_objects(10)
    for i in new_image:
        for j in i:
            j+=init_list_of_objects(19)

    for m in image:
        for j in range(len(m)):
            for k in range(len(m[0])):
                for l in range(len(m[0][0])):
                    new_image[j][k][l].append(m[j][k][l])
    new_image = np.array(new_image)
    return new_image
       
def entire_image_simple(data,event_number):
    x = range(0,9)
    image = []
    for i in x:
        tile_preprocessed = tile_simple(data,event_number,i)
        tile_done = tile_preprocessed
        image.append(tile_done)
    new_image = init_list_of_objects(60)
    for i in new_image:
        i+=init_list_of_objects(10)
    for i in new_image:
        for j in i:
            j+=init_list_of_objects(19)

    for m in image:
        for j in range(len(m)):
            for k in range(len(m[0])):
                for l in range(len(m[0][0])):
                    new_image[j][k][l].append(m[j][k][l])
    new_image = np.array(new_image)
    return new_image



    


    

