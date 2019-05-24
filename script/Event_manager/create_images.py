#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys,os
from I3Hexagon import I3Hexagon
import pickle as pkl

#data_path = '/data/user/amedina/DNN/'

#file_names = os.listdir(data_path)

#data_files = []
#label_files = []

#for i in file_names:
#    if 'labels' in i:
#        label_files.append(i)
#    else:
#        data_files.append(i)

#data_files = sorted([data_path + i for i in data_files])
#label_files = sorted([data_path + i for i in label_files])

data_frames_labels = []
data_frames = []

data_files = ['/data/user/amedina/DNN/data/'+sys.argv[1]+'.hdf5']
label_files = ['/data/user/amedina/DNN/labels/'+sys.argv[1]+'_labels.hdf5']

number = sys.argv[2]

for i in label_files:
    data_frames_labels.append(pd.read_hdf(i))
    print("reading file %s"%(i))
    
for i in data_files:
    data_frames.append(pd.read_hdf(i))
    print("reading file %s"%(i))



np.set_printoptions(threshold=sys.maxsize)

def tile(data,event_number,feature):
    x = I3Hexagon()
    x.initiate_array()
    y = []
    my_data = np.array(data.loc[event_number,feature])
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
    max_value = max(image.flatten())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = image[i][j]/max_value
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
        

list_of_images = {}

for i in data_frames:
    x = list(set(zip(*list(i.index))[0]))
    for j in x:
        list_of_images[j] = [entire_image(i,j),np.array(data_frames_labels[0].loc[int(filter(str.isdigit, j))-1])]
 
fileObject = open('/data/user/amedina/DNN/images/'+'images_%s.pkl'%(number), 'wb')
pkl.dump(list_of_images, fileObject,pkl.HIGHEST_PROTOCOL)
fileObject.close()
       
#np.savez('/data/user/amedina/DNN/images/'+'images_%s.npz'%(number),**list_of_images)
        
#x = entire_image(data_frames[0],'Event Number 1')

#y = data_frames[0]

#print(list(set(zip(*list(y.index))[0])))

#from PIL import Image
#import PIL.ImageOps

#img = Image.fromarray(x)
#img = img.convert("L")
#inverted_image = PIL.ImageOps.invert(img)
#inverted_image.save('my2.png')


    


    

