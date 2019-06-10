#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys,os
from I3Hexagon import I3Hexagon
from create_images import *
import h5py

file_path = '/fs/scratch/PAS1495/amedina/'
y = os.listdir(file_path+'images')
file_names = []
for i in y:
    file_name,file_extension = os.path.splitext(i)
    file_names.append(file_name)

image_files = []
label_files= []

for i in file_names:
    image_files.append(file_path+'images/'+i+'.hdf5')
    label_files.append(file_path+'labels/'+i+'_labels.hdf5')

number = range(len(image_files))

data_files = list(zip(image_files,label_files,number))

def image_processing(i):
    list_of_images = {}
    print(i[0],i[1])
    file1 = pd.read_hdf(i[0])
    file2 = pd.read_hdf(i[1])
    x = list(set(zip(*list(file1.index))[0]))
    for j in x:
        list_of_images[j] = [entire_image_3D(file1,j),np.array(file2.loc[int(filter(str.isdigit, j))-1])]
    np.save(file_path+'processed_3D/'+'images_%s.npy'%(i[2]),list_of_images)

import multiprocessing
from multiprocessing import Pool

if __name__ == '__main__':
    pool = Pool(multiprocessing.cpu_count())                         
    pool.map(image_processing, data_files) 
       



    


    

