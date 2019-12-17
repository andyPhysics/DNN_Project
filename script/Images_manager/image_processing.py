#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys,os
from I3Hexagon import I3Hexagon
from create_images import *
import h5py

input1 = sys.argv[1]
input2 = sys.argv[2]
file_path = '/data/user/amedina/DNN/'

file_names = [input1]


image_files = []
label_files= []

for i in file_names:
    image_files.append(file_path+'data/'+i+'.hdf5')
    label_files.append(file_path+'labels/'+i+'_labels.hdf5')

number = [int(input2)]
data_files = list(zip(image_files,label_files,number))

list_of_images = {}
print(image_files[0],label_files[0])
file1 = pd.read_hdf(image_files[0])
file2 = pd.read_hdf(label_files[0])
x = []
y = set(list(zip(*list(file1.index)))[0])
for i in y:
    x.append(i)

def make_files(j):
    j1 = j.split(" ")[-1]
    results = [j, np.array([entire_image_3D(file1,j),np.array(file2.loc[int(j1)-1])])]
    return results

import multiprocessing
from multiprocessing import Pool

pool = Pool(multiprocessing.cpu_count())                         
values = pool.map(make_files,x) 
for i in values:
    list_of_images[i[0]] = i[1]
np.savez('/data/user/amedina/'+'processed_simple/'+'images_%s.npz'%(number[0]),list_of_images)



    


    

