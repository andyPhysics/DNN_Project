import numpy as np
import sys,os
import matplotlib                                                                                                                                        
matplotlib.use('AGG')                                                                                                                                    
import matplotlib.pyplot as plt                                                                                                                          
import pandas as pd

file_path = '/users/PAS1495/amedina/work/project/data/'
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
    count+=1
    print(count)
    if count > 200:
        for key in x.keys():
            images.append(np.array(x[key][0]))
            labels.append(np.array(x[key][1]))
    if count == 400:
        break

images = np.array(images)
images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
labels = np.array(labels)

zenith_angles = []

for i in labels:
    zenith_angles.append(i[1])

print(zenith_angles)
