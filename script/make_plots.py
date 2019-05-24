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

print('Loading Model')
from keras.models import load_model
model = load_model('/users/PAS1495/amedina/work/project/script/Network/best_model_regression.h5')

print('Predicting Zenith')
predicted_zenith = model.predict(images)
true_zenith = []
true_energy = []

print(predicted_zenith)

for i in labels:
    true_zenith.append(i[1])
    true_energy.append(i[0])
print(true_zenith)

delta_zenith = []
for i,j in zip(predicted_zenith,true_zenith):
    delta_zenith.append(abs(i-j))

print('Plotting Figure')
plt.figure(1)
plt.hist2d(true_energy,delta_zenith)
plt.xlabel('Energy(GeV)')
plt.ylabel('Zenith Error(Rad)')
plt.savefig('Angular_resolution2_heat.png')


