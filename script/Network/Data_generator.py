from keras.utils import Sequence
import numpy as np
import sys,os

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        x = np.load(i,allow_pickle=True,encoding='latin1').item()
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

def get_cos_values(zenith,azimuth):
    cos1 = []
    cos2 = []
    cos3 = []
    for i,j in zip(zenith,azimuth):
        cos1.append(np.sin(i) * np.cos(j))
        cos2.append(np.sin(i) * np.sin(j))
        cos3.append(np.cos(i))
    return np.array(cos1),np.array(cos2),np.array(cos3)


class Data_generator(Sequence):

    def __init__(self,directory,batch_size,activation_function='sigmoid',percent=1,shuffle=False,first_iter=False,augmentations=None):
        y = os.listdir(directory)
        self.files = []
        for i in y:
            self.files.append(directory+i)
        self.files = self.files[0:int(np.ceil(len(self.files)*percent))]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.activation_function = activation_function
        self.on_epoch_end()
        self.first_iter = first_iter
        self.augment = augmentations

    def __len__(self):
        return int(np.floor(len(self.files)/float(self.batch_size)))

    def __getitem__(self,index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.files[k] for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)
        
        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,self_IDs_temp):
        images,labels = load_files(self_IDs_temp)
        zenith_values = get_feature(labels,1)
        azimuth_values = get_feature(labels,2)
        if self.first_iter == True:
            import random
            check = list(zip(zenith_values,azimuth_values,images))
            new_values = []
            z = max(zenith_values)
            for i in check:
                if i[0] > z/2.13:
                    new_values.append(i)
                else:
                    n = abs(np.random.poisson())
                    if n < 1.0:
                        new_values.append(i)
                    else:
                        continue
            zenith_values = np.array(list(zip(*new_values))[0])
            azimuth_values = np.array(list(zip(*new_values))[1])
            images = np.array(list(zip(*new_values))[2],dtype=np.uint8)
        azimuth_values_new = [(i-np.pi)/np.pi for i in azimuth_values]
        cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values)
        cos_values = np.array(list(zip(cos1,cos2,cos3,azimuth_values_new)))

        return images,cos_values
        
        
        


    

