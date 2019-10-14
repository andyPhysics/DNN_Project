from keras.utils import Sequence
import numpy as np
import sys,os

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        x = np.load(i,allow_pickle=True,encoding='latin1').item()
#        print("Loading file: %s"%(i))
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

class Data_generator(Sequence):

    def __init__(self,directory,batch_size,shuffle=False):
        y = os.listdir(directory)
        self.files = []
        for i in y:
            self.files.append(directory+i)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

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
        energy = get_feature(labels,0)
        energy = np.log10(energy)
        return images,energy

    
