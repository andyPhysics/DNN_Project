from keras.utils import Sequence
import numpy as np
import sys,os

def load_files(batch):
    images = []
    labels = []
    for i in batch:
        x = np.load(i,allow_pickle=True,encoding='latin1')['arr_0'].item()
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


class Data_generator(Sequence):

    def __init__(self,directory,batch_size,activation_function='sigmoid',percent=1.0,shuffle=False,first_iter=False,augmentations=None):
        y = os.listdir(directory)
        self.files = []
        for i in y:
            self.files.append(directory+i)
        self.files = np.array(self.files)
        self.batch_size = batch_size
        self.files_split = np.array_split(self.files,np.ceil(len(self.files)/self.batch_size))
        self.shuffle = shuffle
        self.activation_function = activation_function
        self.on_epoch_end()
        self.first_iter = first_iter
        self.augment = augmentations

    def __len__(self):
        #length = int(np.ceil(len(self.files)/float(self.batch_size)))
        length = len(self.files_split)
        return length

    def __getitem__(self,index):

        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #list_IDs_temp = [self.files[k] for k in indexes]
        list_IDs_temp = self.files_split[index]

        X, Y = self.__data_generation(list_IDs_temp)
        
        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files_split))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,self_IDs_temp):
        images,labels = load_files(self_IDs_temp)
        pre_zenith_values = get_feature(labels,1)
        pre_azimuth_values = get_feature(labels,2)
        pre_line_fit_az = get_feature(labels,8)
        pre_line_fit_zen = get_feature(labels,9)
        line_fit_status = get_cuts(labels)
        check_zip = list(zip(pre_zenith_values,pre_azimuth_values,pre_line_fit_az,pre_line_fit_zen,line_fit_status))
        
        zenith_values = []
        azimuth_values = []
        line_fit_az = []
        line_fit_zen = []
        
        for i in check_zip:
            zenith_values.append(i[0])
            azimuth_values.append(i[1])
            line_fit_az.append(i[2])
            line_fit_zen.append(i[3])
                        
        zenith_values = np.array(zenith_values)
        azimuth_values = np.array(azimuth_values)
        line_fit_az = np.array(line_fit_az)
        line_fit_zen = np.array(line_fit_zen)

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
        cos1_line,cos2_line,cos3_line = get_cos_values(line_fit_zen,line_fit_az,self.activation_function)
        cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values,self.activation_function)
        cos_values = np.array(list(zip(cos1,cos2,cos3)))
        cos_values_line = np.array(list(zip(cos1_line,cos2_line,cos3_line)))
        return [images,cos_values_line],cos_values
        
        
        


    

