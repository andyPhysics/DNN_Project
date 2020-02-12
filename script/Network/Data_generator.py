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
            values = np.array(x[key][0])
            images.append(values)
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
    for i,j in zip(zenith,azimuth):
        cos1.append(i/np.pi)
        if j < np.pi:
            cos2.append(j/np.pi)
            cos3.append(0)
        elif j >= np.pi:
            cos2.append((j-np.pi)/np.pi)
            cos3.append(1)
    
    return np.array(cos1),np.array(cos2),np.array(cos3)


class Data_generator(Sequence):

    def __init__(self,directory,batch_size,activation_function='sigmoid',percent=1.0,shuffle=False,first_iter=False,augmentations=None,up = 0):
        y = os.listdir(directory)
        self.files = []
        import random
        random.seed(10)

        for i in y:
            if random.uniform(0,1) < percent:
                self.files.append(directory+i)

        self.files = np.array(self.files)
        self.batch_size = batch_size
        self.files_split = np.array_split(self.files,np.ceil(len(self.files)/self.batch_size))
        self.shuffle = shuffle
        self.activation_function = activation_function
        self.on_epoch_end()
        self.first_iter = first_iter
        self.augment = augmentations
        self.up = up

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
        energy = get_feature(labels,0)
        check_zip = list(zip(pre_zenith_values,pre_azimuth_values,pre_line_fit_az,pre_line_fit_zen,line_fit_status,energy))
        
        zenith_values = []
        azimuth_values = []
        line_fit_az = []
        line_fit_zen = []
        line_fit_stat = []
        energy_new = []
        
        for i in check_zip:
            zenith_values.append(i[0])
            azimuth_values.append(i[1])
            line_fit_az.append(i[2])
            line_fit_zen.append(i[3])
            line_fit_stat.append(i[4])
            energy_new.append(i[5])
                        
        zenith_values = np.array(zenith_values)
        azimuth_values = np.array(azimuth_values)
        line_fit_az = np.array(line_fit_az)
        line_fit_zen = np.array(line_fit_zen)
        line_fit_stat = np.array(line_fit_stat)
        energy_new = np.array(energy_new)

        if self.up == 0:
            check = list(zip(zenith_values,azimuth_values,line_fit_az,line_fit_zen,images,line_fit_stat,energy_new))
            new_values = []
            for i in check:
                new_values.append(i)
               
            zenith_values = np.array(list(zip(*new_values))[0])
            azimuth_values = np.array(list(zip(*new_values))[1])
            images = np.array(list(zip(*new_values))[4],dtype=np.uint8)
            line_fit_az = np.array(list(zip(*new_values))[2])
            line_fit_zen = np.array(list(zip(*new_values))[3])


        elif self.up == 1:
            check = list(zip(zenith_values,azimuth_values,line_fit_az,line_fit_zen,images,line_fit_stat,energy_new))
            new_values = []
            for i in check:
                if np.log10(i[6]) < 5: 
                    new_values.append(i)
                
            zenith_values = np.array(list(zip(*new_values))[0])
            azimuth_values = np.array(list(zip(*new_values))[1])
            images = np.array(list(zip(*new_values))[4],dtype=np.uint8)
            line_fit_az = np.array(list(zip(*new_values))[2])
            line_fit_zen = np.array(list(zip(*new_values))[3])

        elif self.up == 2:
            check = list(zip(zenith_values,azimuth_values,line_fit_az,line_fit_zen,images,line_fit_stat,energy_new))
            new_values = []
            for i in check:
                if np.log10(i[6]) > 5:
                    new_values.append(i)

            zenith_values = np.array(list(zip(*new_values))[0])
            azimuth_values = np.array(list(zip(*new_values))[1])
            images = np.array(list(zip(*new_values))[4],dtype=np.uint8)
            line_fit_az = np.array(list(zip(*new_values))[2])
            line_fit_zen = np.array(list(zip(*new_values))[3])

        cos1_line,cos2_line,cos3_line = get_cos_values(line_fit_zen,line_fit_az,self.activation_function)
        cos1,cos2,cos3 = get_cos_values(zenith_values,azimuth_values,self.activation_function)
        cos_values = np.array(list(zip(cos1,cos2,cos3)))
        cos_values_line = np.array(list(zip(cos1_line,cos2_line,cos3_line)))
        return [images],[cos1,cos2,cos3]
        
        
        


    

