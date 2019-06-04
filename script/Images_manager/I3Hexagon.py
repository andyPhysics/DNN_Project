#This is code to define a hexagon that is going to hold one layer of an icecube event

import numpy as np
import sys,os
import h5py


class I3Hexagon:
    def __init__(self):
        self.end_array = []

    def initiate_array(self):

        #The following setup is used to create a hexagon image with numbers 1-78 filling particular bins that are similar to what IceCube has labeled. 
        self.end_array = []

        start = [[1,72],[2,73],[3,74],[4,67],[5,59],[6,50],[7,78],[14,77],[22,76],[31,75]]
        add = [0,7,8,9,10,10,10,9,8,7]

        lists = []

        for i in start:
            list_add = []
            x = i[0]
            if i[0] > 6:
                del(add[1])
            for j in add:
                if x < i[1]:
                    x += j
                    list_add.append(x)
            lists.append(list_add)

        list_sorted = [6,5,4,3,2,1,7,14,22,31]
        for i in list_sorted:
            for j in lists:
                if j[0] == i:
                    list_sorted[list_sorted.index(i)] = j

        start_index = [5,4,3,2,1,0,1,2,3,4]
        count = 0

        for i in list_sorted:
            x = np.zeros(19)
            y = start_index[count]
            for k in i:
                x[y] = k
                y += 2
            count += 1
            self.end_array.append(x)


    def fill_array(self,data):

        #This function takes in a data array that is equal to 78 and replaces the elements in self.end_array with the corresponding values. 
        if len(data) != 78:
            print("Size of data array not equal to 78")
            return

        for i in self.end_array:
            for j in i:
                if j == 0:
                    continue
                else:
                    x = np.where(self.end_array == j)
                    self.end_array[x[0][0]][x[1][0]] = data[int(j)-1]
        

    def reset_array(self):
        self.initiate_array()
    


    



