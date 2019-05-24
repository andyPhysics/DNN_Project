#!/usr/bin/env bash
i=0

for file in /data/user/amedina/DNN/data/*.hdf5
do 
    
    path=${file##*/}
    base=${path%.hdf5}
    echo $base $i
    ((++i))
done
