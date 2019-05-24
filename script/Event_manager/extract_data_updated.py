#!/usr/bin/env python

# Read IceCube files and create training file (hdf5)
#   Modified from code written by Claudio Kopper
#   get_observable_features = access data from IceCube files
#   read_files = read in files and add truth labels
#   Can take 1 or multiple files
#   Input:
#       -i input: name of input file, include path
#       -n name: name for output file, automatically puts in my scratch
#  NOTE: need to change output path on line 226
##############################
#Author: Jessie Micallef
## ASSUMPTIONS: All strings >=79 are DeepCore

import numpy
import h5py
import argparse

from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units

from collections import OrderedDict
import itertools
import random

import sys,os

## Create ability to change settings from terminal ##
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input",type=str,default='Level5_IC86.2013_genie_numu.014640.00000?.i3.bz2',
                    dest="input_file", help="name of the input file")
parser.add_argument("-n", "--name",type=str,default='Level5_IC86.2013_genie_numu.014640.00000X',
                    dest="output_name",help="name for output file (no path)")
args = parser.parse_args()
input_file = args.input_file
output_name = args.output_name


def get_observable_features(frame):
    """
    Load observable features from IceCube files
    Receives:
        frame = IceCube object type from files
    Returns:
        observable_features: Observables dictionary
    """

    ice_pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'InIcePulses')

#Look inside ice pulses and get stats on charges and time
    store_string = []

    IC_strings = range(1,87,1)
    
# Variable information: sum charges, sum charge <500ns, sum charge <100ns, time first pulse, time when 20 % of charge, time when 50% charge, Time of last pulse, Charge weighted mean time of pulses, Charge weighted standard deviation of pulse times
# So far only looking at the DOMs that are not in deep core which is for strings greater than or equal to 79

    array_IC = numpy.zeros([9,60,len(IC_strings)])
    count_outside = 0
    charge_outside = 0

    for omkey, pulselist in ice_pulses:
        dom_index =  omkey.om-1
        string_val = omkey.string
        timelist = []
        chargelist = []

        string_index = IC_strings.index(string_val)

        for pulse in pulselist:
            
            timelist.append(pulse.time)
            chargelist.append(pulse.charge)

            count_outside +=1
            charge_outside += pulse.charge
                

        charge_array = numpy.array(chargelist)
        time_array = numpy.array(timelist)
        time_shifted = [ (t_value - time_array[0]) for t_value in time_array ]
        time_shifted = numpy.array(time_shifted)
        mask_500 = time_shifted<500
        mask_100 = time_shifted<100

        # Check that pulses are sorted
        for i_t,time in enumerate(time_array):
            assert time == sorted(time_array)[i_t], "Pulses are not pre-sorted!"

        # Find time when 20% and 50% of charge has hit DOM
        sum_charge = numpy.cumsum(chargelist)
        flag_20p = False
        for sum_index,current_charge in enumerate(sum_charge):
            if charge_array[-1] == 0:
                time_20p = 0
                time_50p = 0
                break
            if current_charge/float(charge_array[-1]) > 0.2 and flag_20p == False:
                time_20p = sum_index
                flag_20p = True
            if current_charge/float(charge_array[-1]) > 0.5:
                time_50p = sum_index
                break
            
        weighted_avg_time = numpy.average(time_array,weights=charge_array)
        weighted_std_time = numpy.sqrt( numpy.average((time_array - weighted_avg_time)**2, weights=charge_array) )

        array_IC[0,dom_index,string_index] = sum(chargelist)
        array_IC[1,dom_index,string_index] = sum(charge_array[mask_500])
        array_IC[2,dom_index,string_index] = sum(charge_array[mask_100])
        array_IC[3,dom_index,string_index] = time_array[0]
        array_IC[4,dom_index,string_index] = time_array[time_20p]
        array_IC[5,dom_index,string_index] = time_array[time_50p]
        array_IC[6,dom_index,string_index] = time_array[-1]
        array_IC[7,dom_index,string_index] = weighted_avg_time
        array_IC[8,dom_index,string_index] = weighted_std_time
        
        
    return array_IC 

def read_files(filename_list, drop_fraction_of_tracks=0.88, drop_fraction_of_cascades=0.00):
    """
    Read list of files, make sure they pass L5 cuts, create truth labels
    Receives:
        filename_list = list of strings, filenames to read data from
        drop_fraction_of_tracks = how many track events to drop
        drop_fraction_of_cascades = how many cascade events to drop
                                --> track & cascade not evenly simulated
    Returns:
        output_features = dict from observable features, passed to here
        output_labels = dict with output labels
    """
    output_features_IC = []
    output_labels = []

    for event_file_name in filename_list:
        print("reading file: {}".format(event_file_name))
        event_file = dataio.I3File(event_file_name)
        
        event_number = 0

        while event_file.more():
            frame = event_file.pop_physics()

            # some truth labels (we do *not* have these in real data and would like to figure out what they are)
            truth_labels = dict(
#                neutrino = frame["trueNeutrino"],
#                muon = frame['trueMuon'],
#                cascade = frame['trueCascade'],
                nu_x = frame["MCPrimary"].pos.x,
                nu_y = frame["MCPrimary"].pos.y,
                nu_z = frame["MCPrimary"].pos.z,
                nu_zenith = frame["MCPrimary"].dir.zenith,
                nu_azimuth = frame["MCPrimary"].dir.azimuth,
                nu_energy = frame["MCPrimary"].energy,
                nu_time = frame["MCPrimary"].time,
                track_length = frame["MCPrimary"].length,
                isTrack = frame["is_track"],   # it is a cascade with a track
                isCascade = frame["is_cascade"], # it is just a cascade
                isOther = frame["is_track"] and frame["is_cascade"] # it is something else (should not happen)
            )

            # input file sanity check: this should not print anything since "isOther" should always be false
            if truth_labels['isOther']:
                print(frame['I3MCWeightDict'])

            # Decide how many track events to keep
            if truth_labels['isTrack'] and random.random() < drop_fraction_of_tracks:
                continue

            # Decide how many cascade events to keep
            if truth_labels['isCascade'] and random.random() < drop_fraction_of_cascades:
                continue

            # Only look at "low energy" events for now
            #if truth_labels["nu_energy"] > 60.0:
            #    continue
            
            # Cut to only use events with true vertex in DeepCore
            #radius = 90
            #x_origin = 54
            #y_origin = -36
            #shift_x = truth_labels["nu_x"] - x_origin
            #shift_y = truth_labels["nu_y"] - y_origin
            #z_val = truth_labels["nu_z"]
            #radius_calculation = numpy.sqrt(shift_x**2+shift_y**2)
            #if( radius_calculation > radius or z_val > 192 or z_val < -505 ):
            #    continue

            
            # regression variables
            output_labels.append( numpy.array([ float(truth_labels['nu_energy']),float(truth_labels['nu_zenith']),float(truth_labels['nu_azimuth']),float(truth_labels['nu_time']),float(truth_labels['track_length']),float(truth_labels['nu_x']),float(truth_labels['nu_y']),float(truth_labels['nu_z']) ]) )

            IC_array  = numpy.array(get_observable_features(frame))

            output_features_IC.append(IC_array.reshape([IC_array.shape[0]*IC_array.shape[1],IC_array.shape[2]]))
            


        # close the input file once we are done
        del event_file

    output_features_IC=numpy.asarray(output_features_IC)
    output_labels=numpy.asarray(output_labels)

    return output_features_IC, output_labels

#Construct list of filenames
import glob
import pandas as pd

file_name = input_file

event_file_names = sorted(glob.glob(file_name))
assert event_file_names,"No files loaded, please check path."

#Call function to read and label files
#Currently set to ONLY get track events, no cascades!!! #
features_IC, labels = read_files(event_file_names, drop_fraction_of_tracks=0.0,drop_fraction_of_cascades=1.0)

features_IC_output = features_IC.reshape([features_IC.shape[0]*features_IC.shape[1],features_IC.shape[2]])


#This is some code to put all the data into a dataframe that could prove useful for analyzing data

Event_index = []
String_index = range(1,87,1)
DOM_index = []
Feature_index = []

for event in range(features_IC.shape[0]):
    event_list = ["Event Number %s"%(event+1)]*features_IC.shape[1]
    Event_index += event_list

for i in range(9*features_IC.shape[0]):
    doms = range(1,61,1)
    DOM_index += doms

for g in range(features_IC.shape[0]):
    feature_list = []
    for f in range(9):
        feature = [f]*60
        Feature_index += feature
    
mux = pd.MultiIndex.from_arrays([Event_index,
                                 Feature_index,
                                  DOM_index], 
                                  names=['Event Number','Feature','DOM'])

df = pd.DataFrame(features_IC_output, index=mux, columns=String_index)
df1 = pd.DataFrame(labels)

#Save output to hdf5 file
## HARDCODED OUTPUT PATH! ##
output_path = "/data/user/amedina/DNN/" + output_name + ".hdf5"

df.to_hdf(output_path, key='df', mode='w')
df1.to_hdf("/data/user/amedina/DNN/" + output_name + '_labels.hdf5',key='df_labels',mode='w')
#data_frame.to_hdf(output_path,format='table',data_columns=True)
