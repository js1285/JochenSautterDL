# tools for racing

import os
# import sys
# import glob
import json
import ijson
# import datetime
import pickle


import numpy as np
import pandas as pd
from sklearn import preprocessing
from random import shuffle, seed
# import matplotlib.pyplot as plt

# global variables
seed_no = 42
 

def pickle_from_file(fname):
    
    path = os.path.join("plots/", fname+".pickle")      
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
        
    return obj
    
    
def pickle_to_file(obj, fname):
    
    path = os.path.join("plots/", fname+".pickle")      
    
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def get_run_name(prefix="run", additional=""):
    return "_".join([prefix, 
                     datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                     additional])


'''   ***************** data structure ****************
raw data:
[
    {
        "Distance": 1834.3918374754314,
        "Duration": 297.003,
        "Speed": 6.1763411058993727,
        "SpeedOverall": 3.8983736568973764,
        "SpeedBefore": 0.0,
        "RelativSpeed": 1.5843378930522984,
        "Slope": 2.5514632180345065,
        "MinSlope": -2.4840874009198859,
        "MaxSlope": 5.75172251034925,
        "Segments": [
            {
                "Distance": 65.430754,
                "StartElevation": 1350.273681640625,
                "EndElevation": 1350.8046875,
                "Slope": 0.46572029378986862,
                "Surface": "ASP"
            },

data structure after loading: tuple (params,segments),

with params array of arrays with following features, 

    'distance': 1834.3918374754314,
    'duration': 297.003,
    'speed': 6.1763411058993727,
    'speed_overall': 3.8983736568973764,
    'relative_speed': 1.5843378930522984,
    'slope': 2.5514632180345065,
    'min_slope': -2.4840874009198859,
    'max_slope': 5.75172251034925,
    'avg_height': 1000
    
and segments array of array of arrays each featuring:

    distance, 
    delta_h,      # delta height
    ASP,          # one hot asphalt
    FOR,          # one hot other surfaces
    TR2,          # ...
    PSE,          # ...
    
    where arrays in axis=1 (sections with segments lists of different lenghts) 
    are padded with zero-segment-arrays up to lenght of longest list 
    in order to get even structure (needed for batchwise processing by LSTM)
'''
    
    
# get average height of section from segments (absolute height might be a relevant for performance)
def get_avg_height(raw_line):
    
    weighted_height, cum_distance = 0, 0
    for seg in raw_line:
        weighted_height += (seg['StartElevation']+seg['EndElevation'])/2 * seg['Length']
        # weighted_height += seg['Elevation'] * seg['Length']
    
        cum_distance += seg['Length']
    
    return weighted_height/cum_distance

    
# figure out length of longest among segment lists and pad all lists with zero_segments
def zero_pad_segments(segment_list_list):

    max_len = 0
    for segment_list in segment_list_list:
        # print("number of segments", len(segment_list))
        max_len = max(max_len,len(segment_list))
    print("padding segments with zeros up to length of: ", max_len)
          
    zero_segment = np.zeros(segment_list_list[0][0].shape[0])
    for segment_list in segment_list_list:
        for i in range(max_len-len(segment_list)):
            segment_list.append(zero_segment)
        

# get features for segments, raw_line holds list of segments of one section     
def get_segment_list(raw_line, total_len):

    seg_list = []
    for raw_seg in raw_line:
        # print("raw_seg", raw_seg['Length'])
        seg_len = raw_seg['Length']
        delta_h = raw_seg['EndElevation'] - raw_seg['StartElevation']
        
        if seg_len < 0.001:
            if delta_h > 0.01:
                print("error seg_len is {}, while delta_h is {}".format(seg_len, delta_h))
        else:
            slope = (raw_seg['EndElevation'] - raw_seg['StartElevation']) / raw_seg['Length']
            relative_len = seg_len / total_len
            seg_list.append(np.array([slope, relative_len]))
    
    return seg_list
    
        # seg_arr = np.array([slope,
                            
#                       raw_seg['EndElevation'] - raw_seg['StartElevation'],
#                      1 if raw_seg['Surface'] == 'ASP' else 0,
#                      1 if raw_seg['Surface'] == 'FOR' else 0,
#                      1 if raw_seg['Surface'] == 'TR2' else 0,
#                      1 if raw_seg['Surface'] == 'DFT' else 0])
    
    
#     return [np.array([raw_seg['Slope'],
#                       raw_seg['EndElevation'] - raw_seg['StartElevation'],
#                      1 if raw_seg['Surface'] == 'ASP' else 0,
#                      1 if raw_seg['Surface'] == 'FOR' else 0,
#                      1 if raw_seg['Surface'] == 'TR2' else 0,
#                      1 if raw_seg['Surface'] == 'DFT' else 0]) 
#            for raw_seg in raw_line]    

    
# zero_mean unit variance normalisation of all features except the one-hot encoded ones
def normalise_segments(segment_list_list):
    
    # get total no of segments and mean values of distances and delta heights
    seg_no, total_dist, total_dh = 0, 0, 0
    for segment_list in segment_list_list:
        for segment in segment_list:
            seg_no +=1
            total_dist += segment[0]
            total_dh += segment[1]
    
    avg_dist = total_dist / seg_no
    avg_dh   = total_dh / seg_no
    print("avg_dist", avg_dist)
    print("avg_dh", avg_dh)  

    # get variances of distances and delta heights
    total_dist_var, total_dh_var = 0, 0
    for segment_list in segment_list_list:
        for segment in segment_list:
            total_dist_var += (segment[0] - avg_dist)**2
            total_dh_var += (segment[1] - avg_dh)**2
    
    avg_dist_var = total_dist_var / seg_no
    avg_dh_var   = total_dh_var / seg_no
    print("avg_dist_var", avg_dist_var)
    print("avg_dh_var", avg_dh_var)  
    print("avg_dist_stdev", avg_dist_var**0.5)
    print("avg_dh_stdev", avg_dh_var**0.5)  

    
    # normalise data
    for segment_list in segment_list_list:
        for segment in segment_list:
            gaga = 1
            # no zero mean, because distances always positive
            # normalisation via average distance
            segment[0] = segment[0] / (avg_dist_var**0.5)
            # normalise on zero mean / unit variance 
            # segment[0] = (segment[0] - avg_dist) / (avg_dist_var**0.5)
            segment[1] = (segment[1] - avg_dh) / (avg_dh_var**0.5)

def normalise_Y(Y):
    return 1
    

def check_data_integrity(raw_data, params, segments_arr_arr):

    error = False
    line_count = 0
    for raw_line, param, segments_arr in zip(raw_data, params, segments_arr_arr):
        line_count += 1
        section_len = 0
        for raw_segment, segment in zip(raw_line["zegments"], segments_arr):
            section_len += raw_segment["Length"]
        if abs(section_len - raw_line["SectionLength"]) > 0.01:
            print("error at section with startdistance {} for runner with \
                  reference number {}".format(raw_line["StartDistance"], 
                                              raw_line["ReferenceNumber"]))
            error = True
        # check for integrity of height...
    if error == False:
        print("data integrity checked OK for {} sections".format(line_count))
        
    return error
               
    
    
def load_data(path='data/engadin_5000.json', from_to = None, integrity_check = False):
    
    with open(path, 'r') as fh:
        raw_data = json.load(fh)
    
    # run_count = 0
    # current_runner = ''
    runner_list = []
    param_arr_list, segment_list_list, Y_list = [], [], []    
    for raw_line in (raw_data[from_to[0]:from_to[1]] if from_to != None else raw_data):
        if not raw_line['ReferenceNumber'] in runner_list:
            runner_list.append(raw_line['ReferenceNumber'])
        # if current_runner != raw_line['ReferenceNumber']:
        #    run_count += 1
        #    current_runner = raw_line['ReferenceNumber']
        
        param_arr_list.append(np.array(
           [# raw_line['Distance'],
            # raw_line['Duration'],
            # raw_line['Speed'],
            1 / raw_line['ParticipantPerformance'],    # take inverse, as small is beautiful here
            # raw_line['SpeedBefore'],
            # raw_line['RelativSpeed'],
            # raw_line['Slope'],
            #raw_line['MinSlope'],
            #raw_line['MaxSlope'],
            #get_avg_height(raw_line['zegments'])
           ]))
        
        segment_list_list.append(get_segment_list(raw_line['zegments'], 
                                                  raw_line['SectionLength']))
        
        Y_list.append(raw_line['Speed'])
    
    print("len runner_list", len(runner_list))
    print("runner_list", runner_list)
    
    # for test purposes carve out segment of list before normalisation etc.
    if from_to != None:
        param_arr_list = param_arr_list[from_to[0]:from_to[1]]
        segment_list_list = segment_list_list[from_to[0]:from_to[1]]
        Y_list = Y_list[from_to[0]:from_to[1]]
    
    
    normalise_segments(segment_list_list)   # !!!
    zero_pad_segments(segment_list_list)
    segments = np.array(segment_list_list)
    
    params = preprocessing.scale(np.array(param_arr_list))   # !!!
    # params = np.array(param_arr_list)   # instead of scale...
    
    # normalise segments manually !!!!! (first get mean and variance over all segments in list_list during 
    # get_segment_list(), then adapt --> do not use padded zeros for mean ...! dont alter one_hot encoding 
    # and padded zeros.
    
    # segments = preprocessing.scale(np.array(segment_list_list))

    # Y = preprocessing.scale(np.array(Y_list))   
    # scaling output leaves performance unchanged (several tests...)
    Y = np.array(Y_list)
    
    idx = np.arange(Y.shape[0])
    seed(42)
    # shuffle(idx)

    params  = params[idx]
    segments = segments[idx]
    Y = Y[idx]
    
    if integrity_check:
        check_data_integrity(raw_data, params, segments)
    
    return params, segments, Y

    

# write jason file with right indentation, line breaks, and properly sorted dicts                   
def write_clean_jason(read_path, write_path, from_to=None):

    
    with open(read_path, 'r') as fh:
        tmp = json.load(fh)
    
    tmp_new = []    
    for dict in tmp:
        dict_new = {
            "NormalizedSpeed": dict["NormalizedSpeed"],
            "SlopeStdDev": dict["SlopeStdDev"],
            "Duration": dict["Duration"],
            "EndDistance": dict["EndDistance"],
            "StartDistance": dict["StartDistance"],
            "Speed": dict["Speed"],
            "Slope": dict["NormalizedSpeed"],
            "MaxSlope": dict["MaxSlope"],
            "ParticipantPerformance": dict["ParticipantPerformance"],
            "RouteNumber": dict["RouteNumber"],
            "ReferenceNumber": dict["ReferenceNumber"],
            "DirectionChange": dict["DirectionChange"],
            "MinSlope": dict["MinSlope"],
            "SectionLength": dict["Length"],
            "zegments": dict["Segments"] if "Segments" in dict else dict["zegments"]
        }
        tmp_new.append(dict_new)

    print("now write new file ", write_path)

    with open(write_path, 'w') as fp:
        if from_to == None:
            json.dump(tmp_new, fp, sort_keys=True, indent=4)
        else:
            json.dump(tmp_new[from_to[0]:from_to[1]], fp, sort_keys=True, indent=4)                       