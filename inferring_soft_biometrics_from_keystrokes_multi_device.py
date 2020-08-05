# -*- coding: utf-8 -*-
"""Code for reproducing the results reported in the paper entitled, 
"On the Inference of Soft Biometrics from Typing Patterns Collected in a Multi-device Environment"
SUBMISSION: BigMM 2020
PAPER ID: "144"
"""

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/My Drive/BBMAS_Keystrokes_only/')

!pwd

!pip install KDEpy
!pip install pymrmr

import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
from collections import defaultdict
import pickle
from KDEpy import FFTKDE, TreeKDE
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from math import sqrt
from scipy.stats import gaussian_kde
from operator import itemgetter
import shutil
import math

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import kurtosis,skew
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pymrmr
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

import torch
from torch.utils.data import *
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.autograd import Variable

"""# **Fixed Text Extraction**"""

# Extract all fixed text time stamps to retrieve fixed text data
annotations_dir = './FreeTextMarkerUnixTimeStamp/'

desktop_annotations = pd.read_csv(annotations_dir+'Desktop_Freetext.csv').values
phone_annotations = pd.read_csv(annotations_dir+'Phone_Freetext.csv').values
tablet_annotations = pd.read_csv(annotations_dir+'Tablet_Freetext.csv').values

desktop_calibrated_annotations = {}
for annot in desktop_annotations:
    desktop_calibrated_annotations[int(annot[0])] = int(annot[1])

phone_calibrated_annotations = {}
for annot in phone_annotations:
    phone_calibrated_annotations[int(annot[0])] = int(annot[1])

tablet_calibrated_annotations = {}
for annot in tablet_annotations:
    tablet_calibrated_annotations[int(annot[0])] = int(annot[1])

# A function that takes in the entire raw keystroke data and a target fixed text timestamp, 
# and returns all the keystrokes upto that timestamp

def return_target_csv(data, target_time):
      target_csv = []
      for i, data_item in enumerate(data):
          if(int(data_item[3])<int(target_time)):
              target_csv.append([data_item[0], data_item[1], data_item[2], data_item[3]])
      return np.asarray(target_csv)

# Extract new CSVs for desktop fixed text
data_dir = 'Desktop/'
target_dir = 'Desktop_fixed_text/'

user_files = os.listdir(data_dir)
for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(data_dir+user_file)
        user_data = data_frame.values
        curr_user_ind = int(user_file[user_file.find('r')+1:user_file.find('.')])
        try:
            target_time_stamp = desktop_calibrated_annotations[curr_user_ind]
            csv = return_target_csv(user_data, target_time_stamp)
        except KeyError as e:
            csv = user_data

        f = open(target_dir+'User'+str(curr_user_ind)+'.csv', 'w')
        for line in csv:
            f.write('"'+str(line[0])+'","'+str(line[1])+'","'+str(line[2])+'","'+str(line[3])+'"\n')
        f.close()

# Extract new CSVs for phone fixed text
data_dir = 'Phone/'
target_dir = 'Phone_fixed_text/'

user_files = os.listdir(data_dir)
for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(data_dir+user_file)
        user_data = data_frame.values
        curr_user_ind = int(user_file[user_file.find('r')+1:user_file.find('.')])
        try:
            target_time_stamp = phone_calibrated_annotations[curr_user_ind]
            csv = return_target_csv(user_data, target_time_stamp)
        except KeyError as e:
            csv = user_data

        f = open(target_dir+'User'+str(curr_user_ind)+'.csv', 'w')
        for line in csv:
            f.write('"'+str(line[0])+'","'+str(line[1])+'","'+str(line[2])+'","'+str(line[3])+'"\n')
        f.close()

# Extract new CSVs for tablet fixed text
data_dir = 'Tablet/'
target_dir = 'Tablet_fixed_text/'

user_files = os.listdir(data_dir)
for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(data_dir+user_file)
        user_data = data_frame.values
        curr_user_ind = int(user_file[user_file.find('r')+1:user_file.find('.')])
        try:
            target_time_stamp = tablet_calibrated_annotations[curr_user_ind]
            csv = return_target_csv(user_data, target_time_stamp)
        except KeyError as e:
            csv = user_data

        f = open(target_dir+'User'+str(curr_user_ind)+'.csv', 'w')
        for line in csv:
            f.write('"'+str(line[0])+'","'+str(line[1])+'","'+str(line[2])+'","'+str(line[3])+'"\n')
        f.close()

"""# **KHT feature (Unigraph) extraction utilities**"""

# get KHT feature based on current key and timing values
def get_KHT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)

    for i, (key, timing) in enumerate(keys_in_pipeline):
          if(search_key==key):
              mask[i] = 0
              kht = int(float(search_key_timing))-int(float(timing))
              non_zero_indices = np.nonzero(mask) 
              if(len(non_zero_indices)>0):
                  keys_in_pipeline = keys_in_pipeline[non_zero_indices]
              else:
                  keys_in_pipeline = []
              return keys_in_pipeline, kht

    return keys_in_pipeline, None

# function to get KHT feature dictionary for a given user
def get_KHT_features(data):
    feature_dictionary = {}
    keys_in_pipeline = []

    for row_idx in range(len(data)):
        keys_in_pipeline = list(keys_in_pipeline)
        curr_key = data[row_idx][1]
        curr_direction = data[row_idx][2]
        curr_timing = data[row_idx][3]

        if(curr_direction==0):
            keys_in_pipeline.append([curr_key, curr_timing])

        if(curr_direction==1):
            keys_in_pipeline, curr_kht = get_KHT(keys_in_pipeline, curr_key, curr_timing)
            if(curr_kht is None):
                  continue
            else:
                  if(curr_key in list(feature_dictionary.keys())):
                        feature_dictionary[curr_key].append(curr_kht)
                  else:
                        feature_dictionary[curr_key] = []
                        feature_dictionary[curr_key].append(curr_kht)

    return feature_dictionary

"""# **KIT Data (Digraph) Pre-Processing**"""

# get KIT feature based on current key and timing values
def get_timings_KIT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    for i, (key, timing) in enumerate(keys_in_pipeline):
          if(search_key==key):
              mask[i] = 0
              non_zero_indices = np.nonzero(mask) 

              if(len(non_zero_indices)>0):
                  keys_in_pipeline = keys_in_pipeline[non_zero_indices]
              else:
                  keys_in_pipeline = []

              return keys_in_pipeline, timing, search_key_timing
    return keys_in_pipeline, None, None

# function to get KIT data frame with key, press_time, release_time for a given user
def get_dataframe_KIT(data):
    """ Input: data  Output: Dataframe with (key, press_time, release_time)""" 
    feature_dictionary = {}
    keys_in_pipeline = []
    result_key = []
    press = []
    release = []
    for row_idx in range(len(data)):
        keys_in_pipeline = list(keys_in_pipeline)
        curr_key = data[row_idx][1]
        curr_direction = data[row_idx][2]
        curr_timing = data[row_idx][3]

        if(curr_direction==0):
            keys_in_pipeline.append([curr_key, curr_timing])

        if(curr_direction==1):
            keys_in_pipeline, curr_start, curr_end = get_timings_KIT(keys_in_pipeline, curr_key, curr_timing)
            if(curr_start is None):
                continue
            else:
                result_key.append(curr_key)
                press.append(curr_start)
                release.append(curr_end)

    resultant_data_frame = pd.DataFrame(list(zip(result_key, press, release)),
               columns =['Key', 'Press_Time', 'Release_Time']) 
    return resultant_data_frame

"""# **KIT feature (Digraph) extraction utilities**"""

# function to get Flight1 KIT feature dictionary for a given user
def get_KIT_features_F1(data):
    """ Input: keystroke data, Output: Dictionary of (next_key_press - current_key_release) """
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if(row_idx + 1 >= len(data)):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx+1][1]
        
        if(str(curr_key)+str(next_key) in list(feature_dictionary.keys())):
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))
        else:
            feature_dictionary[str(curr_key)+str(next_key)] = []
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))

    return feature_dictionary  

# function to get Flight2 KIT feature dictionary for a given user
def get_KIT_features_F2(data):
    """ Input: keystroke data, Output: Dictionary of (next_key_press - current_key_press) """
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if(row_idx + 1 >= len(data)):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx+1][1]
        if(str(curr_key)+str(next_key) in list(feature_dictionary.keys())):
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))
        else:
            feature_dictionary[str(curr_key)+str(next_key)] = []
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))

    return feature_dictionary  

# function to get Flight3 KIT feature dictionary for a given user
def get_KIT_features_F3(data):
    """ Input: keystroke data, Output: Dictionary of (next_key_release - current_key_release) """
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if(row_idx + 1 >= len(data)):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx+1][2]
        if(str(curr_key)+str(next_key) in list(feature_dictionary.keys())):
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))
        else:
            feature_dictionary[str(curr_key)+str(next_key)] = []
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))

    return feature_dictionary  

# function to get Flight3 KIT feature dictionary for a given user
def get_KIT_features_F4(data):
    """ Input: keystroke data, Output: Dictionary of (next_key_release - current_key_press) """
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if(row_idx + 1 >= len(data)):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx+1][2]
        if(str(curr_key)+str(next_key) in list(feature_dictionary.keys())):
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))
        else:
            feature_dictionary[str(curr_key)+str(next_key)] = []
            feature_dictionary[str(curr_key)+str(next_key)].append(int(float(next_timing))-int(float(curr_timing)))

    return feature_dictionary

"""# **Word Level feature extraction utilities**"""

''' function to return word level statistics for each extracted word.
These features are as follows:
1) Word hold time
2) Average, Standard Deviation and Median of all key hold times in the word
3) Average, Standard Deviation and Median of all flight 1 features for all digraphs in the word
4) Average, Standard Deviation and Median of all flight 2 features for all digraphs in the word
5) Average, Standard Deviation and Median of all flight 3 features for all digraphs in the word
6) Average, Standard Deviation and Median of all flight 4 features for all digraphs in the word
'''
def get_advanced_word_level_features(words_in_pipeline):
    def get_word_hold(words_in_pipeline):
        return int(float(words_in_pipeline[-1][2])) - int(float(words_in_pipeline[0][1]))
    
    def get_avg_std_median_key_hold(words_in_pipeline):
        key_holds = []
        for _, press, release in words_in_pipeline:
            key_holds.append(int(float(release))-int(float(press)))
        return np.mean(key_holds), np.std(key_holds), np.median(key_holds)

    def get_avg_std_median_flights(words_in_pipeline):
        flights_1 = []
        flights_2 = []
        flights_3 = []
        flights_4 = []
        for i in range(len(words_in_pipeline)-1):
            k1_r = words_in_pipeline[i][2]
            k1_p = words_in_pipeline[i][1]
            k2_r = words_in_pipeline[i+1][2]
            k2_p = words_in_pipeline[i+1][1]
            flights_1.append(int(float(k2_p))-int(float(k1_r)))
            flights_2.append(int(float(k2_r))-int(float(k1_r)))
            flights_3.append(int(float(k2_p))-int(float(k1_p)))
            flights_4.append(int(float(k2_r))-int(float(k1_p)))
        return np.mean(flights_1), np.std(flights_1), np.median(flights_1), np.mean(flights_2), np.std(flights_2), np.median(flights_2), np.mean(flights_3), np.std(flights_3), np.median(flights_3), np.mean(flights_4), np.std(flights_4), np.median(flights_4)

    wh = get_word_hold(words_in_pipeline)
    avg_kh, std_kh, median_kh = get_avg_std_median_key_hold(words_in_pipeline)
    avg_flight1, std_flight1, median_flight1, avg_flight2, std_flight2, median_flight2, avg_flight3, std_flight3, median_flight3, avg_flight4, std_flight4, median_flight4 = get_avg_std_median_flights(words_in_pipeline)
    return [wh, avg_kh, std_kh, median_kh, avg_flight1, std_flight1, median_flight1, avg_flight2, std_flight2, median_flight2, avg_flight3, std_flight3, median_flight3, avg_flight4, std_flight4, median_flight4]

# function to get the advanced word level features of every user
def get_advanced_word_features(processed_data):
    words_in_pipeline = []
    feature_dictionary = {}

    ignore_keys = ['LCTRL', 'RSHIFT', 'TAB', 'DOWN']
    delimiter_keys = ['SPACE', '.', ',', 'RETURN']

    for row_idx in range(len(processed_data)):
        curr_key = processed_data[row_idx][1]
        curr_press = processed_data[row_idx][2]
        curr_release = processed_data[row_idx][3]

        if(curr_key in ignore_keys):
              continue

        if(curr_key in delimiter_keys):
            if(len(words_in_pipeline)>0):
                advanced_word_features = get_advanced_word_level_features(words_in_pipeline)
                key_word = ''
                for char, _, _ in words_in_pipeline:
                    key_word=key_word+str(char)
                
                if(key_word in list(feature_dictionary.keys())):
                    feature_dictionary[key_word].append(advanced_word_features)
                else:
                    feature_dictionary[key_word] = []
                    feature_dictionary[key_word].append(advanced_word_features)
            words_in_pipeline = []
            continue

        if(curr_key=='BACKSPACE'):
              words_in_pipeline = words_in_pipeline[:-1]
              continue

        words_in_pipeline.append([curr_key, curr_press, curr_release])

    return feature_dictionary

"""# **Feature extraction and serialization**"""

# get the entire feature dictionary given the modality (Desktop, Phone, Tablet)
def get_all_users_features_KHT(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        user_data = data_frame.values
        user_feat_dict = get_KHT_features(user_data)
        users_feat_dict[i+1] = user_feat_dict

    return users_feat_dict

def get_all_users_features_KIT(directory):
    users_feat_dict_f1 = {}
    users_feat_dict_f2 = {}
    users_feat_dict_f3 = {}
    users_feat_dict_f4 = {}
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        data_frame = get_dataframe_KIT(data_frame.values)
        user_data = data_frame.values
        
        user_feat_dict_f1 = get_KIT_features_F1(user_data)
        users_feat_dict_f1[i+1] = user_feat_dict_f1

        user_feat_dict_f2 = get_KIT_features_F2(user_data)
        users_feat_dict_f2[i+1] = user_feat_dict_f2

        user_feat_dict_f3 = get_KIT_features_F3(user_data)
        users_feat_dict_f3[i+1] = user_feat_dict_f3

        user_feat_dict_f4 = get_KIT_features_F4(user_data)
        users_feat_dict_f4[i+1] = user_feat_dict_f4

    return users_feat_dict_f1, users_feat_dict_f2, users_feat_dict_f3, users_feat_dict_f4


def get_all_users_features_word(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        user_data = data_frame.values
        user_feat_dict = get_word_features(user_data)
        users_feat_dict[i+1] = user_feat_dict

    return users_feat_dict

def get_all_users_features_advanced_word(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        user_data = data_frame.values
        processed_data = get_dataframe_KIT(user_data)
        processed_data = np.c_[np.arange(len(processed_data)), processed_data]
        processed_data = processed_data[np.argsort(processed_data[:, 2])]
        user_feat_dict = get_advanced_word_features(processed_data)
        users_feat_dict[i+1] = user_feat_dict

    return users_feat_dict

# get the entire feature dictionary given the modality for fixed text (Desktop, Phone, Tablet)
def get_all_users_features_KHT_fixed(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        user_data = data_frame.values
        user_feat_dict = get_KHT_features(user_data)
        users_feat_dict[i+1] = user_feat_dict

    return users_feat_dict

def get_all_users_features_KIT_fixed(directory):
    users_feat_dict_f1 = {}
    users_feat_dict_f2 = {}
    users_feat_dict_f3 = {}
    users_feat_dict_f4 = {}
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        data_frame = get_dataframe_KIT(data_frame.values)
        user_data = data_frame.values
        
        user_feat_dict_f1 = get_KIT_features_F1(user_data)
        users_feat_dict_f1[i+1] = user_feat_dict_f1

        user_feat_dict_f2 = get_KIT_features_F2(user_data)
        users_feat_dict_f2[i+1] = user_feat_dict_f2

        user_feat_dict_f3 = get_KIT_features_F3(user_data)
        users_feat_dict_f3[i+1] = user_feat_dict_f3

        user_feat_dict_f4 = get_KIT_features_F4(user_data)
        users_feat_dict_f4[i+1] = user_feat_dict_f4

    return users_feat_dict_f1, users_feat_dict_f2, users_feat_dict_f3, users_feat_dict_f4


def get_all_users_features_word_fixed(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        user_data = data_frame.values
        user_feat_dict = get_word_features(user_data)
        users_feat_dict[i+1] = user_feat_dict

    return users_feat_dict

def get_all_users_features_advanced_word_fixed(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        data_frame = pd.read_csv(directory+user_file)
        user_data = data_frame.values
        processed_data = get_dataframe_KIT(user_data)
        processed_data = np.c_[np.arange(len(processed_data)), processed_data]
        processed_data = processed_data[np.argsort(processed_data[:, 2])]
        user_feat_dict = get_advanced_word_features(processed_data)
        users_feat_dict[i+1] = user_feat_dict

    return users_feat_dict

# KHT feature extraction for Desktop
desktop_kht_features = get_all_users_features_KHT('Desktop/')

pickle.dump(desktop_kht_features, open('desktop_kht_feature_dictionary.pickle', 'wb'))

# KHT fixed text feature extraction for Desktop
desktop_kht_features_fixed = get_all_users_features_KHT_fixed('Desktop_fixed_text/')

pickle.dump(desktop_kht_features_fixed, open('desktop_kht_feature_dictionary_fixed.pickle', 'wb'))

# KHT feature extraction for Phone
phone_kht_features = get_all_users_features_KHT('Phone/')

pickle.dump(phone_kht_features, open('phone_kht_feature_dictionary.pickle', 'wb'))

# KHT fixed text feature extraction for Phone
phone_kht_features_fixed = get_all_users_features_KHT_fixed('Phone_fixed_text/')

pickle.dump(phone_kht_features_fixed, open('phone_kht_feature_dictionary_fixed.pickle', 'wb'))

# KHT feature extraction for Tablet
tablet_kht_features = get_all_users_features_KHT('Tablet/')

pickle.dump(tablet_kht_features, open('tablet_kht_feature_dictionary.pickle', 'wb'))

# KHT fixed text feature extraction for Tablet
tablet_kht_features_fixed = get_all_users_features_KHT_fixed('Tablet_fixed_text/')

pickle.dump(tablet_kht_features_fixed, open('tablet_kht_feature_dictionary_fixed.pickle', 'wb'))

# KIT feature extraction for Desktop
desktop_kit_features_f1, desktop_kit_features_f2, desktop_kit_features_f3, desktop_kit_features_f4 = get_all_users_features_KIT('Desktop/')

pickle.dump(desktop_kit_features_f1, open('desktop_kit_feature_f1_dictionary.pickle', 'wb'))
pickle.dump(desktop_kit_features_f2, open('desktop_kit_feature_f2_dictionary.pickle', 'wb'))
pickle.dump(desktop_kit_features_f3, open('desktop_kit_feature_f3_dictionary.pickle', 'wb'))
pickle.dump(desktop_kit_features_f4, open('desktop_kit_feature_f4_dictionary.pickle', 'wb'))

# KIT fixed text feature extraction for Desktop
desktop_kit_features_f1_fixed, desktop_kit_features_f2_fixed, desktop_kit_features_f3_fixed, desktop_kit_features_f4_fixed = get_all_users_features_KIT_fixed('Desktop_fixed_text/')

pickle.dump(desktop_kit_features_f1_fixed, open('desktop_kit_feature_f1_dictionary_fixed.pickle', 'wb'))
pickle.dump(desktop_kit_features_f2_fixed, open('desktop_kit_feature_f2_dictionary_fixed.pickle', 'wb'))
pickle.dump(desktop_kit_features_f3_fixed, open('desktop_kit_feature_f3_dictionary_fixed.pickle', 'wb'))
pickle.dump(desktop_kit_features_f4_fixed, open('desktop_kit_feature_f4_dictionary_fixed.pickle', 'wb'))

# KIT feature extraction for Phone
phone_kit_features_f1, phone_kit_features_f2, phone_kit_features_f3, phone_kit_features_f4 = get_all_users_features_KIT('Phone/')

pickle.dump(phone_kit_features_f1, open('phone_kit_feature_f1_dictionary.pickle', 'wb'))
pickle.dump(phone_kit_features_f2, open('phone_kit_feature_f2_dictionary.pickle', 'wb'))
pickle.dump(phone_kit_features_f3, open('phone_kit_feature_f3_dictionary.pickle', 'wb'))
pickle.dump(phone_kit_features_f4, open('phone_kit_feature_f4_dictionary.pickle', 'wb'))

# KIT fixed text feature extraction for Phone
phone_kit_features_f1_fixed, phone_kit_features_f2_fixed, phone_kit_features_f3_fixed, phone_kit_features_f4_fixed = get_all_users_features_KIT_fixed('Phone_fixed_text/')

pickle.dump(phone_kit_features_f1_fixed, open('phone_kit_feature_f1_dictionary_fixed.pickle', 'wb'))
pickle.dump(phone_kit_features_f2_fixed, open('phone_kit_feature_f2_dictionary_fixed.pickle', 'wb'))
pickle.dump(phone_kit_features_f3_fixed, open('phone_kit_feature_f3_dictionary_fixed.pickle', 'wb'))
pickle.dump(phone_kit_features_f4_fixed, open('phone_kit_feature_f4_dictionary_fixed.pickle', 'wb'))

# KIT feature extraction for Tablet
tablet_kit_features_f1, tablet_kit_features_f2, tablet_kit_features_f3, tablet_kit_features_f4 = get_all_users_features_KIT('Tablet/')

pickle.dump(tablet_kit_features_f1, open('tablet_kit_feature_f1_dictionary.pickle', 'wb'))
pickle.dump(tablet_kit_features_f2, open('tablet_kit_feature_f2_dictionary.pickle', 'wb'))
pickle.dump(tablet_kit_features_f3, open('tablet_kit_feature_f3_dictionary.pickle', 'wb'))
pickle.dump(tablet_kit_features_f4, open('tablet_kit_feature_f4_dictionary.pickle', 'wb'))

# KIT fixed text feature extraction for Tablet
tablet_kit_features_f1_fixed, tablet_kit_features_f2_fixed, tablet_kit_features_f3_fixed, tablet_kit_features_f4_fixed = get_all_users_features_KIT_fixed('Tablet_fixed_text/')

pickle.dump(tablet_kit_features_f1_fixed, open('tablet_kit_feature_f1_dictionary_fixed.pickle', 'wb'))
pickle.dump(tablet_kit_features_f2_fixed, open('tablet_kit_feature_f2_dictionary_fixed.pickle', 'wb'))
pickle.dump(tablet_kit_features_f3_fixed, open('tablet_kit_feature_f3_dictionary_fixed.pickle', 'wb'))
pickle.dump(tablet_kit_features_f4_fixed, open('tablet_kit_feature_f4_dictionary_fixed.pickle', 'wb'))

# advanced word feature extraction for Desktop
desktop_advanced_word_features = get_all_users_features_advanced_word('Desktop/')

pickle.dump(desktop_advanced_word_features, open('desktop_advanced_word_feature_dictionary.pickle', 'wb'))

# advanced word fixed text feature extraction for Desktop
desktop_advanced_word_features_fixed = get_all_users_features_advanced_word_fixed('Desktop_fixed_text/')

pickle.dump(desktop_advanced_word_features_fixed, open('desktop_advanced_word_feature_dictionary_fixed.pickle', 'wb'))

# advanced word feature extraction for Phone
phone_advanced_word_features = get_all_users_features_advanced_word('Phone/')

pickle.dump(phone_advanced_word_features, open('phone_advanced_word_feature_dictionary.pickle', 'wb'))

# advanced word fixed text feature extraction for Phone
phone_advanced_word_features_fixed = get_all_users_features_advanced_word_fixed('Phone_fixed_text/')

pickle.dump(phone_advanced_word_features_fixed, open('phone_advanced_word_feature_dictionary_fixed.pickle', 'wb'))

# advanced word feature extraction for Tablet
tablet_advanced_word_features = get_all_users_features_advanced_word('Tablet/')

pickle.dump(tablet_advanced_word_features, open('tablet_advanced_word_feature_dictionary.pickle', 'wb'))

# advanced word fixed text feature extraction for Tablet
tablet_advanced_word_features_fixed = get_all_users_features_advanced_word_fixed('Tablet_fixed_text/')

pickle.dump(tablet_advanced_word_features_fixed, open('tablet_advanced_word_feature_dictionary_fixed.pickle', 'wb'))

"""# **Outlier Removal Utility**"""

# Remove outlier points in the distribution using the 1.5IQR rule
def remove_outliers(x):
    a = np.asarray(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    result = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            result.append(y)
    return result

"""# **Feature processing for Classification/Regression tasks**"""

def get_features(features):
    """ Input: All feature dictionary Output: Feature matrix with unique columns"""
    feature_set = []
    for key1 in features:
        for key2 in features[key1]:
            feature_set.append(key2)

    # Getting unique columns by removing repeated keys
    unique_feature_set = set(feature_set)
    unique_feature_set = list(unique_feature_set)

    size = len(unique_feature_set)
    rows, cols = (len(features), len(unique_feature_set))
    feature_vector = [[0 for x in range(len(cols))] for x in range(rows)]

    # Updating feature matrix based on present features in dictionary
    for key1 in tqdm(features):
        for key2 in features[key1]:
            for j in range(len(unique_feature_set)):
                if unique_feature_set[j] == key2:
                    temp = abs(np.median(features[key1][key2]))
                    feature_vector[(key1)-1][j] = int(temp)
                    break
                else:
                    feature_vector[(key1)-1][j] = 0
    return feature_vector

top_feature_KIT_Desktop_F1 = ["in","nSPACE", "et", "lSPACE", "oSPACE", "ca", "iSPACE", "pl", "ve", "ha", "ne", "da", "he", "wi"]
top_feature_KIT_Tablet_F1 = ["me", "is", "ne", "ha"]
top_feature_KIT_Tablet_F2 = ["ha", "is", "me"]
top_feature_KIT_Tablet_F3 = ["ne"]
top_feature_KIT_Phone_F1 = ["BACKSPACEBACKSPACE"]
top_feature_KIT_Phone_F2 = ["BACKSPACEBACKSPACE"]
top_feature_KIT_Phone_F4 = ["BACKSPACEBACKSPACE"]

def top_feature_KIT(pickle_file, top_feature):
    kit_feature_dictionary = pickle.load(open(pickle_file, 'rb'))
    selected_top_feature = [[0 for x in range(len(top_feature))] for x in range(116)] 
    for key1 in kit_feature_dictionary:
        if key1 == 117:
            break
        for i in range(len(top_feature)):
            for key2 in kit_feature_dictionary[key1]:
                if (str(top_feature[i]) == str(key2)):
                    selected_top_feature[key1-1][i] = np.median(kit_feature_dictionary[key1][key2])
                    break
    return selected_top_feature

def concatenated_feature_matrix_KIT():
    feature_KIT_Desktop_F1 = top_feature_KIT("desktop_kit_feature_f1_dictionary.pickle", top_feature_KIT_Desktop_F1)
    feature_KIT_Tablet_F1 = top_feature_KIT("tablet_kit_feature_f1_dictionary.pickle", top_feature_KIT_Tablet_F1)
    feature_KIT_Tablet_F2 = top_feature_KIT("tablet_kit_feature_f2_dictionary.pickle", top_feature_KIT_Tablet_F2)
    feature_KIT_Tablet_F3 = top_feature_KIT("tablet_kit_feature_f3_dictionary.pickle", top_feature_KIT_Tablet_F3)
    feature_KIT_Phone_F1 = top_feature_KIT("phone_kit_feature_f1_dictionary.pickle", top_feature_KIT_Phone_F1)
    feature_KIT_Phone_F2 = top_feature_KIT("phone_kit_feature_f2_dictionary.pickle", top_feature_KIT_Phone_F2)
    feature_KIT_Phone_F4 = top_feature_KIT("phone_kit_feature_f4_dictionary.pickle", top_feature_KIT_Phone_F4)
    return np.concatenate((np.array(feature_KIT_Desktop_F1), np.array(feature_KIT_Tablet_F1), np.array(feature_KIT_Tablet_F2), np.array(feature_KIT_Tablet_F3), np.array(feature_KIT_Phone_F1), np.array(feature_KIT_Phone_F2), np.array(feature_KIT_Phone_F4)), axis=1)

"""## **KHT (Unigraph) Features (Desktop Only)**"""

'''Desktop Only KHT
0.02974 ./KDE_plots/Desktop/Gender/KHT/kht_v.png v
0.0202 ./KDE_plots/Desktop/Gender/KHT/kht_n.png n
0.01732 ./KDE_plots/Desktop/Gender/KHT/kht_s.png s
0.01723 ./KDE_plots/Desktop/Gender/KHT/kht_h.png h
0.01613 ./KDE_plots/Desktop/Gender/KHT/kht_BACKSPACE.png BACKSPACE
0.01502 ./KDE_plots/Desktop/Gender/KHT/kht_r.png r
0.01446 ./KDE_plots/Desktop/Gender/KHT/kht_u.png u
0.01405 ./KDE_plots/Desktop/Gender/KHT/kht_m.png m
0.01368 ./KDE_plots/Desktop/Gender/KHT/kht_i.png i
0.0134 ./KDE_plots/Desktop/Gender/KHT/kht_p.png p
0.01212 ./KDE_plots/Desktop/Gender/KHT/kht_t.png t
0.01141 ./KDE_plots/Desktop/Gender/KHT/kht_d.png d
0.00996 ./KDE_plots/Desktop/Gender/KHT/kht_o.png o
0.00845 ./KDE_plots/Desktop/Gender/KHT/kht_SPACE.png SPACE
0.00775 ./KDE_plots/Desktop/Gender/KHT/kht_l.png l
0.00595 ./KDE_plots/Desktop/Gender/KHT/kht_q.png q
0.00324 ./KDE_plots/Desktop/Gender/KHT/kht_e.png e
0.00307 ./KDE_plots/Desktop/Gender/KHT/kht_..png .
0.00234 ./KDE_plots/Desktop/Gender/KHT/kht_c.png c
0.00215 ./KDE_plots/Desktop/Gender/KHT/kht_w.png w
0.00175 ./KDE_plots/Desktop/Gender/KHT/kht_y.png y
0.00056 ./KDE_plots/Desktop/Gender/KHT/kht_f.png f
0.00052 ./KDE_plots/Desktop/Gender/KHT/kht_a.png a'''

feature_list_Desktop_KHT = ["v", "n", "s", "h", "BACKSPACE", "r", "u", "m", "i","p","t","d", "o", "SPACE", "l", "q", "e", ".", "c", "w", "f", "a"]

"""## **KHT (Unigraph) Features (Phone Only)**"""

'''Phone only KHT
0.01494 ./KDE_plots/Phone/Gender/KHT/kht_q.png q
0.01181 ./KDE_plots/Phone/Gender/KHT/kht_BACKSPACE.png BACKSPACE
0.01177 ./KDE_plots/Phone/Gender/KHT/kht_w.png w
0.00993 ./KDE_plots/Phone/Gender/KHT/kht_..png .
0.00887 ./KDE_plots/Phone/Gender/KHT/kht_s.png s
0.00668 ./KDE_plots/Phone/Gender/KHT/kht_l.png l
0.00568 ./KDE_plots/Phone/Gender/KHT/kht_g.png g
0.00526 ./KDE_plots/Phone/Gender/KHT/kht_a.png a
0.00451 ./KDE_plots/Phone/Gender/KHT/kht_y.png y
0.00388 ./KDE_plots/Phone/Gender/KHT/kht_e.png e
0.00294 ./KDE_plots/Phone/Gender/KHT/kht_SPACE.png SPACE
0.00266 ./KDE_plots/Phone/Gender/KHT/kht_i.png i
0.00261 ./KDE_plots/Phone/Gender/KHT/kht_o.png o
0.00257 ./KDE_plots/Phone/Gender/KHT/kht_m.png m
0.00239 ./KDE_plots/Phone/Gender/KHT/kht_p.png p
0.00217 ./KDE_plots/Phone/Gender/KHT/kht_r.png r
0.00203 ./KDE_plots/Phone/Gender/KHT/kht_d.png d
0.00139 ./KDE_plots/Phone/Gender/KHT/kht_n.png n
0.00133 ./KDE_plots/Phone/Gender/KHT/kht_h.png h
0.00124 ./KDE_plots/Phone/Gender/KHT/kht_u.png u
0.00113 ./KDE_plots/Phone/Gender/KHT/kht_v.png v
0.00092 ./KDE_plots/Phone/Gender/KHT/kht_c.png c
0.00081 ./KDE_plots/Phone/Gender/KHT/kht_t.png t
0.00065 ./KDE_plots/Phone/Gender/KHT/kht_f.png f'''

feature_list_Phone_KHT = ["q", "BACKSPACE", "w", ".", "s", "l", "g", "a", "y", "e", "SPACE", "i", "o", "m", "p", "r", "d", "n", "h", "u", "v", "c", "t", "f"]

"""## **KHT (Unigraph) Features (Tablet Only)**"""

'''Tablet Only KHT
0.01702 ./KDE_plots/Tablet/Gender/KHT/kht_BACKSPACE.png BACKSPACE
0.0148 ./KDE_plots/Tablet/Gender/KHT/kht_r.png r
0.01267 ./KDE_plots/Tablet/Gender/KHT/kht_v.png v
0.01234 ./KDE_plots/Tablet/Gender/KHT/kht_p.png p
0.01105 ./KDE_plots/Tablet/Gender/KHT/kht_c.png c
0.00729 ./KDE_plots/Tablet/Gender/KHT/kht_e.png e
0.00532 ./KDE_plots/Tablet/Gender/KHT/kht_..png .
0.00407 ./KDE_plots/Tablet/Gender/KHT/kht_f.png f
0.00364 ./KDE_plots/Tablet/Gender/KHT/kht_a.png a
0.00274 ./KDE_plots/Tablet/Gender/KHT/kht_SPACE.png SPACE
0.0023 ./KDE_plots/Tablet/Gender/KHT/kht_d.png d
0.00227 ./KDE_plots/Tablet/Gender/KHT/kht_h.png h
0.00219 ./KDE_plots/Tablet/Gender/KHT/kht_l.png l
0.00218 ./KDE_plots/Tablet/Gender/KHT/kht_b.png b
0.00202 ./KDE_plots/Tablet/Gender/KHT/kht_u.png u
0.00181 ./KDE_plots/Tablet/Gender/KHT/kht_t.png t
0.0015 ./KDE_plots/Tablet/Gender/KHT/kht_o.png o
0.00146 ./KDE_plots/Tablet/Gender/KHT/kht_m.png m
0.00139 ./KDE_plots/Tablet/Gender/KHT/kht_y.png y
0.00132 ./KDE_plots/Tablet/Gender/KHT/kht_i.png i
0.0012 ./KDE_plots/Tablet/Gender/KHT/kht_n.png n
0.00093 ./KDE_plots/Tablet/Gender/KHT/kht_s.png s
0.00064 ./KDE_plots/Tablet/Gender/KHT/kht_w.png w'''

feature_list_Tablet_KHT = ["BACKSPACE", "r", "v", "p", "c", "e", ".", "f", "a", "SPACE", "d", "h", "l", "b", "u", "t", "o", "m", "y", "i", "n", "s", "w"]

"""## **KIT (Digraph) Features (Desktop Only)**"""

feature_list_Desktop_KIT_1 = ['ir', 'ot', 've', 'no', 'ha', 'he', 'to', 'rd', 'hSPACE', 'is', 'co', 'un', 'pSPACE', 'di', 'fi', 'ni', 'pl', 'ne', 'fSPACE', 'hi', 'tSPACE', 'es', 'eSPACE', 'on', 'dSPACE', 'oSPACE', 'rl', 'me', 'le', 'wi', 'if', 'nc', 'lSPACE', 'SPACEs', 'SPACEa', 'nt', 'pe', 'or', 'iSPACE', 'wo', 'SPACEt', 'ca', 'nd', 'ce', 'ly', 'ov', 'te', 'ed', 'th', 'tw', 'ff', 'in', 'aSPACE', 'SPACEl', 'li', 'se', 'it', 'la', 'SPACEn', 'of', 'ul', 'ec', 'ct', 'yp', 'et', 'SPACEc', 'io', 'sa', 'ts', 're', 'SPACEf', 'ti', 'il', 'ap', 'ue', 'ySPACE', 'ds', 'SPACEh', 'el', 'da', 'SPACEo', 'ol', 'll', 'ef', 'SPACEp', 'ta', 'st', 'am', 'SPACEw', 'ar', 'rs', 'en', 'er', 'sSPACE', 'qu', 's.', 'at', '.SPACE', 'nSPACE', 'SPACEu', 'SPACEm', 'as', 'SPACEi', 'e.', 'SPACEd', 'si', 'ee', 'ss']

feature_list_Desktop_KIT_2 = ['rd', 'rl', 'to', 'un', 'ap', 'ot', 'el', 'ol', 'ir', 'ef', 'rs', 'ov', 'yp', 'ly', 'nSPACE', 'SPACEm', 'ue', 'if', 'no', 'nt', 've', 'ed', 'SPACEs', 'iSPACE', 'es', 'co', 'ee', 'SPACEn', 'wi', 'st', 'ni', 'it', 'ct', 'nc', 'et', 'hi', 'ce', 'pSPACE', 'oSPACE', 'hSPACE', 'SPACEf', 'pe', 'sa', 'se', 're', 'of', 'ts', 'SPACEu', 'di', 'la', 'ff', 'si', 'aSPACE', 'ta', 'ti', 'en', 'ySPACE', 'pl', 'ss', 'da', 'fSPACE', 'li', 'tw', 'lSPACE', 'th', 'SPACEc', 'il', 'ne', 'is', 'te', 'or', 'le', 'ul', 'SPACEp', 'as', 'SPACEl', 's.', '.SPACE', 'in', 'me', 'ds', 'dSPACE', 'SPACEa', 'er', 'nd', 'SPACEt', 'fi', 'he', 'ha', 'qu', 'll', 'ec', 'am', 'SPACEd', 'on', 'ca', 'at', 'sSPACE', 'SPACEi', 'SPACEo', 'wo', 'tSPACE', 'SPACEh', 'io', 'eSPACE', 'e.', 'SPACEw', 'ar']

feature_list_Desktop_KIT_3 = ['un', 'to', 'li', 'rl', 'es', 'nt', 'ce', 'hSPACE', 'th', 'iSPACE', 'rd', 'fi', 'fSPACE', 'el', 'pl', 'si', 'ef', 'en', 'ff', 'SPACEs', 'oSPACE', 'ed', 'co', 'ov', 'ap', 'ni', 'if', 'dSPACE', 'il', 'll', 'SPACEn', 'sa', 'ds', 'ot', 'as', 'ir', 'et', 'pSPACE', 'SPACEc', 'tw', 'ca', 've', 'he', 'it', 'lSPACE', 'SPACEt', 'ar', 'SPACEf', 'in', 'ue', 'am', 'ct', 'la', 'no', 'wi', 'nSPACE', 'ti', 'ne', 'ec', 'SPACEp', 'ta', 'is', 'st', 'rs', 'pe', 'ss', 'or', 're', 'hi', 'io', 'ts', 'SPACEw', 'se', 'aSPACE', 'ySPACE', 'ha', 'nc', 'ol', 'da', 'of', 'SPACEu', 'le', 'ly', 'SPACEm', 'SPACEl', 'on', 'at', 's.', 'sSPACE', 'yp', 'di', 'SPACEa', 'te', 'ee', 'SPACEd', 'eSPACE', 'er', 'SPACEi', 'ul', '.SPACE', 'me', 'qu', 'e.', 'SPACEo', 'wo', 'SPACEh', 'nd', 'tSPACE']

feature_list_Desktop_KIT_4 = ['un', 'pl', 'rl', 'rd', 'ol', 'in', 'as', 'ds', 'en', 'me', 'ef', 'to', 'iSPACE', 'll', 'ir', 'nt', 'fSPACE', 'ot', 'at', 'ar', 'ee', 'if', 'am', 'fi', 'el', 'SPACEm', 'co', 'ed', 'ap', 'SPACEu', 'di', 'aSPACE', 'hSPACE', 'ly', 'hi', 'ov', 'it', 'SPACEn', 'ct', 'SPACEi', 'ff', 'ca', 'tw', 'ce', 'SPACEf', 'es', 'wo', 'ni', 'si', 'eSPACE', 'et', 'ti', 'th', 'nc', 'ySPACE', 'da', 'SPACEs', 'qu', 'pe', 'er', 'SPACEl', 'or', 'oSPACE', 'nSPACE', 'li', 'SPACEc', 'ul', 'pSPACE', 'st', 'of', 'il', 'wi', 'on', 'SPACEp', 'la', 'yp', 's.', 'SPACEt', 'sSPACE', 'rs', 'dSPACE', 'he', 'sa', 'ec', 'te', 'lSPACE', 'ta', 'ha', 'SPACEw', 'nd', 'ss', 'ue', 'ts', 'no', 'ne', 've', 'io', 'is', 'SPACEd', '.SPACE', 'SPACEa', 'e.', 'SPACEo', 'se', 'SPACEh', 'tSPACE', 'le', 're']

"""## **KIT (Digraph) Features (Phone Only)**"""

feature_list_Phone_KIT_1 = ['am', 'wo', 'pl', 'me', 'fu', 'le', 'en', 'dSPACE', 'is', 'ha', 'ti', 'ce', 'er', 'to', 'of', 'nd', 'st', 'lSPACE', 'SPACEo', 'nt', 've', 'ct', 'nc', 'se', 'BACKSPACEBACKSPACE', 'aSPACE', 'at', 'co', 'ff', 'ed', 'or', 'ni', 'te', 'ta', 'io', 'SPACEf', 'wi', 'da', 'qu', 'li', 'sSPACE', 'eSPACE', 'oSPACE', 'ol', 'rd', 'ee', 'th', 'nSPACE', 'if', 'in', 'he', 'SPACEw', 'ySPACE', 'hi', 'fSPACE', 'as', 'll', 'SPACEu', 'es', 'on', 'ec', 'iq', 'ar', 'SPACEt', 'e.', 'SPACEa', 'SPACEi', 're', 'SPACEd', 'tSPACE', 'SPACEs']

feature_list_Phone_KIT_2 = ['BACKSPACEBACKSPACE', 'am', 'wo', 'me', 'le', 'rd', 'te', 'pl', 'SPACEo', 'fu', 'er', 'ce', 'lSPACE', 'dSPACE', 'st', 'is', 'of', 'da', 'to', 'ta', 'ct', 've', 'se', 'nd', 'ti', 'at', 'qu', 'ec', 'en', 'wi', 'if', 'fSPACE', 'nc', 'iq', 'or', 'eSPACE', 'ed', 'th', 'as', 'nt', 'nSPACE', 'sSPACE', 'ha', 'aSPACE', 'SPACEw', 'SPACEu', 'ff', 'co', 'ni', 'io', 'hi', 'he', 'li', 'ol', 'es', 'ee', 'ySPACE', 'SPACEf', 'oSPACE', 'e.', 'll', 'SPACEt', 'in', 'SPACEa', 'on', 're', 'SPACEd', 'tSPACE', 'SPACEs', 'ar', 'SPACEi']

feature_list_Phone_KIT_3 = ['wo', 'am', 'le', 'se', 'fu', 've', 'to', 'dSPACE', 'SPACEo', 'st', 'qu', 'en', 'me', 'ni', 'io', 'ol', 'te', 'er', 'ti', 'da', 'rd', 'wi', 'at', 'ce', 'co', 'ed', 'aSPACE', 'of', 'iq', 'ta', 'pl', 'eSPACE', 'sSPACE', 'ct', 'fSPACE', 'nd', 'lSPACE', 'th', 'nt', 'ee', 'is', 'es', 'hi', 'nSPACE', 'li', 'nc', 'SPACEw', 'SPACEf', 'SPACEu', 'or', 're', 'ec', 'ha', 'on', 'as', 'ySPACE', 'oSPACE', 'tSPACE', 'ar', 'ff', 'if', 'e.', 'in', 'SPACEt', 'BACKSPACEBACKSPACE', 'SPACEa', 'll', 'he', 'SPACEd', 'SPACEi', 'SPACEs']

feature_list_Phone_KIT_4 = ['wo', 'am', 'le', 'te', 'dSPACE', 'of', 've', 'ta', 'st', 'ff', 'io', 'se', 'ed', 'SPACEo', 'me', 'ce', 'to', 'er', 'th', 'fu', 'da', 'ee', 'rd', 'iq', 'if', 'as', 'wi', 'qu', 'lSPACE', 'nd', 'ni', 'fSPACE', 'aSPACE', 'ec', 'en', 'ti', 'at', 'co', 'nc', 'eSPACE', 'es', 're', 'll', 'BACKSPACEBACKSPACE', 'nSPACE', 'sSPACE', 'ct', 'nt', 'hi', 'SPACEu', 'ol', 'SPACEw', 'pl', 'in', 'is', 'or', 'li', 'ha', 'e.', 'he', 'ySPACE', 'SPACEf', 'on', 'oSPACE', 'SPACEi', 'tSPACE', 'SPACEd', 'SPACEs', 'SPACEa', 'SPACEt', 'ar']

"""## **KIT (Digraph) Features (Tablet Only)**"""

feature_list_Tablet_KIT_1 = ['ne', 'ot', 'or', 'is', 'me', 'in', 'en', 'ySPACE', 'nSPACE', 'BACKSPACEBACKSPACE', 'he', 'on', 'si', 'lSPACE', 'nd', 'dSPACE', 'nt', 'eSPACE', 'no', 'sSPACE', 'to', 'co', 'oSPACE', 'it', 'tSPACE', 'SPACEl', 'SPACEBACKSPACE', 'SPACEp', 'll', 'at', 'ha', 'SPACEf', 'er', 'hi', 're', 've', 'ol', 'SPACEc', 'SPACEm', 'ty', 'SPACEn', 'ar', 'ff', 'ec', 'se', 'th', 'st', 'SPACEo', 'SPACEs', 'SPACEi', 'SPACEt', 'te', 'SPACEa', 'SPACEw']

feature_list_Tablet_KIT_2 = ['BACKSPACEBACKSPACE', 'me', 'si', 'or', 'ot', 'ySPACE', 'he', 'nSPACE', 'en', 'sSPACE', 'nt', 'tSPACE', 'is', 'it', 'SPACEc', 'lSPACE', 'no', 'SPACEBACKSPACE', 'in', 'nd', 'on', 'ne', 'SPACEl', 'co', 'ff', 'SPACEf', 'SPACEp', 'dSPACE', 'eSPACE', 'to', 'ar', 'SPACEm', 'st', 'ty', 've', 'SPACEn', 'th', 'ha', 'hi', 'SPACEo', 'te', 'ol', 'll', 'se', 'SPACEi', 'oSPACE', 'SPACEs', 'er', 'at', 'SPACEt', 're', 'SPACEa', 'SPACEw', 'ec']

feature_list_Tablet_KIT_3 = ['or', 'ySPACE', 'er', 'si', 'ot', 'en', 'nSPACE', 'dSPACE', 'eSPACE', 'tSPACE', 'nt', 'lSPACE', 'no', 'sSPACE', 'SPACEl', 'SPACEp', 'ff', 'co', 'in', 'is', 'me', 'nd', 'ol', 'SPACEf', 'on', 'hi', 'to', 'it', 'he', 'ar', 'te', 'SPACEn', 'ty', 'SPACEBACKSPACE', 'BACKSPACEBACKSPACE', 'ne', 'SPACEm', 'st', 'SPACEc', 'th', 'SPACEt', 'at', 've', 'SPACEo', 'SPACEi', 'ha', 'll', 'oSPACE', 're', 'SPACEs', 'SPACEa', 'SPACEw', 'se', 'ec']

feature_list_Tablet_KIT_4 = ['si', 'BACKSPACEBACKSPACE', 'nSPACE', 'ff', 'ySPACE', 'or', 'en', 'hi', 'nt', 'SPACEc', 'ot', 'ar', 'me', 'nd', 'no', 'is', 'lSPACE', 'on', 'co', 'tSPACE', 'SPACEl', 've', 'st', 'ha', 'ol', 'te', 'in', 'SPACEf', 'SPACEBACKSPACE', 'it', 'th', 'to', 'SPACEn', 'ty', 'll', 'SPACEm', 'he', 'er', 'dSPACE', 'SPACEp', 'oSPACE', 'SPACEo', 'ne', 'at', 'eSPACE', 'ec', 'SPACEs', 'sSPACE', 'se', 're', 'SPACEi', 'SPACEa', 'SPACEt', 'SPACEw']

"""## **Word level Features (Desktop Only)**"""

feature_dict_advanced_word_Desktop = {'if': [2, 1, 3, 4, 6, 0, 13, 15, 10, 12, 7, 9], 'this': [6, 1, 3, 2, 8, 10, 4, 5, 12, 13, 15, 0, 7, 9, 14, 11], 'have': [6, 13, 3, 1, 0, 7, 4, 8, 15, 5, 12, 10, 9, 14, 2, 11], 'me': [4, 6, 0, 13, 15, 2, 7, 9, 10, 12, 1, 3], 'with': [6, 13, 9, 0, 12, 4, 10, 15, 7, 1, 5, 11, 8, 2, 14, 3], 'to': [4, 6, 0, 13, 15, 10, 12, 7, 9, 1, 3, 2], 'sentences': [13, 6, 5, 8, 11, 14, 3, 4, 10, 0, 1, 7, 15, 2, 12, 9], 'not': [1, 2, 3, 4, 6, 10, 12, 13, 15, 8, 11, 0, 7, 9, 5, 14], 'type': [1, 3, 11, 13, 5, 14, 4, 10, 2, 8, 12, 7, 0, 6, 15, 9], 'words': [11, 14, 8, 5, 1, 6, 9, 4, 13, 15, 3, 7, 0, 12, 10, 2], 'will': [1, 11, 5, 3, 15, 2, 13, 14, 12, 4, 9, 10, 8, 0, 6, 7], 'carefully': [1, 4, 3, 10, 0, 15, 14, 7, 13, 5, 8, 11, 9, 12, 6, 2], 'different': [6, 3, 8, 15, 9, 1, 12, 5, 14, 4, 11, 7, 13, 10, 0, 2], 'two': [1, 14, 11, 3, 8, 5, 13, 15, 7, 9, 10, 12, 4, 6, 0, 2], 'see': [14, 8, 13, 15, 3, 0, 1, 10, 12, 5, 7, 9, 4, 6, 2, 11], 'first': [1, 5, 8, 14, 6, 4, 10, 3, 0, 15, 12, 7, 2, 9, 13, 11], 'sample': [4, 5, 8, 1, 14, 13, 15, 10, 3, 0, 11, 2, 7, 6, 12, 9], 'sets': [6, 3, 1, 9, 10, 12, 7, 5, 4, 13, 8, 15, 0, 11, 14, 2], 'that': [6, 1, 3, 7, 0, 8, 10, 2, 9, 14, 5, 12, 4, 13, 15, 11], 'overlap': [1, 10, 14, 15, 6, 3, 8, 9, 11, 2, 0, 7, 5, 13, 12, 4], 'collection': [15, 6, 10, 1, 12, 14, 7, 8, 13, 4, 11, 0, 9, 3, 5, 2], 'is': [4, 6, 2, 10, 12, 1, 3, 7, 9, 0, 13, 15], 'there': [1, 3, 2, 6, 4, 8, 5, 11, 15, 10, 14, 7, 9, 13, 0, 12], 'data': [1, 3, 8, 2, 4, 5, 7, 9, 13, 0, 6, 10, 11, 12, 15, 14], 'of': [1, 3, 4, 6, 10, 12, 0, 13, 15, 7, 9, 2], 'test': [13, 11, 0, 14, 15, 5, 3, 1, 7, 4, 8, 6, 2, 10, 12, 9], 'are': [2, 5, 13, 15, 4, 6, 8, 3, 10, 12, 0, 11, 1, 14, 7, 9], 'lines': [6, 11, 5, 15, 4, 13, 10, 3, 8, 14, 12, 1, 0, 9, 2, 7], 'the': [1, 4, 6, 2, 8, 10, 12, 7, 9, 3, 13, 15, 0, 11, 5, 14], 'in': [1, 3, 4, 6, 0, 13, 15, 7, 9, 10, 12, 2], 'selected': [13, 0, 15, 5, 1, 6, 9, 3, 14, 12, 10, 8, 7, 11, 4, 2], 'a': [0, 1, 3], 'i': [0, 1, 3]}

"""## **Word level Features (Phone Only)**"""

feature_dict_advanced_word_Phone = {'sample': [8, 14, 11, 5, 10, 7, 13, 0, 4, 2, 1, 6, 9, 3, 12, 15], 'data': [10, 7, 0, 13, 12, 4, 15, 2, 9, 8, 1, 3, 6, 5, 14, 11], 'is': [4, 6, 2, 7, 9, 10, 12, 0, 13, 15, 1, 3], 'the': [0, 13, 15, 2, 7, 9, 8, 11, 14, 10, 12, 5, 4, 6, 3, 1], 'this': [4, 13, 2, 1, 3, 15, 0, 11, 5, 10, 7, 12, 14, 6, 9, 8], 'of': [0, 13, 15, 7, 9, 10, 12, 4, 6, 1, 3, 2], 'are': [2, 1, 3, 0, 4, 6, 13, 15, 8, 10, 12, 5, 14, 11, 7, 9], 'to': [7, 9, 4, 6, 0, 13, 15, 10, 12, 1, 3, 2], 'if': [0, 13, 15, 2, 10, 12, 7, 9, 4, 6, 1, 3], 'a': [0, 1, 3], 'i': [0, 1, 3]}

"""## **Word level Features (Tablet Only)**"""

feature_dict_advanced_word_Tablet = {'to': [2, 7, 9, 4, 6, 10, 12, 1, 3, 0, 13, 15], 'the': [2, 10, 12, 11, 4, 6, 1, 7, 9, 14, 0, 3, 5, 8, 13, 15]}

"""## **Extracting top 25 Features (Desktop Only)**"""

'''0.07513 ./KDE_plots/Desktop/Gender/std_kht/std_kht_if.png if std_kht
0.07369 ./KDE_plots/Desktop/Gender/median_f1/median_f1_this.png this median_f1
0.06316 ./KDE_plots/Desktop/Gender/median_f1/median_f1_have.png have median_f1
0.06005 ./KDE_plots/Desktop/Gender/avg_f1/avg_f1_me.png me avg_f1
0.06005 ./KDE_plots/Desktop/Gender/median_f1/median_f1_me.png me median_f1
0.05933 ./KDE_plots/Desktop/Gender/median_f1/median_f1_with.png with median_f1
0.05066 ./KDE_plots/Desktop/Gender/avg_f1/avg_f1_to.png to avg_f1
0.05066 ./KDE_plots/Desktop/Gender/median_f1/median_f1_to.png to median_f1
0.04587 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_if.png if avg_kht
0.04587 ./KDE_plots/Desktop/Gender/median_kht/median_kht_if.png if median_kht
0.04518 ./KDE_plots/Desktop/Gender/avg_f4/avg_f4_sentences.png sentences avg_f4
0.04372 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_not.png not avg_kht
0.04145 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_this.png this avg_kht
0.04096 ./KDE_plots/Desktop/Gender/avg_f1/avg_f1_if.png if avg_f1
0.04096 ./KDE_plots/Desktop/Gender/median_f1/median_f1_if.png if median_f1
0.04087 ./KDE_plots/Desktop/Gender/median_f1/median_f1_sentences.png sentences median_f1
0.04017 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_type.png type avg_kht
0.03882 ./KDE_plots/Desktop/Gender/std_f3/std_f3_words.png words std_f3
0.03828 ./KDE_plots/Desktop/Gender/std_f4/std_f4_words.png words std_f4
0.03801 ./KDE_plots/Desktop/Gender/std_f2/std_f2_words.png words std_f2
0.03673 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_will.png will avg_kht
0.03597 ./KDE_plots/Desktop/Gender/median_kht/median_kht_this.png this median_kht
0.03567 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_carefully.png carefully avg_kht
0.03547 ./KDE_plots/Desktop/Gender/wht/wht_if.png if wht
0.03547 ./KDE_plots/Desktop/Gender/avg_f4/avg_f4_if.png if avg_f4'''
word_feature_id_mapping = {1: 'wht', 2: 'avg_kht', 3: 'std_kht', 4: 'median_kht', 5: 'avg_f1', 6: 'std_f1', 7: 'median_f1', 8: 'avg_f2', 9: 'std_f2', 10: 'median_f2', 11: 'avg_f3', 12: 'std_f3', 13: 'median_f3', 14: 'avg_f4', 15: 'std_f4', 16: 'median_f4'}

top_feature_advanced_word_Desktop_only_map = {}
top_feature_advanced_word_Desktop_only_map["if"] = [2, 1, 3, 4, 6, 0, 13]
top_feature_advanced_word_Desktop_only_map["this"] = [6, 1, 3]
top_feature_advanced_word_Desktop_only_map["have"] = [6]
top_feature_advanced_word_Desktop_only_map["me"] = [4, 6]
top_feature_advanced_word_Desktop_only_map["with"] = [6]
top_feature_advanced_word_Desktop_only_map["to"] = [4, 6]
top_feature_advanced_word_Desktop_only_map["sentences"] = [13, 6]
top_feature_advanced_word_Desktop_only_map["not"] = [1]
top_feature_advanced_word_Desktop_only_map["type"] = [1]
top_feature_advanced_word_Desktop_only_map["words"] = [11, 8, 14]
top_feature_advanced_word_Desktop_only_map["will"] = [1]
top_feature_advanced_word_Desktop_only_map["carefully"] = [1]

"""## **Extracting top 25 Features (Phone Only)**"""

'''0.13533 ./KDE_plots/Phone/Gender/std_f2/std_f2_sample.png sample std_f2
0.13211 ./KDE_plots/Phone/Gender/std_f4/std_f4_sample.png sample std_f4
0.12659 ./KDE_plots/Phone/Gender/std_f3/std_f3_sample.png sample std_f3
0.12489 ./KDE_plots/Phone/Gender/std_f1/std_f1_sample.png sample std_f1
0.07055 ./KDE_plots/Phone/Gender/avg_f3/avg_f3_sample.png sample avg_f3
0.06776 ./KDE_plots/Phone/Gender/avg_f2/avg_f2_sample.png sample avg_f2
0.06308 ./KDE_plots/Phone/Gender/avg_f4/avg_f4_sample.png sample avg_f4
0.05921 ./KDE_plots/Phone/Gender/wht/wht_sample.png sample wht
0.05472 ./KDE_plots/Phone/Gender/avg_f1/avg_f1_sample.png sample avg_f1
0.03678 ./KDE_plots/Phone/Gender/avg_f3/avg_f3_data.png data avg_f3
0.03271 ./KDE_plots/Phone/Gender/avg_f2/avg_f2_data.png data avg_f2
0.03012 ./KDE_plots/Phone/Gender/avg_f1/avg_f1_is.png is avg_f1
0.03012 ./KDE_plots/Phone/Gender/median_f1/median_f1_is.png is median_f1
0.02885 ./KDE_plots/Phone/Gender/wht/wht_data.png data wht
0.02593 ./KDE_plots/Phone/Gender/avg_f4/avg_f4_data.png data avg_f4
0.02493 ./KDE_plots/Phone/Gender/median_f3/median_f3_data.png data median_f3
0.02488 ./KDE_plots/Phone/Gender/std_kht/std_kht_is.png is std_kht
0.02463 ./KDE_plots/Phone/Gender/avg_f1/avg_f1_data.png data avg_f1
0.02393 ./KDE_plots/Phone/Gender/wht/wht_the.png the wht
0.02333 ./KDE_plots/Phone/Gender/avg_f4/avg_f4_the.png the avg_f4
0.02333 ./KDE_plots/Phone/Gender/median_f4/median_f4_the.png the median_f4
0.02216 ./KDE_plots/Phone/Gender/avg_f2/avg_f2_is.png is avg_f2
0.02216 ./KDE_plots/Phone/Gender/median_f2/median_f2_is.png is median_f2
0.02151 ./KDE_plots/Phone/Gender/avg_f3/avg_f3_is.png is avg_f3
0.02151 ./KDE_plots/Phone/Gender/median_f3/median_f3_is.png is median_f3'''


word_feature_id_mapping = {1: 'wht', 2: 'avg_kht', 3: 'std_kht', 4: 'median_kht', 5: 'avg_f1', 6: 'std_f1', 7: 'median_f1', 8: 'avg_f2', 9: 'std_f2', 10: 'median_f2', 11: 'avg_f3', 12: 'std_f3', 13: 'median_f3', 14: 'avg_f4', 15: 'std_f4', 16: 'median_f4'}

top_feature_advanced_word_Phone_only_map = {}
top_feature_advanced_word_Phone_only_map["sample"] = [0, 4, 5, 8, 11, 14, 7, 10, 13]
top_feature_advanced_word_Phone_only_map["data"] = [0, 4, 7, 10, 13, 12]
top_feature_advanced_word_Phone_only_map["is"] = [2, 4, 6, 7, 9, 10, 12]
top_feature_advanced_word_Phone_only_map["the"] = [0, 13, 15]

"""## **Extracting top 25 Features (Tablet Only)**"""

'''0.02673 ./KDE_plots/Tablet/Gender/std_kht/std_kht_to.png to std_kht
0.0128 ./KDE_plots/Tablet/Gender/std_kht/std_kht_the.png the std_kht
0.01215 ./KDE_plots/Tablet/Gender/avg_f3/avg_f3_the.png the avg_f3
0.01215 ./KDE_plots/Tablet/Gender/median_f3/median_f3_the.png the median_f3
0.01017 ./KDE_plots/Tablet/Gender/avg_f2/avg_f2_to.png to avg_f2
0.01017 ./KDE_plots/Tablet/Gender/median_f2/median_f2_to.png to median_f2
0.00932 ./KDE_plots/Tablet/Gender/avg_f1/avg_f1_to.png to avg_f1
0.00932 ./KDE_plots/Tablet/Gender/median_f1/median_f1_to.png to median_f1
0.00847 ./KDE_plots/Tablet/Gender/std_f3/std_f3_the.png the std_f3
0.00846 ./KDE_plots/Tablet/Gender/avg_f1/avg_f1_the.png the avg_f1
0.00846 ./KDE_plots/Tablet/Gender/median_f1/median_f1_the.png the median_f1
0.00629 ./KDE_plots/Tablet/Gender/avg_f3/avg_f3_to.png to avg_f3
0.00629 ./KDE_plots/Tablet/Gender/median_f3/median_f3_to.png to median_f3
0.00601 ./KDE_plots/Tablet/Gender/avg_kht/avg_kht_the.png the avg_kht
0.00545 ./KDE_plots/Tablet/Gender/avg_f2/avg_f2_the.png the avg_f2
0.00545 ./KDE_plots/Tablet/Gender/median_f2/median_f2_the.png the median_f2
0.00509 ./KDE_plots/Tablet/Gender/std_f4/std_f4_the.png the std_f4
0.00492 ./KDE_plots/Tablet/Gender/wht/wht_the.png the wht
0.00447 ./KDE_plots/Tablet/Gender/median_kht/median_kht_the.png the median_kht
0.0044 ./KDE_plots/Tablet/Gender/std_f1/std_f1_the.png the std_f1
0.00418 ./KDE_plots/Tablet/Gender/std_f2/std_f2_the.png the std_f2
0.00383 ./KDE_plots/Tablet/Gender/avg_kht/avg_kht_to.png to avg_kht
0.00383 ./KDE_plots/Tablet/Gender/median_kht/median_kht_to.png to median_kht
0.00328 ./KDE_plots/Tablet/Gender/wht/wht_to.png to wht
0.00328 ./KDE_plots/Tablet/Gender/avg_f4/avg_f4_to.png to avg_f4'''

word_feature_id_mapping = {1: 'wht', 2: 'avg_kht', 3: 'std_kht', 4: 'median_kht', 5: 'avg_f1', 6: 'std_f1', 7: 'median_f1', 8: 'avg_f2', 9: 'std_f2', 10: 'median_f2', 11: 'avg_f3', 12: 'std_f3', 13: 'median_f3', 14: 'avg_f4', 15: 'std_f4', 16: 'median_f4'}

top_feature_advanced_word_Tablet_only_map = {}
top_feature_advanced_word_Tablet_only_map["to"] = [0, 1, 2, 3, 4, 6, 7, 9, 10, 12, 13]
top_feature_advanced_word_Tablet_only_map["the"] = [0, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]

"""## **Extracting top 25 Features (Combined = Desktop + Phone)**"""

'''0.13533 ./KDE_plots/Phone/Gender/std_f2/std_f2_sample.png sample std_f2
0.13211 ./KDE_plots/Phone/Gender/std_f4/std_f4_sample.png sample std_f4
0.12659 ./KDE_plots/Phone/Gender/std_f3/std_f3_sample.png sample std_f3
0.12489 ./KDE_plots/Phone/Gender/std_f1/std_f1_sample.png sample std_f1
0.07513 ./KDE_plots/Desktop/Gender/std_kht/std_kht_if.png if std_kht
0.07369 ./KDE_plots/Desktop/Gender/median_f1/median_f1_this.png this median_f1
0.07055 ./KDE_plots/Phone/Gender/avg_f3/avg_f3_sample.png sample avg_f3
0.06776 ./KDE_plots/Phone/Gender/avg_f2/avg_f2_sample.png sample avg_f2
0.06316 ./KDE_plots/Desktop/Gender/median_f1/median_f1_have.png have median_f1
0.06308 ./KDE_plots/Phone/Gender/avg_f4/avg_f4_sample.png sample avg_f4
0.06005 ./KDE_plots/Desktop/Gender/avg_f1/avg_f1_me.png me avg_f1
0.06005 ./KDE_plots/Desktop/Gender/median_f1/median_f1_me.png me median_f1
0.05933 ./KDE_plots/Desktop/Gender/median_f1/median_f1_with.png with median_f1
0.05921 ./KDE_plots/Phone/Gender/wht/wht_sample.png sample wht
0.05472 ./KDE_plots/Phone/Gender/avg_f1/avg_f1_sample.png sample avg_f1
0.05066 ./KDE_plots/Desktop/Gender/avg_f1/avg_f1_to.png to avg_f1
0.05066 ./KDE_plots/Desktop/Gender/median_f1/median_f1_to.png to median_f1
0.04587 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_if.png if avg_kht
0.04587 ./KDE_plots/Desktop/Gender/median_kht/median_kht_if.png if median_kht
0.04518 ./KDE_plots/Desktop/Gender/avg_f4/avg_f4_sentences.png sentences avg_f4
0.04372 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_not.png not avg_kht
0.04145 ./KDE_plots/Desktop/Gender/avg_kht/avg_kht_this.png this avg_kht
0.04096 ./KDE_plots/Desktop/Gender/avg_f1/avg_f1_if.png if avg_f1
0.04096 ./KDE_plots/Desktop/Gender/median_f1/median_f1_if.png if median_f1
0.04087 ./KDE_plots/Desktop/Gender/median_f1/median_f1_sentences.png sentences median_f1'''
word_feature_id_mapping = {1: 'wht', 2: 'avg_kht', 3: 'std_kht', 4: 'median_kht', 5: 'avg_f1',
                           6: 'std_f1', 7: 'median_f1', 8: 'avg_f2', 9: 'std_f2', 10: 'median_f2',
                           11: 'avg_f3', 12: 'std_f3', 13: 'median_f3', 14: 'avg_f4', 15: 'std_f4', 
                           16: 'median_f4'}


top_feature_advanced_word_Desktop_map_DP = {}
top_feature_advanced_word_Desktop_map_DP["if"] = [1, 2, 3, 4, 6]
top_feature_advanced_word_Desktop_map_DP["this"] = [1, 6]
top_feature_advanced_word_Desktop_map_DP["have"] = [6]
top_feature_advanced_word_Desktop_map_DP["me"] = [4, 6]
top_feature_advanced_word_Desktop_map_DP["with"] = [6]
top_feature_advanced_word_Desktop_map_DP["to"] = [4, 6]
top_feature_advanced_word_Desktop_map_DP["sentences"] = [6, 13]
top_feature_advanced_word_Desktop_map_DP["not"] = [1]

top_feature_advanced_word_Phone_map_DP = {}
top_feature_advanced_word_Phone_map_DP["sample"] = [0, 4, 5, 7, 8, 10, 11, 13, 14]

"""## **Code to extract top 25 word level Features (Combined)**"""

# List of top 25 features and their equivalent position
top_feature_advanced_word_Desktop_map = {}
top_feature_advanced_word_Desktop_map["selected"] = [0,7,10]
top_feature_advanced_word_Desktop_map["me"] = [4,6]
top_feature_advanced_word_Desktop_map["if"] = [4,6]
top_feature_advanced_word_Desktop_map["that"] = [4]
top_feature_advanced_word_Desktop_map["sample"] = [4]
top_feature_advanced_word_Desktop_map["test"] = [4]
top_feature_advanced_word_Desktop_map["have"] = [4]
top_feature_advanced_word_Desktop_map["data"] = [1,4]
top_feature_advanced_word_Desktop_map["with"] = [13]
top_feature_advanced_word_Desktop_map["will"] = [13]

top_feature_advanced_word_Phone_map = {}
top_feature_advanced_word_Phone_map["sample"] = [7,14]

top_feature_advanced_word_Tablet_map = {}
top_feature_advanced_word_Tablet_map["have"] = [4,6,9,12]
top_feature_advanced_word_Tablet_map["is"] = [4,6,10,12]

# Calculated median of required features
def top_feature_advanced_word(pickle_file, feature_dict):
    temp_features = pickle.load(open(pickle_file, 'rb'))
    selected_top_feature = []
    top_feature_list = [[0 for x in range(len(temp_features))] for x in range(116)] 
    #for key in feature_dict:
    for key1 in temp_features:
        if key1 == 117:
            break
        temp = [] 
        for key2 in temp_features[key1]:
            for key in feature_dict:
                if key2 == key:              
                    for i in (feature_dict[key]):
                        # Removing outliers
                        temp_without_outlier = remove_outliers(np.array(temp_features[key1][key2])[:,i])

                        # Median feature
                        #temp.append(np.median(temp_without_outlier))

                        '''# Mean feature
                        temp.append(np.mean(temp_without_outlier))

                        #IQR feature
                        a = np.asarray(temp_without_outlier)
                        upper_quartile = np.percentile(a, 75)
                        lower_quartile = np.percentile(a, 25)
                        IQR = (upper_quartile - lower_quartile) * 1.5
                        temp.append(IQR)
                        temp.append(upper_quartile)
                        temp.append(lower_quartile)

                        # Kurtosis feature
                        temp.append(kurtosis(temp_without_outlier))

                        # Skew features
                        temp.append(skew(temp_without_outlier))'''

                        # Mean-Median
                        temp.append(np.mean(temp_without_outlier) - np.median(temp_without_outlier))
                    break
            top_feature_list[key1-1] = list(temp)          
    return top_feature_list

# Combines data from multiple devices (desktop, phone, tablet)
def combine_top_advanced_word():
    desktop_features = top_feature_advanced_word('desktop_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Desktop_map)
    phone_features = top_feature_advanced_word('phone_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Phone_map)
    tablet_features = top_feature_advanced_word('tablet_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Tablet_map)
    return np.concatenate((np.array(desktop_features), np.array(phone_features), np.array(tablet_features)), axis=1)

def combine_top_advanced_word_desktop_only():
    desktop_features = top_feature_advanced_word('desktop_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Desktop_only_map)
    return np.array(desktop_features)

def combine_top_advanced_word_phone_only():
    desktop_features = top_feature_advanced_word('desktop_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Phone_only_map)
    return np.array(desktop_features)

def combine_top_advanced_word_tablet_only():
    desktop_features = top_feature_advanced_word('desktop_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Tablet_only_map)
    return np.array(desktop_features)

def combine_top_advanced_word_desktop_and_phone_only():
    desktop_features = top_feature_advanced_word('desktop_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Desktop_map_DP)
    phone_features = top_feature_advanced_word('phone_advanced_word_feature_dictionary.pickle', top_feature_advanced_word_Phone_map_DP)
    return np.concatenate((np.array(desktop_features), np.array(phone_features)), axis=1)

# utilities to get the most commonly occurring words and corresponding features for all users
word_feature_id_mapping = {1: 'wht', 2: 'avg_kht', 3: 'std_kht', 4: 'median_kht', 5: 'avg_f1', 6: 'std_f1', 7: 'median_f1', 8: 'avg_f2', 9: 'std_f2', 10: 'median_f2', 11: 'avg_f3', 12: 'std_f3', 13: 'median_f3', 14: 'avg_f4', 15: 'std_f4', 16: 'median_f4'}

def get_key_array_for_user(user_key, user_dict):
    key_array_lengths = []
    for key in user_dict[user_key]:
        key_array_lengths.append((len(user_dict[user_key][key]), key))
    final_keys_array = []
    for key in key_array_lengths:
        final_keys_array.append(key[1])
    return final_keys_array

def get_top_word_keys(device):
    advanced_word_feat_dict = pickle.load(open(device+'_advanced_word_feature_dictionary.pickle', 'rb'))
    final_key_set = get_key_array_for_user(1, advanced_word_feat_dict)
    for user_id in advanced_word_feat_dict:
        user_key_array = get_key_array_for_user(user_id, advanced_word_feat_dict)
        final_key_set = set(user_key_array).intersection(final_key_set)
    return final_key_set

def get_advanced_word_values_given_user_and_key(advanced_word_feat_dict, user_key, key, word_feat_id):
    try:
        temp = list(np.asarray(advanced_word_feat_dict[user_key+1][key])[:, word_feat_id])
        for k in temp:
            if(math.isnan(k)):
                return int(0)
        return abs(np.median(np.asarray(advanced_word_feat_dict[user_key+1][key])[:, word_feat_id]))
    except KeyError as e:
        return None

def get_advanced_word_features(device):
    """ Input: All feature dictionary Output: Feature matrix with unique columns"""
    features = pickle.load(open(device+'_advanced_word_feature_dictionary.pickle', 'rb'))

    # Getting unique columns by removing repeated keys
    feature_set = list(get_top_word_keys(device))
    val = []
    rows, cols = (len(features), len(feature_set))
    feature_vector = [[0 for x in range(cols * 16)] for x in range(rows)] 

    for i, key in enumerate(feature_set):
        for j in range(16):
            val.append(get_advanced_word_values_given_user_and_key(features, i, key, j))
        feature_vector[i] = val

    return feature_vector

"""## **Feature Matrix for Desktop (KHT (Unigraph) + KIT (Digraph) + Word)**"""

# return all the desktop features for free text
def get_desktop_features():
    desktop_features_KHT = top_feature_KIT("desktop_kht_feature_dictionary.pickle",feature_list_Desktop_KHT)
    desktop_features_KIT_1 = top_feature_KIT("desktop_kit_feature_f1_dictionary.pickle", feature_list_Desktop_KIT_1)
    desktop_features_KIT_2 = top_feature_KIT("desktop_kit_feature_f2_dictionary.pickle", feature_list_Desktop_KIT_2)
    desktop_features_KIT_3 = top_feature_KIT("desktop_kit_feature_f3_dictionary.pickle", feature_list_Desktop_KIT_3)
    desktop_features_KIT_4 = top_feature_KIT("desktop_kit_feature_f4_dictionary.pickle", feature_list_Desktop_KIT_4)

    desktop_features_advanced = top_feature_advanced_word('desktop_advanced_word_feature_dictionary.pickle', feature_dict_advanced_word_Desktop)
    return np.concatenate((np.array(desktop_features_KHT), np.array(desktop_features_advanced), np.array(desktop_features_KIT_1), np.array(desktop_features_KIT_2), np.array(desktop_features_KIT_3), np.array(desktop_features_KIT_4)), axis=1)

# return all the desktop features for fixed text
def get_desktop_features_fixed():
    desktop_features_KHT_fixed = top_feature_KIT("desktop_kht_feature_dictionary_fixed.pickle",feature_list_Desktop_KHT)
    desktop_features_KIT_1_fixed = top_feature_KIT("desktop_kit_feature_f1_dictionary_fixed.pickle", feature_list_Desktop_KIT_1)
    desktop_features_KIT_2_fixed = top_feature_KIT("desktop_kit_feature_f2_dictionary_fixed.pickle", feature_list_Desktop_KIT_2)
    desktop_features_KIT_3_fixed = top_feature_KIT("desktop_kit_feature_f3_dictionary_fixed.pickle", feature_list_Desktop_KIT_3)
    desktop_features_KIT_4_fixed = top_feature_KIT("desktop_kit_feature_f4_dictionary_fixed.pickle", feature_list_Desktop_KIT_4)

    desktop_features_advanced_fixed = top_feature_advanced_word('desktop_advanced_word_feature_dictionary_fixed.pickle', feature_dict_advanced_word_Desktop)
    return np.concatenate((np.array(desktop_features_KHT_fixed), np.array(desktop_features_advanced_fixed), np.array(desktop_features_KIT_1_fixed), np.array(desktop_features_KIT_2_fixed), np.array(desktop_features_KIT_3_fixed), np.array(desktop_features_KIT_4_fixed)), axis=1)

"""## **Feature Matrix for Phone (KHT (Unigraph) + KIT (Digraph) + Word)**"""

# return all the phone features for free text
def get_phone_features():
    phone_features_KHT = top_feature_KIT("phone_kht_feature_dictionary.pickle",feature_list_Phone_KHT)
    phone_features_KIT_1 = top_feature_KIT("phone_kit_feature_f1_dictionary.pickle", feature_list_Phone_KIT_1)
    phone_features_KIT_2 = top_feature_KIT("phone_kit_feature_f2_dictionary.pickle", feature_list_Phone_KIT_2)
    phone_features_KIT_3 = top_feature_KIT("phone_kit_feature_f3_dictionary.pickle", feature_list_Phone_KIT_3)
    phone_features_KIT_4 = top_feature_KIT("phone_kit_feature_f4_dictionary.pickle", feature_list_Phone_KIT_4)

    phone_features_advanced = top_feature_advanced_word('phone_advanced_word_feature_dictionary.pickle', feature_dict_advanced_word_Phone)
    return np.concatenate((np.array(phone_features_KHT), np.array(phone_features_advanced), np.array(phone_features_KIT_1), np.array(phone_features_KIT_2), np.array(phone_features_KIT_3), np.array(phone_features_KIT_4)), axis=1)

# return all the phone features for fixed text
def get_phone_features_fixed():
    phone_features_KHT_fixed = top_feature_KIT("phone_kht_feature_dictionary_fixed.pickle",feature_list_Phone_KHT)
    phone_features_KIT_1_fixed = top_feature_KIT("phone_kit_feature_f1_dictionary_fixed.pickle", feature_list_Phone_KIT_1)
    phone_features_KIT_2_fixed = top_feature_KIT("phone_kit_feature_f2_dictionary_fixed.pickle", feature_list_Phone_KIT_2)
    phone_features_KIT_3_fixed = top_feature_KIT("phone_kit_feature_f3_dictionary_fixed.pickle", feature_list_Phone_KIT_3)
    phone_features_KIT_4_fixed = top_feature_KIT("phone_kit_feature_f4_dictionary_fixed.pickle", feature_list_Phone_KIT_4)

    phone_features_advanced_fixed = top_feature_advanced_word('phone_advanced_word_feature_dictionary.pickle', feature_dict_advanced_word_Phone)
    return np.concatenate((np.array(phone_features_KHT_fixed), np.array(phone_features_advanced_fixed), np.array(phone_features_KIT_1_fixed), np.array(phone_features_KIT_2_fixed), np.array(phone_features_KIT_3_fixed), np.array(phone_features_KIT_4_fixed)), axis=1)

"""## **Feature Matrix for Tablet (KHT (Unigraph) + KIT (Digraph) + Word)**"""

# return all the tablet features for free text
def get_tablet_features():
    tablet_features_KHT = top_feature_KIT("phone_kht_feature_dictionary.pickle",feature_list_Tablet_KHT)
    tablet_features_KIT_1 = top_feature_KIT("phone_kit_feature_f1_dictionary.pickle", feature_list_Tablet_KIT_1)
    tablet_features_KIT_2 = top_feature_KIT("phone_kit_feature_f2_dictionary.pickle", feature_list_Tablet_KIT_2)
    tablet_features_KIT_3 = top_feature_KIT("phone_kit_feature_f3_dictionary.pickle", feature_list_Tablet_KIT_3)
    tablet_features_KIT_4 = top_feature_KIT("phone_kit_feature_f4_dictionary.pickle", feature_list_Tablet_KIT_4)

    tablet_features_advanced = top_feature_advanced_word('tablet_advanced_word_feature_dictionary.pickle', feature_dict_advanced_word_Tablet)
    return np.concatenate((np.array(tablet_features_KHT), np.array(tablet_features_advanced), np.array(tablet_features_KIT_1), np.array(tablet_features_KIT_2), np.array(tablet_features_KIT_3), np.array(tablet_features_KIT_4)), axis=1)

# return all the tablet features for fixed text
def get_tablet_features_fixed():
    tablet_features_KHT_fixed = top_feature_KIT("phone_kht_feature_dictionary_fixed.pickle",feature_list_Tablet_KHT)
    tablet_features_KIT_1_fixed = top_feature_KIT("phone_kit_feature_f1_dictionary_fixed.pickle", feature_list_Tablet_KIT_1)
    tablet_features_KIT_2_fixed = top_feature_KIT("phone_kit_feature_f2_dictionary_fixed.pickle", feature_list_Tablet_KIT_2)
    tablet_features_KIT_3_fixed = top_feature_KIT("phone_kit_feature_f3_dictionary_fixed.pickle", feature_list_Tablet_KIT_3)
    tablet_features_KIT_4_fixed = top_feature_KIT("phone_kit_feature_f4_dictionary_fixed.pickle", feature_list_Tablet_KIT_4)

    tablet_features_advanced_fixed = top_feature_advanced_word('tablet_advanced_word_feature_dictionary.pickle', feature_dict_advanced_word_Tablet)
    return np.concatenate((np.array(tablet_features_KHT_fixed), np.array(tablet_features_advanced_fixed), np.array(tablet_features_KIT_1_fixed), np.array(tablet_features_KIT_2_fixed), np.array(tablet_features_KIT_3_fixed), np.array(tablet_features_KIT_4_fixed)), axis=1)

"""## **Feature Matrix for Combined (KHT (Unigraph) + KIT (Digraph) + Word)**"""

# return all the combined (desktop+phone+tablet) features for free text
def get_combined_features():
    desktop_features_combined = get_desktop_features()
    phone_features_combined = get_phone_features()
    tablet_features_combined = get_tablet_features()
    return np.concatenate((np.array(desktop_features_combined), np.array(phone_features_combined), np.array(tablet_features_combined)), axis = 1)

# return all the combined (desktop+phone+tablet) features for fixed text
def get_combined_features_fixed():
    desktop_features_combined_fixed = get_desktop_features_fixed()
    phone_features_combined_fixed = get_phone_features_fixed()
    tablet_features_combined_fixed = get_tablet_features_fixed()
    return np.concatenate((np.array(desktop_features_combined_fixed), np.array(phone_features_combined_fixed), np.array(tablet_features_combined_fixed)), axis = 1)

"""# **Machine Learning Models: Classification tasks**"""

!pip install skfeature-chappers

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from skfeature.function.information_theoretical_based import MRMR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

def compare_classification(label_name, feature_type, top_n_features, model):
''' Function to process the data, apply oversampling techniques (SMOTE) and run the classification model specified using GridSearchCV
Input:  label_name: The task to be performed (Gender, Major/Minor, Typing Style)
        feature_type: The feature set to be used (Desktop, Phone, Tablet, Combined)
        top_n_features: Thu number of features to be selected using Mutual Info criterion
        model: The ML model to train and evaluate
Output: accuracy scores, best hyperparameters of the gridsearch run
'''
    # Creating class label vector using metadata
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if(label_name == "Typing Style"):
        for i in range(117):
            if(Y_vector[i] == 'a'):
                Y_vector[i] = 0
            if(Y_vector[i] == 'b'):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 2

    if(label_name == "Major/Minor"):
        string1 = "Computer"
        string2 = "CS"
        for i, v in enumerate(Y_vector):
            if (type(Y_vector[i]) == float):
                Y_vector[i] = 1
                continue
            if "LEFT" in Y_vector[i]:
                Y_vector[i] = 0
            if string1 in v or string2 in v:
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    if (label_name == "Gender" or label_name == "Ethnicity"):
        for i in range(116):
            if(Y_values[i] == 'M' or Y_values[i] == "Asian"):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype('int')

    X_matrix = feature_type

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)

    X_matrix_new = SelectKBest(mutual_info_classif, k=top_n_features).fit_transform(X_matrix, Y_vector)
    X_matrix_new, Y_vector = SMOTE(kind='svm').fit_sample(X_matrix_new, Y_vector)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix_new, Y_vector, test_size=0.3, random_state=0)

    if model == "SVM":    
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                     'C': [0.1, 1, 10, 100, 1000, 10000]},
                     {'kernel': ['poly'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                     'C': [0.1, 1, 10, 100, 1000, 10000]},
                     {'kernel': ['sigmoid'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
                     'C': [0.1, 1, 10, 100, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000,10000]}]

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='accuracy', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "DTree":
        tuned_parameters = {
            'criterion':["gini", "entropy"],
            'splitter': ["best", "random"],
            'max_leaf_nodes': list(range(2, 100)),
            'min_samples_split': [2, 3, 4, 6, 8, 10],
        }
        clf = GridSearchCV(
            DecisionTreeClassifier(random_state=42), tuned_parameters, scoring='accuracy', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "RForest":
        tuned_parameters = {'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400, 600, 800, 1000]
        }
        clf = GridSearchCV(
            RandomForestClassifier(), tuned_parameters, scoring='accuracy', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "XGBoost":
        tuned_parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
        clf = GridSearchCV(
            xgb.XGBClassifier(), tuned_parameters, scoring='accuracy', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "ABoost":
        tuned_parameters = {'n_estimators':[10, 50, 100, 200, 500, 1000],'learning_rate':[0.00001, 0.001, 0.01, 0.1], 'algorithm':["SAMME", "SAMME.R"]}
        clf = GridSearchCV(
            AdaBoostClassifier(), tuned_parameters, scoring='accuracy', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "MLP":
        tuned_parameters = {
            'hidden_layer_sizes': [(10,), (50,),(70,),(90,), (100,), (120,),(150,)],
            'activation': ['tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.01, 0.1],
            'learning_rate': ['constant','adaptive'],
            'max_iter':[100, 1000],
        }
        clf = GridSearchCV(
            MLPClassifier(), tuned_parameters, scoring='accuracy', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "NB":
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        return accuracy_score(y_true, y_pred), None, None

"""### **Run Free Text classification tasks**"""

# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

# function to call the compare_classification function for the specified model, feature_type and task
def classification_results(problem, feature_type, model):
    num_features = []
    accuracy = []
    hyper = []
    val_score = []
    for i in range(5,105,5):
        res, setup, val = compare_classification(problem, feature_type, i, model)
        num_features.append(i)
        accuracy.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(accuracy)
    print(hyper)
    #print(val_score)

class_problems = ["Gender", "Typing Style", "Major/Minor"]
models = ["NB", "ABoost", "SVM", "DTree", "XGBoost", "MLP"]

for model in models:
    print("###########################################################################################")
    print(model)
    for class_problem in class_problems:
        print(class_problem)
        print("Desktop")
        classification_results(class_problem, get_desktop_features(), model)
        print("Phone")
        classification_results(class_problem, get_phone_features(), model)
        print("Tablet")
        classification_results(class_problem, get_tablet_features(), model)
        print("Combined")
        classification_results(class_problem, get_combined_features(), model)
        print()
        print("-----------------------------------------------------------------------------------------")

"""### **Run Fixed Text classification tasks**"""

# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

# function to call the compare_classification function for the specified model, feature_type and task
def classification_results(problem, feature_type, model):
    num_features = []
    accuracy = []
    hyper = []
    val_score = []
    for i in range(5,105,5):
        res, setup, val = compare_classification(problem, feature_type, i, model)
        num_features.append(i)
        accuracy.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(accuracy)
    print(hyper)
    #print(val_score)

class_problems = ["Gender", "Typing Style", "Major/Minor"]
models = ["NB", "SVM", "DTree", "ABoost", "MLP", "XGBoost"]

for model in models:
    print("###########################################################################################")
    print(model)
    for class_problem in class_problems:
        print(class_problem)
        print("Desktop")
        classification_results(class_problem, get_desktop_features_fixed(), model)
        print("Phone")
        classification_results(class_problem, get_phone_features_fixed(), model)
        print("Tablet")
        classification_results(class_problem, get_tablet_features_fixed(), model)
        print("Combined")
        classification_results(class_problem, get_combined_features_fixed(), model)
        print()
        print("-----------------------------------------------------------------------------------------")

"""# **Machine Learning Models: Regression tasks**"""

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

def compare_regression(label_name, feature_type, top_n_features, model):
''' Function to process the data and run the regression model specified using GridSearchCV
Input:  label_name: The task to be performed (Age, Height)
        feature_type: The feature set to be used (Desktop, Phone, Tablet, Combined)
        top_n_features: Thu number of features to be selected using Mutual Info criterion
        model: The ML model to train and evaluate
Output: accuracy scores, best hyperparameters of the gridsearch run
'''
    # Creating class label vector using metadata
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()

    Y_vector = np.asarray(Y_values)

    Y_vector = Y_vector[:-1]
    Y_vector = Y_vector.astype('int')

    X_matrix = feature_type

    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)

    np.random.seed(0)
    X_matrix_new = SelectKBest(mutual_info_classif, k=top_n_features).fit_transform(X_matrix, Y_vector)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix_new, Y_vector, test_size=0.3, random_state=0)

    if model == "SVM":    
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': ['scale','auto'],
                     'C': [0.1, 1, 10, 100, 1000]},
                     {'kernel': ['poly'], 'gamma': ['scale','auto'],
                     'C': [0.1, 1, 10, 100, 1000]},
                     {'kernel': ['sigmoid'], 'gamma': ['scale','auto'],
                     'C': [0.1, 1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

        clf = GridSearchCV(
            SVR(), tuned_parameters, scoring='neg_mean_absolute_error', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "Lasso":    
        # Set the parameters by cross-validation
        tuned_parameters = {
            'alpha': [0.2, 0.4, 0.6, 0.8, 1],
            'selection':['cyclic', 'random'],
        }

        clf = GridSearchCV(
            Lasso(), tuned_parameters, scoring='neg_mean_absolute_error', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_
      
    if model == "Ridge":    
        # Set the parameters by cross-validation
        tuned_parameters = {
            'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01],
        }

        clf = GridSearchCV(
            Ridge(), tuned_parameters, scoring='neg_mean_absolute_error', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_
    
    if model == "KNN":    
        # Set the parameters by cross-validation
        tuned_parameters = {
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20],
        }

        clf = GridSearchCV(
            KNeighborsRegressor(), tuned_parameters, scoring='neg_mean_absolute_error', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

    if model == "XGB":    
        # Set the parameters by cross-validation
        tuned_parameters = {
              'objective':['reg:linear'],
              'learning_rate': [.01, 0.1, .001], #so called `eta` value
              'max_depth': [5, 10, 15, 20],
              'min_child_weight': [4, 8],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.5, 0.7, 1.0],
              'n_estimators': [100, 500, 800]}

        clf = GridSearchCV(
            XGBRegressor(), tuned_parameters, scoring='neg_mean_absolute_error', return_train_score=True
        )
        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        return mean_absolute_error(y_true, y_pred), clf.best_params_, clf.cv_results_

"""### **Run Free Text regression tasks**"""

# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

# function to call the compare_regression function for the specified model, feature_type and task
def regression_results(problem, feature_type, model):
    num_features = []
    mae = []
    hyper = []
    val_score = []
    for i in range(5,105,5):
        res, setup, val = compare_regression(problem, feature_type, i, model)
        num_features.append(i)
        mae.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(mae)
    print(hyper)
    #print(val_score)

class_problems = ["Age", "Height"]
models = ["SVM", "KNN", "XGBoost"]

for model in models:
    print("###########################################################################################")
    print(model)
    for class_problem in class_problems:
        print(class_problem)
        print("Desktop")
        regression_results(class_problem, get_desktop_features(), model)
        print("Phone")
        regression_results(class_problem, get_phone_features(), model)
        print("Tablet")
        regression_results(class_problem, get_tablet_features(), model)
        print("Combined")
        regression_results(class_problem, get_combined_features(), model)
        print()
        print("-----------------------------------------------------------------------------------------")

"""### **Run Fixed Text regression tasks**"""

# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=ConvergenceWarning)

# function to call the compare_regression function for the specified model, feature_type and task
def regression_results(problem, feature_type, model):
    num_features = []
    mae = []
    hyper = []
    val_score = []
    for i in range(5,105,5):
        res, setup, val = compare_regression(problem, feature_type, i, model)
        num_features.append(i)
        mae.append(res)
        hyper.append(setup)
        val_score.append(val)
    print(num_features)
    print(mae)
    print(hyper)
    #print(val_score)

class_problems = ["Age", "Height"]
models = ["SVM", "KNN", "XGBoost"]

for model in models:
    print("###########################################################################################")
    print(model)
    for class_problem in class_problems:
        print(class_problem)
        print("Desktop")
        regression_results(class_problem, get_desktop_features_fixed(), model)
        print("Phone")
        regression_results(class_problem, get_phone_features_fixed(), model)
        print("Tablet")
        regression_results(class_problem, get_tablet_features_fixed(), model)
        print("Combined")
        regression_results(class_problem, get_combined_features_fixed(), model)
        print()
        print("-----------------------------------------------------------------------------------------")

"""# **Deep Learning Models**

## **Data Pre-Processing Utilities**
"""

# Create the appropriate train-test splits for free text classification tasks to align with the ML models 
def get_train_test_splits(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if(label_name == "Typing Style"):
        for i in range(117): 
            if(Y_vector[i] == 'a'):
                Y_vector[i] = 0
            elif(Y_vector[i] == 'b'):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 2

    if(label_name == "Major/Minor"):
        string1 = "Computer"
        string2 = "CS"
        for i, v in enumerate(Y_vector):
            if (type(Y_vector[i]) == float):
                Y_vector[i] = 1
                continue
            if "LEFT" in Y_vector[i]:
                Y_vector[i] = 0
            if string1 in v or string2 in v:
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    if (label_name == "Gender" or label_name == "Ethnicity"):
        for i in range(116):
            if(Y_values[i] == 'M' or Y_values[i] == "Asian"):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype('int')

    #uncomment one of the below four lines for the required feature set
    X_matrix = get_combined_features()
    # X_matrix = get_desktop_features()
    # X_matrix = get_phone_features()
    # X_matrix = get_tablet_features()

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)
    X_matrix_new = X_matrix

    print(X_matrix_new.shape)    
    X_matrix_new, Y_vector = SMOTE(kind='svm').fit_sample(X_matrix_new, Y_vector)
    return X_matrix_new, Y_vector

# Create the appropriate train-test splits for fixed text classification tasks to align with the ML models 
def get_train_test_splits_fixed(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if(label_name == "Typing Style"):
        for i in range(117): 
            if(Y_vector[i] == 'a'):
                Y_vector[i] = 0
            elif(Y_vector[i] == 'b'):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 2

    if(label_name == "Major/Minor"):
        string1 = "Computer"
        string2 = "CS"
        for i, v in enumerate(Y_vector):
            if (type(Y_vector[i]) == float):
                Y_vector[i] = 1
                continue
            if "LEFT" in Y_vector[i]:
                Y_vector[i] = 0
            if string1 in v or string2 in v:
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    if (label_name == "Gender" or label_name == "Ethnicity"):
        for i in range(116):
            if(Y_values[i] == 'M' or Y_values[i] == "Asian"):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype('int')

    # uncomment one of the below four lines for the required feature set
    X_matrix = get_combined_features_fixed()
    # X_matrix = get_desktop_features_fixed()
    # X_matrix = get_phone_features_fixed()
    # X_matrix = get_tablet_features_fixed()

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)
    X_matrix_new = X_matrix

    print(X_matrix_new.shape)    

    X_matrix_new, Y_vector = SMOTE(kind='svm').fit_sample(X_matrix_new, Y_vector)
    return X_matrix_new, Y_vector

# Create the appropriate train-test splits for free text regression tasks to align with the ML models 
def get_train_test_splits_reg(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if(label_name == "Typing Style"):
        for i in range(117): 
            if(Y_vector[i] == 'a'):
                Y_vector[i] = 0
            elif(Y_vector[i] == 'b'):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 2

    if(label_name == "Major/Minor"):
        string1 = "Computer"
        string2 = "CS"
        for i, v in enumerate(Y_vector):
            if (type(Y_vector[i]) == float):
                Y_vector[i] = 1
                continue
            if "LEFT" in Y_vector[i]:
                Y_vector[i] = 0
            if string1 in v or string2 in v:
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    if (label_name == "Gender" or label_name == "Ethnicity"):
        for i in range(116):
            if(Y_values[i] == 'M' or Y_values[i] == "Asian"):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype('int')

    # uncomment one of the below four lines for the required feature set
    X_matrix = get_combined_features()
    # X_matrix = get_desktop_features()
    # X_matrix = get_phone_features()
    # X_matrix = get_tablet_features()

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)
    X_matrix_new = X_matrix

    print(X_matrix_new.shape)    

    return X_matrix_new, Y_vector

# Create the appropriate train-test splits for fixed text regression tasks to align with the ML models 
def get_train_test_splits_reg_fixed(label_name):
    demographics_data_frame = pd.read_csv("Demographics.csv")
    Y_values = demographics_data_frame[label_name].to_numpy()
    Y_vector = np.asarray(Y_values)

    if(label_name == "Typing Style"):
        for i in range(117): 
            if(Y_vector[i] == 'a'):
                Y_vector[i] = 0
            elif(Y_vector[i] == 'b'):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 2

    if(label_name == "Major/Minor"):
        string1 = "Computer"
        string2 = "CS"
        for i, v in enumerate(Y_vector):
            if (type(Y_vector[i]) == float):
                Y_vector[i] = 1
                continue
            if "LEFT" in Y_vector[i]:
                Y_vector[i] = 0
            if string1 in v or string2 in v:
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    if (label_name == "Gender" or label_name == "Ethnicity"):
        for i in range(116):
            if(Y_values[i] == 'M' or Y_values[i] == "Asian"):
                Y_vector[i] = 1
            else:
                Y_vector[i] = 0

    Y_vector = Y_vector[:-1]
    Y_values = Y_values[:-1]
    Y_vector = Y_vector.astype('int')

    # uncomment one of the below four lines for the required feature set
    X_matrix = get_combined_features_fixed()
    # X_matrix = get_desktop_features_fixed()
    # X_matrix = get_phone_features_fixed()
    # X_matrix = get_tablet_features_fixed()

    # Normalizing features and selecting top 20
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    X_matrix = preprocessing.scale(X_matrix)
    np.random.seed(0)
    X_matrix_new = X_matrix

    print(X_matrix_new.shape)    

    return X_matrix_new, Y_vector

"""## **FC network for classification/regression**"""

# A simple four layered NN for classification tasks
class FC_Net(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(FC_Net, self).__init__()
        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)

    def forward(self, feats):
        out = self.dropout_2(self.relu(self.fc1(feats)))
        out = self.dropout_2(self.relu(self.fc2(out)))
        out = self.dropout_2(self.relu(self.fc3(out)))
        out = self.fc4(out)
        return out

# A simple four layered NN for regression tasks
class Reg_FC_Net(nn.Module):
    def __init__(self, input_dims):
        super(Reg_FC_Net, self).__init__()
        self.fc1 = nn.Linear(input_dims,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)

    def forward(self, feats):
        out = self.relu(self.fc1(feats))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        return out

"""### **Run Free Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
 
	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num))
		split_num += 1 

		x_train, x_val = X_train[train_index], X_train[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]

		fcn = FC_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(x_train)
		train_tensor_y = torch.Tensor(y_train)
		val_tensor_x = torch.Tensor(x_val)
		val_tensor_y = torch.Tensor(y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()					

				if(100*val_correct/val_total>50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
 
	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num))
		split_num += 1

		x_train, x_val = X_train[train_index], X_train[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]

		fcn = FC_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(x_train)
		train_tensor_y = torch.Tensor(y_train)
		val_tensor_x = torch.Tensor(x_val)
		val_tensor_y = torch.Tensor(y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()					

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
 
	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num))
		split_num += 1

		x_train, x_val = X_train[train_index], X_train[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]

		fcn = FC_Net(X_train.shape[1], 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(x_train)
		train_tensor_y = torch.Tensor(y_train)
		val_tensor_x = torch.Tensor(x_val)
		val_tensor_y = torch.Tensor(y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()					

				if(100*val_correct/val_total>20):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_FC_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(40):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_FC_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(40):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""### **Run Fixed Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
 
	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num))
		split_num += 1

		x_train, x_val = X_train[train_index], X_train[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]

		fcn = FC_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(x_train)
		train_tensor_y = torch.Tensor(y_train)
		val_tensor_x = torch.Tensor(x_val)
		val_tensor_y = torch.Tensor(y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()					

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
 
	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num))
		split_num += 1

		x_train, x_val = X_train[train_index], X_train[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]

		fcn = FC_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(x_train)
		train_tensor_y = torch.Tensor(y_train)
		val_tensor_x = torch.Tensor(x_val)
		val_tensor_y = torch.Tensor(y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()					

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
 
	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num))
		split_num += 1

		x_train, x_val = X_train[train_index], X_train[val_index]
		y_train, y_val = Y_train[train_index], Y_train[val_index]

		fcn = FC_Net(X_train.shape[1], 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(x_train)
		train_tensor_y = torch.Tensor(y_train)
		val_tensor_x = torch.Tensor(x_val)
		val_tensor_y = torch.Tensor(y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()					

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_FC_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(40):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_FC_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(40):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""## **CNN network for classification/regression**"""

# CNN model for classification tasks
class CNN_Net(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=True)

        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()

    def forward(self, feats):
        out = self.bn1(self.relu(self.conv1(feats)))
        out = self.bn2(self.relu(self.conv2(out)))
        out = self.bn3(self.relu(self.conv3(out)))
        out = self.bn4(self.relu(self.conv4(out)))
        out = out.view(-1, out.shape[1]*out.shape[2]*out.shape[3])
        out = self.dropout_2(self.relu(self.fc1(out)))
        out = self.dropout_2(self.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

x = torch.randn((10, 1, 40, 40))
CNN_Net(1, 2).forward(x)

# CNN model for regression tasks
class Reg_CNN_Net(nn.Module):
    def __init__(self, input_dims):
        super(Reg_CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=True)

        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(); self.sig = nn.Sigmoid()

    def forward(self, feats):
        out = self.bn1(self.relu(self.conv1(feats)))
        out = self.bn2(self.relu(self.conv2(out)))
        out = self.bn3(self.relu(self.conv3(out)))
        out = self.bn4(self.relu(self.conv4(out)))
        out = out.view(-1, out.shape[1]*out.shape[2]*out.shape[3])
        out = self.dropout_2(self.relu(self.fc1(out)))
        out = self.dropout_2(self.relu(self.fc2(out)))
        # out = self.fc3(out)
        out = self.sig(self.fc3(out))
        return out*60

x = torch.randn((10, 1, 40, 40))
CNN_Net(1, 2).forward(x)

"""### **Run Free Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num)); split_num += 1
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = CNN_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num)); split_num += 1
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = CNN_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num)); split_num += 1
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = CNN_Net(X_train.shape[1], 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>10):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_CNN_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()

					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
		train_error = mean_absolute_error(train_array_true, train_array_preds)      
		val_error = mean_absolute_error(val_array_true, val_array_preds)      
		test_error = mean_absolute_error(test_array_true, test_array_preds)      

		if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_CNN_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()

					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
		train_error = mean_absolute_error(train_array_true, train_array_preds)      
		val_error = mean_absolute_error(val_array_true, val_array_preds)      
		test_error = mean_absolute_error(test_array_true, test_array_preds)      

		if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""### **Run Fixed Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num)); split_num += 1
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = CNN_Net(X_train.shape[1], 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num)); split_num += 1
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = CNN_Net(X_train.shape[1], 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		print('Split Num: '+str(split_num)); split_num += 1
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = CNN_Net(X_train.shape[1], 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>60):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_CNN_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()

					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
		train_error = mean_absolute_error(train_array_true, train_array_preds)      
		val_error = mean_absolute_error(val_array_true, val_array_preds)      
		test_error = mean_absolute_error(test_array_true, test_array_preds)      

		if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)
	X_matrix_new = SelectKBest(mutual_info_classif, k=256).fit_transform(X_matrix_new, Y_vector)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 1, 40, 40))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_CNN_Net(X_train.shape[1])
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()

					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
		train_error = mean_absolute_error(train_array_true, train_array_preds)      
		val_error = mean_absolute_error(val_array_true, val_array_preds)      
		test_error = mean_absolute_error(test_array_true, test_array_preds)      

		if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""## **LSTM network for classification/regression**"""

# LSTM model for classification tasks
class LSTM_Net(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_classes):
        super(LSTM_Net, self).__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=3)

        self.fc = nn.Linear(self.hidden_size, num_classes)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

    def forward(self, feats):        
        h_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))

        h_0 = h_0.cuda()
        c_0 = c_0.cuda()

        out, (final_h, final_c) = self.lstm(feats, (h_0, c_0))
        out = self.fc(final_h[-1])
        return out

LSTM_Net(10, 547, 10, 2).cuda().forward(x)

# LSTM model for regression tasks
class Reg_LSTM_Net(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size):
        super(Reg_LSTM_Net, self).__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=3)

        self.fc = nn.Linear(self.hidden_size, 1)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

    def forward(self, feats):        
        h_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))

        h_0 = h_0.cuda()
        c_0 = c_0.cuda()

        out, (final_h, final_c) = self.lstm(feats, (h_0, c_0))
        out = self.relu(self.fc(final_h[-1]))
        return out*100 #return out 

LSTM_Net(10, 547, 10, 2).cuda().forward(x)

"""### **Run Free Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = LSTM_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>40):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = LSTM_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>40):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = LSTM_Net(10, 547, 10, 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>10):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_LSTM_Net(10, 547, 10)

		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_LSTM_Net(10, 547, 10)

		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""### **Run Fixed Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = LSTM_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>40):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = LSTM_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>40):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = LSTM_Net(10, 547, 10, 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>40):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_LSTM_Net(10, 547, 10)

		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_LSTM_Net(10, 547, 10)

		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []					
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())
				
				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""## **RNN network for classification/regression**"""

# RNN model for classification tasks
class RNN_Net(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_classes):
        super(RNN_Net, self).__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=3)

        self.fc = nn.Linear(self.hidden_size, num_classes)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

    def forward(self, feats):   
        h_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))

        h_0 = h_0.cuda()

        out, final_h = self.rnn(feats, h_0)
        out = self.fc(final_h[-1])
        return out

x = torch.randn((10, 3, 547)).cuda()
RNN_Net(10, 547, 10, 2).cuda().forward(x)

# RNN model for regression tasks
class Reg_RNN_Net(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size):
        super(Reg_RNN_Net, self).__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers=3)

        self.fc = nn.Linear(self.hidden_size, 1)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.dropout_1 = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()

    def forward(self, feats):   
        h_0 = Variable(torch.zeros(3, self.batch_size, self.hidden_size))

        h_0 = h_0.cuda()

        out, final_h = self.rnn(feats, h_0)
        out = self.relu(self.fc(final_h[-1]))
        return out*100

"""### **Run Free Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = RNN_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>=50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = RNN_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>=50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = RNN_Net(10, 547, 10, 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>=10):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_RNN_Net(10, 547, 10)

		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_RNN_Net(10, 547, 10)

		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<100):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")

"""### **Run Fixed Text tasks**"""

# Gender classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = RNN_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>=50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Gender")

# Major/Minor classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = RNN_Net(10, 547, 10, 2)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>=50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Major/Minor")

# Typing Style classification
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = RNN_Net(10, 547, 10, 3)
		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.CrossEntropyLoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs, y.long())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_correct = 0
					train_total = 0
					val_correct = 0
					val_total = 0
					test_correct = 0
					test_total = 0

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						train_total += y.size(0)
						train_correct += (y==predicted).sum().item()

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						val_total += y.size(0)
						val_correct += (y==predicted).sum().item()

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						_, predicted = torch.max(outputs.data, 1)
						test_total += y.size(0)
						test_correct += (y==predicted).sum().item()

				if(100*val_correct/val_total>=50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Acc: '+str(100*train_correct/train_total)+', Val Acc: '+str(100*val_correct/val_total)+', Test Acc: '+str(100*test_correct/test_total))
train_model("Typing Style")

# Age regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_RNN_Net(10, 547, 10)

		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Age")

# Height regression
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def weights_init(layer):
	if isinstance(layer, nn.Linear):
		layer.bias.data.zero_()
		nn.init.kaiming_uniform_(layer.weight.data)

def train_model(label_type):
	X_matrix_new, Y_vector = get_train_test_splits_reg_fixed(label_type)
	X_matrix_new = np.resize(X_matrix_new, (X_matrix_new.shape[0], 3, 547))

	X_train, X_test, Y_train, Y_test = train_test_split(X_matrix_new, Y_vector, test_size=0.3, random_state=0)
	kf = StratifiedKFold(n_splits=3)
	kf.get_n_splits(X_train)

	split_num = 0

	for train_index, val_index in kf.split(X_train, Y_train):
		X_train, X_val = X_matrix_new[train_index], X_matrix_new[val_index]
		Y_train, Y_val = Y_vector[train_index], Y_vector[val_index]

		fcn = Reg_RNN_Net(10, 547, 10)

		fcn.apply(weights_init)
		optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
		loss_func = nn.MSELoss()

		fcn.cuda()
		loss_func.cuda()

		train_tensor_x = torch.Tensor(X_train)
		train_tensor_y = torch.Tensor(Y_train)
		val_tensor_x = torch.Tensor(X_val)
		val_tensor_y = torch.Tensor(Y_val)
		test_tensor_x = torch.Tensor(X_test)
		test_tensor_y = torch.Tensor(Y_test)

		train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)

		val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
		val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False)

		test_dataset = torch.utils.data.TensorDataset(test_tensor_x, test_tensor_y)
		test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, drop_last=False)

		for epoch in range(10):
			for itr, (x, y) in enumerate(train_dataloader):
				fcn.train()
				x = x.cuda()
				y = y.cuda()

				# print(x.shape)
		
				if(x.shape[0]!=10):
						continue

				outputs = fcn(x)
				loss = loss_func(outputs.view(-1), y.float())

				params = list(fcn.parameters())
				l1_regularization, l2_regularization = torch.norm(params[0], 1), torch.norm(params[0], 2)

				for param in params:
					l1_regularization += torch.norm(param, 1)
					# l2_regularization += torch.norm(param, 2)

				reg_1 = Variable(l1_regularization)
				# reg_2 = Variable(l2_regularization)

				loss += reg_1

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				with torch.no_grad():
					fcn.eval()
					train_array_preds = []
					val_array_preds = []
					test_array_preds = []
					train_array_true = []
					val_array_true = []
					test_array_true = []

					for (x, y) in train_dataloader:
						x = x.cuda()
						y = y.cuda()

						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						train_array_preds = train_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						train_array_true = train_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in val_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						val_array_preds = val_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						val_array_true = val_array_true + list(y.float().detach().cpu().numpy())

					for (x, y) in test_dataloader:
						x = x.cuda()
						y = y.cuda()
			
						if(x.shape[0]!=10):
								continue

						outputs = fcn(x)
						test_array_preds = test_array_preds + list(outputs.view(-1).detach().cpu().numpy())
						test_array_true = test_array_true + list(y.float().detach().cpu().numpy())

				train_error = mean_absolute_error(train_array_true, train_array_preds)      
				val_error = mean_absolute_error(val_array_true, val_array_preds)
				test_error = mean_absolute_error(test_array_true, test_array_preds)

				if(val_error<50):
					print('Epoch: '+str(epoch)+', Itr: '+str(itr)+', Loss: '+str(loss.item())+', Train Error: '+str(train_error)+', Val Error: '+str(val_error)+', Test Error: '+str(test_error))
train_model("Height")
