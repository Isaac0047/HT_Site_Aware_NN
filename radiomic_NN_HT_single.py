# This code performs the radiogenomic analysis
# 0-13 Zero 14-31 First 32-55 GLCM 56-69 GLDM 70-85 GLRLM 86-101 GLSZM 102-106 NGTDM

import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd


#%% Section 1
###############################################################################
###############################################################################

#%% Load the dataset
file_path = '/Users/hafeng/Documents/Postdoc_Research_Meetings/Medical_BLT/HT_Data/medical/'
label = pd.read_csv(file_path + '250520HTLabel.csv')
cli_dose = pd.read_csv(file_path + 'cli+dose.csv')

dose_path  = file_path + 'dose_feature/'
image_path = file_path + 'image_feature/'

label_matrix = label.iloc[:,-1].to_numpy()
label_name   = label['ID']

# cli_dose = cli_dose.loc[:, ~cli_dose.columns.str.contains('pq_')]

#%% Load the dataset
file_path2 = '/Users/hafeng/Documents/Postdoc_Research_Meetings/Medical_BLT/HT_Data/medical2/'
label2     = pd.read_csv(file_path2 + 'clinicalsLabels.csv')
cli_dose2  = pd.read_csv(file_path2 + 'output3.csv')
level      = pd.read_csv(file_path2 + 'level4.csv')

level.iloc[:, 1] = pd.to_numeric(level.iloc[:, 1], errors='coerce')
level.iloc[:, 2] = pd.to_numeric(level.iloc[:, 2], errors='coerce')
level.iloc[:, 3] = pd.to_numeric(level.iloc[:, 3], errors='coerce')
level.iloc[:, 4] = pd.to_numeric(level.iloc[:, 4], errors='coerce')

level.iloc[:, 1] = (level.iloc[:, 1] >= 2).astype(int)
level.iloc[:, 2] = (level.iloc[:, 2] >= 2).astype(int)
level.iloc[:, 3] = (level.iloc[:, 3] >= 2).astype(int)
level.iloc[:, 4] = (level.iloc[:, 4] >= 2).astype(int)

dose_path2  = file_path2
image_path2 = file_path2

demo_df = label2.iloc[:,:-1]

label_matrix2 = label2.iloc[:,-1].to_numpy()
label_name2   = label2['ID']

# cli_dose2 = cli_dose2.loc[:, ~cli_dose2.columns.str.contains('pq_')]
cli_dose2.iloc[:, 0] = cli_dose2.iloc[:, 0].astype(str)

common = set(label2.iloc[:, 0]).intersection(set(cli_dose2.iloc[:, 0]))
print(common)

#%% Process the dataset
# Dose Image Needs to get rid of '1146297.nii.gz'
label_list   = []
image_gg_feature_list  = []
image_pq_feature_list  = []
image_qg_feature_list  = []
image_ydz_feature_list = [] 

dose_gg_feature_list  = []
dose_pq_feature_list  = []
dose_qg_feature_list  = []
dose_ydz_feature_list = [] 

HT_list          = []
cli_dose_list    = []
demo_list        = []
failed_list = ['1178436.nii.gz','24052104.nii.gz','1212466.nii.gz','1332354.nii.gz']
# failed_list = ['1178436.nii.gz']
# failed_list = ['1146297.nii.gz','1178436.nii.gz','24052104.nii.gz','1212466.nii.gz']

for item in os.listdir(file_path+'/images/'):
    
    if item not in failed_list:
        print(item)
        label_val  = label[label.iloc[:, 0] == item].iloc[0, 1]
        label_list.append(label_val)
        
        cli_feature = cli_dose[cli_dose.iloc[:,0] == item].iloc[0,:]
        cli_dose_list.append(cli_feature)
        
        demo_feature = demo_df[demo_df.iloc[:,0] == int(item[:-7])].iloc[0,1:]
        demo_list.append(demo_feature)
        
        image_gg_feature = pd.read_excel(image_path+item[:-7]+'_gg_all.xlsx', usecols='B', skiprows=23)
        image_gg_feature_list.append(list(np.squeeze(image_gg_feature.to_numpy())))
        
        image_pq_feature = pd.read_excel(image_path+item[:-7]+'_pq_all.xlsx', usecols='B', skiprows=23)
        image_pq_feature_list.append(list(np.squeeze(image_pq_feature.to_numpy())))
        
        image_qg_feature = pd.read_excel(image_path+item[:-7]+'_qg_all.xlsx', usecols='B', skiprows=23)
        image_qg_feature_list.append(list(np.squeeze(image_qg_feature.to_numpy())))
        
        image_ydz_feature = pd.read_excel(image_path+item[:-7]+'_ydz_all.xlsx', usecols='B', skiprows=23)
        image_ydz_feature_list.append(list(np.squeeze(image_ydz_feature.to_numpy())))
        
        dose_gg_feature = pd.read_excel(dose_path+item[:-7]+'_gg_all.xlsx', usecols='B', skiprows=23)
        dose_gg_feature_list.append(list(np.squeeze(dose_gg_feature.to_numpy())))
        
        dose_pq_feature = pd.read_excel(dose_path+item[:-7]+'_pq_all.xlsx', usecols='B', skiprows=23)
        dose_pq_feature_list.append(list(np.squeeze(dose_pq_feature.to_numpy())))
         
        dose_qg_feature = pd.read_excel(dose_path+item[:-7]+'_qg_all.xlsx', usecols='B', skiprows=23)
        dose_qg_feature_list.append(list(np.squeeze(dose_qg_feature.to_numpy())))
        
        dose_ydz_feature = pd.read_excel(dose_path+item[:-7]+'_ydz_all.xlsx', usecols='B', skiprows=23)
        dose_ydz_feature_list.append(list(np.squeeze(dose_ydz_feature.to_numpy())))

#%% Obtain the feature names

feature_name = pd.read_excel(image_path+item[:-7]+'_gg_all.xlsx', usecols='A', skiprows=23).iloc[:,0].to_list()
# feature_name = pd.read_excel(image_path+item[:-7]+'_gg_all.xlsx', usecols='A', skiprows=23, nrows=107).iloc[:,0].to_list()

cli_name     = ['V5gy', 'V10gy', 'V20gy', 'V30gy', 'V35gy', 'V40gy', 'V45gy', 'V50gy', 'D1p', 'D2p', 'MaxDose', 'MeanDose']

#%% Reformat the dataset

# demo_feature_matrix      = np.array(cli_dose_list)[:,1:8].astype(float)
demo_feature_matrix      = np.array(demo_list).astype(float)
cli_feature_matrix       = np.array(cli_dose_list)[:,8:].astype(float)

image_gg_feature_matrix  = np.array(image_gg_feature_list)
image_pq_feature_matrix  = np.array(image_pq_feature_list)
image_qg_feature_matrix  = np.array(image_qg_feature_list)
image_ydz_feature_matrix = np.array(image_ydz_feature_list)
dose_gg_feature_matrix   = np.array(dose_gg_feature_list)
dose_pq_feature_matrix   = np.array(dose_pq_feature_list)
dose_qg_feature_matrix   = np.array(dose_qg_feature_list)
dose_ydz_feature_matrix  = np.array(dose_ydz_feature_list)
label_matrix             = np.array(label_list)

gg_feature_matrix  = np.concatenate((image_gg_feature_matrix,  dose_gg_feature_matrix), axis=1)
pq_feature_matrix  = np.concatenate((image_pq_feature_matrix,  dose_pq_feature_matrix), axis=1)
qg_feature_matrix  = np.concatenate((image_qg_feature_matrix,  dose_qg_feature_matrix), axis=1)
ydz_feature_matrix = np.concatenate((image_ydz_feature_matrix, dose_ydz_feature_matrix), axis=1)

#%% Log-Transform of the feature and stacking
demo_feature_matrix         = np.log1p(demo_feature_matrix)
cli_feature_matrix[:,8:12]  = np.log1p(cli_feature_matrix[:,8:12])
cli_feature_matrix[:,20:24] = np.log1p(cli_feature_matrix[:,20:24])
cli_feature_matrix[:,32:36] = np.log1p(cli_feature_matrix[:,32:36])
cli_feature_matrix[:,44:]   = np.log1p(cli_feature_matrix[:,44:])

#%% Section 2
###############################################################################
###############################################################################

#%% Process the dataset
# Dose Image Needs to get rid of '1146297.nii.gz'
label_list2   = []
image_gg_feature_list2  = []
image_pq_feature_list2  = []
image_qg_feature_list2  = []
image_ydz_feature_list2 = [] 

dose_gg_feature_list2  = []
dose_pq_feature_list2  = []
dose_qg_feature_list2  = []
dose_ydz_feature_list2 = [] 

HT_list2          = []
demo_list2        = []
cli_dose_list2    = []
# failed_list2      = ['1298384','25041705']
# failed_list = ['1146297.nii.gz','1178436.nii.gz','24052104.nii.gz','1212466.nii.gz']
search_path2 = '/Users/hafeng/Documents/Postdoc_Research_Meetings/Medical_BLT/HT_Data2/'
valid_item   = []

for item in os.listdir(search_path2):
    
    if not item.startswith('.DS'):
    
        if int(item) in label2.iloc[:, 0].values and item in cli_dose2.iloc[:,0].values:
            print(f'Working on Patient {item}')
            valid_item.append(item)
            
            label_val2   = label2[label2.iloc[:, 0] == int(item)].iloc[0, -1]
            label_list2.append(label_val2)
            
            cli_feature2 = cli_dose2[cli_dose2.iloc[:,0] == item].iloc[0,:]
            cli_dose_list2.append(cli_feature2)
            
            demo_feature2 = demo_df[demo_df.iloc[:,0] == int(item)].iloc[0,1:]
            demo_list2.append(demo_feature2)
            
            HT_feature2  = level[level.iloc[:,0] == int(item)].iloc[0,:]
            HT_list2.append(HT_feature2)
            
            image_gg_feature2  = pd.read_excel(image_path2+item+'_gg_image.xlsx', usecols='B', skiprows=23)
            image_gg_feature_list2.append(list(np.squeeze(image_gg_feature2.to_numpy())))
            
            image_pq_feature2  = pd.read_excel(image_path2+item+'_pq_image.xlsx', usecols='B', skiprows=23)
            image_pq_feature_list2.append(list(np.squeeze(image_pq_feature2.to_numpy())))
            
            image_qg_feature2  = pd.read_excel(image_path2+item+'_qg_image.xlsx', usecols='B', skiprows=23)
            image_qg_feature_list2.append(list(np.squeeze(image_qg_feature2.to_numpy())))
            
            image_ydz_feature2 = pd.read_excel(image_path2+item+'_ydz_image.xlsx', usecols='B', skiprows=23)
            image_ydz_feature_list2.append(list(np.squeeze(image_ydz_feature2.to_numpy())))
            
            dose_gg_feature2   = pd.read_excel(dose_path2+item+'_gg_dose.xlsx', usecols='B', skiprows=23)
            dose_gg_feature_list2.append(list(np.squeeze(dose_gg_feature2.to_numpy())))
            
            dose_pq_feature2   = pd.read_excel(dose_path2+item+'_pq_dose.xlsx', usecols='B', skiprows=23)
            dose_pq_feature_list2.append(list(np.squeeze(dose_pq_feature2.to_numpy())))
             
            dose_qg_feature2   = pd.read_excel(dose_path2+item+'_qg_dose.xlsx', usecols='B', skiprows=23)
            dose_qg_feature_list2.append(list(np.squeeze(dose_qg_feature2.to_numpy())))
            
            dose_ydz_feature2  = pd.read_excel(dose_path2+item+'_ydz_dose.xlsx', usecols='B', skiprows=23)
            dose_ydz_feature_list2.append(list(np.squeeze(dose_ydz_feature2.to_numpy())))
            
        else:
            print(f'Not found Patient {item}')

#%% Reformat the dataset 2

# demo_feature_matrix2      = np.array(cli_dose_list2)[:,1:8].astype(float)
demo_feature_matrix2      = np.array(demo_list2).astype(float)
cli_feature_matrix2       = np.array(cli_dose_list2)[:,1:].astype(float)
HT_feature_matrix2        = np.array(HT_list2)[:,1:].astype(float)

image_gg_feature_matrix2  = np.array(image_gg_feature_list2)
image_pq_feature_matrix2  = np.array(image_pq_feature_list2)
image_qg_feature_matrix2  = np.array(image_qg_feature_list2)
image_ydz_feature_matrix2 = np.array(image_ydz_feature_list2)
dose_gg_feature_matrix2   = np.array(dose_gg_feature_list2)
dose_pq_feature_matrix2   = np.array(dose_pq_feature_list2)
dose_qg_feature_matrix2   = np.array(dose_qg_feature_list2)
dose_ydz_feature_matrix2  = np.array(dose_ydz_feature_list2)
label_matrix2             = np.array(label_list2)

gg_feature_matrix  = np.concatenate((image_gg_feature_matrix2,  dose_gg_feature_matrix2),  axis=1)
pq_feature_matrix  = np.concatenate((image_pq_feature_matrix2,  dose_pq_feature_matrix2),  axis=1)
qg_feature_matrix  = np.concatenate((image_qg_feature_matrix2,  dose_qg_feature_matrix2),  axis=1)
ydz_feature_matrix = np.concatenate((image_ydz_feature_matrix2, dose_ydz_feature_matrix2), axis=1)

#%% Log-Transform of the feature and stacking

demo_feature_matrix2         = np.log1p(demo_feature_matrix2)
cli_feature_matrix2[:,8:12]  = np.log1p(cli_feature_matrix2[:,8:12])
cli_feature_matrix2[:,20:24] = np.log1p(cli_feature_matrix2[:,20:24])
cli_feature_matrix2[:,32:36] = np.log1p(cli_feature_matrix2[:,32:36])
cli_feature_matrix2[:,44:]   = np.log1p(cli_feature_matrix2[:,44:])

#%%

def standard_scaler_fit(X):
    """Compute mean and std for each feature (column)"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def standard_scaler_transform(X, mean, std):
    """Normalize X using precomputed mean and std"""
    # Avoid division by zero
    std_replaced = np.where(std == 0, 1, std)
    return (X - mean) / std_replaced

def minmax_scaler_fit(X):
    """Compute min and max for each feature (column)"""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return min_val, max_val

def minmax_scaler_transform(X, min_val, max_val):
    """Normalize X to [0, 1] using precomputed min and max"""
    # Avoid division by zero (when max == min)
    range_replaced = np.where((max_val - min_val) == 0, 1, max_val - min_val)
    return (X - min_val) / range_replaced

#%% Concatenate two groups together

image_gg_feature_matrix_all  = np.concatenate((image_gg_feature_matrix,  image_gg_feature_matrix2),  axis=0)
image_qg_feature_matrix_all  = np.concatenate((image_qg_feature_matrix,  image_qg_feature_matrix2),  axis=0)
image_pq_feature_matrix_all  = np.concatenate((image_pq_feature_matrix,  image_pq_feature_matrix2),  axis=0)
image_ydz_feature_matrix_all = np.concatenate((image_ydz_feature_matrix, image_ydz_feature_matrix2), axis=0)
dose_gg_feature_matrix_all   = np.concatenate((dose_gg_feature_matrix,   dose_gg_feature_matrix2),   axis=0)
dose_qg_feature_matrix_all   = np.concatenate((dose_qg_feature_matrix,   dose_qg_feature_matrix2),   axis=0)
dose_pq_feature_matrix_all   = np.concatenate((dose_pq_feature_matrix,   dose_pq_feature_matrix2),   axis=0)
dose_ydz_feature_matrix_all  = np.concatenate((dose_ydz_feature_matrix,  dose_ydz_feature_matrix2),  axis=0)
cli_feature_matrix_all       = np.concatenate((cli_feature_matrix,       cli_feature_matrix2),       axis=0)
demo_feature_matrix_all      = np.concatenate((demo_feature_matrix,      demo_feature_matrix2),      axis=0)

#%% Combine into dataframe

# data_index = 'single'
# data_index = 'second'
data_index = 'all'

if data_index == 'all':
    image_gg_radiomic    = pd.DataFrame(image_gg_feature_matrix_all,  columns=feature_name)
    image_pq_radiomic    = pd.DataFrame(image_pq_feature_matrix_all,  columns=feature_name)
    image_qg_radiomic    = pd.DataFrame(image_qg_feature_matrix_all,  columns=feature_name)
    image_ydz_radiomic   = pd.DataFrame(image_ydz_feature_matrix_all, columns=feature_name)
    dose_gg_radiomic     = pd.DataFrame(dose_gg_feature_matrix_all,   columns=feature_name)
    dose_pq_radiomic     = pd.DataFrame(dose_pq_feature_matrix_all,   columns=feature_name)
    dose_qg_radiomic     = pd.DataFrame(dose_qg_feature_matrix_all,   columns=feature_name)
    dose_ydz_radiomic    = pd.DataFrame(dose_ydz_feature_matrix_all,  columns=feature_name)
    cli_feature_radiomic = pd.DataFrame(cli_feature_matrix_all,       columns=cli_name*4)
elif data_index == 'single':
    image_gg_radiomic    = pd.DataFrame(image_gg_feature_matrix,  columns=feature_name)
    image_pq_radiomic    = pd.DataFrame(image_pq_feature_matrix,  columns=feature_name)
    image_qg_radiomic    = pd.DataFrame(image_qg_feature_matrix,  columns=feature_name)
    image_ydz_radiomic   = pd.DataFrame(image_ydz_feature_matrix, columns=feature_name)
    dose_gg_radiomic     = pd.DataFrame(dose_gg_feature_matrix,   columns=feature_name)
    dose_pq_radiomic     = pd.DataFrame(dose_pq_feature_matrix,   columns=feature_name)
    dose_qg_radiomic     = pd.DataFrame(dose_qg_feature_matrix,   columns=feature_name)
    dose_ydz_radiomic    = pd.DataFrame(dose_ydz_feature_matrix,  columns=feature_name)
    cli_feature_radiomic = pd.DataFrame(cli_feature_matrix,       columns=cli_name*4)
elif data_index == 'second':
    image_gg_radiomic    = pd.DataFrame(image_gg_feature_matrix2,  columns=feature_name)
    image_pq_radiomic    = pd.DataFrame(image_pq_feature_matrix2,  columns=feature_name)
    image_qg_radiomic    = pd.DataFrame(image_qg_feature_matrix2,  columns=feature_name)
    image_ydz_radiomic   = pd.DataFrame(image_ydz_feature_matrix2, columns=feature_name)
    dose_gg_radiomic     = pd.DataFrame(dose_gg_feature_matrix2,   columns=feature_name)
    dose_pq_radiomic     = pd.DataFrame(dose_pq_feature_matrix2,   columns=feature_name)
    dose_qg_radiomic     = pd.DataFrame(dose_qg_feature_matrix2,   columns=feature_name)
    dose_ydz_radiomic    = pd.DataFrame(dose_ydz_feature_matrix2,  columns=feature_name)
    cli_feature_radiomic = pd.DataFrame(cli_feature_matrix2,       columns=cli_name*4)

#%% Regression
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import random
# from mrmr import mrmr_classif

# X_numeric = image_qg_feature_matrix
# X_numeric = np.concatenate((ydz_feature_matrix, cli_feature_matrix),axis=1)
# X_numeric = np.concatenate((gg_feature_matrix, pq_feature_matrix, qg_feature_matrix, ydz_feature_matrix),axis=1)  # or X.values
# X_numeric = np.concatenate((gg_feature_matrix, pq_feature_matrix, qg_feature_matrix, ydz_feature_matrix, cli_feature_matrix),axis=1)  # or X.values

# X_numeric = np.concatenate((dose_qg_feature_matrix, cli_feature_matrix),axis=1)
# X_numeric = np.concatenate((image_gg_feature_matrix, image_pq_feature_matrix, image_qg_feature_matrix, image_ydz_feature_matrix),axis=1)  # or X.values
# X_numeric = np.concatenate((dose_gg_feature_matrix, dose_pq_feature_matrix, dose_qg_feature_matrix, dose_ydz_feature_matrix),axis=1)  # or X.values
# X_numeric = np.concatenate((image_gg_feature_matrix, image_pq_feature_matrix, image_qg_feature_matrix, image_ydz_feature_matrix),axis=1)  # or X.values
# X_numeric  = np.concatenate((dose_gg_feature_matrix, dose_pq_feature_matrix, dose_qg_feature_matrix, dose_ydz_feature_matrix),axis=1)  # or X.values
# X_numeric = np.concatenate((image_gg_feature_matrix, image_pq_feature_matrix, image_qg_feature_matrix, image_ydz_feature_matrix, dose_gg_feature_matrix, dose_pq_feature_matrix, dose_qg_feature_matrix, dose_ydz_feature_matrix),axis=1)
# X_numeric = cli_feature_matrix

# temp_df = ce_shape_clean.add_suffix('_ce')
# temp_df = image_gg_radiomic.add_suffix('_pelvis')
# temp_df = pd.merge(temp_df, image_pq_radiomic.add_suffix('_cavity'), left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, image_qg_radiomic.add_suffix('_iliac'), left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, image_ydz_radiomic.add_suffix('_lum'), left_index=True, right_index=True)
temp_df = dose_gg_radiomic.add_suffix('_dose_pelvis')
# temp_df = pd.merge(temp_df, dose_gg_radiomic.add_suffix('_dose_pelvis'), left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, dose_pq_radiomic.add_suffix('_dose_cavity'), left_index=True, right_index=True)
temp_df = pd.merge(temp_df, dose_qg_radiomic.add_suffix('_dose_iliac'), left_index=True, right_index=True)
temp_df = pd.merge(temp_df, dose_ydz_radiomic.add_suffix('_dose_lum'), left_index=True, right_index=True)

# mean, std  = standard_scaler_fit(X_numeric)
# min_val, max_val = minmax_scaler_fit(X_numeric)

X_numeric  = temp_df.to_numpy()

# X_numeric  = scaler.fit_transform(X_numeric)
if data_index == 'single':
    y_numeric  = label_matrix
elif data_index == 'second':
    y_numeric  = label_matrix2
    # y_nemeric  = HT_feature_matrix2[:,3]
elif data_index == 'all':
    y_numeric  = np.concatenate((label_matrix, label_matrix2), axis=0)

# Convert your object array to float
X = np.array(X_numeric, dtype=np.float32)
y = np.array(y_numeric).astype(np.float32)

from sklearn.decomposition import PCA
# pca = PCA(n_components=60)  # or 20–50 depending on your variance explained
# X_scaled = pca.fit_transform(X_scaled)  # use scaled features
from sklearn.feature_selection import SelectKBest, f_classif

# selector    = SelectKBest(score_func=f_classif, k=100)  # e.g., top 100

def Feature_Select(X, X_validate, X_validate0, y, top_num):
    X_main  = X[:, :-48]   # shape (m, n-50)
    X_extra = X[:, -48:]  # shape (m, 50)
    selector        = SelectKBest(score_func=f_classif, k=top_num)  # e.g., top 100
    X_main_selected = selector.fit_transform(X_main, y)
    X_combined      = np.hstack((X_main_selected, X_extra))
    
    X_main_val     = X_validate[:, :-48]   # shape (m, n-50)
    X_val_extra    = X_validate[:, -48:]  # shape (m, 50)
    X_val_selected = selector.transform(X_main_val)
    X_val_combined = np.hstack((X_val_selected, X_val_extra))
    
    X_main_val0     = X_validate0[:, :-48]   # shape (m, n-50)
    X_val_extra0    = X_validate0[:, -48:]  # shape (m, 50)
    X_val0_selected = selector.transform(X_main_val0)
    X_val0_combined = np.hstack((X_val0_selected, X_val_extra0))
    
    return X_combined, X_val_combined, X_val0_combined

def Feature_Select1(X, X_validate, X_validate0, y, top_num):
    X_main          = X
    selector        = SelectKBest(score_func=f_classif, k=top_num)  # e.g., top 100
    X_main_selected = selector.fit_transform(X_main, y)
    X_combined      = X_main_selected
    
    return X_combined

selector1   = SelectKBest(score_func=f_classif, k=100)  # e.g., top 100
# selector1.fit(X,y)
X = selector1.fit_transform(X, y)

# Get the actual indices from the original 7000 features
selected_indices = np.where(selector1.get_support())[0]
print("Selected indices from original feature space:", selected_indices)

# X = Feature_Select1(X, X_validate, X_validate0, y, 100)

scaler = MinMaxScaler()
# scaler = StandardScaler()
X = scaler.fit_transform(X)

#%%
# X = cli_feature_matrix
# add_index  = 'add'
add_index = 'none'

# demo_index = 'add'
demo_index = 'none'
# demo_index = 'pure'
# demo_index = 'cli'

# X       = np.concatenate((X, demo_feature_matrix), axis=1)

if demo_index == 'add':
    if add_index == 'none':
        if data_index == 'single':
            X = np.concatenate((X, demo_feature_matrix),     axis=1)
        elif data_index == 'second':
            X = np.concatenate((X, demo_feature_matrix2),    axis=1)
        elif data_index == 'all':
            X = np.concatenate((X, demo_feature_matrix_all), axis=1)
    
    elif add_index == 'add':
        if data_index == 'single':
            X = np.concatenate((X, cli_feature_matrix,     demo_feature_matrix),     axis=1)
        elif data_index == 'second':
            X = np.concatenate((X, cli_feature_matrix2,    demo_feature_matrix2),    axis=1)
        elif data_index == 'all':
            X = np.concatenate((X, cli_feature_matrix_all, demo_feature_matrix_all), axis=1)

elif demo_index == 'none':
    if add_index == 'none':
        if data_index == 'single':
            X = X
        elif data_index == 'second':
            X = X
        elif data_index == 'all':
            X = X
    
    elif add_index == 'add':
        if data_index == 'single':
            X = np.concatenate((X, cli_feature_matrix), axis=1)
        elif data_index == 'second':
            X = np.concatenate((X, cli_feature_matrix2), axis=1)
        elif data_index == 'all':
            X = np.concatenate((X, cli_feature_matrix_all), axis=1)

elif demo_index == 'pure':
    if data_index == 'single':
        X = cli_feature_matrix
    elif data_index == 'second':
        X = cli_feature_matrix2
    elif data_index == 'all':
        X = cli_feature_matrix_all
        
elif demo_index == 'cli':
    if data_index == 'single':
        X = np.concatenate((cli_feature_matrix,     demo_feature_matrix),     axis=1)
    elif data_index == 'second':
        X = np.concatenate((cli_feature_matrix2,    demo_feature_matrix2),    axis=1)
    elif data_index == 'all':
        X = np.concatenate((cli_feature_matrix_all, demo_feature_matrix_all), axis=1)
        

#%% Remove abnormal rows

All_Index    = 140
Second_Index = 45

if data_index   == 'second':
    X  = np.delete(X, 45, axis=0)
    y  = np.delete(y, 45, axis=0)
elif data_index == 'all':
    X  = np.delete(X, [56,57,140], axis=0)
    y  = np.delete(y, [56,57,140], axis=0)

#%% Remove NaN
# X = cli_feature_matrix_all[:,[5,17,29,41]]
# X = cli_feature_matrix_all

mask = ~np.isnan(X).any(axis=1)

X = X[mask]
y = y[mask]

#%% Plot out the features and double check

plt.figure()
plt.plot(X[:,32])
plt.title('Feature')

#%%
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Neural network model for binary classification
class BinaryRadiomicsNet(nn.Module):
    def __init__(self, input_dim):
        super(BinaryRadiomicsNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(64, 1)  # 1 output for binary classification
            # nn.Sigmoid()    # Final sigmoid activation
        )

    def forward(self, x):
        return self.model(x)  # logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kfold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
best_auc = -1
best_model_state = None
best_fold = -1
preds_nn = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[test_idx], dtype=torch.float32)
    y_train, y_test = torch.tensor(y[train_idx], dtype=torch.float32), torch.tensor(y[test_idx], dtype=torch.float32)

    model = BinaryRadiomicsNet(input_dim=X_train.shape[1]).to(device)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Training
    model.train()
    for epoch in range(5000):  # Adjust epochs
        optimizer.zero_grad()
        outputs = model(X_train.to(device)).squeeze()
        loss    = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).squeeze().cpu().numpy()
        auc   = roc_auc_score(y[test_idx], preds)
        auc_scores.append(auc)
        print(f"AUC for fold {fold + 1}: {auc:.4f}")
        preds_nn.append(preds)
        
        #  Save best model state
        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()
            best_fold = fold + 1

print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")

# Reload best model
best_model = BinaryRadiomicsNet(input_dim=X_train.shape[1]).to(device)
best_model.load_state_dict(best_model_state)

from scipy import stats

mean_auc = np.mean(auc_scores)
std_auc  = np.std(auc_scores, ddof=1)
ci95     = stats.t.interval(0.95, len(auc_scores) - 1, loc=mean_auc, scale=std_auc / np.sqrt(len(auc_scores)))

print(f"\nAverage AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

#%% NN Testing Set

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_auc_scores = []
val_auc_scores  = []

best_auc = -1
best_model_state = None

for fold, (trainval_idx, test_idx) in enumerate(kfold.split(X, y)):

    print(f"\n===== Fold {fold+1} =====")

    # Outer split
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Inner split (Train / Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Model
    model = BinaryRadiomicsNet(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        weight_decay=1e-4
    )

    # Training
    model.train()

    for epoch in range(5000):

        optimizer.zero_grad()

        outputs = model(X_train.to(device)).squeeze()

        loss = criterion(outputs, y_train.to(device))

        loss.backward()
        optimizer.step()

    # Validation
    model.eval()

    with torch.no_grad():

        val_preds = model(X_val.to(device)).squeeze().cpu().numpy()

    val_auc = roc_auc_score(y_val, val_preds)
    val_auc_scores.append(val_auc)

    print(f"Validation AUC: {val_auc:.4f}")

    # Test evaluation
    with torch.no_grad():

        test_preds = model(X_test_tensor.to(device)).squeeze().cpu().numpy()

    test_auc = roc_auc_score(y_test, test_preds)

    print(f"Test AUC: {test_auc:.4f}")

    test_auc_scores.append(test_auc)

    # Save best model
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = model.state_dict()

# ==========================
# Final Test Statistics
# ==========================

mean_val_auc = np.mean(val_auc_scores)
std_val_auc = np.std(val_auc_scores, ddof=1)

mean_test_auc = np.mean(test_auc_scores)
std_test_auc = np.std(test_auc_scores, ddof=1)

ci95 = stats.t.interval(
    0.95,
    len(test_auc_scores)-1,
    loc=mean_test_auc,
    scale=std_test_auc/np.sqrt(len(test_auc_scores))
)

ci95_cv = stats.t.interval(
    0.95,
    len(val_auc_scores)-1,
    loc=mean_val_auc,
    scale=std_test_auc/np.sqrt(len(val_auc_scores))
)

print("\n======================")
print(f"Mean Test AUC: {mean_test_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

print(f"Mean CV AUC: {mean_val_auc:.4f}")
print(f"95% CI: ({ci95_cv[0]:.4f}, {ci95_cv[1]:.4f})")

#%% Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

auc_rf = []
auc_ucsf   = []
best_auc   = -1
best_model_clf = None
best_fold  = -1
preds_rf   = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, preds)
    auc_rf.append(auc)
    print(f"AUC for fold {fold + 1}: {auc:.4f}")
    preds_rf.append(preds)
    
    # preds = rf.predict_proba(X_upenn)[:, 1]
    # auc1   = roc_auc_score(y_upenn, preds)
    # auc_ucsf.append(auc1)
    # print(f"AUC for fold {fold + 1}: {auc1:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model_rf = rf
        best_fold     = fold + 1

mean_auc = np.mean(auc_rf)
std_auc  = np.std(auc_rf, ddof=1)
ci95     = stats.t.interval(0.95, len(auc_rf) - 1, loc=mean_auc, scale=std_auc / np.sqrt(len(auc_rf)))
print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")
print(f"Average AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

# mean_auc = np.mean(auc_ucsf)
# std_auc  = np.std(auc_ucsf, ddof=1)
# ci95     = stats.t.interval(0.95, len(auc_ucsf) - 1, loc=mean_auc, scale=std_auc / np.sqrt(len(auc_ucsf)))
# print(f"\nAverage AUC: {mean_auc:.4f}")
# print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

#%% RF Test Set

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import stats
import numpy as np

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_auc_rf = []
val_auc_rf  = []

best_auc = -1
best_model_rf = None
best_fold = -1

for fold, (trainval_idx, test_idx) in enumerate(kfold.split(X, y)):

    print(f"\n===== Fold {fold+1} =====")

    # Outer split
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Inner split (Train / Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=42
    )

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # -------- Validation AUC --------
    val_preds = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print(f"Validation AUC: {val_auc:.4f}")
    val_auc_rf.append(val_auc)

    # -------- Test AUC --------
    test_preds = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)

    print(f"Test AUC: {test_auc:.4f}")
    test_auc_rf.append(test_auc)

    # Save best model based on validation
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_rf = rf
        best_fold = fold + 1


# =============================
# Final Test Statistics
# =============================

mean_val_auc = np.mean(val_auc_rf)
std_val_auc  = np.std(val_auc_rf, ddof=1)

mean_test_auc = np.mean(test_auc_rf)
std_test_auc  = np.std(test_auc_rf, ddof=1)

ci95 = stats.t.interval(
    0.95,
    len(test_auc_rf)-1,
    loc=mean_test_auc,
    scale=std_test_auc / np.sqrt(len(test_auc_rf))
)

ci95_cv = stats.t.interval(
    0.95,
    len(val_auc_rf)-1,
    loc=mean_val_auc,
    scale=std_test_auc/np.sqrt(len(val_auc_rf))
)

print("\n======================")
print(f"Mean Test AUC: {mean_test_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

print(f"Mean CV AUC: {mean_val_auc:.4f}")
print(f"95% CI: ({ci95_cv[0]:.4f}, {ci95_cv[1]:.4f})")

#%% TPOT

from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
from scipy import stats

import numpy as np
# Compatibility patch for NumPy >= 2.0
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
    
auc_scores = []
auc_ucsf   = []
best_auc = -1
best_model_tpot = None
best_fold = -1
preds_tpot = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # TPOT classifier (generations and population size can be tuned)
    # tpot = TPOTClassifier(generations=5, population_size=5, random_state=42)
    tpot   = GradientBoostingClassifier(n_estimators=100, max_depth=None, random_state=42)
    # tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, scoring='roc_auc', random_state=42, n_jobs=-1)
    tpot.fit(X_train, y_train)

    preds = tpot.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    auc_scores.append(auc)
    print(f"AUC for fold {fold + 1}: {auc:.4f}")
    preds_tpot.append(preds)
    
    # preds = tpot.predict(X_upenn)
    # auc = roc_auc_score(y_upenn, preds)
    # auc_ucsf.append(auc)
    # print(f"AUC for fold {fold + 1}: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model_tpot = tpot
        best_fold = fold + 1

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores, ddof=1)
ci95 = stats.t.interval(0.95, len(auc_scores) - 1, loc=mean_auc, scale=std_auc / np.sqrt(len(auc_scores)))

print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")
print(f"Average AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

# mean_auc = np.mean(auc_ucsf)
# std_auc  = np.std(auc_ucsf, ddof=1)
# ci95     = stats.t.interval(0.95, len(auc_ucsf) - 1, loc=mean_auc, scale=std_auc / np.sqrt(len(auc_ucsf)))
# print(f"\nAverage AUC: {mean_auc:.4f}")
# print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

#%% GB Testing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import stats
import numpy as np

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_auc_rf = []
val_auc_rf  = []

best_auc = -1
best_model_rf = None
best_fold = -1

for fold, (trainval_idx, test_idx) in enumerate(kfold.split(X, y)):

    print(f"\n===== Fold {fold+1} =====")

    # Outer split
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Inner split (Train / Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=42
    )

    # Train Random Forest
    rf = GradientBoostingClassifier(n_estimators=100, max_depth=None, random_state=42)

    rf.fit(X_train, y_train)

    # -------- Validation AUC --------
    val_preds = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print(f"Validation AUC: {val_auc:.4f}")
    val_auc_rf.append(val_auc)

    # -------- Test AUC --------
    test_preds = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)

    print(f"Test AUC: {test_auc:.4f}")
    test_auc_rf.append(test_auc)

    # Save best model based on validation
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_rf = rf
        best_fold = fold + 1


# =============================
# Final Test Statistics
# =============================

mean_val_auc = np.mean(val_auc_rf)
std_val_auc  = np.std(val_auc_rf, ddof=1)

mean_test_auc = np.mean(test_auc_rf)
std_test_auc  = np.std(test_auc_rf, ddof=1)

ci95 = stats.t.interval(
    0.95,
    len(test_auc_rf)-1,
    loc=mean_test_auc,
    scale=std_test_auc / np.sqrt(len(test_auc_rf))
)

ci95_cv = stats.t.interval(
    0.95,
    len(val_auc_rf)-1,
    loc=mean_val_auc,
    scale=std_test_auc/np.sqrt(len(val_auc_rf))
)

print("\n======================")
print(f"Mean Test AUC: {mean_test_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

print(f"Mean CV AUC: {mean_val_auc:.4f}")
print(f"95% CI: ({ci95_cv[0]:.4f}, {ci95_cv[1]:.4f})")

#%% CLF training

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

auc_scores = []
best_auc = -1
best_model_clf = None
best_fold = -1
preds_clf = []
true_label = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train logistic regression
    clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000)
    # clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)

    # Predict probabilities for class 1
    preds = clf.predict_proba(X_test)[:, 1]
    preds_clf.append(preds)
    true_label.append(y_test)

    auc = roc_auc_score(y_test, preds)
    auc_scores.append(auc)
    print(f"AUC for fold {fold + 1}: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model_clf = clf
        best_fold = fold + 1

# After loop
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores, ddof=1)
ci95 = stats.t.interval(0.95, len(auc_scores) - 1, loc=mean_auc, scale=std_auc / np.sqrt(len(auc_scores)))

print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")
print(f"Average AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

#%% CLF Testing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import stats
import numpy as np

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_auc_rf = []
val_auc_rf  = []

best_auc = -1
best_model_rf = None
best_fold = -1

for fold, (trainval_idx, test_idx) in enumerate(kfold.split(X, y)):

    print(f"\n===== Fold {fold+1} =====")

    # Outer split
    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Inner split (Train / Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        stratify=y_trainval,
        random_state=42
    )

    # Train Random Forest
    rf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000)

    rf.fit(X_train, y_train)

    # -------- Validation AUC --------
    val_preds = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print(f"Validation AUC: {val_auc:.4f}")
    val_auc_rf.append(val_auc)

    # -------- Test AUC --------
    test_preds = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)


    print(f"Test AUC: {test_auc:.4f}")
    test_auc_rf.append(test_auc)

    # Save best model based on validation
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_rf = rf
        best_fold = fold + 1

# =============================
# Final Test Statistics
# =============================

mean_val_auc = np.mean(val_auc_rf)
std_val_auc  = np.std(val_auc_rf, ddof=1)

mean_test_auc = np.mean(test_auc_rf)
std_test_auc  = np.std(test_auc_rf, ddof=1)

ci95 = stats.t.interval(
    0.95,
    len(test_auc_rf)-1,
    loc=mean_test_auc,
    scale=std_test_auc / np.sqrt(len(test_auc_rf))
)

ci95_cv = stats.t.interval(
    0.95,
    len(val_auc_rf)-1,
    loc=mean_val_auc,
    scale=std_test_auc/np.sqrt(len(val_auc_rf))
)

print("\n======================")
print(f"Mean Test AUC: {mean_test_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

print(f"Mean CV AUC: {mean_val_auc:.4f}")
print(f"95% CI: ({ci95_cv[0]:.4f}, {ci95_cv[1]:.4f})")


#%% Obtain 95% CI values
from scipy import stats

def compute_95CI(lists):
    
    mean_acc = np.mean(lists)
    std_acc  = np.std(lists, ddof=1)  # use ddof=1 for sample standard deviation
    
    n = len(lists)
    confidence = 0.95
    alpha = 1 - confidence
    
    # t critical value for 95% CI and df=n-1
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_of_error = t_crit * (std_acc / np.sqrt(n))
    
    ci_lower = mean_acc - margin_of_error
    ci_upper = mean_acc + margin_of_error
    print(f"Mean accuracy: {mean_acc:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# def hazard_ratio(X1,X2):
#     X1 = torch.tensor(X1)
#     X2 = torch.tensor(X2)
    
#     return torch.sigmoid(X1) / torch.sigmoid(X2)

def hazard_ratio(X1,X2):
    X1 = torch.tensor(X1)
    X2 = torch.tensor(X2)
    
    return torch.exp(X2-X1)

#%% Plot out the statistical information
xx = np.linspace(0,7,8)
labels = ['V5gy', 'V10gy', 'V20gy', 'V30gy', 'V35gy', 'V40gy', 'V45gy', 'V50gy']

ii = 2

plt.figure()
plt.plot(xx,np.squeeze(cli_fea0[ii*12:8+ii*12]), 'b*')
plt.plot(xx,np.squeeze(cli_fea[ii*12:8+ii*12]), 'r*')
# plt.title('Femural Head Dose Statistical Comparison')
plt.legend(['VMAT Plan', 'UHPP Plan'], fontsize=12)
plt.xticks(ticks=xx, labels=labels)
plt.xticks(fontsize=12)  # x-axis tick numbers or labels
plt.ylabel('Values', fontsize=12)
plt.yticks(fontsize=12)  # y-axis tick numbers

xx = np.linspace(0,3,4)
labels = ['D1p','D2p','MaxDose','MeanDose']

plt.figure()
plt.plot(xx-0.05,np.squeeze(cli_fea0[ii*12+8:ii*12+12]), 'b*')
plt.plot(xx,np.squeeze(cli_fea[ii*12+8:ii*12+12]), 'r*')
# plt.title('Femural Head Dose Statistical Comparison')
plt.legend(['VMAT Plan', 'UHPP Plan'], fontsize=12)
plt.xticks(ticks=xx, labels=labels, fontsize=12)
plt.xticks(fontsize=12)  # x-axis tick numbers or labels
plt.ylabel('Values', fontsize=12)
plt.yticks(fontsize=12)  # y-axis tick numbers

#%% Plotting the graphs

gg_v0 = cli_fea0[0*12:8+0*12]
gg_v  = cli_fea[0*12:8+0*12]
gg_d0 = cli_fea0[0*12+8:0*12+12]
gg_d  = cli_fea[0*12+8:0*12+12]

qg_v0 = cli_fea0[2*12:8+2*12]
qg_v  = cli_fea[2*12:8+2*12]
qg_d0 = cli_fea0[2*12+8:2*12+12]
qg_d  = cli_fea[2*12+8:2*12+12]

ydz_v0 = cli_fea0[3*12:8+3*12]
ydz_v  = cli_fea[3*12:8+3*12]
ydz_d0 = cli_fea0[3*12+8:3*12+12]
ydz_d  = cli_fea[3*12+8:3*12+12]

xx = np.linspace(0, 7, 8)
labels = ['V5gy', 'V10gy', 'V20gy', 'V30gy', 'V35gy', 'V40gy', 'V45gy', 'V50gy']

plt.figure()
plt.plot(xx, np.squeeze(gg_v0), 'b*')
plt.plot(xx, np.squeeze(gg_v),  'bo')
plt.plot(xx, np.squeeze(qg_v0), 'g*')
plt.plot(xx, np.squeeze(qg_v),  'go')
plt.plot(xx, np.squeeze(ydz_v0),'r*')
plt.plot(xx, np.squeeze(ydz_v), 'ro')

# First legend: marker meaning (VMAT vs UHPP)
legend1 = plt.legend(['VMAT Plan', 'UHPP Plan'], fontsize=12, loc='center right')
plt.gca().add_artist(legend1)  # Keep the first legend visible

# Second legend: color meaning (blue/green/red)
color_handles = [
    Line2D([0], [0], color='b', lw=3, label='Lower Pelvis (Blue)'),
    Line2D([0], [0], color='g', lw=3, label='Iliac Bone (Green)'),
    Line2D([0], [0], color='r', lw=3, label='Lumbosacral\nVertebrae (Red)')
]
plt.legend(handles=color_handles, fontsize=12, loc='upper right')

# Axes labels and ticks
plt.xticks(ticks=xx, labels=labels, fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

xx = np.linspace(0, 3, 4)
labels = ['D1p', 'D2p', 'MaxDose', 'MeanDose']

plt.figure(facecolor='white')
# plt.rcParams['axes.facecolor'] = 'white'  # make the plot (axes) background white
plt.plot(xx, np.squeeze(gg_d0), 'b*')
plt.plot(xx, np.squeeze(gg_d),  'bo')
plt.plot(xx, np.squeeze(qg_d0), 'g*')
plt.plot(xx, np.squeeze(qg_d),  'go')
plt.plot(xx, np.squeeze(ydz_d0),'r*')
plt.plot(xx, np.squeeze(ydz_d), 'ro')

# First legend (marker type legend)
legend1 = plt.legend(['VMAT Plan', 'UHPP Plan'], fontsize=12, loc='lower left')

# Add the first legend back to the axes
plt.gca().add_artist(legend1)

# Second legend (color meaning)
color_handles = [
    Line2D([0], [0], color='b', lw=3, label='Lower Pelvis (Blue)'),
    Line2D([0], [0], color='g', lw=3, label='Iliac Bone (Green)'),
    Line2D([0], [0], color='r', lw=3, label='Lumbosacral\nVertebrae (Red)')
]
plt.legend(handles=color_handles, fontsize=12, loc='center left')

plt.xticks(ticks=xx, labels=labels, fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

xx = np.linspace(0, 7, 8)
labels = ['V5gy', 'V10gy', 'V20gy', 'V30gy', 'V35gy', 'V40gy', 'V45gy', 'V50gy']

# --- White background setup ---
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# --- Plot data ---
plt.plot(xx, np.squeeze(gg_v0), 'b*')
plt.plot(xx, np.squeeze(gg_v),  'bo')
plt.plot(xx, np.squeeze(qg_v0), 'g*')
plt.plot(xx, np.squeeze(qg_v),  'go')
plt.plot(xx, np.squeeze(ydz_v0),'r*')
plt.plot(xx, np.squeeze(ydz_v), 'ro')

# --- Axis + spine color fixes ---
for spine in ax.spines.values():
    spine.set_color('black')
ax.tick_params(colors='black')  # tick color
ax.yaxis.label.set_color('black')
ax.xaxis.label.set_color('black')

# --- Legends ---
legend1 = plt.legend(['VMAT Plan', 'UHPP Plan'], fontsize=12, loc='center right')
ax.add_artist(legend1)

color_handles = [
    Line2D([0], [0], color='b', lw=3, label='Lower Pelvis (Blue)'),
    Line2D([0], [0], color='g', lw=3, label='Iliac Bone (Green)'),
    Line2D([0], [0], color='r', lw=3, label='Lumbosacral\nVertebrae (Red)')
]
plt.legend(handles=color_handles, fontsize=12, loc='upper right')

# --- Labels and ticks ---
plt.xticks(ticks=xx, labels=labels, fontsize=12)
plt.ylabel('Values', fontsize=14)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

xx = np.linspace(0, 3, 4)
labels = ['D1p', 'D2p', 'MaxDose', 'MeanDose']

# --- White background setup ---
plt.figure(facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# --- Plot data ---
plt.plot(xx, np.squeeze(gg_d0), 'b*')
plt.plot(xx, np.squeeze(gg_d),  'bo')
plt.plot(xx, np.squeeze(qg_d0), 'g*')
plt.plot(xx, np.squeeze(qg_d),  'go')
plt.plot(xx, np.squeeze(ydz_d0),'r*')
plt.plot(xx, np.squeeze(ydz_d), 'ro')

# --- Axis + spine color fixes ---
for spine in ax.spines.values():
    spine.set_color('black')
ax.tick_params(colors='black')  # tick color
ax.yaxis.label.set_color('black')
ax.xaxis.label.set_color('black')

# --- Legends ---
legend1 = plt.legend(['VMAT Plan', 'UHPP Plan'], fontsize=12, loc='lower left')
ax.add_artist(legend1)

color_handles = [
    Line2D([0], [0], color='b', lw=3, label='Lower Pelvis (Blue)'),
    Line2D([0], [0], color='g', lw=3, label='Iliac Bone (Green)'),
    Line2D([0], [0], color='r', lw=3, label='Lumbosacral\nVertebrae (Red)')
]
plt.legend(handles=color_handles, fontsize=12, loc='center left')

# --- Labels and ticks ---
plt.xticks(ticks=xx, labels=labels, fontsize=12)
plt.ylabel('Values', fontsize=14)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

#%% Perform SHAP Analysis
import shap
# selected_feature_names = temp_df.columns[selector.get_support()]

baseline_name = ['V5gy', 'V10gy', 'V20gy', 'V30gy', 'V35gy', 'V40gy', 'V45gy', 'V50gy', 'D1p', 'D2p', 'MaxDose', 'MeanDose']
pelvis_name   = [name + '_pelvis' for name in baseline_name]
iliac_name    = [name + '_iliac' for name in baseline_name]
lumber_name   = [name + '_lumbo' for name in baseline_name]

# explainer = shap.Explainer(model, torch.tensor(X_train))  # If model is Keras
# shap_values = explainer(X_test, max_evals=5000)
# shap.summary_plot(shap_values, X_test, feature_names=selected_feature_names)

selected_feature_names_rad  = temp_df.columns[selector1.get_support()].to_list()
selected_feature_names_dosi = pelvis_name + iliac_name + lumber_name
selected_feature_names_demo = ['age','age_norm','figo','bmi','bmi_norm','pathology','vmat1']

selected_feature_names      = selected_feature_names_rad + selected_feature_names_dosi + selected_feature_names_demo

explainer   = shap.Explainer(clf, X_train)  # If model is Keras
shap_values = explainer(X_test)
# shap.summary_plot(shap_values, X_test, feature_names=selected_feature_names)

# Set figure size before plotting
plt.figure(figsize=(20, 10))  # Larger figure
shap.summary_plot(shap_values, X_test, feature_names=selected_feature_names, max_display=10)
plt.tight_layout(pad=2.0)  # Automatically adjust layout
plt.show()

# Extract SHAP values
sv = shap_values.values  # shape: (n_samples, n_features)

# Compute mean absolute SHAP per feature
shap_mean_abs = np.abs(sv).mean(axis=0)  # shape: (n_features,)

# Get sorted indices (largest importance first)
sorted_indices = np.argsort(shap_mean_abs)[::-1]

# Top N features
top_n = 10
top_indices = sorted_indices[:top_n]

print("Top feature indices:", top_indices)
print("Top feature names:", [selected_feature_names[i] for i in top_indices])

#%% Plot the hierarchical clustering

label_name = ['Label']
gene_filter = pd.DataFrame(y, columns=label_name)

# Summary
final_df1  = temp_df
merged_df1 = pd.merge(final_df1.reset_index(drop=True), gene_filter.reset_index(drop=True), left_index=True, right_index=True)

shap_names = [selected_feature_names[i] for i in top_indices]

# shap_names = [c for c in shap_names if c in merged_df1.columns]
# feature_hm = merged_df1[shap_names]
# print(feature_hm.shape)

# feature_hm = merged_df1[shap_names]

# print("merged_df1.shape:", merged_df1.shape)
# print("Number of SHAP features found:", len(shap_names))
# print("feature_hm.shape:", feature_hm.shape)

# df_hm = feature_hm
# df_hm = pd.merge(df_hm, mgmt_hm, left_index=True, right_index=True) 
# df_hm = pd.merge(df_hm, egfr_hm, left_index=True, right_index=True) 
# df_hm = pd.merge(df_hm, tp53_hm, left_index=True, right_index=True) 
# df_hm = pd.merge(df_hm, pten_hm, left_index=True, right_index=True) 

# Count for radiomic types

count = sum('firstorder' in name for name in selected_feature_names)
print(count)

count = sum('glcm'  in name for name in selected_feature_names)
print(count)

count = sum('gldm'  in name for name in selected_feature_names)
print(count)

count = sum('glrlm' in name for name in selected_feature_names)
print(count)

count = sum('glszm' in name for name in selected_feature_names)
print(count)

count = sum('ngtdm' in name for name in selected_feature_names)
print(count)

count = sum('shape' in name for name in selected_feature_names)
print(count)

count = sum('wavelet' in name for name in selected_feature_names)
print(count)

# indices = np.where(np.abs(np.squeeze(X_validate0)) > 10)[0]
# print("Indices:", indices)

# indices = np.where(np.abs(np.squeeze(X_validate)) > 50)[0]
# print("Indices:", indices)

#%%
import seaborn as sns

gene_cols = ['Label']
feature_cols = shap_names

# Compute correlations
corr_matrix = pd.DataFrame(index=feature_cols, columns=gene_cols)

for g in gene_cols:
    for f in feature_cols:
        corr_matrix.loc[f, g] = merged_df1[f].corr(merged_df1[g])

# Convert to numeric (corr returns float or NaN)
# corr_matrix = corr_matrix.rename(columns={'Survival Status (1-dead,0-alive)': 'Survival'})
corr_matrix = corr_matrix.astype(float)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature–Gene Correlation Heatmap')
plt.tight_layout()
plt.show()

sns.set(font_scale=1.3)
# Hierarchical clustering heatmap
g = sns.clustermap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    method='average',
    metric='euclidean',   # works better for 1D vectors
    figsize=(6, 10),
    col_cluster=False     # disable column clustering
)
# Rotate bottom (column) labels
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=0, ha='right')

# Add space at the bottom
g.fig.subplots_adjust(bottom=0.25)
plt.show()

#%% Count for feature percentages

count = sum('wavelet' in name for name in selected_feature_names)
print(count)

count = sum('pelvis' in name for name in selected_feature_names)
print(count)

count = sum('iliac' in name for name in selected_feature_names)
print(count)

count = sum('lum' in name for name in selected_feature_names)
print(count)


#%% Generate the ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

for i in range(len(true_label)):
    
    fpr_nn, tpr_nn, _ = roc_curve(true_label[i], preds_nn[i])
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    
    fpr_clf, tpr_clf, _ = roc_curve(true_label[i], preds_clf[i])
    roc_auc_clf = auc(fpr_clf, tpr_clf)
    
    fpr_rf, tpr_rf, _ = roc_curve(true_label[i], preds_rf[i])
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    fpr_tpot, tpr_tpot, _ = roc_curve(true_label[i], preds_tpot[i])
    roc_auc_tpot = auc(fpr_tpot, tpr_tpot)
    
    
    plt.figure()
    plt.plot(fpr_nn, tpr_nn, color='red', lw=1.5, 
             label=f'NN (AUC = {roc_auc_nn:.2f})')
    plt.plot(fpr_clf, tpr_clf, color='darkorange', lw=1.5, 
             label=f'LR (AUC = {roc_auc_clf:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='darkgreen', lw=1.5, 
             label=f'RF (AUC = {roc_auc_rf:.2f})')
    plt.plot(fpr_tpot, tpr_tpot, color='blue', lw=1.5, 
             label=f'TPOT (AUC = {roc_auc_tpot:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')  # diagonal line
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) on folds {i+1}')
    plt.legend(loc='lower right')
    plt.show()











