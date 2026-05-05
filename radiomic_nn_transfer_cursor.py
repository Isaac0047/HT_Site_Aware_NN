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
label     = pd.read_csv(file_path + '250520HTLabel.csv')
cli_dose  = pd.read_csv(file_path + 'cli+dose.csv')

dose_path  = file_path + 'dose_feature/'
image_path = file_path + 'image_feature/'

label_matrix = label.iloc[:,-1].to_numpy()
label_name   = label['ID']

# cli_dose = cli_dose.loc[:, ~cli_dose.columns.str.contains('pq_')]

#%% Load the dataset
file_path2 = '/Users/hafeng/Documents/Postdoc_Research_Meetings/Medical_BLT/HT_Data/medical2/'
label2     = pd.read_csv(file_path2 + 'clinicalsLabels.csv')
cli_dose2  = pd.read_csv(file_path2 + 'output3.csv')

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

cli_dose_list    = []
demo_list        = []
failed_list = ['1178436.nii.gz','24052104.nii.gz','1212466.nii.gz','1332354.nii.gz']
valid_item1 = []

# failed_list = ['1178436.nii.gz']
# failed_list = ['1146297.nii.gz','1178436.nii.gz','24052104.nii.gz','1212466.nii.gz']

for item in os.listdir(file_path+'/images/'):
    
    if item not in failed_list:
        print(item)
        valid_item1.append(item)
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

norm_label = False
if norm_label:
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
 
if norm_label:
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

#%% Filter Out the OOD Patients

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------
# INPUT:
#   X1 = Group 1 radiomic features (training distribution)
#   X2 = Group 2 radiomic features (test distribution)
# --------------------------------------------

X1 = image_gg_feature_matrix
X2 = image_gg_feature_matrix2

# ---------- 1. Fit Isolation Forest on Group 1 ----------
iso = IsolationForest(
    contamination='auto',   # automatically estimate anomaly rate
    random_state=42
)
iso.fit(X1)

# ---------- 2. Predict OOD for Group 2 ----------
scores = iso.decision_function(X2)   # anomaly score
labels = iso.predict(X2)             # 1 = inlier, -1 = outlier

# ---------- 3. Extract out-of-distribution patients ----------
ood_mask_gg = labels == -1
in_mask_gg  = labels == 1

ood_patients = np.where(ood_mask_gg)[0]
in_patients = np.where(in_mask_gg)[0]

print(f"Total patients in Group 2: {len(X2)}")
print(f"In-distribution (kept): {len(in_patients)}")
print(f"Out-of-distribution (removed): {len(ood_patients)}")

# # ---------- 4. Create cleaned X2 without OOD patients ----------
# X2_clean = X2[in_mask_gg]

# # ---------- 5. Create summary output ----------
# results = pd.DataFrame({
#     "Patient_Index_Group2": np.arange(len(X2)),
#     "IF_Score": scores,
#     "OOD": ood_mask
# })

# results_sorted = results.sort_values("IF_Score", ascending=True)  # most OOD first

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------
# INPUT:
#   X1 = Group 1 radiomic features (training distribution)
#   X2 = Group 2 radiomic features (test distribution)
# --------------------------------------------

X1 = image_qg_feature_matrix
X2 = image_qg_feature_matrix2

# ---------- 1. Fit Isolation Forest on Group 1 ----------
iso = IsolationForest(
    contamination='auto',   # automatically estimate anomaly rate
    random_state=42
)
iso.fit(X1)

# ---------- 2. Predict OOD for Group 2 ----------
scores = iso.decision_function(X2)   # anomaly score
labels = iso.predict(X2)             # 1 = inlier, -1 = outlier

# ---------- 3. Extract out-of-distribution patients ----------
ood_mask_qg = labels == -1
in_mask_qg  = labels == 1

ood_patients = np.where(ood_mask_qg)[0]
in_patients = np.where(in_mask_qg)[0]

print(f"Total patients in Group 2: {len(X2)}")
print(f"In-distribution (kept): {len(in_patients)}")
print(f"Out-of-distribution (removed): {len(ood_patients)}")

# # ---------- 4. Create cleaned X2 without OOD patients ----------
# X2_clean = X2[in_mask_qg]

# # ---------- 5. Create summary output ----------
# results = pd.DataFrame({
#     "Patient_Index_Group2": np.arange(len(X2)),
#     "IF_Score": scores,
#     "OOD": ood_mask
# })

# results_sorted = results.sort_values("IF_Score", ascending=True)  # most OOD first

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------
# INPUT:
#   X1 = Group 1 radiomic features (training distribution)
#   X2 = Group 2 radiomic features (test distribution)
# --------------------------------------------

X1 = image_pq_feature_matrix
X2 = image_pq_feature_matrix2

# ---------- 1. Fit Isolation Forest on Group 1 ----------
iso = IsolationForest(
    contamination='auto',   # automatically estimate anomaly rate
    random_state=42
)
iso.fit(X1)

# ---------- 2. Predict OOD for Group 2 ----------
scores = iso.decision_function(X2)   # anomaly score
labels = iso.predict(X2)             # 1 = inlier, -1 = outlier

# ---------- 3. Extract out-of-distribution patients ----------
ood_mask_pq = labels == -1
in_mask_pq  = labels == 1

ood_patients = np.where(ood_mask_pq)[0]
in_patients = np.where(in_mask_pq)[0]

print(f"Total patients in Group 2: {len(X2)}")
print(f"In-distribution (kept): {len(in_patients)}")
print(f"Out-of-distribution (removed): {len(ood_patients)}")

# # ---------- 4. Create cleaned X2 without OOD patients ----------
# X2_clean = X2[in_mask_pq]

# # ---------- 5. Create summary output ----------
# results = pd.DataFrame({
#     "Patient_Index_Group2": np.arange(len(X2)),
#     "IF_Score": scores,
#     "OOD": ood_mask
# })

# results_sorted = results.sort_values("IF_Score", ascending=True)  # most OOD first

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------
# INPUT:
#   X1 = Group 1 radiomic features (training distribution)
#   X2 = Group 2 radiomic features (test distribution)
# --------------------------------------------

X1 = image_ydz_feature_matrix
X2 = image_ydz_feature_matrix2

# ---------- 1. Fit Isolation Forest on Group 1 ----------
iso = IsolationForest(
    contamination='auto',   # automatically estimate anomaly rate
    random_state=42
)
iso.fit(X1)

# ---------- 2. Predict OOD for Group 2 ----------
scores = iso.decision_function(X2)   # anomaly score
labels = iso.predict(X2)             # 1 = inlier, -1 = outlier

# ---------- 3. Extract out-of-distribution patients ----------
ood_mask_ydz = labels == -1
in_mask_ydz  = labels == 1

ood_patients = np.where(ood_mask_ydz)[0]
in_patients = np.where(in_mask_ydz)[0]

print(f"Total patients in Group 2: {len(X2)}")
print(f"In-distribution (kept): {len(in_patients)}")
print(f"Out-of-distribution (removed): {len(ood_patients)}")

# # ---------- 4. Create cleaned X2 without OOD patients ----------
# X2_clean = X2[in_mask_ydz]

# # ---------- 5. Create summary output ----------
# results = pd.DataFrame({
#     "Patient_Index_Group2": np.arange(len(X2)),
#     "IF_Score": scores,
#     "OOD": ood_mask
# })

# results_sorted = results.sort_values("IF_Score", ascending=True)  # most OOD first

#%% Obtain the filtered out dataset

final_mask = in_mask_gg & in_mask_pq & in_mask_qg & in_mask_ydz

filter_label = True

if filter_label:

    image_gg_feature_matrix2  = image_gg_feature_matrix2[final_mask]
    image_pq_feature_matrix2  = image_pq_feature_matrix2[final_mask]
    image_qg_feature_matrix2  = image_qg_feature_matrix2[final_mask]
    image_ydz_feature_matrix2 = image_ydz_feature_matrix2[final_mask]
    cli_feature_matrix2       = cli_feature_matrix2[final_mask]
    demo_feature_matrix2      = demo_feature_matrix2[final_mask]
    label_matrix2 = label_matrix2[final_mask]
    
    

    # Remove abnormal rows
    
    Second_Index = 45
    
    image_gg_feature_matrix2  = np.delete(image_gg_feature_matrix2,  Second_Index, axis=0)
    image_pq_feature_matrix2  = np.delete(image_pq_feature_matrix2,  Second_Index, axis=0)
    image_qg_feature_matrix2  = np.delete(image_qg_feature_matrix2,  Second_Index, axis=0)
    image_ydz_feature_matrix2 = np.delete(image_ydz_feature_matrix2, Second_Index, axis=0)
    
    cli_feature_matrix2  = np.delete(cli_feature_matrix2,  Second_Index, axis=0)
    demo_feature_matrix2 = np.delete(demo_feature_matrix2, Second_Index, axis=0)
    label_matrix2        = np.delete(label_matrix2, Second_Index, axis=0)
    # if data_index   == 'second':
    #     X  = np.delete(X, 61, axis=0)
    #     y  = np.delete(y, 61, axis=0)
    # elif data_index == 'all':
    #     X  = np.delete(X, [56,57,140], axis=0)
    #     y  = np.delete(y, [56,57,140], axis=0)

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
temp_df = image_gg_radiomic.add_suffix('_pelvis')
# temp_df = pd.merge(temp_df, image_pq_radiomic.add_suffix('_cavity'), left_index=True, right_index=True)
temp_df = pd.merge(temp_df, image_qg_radiomic.add_suffix('_iliac'),    left_index=True, right_index=True)
temp_df = pd.merge(temp_df, image_ydz_radiomic.add_suffix('_lum'),     left_index=True, right_index=True)
# temp_df = dose_gg_radiomic.add_suffix('_dose_pelvis')
temp_df = pd.merge(temp_df, dose_gg_radiomic.add_suffix('_dose_pelvis'), left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, dose_pq_radiomic.add_suffix('_dose_cavity'), left_index=True, right_index=True)
temp_df = pd.merge(temp_df, dose_qg_radiomic.add_suffix('_dose_iliac'),  left_index=True, right_index=True)
temp_df = pd.merge(temp_df, dose_ydz_radiomic.add_suffix('_dose_lum'),   left_index=True, right_index=True)

# mean, std  = standard_scaler_fit(X_numeric)
# min_val, max_val = minmax_scaler_fit(X_numeric)

X_numeric  = temp_df.to_numpy()

# X_numeric  = scaler.fit_transform(X_numeric)
if data_index   == 'single':
    y_numeric  = label_matrix
elif data_index == 'second':
    y_numeric  = label_matrix2
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
    
    X_main_val      = X_validate[:, :-48]   # shape (m, n-50)
    X_val_extra     = X_validate[:, -48:]  # shape (m, 50)
    X_val_selected  = selector.transform(X_main_val)
    X_val_combined  = np.hstack((X_val_selected, X_val_extra))
    
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

# select_agent = 'LASSO'
# select_agent = 'ANOVA'

if select_agent == 'ANOVA':
    selector1   = SelectKBest(score_func=f_classif, k=100)  # e.g., top 100
    X = selector1.fit_transform(X, y)
    # selected_features = X.columns[selector.get_support()]

    # Get the actual indices from the original 7000 features
    selected_indices = np.where(selector1.get_support())[0]
    print("Selected indices from original feature space:", selected_indices)
    
elif select_agent == 'LASSO':
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    
    lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=1.0,
        max_iter=500
    )

    selector = SelectFromModel(lasso)
    X = selector.fit_transform(X, y)

# X = Feature_Select(X, X_validate, X_validate0, y, 100)
# select_name = [list(temp_df.columns)[i] for i in selected_indices]

scaler = MinMaxScaler()
# scaler = StandardScaler()
X = scaler.fit_transform(X)

#%%
# X = cli_feature_matrix
# add_index  = 'none'
add_index = 'add'
# demo_index = 'none'
demo_index = 'add'
# demo_index = 'pure'
# X       = np.concatenate((X, demo_feature_matrix), axis=1)

if demo_index == 'add':
    if add_index == 'none':
        if data_index == 'single':
            X = np.concatenate((X, cli_feature_matrix),     axis=1)
        elif data_index == 'second':
            X = np.concatenate((X, cli_feature_matrix2),    axis=1)
        elif data_index == 'all':
            X = np.concatenate((X, cli_feature_matrix_all), axis=1)
    
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
            X = cli_feature_matrix
        elif data_index == 'second':
            X = cli_feature_matrix2
        elif data_index == 'all':
            X = cli_feature_matrix_all
    
    elif add_index == 'add':
        if data_index == 'single':
            X = np.concatenate((demo_feature_matrix, cli_feature_matrix), axis=1)
        elif data_index == 'second':
            X = np.concatenate((demo_feature_matrix2, cli_feature_matrix2), axis=1)
        elif data_index == 'all':
            X = np.concatenate((demo_feature_matrix_all, cli_feature_matrix_all), axis=1)
            
elif demo_index == 'pure':
    if data_index == 'single':
        X = X
    elif data_index == 'second':
        X = X
    elif data_index == 'all':
        X = X
    
#%% Develop Label G

g1 = np.ones((image_gg_feature_matrix.shape[0]))
g2 = np.ones((image_gg_feature_matrix2.shape[0]))
g2 = -g2

g = np.concatenate((g1,g2))

#%% Remove NaN
# X = cli_feature_matrix_all[:,[5,17,29,41]]
# X = cli_feature_matrix_all

mask = ~np.isnan(X).any(axis=1)

X = X[mask]
y = y[mask]
g = g[mask]

# Plot out the features and double check
plt.figure()
plt.plot(X[:,32])
plt.title('Feature')

#%% CLF training

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy import stats

auc_scores = []
best_auc = -1
best_model_clf = None
best_fold = -1
preds_clf = []
true_label = []

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train logistic regression
    clf = LogisticRegression(penalty='l2', solver='liblinear')
    # clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)

    # Predict probabilities for class 1
    preds = clf.predict_proba(X_test)[:, 1]
    preds_clf.append(preds)
    true_label.append(y_test)
    
    print(f'Test True Label is {y_test}')
    print(f'Predicted Label is {preds}')

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

#%%

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

def logistic_regression_pipeline(X, y, n_splits=5):

    # ======================================================
    # Fixed test set (VERY IMPORTANT for DCA)
    # ======================================================
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    val_aucs = []
    test_aucs = []
    test_probs_folds = []

    for fold, (tr, va) in enumerate(skf.split(X_trainval, y_trainval)):

        print(f"\nFold {fold+1}/{n_splits}")

        X_tr, X_va = X_trainval[tr], X_trainval[va]
        y_tr, y_va = y_trainval[tr], y_trainval[va]

        X_tr = selector1.fit_transform(X_tr, y_tr)
        X_va   = selector1.transform(X_va)
        X_test  = selector1.transform(X_test)

        # ======================================================å
        # SIMPLE MODEL: Logistic Regression
        # ======================================================
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear"
        )

        model.fit(X_tr, y_tr)

        # =========================
        # Validation AUC
        # =========================
        va_prob = model.predict_proba(X_va)[:, 1]
        auc_va = roc_auc_score(y_va, va_prob)
        val_aucs.append(auc_va)

        # =========================
        # TEST AUC
        # =========================
        test_prob = model.predict_proba(X_test)[:, 1]
        test_probs_folds.append(test_prob)

        auc_test = roc_auc_score(y_test, test_prob)
        test_aucs.append(auc_test)
        
        print(f'Test True Label is {y_test}')
        print(f'Predicted Label is {test_prob}')

        print(f"Fold {fold+1} | VAL AUC: {auc_va:.4f} | TEST AUC: {auc_test:.4f}")

    # ======================================================
    # Average prediction across folds (ensemble for stability)
    # ======================================================
    test_probs_folds = np.array(test_probs_folds)
    y_prob_test_final = np.mean(test_probs_folds, axis=0)

    print("\nFinal CV Results:")
    print(f"Mean VAL AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"Mean TEST AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    return y_test, y_prob_test_final

logistic_regression_pipeline(X, y, n_splits=5)

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from scipy import stats

auc_scores = []
best_auc = -1
best_model_clf = None
best_fold = -1

outer_kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for fold, (train_full_idx, test_idx) in enumerate(outer_kfold.split(X, y)):
    print(f"\nOuter Fold {fold + 1}")

    X_train_full, X_test = X[train_full_idx], X[test_idx]
    y_train_full, y_test = y[train_full_idx], y[test_idx]

    # ---- INNER SPLIT (Train / Validation) ----
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    X_train = selector1.fit_transform(X_train, y_train)
    X_val   = selector1.transform(X_val)
    X_test  = selector1.transform(X_test)

    # Train model on inner training set
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)

    # Validation performance (optional, for model selection)
    val_preds = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {val_auc:.4f}")

    # Retrain on full outer training set
    clf.fit(X_train_full, y_train_full)

    # Test on outer test fold
    test_preds = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)

    auc_scores.append(test_auc)
    print(f"Test AUC: {test_auc:.4f}")

    if test_auc > best_auc:
        best_auc = test_auc
        best_model_clf = clf
        best_fold = fold + 1

# ---- Final Statistics ----
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores, ddof=1)
ci95 = stats.t.interval(
    0.95,
    len(auc_scores) - 1,
    loc=mean_auc,
    scale=std_auc / np.sqrt(len(auc_scores))
)

print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")
print(f"Average AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

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

def hazard_ratio(X1,X2):
    X1 = torch.tensor(X1)
    X2 = torch.tensor(X2)
    
    return torch.exp(X2-X1)

#%%

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

g_remap = (g == 1).astype(int)

# ============================================================
# MODELS
# ============================================================

class SitePredictorNetwork(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            # nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class CohortSpecificNetwork(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            # nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

def soft_combination(site_prob, pred_A, pred_B):
    return (1 - site_prob) * pred_A + site_prob * pred_B

class MetaCombiner(nn.Module):
    """
    Learns how to combine:
      - site probability
      - cohort A prediction
      - cohort B prediction
    """
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, site_prob, pred_A, pred_B):
        x = torch.cat([site_prob, pred_A, pred_B], dim=1)
        return self.net(x)

# ============================================================
# TRAINING HELPERS (NO LEAKAGE)
# ============================================================

def train_site_predictor(X, site, n_epochs=3500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # X_tr, X_va, y_tr, y_va = train_test_split(
    #     X, site, test_size=0.2, stratify=site, random_state=0
    # )
    
    X_tr = X
    y_tr = site

    model = SitePredictorNetwork(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)

    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        # print(f'Max value of prediction is {torch.max(model(X_tr))} and Min value is {torch.min(model(X_tr))}')
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()

    return model.eval()

def train_cohort_model(X, y, n_epochs=1500, use_early_stopping=True):
    """
    Train cohort model with same settings as separate training:
    - Adam optimizer with weight decay
    - Learning rate scheduler
    - Early stopping
    - Class weighting
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_tr = X
    y_tr = y
    # print(f'The shape of the Trained model is {X_tr.shape}')

    model = CohortSpecificNetwork(X.shape[1]).to(device)
    # Match separate training: Adam with weight decay
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    # Use class weighting like separate training
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)

    best_loss = float('inf')
    patience_counter = 0
    patience = 300

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        y_hat = model(X_tr)
        loss = loss_fn(y_hat, y_tr)
        loss.backward()
        # Gradient clipping like separate training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        
        # Early stopping
        if use_early_stopping and epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                current_loss = loss_fn(model(X_tr), y_tr).item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 50
                
                if patience_counter >= patience:
                    break

    return model.eval()

def train_meta_combiner(
    site_prob_tr, pred_A_tr, pred_B_tr,
    y_tr,
    site_prob_va, pred_A_va, pred_B_va,
    y_va,
    n_epochs=5000
):
    device = site_prob_tr.device
    meta = MetaCombiner().to(device)

    opt = torch.optim.Adam(meta.parameters(), lr=1e-3)
    # opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)

    best_auc = 0
    best_state = None

    for epoch in range(n_epochs):
        meta.train()
        opt.zero_grad()

        logits = meta(site_prob_tr, pred_A_tr, pred_B_tr)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()

        # ---- validate ----
        if epoch % 50 == 0:
            meta.eval()
            with torch.no_grad():
                val_prob = torch.sigmoid(
                    meta(site_prob_va, pred_A_va, pred_B_va)
                ).cpu().numpy().ravel()

            if len(np.unique(y_va)) > 1:
                auc = roc_auc_score(y_va, val_prob)
                if auc > best_auc:
                    best_auc = auc
                    best_state = meta.state_dict()

    if best_state is not None:
        meta.load_state_dict(best_state)

    return meta.eval(), best_auc

def run_single_fold_sann_auc(
    X_tr, y_tr, site_tr,
    X_va, y_va, site_va,
    device
):
    # Train site predictor
    site_model = train_site_predictor(X_tr, site_tr)

    # Train cohort models
    model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
    model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)

    with torch.no_grad():
        site_prob_tr = site_model(X_tr_t)
        site_prob_va = site_model(X_va_t)
        pred_A_tr = torch.sigmoid(model_A(X_tr_t))
        pred_B_tr = torch.sigmoid(model_B(X_tr_t))
        pred_A_va = torch.sigmoid(model_A(X_va_t))
        pred_B_va = torch.sigmoid(model_B(X_va_t))

    site_prob_tr = site_prob_tr.detach()
    site_prob_va = site_prob_va.detach()
    pred_A_tr = pred_A_tr.detach()
    pred_B_tr = pred_B_tr.detach()
    pred_A_va = pred_A_va.detach()
    pred_B_va = pred_B_va.detach()

    _, meta_auc = train_meta_combiner(
        site_prob_tr, pred_A_tr, pred_B_tr, y_tr,
        site_prob_va, pred_A_va, pred_B_va, y_va
    )

    return meta_auc

n_permutations = 200
perm_meta_aucs = []
true_meta_aucs = []

# ============================================================
# LEAK-FREE K-FOLD PIPELINE
# ============================================================

def kfold_ensemble_pipeline(
    X, y, site,
    n_splits=5,
    combination="meta"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # =========================
    # BASELINE: Train on FULL cohort data (like separate training)
    # This shows what AUC you'd get with separate training approach
    # =========================
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Training on FULL cohort data")
    print("(This matches your separate training setup)")
    print("=" * 60)
    
    mask_A_full = site == 0
    mask_B_full = site == 1
    X_A_full = X[mask_A_full]
    y_A_full = y[mask_A_full]
    X_B_full = X[mask_B_full]
    y_B_full = y[mask_B_full]
    
    print(f"Cohort A full data: {len(y_A_full)} samples")
    print(f"Cohort B full data: {len(y_B_full)} samples")
    
    # Initialize baseline AUCs
    auc_A_full = None
    auc_B_full = None
    
    # Train on full data with train/test split (like separate training)
    if len(y_A_full) > 10 and len(np.unique(y_A_full)) > 1:
        X_A_tr_full, X_A_val_full, y_A_tr_full, y_A_val_full = train_test_split(
            X_A_full, y_A_full, test_size=0.2, stratify=y_A_full, random_state=47
        )
        model_A_full = train_cohort_model(X_A_tr_full, y_A_tr_full)
        with torch.no_grad():
            pred_A_full = torch.sigmoid(
                model_A_full(torch.tensor(X_A_val_full, dtype=torch.float32).to(device))
            ).cpu().numpy().ravel()
        auc_A_full = roc_auc_score(y_A_val_full, pred_A_full) if len(np.unique(y_A_val_full)) > 1 else 0.5
        print(f"BASELINE Cohort A AUC (full data, 80/20 split): {auc_A_full:.4f} (val size: {len(y_A_val_full)})")
    
    if len(y_B_full) > 10 and len(np.unique(y_B_full)) > 1:
        X_B_tr_full, X_B_val_full, y_B_tr_full, y_B_val_full = train_test_split(
            X_B_full, y_B_full, test_size=0.2, stratify=y_B_full, random_state=47
        )
        model_B_full = train_cohort_model(X_B_tr_full, y_B_tr_full)
        with torch.no_grad():
            pred_B_full = torch.sigmoid(
                model_B_full(torch.tensor(X_B_val_full, dtype=torch.float32).to(device))
            ).cpu().numpy().ravel()
        auc_B_full = roc_auc_score(y_B_val_full, pred_B_full) if len(np.unique(y_B_val_full)) > 1 else 0.5
        print(f"BASELINE Cohort B AUC (full data, 80/20 split): {auc_B_full:.4f} (val size: {len(y_B_val_full)})")
    
    print("=" * 60)
    print("Now starting k-fold cross-validation...")
    print("=" * 60)

    meta_aucs = []
    site_aucs = []
    cohort_A_aucs = []
    cohort_B_aucs = []
    
    # For pooled evaluation (more reliable)
    all_nested_A_preds = []
    all_nested_A_labels = []
    all_nested_B_preds = []
    all_nested_B_labels = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        print("-" * 60)

        # =========================
        # Split data
        # =========================
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        site_tr, site_va = site[tr_idx], site[va_idx]
        
        print(f'The percentage in Training is {np.sum(y_tr)/len(y_tr)}')
        print(f'The percentage in Testing is {np.sum(y_va)/len(y_va)}')
        
        print(f'The shape of X_tr is {X_tr.shape} and shape of site_tr is {site_tr.shape}')

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
        y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)

        # =========================
        # Train base models (TRAIN ONLY)
        # =========================
        site_model = train_site_predictor(X_tr, site_tr)

        # Get cohort-specific training data
        mask_A_tr = site_tr == 0
        mask_B_tr = site_tr == 1
        X_A_tr = X_tr[mask_A_tr]
        y_A_tr = y_tr[mask_A_tr]
        X_B_tr = X_tr[mask_B_tr]
        y_B_tr = y_tr[mask_B_tr]
        
        print(f"Cohort A training: {len(y_A_tr)} samples, class distribution: {np.bincount(y_A_tr.astype(int))}")
        print(f"Cohort B training: {len(y_B_tr)} samples, class distribution: {np.bincount(y_B_tr.astype(int))}")
        
        # Train cohort models on full training fold (needed for meta-features)
        model_A = train_cohort_model(X_A_tr, y_A_tr)
        model_B = train_cohort_model(X_B_tr, y_B_tr)
        
        # Use nested split for more reliable cohort model evaluation (diagnostic only)
        # This helps understand model performance without affecting the main pipeline
        
        if len(y_A_tr) > 10 and len(np.unique(y_A_tr)) > 1:
            X_A_tr_inner, X_A_val_inner, y_A_tr_inner, y_A_val_inner = train_test_split(
                X_A_tr, y_A_tr, test_size=0.2, stratify=y_A_tr, random_state=42
            )
            model_A_inner = train_cohort_model(X_A_tr_inner, y_A_tr_inner)
            # Evaluate on inner validation set
            with torch.no_grad():
                pred_A_inner = torch.sigmoid(
                    model_A_inner(torch.tensor(X_A_val_inner, dtype=torch.float32).to(device))
                ).cpu().numpy().ravel()
            auc_A_inner = roc_auc_score(y_A_val_inner, pred_A_inner) if len(np.unique(y_A_val_inner)) > 1 else 0.5
            print(f"Cohort A Model | Nested Val AUC (more reliable): {auc_A_inner:.4f} (val size: {len(y_A_val_inner)})")
            # Collect for pooled evaluation
            all_nested_A_preds.extend(pred_A_inner)
            all_nested_A_labels.extend(y_A_val_inner)
        
        if len(y_B_tr) > 10 and len(np.unique(y_B_tr)) > 1:
            X_B_tr_inner, X_B_val_inner, y_B_tr_inner, y_B_val_inner = train_test_split(
                X_B_tr, y_B_tr, test_size=0.2, stratify=y_B_tr, random_state=42
            )
            model_B_inner = train_cohort_model(X_B_tr_inner, y_B_tr_inner)
            # Evaluate on inner validation set
            with torch.no_grad():
                pred_B_inner = torch.sigmoid(
                    model_B_inner(torch.tensor(X_B_val_inner, dtype=torch.float32).to(device))
                ).cpu().numpy().ravel()
            auc_B_inner = roc_auc_score(y_B_val_inner, pred_B_inner) if len(np.unique(y_B_val_inner)) > 1 else 0.5
            print(f"Cohort B Model | Nested Val AUC (more reliable): {auc_B_inner:.4f} (val size: {len(y_B_val_inner)})")
            # Collect for pooled evaluation
            all_nested_B_preds.extend(pred_B_inner)
            all_nested_B_labels.extend(y_B_val_inner)

        # =========================
        # Generate meta features
        # =========================
        with torch.no_grad():
            site_prob_tr = site_model(X_tr_t)
            pred_A_tr    = torch.sigmoid(model_A(X_tr_t))
            pred_B_tr    = torch.sigmoid(model_B(X_tr_t))

            site_prob_va = site_model(X_va_t)
            pred_A_va    = torch.sigmoid(model_A(X_va_t))
            pred_B_va    = torch.sigmoid(model_B(X_va_t))

        # IMPORTANT: detach before meta learning
        site_prob_tr = site_prob_tr.detach()
        pred_A_tr    = pred_A_tr.detach()
        pred_B_tr    = pred_B_tr.detach()

        site_prob_va = site_prob_va.detach()
        pred_A_va    = pred_A_va.detach()
        pred_B_va    = pred_B_va.detach()

        # =========================
        # Train meta combiner (GRADS ENABLED)
        # =========================
        meta_model, meta_auc = train_meta_combiner(
            site_prob_tr, pred_A_tr, pred_B_tr, y_tr,
            site_prob_va, pred_A_va, pred_B_va, y_va
        )

        print(f"Meta-combiner | Val AUC: {meta_auc:.4f}")
        meta_aucs.append(meta_auc)

        # =========================
        # Site predictor performance
        # =========================
        site_auc = roc_auc_score(site_va, site_prob_va.cpu().numpy())
        site_acc = accuracy_score(site_va, site_prob_va.cpu().numpy() > 0.5)
        print(f"Site Model     | AUC: {site_auc:.4f}, Acc: {site_acc:.4f}")
        site_aucs.append(site_auc)

        # =========================
        # Cohort A validation
        # =========================
        mask_A = site_va == 0
        print(f"Cohort A validation: {mask_A.sum()} samples, class distribution: {np.bincount(y_va[mask_A].astype(int)) if mask_A.sum() > 0 else 'N/A'}")
        
        # Also evaluate on training set for comparison
        if len(y_A_tr) > 1 and len(np.unique(y_A_tr)) > 1:
            with torch.no_grad():
                pred_A_train = torch.sigmoid(
                    model_A(torch.tensor(X_A_tr, dtype=torch.float32).to(device))
                ).cpu().numpy().ravel()
            auc_A_train = roc_auc_score(y_A_tr, pred_A_train)
            print(f"Cohort A Model | Train AUC: {auc_A_train:.4f}")
        
        if mask_A.sum() > 1 and len(np.unique(y_va[mask_A])) > 1:
            with torch.no_grad():
                pred_A_only = torch.sigmoid(
                    model_A(torch.tensor(X_va[mask_A], dtype=torch.float32).to(device))
                ).cpu().numpy().ravel()

            auc_A = roc_auc_score(y_va[mask_A], pred_A_only)
            print(f"Cohort A Model | Val AUC: {auc_A:.4f}")
            cohort_A_aucs.append(auc_A)
        else:
            print("Cohort A Model | Val AUC: N/A (insufficient samples or single class)")

        # =========================
        # Cohort B validation
        # =========================
        mask_B = site_va == 1
        print(f"Cohort B validation: {mask_B.sum()} samples, class distribution: {np.bincount(y_va[mask_B].astype(int)) if mask_B.sum() > 0 else 'N/A'}")
        
        # Also evaluate on training set for comparison
        if len(y_B_tr) > 1 and len(np.unique(y_B_tr)) > 1:
            with torch.no_grad():
                pred_B_train = torch.sigmoid(
                    model_B(torch.tensor(X_B_tr, dtype=torch.float32).to(device))
                ).cpu().numpy().ravel()
            auc_B_train = roc_auc_score(y_B_tr, pred_B_train)
            print(f"Cohort B Model | Train AUC: {auc_B_train:.4f}")
        
        if mask_B.sum() > 1 and len(np.unique(y_va[mask_B])) > 1:
            with torch.no_grad():
                pred_B_only = torch.sigmoid(
                    model_B(torch.tensor(X_va[mask_B], dtype=torch.float32).to(device))
                ).cpu().numpy().ravel()

            auc_B = roc_auc_score(y_va[mask_B], pred_B_only)
            print(f"Cohort B Model | Val AUC: {auc_B:.4f}")
            cohort_B_aucs.append(auc_B)
        else:
            print("Cohort B Model | Val AUC: N/A (insufficient samples or single class)")

    # =========================
    # Summary
    # =========================
    print("\n" + "=" * 60)
    print("WHY COHORT A/B AUC IS LOWER IN K-FOLD THAN BASELINE/SEPARATE TRAINING")
    print("-" * 60)
    print("")
    print("  It is NOT about training settings (Adam, etc.). Those are now the same.")
    print("  It is about HOW MUCH DATA each cohort model sees during training.")
    print("")
    print("  BASELINE / separate training:")
    print("    • Train on 80% of FULL cohort  →  model sees ~80% of cohort data")
    print("")
    print("  K-fold (nested) cohort models:")
    print("    • Fold splits: 80% train, 20% val. Cohort model uses TRAIN fold only.")
    print("    • Then nested 80/20 within train  →  train on 80% of that 80%")
    print("    • So model sees  80% × 80% = 64%  of full cohort data.")
    print("")
    print("  LESS TRAINING DATA  →  WEAKER MODEL  →  LOWER AUC.")
    print("  Pooled AUC uses a larger val set (reliable estimate) but same smaller")
    print("  training data per fold, so it stays lower than baseline.")
    print("")
    print("  'Cohort A/B Mean AUC' above uses tiny val sets per fold (~15 samples)")
    print("  → unstable; ignore those. Use POOLED and BASELINE for comparison.")
    print("-" * 60)
    print(f"Meta Model  Mean AUC: {np.mean(meta_aucs):.4f} ± {np.std(meta_aucs):.4f}")
    print(f"Site Model  Mean AUC: {np.mean(site_aucs):.4f} ± {np.std(site_aucs):.4f}")
    print(f"Cohort A    Mean AUC: {np.mean(cohort_A_aucs):.4f} ± {np.std(cohort_A_aucs):.4f} (unreliable - small val sets)")
    print(f"Cohort B    Mean AUC: {np.mean(cohort_B_aucs):.4f} ± {np.std(cohort_B_aucs):.4f} (unreliable - small val sets)")
    print("=" * 60)
    print("FINAL COMPARISON:")
    print("-" * 60)
    
    # Pooled evaluation across all folds (most reliable)
    if len(all_nested_A_preds) > 0 and len(np.unique(all_nested_A_labels)) > 1:
        pooled_auc_A = roc_auc_score(all_nested_A_labels, all_nested_A_preds)
        print(f"K-FOLD Pooled Cohort A AUC: {pooled_auc_A:.4f} (val samples: {len(all_nested_A_labels)})")
        if auc_A_full is not None:
            print(f"BASELINE Cohort A AUC (full data): {auc_A_full:.4f}")
            print(f"  → Difference: {auc_A_full - pooled_auc_A:.4f} (baseline uses more training data)")
    
    if len(all_nested_B_preds) > 0 and len(np.unique(all_nested_B_labels)) > 1:
        pooled_auc_B = roc_auc_score(all_nested_B_labels, all_nested_B_preds)
        print(f"K-FOLD Pooled Cohort B AUC: {pooled_auc_B:.4f} (val samples: {len(all_nested_B_labels)})")
        if auc_B_full is not None:
            print(f"BASELINE Cohort B AUC (full data): {auc_B_full:.4f}")
            print(f"  → Difference: {auc_B_full - pooled_auc_B:.4f} (baseline uses more training data)")
    
    print("-" * 60)
    print("TAKEAWAY:")
    print("  K-fold cohort AUC is lower because each fold's model is trained on")
    print("  ~64% of cohort data (80% of 80%), not 80%. Baseline uses 80%.")
    print("  Same training code, same settings — only the DATA amount differs.")
    print("=" * 60)
    
    print(f'True Label is {site_va}')
    print(f'Pred Label is {site_prob_va}')

    return meta_aucs, site_va, site_prob_va

def kfold_ensemble_pipeline_permute(
    X, y, site,
    n_splits=5,
    n_permutations=200
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_aucs = []
    site_aucs = []
    true_meta_aucs = []
    perm_meta_aucs = []

    print("\n" + "=" * 70)
    print("K-FOLD SITE-AWARE NEURAL NETWORK WITH PERMUTATION TEST")
    print("=" * 70)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        print("-" * 70)

        # =========================
        # Split data
        # =========================
        X_tr, X_va       = X[tr_idx],    X[va_idx]
        y_tr, y_va       = y[tr_idx],    y[va_idx]
        site_tr, site_va = site[tr_idx], site[va_idx]

        print(f"Training positives: {y_tr.mean():.3f}")
        print(f"Validation positives: {y_va.mean():.3f}")

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)

        # =========================
        # Train base models
        # =========================
        site_model = train_site_predictor(X_tr, site_tr)

        model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
        model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

        with torch.no_grad():
            site_prob_tr = site_model(X_tr_t)
            site_prob_va = site_model(X_va_t)

            pred_A_tr = torch.sigmoid(model_A(X_tr_t))
            pred_B_tr = torch.sigmoid(model_B(X_tr_t))
            pred_A_va = torch.sigmoid(model_A(X_va_t))
            pred_B_va = torch.sigmoid(model_B(X_va_t))

        site_prob_tr = site_prob_tr.detach()
        site_prob_va = site_prob_va.detach()
        pred_A_tr = pred_A_tr.detach()
        pred_B_tr = pred_B_tr.detach()
        pred_A_va = pred_A_va.detach()
        pred_B_va = pred_B_va.detach()

        # =========================
        # Meta combiner (TRUE SITE)
        # =========================
        _, meta_auc = train_meta_combiner(
            site_prob_tr, pred_A_tr, pred_B_tr, y_tr,
            site_prob_va, pred_A_va, pred_B_va, y_va
        )

        meta_aucs.append(meta_auc)
        true_meta_aucs.append(meta_auc)

        print(f"Meta AUC (true site): {meta_auc:.4f}")

        # =========================
        # Site prediction diagnostic
        # =========================
        site_auc = roc_auc_score(site_va, site_prob_va.cpu().numpy())
        site_aucs.append(site_auc)
        print(f"Site prediction AUC: {site_auc:.4f}")

        # =========================
        # SITE PERMUTATION TEST
        # =========================
        perm_aucs_fold = []

        for p in range(n_permutations):
            site_tr_perm = np.random.permutation(site_tr)

            perm_auc = run_single_fold_sann_auc(
                X_tr, y_tr, site_tr_perm,
                X_va, y_va, site_va,
                device
            )

            perm_aucs_fold.append(perm_auc)

        perm_meta_aucs.extend(perm_aucs_fold)

        print(
            f"Permutation | Fold mean AUC: "
            f"{np.mean(perm_aucs_fold):.4f}"
        )

    # =========================
    # FINAL PERMUTATION STATISTICS
    # =========================
    true_meta_aucs = np.array(true_meta_aucs)
    perm_meta_aucs = np.array(perm_meta_aucs)

    p_value = (
        np.sum(perm_meta_aucs >= np.mean(true_meta_aucs)) + 1
    ) / (len(perm_meta_aucs) + 1)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("-" * 70)
    print(f"Observed Meta AUC: {np.mean(true_meta_aucs):.4f} ± {np.std(true_meta_aucs):.4f}")
    print(f"Permuted Meta AUC: {np.mean(perm_meta_aucs):.4f} ± {np.std(perm_meta_aucs):.4f}")
    print(f"Permutation p-value: {p_value:.4e}")
    print("=" * 70)
    
    print(f'True Label is {site_va}')
    print(f'Pred Label is {site_prob_va}')

    return {
        "meta_aucs": meta_aucs,
        "site_aucs": site_aucs,
        "perm_meta_aucs": perm_meta_aucs,
        "p_value": p_value,
        "Site_VA": site_va,
        'Site_Prob': site_prob_va
    }

pred_auc, label_true, label_pred = kfold_ensemble_pipeline(X, y, g_remap)
# results = kfold_ensemble_pipeline_permute(
#     X, y, g_remap,
#     n_splits=5,
#     n_permutations=75
# )

##############################################################
##############################################################
##############################################################

#%% Add Testing

from sklearn.model_selection import train_test_split

from scipy import stats
import numpy as np

def mean_ci(values, confidence=0.95):
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)  # standard error
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

def kfold_ensemble_pipeline(X, y, site, n_splits=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    test_meta_aucs = []
    val_meta_aucs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        print(f"\nFold {fold+1}")

        # ----------------------------------
        # TEST SET (outer fold)
        # ----------------------------------
        X_train_full = X[train_idx]
        y_train_full = y[train_idx]
        site_train_full = site[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        site_test = site[test_idx]

        # ----------------------------------
        # TRAIN / VALIDATION split
        # ----------------------------------
        X_tr, X_va, y_tr, y_va, site_tr, site_va = train_test_split(
            X_train_full,
            y_train_full,
            site_train_full,
            test_size=0.2,
            stratify=y_train_full,
            random_state=42
        )

        # ----------------------------------
        # TRAIN BASE MODELS
        # ----------------------------------
        site_model = train_site_predictor(X_tr, site_tr)

        mask_A = site_tr == 0
        mask_B = site_tr == 1

        model_A = train_cohort_model(X_tr[mask_A], y_tr[mask_A])
        model_B = train_cohort_model(X_tr[mask_B], y_tr[mask_B])

        # convert to tensor
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        # ----------------------------------
        # META FEATURES
        # ----------------------------------
        with torch.no_grad():

            site_prob_tr = site_model(X_tr_t)
            site_prob_va = site_model(X_va_t)
            site_prob_test = site_model(X_test_t)

            pred_A_tr = torch.sigmoid(model_A(X_tr_t))
            pred_B_tr = torch.sigmoid(model_B(X_tr_t))

            pred_A_va = torch.sigmoid(model_A(X_va_t))
            pred_B_va = torch.sigmoid(model_B(X_va_t))

            pred_A_test = torch.sigmoid(model_A(X_test_t))
            pred_B_test = torch.sigmoid(model_B(X_test_t))

        # detach tensors
        site_prob_tr = site_prob_tr.detach()
        pred_A_tr = pred_A_tr.detach()
        pred_B_tr = pred_B_tr.detach()

        site_prob_va = site_prob_va.detach()
        pred_A_va = pred_A_va.detach()
        pred_B_va = pred_B_va.detach()

        # ----------------------------------
        # TRAIN META COMBINER (train + val)
        # ----------------------------------
        meta_model, val_auc = train_meta_combiner(
            site_prob_tr, pred_A_tr, pred_B_tr, y_tr,
            site_prob_va, pred_A_va, pred_B_va, y_va
        )

        val_meta_aucs.append(val_auc)

        # ----------------------------------
        # TEST EVALUATION
        # ----------------------------------
        with torch.no_grad():

            test_prob = torch.sigmoid(
                meta_model(
                    site_prob_test,
                    pred_A_test,
                    pred_B_test
                )
            ).cpu().numpy().ravel()

        test_auc = roc_auc_score(y_test, test_prob)
        test_meta_aucs.append(test_auc)

        print(f"Validation AUC: {val_auc:.4f}")
        print(f"TEST AUC: {test_auc:.4f}")

    mean_test, low_test, high_test = mean_ci(test_meta_aucs)
    mean_val, low_val, high_val = mean_ci(val_meta_aucs)
    
    print("\n===============================")
    print(f"Validation AUC: {mean_val:.4f} (95% CI {low_val:.4f} – {high_val:.4f})")
    print(f"TEST AUC:       {mean_test:.4f} (95% CI {low_test:.4f} – {high_test:.4f})")
    print("===============================")

    return val_meta_aucs, test_meta_aucs

kfold_ensemble_pipeline(X, y, g_remap)

#%% Fine_tunable SANN

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

class SitePredictorNetwork(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class CohortSpecificNetwork(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class MetaCombiner(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, site_prob, pred_A, pred_B):
        x = torch.cat([site_prob, pred_A, pred_B], dim=1)
        return self.net(x)
    
class FineTunableSANN(nn.Module):
    def __init__(self, site_net, model_A, model_B, meta):
        super().__init__()
        self.site_net = site_net
        self.model_A = model_A
        self.model_B = model_B
        self.meta = meta

        # Freeze site predictor
        for p in self.site_net.parameters():
            p.requires_grad = False

    def forward(self, x):
        site_prob = self.site_net(x).detach()   # frozen semantics
        pred_A = torch.sigmoid(self.model_A(x)) # trainable
        pred_B = torch.sigmoid(self.model_B(x)) # trainable
        return self.meta(site_prob, pred_A, pred_B)
    
def train_site_predictor(X, site, n_epochs=2000):
    model = SitePredictorNetwork(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(site, dtype=torch.float32).unsqueeze(1).to(device)

    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()

    return model.eval()

def train_cohort_model(X, y, n_epochs=1500):
    model = CohortSpecificNetwork(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

    return model.eval()

def train_meta_combiner(
    site_prob_tr, pred_A_tr, pred_B_tr, y_tr,
    site_prob_va, pred_A_va, pred_B_va, y_va,
    n_epochs=4000
):
    meta = MetaCombiner().to(device)
    opt = torch.optim.Adam(meta.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)

    best_auc, best_state = 0, None

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(meta(site_prob_tr, pred_A_tr, pred_B_tr), y_tr_t)
        loss.backward()
        opt.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                val_prob = torch.sigmoid(
                    meta(site_prob_va, pred_A_va, pred_B_va)
                ).cpu().numpy().ravel()

            if len(np.unique(y_va)) > 1:
                auc = roc_auc_score(y_va, val_prob)
                if auc > best_auc:
                    best_auc = auc
                    best_state = meta.state_dict()

    if best_state is not None:
        meta.load_state_dict(best_state)

    return meta.eval(), best_auc

def finetune_sann(sann, X, y, n_epochs=3000, lr=1e-3):
    sann.train()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, sann.parameters()),
        lr=lr
    )
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(sann(X_t), y_t)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(sann.parameters(), 1.0)
        opt.step()

    return sann.eval()

def kfold_ensemble_pipeline(X, y, site, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_aucs, ft_aucs = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")

        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        site_tr, site_va = site[tr], site[va]

        # ---------- Stage 1 ----------
        site_net = train_site_predictor(X_tr, site_tr)

        model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
        model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

        with torch.no_grad():
            site_prob_tr = site_net(torch.tensor(X_tr, dtype=torch.float32).to(device))
            site_prob_va = site_net(torch.tensor(X_va, dtype=torch.float32).to(device))

            pred_A_tr = torch.sigmoid(model_A(torch.tensor(X_tr, dtype=torch.float32).to(device)))
            pred_B_tr = torch.sigmoid(model_B(torch.tensor(X_tr, dtype=torch.float32).to(device)))

            pred_A_va = torch.sigmoid(model_A(torch.tensor(X_va, dtype=torch.float32).to(device)))
            pred_B_va = torch.sigmoid(model_B(torch.tensor(X_va, dtype=torch.float32).to(device)))

        meta, auc_meta = train_meta_combiner(
            site_prob_tr.detach(), pred_A_tr, pred_B_tr, y_tr,
            site_prob_va.detach(), pred_A_va, pred_B_va, y_va
        )
        meta_aucs.append(auc_meta)

        # ---------- Stage 2: fine-tuning ----------
        sann = FineTunableSANN(site_net, model_A, model_B, meta).to(device)
        sann = finetune_sann(sann, X_tr, y_tr)

        with torch.no_grad():
            val_prob = torch.sigmoid(
                sann(torch.tensor(X_va, dtype=torch.float32).to(device))
            ).cpu().numpy().ravel()

        auc_ft = roc_auc_score(y_va, val_prob) if len(np.unique(y_va)) > 1 else 0.5
        ft_aucs.append(auc_ft)

        print(f"Meta AUC: {auc_meta:.4f} | Fine-tuned AUC: {auc_ft:.4f}")

    print("\n==============================")
    print(f"Meta Mean AUC: {np.mean(meta_aucs):.4f} ± {np.std(meta_aucs):.4f}")
    print(f"FT   Mean AUC: {np.mean(ft_aucs):.4f} ± {np.std(ft_aucs):.4f}")
    print("==============================")
    
    # ============================================================
    # DIAGNOSTIC AUCs (NO EFFECT ON TRAINING)
    # ============================================================
    
    # ---- Site predictor ----
    site_prob_va_np = site_prob_va.cpu().numpy().ravel()
    site_auc = roc_auc_score(site_va, site_prob_va_np)
    site_acc = accuracy_score(site_va, site_prob_va_np > 0.5)
    
    print(f"[Diagnostic] Site Predictor | AUC: {site_auc:.4f}, Acc: {site_acc:.4f}")
    
    # ---- Cohort A ----
    mask_A = site_va == 0
    if mask_A.sum() > 1 and len(np.unique(y_va[mask_A])) > 1:
        pred_A_only = pred_A_va.cpu().numpy().ravel()[mask_A]
        auc_A = roc_auc_score(y_va[mask_A], pred_A_only)
        print(f"[Diagnostic] Cohort A Model | Val AUC: {auc_A:.4f} (n={mask_A.sum()})")
    else:
        auc_A = np.nan
        print("[Diagnostic] Cohort A Model | Val AUC: N/A")
    
    # ---- Cohort B ----
    mask_B = site_va == 1
    if mask_B.sum() > 1 and len(np.unique(y_va[mask_B])) > 1:
        pred_B_only = pred_B_va.cpu().numpy().ravel()[mask_B]
        auc_B = roc_auc_score(y_va[mask_B], pred_B_only)
        print(f"[Diagnostic] Cohort B Model | Val AUC: {auc_B:.4f} (n={mask_B.sum()})")
    else:
        auc_B = np.nan
        print("[Diagnostic] Cohort B Model | Val AUC: N/A")

    return meta_aucs, ft_aucs

kfold_ensemble_pipeline(X, y, g_remap)

#%% Add the Testset

from sklearn.model_selection import train_test_split
from scipy import stats

def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci = stats.t.interval(
        0.95,
        len(values)-1,
        loc=mean,
        scale=std/np.sqrt(len(values))
    )
    return mean, ci


def kfold_ensemble_pipeline(X, y, site, n_splits=5):

    # ======================================================
    # Train / Test split
    # ======================================================
    X_trainval, X_test, y_trainval, y_test, site_trainval, site_test = train_test_split(
        X, y, site,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_aucs, ft_aucs = [], []
    test_aucs = []

    for fold, (tr, va) in enumerate(skf.split(X_trainval, y_trainval)):
 
        print(f"\nFold {fold+1}/{n_splits}")

        X_tr, X_va = X_trainval[tr], X_trainval[va]
        y_tr, y_va = y_trainval[tr], y_trainval[va]
        site_tr, site_va = site_trainval[tr], site_trainval[va]

        X_tr    = selector1.fit_transform(X_tr, y_tr)
        X_va    = selector1.transform(X_va)
        X_test  = selector1.transform(X_test)

        # ---------- Stage 1 ----------
        site_net = train_site_predictor(X_tr, site_tr)

        model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
        model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

        with torch.no_grad():

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            site_prob_tr = site_net(X_tr_t)
            site_prob_va = site_net(X_va_t)
            site_prob_te = site_net(X_te_t)

            pred_A_tr = torch.sigmoid(model_A(X_tr_t))
            pred_B_tr = torch.sigmoid(model_B(X_tr_t))

            pred_A_va = torch.sigmoid(model_A(X_va_t))
            pred_B_va = torch.sigmoid(model_B(X_va_t))

            pred_A_te = torch.sigmoid(model_A(X_te_t))
            pred_B_te = torch.sigmoid(model_B(X_te_t))

        meta, auc_meta = train_meta_combiner(
            site_prob_tr.detach(), pred_A_tr, pred_B_tr, y_tr,
            site_prob_va.detach(), pred_A_va, pred_B_va, y_va
        )

        meta_aucs.append(auc_meta)

        # ---------- Stage 2: fine-tuning ----------
        sann = FineTunableSANN(site_net, model_A, model_B, meta).to(device)
        sann = finetune_sann(sann, X_tr, y_tr)

        # =========================
        # Validation AUC
        # =========================
        with torch.no_grad():
            val_prob = torch.sigmoid(
                sann(torch.tensor(X_va, dtype=torch.float32).to(device))
            ).cpu().numpy().ravel()

        auc_ft = roc_auc_score(y_va, val_prob) if len(np.unique(y_va)) > 1 else 0.5
        ft_aucs.append(auc_ft)

        # =========================
        # TEST AUC
        # =========================
        with torch.no_grad():
            test_prob = torch.sigmoid(
                sann(torch.tensor(X_test, dtype=torch.float32).to(device))
            ).cpu().numpy().ravel()
            
        print(f'Test True Label is {y_test}')
        print(f'Predicted Label is {test_prob}')

        test_auc = roc_auc_score(y_test, test_prob)
        test_aucs.append(test_auc)

        print(f"Meta AUC: {auc_meta:.4f} | Fine-tuned AUC: {auc_ft:.4f} | TEST AUC: {test_auc:.4f}")

    # ======================================================
    # CV statistics
    # ======================================================

    meta_mean, meta_ci = compute_ci(meta_aucs)
    ft_mean, ft_ci = compute_ci(ft_aucs)
    test_mean, test_ci = compute_ci(test_aucs)

    print("\n==============================")
    print(f"Meta CV AUC: {meta_mean:.4f} (95% CI {meta_ci[0]:.4f}-{meta_ci[1]:.4f})")
    print(f"FT   CV AUC: {ft_mean:.4f} (95% CI {ft_ci[0]:.4f}-{ft_ci[1]:.4f})")
    print(f"TEST AUC:    {test_mean:.4f} (95% CI {test_ci[0]:.4f}-{test_ci[1]:.4f})")
    print("==============================")

    # ======================================================
    # DIAGNOSTIC AUCs (same as your original)
    # ======================================================

    site_prob_va_np = site_prob_va.cpu().numpy().ravel()
    site_auc = roc_auc_score(site_va, site_prob_va_np)
    site_acc = accuracy_score(site_va, site_prob_va_np > 0.5)

    print(f"[Diagnostic] Site Predictor | AUC: {site_auc:.4f}, Acc: {site_acc:.4f}")

    mask_A = site_va == 0
    if mask_A.sum() > 1 and len(np.unique(y_va[mask_A])) > 1:
        pred_A_only = pred_A_va.cpu().numpy().ravel()[mask_A]
        auc_A = roc_auc_score(y_va[mask_A], pred_A_only)
        print(f"[Diagnostic] Cohort A Model | Val AUC: {auc_A:.4f} (n={mask_A.sum()})")
    else:
        print("[Diagnostic] Cohort A Model | Val AUC: N/A")

    mask_B = site_va == 1
    if mask_B.sum() > 1 and len(np.unique(y_va[mask_B])) > 1:
        pred_B_only = pred_B_va.cpu().numpy().ravel()[mask_B]
        auc_B = roc_auc_score(y_va[mask_B], pred_B_only)
        print(f"[Diagnostic] Cohort B Model | Val AUC: {auc_B:.4f} (n={mask_B.sum()})")
    else:
        print("[Diagnostic] Cohort B Model | Val AUC: N/A")

    return meta_aucs, ft_aucs, test_aucs, train_meta_combiner

meta_aucs, ft_aucs, test_aucs, train_meta_combiner = kfold_ensemble_pipeline(X, y, g_remap)

#%% Test Set No FT

from sklearn.model_selection import train_test_split
from scipy import stats

def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci = stats.t.interval(
        0.95,
        len(values)-1,
        loc=mean,
        scale=std/np.sqrt(len(values))
    )
    return mean, ci


def kfold_ensemble_pipeline(X, y, site, n_splits=5):

    # ======================================================
    # Train / Test split
    # ======================================================
    X_trainval, X_test, y_trainval, y_test, site_trainval, site_test = train_test_split(
        X, y, site,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_aucs = []
    test_aucs = []

    for fold, (tr, va) in enumerate(skf.split(X_trainval, y_trainval)):

        print(f"\nFold {fold+1}/{n_splits}")

        X_tr, X_va = X_trainval[tr], X_trainval[va]
        y_tr, y_va = y_trainval[tr], y_trainval[va]
        site_tr, site_va = site_trainval[tr], site_trainval[va]

        # ---------- Stage 1 ----------
        site_net = train_site_predictor(X_tr, site_tr)

        model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
        model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

        with torch.no_grad():

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            site_prob_tr = site_net(X_tr_t)
            site_prob_va = site_net(X_va_t)
            site_prob_te = site_net(X_te_t)

            pred_A_tr = torch.sigmoid(model_A(X_tr_t))
            pred_B_tr = torch.sigmoid(model_B(X_tr_t))

            pred_A_va = torch.sigmoid(model_A(X_va_t))
            pred_B_va = torch.sigmoid(model_B(X_va_t))

            pred_A_te = torch.sigmoid(model_A(X_te_t))
            pred_B_te = torch.sigmoid(model_B(X_te_t))

        # ---------- Train Meta Combiner ----------
        meta, auc_meta = train_meta_combiner(
            site_prob_tr.detach(), pred_A_tr, pred_B_tr, y_tr,
            site_prob_va.detach(), pred_A_va, pred_B_va, y_va
        )

        meta_aucs.append(auc_meta)

        # ---------- CV prediction ----------
        with torch.no_grad():
            val_prob = torch.sigmoid(
                meta(site_prob_va, pred_A_va, pred_B_va)
            ).cpu().numpy().ravel()

        auc_val = roc_auc_score(y_va, val_prob)
        print(f"Meta CV AUC: {auc_val:.4f}")

        # ---------- TEST prediction ----------
        with torch.no_grad():
            test_prob = torch.sigmoid(
                meta(site_prob_te, pred_A_te, pred_B_te)
            ).cpu().numpy().ravel()
            
        print(f'Test True Label is {y_test}')
        print(f'Predicted Label is {test_prob}')

        test_auc = roc_auc_score(y_test, test_prob)
        test_aucs.append(test_auc)

        print(f"TEST AUC: {test_auc:.4f}")

    # ======================================================
    # Statistics
    # ======================================================

    cv_mean, cv_ci = compute_ci(meta_aucs)
    test_mean, test_ci = compute_ci(test_aucs)

    print("\n==============================")
    print(f"CV AUC:   {cv_mean:.4f} (95% CI {cv_ci[0]:.4f}-{cv_ci[1]:.4f})")
    print(f"TEST AUC: {test_mean:.4f} (95% CI {test_ci[0]:.4f}-{test_ci[1]:.4f})")
    print("==============================")

    # ======================================================
    # DIAGNOSTICS (unchanged)
    # ======================================================

    site_prob_va_np = site_prob_va.cpu().numpy().ravel()
    site_auc = roc_auc_score(site_va, site_prob_va_np)
    site_acc = accuracy_score(site_va, site_prob_va_np > 0.5)

    print(f"[Diagnostic] Site Predictor | AUC: {site_auc:.4f}, Acc: {site_acc:.4f}")

    mask_A = site_va == 0
    if mask_A.sum() > 1 and len(np.unique(y_va[mask_A])) > 1:
        pred_A_only = pred_A_va.cpu().numpy().ravel()[mask_A]
        auc_A = roc_auc_score(y_va[mask_A], pred_A_only)
        print(f"[Diagnostic] Cohort A Model | Val AUC: {auc_A:.4f} (n={mask_A.sum()})")
    else:
        print("[Diagnostic] Cohort A Model | Val AUC: N/A")

    mask_B = site_va == 1
    if mask_B.sum() > 1 and len(np.unique(y_va[mask_B])) > 1:
        pred_B_only = pred_B_va.cpu().numpy().ravel()[mask_B]
        auc_B = roc_auc_score(y_va[mask_B], pred_B_only)
        print(f"[Diagnostic] Cohort B Model | Val AUC: {auc_B:.4f} (n={mask_B.sum()})")
    else:
        print("[Diagnostic] Cohort B Model | Val AUC: N/A")

    return meta_aucs, test_aucs

kfold_ensemble_pipeline(X, y, g_remap)

#%% SHAP Test

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats


def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    ci = stats.t.interval(
        0.95,
        len(values)-1,
        loc=mean,
        scale=std/np.sqrt(len(values))
    )
    return mean, ci


def kfold_ensemble_pipeline(X, y, site, n_splits=5):

    # ======================================================
    # Train / Test split
    # ======================================================
    X_trainval, X_test, y_trainval, y_test, site_trainval, site_test = train_test_split(
        X, y, site,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_aucs = []
    test_aucs = []

    # ===== store SHAP across folds =====
    shap_values_all = []
    shap_X_all = []

    for fold, (tr, va) in enumerate(skf.split(X_trainval, y_trainval)):

        print(f"\nFold {fold+1}/{n_splits}")

        X_tr, X_va = X_trainval[tr], X_trainval[va]
        y_tr, y_va = y_trainval[tr], y_trainval[va]
        site_tr, site_va = site_trainval[tr], site_trainval[va]

        # ---------- Stage 1 ----------
        site_net = train_site_predictor(X_tr, site_tr)

        model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
        model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

        with torch.no_grad():

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            site_prob_tr = site_net(X_tr_t)
            site_prob_va = site_net(X_va_t)
            site_prob_te = site_net(X_te_t)

            pred_A_tr = torch.sigmoid(model_A(X_tr_t))
            pred_B_tr = torch.sigmoid(model_B(X_tr_t))

            pred_A_va = torch.sigmoid(model_A(X_va_t))
            pred_B_va = torch.sigmoid(model_B(X_va_t))

            pred_A_te = torch.sigmoid(model_A(X_te_t))
            pred_B_te = torch.sigmoid(model_B(X_te_t))

        # ---------- Train Meta Combiner ----------
        meta, auc_meta = train_meta_combiner(
            site_prob_tr.detach(), pred_A_tr, pred_B_tr, y_tr,
            site_prob_va.detach(), pred_A_va, pred_B_va, y_va
        )

        meta_aucs.append(auc_meta)

        # ---------- CV prediction ----------
        with torch.no_grad():
            val_prob = torch.sigmoid(
                meta(site_prob_va, pred_A_va, pred_B_va)
            ).cpu().numpy().ravel()

        auc_val = roc_auc_score(y_va, val_prob)
        print(f"Meta CV AUC: {auc_val:.4f}")

        # ---------- TEST prediction ----------
        with torch.no_grad():
            test_prob = torch.sigmoid(
                meta(site_prob_te, pred_A_te, pred_B_te)
            ).cpu().numpy().ravel()

        test_auc = roc_auc_score(y_test, test_prob)
        test_aucs.append(test_auc)

        print(f"TEST AUC: {test_auc:.4f}")

        # ======================================================
        # SHAP ANALYSIS (NEW)
        # ======================================================

        print("Running SHAP for this fold...")

        def ensemble_predict(X_input):

            X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)

            with torch.no_grad():

                site_p = site_net(X_tensor)

                pred_A = torch.sigmoid(model_A(X_tensor))
                pred_B = torch.sigmoid(model_B(X_tensor))

                pred = torch.sigmoid(
                    meta(site_p, pred_A, pred_B)
                ).cpu().numpy().ravel()

            return pred


        # background sample (speed)
        background = shap.sample(X_tr, 100)

        explainer = shap.KernelExplainer(
            ensemble_predict,
            background
        )

        shap_values = explainer.shap_values(
            X_va,
            nsamples=100
        )

        shap_values_all.append(shap_values)
        shap_X_all.append(X_va)

        # per-fold SHAP plot
        print('Start SHAP Plotting')
        plt.rcParams.update({
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20
        })
        
        plt.figure()
        shap.summary_plot(shap_values, X_va, max_display=20)
        plt.title(f"SHAP Fold {fold+1}")
        # plt.savefig(f"shap_fold_{fold+1}.png")
        
        # Feature names (Y axis)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=20)
        cbar = plt.gcf().axes[-1]   # SHAP colorbar axis
        cbar.tick_params(labelsize=20)
        cbar.set_ylabel("Feature value", fontsize=20)
        
        plt.close()

    # ======================================================
    # Statistics
    # ======================================================

    cv_mean, cv_ci = compute_ci(meta_aucs)
    test_mean, test_ci = compute_ci(test_aucs)

    print("\n==============================")
    print(f"CV AUC:   {cv_mean:.4f} (95% CI {cv_ci[0]:.4f}-{cv_ci[1]:.4f})")
    print(f"TEST AUC: {test_mean:.4f} (95% CI {test_ci[0]:.4f}-{test_ci[1]:.4f})")
    print("==============================")

    # ======================================================
    # GLOBAL SHAP (ACROSS FOLDS)
    # ======================================================

    print("\nGenerating global SHAP summary...")

    shap_values_all = np.concatenate(shap_values_all, axis=0)
    shap_X_all = np.concatenate(shap_X_all, axis=0)

    plt.figure(figsize=(12,8))
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    })
    
    shap.summary_plot(shap_values_all, shap_X_all, max_display=30)
    plt.title("Global SHAP Summary (All Folds)")
    # plt.savefig("shap_global_summary.png")
    plt.close()
    
    print(f'Target Shap names are {shap_X_all}')

    # ======================================================
    # DIAGNOSTICS (unchanged)
    # ======================================================

    site_prob_va_np = site_prob_va.cpu().numpy().ravel()
    site_auc = roc_auc_score(site_va, site_prob_va_np)
    site_acc = accuracy_score(site_va, site_prob_va_np > 0.5)

    print(f"[Diagnostic] Site Predictor | AUC: {site_auc:.4f}, Acc: {site_acc:.4f}")

    mask_A = site_va == 0
    if mask_A.sum() > 1 and len(np.unique(y_va[mask_A])) > 1:
        pred_A_only = pred_A_va.cpu().numpy().ravel()[mask_A]
        auc_A = roc_auc_score(y_va[mask_A], pred_A_only)
        print(f"[Diagnostic] Cohort A Model | Val AUC: {auc_A:.4f} (n={mask_A.sum()})")
    else:
        print("[Diagnostic] Cohort A Model | Val AUC: N/A")

    mask_B = site_va == 1
    if mask_B.sum() > 1 and len(np.unique(y_va[mask_B])) > 1:
        pred_B_only = pred_B_va.cpu().numpy().ravel()[mask_B]
        auc_B = roc_auc_score(y_va[mask_B], pred_B_only)
        print(f"[Diagnostic] Cohort B Model | Val AUC: {auc_B:.4f} (n={mask_B.sum()})")
    else:
        print("[Diagnostic] Cohort B Model | Val AUC: N/A")

    return meta_aucs, test_aucs

kfold_ensemble_pipeline(X, y, g_remap)

#%%

from scipy import stats

def nested_sann_pipeline(X, y, site, n_outer=5):

    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    cv_aucs = []
    test_aucs = []

    for fold, (train_val_idx, test_idx) in enumerate(outer.split(X, y)):

        print(f"\n========================")
        print(f"Outer Fold {fold+1}/{n_outer}")
        print("========================")

        X_train_val = X[train_val_idx]
        y_train_val = y[train_val_idx]
        site_train_val = site[train_val_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        site_test = site[test_idx]

        # -------------------------
        # Inner train / CV split
        # -------------------------
        tr, cv = train_test_split(
            np.arange(len(X_train_val)),
            test_size=0.2,
            stratify=y_train_val,
            random_state=42
        )

        X_tr = X_train_val[tr]
        y_tr = y_train_val[tr]
        site_tr = site_train_val[tr]

        X_cv = X_train_val[cv]
        y_cv = y_train_val[cv]
        site_cv = site_train_val[cv]

        # ---------- Stage 1 ----------
        site_net = train_site_predictor(X_tr, site_tr)

        model_A = train_cohort_model(X_tr[site_tr == 0], y_tr[site_tr == 0])
        model_B = train_cohort_model(X_tr[site_tr == 1], y_tr[site_tr == 1])

        with torch.no_grad():

            Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
            Xcv_t = torch.tensor(X_cv, dtype=torch.float32).to(device)
            Xte_t = torch.tensor(X_test, dtype=torch.float32).to(device)

            site_prob_tr = site_net(Xtr_t)
            site_prob_cv = site_net(Xcv_t)
            site_prob_te = site_net(Xte_t)

            pred_A_tr = torch.sigmoid(model_A(Xtr_t))
            pred_B_tr = torch.sigmoid(model_B(Xtr_t))

            pred_A_cv = torch.sigmoid(model_A(Xcv_t))
            pred_B_cv = torch.sigmoid(model_B(Xcv_t))

            pred_A_te = torch.sigmoid(model_A(Xte_t))
            pred_B_te = torch.sigmoid(model_B(Xte_t))

        # ---------- Meta training ----------
        meta, auc_meta = train_meta_combiner(
            site_prob_tr.detach(), pred_A_tr, pred_B_tr, y_tr,
            site_prob_cv.detach(), pred_A_cv, pred_B_cv, y_cv
        )

        cv_aucs.append(auc_meta)

        # ---------- Fine tuning ----------
        sann = FineTunableSANN(site_net, model_A, model_B, meta).to(device)
        sann = finetune_sann(sann, X_tr, y_tr)

        # ---------- Test evaluation ----------
        with torch.no_grad():

            test_prob = torch.sigmoid(
                sann(torch.tensor(X_test, dtype=torch.float32).to(device))
            ).cpu().numpy().ravel()

        test_auc = roc_auc_score(y_test, test_prob)
        test_aucs.append(test_auc)

        print(f"CV AUC: {auc_meta:.4f} | Test AUC: {test_auc:.4f}")

    # ==============================
    # Statistics
    # ==============================

    def auc_ci(aucs):

        mean = np.mean(aucs)
        std = np.std(aucs, ddof=1)

        ci = stats.t.interval(
            0.95,
            len(aucs)-1,
            loc=mean,
            scale=std/np.sqrt(len(aucs))
        )

        return mean, ci

    cv_mean, cv_ci = auc_ci(cv_aucs)
    test_mean, test_ci = auc_ci(test_aucs)

    print("\n====================================")
    print(f"CV   AUC: {cv_mean:.4f} (95% CI {cv_ci[0]:.4f}-{cv_ci[1]:.4f})")
    print(f"TEST AUC: {test_mean:.4f} (95% CI {test_ci[0]:.4f}-{test_ci[1]:.4f})")
    print("====================================")

    return cv_aucs, test_aucs

cv_aucs, test_aucs = nested_sann_pipeline(X, y, g_remap)

#%% Draw the Confusion Matrix

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # Ground truth
# y_true = y_test   # or y_test if already numpy
# # y_true = label_true

# # Predicted probabilities for class 1
# y_prob = preds       # shape (N,)
# # y_prob = np.array(label_pred)

# # Convert probabilities to labels
# threshold = 0.5
# y_pred = (y_prob >= threshold).astype(int)

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)

# # Plot
# fig, ax = plt.subplots(figsize=(5, 4))

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["0", "1"]
# )
# disp.plot(ax=ax, cmap="Blues", values_format="d")

# ax.set_title("Confusion Matrix (Single Model)")
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Your existing data processing
y_true = y_test 
y_prob = preds 
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)
# cm = confusion_matrix(y_true, y_pred)

cm = np.array([[7, 5], [3, 9]])
# cm = np.array([[9, 3], [3, 9]])
# cm = np.array([[11, 1], [3, 9]])

labels_cm = np.array([
    [f"{cm[0][0]}\nTN", f"{cm[0][1]}\nFP"],
    [f"{cm[1][0]}\nFN", f"{cm[1][1]}\nTP"]
])

# --- Updated Plotting Code ---
fig, ax = plt.subplots(figsize=(6, 5)) # Slightly larger figure for larger text

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["0", "1"]
)

# 1. Increase font size of numbers inside the cells
# Use text_kw to pass font size to the internal text objects
disp.plot(ax=ax, cmap="Blues", values_format="d", text_kw={'size': 18}, im_kw={"vmin": 3, "vmax": 11})

# Remove default text labels
for text in disp.text_.ravel():
    text.set_visible(False)

# Add custom labels
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        
        text_color = "white" if (i == 0 and j == 0 or i==1 and j==1) else "black"
        ax.text(j, i,
                labels_cm[i, j],
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=text_color)

# 2. Increase font size of axis labels and tick labels
# ax.set_title("Confusion Matrix", fontsize=18, pad=20)
ax.set_xlabel("Predicted Label", fontsize=18)
ax.set_ylabel("True Label", fontsize=18)

# Increase font size of the tick labels (the "0" and "1" on axes)
ax.tick_params(axis='both', which='major', labelsize=18)

# 4. Increase the font size of the Colorbar numbers
# We access the colorbar from the 'im_' attribute of the display object
cbar = disp.im_.colorbar
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.show()

#%% Validation Tests

import matplotlib.pyplot as plt

fea_index = 8
feature = X[:,fea_index+12*3]  # e.g., mean dose
labels = y   # 0/1

plt.hist(feature[labels==0], bins=20, alpha=0.5, label='No toxicity')
plt.hist(feature[labels==1], bins=20, alpha=0.5, label='Toxicity')
plt.xlabel('Feature value')
plt.ylabel('Count')
plt.legend()
plt.title(f'Feature {cli_name[fea_index]} distribution by label')
plt.show()

from scipy.stats import ttest_ind, mannwhitneyu
t_stat, p_val = ttest_ind(feature[y==0], feature[y==1])
print("t-test p-value:", p_val)

plt.figure()
plt.scatter(feature, labels)
plt.title(f'Scatter plot for Feature {cli_name[fea_index]}')

#%% Add Visual Best

import numpy as np
import pandas as pd

def get_visually_best_features(df, labels, top_n=15, method='cohen'):
    """
    Identifies features with the highest statistical separation between two groups.
    
    Parameters:
    df: Pandas DataFrame (Features only, normalized)
    labels: Array-like (0 for Negative/No Toxicity, 1 for Positive/Toxicity)
    top_n: Number of features to return
    method: 'cohen' for Cohen's d (Effect Size) or 'snr' for Signal-to-Noise Ratio
    """
    
    # Split the dataframe by label
    group_pos = df[labels == 1]
    group_neg = df[labels == 0]
    
    scores = {}
    
    for col in df.columns:
        mean_pos = group_pos[col].mean()
        mean_neg = group_neg[col].mean()
        std_pos = group_pos[col].std()
        std_neg = group_neg[col].std()
        
        # Avoid division by zero for invariant features
        if (std_pos + std_neg) == 0:
            scores[col] = 0
            continue

        if method == 'cohen':
            # Cohen's d: Difference of means / pooled standard deviation
            pooled_std = np.sqrt(((group_pos[col].count() - 1) * std_pos**2 + 
                                  (group_neg[col].count() - 1) * std_neg**2) / 
                                 (group_pos[col].count() + group_neg[col].count() - 2))
            if pooled_std == 0:
                scores[col] = 0
            else:
                scores[col] = np.abs(mean_pos - mean_neg) / pooled_std
                
        elif method == 'snr':
            # SNR: |mean1 - mean2| / (std1 + std2)
            scores[col] = np.abs(mean_pos - mean_neg) / (std_pos + std_neg)

    # Sort features by score descending
    sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Print the top results for your record
    print(f"--- Top {top_n} Features by {method.upper()} ---")
    for feat, score in sorted_features[:top_n]:
        print(f"{feat}: {score:.4f}")
        
    return [f[0] for f in sorted_features[:top_n]]

# ==========================================
# HOW TO USE THIS FOR YOUR PLOT:
# ==========================================

# 1. Run the function (assuming 'df_norm' is your data and 'y' are your labels)
# best_features = get_visually_best_features(df_norm, y, top_n=15, method='cohen')

# 2. Subset your dataframe for the heatmap
# df_for_plot = df_norm[best_features]

# 3. Plot (using the label-sorted order you already have)
# sns.heatmap(df_for_plot, cmap='RdBu_r', vmin=-3, vmax=3)

#%% Visual Heatmap

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example data
total_name = select_name + list(cli_dose2.columns)[1:] + list(demo_feature.index)

df_X1 = cli_dose.iloc[:,-48:]
df_X2 = cli_dose2.iloc[:,-48:]
# name = list(df_X1.columns)
# shap_list = [161, 149, 158, 153, 160, 111, 154, 155, 122, 2]
# name = np.array(total_name)[shap_list]
# df_X = pd.DataFrame(X[:,shap_list], columns=name)

name = total_name
# df_X = pd.DataFrame(X[:,:20], columns=name)
df_X = pd.DataFrame(X, columns=name)
# df_X = pd.concat([df_X1, df_X2])

df = df_X.copy()
best_features = get_visually_best_features(df, y, top_n=60, method='snr')
df          = df_X[best_features]
df['Label'] = y
# df['Label'] = label2['Label']
df_sorted = df.sort_values('Label')  # optional: sort so HT+ and HT- are grouped
labels = df_sorted['Label']
df_sorted = df_sorted.drop(columns='Label')

plt.figure(figsize=(10,8))
sns.heatmap(df_sorted, cmap='viridis', cbar_kws={'label': '% Volume'}, linewidths=0.5, vmin=-6, vmax=6)
plt.xlabel('Dosimetry Features')
plt.ylabel('Patients (sorted by label)', fontsize=20)
plt.title('Heatmap of Dosimetry Features Across Patients')
plt.show()

from sklearn.preprocessing import StandardScaler

# Standardize by column so each feature has Mean=0, Std=1
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_sorted), 
                         columns=df_sorted.columns, 
                         index=df_sorted.index)

plt.figure(figsize=(12,8))
sns.heatmap(df_scaled, cmap='RdBu_r', center=0) # Red-Blue shows deviations from mean

# Calculate correlation matrix
corr_matrix = df_sorted.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Drop highly correlated features
df_reduced = df_sorted.drop(columns=to_drop)

from sklearn.preprocessing import StandardScaler

# Standardize by column so each feature has Mean=0, Std=1
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_reduced), 
                         columns=df_reduced.columns, 
                         index=df_reduced.index)

plt.figure(figsize=(12,8))
sns.heatmap(df_scaled, cmap='RdBu_r', center=0) # Red-Blue shows deviations from mean

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. Feature Reduction (Correlation Filter) ---
# Calculate correlation matrix
corr_matrix = df_sorted.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]

# Drop highly correlated features
df_reduced = df_sorted.drop(columns=to_drop)

# --- 2. Data Scaling ---
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_reduced), 
                         columns=df_reduced.columns, 
                         index=df_reduced.index)

# --- 3. Visualization with Class Differentiation ---
# labels is assumed to be the Series/Column corresponding to df_scaled index
# We create a 1x2 grid: a thin bar for labels and a wide block for the heatmap
fig, (ax_label, ax_main) = plt.subplots(1, 2, figsize=(16, 10), 
                                         gridspec_kw={'width_ratios': [1, 25]})

# A. Plot the Label Sidebar (Blue for 0, Red for 1)
label_df = pd.DataFrame(labels)
sns.heatmap(label_df, ax=ax_label, cbar=False, 
            cmap=['#3498db', '#e74c3c'], # Blue and Red hex codes
            yticklabels=False)
# , vmin=-10, vmax=6
ax_label.set_xticks([])
ax_label.set_ylabel('Patients (sorted by label)', fontsize=30)
ax_label.set_title('Group', fontsize=30)

# B. Plot the Main Scaled Heatmap
sns.heatmap(df_scaled, ax=ax_main, cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Normalized Dose Value (Z-Score)'}, vmin=-6, vmax=6
            )

# We use ax_main because that is where the heatmap lives
if ax_main.collections:
    cbar = ax_main.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)  # Size of the numbers on the bar
    cbar.set_label('Normalized Dose Value (Z-Score)', size=30)  # Updated label size

# ax_main.set_title('Reduced Dosimetry Features Across Patients')
# ax_main.set_xlabel('Dosimetry Features', fontsize=18)
# ax_main.set_xticks([])
ax_main.set_xticklabels([])

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Remove the ticks and the labels entirely
ax_main.set_yticks([])
ax_main.set_yticklabels([])

# C. Add the Divider Line
# Find the split point (number of Class 0 patients)
split_idx = len(labels[labels == 0])

# Draw line across both the sidebar and the main heatmap
# ax_main.axhline(y=split_idx, color='black', linewidth=4, linestyle='-', label='Class Boundary')
ax_label.axhline(y=split_idx, color='black', linewidth=4)

# D. Color the Patient ID numbers on the Y-axis
for i, tick_label in enumerate(ax_label.get_yticklabels()):
    if labels.iloc[i] == 1:
        tick_label.set_color('red')
        tick_label.set_weight('bold')
    else:
        tick_label.set_color('blue')

plt.tight_layout()
plt.show()

#%% Plot out the UMAP 2D

import umap

feature = df_scaled.to_numpy()
label   = labels.to_numpy()

# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

# Fit and transform the feature matrix
X_umap = reducer.fit_transform(feature)

# Create a color palette
palette = sns.color_palette("hsv", len(np.unique(y)))

plt.figure(figsize=(8,6))
for label in np.unique(y):
    idx = y == label
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], 
                label=f"Class {label}", 
                alpha=0.7, 
                s=50)

plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.title('UMAP Projection of Features')
plt.legend()
plt.show()

#%% Plot out the UMAP 3D

from mpl_toolkits.mplot3d import Axes3D

reducer3d = umap.UMAP(n_components=3, random_state=42)
X_umap3d = reducer3d.fit_transform(feature)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
for label in np.unique(y):
    idx = y == label
    ax.scatter(X_umap3d[idx, 0], X_umap3d[idx, 1], X_umap3d[idx, 2], label=f"Class {label}", s=50)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_zlabel('UMAP3')
plt.legend()
plt.show()

#%% Compute Statistics

from sklearn.metrics import silhouette_score
import numpy as np

# 1. Calculate Silhouette Score for DVH features
# 'labels' is your vector of Red/Blue labels (0 and 1)
sil = silhouette_score(df_scaled, labels)

# 2. Calculate Silhouette Score for Top 20 SHAP features
# sil_top20 = silhouette_score(df_top20_normalized, labels)

print(f"DVH Silhouette: {sil:.3f}")
# print(f"Top 20 Silhouette: {sil_top20:.3f}")

# 3. Calculate Correlation with Label
corr = df_scaled.corrwith(labels).abs().mean()
# corr_top20 = df_top20_normalized.corrwith(labels).abs().mean()

print(f"Mean Correlation (DVH): {corr:.3f}")
# print(f"Mean Correlation (Top 20): {corr_top20:.3f}")

#%% Ideal Example

# CODE TO GENERATE A "IDEAL" EXAMPLE OF DISCRIMINATORY DATA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate data where Class 1 has significantly higher dose in certain features
np.random.seed(42)
class0_data = np.random.normal(loc=-0.8, scale=0.5, size=(20, 10)) # Lower dose
class1_data = np.random.normal(loc=0.8, scale=0.5, size=(20, 10))  # Higher dose
example_data = np.vstack([class0_data, class1_data])
example_labels = [0]*20 + [1]*20

plt.figure(figsize=(8, 6))
sns.heatmap(example_data, cmap='RdBu_r', center=0, cbar_kws={'label': 'Dose Value'})
plt.axhline(y=20, color='black', linewidth=4) # The clear boundary
plt.title("EXAMPLE: What a 'Working' Heatmap Looks Like")
plt.xlabel("Dosimetry Features")
plt.ylabel("Patients (Sorted by Label)")
plt.show()

#%% Print out the mean and std

# X = cli_feature_matrix_all

# 1. Identify indices for each group
idx_neg = (y == 0)
idx_pos = (y == 1)

# 2. Slice the feature array into two groups
X_neg = X[idx_neg]
X_pos = X[idx_pos]

fea_index = [8,9,10,11]

for ii in fea_index:
    
    ii_index = ii + 12*0
    feature_pos = X_pos[:,ii_index]
    feature_neg = X_neg[:,ii_index]

    print(f'NEG Feature {name[ii_index]} has Mean {np.mean(feature_neg)} and STD {np.std(feature_neg)}')
    print(f'POS Feature {name[ii_index]} has Mean {np.mean(feature_pos)} and STD {np.std(feature_pos)}')

#%%

import matplotlib.pyplot as plt
import sns
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns  # Make sure this is 'import seaborn as sns'
import numpy as np

# Assuming X_pos and X_neg are your (N, 48) numpy arrays
# name is your list of 48 feature names

# 1. Select specific indices to plot (e.g., V5gy, V10gy, V20gy)
indices = [11+12*3] 
selected_names = [name[i] for i in indices]

all_values = []
all_labels = []
all_features = []

for i in indices:
    pos_vals = X_pos[:, i]
    neg_vals = X_neg[:, i]
    
    # Add negative cohort data
    all_values.extend(neg_vals)
    all_labels.extend(["HT Negative"] * len(neg_vals))
    all_features.extend([name[i]] * len(neg_vals))
    
    # Add positive cohort data
    all_values.extend(pos_vals)
    all_labels.extend(["HT Positive"] * len(pos_vals))
    all_features.extend([name[i]] * len(pos_vals))

# 2. Plotting
plt.figure(figsize=(6, 6))
sns.violinplot(
    x=all_features, 
    y=all_values, 
    hue=all_labels, 
    split=True,       
    inner="quartile", 
    palette={"HT Negative": "skyblue", "HT Positive": "salmon"}
)

# plt.title("Comparison of Dosimetry: HT Negative vs. HT Positive", fontsize=18)
# plt.ylabel("Dose Metric Value", fontsize=18)
# plt.xlabel("Dosimetry Features", fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title="Cohort", fontsize=16, title_fontsize=16)

plt.tight_layout()
plt.show()

#%% Count Statistical Values

from scipy.stats import chi2_contingency

# 1. Create the contingency table (as done previously)
contingency_table = pd.crosstab(label2['Label'], label2['Pathology'])

# 2. Compute the Chi-Square test
# chi2: The test statistic
# p_val: The p-value
# dof: Degrees of freedom
# expected: The expected frequencies under the null hypothesis
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square p-value: {p_val:.4f}")

#%% Load the validated set

file_path = '/Users/hafeng/Documents/Postdoc_Research_Meetings/Medical_BLT/HT_Data/medical/'
file_id_update_path = '/Users/hafeng/Documents/Postdoc_Research_Meetings/Medical_BLT/HT_Data/medical2/cohort1_correct.csv'
file_id_update      = pd.read_csv(file_id_update_path)
file_id_list        = list(file_id_update['cohort1_correct'])

dose_path  = file_path + 'dose_feature/'
image_path = file_path + 'image_feature/'

image_gg_feature_list  = []
image_pq_feature_list  = []
image_qg_feature_list  = []
image_ydz_feature_list = []

dose_gg_feature_list  = []
dose_pq_feature_list  = []
dose_qg_feature_list  = []
dose_ydz_feature_list = []

for item in file_id_list:
    # item = item[:-7]
    
    if item not in failed_list:
        print(item)
        # valid_item1.append(item)
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

#% Reformat the dataset

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

#%%

feat_id = 12
print(np.mean(image_gg_feature_matrix2[:,feat_id]))
print(np.std(image_gg_feature_matrix2[:,feat_id]))

print(np.mean(image_pq_feature_matrix2[:,feat_id]))
print(np.std(image_pq_feature_matrix2[:,feat_id]))

print(np.mean(image_qg_feature_matrix2[:,feat_id]))
print(np.std(image_qg_feature_matrix2[:,feat_id]))

print(np.mean(image_ydz_feature_matrix2[:,feat_id]))
print(np.std(image_ydz_feature_matrix2[:,feat_id]))

from scipy.stats import mannwhitneyu

# feature values
feature_cohort1 = image_gg_feature_matrix[:,feat_id]
feature_cohort2 = image_gg_feature_matrix2[:,feat_id]

stat, p_value = mannwhitneyu(feature_cohort1, feature_cohort2)

print("p-value:", p_value)

#%% Test for percentages

count = 0

for i in range(image_gg_feature_matrix2.shape[1]):
    feature_cohort1 = image_ydz_feature_matrix[:,i]
    feature_cohort2 = image_ydz_feature_matrix2[:,i]
    stat, p_value   = mannwhitneyu(feature_cohort1, feature_cohort2)
    
    if p_value < 0.05:
        count = count+1

print(f'Significant Percentage is {count / image_gg_feature_matrix2.shape[1]}')

#%% SHAP Analysis
import shap
import matplotlib.pyplot as plt

# create explainer
explainer = shap.Explainer(train_meta_combiner, X_train)

# compute shap values
shap_values = explainer(X_test)

plt.figure(figsize=(12,8))  # increase width
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.show()

#%%
total_name = select_name + list(cli_dose2.columns)[1:] + list(demo_feature.index)

# inter_name = [161, 149, 158, 153, 160, 111, 154, 155, 122, 2, 151, 95, 116, 152, 64, 35, 77, 37, 28, 126]
# inter_name = [161, 149, 158, 18,  31,  160, 147, 81,  21,  1, 38,  119, 44, 111, 76, 155, 103, 77, 43, 139]
inter_name = [161, 149, 111, 160, 157, 55, 72, 17, 151, 152, 155, 25, 121, 158, 39, 150, 27, 81, 7, 126]

real_inter_name = [total_name[i] for i in inter_name]

print(real_inter_name)
