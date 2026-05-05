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
    dose_gg_feature_matrix2   = dose_gg_feature_matrix2[final_mask]
    dose_pq_feature_matrix2   = dose_pq_feature_matrix2[final_mask]
    dosee_qg_feature_matrix2  = dose_qg_feature_matrix2[final_mask]
    dose_ydz_feature_matrix2  = dose_ydz_feature_matrix2[final_mask]
    cli_feature_matrix2       = cli_feature_matrix2[final_mask]
    demo_feature_matrix2      = demo_feature_matrix2[final_mask]
    label_matrix2 = label_matrix2[final_mask]

    # Remove abnormal rows
    
    Second_Index = 45
    
    image_gg_feature_matrix2  = np.delete(image_gg_feature_matrix2,  Second_Index, axis=0)
    image_pq_feature_matrix2  = np.delete(image_pq_feature_matrix2,  Second_Index, axis=0)
    image_qg_feature_matrix2  = np.delete(image_qg_feature_matrix2,  Second_Index, axis=0)
    image_ydz_feature_matrix2 = np.delete(image_ydz_feature_matrix2, Second_Index, axis=0)
    
    dose_gg_feature_matrix2  = np.delete(dose_gg_feature_matrix2,  Second_Index, axis=0)
    dose_pq_feature_matrix2  = np.delete(dose_pq_feature_matrix2,  Second_Index, axis=0)
    dose_qg_feature_matrix2  = np.delete(dose_qg_feature_matrix2,  Second_Index, axis=0)
    dose_ydz_feature_matrix2 = np.delete(dose_ydz_feature_matrix2, Second_Index, axis=0)
    
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
# 
# temp_df = ce_shape_clean.add_suffix('_ce')
temp_df = image_gg_radiomic.add_suffix('_pelvis')
# temp_df = pd.merge(temp_df, image_pq_radiomic.add_suffix('_cavity'), left_index=True, right_index=True)
temp_df = pd.merge(temp_df, image_qg_radiomic.add_suffix('_iliac'),    left_index=True, right_index=True)
temp_df = pd.merge(temp_df, image_ydz_radiomic.add_suffix('_lum'),     left_index=True, right_index=True)
# temp_df = dose_gg_radiomic.add_suffix('_dose_pelvis')
# temp_df = pd.merge(temp_df, dose_gg_radiomic.add_suffix('_dose_pelvis'), left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, dose_pq_radiomic.add_suffix('_dose_cavity'), left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, dose_qg_radiomic.add_suffix('_dose_iliac'),  left_index=True, right_index=True)
# temp_df = pd.merge(temp_df, dose_ydz_radiomic.add_suffix('_dose_lum'),   left_index=True, right_index=True)

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

selector1   = SelectKBest(score_func=f_classif, k=100)  # e.g., top 100
# # selector1.fit(X,y)
X = selector1.fit_transform(X, y)

# # Get the actual indices from the original 7000 features
# selected_indices = np.where(selector1.get_support())[0]
# print("Selected indices from original feature space:", selected_indices)

# # X = Feature_Select1(X, X_validate, X_validate0, y, 100)

scaler = MinMaxScaler()
# # scaler = StandardScaler()
X = scaler.fit_transform(X)

#%%
# X = cli_feature_matrix
# add_index  = 'none'
add_index = 'add'
# demo_index = 'none'
# demo_index = 'add'
demo_index = 'pure'
# demo_index = 'all'

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
        X = np.concatenate((cli_feature_matrix, demo_feature_matrix),     axis=1)
    elif data_index == 'second':
        X = np.concatenate((cli_feature_matrix2, demo_feature_matrix2),    axis=1)
    elif data_index == 'all':
        X = np.concatenate((cli_feature_matrix_all, demo_feature_matrix_all), axis=1)

#%% Develop Label G

g1 = np.ones((image_gg_feature_matrix.shape[0]))
g2 = np.ones((image_gg_feature_matrix2.shape[0]))
g2 = -g2

# data_index = 'single'
# data_index = 'second'
# data_index = 'all'

if data_index == 'single':
    g = g1
elif data_index == 'second':
    g = g2
elif data_index == 'all':
    g = np.concatenate((g1,g2))

#%% Remove NaN
# X = cli_feature_matrix_all[:,[5,17,29,41]]
# X = cli_feature_matrix_all

mask = ~np.isnan(X).any(axis=1)

X = X[mask]
y = y[mask]
g = g[mask]

g_remap = (g == 1).astype(int)

# Plot out the features and double check
plt.figure()
plt.plot(X[:,32])
plt.title('Feature')

# Remove certain patients with wrong feature

# if 

# X  = np.delete(X, 108, axis=0)
# y  = np.delete(y, 108, axis=0)
# g  = np.delete(g, 108, axis=0)

#%% Split the data into two groups 

mask1 = (g == 1)
mask2 = (g == -1)
X_G1  = X[mask1] 
X_G2  = X[mask2]
y_G1  = y[mask1]
y_G2  = y[mask2]

#%% Develop the ICC frameworks

X_cohort1 = X_G1
X_cohort2 = X_G2
y_cohort1 = y_G1
y_cohort2 = y_G2

def calculate_icc(data1, data2):
    """
    Calculate ICC(2,1) between two datasets for each feature.
    Higher ICC indicates more stable/reproducible features across cohorts.
    """
    n1, n2 = data1.shape[0], data2.shape[0]
    k = 2  # number of cohorts
    
    icc_values = []
    
    for i in range(data1.shape[1]):
        feat1 = data1[:, i]
        feat2 = data2[:, i]
        
        # Combine the feature values
        all_values = np.concatenate([feat1, feat2])
        
        # Grand mean
        grand_mean = np.mean(all_values)
        
        # Between-group variance (cohort variance)
        mean1 = np.mean(feat1)
        mean2 = np.mean(feat2)
        ms_between = ((mean1 - grand_mean)**2 * n1 + (mean2 - grand_mean)**2 * n2) / (k - 1)
        
        # Within-group variance
        ss_within = np.sum((feat1 - mean1)**2) + np.sum((feat2 - mean2)**2)
        ms_within = ss_within / (n1 + n2 - k)
        
        # ICC(2,1) calculation
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        icc_values.append(icc)
    
    return np.array(icc_values)

# Generate example data (replace with your actual data)
np.random.seed(42)

# Calculate ICC for each feature
icc_scores = calculate_icc(X_cohort1, X_cohort2)

print(f"\nICC scores - Mean: {np.mean(icc_scores):.3f}, Std: {np.std(icc_scores):.3f}")
print(f"ICC range: [{np.min(icc_scores):.3f}, {np.max(icc_scores):.3f}]")

# Select stable features based on ICC threshold
icc_threshold = 0.5  # Adjust this threshold based on your needs
stable_features = icc_scores > icc_threshold

print(f"\nNumber of stable features (ICC > {icc_threshold}): {np.sum(stable_features)}")

# Filter features
X_cohort1_filtered = X_cohort1[:, stable_features]
X_cohort2_filtered = X_cohort2[:, stable_features]

# Combine cohorts
X_combined = np.vstack([X_cohort1_filtered, X_cohort2_filtered])
y_combined = np.concatenate([y_cohort1, y_cohort2])

print(f"Combined dataset shape: {X_combined.shape}")

#%%

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined
)

# Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Predict and calculate AUC
y_pred_proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'='*50}")
print(f"Final Prediction AUC: {auc:.4f}")
print(f"{'='*50}")

# Alternative: Use different ICC threshold
print("\n" + "="*50)
print("Testing different ICC thresholds:")
print("="*50)

#%% Train-CV-Test Split

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import stats

kfold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
best_auc = -1
best_model_state = None
best_fold = -1
preds_nn = []

# for fold, (train_full_idx, test_idx) in enumerate(kfold.split(X_combined, y_combined)):
for fold, (train_full_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nOuter Fold {fold + 1}")

    X_train_full = X[train_full_idx]
    y_train_full = y[train_full_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # -------- Inner Train / Validation Split --------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)

    X_test  = torch.tensor(X_test, dtype=torch.float32)

    model = BinaryRadiomicsNet(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_val_auc = -1
    best_epoch_state = None

    # -------- Training --------
    for epoch in range(5000):

        model.train()
        optimizer.zero_grad()

        outputs = model(X_train.to(device)).squeeze()
        loss = criterion(outputs, y_train.to(device))

        loss.backward()
        optimizer.step()

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze()
            val_preds = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val.numpy(), val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch_state = model.state_dict()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")

    print(f"Best Validation AUC: {best_val_auc:.4f}")

    # -------- Load Best Epoch --------
    model.load_state_dict(best_epoch_state)

    # -------- Test Evaluation --------
    model.eval()
    with torch.no_grad():

        test_logits = model(X_test.to(device)).squeeze()
        test_preds = torch.sigmoid(test_logits).cpu().numpy()

        auc = roc_auc_score(y_test, test_preds)
        auc_scores.append(auc)

        print(f"Test AUC for Fold {fold + 1}: {auc:.4f}")
        preds_nn.append(test_preds)

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()
            best_fold = fold + 1


print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")

# Reload best model
best_model = BinaryRadiomicsNet(input_dim=X.shape[1]).to(device)
best_model.load_state_dict(best_model_state)

mean_auc = np.mean(auc_scores)
std_auc  = np.std(auc_scores, ddof=1)

ci95 = stats.t.interval(
    0.95,
    len(auc_scores)-1,
    loc=mean_auc,
    scale=std_auc/np.sqrt(len(auc_scores))
)

print(f"\nAverage AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")


#%% Feature Final

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

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_combined, y_combined)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[test_idx], dtype=torch.float32)
    y_train, y_test = torch.tensor(y[train_idx], dtype=torch.float32), torch.tensor(y[test_idx], dtype=torch.float32)

    model = BinaryRadiomicsNet(input_dim=X_train.shape[1]).to(device)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Training
    model.train()
    for epoch in range(3000):  # Adjust epochs
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
            best_auc  = auc
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


#%% Feature Selection

# X_temp = X_combined
# y = y_combined

# scaler   = MinMaxScaler()
# X_scaled = scaler.fit_transform(X_temp)

# selector1   = SelectKBest(score_func=f_classif, k=100)  # e.g., top 100
# # selector1.fit(X,y)
# X = selector1.fit_transform(X_scaled, y)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Training
    model.train()
    for epoch in range(3000):  # Adjust epochs
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
            best_auc  = auc
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

#%% Train-CV-Test Split

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import stats

kfold  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
best_auc = -1
best_model_state = None
best_fold = -1
preds_nn = []

for fold, (train_full_idx, test_idx) in enumerate(kfold.split(X, y)):
    print(f"\nOuter Fold {fold + 1}")

    X_train_full = X[train_full_idx]
    y_train_full = y[train_full_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # -------- Inner Train / Validation Split --------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_val   = torch.tensor(X_val, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)

    X_test  = torch.tensor(X_test, dtype=torch.float32)

    model = BinaryRadiomicsNet(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_val_auc = -1
    best_epoch_state = None

    # -------- Training --------
    for epoch in range(5000):

        model.train()
        optimizer.zero_grad()

        outputs = model(X_train.to(device)).squeeze()
        loss = criterion(outputs, y_train.to(device))

        loss.backward()
        optimizer.step()

        # -------- Validation --------
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze()
            val_preds = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val.numpy(), val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch_state = model.state_dict()

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")

    print(f"Best Validation AUC: {best_val_auc:.4f}")

    # -------- Load Best Epoch --------
    model.load_state_dict(best_epoch_state)

    # -------- Test Evaluation --------
    model.eval()
    with torch.no_grad():

        test_logits = model(X_test.to(device)).squeeze()
        test_preds = torch.sigmoid(test_logits).cpu().numpy()

        auc = roc_auc_score(y_test, test_preds)
        auc_scores.append(auc)

        print(f"Test AUC for Fold {fold + 1}: {auc:.4f}")
        preds_nn.append(test_preds)

        if auc > best_auc:
            best_auc = auc
            best_model_state = model.state_dict()
            best_fold = fold + 1


print(f"\nBest AUC: {best_auc:.4f} from Fold {best_fold}")

# Reload best model
best_model = BinaryRadiomicsNet(input_dim=X.shape[1]).to(device)
best_model.load_state_dict(best_model_state)

mean_auc = np.mean(auc_scores)
std_auc  = np.std(auc_scores, ddof=1)

ci95 = stats.t.interval(
    0.95,
    len(auc_scores)-1,
    loc=mean_auc,
    scale=std_auc/np.sqrt(len(auc_scores))
)

print(f"\nAverage AUC: {mean_auc:.4f}")
print(f"95% CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")

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
    auc   = roc_auc_score(y_test, preds)
    auc_scores.append(auc)
    print(f"AUC for fold {fold + 1}: {auc:.4f}")
    preds_tpot.append(preds)
    
    # preds = tpot.predict(X_upenn)
    # auc = roc_auc_score(y_upenn, preds)
    # auc_ucsf.append(auc)
    # print(f"AUC for fold {fold + 1}: {auc:.4f}")

    if auc > best_auc:
        best_auc  = auc
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

