import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Data Path
Path(os.path.join('processed')).mkdir(parents=True, exist_ok=True)

# Define header mapping.
column_mapping = {
    'patientid': 'PatientID',
    'hadmid': 'HospitalID',
    'icustayid': 'ICUStayID',
    'icuintime': 'ICUInTime',
    'icuouttime': 'ICUOutTime',
    'sex': 'Sex',
    'age': 'Age',
    'weight': 'Weight',
    'height': 'Height',
    'outcome': 'Outcome',
    'glucose': 'Glucose',
    'hematocrit': 'Hematocrit',
    'hemoglobin': 'Hemoglobin',
    'ph': 'PH',
    'temperature': 'Temperature',
    'plateletmin': 'PlateletMin',
    'plateletmax': 'PlateletMax',
    'wbcmin': 'WBCMin',
    'wbcmax': 'WBCMax',
    'heartratemean': 'HeartRateMean',
    'sbpmean': 'SBPMean',
    'dbpmean': 'DBPMean',
    'mbpmean': 'MBPMean',
    'respiratoryratemean': 'RespiratoryRateMean',
    'spo2mean': 'SpO2Mean',
    'disease_codes': 'DiseaseCodes',
    'diseases': 'Disease',
    'medications': 'Medication',
    'procedure_codes': 'ProcedureCodes',
    'procedure_names': 'Procedure',
    'apsiii': 'APSIII',
    'apsiii_prob': 'APSIIIProb',
    'apsiii_hr_score': 'APSIII_HR_Score',
    'apsiii_meanbp_score': 'APSIII_MeanBP_Score',
    'apsiii_temp_score': 'APSIII_Temperature_Score',
    'apsiii_resprate_score': 'APSIII_RespRate_Score',
    'apsiii_pao2_aado2_score': 'APSIII_PAO2_AADO2_Score',
    'apsiii_hematocrit_score': 'APSIII_Hematocrit_Score',
    'apsiii_wbc_score': 'APSIII_WBC_Score',
    'apsiii_creatinine_score': 'APSIII_Creatinine_Score',
    'apsiii_uo_score': 'APSIII_UO_Score',
    'apsiii_bun_score': 'APSIII_BUN_Score',
    'apsiii_sodium_score': 'APSIII_Sodium_Score',
    'apsiii_albumin_score': 'APSIII_Albumin_Score',
    'apsiii_bilirubin_score': 'APSIII_Bilirubin_Score',
    'apsiii_glucose_score': 'APSIII_Glucose_Score',
    'apsiii_acidbase_score': 'APSIII_AcidBase_Score',
    'apsiii_gcs_score': 'APSIII_GCS_Score',
    'sapsii': 'SAPSII',
    'sapsii_prob': 'SAPSIIProb',
    'sapsii_age_score': 'SAPSII_Age_Score',
    'sapsii_hr_score': 'SAPSII_HR_Score',
    'sapsii_sysbp_score': 'SAPSII_SysBP_Score',
    'sapsii_temp_score': 'SAPSII_Temperature_Score',
    'sapsii_pao2fio2_score': 'SAPSII_PAO2FIO2_Score',
    'sapsii_uo_score': 'SAPSII_UO_Score',
    'sapsii_bun_score': 'SAPSII_BUN_Score',
    'sapsii_wbc_score': 'SAPSII_WBC_Score',
    'sapsii_potassium_score': 'SAPSII_Potassium_Score',
    'sapsii_sodium_score': 'SAPSII_Sodium_Score',
    'sapsii_bicarbonate_score': 'SAPSII_Bicarbonate_Score',
    'sapsii_bilirubin_score': 'SAPSII_Bilirubin_Score',
    'sapsii_gcs_score': 'SAPSII_GCS_Score',
    'sapsii_comorbidity_score': 'SAPSII_Comorbidity_Score',
    'sapsii_admissiontype_score': 'SAPSII_AdmissionType_Score',
}

# Definition of the Feature List
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
demographic_features = ['Sex', 'Age']
labtest_features = [
    'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature', 'Glucose', 'Hematocrit',
    'PlateletMin', 'PlateletMax', 'WBCMin', 'WBCMax', 'HeartRateMean', 'SBPMean',
    'DBPMean', 'MBPMean', 'RespiratoryRateMean', 'SpO2Mean'
]
normalize_features = ['Age'] + labtest_features + ['LOS']

# ICU score columns
icu_score_columns = [
    'APSIII', 'APSIIIProb', 'APSIII_HR_Score', 'APSIII_MeanBP_Score', 'APSIII_Temperature_Score',
    'APSIII_RespRate_Score', 'APSIII_PAO2_AADO2_Score', 'APSIII_Hematocrit_Score',
    'APSIII_WBC_Score', 'APSIII_Creatinine_Score', 'APSIII_UO_Score', 'APSIII_BUN_Score',
    'APSIII_Sodium_Score', 'APSIII_Albumin_Score', 'APSIII_Bilirubin_Score',
    'APSIII_Glucose_Score', 'APSIII_AcidBase_Score', 'APSIII_GCS_Score',
    'SAPSII', 'SAPSIIProb', 'SAPSII_Age_Score', 'SAPSII_HR_Score', 'SAPSII_SysBP_Score',
    'SAPSII_Temperature_Score', 'SAPSII_PAO2FIO2_Score', 'SAPSII_UO_Score',
    'SAPSII_BUN_Score', 'SAPSII_WBC_Score', 'SAPSII_Potassium_Score',
    'SAPSII_Sodium_Score', 'SAPSII_Bicarbonate_Score', 'SAPSII_Bilirubin_Score',
    'SAPSII_GCS_Score', 'SAPSII_Comorbidity_Score', 'SAPSII_AdmissionType_Score'
]

# Read data
file_path = os.path.join("HiRMD/datasets/mimic-iii/mimiciii_format.csv")
df = pd.read_csv(file_path, on_bad_lines='skip')

df.rename(columns=column_mapping, inplace=True)

# Calculate Record Time
print("Calculating 'RecordTime' for each PatientID based on admittime...")
df['RecordTime'] = df.groupby('PatientID')['admittime'].rank(method='first', ascending=True).astype(int)

# Convert the time field to datetime format.
df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])

# Ensure that the data is sorted by PatientID and RecordTime.
df = df.sort_values(['PatientID', 'RecordTime'])

# Filter out groups where ICUStayID is entirely empty.
print("Filtering groups with at least one non-null 'ICUStayID'...")
df = df.groupby('PatientID').filter(lambda group: group['ICUStayID'].notna().any())

# Calculate the time interval between each discharge and readmission.
df['next_admittime'] = df.groupby('PatientID')['admittime'].shift(-1)
df['LOS'] = (df['dischtime'] - df['admittime']).dt.days

df['time_to_next_admission'] = (df['next_admittime'] - df['dischtime']).dt.days

# Mark for readmission in 30 days.
readmission_threshold = 30
df['Readmission'] = (df['time_to_next_admission'] <= readmission_threshold).astype(int)

df.fillna({'Readmission': 0}, inplace=True)

# Filter out patients with at least 3 records.
df = df.groupby('PatientID').filter(lambda x: len(x) >= 3)
# print(len(df))
# Select the first 48 records for each patient.
df = df.groupby('PatientID').head(48)

# Increase the code for downward and upward interpolation of missing values.
columns_to_process = df.columns.difference(['PatientID', 'DiseaseCodes', 'Disease', 'Medication', 'ProcedureCodes', 'Procedure'])
df[columns_to_process] = (
    df.groupby('PatientID', group_keys=False)[columns_to_process]
    .apply(lambda group: group.ffill().bfill().infer_objects(copy=False))
)
df.reset_index(drop=True, inplace=True)

# Increase the missing value indicator.
print("Adding missing value indicators...")
for col in labtest_features:
    df[f'{col}_missing'] = df[col].isnull().astype(int)

# Impute missing values to prevent normalization failure
df.fillna(0, inplace=True)

# Normalization
print("Normalizing features...")
scaler = MinMaxScaler()

df_normalized = df[normalize_features]
df[normalize_features] = scaler.fit_transform(df_normalized)

# ICU scoring calculation and standardization.
print("Calculating and normalizing ICU scores...")

exclude_cols = ['APSIIIProb', 'SAPSIIProb']
cols_mean = [col for col in icu_score_columns if col not in exclude_cols]
icu_mean = df.groupby('PatientID')[cols_mean].mean()
icu_max = df.groupby('PatientID')[exclude_cols].max()
icu_score_avg = pd.concat([icu_mean, icu_max], axis=1).reset_index()
icu_score_avg[cols_mean] = scaler.fit_transform(icu_score_avg[cols_mean])


# Save ICU scoring data.
icu_output_file = os.path.join("processed", "icu_score_mimic-iii.csv")
icu_score_avg.to_csv(icu_output_file, index=False)
print(f"ICU score file saved to: {icu_output_file}")

# Feature extraction
DDP_columns = ['PatientID', 'RecordTime', 'Disease', 'Medication', 'Procedure']
df_DDP = df[DDP_columns]
label_columns = ['PatientID', 'Outcome', 'LOS', 'Readmission']
df_label = df[label_columns]

EHR_columns = ['PatientID', 'RecordTime', 'Sex', 'Age', 'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature',
               'Glucose', 'Hematocrit', 'PlateletMin', 'PlateletMax', 'WBCMin', 'WBCMax', 'HeartRateMean',
               'SBPMean', 'DBPMean', 'MBPMean', 'RespiratoryRateMean', 'SpO2Mean']
EHR_columns += [f'{col}_missing' for col in labtest_features]
df_EHR = df[EHR_columns]

# Save the results.
df_EHR.to_csv(os.path.join("processed", "EHR_mimic-iii.csv"), index=False)
df_DDP.to_csv(os.path.join("processed", "DDP_mimic-iii.csv"), index=False)
df_label.to_csv(os.path.join("processed", "label_mimic-iii.csv"), index=False)

print("Success for saving files.")

# # mimic-iii mini
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # Data Path
# Path(os.path.join('processed')).mkdir(parents=True, exist_ok=True)
#
# # Define header mapping
# column_mapping = {
#     'patientid': 'PatientID',
#     'hadmid': 'HospitalID',
#     'icustayid': 'ICUStayID',
#     'icuintime': 'ICUInTime',
#     'icuouttime': 'ICUOutTime',
#     'sex': 'Sex',
#     'age': 'Age',
#     'weight': 'Weight',
#     'height': 'Height',
#     'outcome': 'Outcome',
#     'glucose': 'Glucose',
#     'hematocrit': 'Hematocrit',
#     'hemoglobin': 'Hemoglobin',
#     'ph': 'PH',
#     'temperature': 'Temperature',
#     'plateletmin': 'PlateletMin',
#     'plateletmax': 'PlateletMax',
#     'wbcmin': 'WBCMin',
#     'wbcmax': 'WBCMax',
#     'heartratemean': 'HeartRateMean',
#     'sbpmean': 'SBPMean',
#     'dbpmean': 'DBPMean',
#     'mbpmean': 'MBPMean',
#     'respiratoryratemean': 'RespiratoryRateMean',
#     'spo2mean': 'SpO2Mean',
#     'disease_codes': 'DiseaseCodes',
#     'diseases': 'Disease',
#     'medications': 'Medication',
#     'procedure_codes': 'ProcedureCodes',
#     'procedure_names': 'Procedure',
#     'apsiii': 'APSIII',
#     'apsiii_prob': 'APSIIIProb',
#     'apsiii_hr_score': 'APSIII_HR_Score',
#     'apsiii_meanbp_score': 'APSIII_MeanBP_Score',
#     'apsiii_temp_score': 'APSIII_Temperature_Score',
#     'apsiii_resprate_score': 'APSIII_RespRate_Score',
#     'apsiii_pao2_aado2_score': 'APSIII_PAO2_AADO2_Score',
#     'apsiii_hematocrit_score': 'APSIII_Hematocrit_Score',
#     'apsiii_wbc_score': 'APSIII_WBC_Score',
#     'apsiii_creatinine_score': 'APSIII_Creatinine_Score',
#     'apsiii_uo_score': 'APSIII_UO_Score',
#     'apsiii_bun_score': 'APSIII_BUN_Score',
#     'apsiii_sodium_score': 'APSIII_Sodium_Score',
#     'apsiii_albumin_score': 'APSIII_Albumin_Score',
#     'apsiii_bilirubin_score': 'APSIII_Bilirubin_Score',
#     'apsiii_glucose_score': 'APSIII_Glucose_Score',
#     'apsiii_acidbase_score': 'APSIII_AcidBase_Score',
#     'apsiii_gcs_score': 'APSIII_GCS_Score',
#     'sapsii': 'SAPSII',
#     'sapsii_prob': 'SAPSIIProb',
#     'sapsii_age_score': 'SAPSII_Age_Score',
#     'sapsii_hr_score': 'SAPSII_HR_Score',
#     'sapsii_sysbp_score': 'SAPSII_SysBP_Score',
#     'sapsii_temp_score': 'SAPSII_Temperature_Score',
#     'sapsii_pao2fio2_score': 'SAPSII_PAO2FIO2_Score',
#     'sapsii_uo_score': 'SAPSII_UO_Score',
#     'sapsii_bun_score': 'SAPSII_BUN_Score',
#     'sapsii_wbc_score': 'SAPSII_WBC_Score',
#     'sapsii_potassium_score': 'SAPSII_Potassium_Score',
#     'sapsii_sodium_score': 'SAPSII_Sodium_Score',
#     'sapsii_bicarbonate_score': 'SAPSII_Bicarbonate_Score',
#     'sapsii_bilirubin_score': 'SAPSII_Bilirubin_Score',
#     'sapsii_gcs_score': 'SAPSII_GCS_Score',
#     'sapsii_comorbidity_score': 'SAPSII_Comorbidity_Score',
#     'sapsii_admissiontype_score': 'SAPSII_AdmissionType_Score',
# }
#
# # Definition of Feature List
# basic_records = ['PatientID', 'RecordTime']
# target_features = ['Outcome', 'LOS', 'Readmission']
# demographic_features = ['Sex', 'Age']  # Sex is binary, Age is continuous
# labtest_features = [
#     'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature', 'Glucose', 'Hematocrit',
#     'PlateletMin', 'PlateletMax', 'WBCMin', 'WBCMax', 'HeartRateMean', 'SBPMean',
#     'DBPMean', 'MBPMean', 'RespiratoryRateMean', 'SpO2Mean'
# ]
# normalize_features = ['Age'] + labtest_features + ['LOS']
#
# # ICU scoring columns
# icu_score_columns = [
#     'APSIII', 'APSIIIProb', 'APSIII_HR_Score', 'APSIII_MeanBP_Score', 'APSIII_Temperature_Score',
#     'APSIII_RespRate_Score', 'APSIII_PAO2_AADO2_Score', 'APSIII_Hematocrit_Score',
#     'APSIII_WBC_Score', 'APSIII_Creatinine_Score', 'APSIII_UO_Score', 'APSIII_BUN_Score',
#     'APSIII_Sodium_Score', 'APSIII_Albumin_Score', 'APSIII_Bilirubin_Score',
#     'APSIII_Glucose_Score', 'APSIII_AcidBase_Score', 'APSIII_GCS_Score',
#     'SAPSII', 'SAPSIIProb', 'SAPSII_Age_Score', 'SAPSII_HR_Score', 'SAPSII_SysBP_Score',
#     'SAPSII_Temperature_Score', 'SAPSII_PAO2FIO2_Score', 'SAPSII_UO_Score',
#     'SAPSII_BUN_Score', 'SAPSII_WBC_Score', 'SAPSII_Potassium_Score',
#     'SAPSII_Sodium_Score', 'SAPSII_Bicarbonate_Score', 'SAPSII_Bilirubin_Score',
#     'SAPSII_GCS_Score', 'SAPSII_Comorbidity_Score', 'SAPSII_AdmissionType_Score'
# ]
#
# # Read datas
# file_path = os.path.join("HiRMD/datasets/mimic-iii/mimiciii_format.csv")
# df = pd.read_csv(file_path, on_bad_lines='skip')
#
# # Rename Column
# df.rename(columns=column_mapping, inplace=True)
#
# # Calculate Record Time
# print("Calculating 'RecordTime' for each PatientID based on admittime...")
# df['RecordTime'] = df.groupby('PatientID')['admittime'].rank(method='first', ascending=True).astype(int)
#
# # Convert the time field to datetime format.
# df['admittime'] = pd.to_datetime(df['admittime'])
# df['dischtime'] = pd.to_datetime(df['dischtime'])
#
# # Ensure that the data is sorted by PatientID and RecordTime.
# df = df.sort_values(['PatientID', 'RecordTime'])
#
# # Filter out groups where the ICUStayID is entirely empty.
# print("Filtering groups with at least one non-null 'ICUStayID'...")
# df = df.groupby('PatientID').filter(lambda group: group['ICUStayID'].notna().any())
#
# # Calculate the time interval between each discharge and readmission.
# df['next_admittime'] = df.groupby('PatientID')['admittime'].shift(-1)
# df['LOS'] = (df['dischtime'] - df['admittime']).dt.days
#
# df['time_to_next_admission'] = (df['next_admittime'] - df['dischtime']).dt.days
#
# # Re-admission marked for 30 days.
# readmission_threshold = 30
# df['Readmission'] = (df['time_to_next_admission'] <= readmission_threshold).astype(int)
#
# df.fillna({'Readmission': 0}, inplace=True)
#
# # Filter out patients with at least 3 records.
# df = df.groupby('PatientID').filter(lambda x: len(x) >= 3)
# print(len(df))
# # Select the first 48 records for each patient.
# df = df.groupby('PatientID').head(48)
#
# # Add code for downward and upward interpolation of missing values.
# columns_to_process = df.columns.difference(['PatientID', 'DiseaseCodes', 'Disease', 'Medication', 'ProcedureCodes', 'Procedure'])
# df[columns_to_process] = (
#     df.groupby('PatientID', group_keys=False)[columns_to_process]
#     .apply(lambda group: group.ffill().bfill().infer_objects(copy=False))
# )
# df.reset_index(drop=True, inplace=True)
#
# # Add a missing value indicator.
# print("Adding missing value indicators...")
# for col in labtest_features:
#     df[f'{col}_missing'] = df[col].isnull().astype(int)
#
# # Impute missing values to prevent normalization failure
# df.fillna(0, inplace=True)
#
# # Adjust the ratio of death to survival to approximately 0.23.
# print("Adjusting death/survival ratio to approximately 0.23...")
#
# patients = df.drop_duplicates(subset='PatientID')[['PatientID', 'Outcome']]
#
# dead_patients = patients[patients['Outcome'] == 1]
# survived_patients = patients[patients['Outcome'] == 0]
#
# N_s = len(survived_patients)
# desired_N_d = int(0.23 * N_s)
# N_d_current = len(dead_patients)
#
# print(f"Current number of living patients: {N_s}")
# print(f"Current number of deceased patients.: {N_d_current}")
# print(f"Expected number of deceased patients: {desired_N_d}")
#
# if N_d_current > desired_N_d:
#     sampled_dead_patients = dead_patients.sample(n=desired_N_d, random_state=42)
#     print(f"Extracted {desired_N_d} deceased patients to achieve the target ratio.")
# else:
#     sampled_dead_patients = dead_patients
#     print("The current number of deceased patients has met or is below the target ratio, and there is no need for removal.")
#
# # Merged surviving patients and the deceased patients after screening.
# filtered_patients = pd.concat([survived_patients, sampled_dead_patients])
#
# # Filter the main dataset, retaining only the selected patients.
# df = df[df['PatientID'].isin(filtered_patients['PatientID'])]
#
# # Calculate and display the new mortality and survival ratios.
# final_patients = df.drop_duplicates(subset='PatientID')[['PatientID', 'Outcome']]
# final_dead = final_patients[final_patients['Outcome'] == 1]
# final_survived = final_patients[final_patients['Outcome'] == 0]
# final_ratio = len(final_dead) / len(final_survived) if len(final_survived) > 0 else 0
# print(f"Adjusted mortality and survival ratio: {final_ratio:.2f}")
#
# # Normalization
# print("Normalizing features...")
# scaler = MinMaxScaler()
#
# df_normalized = df[normalize_features]
# df[normalize_features] = scaler.fit_transform(df_normalized)
#
# # ICU Score Calculation and Standardization
# print("Calculating and normalizing ICU scores...")
# exclude_cols = ['APSIIIProb', 'SAPSIIProb']
# cols_mean = [col for col in icu_score_columns if col not in exclude_cols]
# icu_mean = df.groupby('PatientID')[cols_mean].mean()
# icu_max = df.groupby('PatientID')[exclude_cols].max()
# icu_score_avg = pd.concat([icu_mean, icu_max], axis=1).reset_index()
# icu_score_avg[cols_mean] = scaler.fit_transform(icu_score_avg[cols_mean])
#
# # Save ICU scoring data.
# icu_output_file = os.path.join("processed", "icu_score_mimic-iii.csv")
# icu_score_avg.to_csv(icu_output_file, index=False)
# print(f"ICU score file saved to: {icu_output_file}")
#
# # Feature Extraction
# DDP_columns = ['PatientID', 'RecordTime', 'Disease', 'Medication', 'Procedure']
# df_DDP = df[DDP_columns]
#
# label_columns = ['Outcome', 'LOS', 'Readmission']
# df_label = df[label_columns]
#
# EHR_columns = ['PatientID', 'RecordTime', 'Sex', 'Age', 'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature',
#                'Glucose', 'Hematocrit', 'PlateletMin', 'PlateletMax', 'WBCMin', 'WBCMax', 'HeartRateMean',
#                'SBPMean', 'DBPMean', 'MBPMean', 'RespiratoryRateMean', 'SpO2Mean']
# # Add a column indicating missing values.
# EHR_columns += [f'{col}_missing' for col in labtest_features]
# df_EHR = df[EHR_columns]
#
# # Save the results.
# df_EHR.to_csv(os.path.join("processed", "EHR_mimic-iii.csv"), index=False)
# df_DDP.to_csv(os.path.join("processed", "DDP_mimic-iii.csv"), index=False)
# df_label.to_csv(os.path.join("processed", "label_mimic-iii.csv"), index=False)
#
# print("Success for saving files.")
