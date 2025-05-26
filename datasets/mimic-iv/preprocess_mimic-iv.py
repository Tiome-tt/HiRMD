import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Data Path
Path(os.path.join('processed')).mkdir(parents=True, exist_ok=True)

# Define the header mapping.
column_mapping = {
    'subject_id': 'PatientID',
    'gender': 'Sex',
    'age': 'Age',
    'admittime_rank': 'RecordTime',
    'height': 'Height',
    'weight': 'Weight',
    'ph': 'PH',
    'hemoglobin': 'Hemoglobin',
    'temperature': 'Temperature',
    'glucose': 'Glucose',
    'wbc': 'White Blood Cell Count',
    'lymphocytes': 'Lymphocytes',
    'hematocrit': 'Hematocrit',
    'platelet': 'Platelet',
    'rbc': 'Red Blood Cell Count',
    'sbp': 'Systolic Blood Pressure',
    'dbp': 'Diastolic Blood Pressure',
    'mbp': 'Mean Blood Pressure',
    'resp_rate': 'Respiratory Rate',
    'heart_rate': 'Heart Rate',
    'surface_name': 'Disease',
    'drug_name': 'Medication',
    'procedures_description': 'Procedure',
    'hosp_los': 'LOS',
    'death_status': 'Outcome',
}

basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
demographic_features = ['Sex', 'Age']
labtest_features = [
    'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature', 'Glucose', 'White Blood Cell Count',
    'Lymphocytes', 'Hematocrit', 'Platelet', 'Red Blood Cell Count', 'Systolic Blood Pressure',
    'Diastolic Blood Pressure', 'Mean Blood Pressure', 'Respiratory Rate', 'Heart Rate'
]
normalize_features = ['Age'] + labtest_features + ['LOS']

# ICU score columns
icu_score_columns = [
    'apsiii', 'apsiii_prob', 'aps_hr_score', 'aps_mbp_score', 'aps_temp_score', 'aps_resp_rate_score',
    'aps_pao2_aado2_score', 'aps_hematocrit_score', 'aps_wbc_score', 'aps_creatinine_score',
    'aps_uo_score', 'aps_bun_score', 'aps_sodium_score', 'aps_albumin_score', 'aps_bilirubin_score',
    'aps_glucose_score', 'aps_acidbase_score', 'aps_gcs_score',
    'sapsii', 'sapsii_prob', 'saps_age_score', 'saps_hr_score', 'saps_sysbp_score', 'saps_temp_score',
    'saps_pao2fio2_score', 'saps_uo_score', 'saps_bun_score', 'saps_wbc_score',
    'saps_potassium_score', 'saps_sodium_score', 'saps_bicarbonate_score', 'saps_bilirubin_score',
    'saps_gcs_score', 'saps_comorbidity_score', 'saps_admissiontype_score'
]

# Read data
file_path = os.path.join("HiRMD/datasets/mimic-iv/mimiciv_format.csv")
df = pd.read_csv(file_path, on_bad_lines='skip')

print("Calculating 'admittime_rank' for each PatientID...")
df['admittime_rank'] = df.groupby('subject_id')['admittime'].rank(method='first', ascending=True)

# Rename Column
df.rename(columns=column_mapping, inplace=True)

# Convert the time field to datetime format.
df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])

# sort
df = df.sort_values(['PatientID', 'RecordTime'])

# Filter out groups where stay_id is entirely empty.
print("Filtering groups with at least one non-null 'stay_id'...")
df = df.groupby('PatientID').filter(lambda group: group['stay_id'].notna().any())

# Calculate the time interval between each discharge and readmission.
df['next_admittime'] = df.groupby('PatientID')['admittime'].shift(-1)
df['time_to_next_admission'] = (df['next_admittime'] - df['dischtime']).dt.days

# Mark for readmission in 30 days.
readmission_threshold = 30
df['Readmission'] = (df['time_to_next_admission'] <= readmission_threshold).astype(int)

df.fillna({'Readmission': 0}, inplace=True)

# Filter out patients with at least 10 records.
df = df.groupby('PatientID').filter(lambda x: len(x) >= 10)

# Select the first 48 records for each patient.
df = df.groupby('PatientID').head(48)

# Increase the code for downward and upward interpolation of missing values.
columns_to_process = df.columns.difference(['PatientID'])
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

# normalization
print("Normalizing features...")
scaler = MinMaxScaler()

df_normalized = df[normalize_features]
df[normalize_features] = scaler.fit_transform(df_normalized)

# ICU Score Calculation and Normalization

print("Calculating and normalizing ICU scores...")
exclude_cols = ['apsiii_prob', 'sapsii_prob']
cols_mean = [col for col in icu_score_columns if col not in exclude_cols]
icu_mean = df.groupby('PatientID')[cols_mean].mean()
icu_max = df.groupby('PatientID')[exclude_cols].max()
icu_score_avg = pd.concat([icu_mean, icu_max], axis=1).reset_index()
icu_score_avg[cols_mean] = scaler.fit_transform(icu_score_avg[cols_mean])


# Save ICU scoring data.
icu_output_file = os.path.join("processed", "icu_score_mimic-iv.csv")
icu_score_avg.to_csv(icu_output_file, index=False)
print(f"ICU score file saved to: {icu_output_file}")

df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})
df['Outcome'] = df.groupby('PatientID')['Outcome'].transform(lambda x: 1 if x.max() == 1 else 0)

# Feature extraction
DDP_columns = ['PatientID', 'RecordTime', 'Disease', 'Medication', 'Procedure']
df_DDP = df[DDP_columns]
label_columns = ['PatientID', 'Outcome', 'LOS', 'Readmission']
df_label = df[label_columns]

EHR_columns = ['PatientID', 'RecordTime', 'Sex', 'Age', 'Height', 'Weight', 'PH', 'Hemoglobin', 'Temperature',
               'Glucose', 'White Blood Cell Count', 'Lymphocytes', 'Hematocrit', 'Platelet', 'Red Blood Cell Count',
               'Heart Rate', 'Respiratory Rate', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Mean Blood Pressure']
# Add a missing value indicator column.
EHR_columns += [f'{col}_missing' for col in labtest_features]
df_EHR = df[EHR_columns]

# Save the results.
df_EHR.to_csv(os.path.join("processed", "EHR_mimic-iv.csv"), index=False)
df_DDP.to_csv(os.path.join("processed", "DDP_mimic-iv.csv"), index=False)
df_label.to_csv(os.path.join("processed", "label_mimic-iv.csv"), index=False)

print("Success for saving files.")
