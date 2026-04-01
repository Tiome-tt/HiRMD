import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Data path
Path(os.path.join('processed')).mkdir(parents=True, exist_ok=True)

# Define the header mapping (adjusted according to the MIMIC-III dataset)
column_mapping = {
    'patientid': 'PatientID',
    'stayid': 'ICUStayID',
    'sex': 'Sex',
    'age': 'Age',
    'admissionheight': 'AdmissionHeight',
    'admissionweight': 'AdmissionWeight',
    'dischargeweight': 'DischargeWeight',
    'outcome': 'Outcome',
    'icu_los_hours': 'ICULOSHours',
    'unitvisitnumber': 'UnitVisitNumber',
    'glucose_min': 'GlucoseMin',
    'glucose_max': 'GlucoseMax',
    'hematocrit_min': 'HematocritMin',
    'hematocrit_max': 'HematocritMax',
    'hemoglobin_min': 'HemoglobinMin',
    'hemoglobin_max': 'HemoglobinMax',
    'ph': 'PH',
    'temperature': 'Temperature',
    'platelet_min': 'PlateletMin',
    'platelet_max': 'PlateletMax',
    'wbc_min': 'WBCMin',
    'wbcmin': 'WBCMin',
    'heartrate': 'HeartRate',
    'respiratoryrate': 'RespiratoryRate',
    'disease_codes': 'DiseaseCodes',
    'diseases': 'Disease',
    'medications': 'Medication',
    'procedure_names': 'Procedure',
    'gcs': 'GCS',
    'gcs_motor': 'GCS_Motor',
    'gcs_verbal': 'GCS_Verbal',
    'gcs_eyes': 'GCS_Eyes',
    'gcs_unable': 'GCS_Unable',
    'gcs_intub': 'GCS_Intub',
    'fall_risk': 'Fall_Risk',
    'delirium_score': 'Delirium_Score',
    'sedation_score': 'Sedation_Score',
    'sedation_goal': 'Sedation_Goal',
    'pain_score': 'Pain_Score',
    'pain_goal': 'Pain_Goal'
}

# Definition of feature list
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome', 'ICULOSHours', 'Readmission']
demographic_features = ['Sex', 'Age']
labtest_features = [
    'AdmissionHeight', 'AdmissionWeight', 'DischargeWeight', 'UnitVisitNumber',
    'GlucoseMin', 'GlucoseMax', 'HematocritMin', 'HematocritMax', 'HemoglobinMin',
    'HemoglobinMax', 'PH', 'Temperature', 'PlateletMin', 'PlateletMax',
    'WBCMin', 'HeartRate', 'RespiratoryRate'
]
normalize_features = ['Age'] + labtest_features + ['ICULOSHours']

# ICU score column
icu_score_columns = [
    'GCS', 'GCS_Motor', 'GCS_Verbal', 'GCS_Eyes', 'GCS_Unable', 'GCS_Intub', 'Fall_Risk',
    'Delirium_Score', 'Sedation_Score', 'Sedation_Goal', 'Pain_Score', 'Pain_Goal'
]

# Read data.
file_path = os.path.join("HiRMD/datasets/eicu/eICU_format.csv")
df = pd.read_csv(file_path, on_bad_lines='skip')

# Rename column
df.rename(columns=column_mapping, inplace=True)

# Replace '>89' in Age with 89.
print("Processing age values...")
df['Age'] = df['Age'].replace('> 89', 89).astype(float)

# Sort by PatientID and UnitVisitNumber.
df.sort_values(by=['PatientID', 'UnitVisitNumber'], inplace=True)

# Group by PatientID and calculate RecordTime.
df['RecordTime'] = df.groupby('PatientID').cumcount() + 1

# Count the number of records per PatientID
record_counts = df.groupby('PatientID')['RecordTime'].count().reset_index()
record_counts.columns = ['PatientID', 'RecordCount']

# Merge the record count information into the original data.
df = df.merge(record_counts, on='PatientID', how='left')

# The number of filtered records is between 3 and 48 for patient records.
df_filtered = df.loc[(df['RecordCount'] >= 3) & (df['RecordCount'] <= 48)].copy()

# Exclude Patient Records with Procedure All Empty
print("Filtering out patients where all Procedure values are NaN...")
df_filtered = df_filtered.groupby('PatientID').filter(lambda group: group['Procedure'].notna().any())

# Increase downward and upward interpolation of missing values.
print("Filling missing values with forward and backward fill...")
columns_to_process = df_filtered.columns.difference(['PatientID', 'DiseaseCodes', 'Disease', 'Medication', 'Procedure'])
df_filtered.loc[:, columns_to_process] = (
    df_filtered.groupby('PatientID', group_keys=False)[columns_to_process]
    .apply(lambda group: group.ffill().bfill().infer_objects(copy=False))
)

# Increase missing value indicator
print("Adding missing value indicators...")
for col in labtest_features:
    df_filtered[f'{col}_missing'] = df_filtered[col].isnull().astype(int)

# Impute missing values to prevent normalization failure.
df_filtered.fillna(0, inplace=True)

# If any record in a certain patient group has an Outcome of 1, then set the Outcome of all records in that group to 1.
print("Updating Outcome for each PatientID group...")
df_filtered['Outcome'] = df_filtered.groupby('PatientID')['Outcome'].transform(lambda x: 1 if x.max() == 1 else 0)

# Normalization
print("Normalizing features...")
scaler = MinMaxScaler()
df_filtered[normalize_features] = scaler.fit_transform(df_filtered[normalize_features])

# ICU scoring calculation and standardization
print("Calculating and normalizing ICU scores...")
icu_score_avg = df_filtered.groupby('PatientID')[icu_score_columns].mean().reset_index()
icu_score_avg[icu_score_columns] = scaler.fit_transform(icu_score_avg[icu_score_columns])

# Save ICU scoring data.
icu_output_file = os.path.join("processed", "icu_score_eicu.csv")
icu_score_avg.to_csv(icu_output_file, index=False)
print(f"ICU score file saved to: {icu_output_file}")

# Feature extraction
print("Extracting features...")
DDP_columns = ['PatientID', 'RecordTime', 'Disease', 'Medication', 'Procedure']
df_DDP = df_filtered[DDP_columns]

label_columns = ['Outcome', 'ICULOSHours']
df_label = df_filtered[label_columns]

EHR_columns = ['PatientID', 'RecordTime', 'Sex', 'Age', 'AdmissionHeight', 'AdmissionWeight', 'DischargeWeight',
               'PH', 'HemoglobinMin', 'HemoglobinMax', 'Temperature', 'GlucoseMin', 'GlucoseMax', 'HematocritMin',
               'HematocritMax', 'PlateletMin', 'PlateletMax', 'WBCMin', 'HeartRate', 'RespiratoryRate']
EHR_columns += [f'{col}_missing' for col in labtest_features]
df_EHR = df_filtered[EHR_columns]

# Save the processed dataset to a CSV file.
print("Saving processed datasets...")
ehr_output_file = os.path.join("processed", "EHR_eicu.csv")
ddp_output_file = os.path.join("processed", "DDP_eicu.csv")
label_output_file = os.path.join("processed", "label_eicu.csv")

df_EHR.to_csv(ehr_output_file, index=False)
df_DDP.to_csv(ddp_output_file, index=False)
df_label.to_csv(label_output_file, index=False)

print(f"EHR data saved to: {ehr_output_file}")
print(f"DDP data saved to: {ddp_output_file}")
print(f"Label data saved to: {label_output_file}")

print("Success for saving files.")


# # mini
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
#
# # Data path
# Path(os.path.join('processed')).mkdir(parents=True, exist_ok=True)
#
# # Table header mapping of numerical definitions (adjusted according to the MIMIC-III dataset)
# column_mapping = {
#     'patientid': 'PatientID',
#     'stayid': 'ICUStayID',
#     'sex': 'Sex',
#     'age': 'Age',
#     'admissionheight': 'AdmissionHeight',
#     'admissionweight': 'AdmissionWeight',
#     'dischargeweight': 'DischargeWeight',
#     'outcome': 'Outcome',
#     'icu_los_hours': 'ICULOSHours',
#     'unitvisitnumber': 'UnitVisitNumber',
#     'glucose_min': 'GlucoseMin',
#     'glucose_max': 'GlucoseMax',
#     'hematocrit_min': 'HematocritMin',
#     'hematocrit_max': 'HematocritMax',
#     'hemoglobin_min': 'HemoglobinMin',
#     'hemoglobin_max': 'HemoglobinMax',
#     'ph': 'PH',
#     'temperature': 'Temperature',
#     'platelet_min': 'PlateletMin',
#     'platelet_max': 'PlateletMax',
#     'wbc_min': 'WBCMin',
#     'wbcmin': 'WBCMin',
#     'heartrate': 'HeartRate',
#     'respiratoryrate': 'RespiratoryRate',
#     'disease_codes': 'DiseaseCodes',
#     'diseases': 'Disease',
#     'medications': 'Medication',
#     'procedure_names': 'Procedure',
#     'gcs': 'GCS',
#     'gcs_motor': 'GCS_Motor',
#     'gcs_verbal': 'GCS_Verbal',
#     'gcs_eyes': 'GCS_Eyes',
#     'gcs_unable': 'GCS_Unable',
#     'gcs_intub': 'GCS_Intub',
#     'fall_risk': 'Fall_Risk',
#     'delirium_score': 'Delirium_Score',
#     'sedation_score': 'Sedation_Score',
#     'sedation_goal': 'Sedation_Goal',
#     'pain_score': 'Pain_Score',
#     'pain_goal': 'Pain_Goal'
# }
#
# # Definition of feature list
# basic_records = ['PatientID', 'RecordTime']
# target_features = ['Outcome', 'ICULOSHours', 'Readmission']
# demographic_features = ['Sex', 'Age']
# labtest_features = [
#     'AdmissionHeight', 'AdmissionWeight', 'DischargeWeight', 'UnitVisitNumber',
#     'GlucoseMin', 'GlucoseMax', 'HematocritMin', 'HematocritMax', 'HemoglobinMin',
#     'HemoglobinMax', 'PH', 'Temperature', 'PlateletMin', 'PlateletMax',
#     'WBCMin', 'HeartRate', 'RespiratoryRate'
# ]
# normalize_features = ['Age'] + labtest_features + ['ICULOSHours']
#
# # ICU score column
# icu_score_columns = [
#     'GCS', 'GCS_Motor', 'GCS_Verbal', 'GCS_Eyes', 'GCS_Unable', 'GCS_Intub', 'Fall_Risk',
#     'Delirium_Score', 'Sedation_Score', 'Sedation_Goal', 'Pain_Score', 'Pain_Goal'
# ]
#
# # Read the data.
# file_path = os.path.join("HiRMD/datasets/eicu/eICU_format.csv")
# df = pd.read_csv(file_path, on_bad_lines='skip')
#
# # Rename column
# df.rename(columns=column_mapping, inplace=True)
#
# # Replace '>89' in Age with 89.
# print("Processing age values...")
# df['Age'] = df['Age'].replace('> 89', 89).astype(float)
#
# # Sort by PatientID and UnitVisitNumber.
# df.sort_values(by=['PatientID', 'UnitVisitNumber'], inplace=True)
#
# # Group by PatientID and calculate RecordTime.
# df['RecordTime'] = df.groupby('PatientID').cumcount() + 1
#
# # Count the number of records per PatientID
# record_counts = df.groupby('PatientID')['RecordTime'].count().reset_index()
# record_counts.columns = ['PatientID', 'RecordCount']
#
# # Merge the record count information into the original data.
# df = df.merge(record_counts, on='PatientID', how='left')
#
# # Patient records with a count of filtered records between 3 and 48.
# df_filtered = df.loc[(df['RecordCount'] >= 3) & (df['RecordCount'] <= 48)].copy()
#
# # Exclude Patient Records with Procedure All Empty
# print("Filtering out patients where all Procedure values are NaN...")
# df_filtered = df_filtered.groupby('PatientID').filter(lambda group: group['Procedure'].notna().any())
#
# # Increase downward and upward interpolation of missing values.
# print("Filling missing values with forward and backward fill...")
# columns_to_process = df_filtered.columns.difference(['PatientID', 'DiseaseCodes', 'Disease', 'Medication', 'Procedure'])
# df_filtered.loc[:, columns_to_process] = (
#     df_filtered.groupby('PatientID', group_keys=False)[columns_to_process]
#     .apply(lambda group: group.ffill().bfill().infer_objects(copy=False))
# )
#
# # Increase missing value indicators.
# print("Adding missing value indicators...")
# for col in labtest_features:
#     df_filtered[f'{col}_missing'] = df_filtered[col].isnull().astype(int)
#
# # Impute missing values to prevent normalization failure.
# df_filtered.fillna(0, inplace=True)
#
# # If any record's Outcome in a certain patient group is 1, then set the Outcome of all records in that group to 1.
# print("Updating Outcome for each PatientID group...")
# df_filtered['Outcome'] = df_filtered.groupby('PatientID')['Outcome'].transform(lambda x: 1 if x.max() == 1 else 0)
#
# # Normalization
# print("Normalizing features...")
# scaler = MinMaxScaler()
# df_filtered[normalize_features] = scaler.fit_transform(df_filtered[normalize_features])
#
# # ICU scoring calculation and standardization
# print("Calculating and normalizing ICU scores...")
# icu_score_avg = df_filtered.groupby('PatientID')[icu_score_columns].mean().reset_index()
# icu_score_avg[icu_score_columns] = scaler.fit_transform(icu_score_avg[icu_score_columns])
#
# # Save ICU scoring data.
# icu_output_file = os.path.join("processed", "icu_score_eicu.csv")
# icu_score_avg.to_csv(icu_output_file, index=False)
# print(f"ICU score file saved to: {icu_output_file}")
#
# # New: Ensure that the ratio of deceased to surviving patients is 0.23.
# print("Adjusting mortality rate...")
# df_filtered['Outcome'] = df_filtered['Outcome'].astype(int)
# death_rate = df_filtered['Outcome'].mean()
# if death_rate > 0.23:
#     print(f"Mortality rate is higher ({death_rate}), reducing the number of patients with Outcome=1...")
#     df_filtered = df_filtered[df_filtered['Outcome'] == 0]
# elif death_rate < 0.23:
#     print(f"Mortality rate is lower ({death_rate}), increasing the number of patients with Outcome=1...")
#     additional_deaths = int(0.23 * len(df_filtered) - df_filtered['Outcome'].sum())
#     df_deaths = df_filtered[df_filtered['Outcome'] == 0].sample(n=additional_deaths, replace=True, random_state=42)
#     df_filtered = pd.concat([df_filtered, df_deaths])
#
# # Control the final record count.
# total_records = len(df_filtered)
# if total_records > 8000:
#     print(f"Total records {total_records} are more than 8000, reducing...")
#     patient_record_counts = df_filtered.groupby('PatientID').size().reset_index(name='RecordCount')
#     patient_record_counts = patient_record_counts.sort_values(by='RecordCount', ascending=False)
#     selected_patients = patient_record_counts.head(2100)
#     df_filtered = df_filtered[df_filtered['PatientID'].isin(selected_patients['PatientID'])]
#
#     # If the number of records exceeds 8000, then remove the records of the patients with the highest number of records.
#     total_records = len(df_filtered)
#     if total_records > 8000:
#         patients_to_remove = patient_record_counts.tail(total_records - 8000)['PatientID']
#         df_filtered = df_filtered[~df_filtered['PatientID'].isin(patients_to_remove)]
#
# DDP_columns = ['PatientID', 'RecordTime', 'Disease', 'Medication', 'Procedure']
# df_DDP = df_filtered[DDP_columns]
#
# label_columns = ['Outcome', 'ICULOSHours']
# df_label = df_filtered[label_columns]
#
# # Save the final processed dataset.
# print(f"Saving processed datasets...")
# ehr_output_file = os.path.join("processed", "EHR_eicu.csv")
# ddp_output_file = os.path.join("processed", "DDP_eicu.csv")
# label_output_file = os.path.join("processed", "label_eicu.csv")
#
# df_filtered.to_csv(ehr_output_file, index=False)
# print(f"EHR data saved to: {ehr_output_file}")
# df_DDP.to_csv(ddp_output_file, index=False)
# print(f"EHR data saved to: {ddp_output_file}")
# df_label.to_csv(label_output_file, index=False)
# print(f"EHR data saved to: {label_output_file}")
#
# print(f"Success: Final dataset contains {len(df_filtered)} records.")
#
