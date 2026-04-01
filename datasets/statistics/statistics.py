import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from TimeAttention.config.config import datasets

# Read the data
ehr_data = pd.read_csv(f'../{datasets}/processed/EHR_{datasets}.csv')
label_data = pd.read_csv(f'../{datasets}/processed/label_{datasets}.csv')

# Merge EHR data and tag data
merged_data = pd.concat([ehr_data[['PatientID']], label_data[['Outcome']]], axis=1)

# Count the total number of records
total_records = merged_data.shape[0]

# PatientID deduplication and merging of death information (if the outcome of any record is 1, the patient is considered dead)
patient_outcome = merged_data.groupby('PatientID')['Outcome'].max()

# Count the number of patients (deduplicated PatientID)
patient_count = merged_data['PatientID'].nunique()

# Count the number of dead and surviving patients
outcome_counts = patient_outcome.value_counts()

# Calculate the ratio of death to survival
death_to_survival_ratio = None
if 1 in outcome_counts and 0 in outcome_counts:
    death_to_survival_ratio = outcome_counts[1] / outcome_counts[0]

# Output the result
print(f"The total number of records: {total_records}")
print(f"Total number of patients: {patient_count}")
print(f"Number of deaths: {outcome_counts.get(1, 0)}")
print(f"Number of surviving patients: {outcome_counts.get(0, 0)}")
print(
    f"The ratio of death to survival: {death_to_survival_ratio if death_to_survival_ratio is not None else 'There were no deaths or survivors in the data'}")

# Count the number of consultation records for each patient
patient_record_counts = ehr_data.groupby('PatientID')['RecordTime'].count()

# Count the average number of records of patients
average_records_per_patient = patient_record_counts.mean()

# Output the average number of records
print(f"Average number of records for patients: {average_records_per_patient:.2f}")

# The number of patients under each number of consultations was counted
record_count_distribution = patient_record_counts.value_counts().sort_index()

# Outputs the distribution of the number of consultation records
print("Distribution of the number of interview records：")
print(record_count_distribution)


# Optimize PlotsOptimize Plotting Functions
def plot_optimized_distribution(distribution, title, log_scale=False, group_interval=None):
    """
    Optimized the drawing function, supported logarithmic scale and group display.

    :p aram distribution: Data distribution (Series)
    :p aram title: The title of the chart
    :p aram log_scale: Whether to use logarithmic scale (default False)
    :p aram group_interval: Whether to group by interval (e.g., 5, 10, default None)
    """
    plt.figure(figsize=(10, 6))

    if group_interval:
        # Display in groups
        bins = np.arange(0, distribution.index.max() + group_interval, group_interval)
        grouped_distribution = distribution.groupby(pd.cut(distribution.index, bins=bins)).sum()
        grouped_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xticks(rotation=45,
                   labels=[f"{int(interval.left)}-{int(interval.right)}" for interval in grouped_distribution.index])
    else:
        distribution.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.xticks(rotation=0)

    plt.title(title)
    plt.xlabel('Number of Records')
    plt.ylabel('Number of Patients (Log Scale)' if log_scale else 'Number of Patients')

    if log_scale:
        plt.yscale('log')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Draw optimized charts

# Use a logarithmic scale to optimize the display
plot_optimized_distribution(record_count_distribution, title=f"Records Distribution on {datasets}", log_scale=True)
