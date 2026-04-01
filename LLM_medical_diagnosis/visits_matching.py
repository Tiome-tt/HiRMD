import os
import pandas as pd
import json
import re
import string
from TimeAttention.config.config import datasets

# Clean and standardize the list items, removing unnecessary spaces, punctuation, and converting to lowercase.
def clean_data(data):
    cleaned = []
    for item in data:
        item = item.strip().lower()
        # Remove punctuation marks
        item = item.translate(str.maketrans('', '', string.punctuation))
        if item:
            cleaned.append(item)
    return cleaned

# Extract the main keywords from the DDP entry (the part separated by the last '|') and clean them up.
def extract_main_term(ddp_entry):
    parts = ddp_entry.split('|')
    if parts:
        main_term = parts[-1].strip().lower()
    else:
        main_term = ddp_entry.strip().lower()
    # Remove punctuation marks
    main_term = main_term.translate(str.maketrans('', '', string.punctuation))
    return main_term

# Read response.jsonl data and extract diseases, medications, and procedures for each PatientID
def load_response_data(file_path):
    response_dict = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'response' not in data:
                    print(f"Missing 'response' field in line: {line}")
                    continue  # Skip lines that don't have a response

                patient_id = data['PatientID']
                response_text = data['response']

                # Extract each part using regular expressions.
                disease_match = re.search(r'Disease:\s*(.*?)(?:\n|$)', response_text, re.IGNORECASE)
                medication_match = re.search(r'Medication:\s*(.*?)(?:\n|$)', response_text, re.IGNORECASE)
                procedure_match = re.search(r'Procedure:\s*(.*?)(?:\n|$)', response_text, re.IGNORECASE)

                diseases_raw = disease_match.group(1) if disease_match else ''
                medications_raw = medication_match.group(1) if medication_match else ''
                procedures_raw = procedure_match.group(1) if procedure_match else ''

                # Clean and convert to a lowercase list.
                response_dict[patient_id] = {
                    'diseases': clean_data(diseases_raw.split(';')) if diseases_raw else [],
                    'medications': clean_data(medications_raw.split(';')) if medications_raw else [],
                    'procedures': clean_data(procedures_raw.split(';')) if procedures_raw else []
                }
            except json.JSONDecodeError as e:
                print(f"JSON decoding error in line: {line}")
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error in line: {line}")
                print(f"Error: {e}")
    return response_dict

# Read the DDP_mimic-iv.csv data.
def load_ddp_mimic_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Define the matching function: check for the existence of an inclusion relationship.
def is_match(ddp_term, response_terms):
    for response_term in response_terms:
        if ddp_term in response_term or response_term in ddp_term:
            return True
    return False

# Filter out non-matching diseases, medications, and procedures based on PatientID.
def filter_patient_data(df, response_data):
    filtered_data = []
    for _, row in df.iterrows():
        patient_id = row['PatientID']
        # Initialize to raw data
        new_disease = row['Disease']
        new_medication = row['Medication']
        new_procedure = row['Procedure']

        if patient_id in response_data:
            # Clean and convert diseases, medications, and procedures to lowercase.
            diseases_ddp = clean_data(row['Disease'].split(';'))
            medications_ddp = clean_data(row['Medication'].split(';'))
            procedures_ddp = clean_data(row['Procedure'].split(';'))

            # Get the corresponding data in response.jsonl
            diseases_response = response_data[patient_id]['diseases']
            medications_response = response_data[patient_id]['medications']
            procedures_response = response_data[patient_id]['procedures']

            # Filtering diseases
            valid_diseases = [
                disease_ddp for disease_ddp in diseases_ddp
                if is_match(extract_main_term(disease_ddp), diseases_response)
            ]
            if valid_diseases:
                new_disease = '; '.join(valid_diseases)

            # Filtering medication
            valid_medications = [
                medication_ddp for medication_ddp in medications_ddp
                if is_match(extract_main_term(medication_ddp), medications_response)
            ]
            if valid_medications:
                new_medication = '; '.join(valid_medications)

            # Filtering procedure
            valid_procedures = [
                procedure_ddp for procedure_ddp in procedures_ddp
                if is_match(extract_main_term(procedure_ddp), procedures_response)
            ]
            if valid_procedures:
                new_procedure = '; '.join(valid_procedures)

        # Add the processed data to filtered_data.
        filtered_data.append({
            'PatientID': patient_id,
            'RecordTime': row['RecordTime'],
            'Disease': new_disease,
            'Medication': new_medication,
            'Procedure': new_procedure
        })

    # Convert all records (including unmatched ones) to a DataFrame.
    return pd.DataFrame(filtered_data)

# The output will be a new CSV file.
def save_filtered_data(df, output_file_path):
    df.to_csv(output_file_path, index=False)

def main():
    response_file_path = f'HiRMD/LLM_medical_diagnosis/processed/{datasets}/response.jsonl'
    ddp_file_path = f'HiRMD/datasets/{datasets}/processed/DDP_{datasets}.csv'
    output_file_path = f'processed/{datasets}/filtered_data.csv'

    # Loading data
    response_data = load_response_data(response_file_path)
    ddp_mimic_data = load_ddp_mimic_data(ddp_file_path)

    # Filter data.
    filtered_data = filter_patient_data(ddp_mimic_data, response_data)

    # Save the results.
    save_filtered_data(filtered_data, output_file_path)

    print(f"The filtered data has been saved to {output_file_path}")

if __name__ == "__main__":
    main()
