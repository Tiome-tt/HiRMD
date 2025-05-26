## Code for paper <HiRMD: Improving In-Hospital Mortality Prediction via Prompting LLMs with High-Risk Medical Cues>

### Datasets
In order to protect the personal data of patients from being leaked, we provide only one data point for each dataset to facilitate the reader's understanding of our experimental setup.

### Steps

Step 1. Download the MIMIC dataset and store it in PostgreSQL. Run the corresponding SQL code, export the `result_table` as a CSV file, and save it with a new name to the designated path, such as `HiRMD/datasets/mimic-iii/mimiciii_format.csv`.

Step 2. Run the corresponding data preprocessing script, such as `HiRMD/datasets/mimic-iii/preprocess_mimic-iii.py`.

Step 3. Modify the configuration file. Then proceed with LLM medical diagnosis by sequentially running `HiRMD/LLM_medical_diagnosis/unique_process.py`, `HiRMD/LLM_medical_diagnosis/LLM_target_extraction.py`, `HiRMD/LLM_medical_diagnosis/visits_matching.py`, and `HiRMD/LLM_medical_diagnosis/LLM_Diagnosis.py`.

Step 4. Finally, run `main.py` to obtain the experimental results.

