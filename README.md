# Code for the paper: *HiRMD: A System for Mortality Prediction via LLM-Based High-Risk Information Extraction and Diagnosis*

## Datasets

To protect patient privacy, we provide only a single sample instance from each dataset. These examples are included solely to help readers understand the data format and the experimental setup used in our work.

## Quick Start

### 🧩**Step 1. Prepare the MIMIC dataset**

Download the MIMIC dataset and import it into a PostgreSQL database.  
Run the provided SQL scripts, export the resulting `result_table` as a CSV file, and save it under an appropriate filename and path, such as:

```
HiRMD/datasets/mimic-iii/mimiciii_format.csv
```

### 🧹**Step 2. Run data preprocessing**

Execute the corresponding preprocessing script for the dataset:

```
HiRMD/datasets/mimic-iii/preprocess_mimic-iii.py
```

### 🩺**Step 3. Configure and run the LLM-based medical diagnosis module**

Modify the configuration file as needed.  
Then sequentially run:

```
HiRMD/LLM_medical_diagnosis/unique_process.py
HiRMD/LLM_medical_diagnosis/LLM_target_extraction.py
HiRMD/LLM_medical_diagnosis/visits_matching.py
HiRMD/LLM_medical_diagnosis/LLM_Diagnosis.py
```

### 🚀**Step 4. Train the model and obtain results**

Finally, run:

```
main.py
```

to reproduce the experimental results reported in the paper.

