import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from TimeAttention.prompt.prompt_ChatGPT import SYSTEMPROMPT, USERPROMPT, CLASSIFICATION_CRITERIA
from TimeAttention.config.config import API_KEY, BASE_URL, MODEL, datasets
# GPT API configuration
OPENAI_API_KEY = API_KEY
GPT_BASE_URL = BASE_URL
GPT_MODEL = MODEL
OUTPUT_DIR = "outputs"


# Read patient data.
def load_patient_data(file_path):
    return pd.read_csv(file_path)


# Dynamically generate output file name
def generate_output_filename():
    return os.path.join(OUTPUT_DIR, f"LLM_Diagnosis_{datasets}.jsonl")


# Construct Prompt
def construct_prompts(patient_data):
    prompts = []
    grouped = patient_data.groupby("PatientID")

    for patient_id, group in grouped:
        # Sort by RecordTime, do not delete records where Disease is empty.
        visits = group.sort_values("RecordTime").to_dict(orient="records")

        # Construct the visit section for each patient.
        visits_str = ""
        for i, visit in enumerate(visits):
            visit_disease = visit.get("Disease", "NAN")
            visit_medication = visit.get("Medication", "NAN")
            visit_procedure = visit.get("Procedure", "NAN")

            visits_str += f"visit{i + 1}:\nDisease: {visit_disease}. Medication: {visit_medication}. Procedure: {visit_procedure}.\n"

        # Generate the final prompt
        filled_prompt = USERPROMPT.format(
            CLASSIFICATION_CRITERIA=CLASSIFICATION_CRITERIA,
            visits=visits_str.strip()
        )
        prompts.append({"PatientID": patient_id, "prompt": filled_prompt})

    return prompts


# Call GPT API (add retry mechanism)
def call_gpt_api(prompt_item, retries=3):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEMPROMPT},
            {"role": "user", "content": prompt_item["prompt"]},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }

    for attempt in range(retries):
        try:
            response = requests.post(GPT_BASE_URL, headers=headers, json=data, timeout=300)
            if response.status_code == 200:
                result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"PatientID": prompt_item["PatientID"], "response": result}
            else:
                error_msg = response.json().get("error", {}).get("message", f"HTTP {response.status_code}")
                print(f"Error for PatientID {prompt_item['PatientID']}: {error_msg}")
                return {"PatientID": prompt_item["PatientID"], "error": error_msg}
        except requests.exceptions.Timeout as e:
            print(f"Timeout for PatientID {prompt_item['PatientID']} on attempt {attempt + 1}: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for PatientID {prompt_item['PatientID']} on attempt {attempt + 1}: {e}")
        except Exception as e:
            print(f"Unexpected error for PatientID {prompt_item['PatientID']} on attempt {attempt + 1}: {e}")

        # If there are still retry attempts available, please continue to try.
        if attempt < retries - 1:
            print(f"Retrying for PatientID {prompt_item['PatientID']}...")

    # If it still fails after multiple retries, record the error.
    return {"PatientID": prompt_item["PatientID"], "error": f"Failed after {retries} retries"}


# 主程序
if __name__ == "__main__":
    input_file = f"HiRMD/LLM_medical_diagnosis/processed/{datasets}/filtered_data.csv"
    output_file = generate_output_filename()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear the contents of the file.
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    # Read data.
    patient_data = load_patient_data(input_file)

    # Construct Prompts
    prompts = construct_prompts(patient_data)

    # Concurrent calls to the GPT API and real-time writing to files.
    with ThreadPoolExecutor(max_workers=5) as executor, open(output_file, "a", encoding="utf-8") as output_f:
        futures = [executor.submit(call_gpt_api, prompt) for prompt in prompts]
        for future in tqdm(as_completed(futures), total=len(prompts), desc="Processing Prompts"):
            result = future.result()
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()  # Write to disk immediately.

    print(f"Results saved to {output_file}.")
