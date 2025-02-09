import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from TimeAttention.config.config import API_KEY, BASE_URL, MODEL, datasets
from TimeAttention.prompt.prompt_target_extraction import SYSTEMPROMPT, USERPROMPT

OPENAI_API_KEY = API_KEY
GPT_BASE_URL = BASE_URL
GPT_MODEL = MODEL
OUTPUT_DIR = f"processed/{datasets}"

# Dynamically generate output file name
def generate_output_filename():
    return os.path.join(OUTPUT_DIR, "response.jsonl")

# Loading patient data.
def load_patient_data(file_path):
    return pd.read_csv(file_path)

# Build a Prompt
def construct_prompts(patient_data):
    prompts = []
    for _, row in patient_data.iterrows():
        patient_id = row["PatientID"]
        disease_list = row["Disease"]
        medication_list = row["Medication"]
        procedure_list = row["Procedure"]

        filled_prompt = USERPROMPT.format(
            disease_list=disease_list,
            medication_list=medication_list,
            procedure_list=procedure_list
        )
        prompts.append({"PatientID": patient_id, "prompt": filled_prompt})
    return prompts

# Invoke GPT API
def call_gpt_api(prompt_item):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEMPROMPT},
            {"role": "user", "content": prompt_item["prompt"]},
        ],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        response = requests.post(GPT_BASE_URL, headers=headers, json=data, timeout=300)
        if response.status_code == 200:
            result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"PatientID": prompt_item["PatientID"], "response": result}
        else:
            error_msg = response.json().get("error", {}).get("message", f"HTTP {response.status_code}")
            print(f"Error for PatientID {prompt_item['PatientID']}: {error_msg}")
            return {"PatientID": prompt_item["PatientID"], "error": error_msg}
    except Exception as e:
        return {"PatientID": prompt_item["PatientID"], "error": str(e)}

if __name__ == "__main__":
    input_file = f"HiRMD/LLM_medical_diagnosis/processed/{datasets}/unique_data.csv"
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
    with ThreadPoolExecutor(max_workers=10) as executor, open(output_file, "a", encoding="utf-8") as output_f:
        futures = [executor.submit(call_gpt_api, prompt) for prompt in prompts]
        for future in tqdm(as_completed(futures), total=len(prompts), desc="Processing Prompts"):
            result = future.result()
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()  # Write to disk immediately.

    print(f"Results saved to {output_file}.")
