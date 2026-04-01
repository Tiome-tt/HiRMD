import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.config import API_KEY, BASE_URL, MODEL, datasets
from prompt.prompt_target_extraction import SYSTEMPROMPT, USERPROMPT

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions

# GPT API Configuration
OPENAI_API_KEY = API_KEY
GPT_BASE_URL = BASE_URL
GPT_MODEL = MODEL
OUTPUT_DIR = f"processed/{datasets}"

# Output file path
def generate_output_filename():
    return os.path.join(OUTPUT_DIR, "response.jsonl")

# Error log file path
def generate_errorlog_filename():
    return os.path.join(OUTPUT_DIR, "error.log")

# Loading patient data.
def load_patient_data(file_path):
    return pd.read_csv(file_path)

# prompts
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

# GPT API request
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
    reraise=False
)
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
        response = requests.post(GPT_BASE_URL, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"PatientID": prompt_item["PatientID"], "response": result}
        else:
            error_msg = response.json().get("error", {}).get("message", f"HTTP {response.status_code}")
            raise requests.exceptions.RequestException(error_msg)
    except requests.exceptions.RequestException as e:
        return {"PatientID": prompt_item["PatientID"], "error": f"RequestException: {str(e)}"}
    except Exception as e:
        return {"PatientID": prompt_item["PatientID"], "error": f"OtherError: {str(e)}"}

if __name__ == "__main__":
    input_file = f"HiRMD/LLM_medical_diagnosis/processed/{datasets}/unique_data.csv"
    output_file = generate_output_filename()
    error_log_file = generate_errorlog_filename()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing_patient_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    pid = record.get("PatientID")
                    if pid and "response" in record:
                        existing_patient_ids.add(pid)
                except json.JSONDecodeError:
                    continue

    patient_data = load_patient_data(input_file)
    prompts = construct_prompts(patient_data)

    filtered_prompts = [p for p in prompts if p["PatientID"] not in existing_patient_ids]
    print(f"Total prompts: {len(prompts)}, Skipping {len(prompts) - len(filtered_prompts)} already completed.")

    with ThreadPoolExecutor(max_workers=10) as executor, \
         open(output_file, "a", encoding="utf-8") as output_f, \
         open(error_log_file, "a", encoding="utf-8") as error_f:

        futures = [executor.submit(call_gpt_api, prompt) for prompt in filtered_prompts]
        for future in tqdm(as_completed(futures), total=len(filtered_prompts), desc="Processing Prompts"):
            result = future.result()
            output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_f.flush()
            if "error" in result:
                error_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                error_f.flush()

    print(f"Results saved to {output_file}. Errors logged to {error_log_file}.")
     patient_data = load_patient_data(input_file)
