import os
import pandas as pd
from TimeAttention.config.config import datasets
input_file = f"HiRMD/datasets/{datasets}/processed/DDP_{datasets}.csv"

# Output folder path
output_dir = f"processed/{datasets}"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "unique_data.csv")

# Read data.
df = pd.read_csv(input_file)

# Define the deduplication processing function.
def process_unique_values(group, column):
    # Separate the content of each record with a semicolon, remove duplicates, and then merge them.
    unique_values = set()
    for values in group[column]:
        if pd.notnull(values):
            unique_values.update([v.strip() for v in values.split(";")])
    return "; ".join(sorted(unique_values))

# Data processing in groups.
processed_data = (
    df.groupby("PatientID")
    .apply(
        lambda group: pd.Series({
            "Disease": process_unique_values(group, "Disease"),
            "Medication": process_unique_values(group, "Medication"),
            "Procedure": process_unique_values(group, "Procedure"),
        })
    )
    .reset_index()
)

# Save the results to a file.
processed_data.to_csv(output_file, index=False)

print(f"The processing is complete! The document has been saved to:{output_file}")
