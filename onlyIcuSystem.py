import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# ========== MIMIC-IV ==========
print("MIMIC-IV ICU System:")

mimiciv_icu_df = pd.read_csv("HiRMD/datasets/mimic-iv/processed/icu_score_mimic-iv.csv")
mimiciv_label_df = pd.read_csv("HiRMD/datasets/mimic-iv/processed/label_mimic-iv.csv")

mimiciv_icu_df = mimiciv_icu_df[["PatientID", "apsiii_prob", "sapsii_prob"]]
mimiciv_label_df = mimiciv_label_df[["PatientID", "Outcome"]]
mimiciv_icu_df["apsiii_outcome"] = (mimiciv_icu_df["apsiii_prob"] > 0.5).astype(int)
mimiciv_icu_df["sapsii_outcome"] = (mimiciv_icu_df["sapsii_prob"] > 0.5).astype(int)
mimiciv_merged_df = pd.merge(mimiciv_icu_df, mimiciv_label_df, on="PatientID", how="inner")

mimiciv_merged_df["apsiii_correct"] = (mimiciv_merged_df["apsiii_outcome"] == mimiciv_merged_df["Outcome"]).astype(int)
mimiciv_merged_df["sapsii_correct"] = (mimiciv_merged_df["sapsii_outcome"] == mimiciv_merged_df["Outcome"]).astype(int)
mimiciv_apsiii_accuracy = mimiciv_merged_df["apsiii_correct"].mean()
mimiciv_sapsii_accuracy = mimiciv_merged_df["sapsii_correct"].mean()

apsiii_auc = roc_auc_score(mimiciv_merged_df["Outcome"], mimiciv_merged_df["apsiii_prob"])
apsiii_auprc = average_precision_score(mimiciv_merged_df["Outcome"], mimiciv_merged_df["apsiii_prob"])
apsiii_f1 = f1_score(mimiciv_merged_df["Outcome"], mimiciv_merged_df["apsiii_outcome"])
sapsii_auc = roc_auc_score(mimiciv_merged_df["Outcome"], mimiciv_merged_df["sapsii_prob"])
sapsii_auprc = average_precision_score(mimiciv_merged_df["Outcome"], mimiciv_merged_df["sapsii_prob"])
sapsii_f1 = f1_score(mimiciv_merged_df["Outcome"], mimiciv_merged_df["sapsii_outcome"])

print(f"APSIII Accuracy: {mimiciv_apsiii_accuracy:.4f}, AUROC: {apsiii_auc:.4f}, AUPRC: {apsiii_auprc:.4f}, F1 Score: {apsiii_f1:.4f}")
print(f"SAPSII Accuracy: {mimiciv_sapsii_accuracy:.4f}, AUROC: {sapsii_auc:.4f}, AUPRC: {sapsii_auprc:.4f}, F1 Score: {sapsii_f1:.4f}")

# ========== MIMIC-III ==========
print("\nMIMIC-III ICU System:")

mimiciii_icu_df = pd.read_csv("HiRMD/datasets/mimic-iii/processed/icu_score_mimic-iii.csv")
mimiciii_label_df = pd.read_csv("HiRMD/datasets/mimic-iii/processed/label_mimic-iii.csv")

mimiciii_icu_df = mimiciii_icu_df[["PatientID", "APSIIIProb", "SAPSIIProb"]]
mimiciii_label_df = mimiciii_label_df[["PatientID", "Outcome"]]
mimiciii_icu_df["apsiii_outcome"] = (mimiciii_icu_df["APSIIIProb"] > 0.5).astype(int)
mimiciii_icu_df["sapsii_outcome"] = (mimiciii_icu_df["SAPSIIProb"] > 0.5).astype(int)
mimiciii_merged_df = pd.merge(mimiciii_icu_df, mimiciii_label_df, on="PatientID", how="inner")

mimiciii_merged_df["apsiii_correct"] = (mimiciii_merged_df["apsiii_outcome"] == mimiciii_merged_df["Outcome"]).astype(int)
mimiciii_merged_df["sapsii_correct"] = (mimiciii_merged_df["sapsii_outcome"] == mimiciii_merged_df["Outcome"]).astype(int)
mimiciii_apsiii_accuracy = mimiciii_merged_df["apsiii_correct"].mean()
mimiciii_sapsii_accuracy = mimiciii_merged_df["sapsii_correct"].mean()

apsiii_auc = roc_auc_score(mimiciii_merged_df["Outcome"], mimiciii_merged_df["APSIIIProb"])
apsiii_auprc = average_precision_score(mimiciii_merged_df["Outcome"], mimiciii_merged_df["APSIIIProb"])
apsiii_f1 = f1_score(mimiciii_merged_df["Outcome"], mimiciii_merged_df["apsiii_outcome"])
sapsii_auc = roc_auc_score(mimiciii_merged_df["Outcome"], mimiciii_merged_df["SAPSIIProb"])
sapsii_auprc = average_precision_score(mimiciii_merged_df["Outcome"], mimiciii_merged_df["SAPSIIProb"])
sapsii_f1 = f1_score(mimiciii_merged_df["Outcome"], mimiciii_merged_df["sapsii_outcome"])

print(f"APSIII Accuracy: {mimiciii_apsiii_accuracy:.4f}, AUROC: {apsiii_auc:.4f}, AUPRC: {apsiii_auprc:.4f}, F1 Score: {apsiii_f1:.4f}")
print(f"SAPSII Accuracy: {mimiciii_sapsii_accuracy:.4f}, AUROC: {sapsii_auc:.4f}, AUPRC: {sapsii_auprc:.4f}, F1 Score: {sapsii_f1:.4f}")