import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from torch.optim.lr_scheduler import StepLR
from imblearn.over_sampling import RandomOverSampler

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 引入配置和你刚才保存的 RETAIN 模型
from config.config import Input_Features, datasets as ds
from retain import RETAIN_Fusion


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EHRFusionDataset(Dataset):
    def __init__(self, X_seqs, y, lens, demo, pids, gpt_dict, icu_dict, icu_features, gpt_dim=14):
        self.X_seqs = X_seqs
        self.y = y
        self.lens = lens
        self.demo = demo
        self.pids = pids
        self.gpt_dict = gpt_dict
        self.icu_dict = icu_dict
        self.icu_features = icu_features
        self.gpt_dim = gpt_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X_seqs[idx]
        y = self.y[idx]
        length = self.lens[idx]
        demo = self.demo[idx]
        pid = self.pids[idx]

        if pid in self.gpt_dict:
            gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32)
        else:
            gpt_feature = torch.zeros(self.gpt_dim, dtype=torch.float32)

        if pid in self.icu_dict:
            icu_vals = [self.icu_dict[pid].get(col, 0.0) for col in self.icu_features]
            icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
        else:
            icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)

        return x, y, length, demo, gpt_feature, icu_feature


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    x_seqs, ys, lens, demos, gpts, icus = zip(*batch)
    x_padded = pad_sequence(x_seqs, batch_first=True, padding_value=0.0)
    ys = torch.tensor(ys, dtype=torch.float32)
    lens = torch.tensor(lens, dtype=torch.long)
    demos = torch.stack(demos)
    gpts = torch.stack(gpts)
    icus = torch.stack(icus)
    return x_padded, ys, lens, demos, gpts, icus


def evaluate_metrics_from_logits(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = float('nan')
    try:
        auprc = average_precision_score(labels, probs)
    except:
        auprc = float('nan')
    try:
        f1 = f1_score(labels, preds)
    except:
        f1 = float('nan')
    return acc, auc, auprc, f1


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for x_batch, y_batch, lens_batch, demo_batch, gpt_batch, icu_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        lens_batch, demo_batch = lens_batch.to(device), demo_batch.to(device)
        gpt_batch, icu_batch = gpt_batch.to(device), icu_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch, demo_batch, lens_batch, gpt_batch, icu_batch).squeeze(1)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y_batch.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, auc, auprc, f1 = evaluate_metrics_from_logits(all_logits, all_labels)
    return avg_loss, acc, auc, auprc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch, lens_batch, demo_batch, gpt_batch, icu_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lens_batch, demo_batch = lens_batch.to(device), demo_batch.to(device)
            gpt_batch, icu_batch = gpt_batch.to(device), icu_batch.to(device)

            logits = model(x_batch, demo_batch, lens_batch, gpt_batch, icu_batch).squeeze(1)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, auc, auprc, f1 = evaluate_metrics_from_logits(all_logits, all_labels)
    return avg_loss, acc, auc, auprc, f1


def main():
    datasets = ds
    data_file = f"/home/user1/MXY/TimeAttention/datasets/{datasets}/processed/EHR_{datasets}.csv"
    label_file = f"/home/user1/MXY/TimeAttention/datasets/{datasets}/processed/label_{datasets}.csv"
    gpt_response_file = f"/home/user1/MXY/TimeAttention/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}_GPT-4o.jsonl"
    icu_file = f"/home/user1/MXY/TimeAttention/datasets/{datasets}/processed/icu_score_{datasets}.csv"

    print("Loading data... (This happens only once)")
    df = pd.read_csv(data_file)
    labels = pd.read_csv(label_file)

    labels['Outcome'] = labels['Outcome'].astype(int)
    df['Outcome'] = labels['Outcome']
    df = df.sort_values(by=['PatientID', 'RecordTime'])

    gpt_dict = {}
    with open(gpt_response_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            gpt_dict[str(item['PatientID'])] = [int(x) for x in item['response'].split(',')]

    icu_df = pd.read_csv(icu_file)
    icu_df['PatientID'] = icu_df['PatientID'].astype(str)
    icu_df = icu_df.set_index('PatientID')
    icu_features = list(icu_df.columns)
    icu_scores_dict = icu_df.to_dict(orient='index')

    input_features = Input_Features

    # 动态判断 demo 维度
    if datasets == "eicu":
        demo_features = ['Sex', 'Age']
    else:
        demo_features = ['Sex', 'Age', 'Height', 'Weight']

    seq_features = [f for f in input_features if f not in demo_features]
    label_column = 'Outcome'

    X_raw_list, y_list, pids, lens_list, demo_list = [], [], [], [], []
    grouped = df.groupby('PatientID')
    for patient_id, group in grouped:
        group = group.sort_values(by='RecordTime')
        X_raw_list.append(group[seq_features].values)
        y_list.append(int(group[label_column].values[-1]))
        pids.append(str(patient_id))
        lens_list.append(len(group))
        demo_list.append(torch.tensor(group[demo_features].iloc[0].values.astype(np.float32), dtype=torch.float32))

    seeds = [42, 1024, 2023, 8888, 9999]
    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Data loaded successfully! Starting {len(seeds)} RETAIN runs on {device}...")

    for run_idx, current_seed in enumerate(seeds):
        print(f"\n{'=' * 20} Run {run_idx + 1}/{len(seeds)} (Seed: {current_seed}) {'=' * 20}")
        set_seed(current_seed)

        X_train_raw, X_temp_raw, y_train, y_temp, pids_train, pids_temp, lens_train, lens_temp, demo_train, demo_temp = train_test_split(
            X_raw_list, y_list, pids, lens_list, demo_list, test_size=0.2, random_state=current_seed
        )
        X_val_raw, X_test_raw, y_val, y_test, pids_val, pids_test, lens_val, lens_test, demo_val, demo_test = train_test_split(
            X_temp_raw, y_temp, pids_temp, lens_temp, demo_temp, test_size=0.5, random_state=current_seed
        )

        scaler = StandardScaler()
        scaler.fit(np.vstack(X_train_raw))

        X_train_seqs = [torch.tensor(scaler.transform(seq), dtype=torch.float32) for seq in X_train_raw]
        X_val_seqs = [torch.tensor(scaler.transform(seq), dtype=torch.float32) for seq in X_val_raw]
        X_test_seqs = [torch.tensor(scaler.transform(seq), dtype=torch.float32) for seq in X_test_raw]

        ros = RandomOverSampler(random_state=current_seed)
        resampled_indices = ros.fit_resample(np.arange(len(y_train)).reshape(-1, 1), y_train)[0].flatten()

        train_dataset = EHRFusionDataset(
            [X_train_seqs[i] for i in resampled_indices], [y_train[i] for i in resampled_indices],
            [lens_train[i] for i in resampled_indices], [demo_train[i] for i in resampled_indices],
            [pids_train[i] for i in resampled_indices], gpt_dict, icu_scores_dict, icu_features, gpt_dim=14
        )
        val_dataset = EHRFusionDataset(X_val_seqs, y_val, lens_val, demo_val, pids_val, gpt_dict, icu_scores_dict,
                                       icu_features, gpt_dim=14)
        test_dataset = EHRFusionDataset(X_test_seqs, y_test, lens_test, demo_test, pids_test, gpt_dict, icu_scores_dict,
                                        icu_features, gpt_dim=14)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

        # 实例化 RETAIN
        model = RETAIN_Fusion(
            input_dim=len(seq_features),
            demo_dim=len(demo_features),  # 动态获取 demo 维度
            gpt_dim=14,
            icu_dim=len(icu_features),
            emb_dim=64,
            alpha_hidden_dim=64,
            beta_hidden_dim=64,
            output_dim=1,
            dropout=0.1
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

        # 模型保存路径修改为 retain
        best_model_path = f"/home/user1/MXY/TimeAttention/baselines/retain_best_model_seed_{current_seed}.pth"

        best_val_auc, epochs_no_improve, patience = 0.0, 0, 3

        for epoch in range(100):
            train_loss, train_acc, train_auc, train_auprc, train_f1 = train_one_epoch(model, train_loader, criterion,
                                                                                      optimizer, device)
            val_loss, val_acc, val_auc, val_auprc, val_f1 = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, test_auc, test_auprc, test_f1 = evaluate(model, test_loader, criterion, device)

        print(
            f"Run {run_idx + 1} Test Results -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")

        all_results.append(
            {'Seed': current_seed, 'Loss': test_loss, 'Accuracy': test_acc, 'AUROC': test_auc, 'AUPRC': test_auprc,
             'F1': test_f1})

    print("\n\n=================== RETAIN FINAL SUMMARY ===================")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    print("\n------------------- MEAN ± STD -------------------")
    for metric in ['Loss', 'Accuracy', 'AUROC', 'AUPRC', 'F1']:
        print(f"{metric + ':':<10} {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}")
    print("=====================================================\n")


if __name__ == "__main__":
    main()