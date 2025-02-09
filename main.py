# GRU
import gc
import logging
import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from imblearn.over_sampling import RandomOverSampler
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from TimeAttention.config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, Dropout, Input_Features
from torch.optim.lr_scheduler import ReduceLROnPlateau


log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'training.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')

# 1. Data loading and preprocessing

data_file = f"HiRMD/datasets/{datasets}/processed/EHR_{datasets}.csv"
label_file = f"HiRMD/datasets/{datasets}/processed/label_{datasets}.csv"
gpt_response_file = f"HiRMD/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}.jsonl"
icu_file = f"HiRMD/datasets/{datasets}/processed/icu_score_{datasets}.csv"

df = pd.read_csv(data_file)
labels = pd.read_csv(label_file)

if len(df) != len(labels):
    raise ValueError("Feature data and label data size do not match!")

labels['Outcome'] = labels['Outcome'].astype(int)

df['Outcome'] = labels['Outcome']
# df['LOS'] = labels['LOS']
# df['Readmission'] = labels['Readmission']
df = df.sort_values(by=['PatientID', 'RecordTime'])

# GPT feature
gpt_dict = {}
with open(gpt_response_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        pid = data['PatientID']
        gpt_seq = [int(x) for x in data['response'].split(',')]
        gpt_dict[pid] = gpt_seq

# ICU score
icu_df = pd.read_csv(icu_file).set_index('PatientID')
icu_features = list(icu_df.columns)
icu_scores_dict = icu_df.to_dict(orient='index')

input_features = Input_Features
label_column = 'Outcome'

all_features = df[input_features].values
scaler = StandardScaler().fit(all_features)

X_list, y_list, pids = [], [], []
grouped = df.groupby('PatientID')
for patient_id, group in grouped:
    group = group.sort_values(by='RecordTime')
    scaled_feat = scaler.transform(group[input_features].values)
    X_list.append(torch.tensor(scaled_feat, dtype=torch.float32))
    y_val = int(group[label_column].values[-1])
    y_list.append(y_val)
    pids.append(patient_id)

X_train, X_temp, y_train, y_temp, pids_train, pids_temp = train_test_split(X_list, y_list, pids, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test, pids_val, pids_test = train_test_split(X_temp, y_temp, pids_temp, test_size=0.5, random_state=42)

X_train_padded = pad_sequence(X_train, batch_first=True, padding_value=0.0)
X_val_padded = pad_sequence(X_val, batch_first=True, padding_value=0.0)
X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=0.0)

y_train = np.array(y_train, dtype=int)
y_val = np.array(y_val, dtype=int)
y_test = np.array(y_test, dtype=int)

X_train_flat = X_train_padded.reshape(X_train_padded.shape[0], -1)

train_arr = np.hstack([X_train_flat, y_train.reshape(-1,1), np.array(pids_train, dtype=object).reshape(-1,1)])
columns = [f"feat_{i}" for i in range(X_train_flat.shape[1])] + ["y", "pid"]
train_df = pd.DataFrame(train_arr, columns=columns)

train_df['y'] = train_df['y'].astype(int)

X_cols = [c for c in train_df.columns if c != 'y']
y_col = 'y'

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(train_df[X_cols], train_df[y_col])

train_df_resampled = pd.concat([X_resampled, pd.DataFrame({'y': y_resampled})], axis=1)
if 'pid' not in train_df_resampled.columns:
    raise KeyError("pid column not found after resampling. Check X_cols definition.")

pids_train_resampled = train_df_resampled['pid'].values
train_df_resampled = train_df_resampled.drop(['y', 'pid'], axis=1)

X_train_res_np = train_df_resampled.values.astype(float)
X_train_resampled = torch.tensor(X_train_res_np.reshape(-1, X_train_padded.shape[1], X_train_padded.shape[2]), dtype=torch.float32)
y_train_resampled = torch.tensor(y_resampled, dtype=torch.float32)

y_train_resampled_long = y_train_resampled.long()

class_weights = (len(y_train_resampled_long) - torch.bincount(y_train_resampled_long)) / torch.bincount(y_train_resampled_long)
pos_weight = class_weights[1] / class_weights[0]

class EHRDataset(Dataset):
    def __init__(self, X, y, pids, gpt_dict, icu_dict, icu_features):
        self.X = X
        self.y = y
        self.pids = pids
        self.gpt_dict = gpt_dict
        self.icu_dict = icu_dict
        self.icu_features = icu_features

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        pid = self.pids[idx]

        if pid in self.gpt_dict:
            gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32)
        else:
            gpt_feature = torch.zeros(14, dtype=torch.float32)

        if pid in self.icu_dict:
            icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
            icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
        else:
            icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)

        return x, y, gpt_feature, icu_feature

train_dataset = EHRDataset(X_train_resampled, y_train_resampled, pids_train_resampled, gpt_dict, icu_scores_dict, icu_features)
val_dataset = EHRDataset(X_val_padded, torch.tensor(y_val, dtype=torch.float32), pids_val, gpt_dict, icu_scores_dict, icu_features)
test_dataset = EHRDataset(X_test_padded, torch.tensor(y_test, dtype=torch.float32), pids_test, gpt_dict, icu_scores_dict, icu_features)

train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)

# 2. Define the multi-head attention mechanism and model.

class DeathRiskPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 gpt_seq_len=14, icu_feature_dim=33 if datasets=="mimic-iv" else 35 if datasets == "mimic-iii" else 12,
                 dropout=Dropout, embed_dim=32, mlp_hidden_dim=32, num_heads=8):
        super(DeathRiskPredictionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
        # self.icu_linear = nn.Linear(icu_feature_dim, embed_dim)
        self.icu_mlp = nn.Sequential(
            nn.Linear(icu_feature_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2 + embed_dim, num_heads=num_heads,
                                                         dropout=dropout)

        fusion_input_dim = hidden_size * 2 + embed_dim
        final_fusion_dim = fusion_input_dim + embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(final_fusion_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, output_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, gpt_feature, icu_feature):
        gru_out, _ = self.gru(x)  # gru_out: (batch_size, seq_len, hidden_size*2)

        final_out = gru_out[:, -1, :]

        gpt_embed = self.gpt_linear(gpt_feature)
        # icu_embed = self.icu_linear(icu_feature)
        icu_embed = self.icu_mlp(icu_feature)

        fusion = torch.cat([final_out, icu_embed], dim=-1)
        fusion = fusion.unsqueeze(0)

        attention_out, _ = self.multihead_attention(fusion, fusion,
                                                    fusion)
        attention_out = attention_out.squeeze(0)

        final_fusion = torch.cat([attention_out, gpt_embed], dim=-1)

        out = self.mlp(final_fusion)
        return out


# 3. Define cross-validation training and evaluation processes

def accuracy_calc(outputs, labels):
    predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


def evaluate_metrics(outputs, labels):
    prob = torch.sigmoid(outputs).cpu().numpy()
    labels = labels.cpu().numpy()
    preds = (prob > 0.5).astype(float)

    acc = (preds == labels).mean()
    try:
        auc = roc_auc_score(labels, prob)
    except:
        auc = float('nan')
    try:
        auprc = average_precision_score(labels, prob)
    except:
        auprc = float('nan')
    try:
        f1 = f1_score(labels, preds)
    except:
        f1 = float('nan')
    return acc, auc, auprc, f1


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for X_batch, y_batch, gpt_batch, icu_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        gpt_batch = gpt_batch.to(device)
        icu_batch = icu_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch, gpt_batch, icu_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        acc = accuracy_calc(outputs, y_batch)
        total_correct += acc * X_batch.size(0)
        total_samples += X_batch.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch, gpt_batch, icu_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            gpt_batch = gpt_batch.to(device)
            icu_batch = icu_batch.to(device)
            outputs = model(X_batch, gpt_batch, icu_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_outputs.append(outputs.squeeze().cpu())
            all_labels.append(y_batch.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)
    return avg_loss, acc, auc, auprc, f1


# 4. Cross-validation training

num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_size = len(input_features)
hidden_size = Hidden_Size
num_layers = Num_Layers
output_size = 1
learning_rate = Learning_Rate
num_epochs = Num_Epochs
icu_feature_dim = len(icu_features)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)

# Cross-validation
fold_results = []
train_losses_all_folds = []
val_losses_all_folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_list, y_list)):
    print(f"Starting Fold {fold + 1}/{num_folds}...")

    try:
        X_train_fold = [X_list[i] for i in train_idx]
        y_train_fold = [y_list[i] for i in train_idx]
        X_val_fold = [X_list[i] for i in val_idx]
        y_val_fold = [y_list[i] for i in val_idx]

        X_train_padded = pad_sequence(X_train_fold, batch_first=True, padding_value=0.0)
        X_val_padded = pad_sequence(X_val_fold, batch_first=True, padding_value=0.0)

        y_train_fold = torch.tensor(y_train_fold, dtype=torch.float32)
        y_val_fold = torch.tensor(y_val_fold, dtype=torch.float32)

        train_dataset = EHRDataset(X_train_padded, y_train_fold, [pids[i] for i in train_idx], gpt_dict,
                                   icu_scores_dict, icu_features)
        val_dataset = EHRDataset(X_val_padded, y_val_fold, [pids[i] for i in val_idx], gpt_dict, icu_scores_dict,
                                 icu_features)
        train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)

        model = DeathRiskPredictionModel(input_size, hidden_size, num_layers, output_size,
                                        gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_val_auc = 0.0
        best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_fold_{fold + 1}.pth")

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
            logging.info(
                f"{datasets}, Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_auc)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_auc': best_val_auc,
                }, best_model_path)
                print(f"Fold {fold + 1}: New best model saved with AUC: {val_auc:.4f}")

        train_losses_all_folds.append(train_losses)
        val_losses_all_folds.append(val_losses)

        # Load the best model and evaluate it on the validation set
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
        fold_results.append((val_acc, val_auc, val_auprc, val_f1))
    except Exception as e:
        print(f"Error in Fold {fold + 1}: {e}")
        fold_results.append((None, None, None, None))

    torch.cuda.empty_cache()
    gc.collect()

# Check the fold_results and choose the best fold
if len(fold_results) == 0 or all(result[1] is None for result in fold_results):
    raise ValueError("No valid fold results. Please check the training process.")

best_fold = np.argmax([result[1] for result in fold_results if result[1] is not None])
best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_fold_{best_fold + 1}.pth")
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Test set evaluation
X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=0.0)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_dataset = EHRDataset(X_test_padded, y_test, pids_test, gpt_dict, icu_scores_dict, icu_features)
test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)

test_loss, test_acc, test_auc, test_auprc, test_f1 = evaluate_model(model, test_loader, criterion, device)
print(
    f"Test Set Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUROC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")


# Draw a learning curve
plt.figure(figsize=(12, 6))

for fold in range(num_folds):
    plt.plot(train_losses_all_folds[fold], label=f'Fold {fold + 1} Train Loss')
    plt.plot(val_losses_all_folds[fold], label=f'Fold {fold + 1} Val Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss per Fold on {datasets}')
plt.legend()
plt.grid(True)
plt.show()


# # Transformer
# import gc
# import logging
# import os
# import json
# import math
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
# from imblearn.over_sampling import RandomOverSampler
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
# import matplotlib.pyplot as plt
# from TimeAttention.config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, \
#     Dropout, Input_Features
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from imblearn.over_sampling import SMOTE
#
# log_dir = 'log'
# os.makedirs(log_dir, exist_ok=True)
#
# log_file_path = os.path.join(log_dir, 'training.log')
# logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')
#
# # 1. Data loading and preprocessing
#
# data_file = f"HiRMD/datasets/{datasets}/processed/EHR_{datasets}.csv"
# label_file = f"HiRMD/datasets/{datasets}/processed/label_{datasets}.csv"
# gpt_response_file = f"HiRMD/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}.jsonl"
# icu_file = f"HiRMD/datasets/{datasets}/processed/icu_score_{datasets}.csv"
#
# df = pd.read_csv(data_file)
# labels = pd.read_csv(label_file)
#
# if len(df) != len(labels):
#     raise ValueError("Feature data and label data size do not match!")
#
# labels['Outcome'] = labels['Outcome'].astype(int)
#
# df['Outcome'] = labels['Outcome']
# # df['LOS'] = labels['LOS']
# # df['Readmission'] = labels['Readmission']
# df = df.sort_values(by=['PatientID', 'RecordTime'])
#
# # GPT feature
# gpt_dict = {}
# with open(gpt_response_file, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         pid = data['PatientID']
#         gpt_seq = [int(x) for x in data['response'].split(',')]
#         gpt_dict[pid] = gpt_seq
#
# # ICU score
# icu_df = pd.read_csv(icu_file).set_index('PatientID')
# icu_features = list(icu_df.columns)
# icu_scores_dict = icu_df.to_dict(orient='index')
#
# input_features = Input_Features
# label_column = 'Outcome'
#
# all_features = df[input_features].values
# scaler = StandardScaler().fit(all_features)
#
# X_list, y_list, pids = [], [], []
# grouped = df.groupby('PatientID')
# for patient_id, group in grouped:
#     group = group.sort_values(by='RecordTime')
#     scaled_feat = scaler.transform(group[input_features].values)
#     X_list.append(torch.tensor(scaled_feat, dtype=torch.float32))
#     y_val = int(group[label_column].values[-1])
#     y_list.append(y_val)
#     pids.append(patient_id)
#
# X_train, X_temp, y_train, y_temp, pids_train, pids_temp = train_test_split(X_list, y_list, pids, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test, pids_val, pids_test = train_test_split(X_temp, y_temp, pids_temp, test_size=0.5, random_state=42)
#
# X_train_padded = pad_sequence(X_train, batch_first=True, padding_value=0.0)
# X_val_padded = pad_sequence(X_val, batch_first=True, padding_value=0.0)
# X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=0.0)
#
# y_train = np.array(y_train, dtype=int)
# y_val = np.array(y_val, dtype=int)
# y_test = np.array(y_test, dtype=int)
#
# X_train_flat = X_train_padded.reshape(X_train_padded.shape[0], -1)
#
# train_arr = np.hstack([X_train_flat, y_train.reshape(-1,1), np.array(pids_train, dtype=object).reshape(-1,1)])
# columns = [f"feat_{i}" for i in range(X_train_flat.shape[1])] + ["y", "pid"]
# train_df = pd.DataFrame(train_arr, columns=columns)
#
# train_df['y'] = train_df['y'].astype(int)
#
# X_cols = [c for c in train_df.columns if c != 'y']
# y_col = 'y'
#
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(train_df[X_cols], train_df[y_col])
# # smote = SMOTE(random_state=42)
# # X_resampled, y_resampled = smote.fit_resample(train_df[X_cols], train_df[y_col])
#
# train_df_resampled = pd.concat([X_resampled, pd.DataFrame({'y': y_resampled})], axis=1)
# if 'pid' not in train_df_resampled.columns:
#     raise KeyError("pid column not found after resampling. Check X_cols definition.")
#
# pids_train_resampled = train_df_resampled['pid'].values
# train_df_resampled = train_df_resampled.drop(['y', 'pid'], axis=1)
#
# X_train_res_np = train_df_resampled.values.astype(float)
# X_train_resampled = torch.tensor(X_train_res_np.reshape(-1, X_train_padded.shape[1], X_train_padded.shape[2]), dtype=torch.float32)
# y_train_resampled = torch.tensor(y_resampled, dtype=torch.float32)
#
# y_train_resampled_long = y_train_resampled.long()
#
# class_weights = (len(y_train_resampled_long) - torch.bincount(y_train_resampled_long)) / torch.bincount(y_train_resampled_long)
# pos_weight = class_weights[1] / class_weights[0]
#
# class EHRDataset(Dataset):
#     def __init__(self, X, y, pids, gpt_dict, icu_dict, icu_features):
#         self.X = X
#         self.y = y
#         self.pids = pids
#         self.gpt_dict = gpt_dict
#         self.icu_dict = icu_dict
#         self.icu_features = icu_features
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         y = self.y[idx]
#         pid = self.pids[idx]
#
#         if pid in self.gpt_dict:
#             gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32)
#         else:
#             gpt_feature = torch.zeros(14, dtype=torch.float32)
#
#         if pid in self.icu_dict:
#             icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
#             icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
#         else:
#             icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)
#
#         return x, y, gpt_feature, icu_feature
#
# train_dataset = EHRDataset(X_train_resampled, y_train_resampled, pids_train_resampled, gpt_dict, icu_scores_dict, icu_features)
# val_dataset = EHRDataset(X_val_padded, torch.tensor(y_val, dtype=torch.float32), pids_val, gpt_dict, icu_scores_dict, icu_features)
# test_dataset = EHRDataset(X_test_padded, torch.tensor(y_test, dtype=torch.float32), pids_test, gpt_dict, icu_scores_dict, icu_features)
#
# train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)
#
# # 2. Define positional coding and Transformer models
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.3, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Calculate the location code
#         pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2, )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :].to(x.device)
#         return self.dropout(x)
#
# class DeathRiskPredictionModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size,
#                  gpt_seq_len=14, icu_feature_dim=33 if datasets=="mimic-iv" else 35 if datasets == "mimic-iii" else 12,
#                  dropout=Dropout, embed_dim=Hidden_Size, mlp_hidden_dim=32, num_heads=8):
#         super(DeathRiskPredictionModel, self).__init__()
#         self.embed_dim = embed_dim
#         self.input_projection = nn.Linear(input_size, embed_dim)
#         self.positional_encoding = PositionalEncoding(embed_dim, dropout)
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
# #         self.icu_linear = nn.Linear(icu_feature_dim, embed_dim)
#         self.icu_mlp = nn.Sequential(
#             nn.Linear(icu_feature_dim, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(mlp_hidden_dim, embed_dim)
#         )
#
#         self.multihead_attention = nn.MultiheadAttention(embed_dim=2 * embed_dim, num_heads=num_heads,
#                                                          dropout=dropout)
#
#         final_fusion_dim = 2 * embed_dim + embed_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(final_fusion_dim, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(mlp_hidden_dim, output_size)
#         )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x, gpt_feature, icu_feature):
#         """
#         x: (batch_size, seq_len, input_size)
#         gpt_feature: (batch_size, gpt_seq_len)
#         icu_feature: (batch_size, icu_feature_dim)
#         """
#         x = self.input_projection(x)
#         x = self.positional_encoding(x)
#
#         x = x.permute(1, 0, 2)
#         transformer_out = self.transformer_encoder(x)
#         transformer_out = transformer_out.permute(1, 0, 2)
#
#         transformer_rep = transformer_out.mean(dim=1)
#
#         # Extract GPT features and ICU features
#         gpt_embed = self.gpt_linear(gpt_feature)
#         # icu_embed = self.icu_linear(icu_feature)
#         icu_embed = self.icu_mlp(icu_feature)
#
#         fusion = torch.cat([transformer_rep, icu_embed], dim=-1)
#
#         fusion = fusion.unsqueeze(0)
#
#         attention_out, _ = self.multihead_attention(fusion, fusion, fusion)
#         attention_out = attention_out.squeeze(0)
#
#         final_fusion = torch.cat([attention_out, gpt_embed], dim=-1)
#
#         out = self.mlp(final_fusion)
#         return out
#
# # 3. Define cross-validation training and evaluation processes
#
# def accuracy_calc(outputs, labels):
#     predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
#     correct = (predictions == labels).sum().item()
#     return correct / len(labels)
#
# def evaluate_metrics(outputs, labels):
#     prob = torch.sigmoid(outputs).cpu().numpy()
#     labels = labels.cpu().numpy()
#     preds = (prob > 0.5).astype(float)
#
#     acc = (preds == labels).mean()
#     try:
#         auc = roc_auc_score(labels, prob)
#     except:
#         auc = float('nan')
#     try:
#         auprc = average_precision_score(labels, prob)
#     except:
#         auprc = float('nan')
#     try:
#         f1 = f1_score(labels, preds)
#     except:
#         f1 = float('nan')
#     return acc, auc, auprc, f1
#
# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
#     for X_batch, y_batch, gpt_batch, icu_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         gpt_batch = gpt_batch.to(device)
#         icu_batch = icu_batch.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(X_batch, gpt_batch, icu_batch)
#         loss = criterion(outputs.squeeze(), y_batch)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#         optimizer.step()
#
#         total_loss += loss.item() * X_batch.size(0)
#         acc = accuracy_calc(outputs, y_batch)
#         total_correct += acc * X_batch.size(0)
#         total_samples += X_batch.size(0)
#     return total_loss / total_samples, total_correct / total_samples
#
# def evaluate_model(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_outputs = []
#     all_labels = []
#     with torch.no_grad():
#         for X_batch, y_batch, gpt_batch, icu_batch in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             gpt_batch = gpt_batch.to(device)
#             icu_batch = icu_batch.to(device)
#             outputs = model(X_batch, gpt_batch, icu_batch)
#             loss = criterion(outputs.squeeze(), y_batch)
#             total_loss += loss.item() * X_batch.size(0)
#             all_outputs.append(outputs.squeeze().cpu())
#             all_labels.append(y_batch.cpu())
#
#     all_outputs = torch.cat(all_outputs)
#     all_labels = torch.cat(all_labels)
#     avg_loss = total_loss / len(loader.dataset)
#     acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)
#     return avg_loss, acc, auc, auprc, f1
#
# # 4. Cross-validation training
#
# num_folds = 5
# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = len(input_features)
# hidden_size = Hidden_Size
# num_layers = Num_Layers
# output_size = 1
# learning_rate = Learning_Rate
# num_epochs = Num_Epochs
# icu_feature_dim = len(icu_features)
# num_heads = 8
#
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
#
# model_save_dir = "saved_models"
# os.makedirs(model_save_dir, exist_ok=True)
#
# # Cross-validation
# fold_results = []
# train_losses_all_folds = []
# val_losses_all_folds = []
#
# for fold, (train_idx, val_idx) in enumerate(skf.split(X_list, y_list)):
#     print(f"Starting Fold {fold + 1}/{num_folds}...")
#
#     try:
#         X_train_fold = [X_list[i] for i in train_idx]
#         y_train_fold = [y_list[i] for i in train_idx]
#         X_val_fold = [X_list[i] for i in val_idx]
#         y_val_fold = [y_list[i] for i in val_idx]
#
#         X_train_padded = pad_sequence(X_train_fold, batch_first=True, padding_value=0.0)
#         X_val_padded = pad_sequence(X_val_fold, batch_first=True, padding_value=0.0)
#
#         y_train_fold = torch.tensor(y_train_fold, dtype=torch.float32)
#         y_val_fold = torch.tensor(y_val_fold, dtype=torch.float32)
#
#         train_dataset = EHRDataset(X_train_padded, y_train_fold, [pids[i] for i in train_idx], gpt_dict,
#                                    icu_scores_dict, icu_features)
#         val_dataset = EHRDataset(X_val_padded, y_val_fold, [pids[i] for i in val_idx], gpt_dict, icu_scores_dict,
#                                  icu_features)
#         train_loader_fold = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
#         val_loader_fold = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
#
#         model = DeathRiskPredictionModel(input_size, hidden_size, num_layers, output_size,
#                                          gpt_seq_len=14, icu_feature_dim=icu_feature_dim,
#                                          dropout=Dropout, embed_dim=hidden_size, mlp_hidden_dim=32, num_heads=num_heads).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#         scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
#
#         best_val_auc = 0.0
#         best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_fold_{fold + 1}.pth")
#
#         train_losses = []
#         val_losses = []
#
#         for epoch in range(num_epochs):
#             train_loss, train_acc = train_model(model, train_loader_fold, criterion, optimizer, device)
#             val_loss, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader_fold, criterion, device)
#             logging.info(
#                 f"{datasets}, Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
#
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#
#             scheduler.step(val_auc)
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr}")
#
#             if val_auc > best_val_auc:
#                 best_val_auc = val_auc
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'val_auc': best_val_auc,
#                 }, best_model_path)
#                 print(f"Fold {fold + 1}: New best model saved with AUC: {val_auc:.4f}")
#
#         train_losses_all_folds.append(train_losses)
#         val_losses_all_folds.append(val_losses)
#
#         # Load the best model and evaluate it on the validation set
#         checkpoint = torch.load(best_model_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader_fold, criterion, device)
#         fold_results.append((val_acc, val_auc, val_auprc, val_f1))
#     except Exception as e:
#         print(f"Error in Fold {fold + 1}: {e}")
#         fold_results.append((None, None, None, None))
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
# # Check the fold_results and choose the best fold
# if len(fold_results) == 0 or all(result[1] is None for result in fold_results):
#     raise ValueError("No valid fold results. Please check the training process.")
#
# best_fold = np.argmax([result[1] for result in fold_results if result[1] is not None])
# best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_fold_{best_fold + 1}.pth")
# checkpoint = torch.load(best_model_path)
# model = DeathRiskPredictionModel(input_size, hidden_size, num_layers, output_size,
#                                   gpt_seq_len=14, icu_feature_dim=icu_feature_dim,
#                                   dropout=Dropout, embed_dim=hidden_size, mlp_hidden_dim=32, num_heads=num_heads).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
#
# # Test set evaluation
# X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=0.0)
# y_test = torch.tensor(y_test, dtype=torch.float32)
# test_dataset = EHRDataset(X_test_padded, y_test, pids_test, gpt_dict, icu_scores_dict, icu_features)
# test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)
#
# test_loss, test_acc, test_auc, test_auprc, test_f1 = evaluate_model(model, test_loader, criterion, device)
# print(
#     f"Test Set Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUROC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")
#
# # Draw a learning curve
# plt.figure(figsize=(12, 6))
#
# for fold in range(num_folds):
#     plt.plot(train_losses_all_folds[fold], label=f'Fold {fold + 1} Train Loss')
#     plt.plot(val_losses_all_folds[fold], label=f'Fold {fold + 1} Val Loss')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title(f'Training and Validation Loss per Fold on {datasets}')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # LSTM
# import gc
# import logging
# import os
# import json
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
# from imblearn.over_sampling import RandomOverSampler
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
# import matplotlib.pyplot as plt
# from TimeAttention.config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, \
#     Dropout, Input_Features
# from torch.optim.lr_scheduler import ReduceLROnPlateau
#
# log_dir = 'log'
# os.makedirs(log_dir, exist_ok=True)
#
# log_file_path = os.path.join(log_dir, 'training.log')
# logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')
#
# # 1. Data loading and preprocessing
#
# data_file = f"HiRMD/datasets/{datasets}/processed/EHR_{datasets}.csv"
# label_file = f"HiRMD/datasets/{datasets}/processed/label_{datasets}.csv"
# gpt_response_file = f"HiRMD/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}.jsonl"
# icu_file = f"HiRMD/datasets/{datasets}/processed/icu_score_{datasets}.csv"
#
# df = pd.read_csv(data_file)
# labels = pd.read_csv(label_file)
#
# if len(df) != len(labels):
#     raise ValueError("Feature data and label data size do not match!")
#
# labels['Outcome'] = labels['Outcome'].astype(int)
#
# df['Outcome'] = labels['Outcome']
# # df['LOS'] = labels['LOS']
# # df['Readmission'] = labels['Readmission']
# df = df.sort_values(by=['PatientID', 'RecordTime'])
#
# # GPT feature
# gpt_dict = {}
# with open(gpt_response_file, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         pid = data['PatientID']
#         gpt_seq = [int(x) for x in data['response'].split(',')]
#         gpt_dict[pid] = gpt_seq
#
# # ICU score
# icu_df = pd.read_csv(icu_file).set_index('PatientID')
# icu_features = list(icu_df.columns)
# icu_scores_dict = icu_df.to_dict(orient='index')
#
# input_features = Input_Features
# label_column = 'Outcome'
#
# all_features = df[input_features].values
# scaler = StandardScaler().fit(all_features)
#
# X_list, y_list, pids = [], [], []
# grouped = df.groupby('PatientID')
# for patient_id, group in grouped:
#     group = group.sort_values(by='RecordTime')
#     scaled_feat = scaler.transform(group[input_features].values)
#     X_list.append(torch.tensor(scaled_feat, dtype=torch.float32))
#     y_val = int(group[label_column].values[-1])
#     y_list.append(y_val)
#     pids.append(patient_id)
#
# X_train, X_temp, y_train, y_temp, pids_train, pids_temp = train_test_split(X_list, y_list, pids, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test, pids_val, pids_test = train_test_split(X_temp, y_temp, pids_temp, test_size=0.5, random_state=42)
#
# X_train_padded = pad_sequence(X_train, batch_first=True, padding_value=0.0)
# X_val_padded = pad_sequence(X_val, batch_first=True, padding_value=0.0)
# X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=0.0)
#
# y_train = np.array(y_train, dtype=int)
# y_val = np.array(y_val, dtype=int)
# y_test = np.array(y_test, dtype=int)
#
# X_train_flat = X_train_padded.reshape(X_train_padded.shape[0], -1)
#
# train_arr = np.hstack([X_train_flat, y_train.reshape(-1,1), np.array(pids_train, dtype=object).reshape(-1,1)])
# columns = [f"feat_{i}" for i in range(X_train_flat.shape[1])] + ["y", "pid"]
# train_df = pd.DataFrame(train_arr, columns=columns)
#
# train_df['y'] = train_df['y'].astype(int)
#
# X_cols = [c for c in train_df.columns if c != 'y']
# y_col = 'y'
#
# ros = RandomOverSampler(random_state=42)
# X_resampled, y_resampled = ros.fit_resample(train_df[X_cols], train_df[y_col])
#
# train_df_resampled = pd.concat([X_resampled, pd.DataFrame({'y': y_resampled})], axis=1)
# if 'pid' not in train_df_resampled.columns:
#     raise KeyError("pid column not found after resampling. Check X_cols definition.")
#
# pids_train_resampled = train_df_resampled['pid'].values
# train_df_resampled = train_df_resampled.drop(['y', 'pid'], axis=1)
#
# X_train_res_np = train_df_resampled.values.astype(float)
# X_train_resampled = torch.tensor(X_train_res_np.reshape(-1, X_train_padded.shape[1], X_train_padded.shape[2]), dtype=torch.float32)
# y_train_resampled = torch.tensor(y_resampled, dtype=torch.float32)
#
# y_train_resampled_long = y_train_resampled.long()
#
# class_weights = (len(y_train_resampled_long) - torch.bincount(y_train_resampled_long)) / torch.bincount(y_train_resampled_long)
# pos_weight = class_weights[1] / class_weights[0]
#
# class EHRDataset(Dataset):
#     def __init__(self, X, y, pids, gpt_dict, icu_dict, icu_features):
#         self.X = X
#         self.y = y
#         self.pids = pids
#         self.gpt_dict = gpt_dict
#         self.icu_dict = icu_dict
#         self.icu_features = icu_features
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         y = self.y[idx]
#         pid = self.pids[idx]
#
#         if pid in self.gpt_dict:
#             gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32)
#         else:
#             gpt_feature = torch.zeros(14, dtype=torch.float32)
#
#         if pid in self.icu_dict:
#             icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
#             icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
#         else:
#             icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)
#
#         return x, y, gpt_feature, icu_feature
#
# train_dataset = EHRDataset(X_train_resampled, y_train_resampled, pids_train_resampled, gpt_dict, icu_scores_dict, icu_features)
# val_dataset = EHRDataset(X_val_padded, torch.tensor(y_val, dtype=torch.float32), pids_val, gpt_dict, icu_scores_dict, icu_features)
# test_dataset = EHRDataset(X_test_padded, torch.tensor(y_test, dtype=torch.float32), pids_test, gpt_dict, icu_scores_dict, icu_features)
#
# train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)
#
#
# # 2. Define multi-head attention mechanisms and models
#
# class DeathRiskPredictionModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size,
#                  gpt_seq_len=14, icu_feature_dim=33 if datasets=="mimic-iv" else 35 if datasets == "mimic-iii" else 12,
#                  dropout=Dropout, embed_dim=32, mlp_hidden_dim=32, num_heads=8):
#         super(DeathRiskPredictionModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
#         self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
# #         self.icu_linear = nn.Linear(icu_feature_dim, embed_dim)
#         self.icu_mlp = nn.Sequential(
#             nn.Linear(icu_feature_dim, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(mlp_hidden_dim, embed_dim)
#         )
#
#         self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2 + embed_dim, num_heads=num_heads,
#                                                          dropout=dropout)
#
#         fusion_input_dim = hidden_size * 2 + embed_dim
#         final_fusion_dim = fusion_input_dim + embed_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(final_fusion_dim, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(mlp_hidden_dim, output_size)
#         )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x, gpt_feature, icu_feature):
#         lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size*2)
#
#         final_out = lstm_out[:, -1, :]
#
#         gpt_embed = self.gpt_linear(gpt_feature)
# #         icu_embed = self.icu_linear(icu_feature)
#         icu_embed = self.icu_mlp(icu_feature)
#
#         fusion = torch.cat([final_out, icu_embed], dim=-1)
#         fusion = fusion.unsqueeze(0)
#
#         attention_out, _ = self.multihead_attention(fusion, fusion,
#                                                     fusion)
#         attention_out = attention_out.squeeze(0)
#
#         final_fusion = torch.cat([attention_out, gpt_embed], dim=-1)
#
#         out = self.mlp(final_fusion)
#         return out
#
#
# # 3. Define cross-validation training and evaluation processes
#
# def accuracy_calc(outputs, labels):
#     predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
#     correct = (predictions == labels).sum().item()
#     return correct / len(labels)
#
#
# def evaluate_metrics(outputs, labels):
#     prob = torch.sigmoid(outputs).cpu().numpy()
#     labels = labels.cpu().numpy()
#     preds = (prob > 0.5).astype(float)
#
#     acc = (preds == labels).mean()
#     try:
#         auc = roc_auc_score(labels, prob)
#     except:
#         auc = float('nan')
#     try:
#         auprc = average_precision_score(labels, prob)
#     except:
#         auprc = float('nan')
#     try:
#         f1 = f1_score(labels, preds)
#     except:
#         f1 = float('nan')
#     return acc, auc, auprc, f1
#
#
# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     total_correct = 0
#     total_samples = 0
#     for X_batch, y_batch, gpt_batch, icu_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         gpt_batch = gpt_batch.to(device)
#         icu_batch = icu_batch.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(X_batch, gpt_batch, icu_batch)
#         loss = criterion(outputs.squeeze(), y_batch)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
#         optimizer.step()
#
#         total_loss += loss.item() * X_batch.size(0)
#         acc = accuracy_calc(outputs, y_batch)
#         total_correct += acc * X_batch.size(0)
#         total_samples += X_batch.size(0)
#     return total_loss / total_samples, total_correct / total_samples
#
#
# def evaluate_model(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     all_outputs = []
#     all_labels = []
#     with torch.no_grad():
#         for X_batch, y_batch, gpt_batch, icu_batch in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             gpt_batch = gpt_batch.to(device)
#             icu_batch = icu_batch.to(device)
#             outputs = model(X_batch, gpt_batch, icu_batch)
#             loss = criterion(outputs.squeeze(), y_batch)
#             total_loss += loss.item() * X_batch.size(0)
#             all_outputs.append(outputs.squeeze().cpu())
#             all_labels.append(y_batch.cpu())
#
#     all_outputs = torch.cat(all_outputs)
#     all_labels = torch.cat(all_labels)
#     avg_loss = total_loss / len(loader.dataset)
#     acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)
#     return avg_loss, acc, auc, auprc, f1
#
#
# # 4. Cross-validation training
#
# num_folds = 5
# skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = len(input_features)
# hidden_size = Hidden_Size
# num_layers = Num_Layers
# output_size = 1
# learning_rate = Learning_Rate
# num_epochs = Num_Epochs
# icu_feature_dim = len(icu_features)
#
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
#
# model_save_dir = "saved_models"
# os.makedirs(model_save_dir, exist_ok=True)
#
# # Cross-validation
# fold_results = []
# train_losses_all_folds = []
# val_losses_all_folds = []
#
# for fold, (train_idx, val_idx) in enumerate(skf.split(X_list, y_list)):
#     print(f"Starting Fold {fold + 1}/{num_folds}...")
#
#     try:
#         X_train_fold = [X_list[i] for i in train_idx]
#         y_train_fold = [y_list[i] for i in train_idx]
#         X_val_fold = [X_list[i] for i in val_idx]
#         y_val_fold = [y_list[i] for i in val_idx]
#
#         X_train_padded = pad_sequence(X_train_fold, batch_first=True, padding_value=0.0)
#         X_val_padded = pad_sequence(X_val_fold, batch_first=True, padding_value=0.0)
#
#         y_train_fold = torch.tensor(y_train_fold, dtype=torch.float32)
#         y_val_fold = torch.tensor(y_val_fold, dtype=torch.float32)
#
#         train_dataset = EHRDataset(X_train_padded, y_train_fold, [pids[i] for i in train_idx], gpt_dict,
#                                    icu_scores_dict, icu_features)
#         val_dataset = EHRDataset(X_val_padded, y_val_fold, [pids[i] for i in val_idx], gpt_dict, icu_scores_dict,
#                                  icu_features)
#         train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
#
#         model = DeathRiskPredictionModel(input_size, hidden_size, num_layers, output_size,
#                                         gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#         scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
#
#         best_val_auc = 0.0
#         best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_fold_{fold + 1}.pth")
#
#         train_losses = []
#         val_losses = []
#
#         for epoch in range(num_epochs):
#             train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
#             val_loss, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
#             logging.info(
#                 f"{datasets}, Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
#
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#
#             scheduler.step(val_auc)
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Current Learning Rate: {current_lr}")
#
#             if val_auc > best_val_auc:
#                 best_val_auc = val_auc
#                 torch.save({
#                     'model_state_dict': model.state_dict(),
#                     'val_auc': best_val_auc,
#                 }, best_model_path)
#                 print(f"Fold {fold + 1}: New best model saved with AUC: {val_auc:.4f}")
#
#         train_losses_all_folds.append(train_losses)
#         val_losses_all_folds.append(val_losses)
#
#         checkpoint = torch.load(best_model_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
#         fold_results.append((val_acc, val_auc, val_auprc, val_f1))
#     except Exception as e:
#         print(f"Error in Fold {fold + 1}: {e}")
#         fold_results.append((None, None, None, None))
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
# if len(fold_results) == 0 or all(result[1] is None for result in fold_results):
#     raise ValueError("No valid fold results. Please check the training process.")
#
# best_fold = np.argmax([result[1] for result in fold_results if result[1] is not None])
# best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_fold_{best_fold + 1}.pth")
# checkpoint = torch.load(best_model_path)
# model.load_state_dict(checkpoint['model_state_dict'])
#
# X_test_padded = pad_sequence(X_test, batch_first=True, padding_value=0.0)
# y_test = torch.tensor(y_test, dtype=torch.float32)
# test_dataset = EHRDataset(X_test_padded, y_test, pids_test, gpt_dict, icu_scores_dict, icu_features)
# test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False)
#
# test_loss, test_acc, test_auc, test_auprc, test_f1 = evaluate_model(model, test_loader, criterion, device)
# print(
#     f"Test Set Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUROC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")
#
#
# # Draw a learning curve
# plt.figure(figsize=(12, 6))
#
# for fold in range(num_folds):
#     plt.plot(train_losses_all_folds[fold], label=f'Fold {fold + 1} Train Loss')
#     plt.plot(val_losses_all_folds[fold], label=f'Fold {fold + 1} Val Loss')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title(f'Training and Validation Loss per Fold on {datasets}')
# plt.legend()
# plt.grid(True)
# plt.show()