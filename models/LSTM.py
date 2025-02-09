import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from imblearn.over_sampling import RandomOverSampler
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from TimeAttention.config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, \
    Dropout, Input_Features
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 1. Data loading and preprocessing

data_file = f"HiRMD/datasets/{datasets}/processed/EHR_{datasets}.csv"
label_file = f"HiRMD/datasets/{datasets}/processed/label_{datasets}.csv"
gpt_response_file = f"HiRMD/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}.jsonl"
icu_file = f"HiRMD/datasets/{datasets}/processed/icu_score_{datasets}.csv"

df = pd.read_csv(data_file)
labels = pd.read_csv(label_file)

if len(df) != len(labels):
    raise ValueError("The number of rows in the feature data and label data do not match!")

df['Outcome'] = labels['Outcome']
# df['LOS'] = labels['LOS']
# df['Readmission'] = labels['Readmission']
df = df.sort_values(by=['PatientID', 'RecordTime'])

# GPT 0/1 sequence features
gpt_dict = {}
with open(gpt_response_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        pid = data['PatientID']
        gpt_seq = [int(x) for x in data['response'].split(',')]
        gpt_dict[pid] = gpt_seq

# GICU scoring data
icu_df = pd.read_csv(icu_file)
icu_df = icu_df.set_index('PatientID')
icu_features = list(icu_df.columns)
icu_scores_dict = icu_df.to_dict(orient='index')

# Define EHR input characteristics.
input_features = Input_Features
label_column = 'Outcome'

# normalize
all_features = df[input_features].values
scaler = StandardScaler().fit(all_features)

X_list, y_list, pids = [], [], []
grouped = df.groupby('PatientID')
for patient_id, group in grouped:
    group = group.sort_values(by='RecordTime')
    scaled_feat = scaler.transform(group[input_features].values)
    X_list.append(torch.tensor(scaled_feat, dtype=torch.float32))
    y_list.append(torch.tensor(group[label_column].values[-1], dtype=torch.float32))
    pids.append(patient_id)

X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
y = torch.tensor(y_list, dtype=torch.float32)

# Dataset split
X_train_full, X_temp, y_train_full, y_temp, pids_train_full, pids_temp = train_test_split(
    X_padded, y, pids, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test, pids_val, pids_test = train_test_split(
    X_temp, y_temp, pids_temp, test_size=0.5, random_state=42)

# Oversampling
ros = RandomOverSampler(random_state=42)
X_train_flat = X_train_full.reshape(X_train_full.shape[0], -1).numpy()
y_train_np = y_train_full.numpy()

train_df = pd.DataFrame(X_train_flat)
train_df['y'] = y_train_np
train_df['pid'] = pids_train_full
train_df.columns = train_df.columns.astype(str)

X_cols = [c for c in train_df.columns if c != 'y']
y_col = 'y'

X_resampled, y_resampled = ros.fit_resample(train_df[X_cols], train_df[y_col])
train_df_resampled = pd.DataFrame(X_resampled, columns=X_cols)
train_df_resampled['y'] = y_resampled

pids_train_resampled = train_df_resampled['pid'].values
train_df_resampled = train_df_resampled.drop(['y', 'pid'], axis=1)
X_train_res_np = train_df_resampled.values

X_train_resampled = torch.tensor(X_train_res_np.reshape(-1, X_train_full.shape[1], X_train_full.shape[2]), dtype=torch.float32)
y_train_resampled = torch.tensor(y_resampled, dtype=torch.float32)
pids_train_resampled = pids_train_resampled.tolist()

# 2. Dataset class
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

        # GPT feature
        if pid in self.gpt_dict:
            gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32)
        else:
            gpt_feature = torch.zeros(14, dtype=torch.float32)

        # ICU feature
        if pid in self.icu_dict:
            icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
            icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
        else:
            icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)

        return x, y, gpt_feature, icu_feature

train_dataset = EHRDataset(X_train_resampled, y_train_resampled, pids_train_resampled, gpt_dict, icu_scores_dict, icu_features)
val_dataset = EHRDataset(X_val, y_val, pids_val, gpt_dict, icu_scores_dict, icu_features)
test_dataset = EHRDataset(X_test, y_test, pids_test, gpt_dict, icu_scores_dict, icu_features)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Model Definition
class FusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 gpt_seq_len=14, icu_feature_dim=33,
                 lstm_dropout=0.2, embed_dim=32, mlp_hidden_dim=64):
        super(FusionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=True)
        self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
        self.icu_linear = nn.Linear(icu_feature_dim, embed_dim)
        fusion_input_dim = hidden_size*2 + embed_dim + embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_hidden_dim),
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
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

    def forward(self, x, gpt_feature, icu_feature):
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1, :]
        gpt_embed = self.gpt_linear(gpt_feature)
        icu_embed = self.icu_linear(icu_feature)
        fusion = torch.cat([final_out, gpt_embed, icu_embed], dim=-1)
        out = self.mlp(fusion)
        return out

# 4. Training, validation, and testing functions
def accuracy_calc(outputs, labels):
    predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
    correct = (predictions == labels).sum().item()
    return correct / len(labels)

def evaluate_metrics(outputs, labels):
    # outputs: [N], raw logits
    # labels: [N], 0 or 1
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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

def evaluate_test_model(model, test_loader, device, criterion):
    return evaluate_model(model, test_loader, criterion, device)

# 5. Training model (early stopping mechanism)
input_size = len(input_features)
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.0005
num_epochs = 100

icu_feature_dim = len(icu_features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel(input_size, hidden_size, num_layers, output_size,
                    gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

best_val_acc = 0.0
epochs_no_improve = 0
patience = 10
best_model_path = 'HiRMD/saved_models/best_lstm.pth'

for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}, Val F1: {val_f1:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Load the best model.
model.load_state_dict(torch.load(best_model_path))
test_loss, test_acc, test_auc, test_auprc, test_f1 = evaluate_test_model(model, test_loader, device, criterion)
print(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, AUPRC: {test_auprc:.4f}, F1: {test_f1:.4f}")

