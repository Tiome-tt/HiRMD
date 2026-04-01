# LSTM
import gc
import logging
import os
import json
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, Dropout, \
    Input_Features
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'training_lstm_ablation.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')


# set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)

    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1

def plot_all_histories_combined(all_collected_histories):
    num_folds_total = len(all_collected_histories)
    if num_folds_total == 0:
        print("No training histories to plot.")
        return

    fig, axs = plt.subplots(nrows=num_folds_total, ncols=3, figsize=(18, num_folds_total * 3), sharex='col')
    fig.suptitle('Ablation Evaluation (LSTM): All 25 Folds Training History', fontsize=18, y=1.0)

    for idx, meta in enumerate(all_collected_histories):
        hist = meta['history']
        s_f_title = f"S:{meta['seed']} F:{meta['fold']}"
        epochs = range(1, len(hist['train_loss']) + 1)

        if num_folds_total == 1:
            ax_row = axs
        else:
            ax_row = axs[idx]

        ax_loss = ax_row[0]
        ax_loss.plot(epochs, hist['train_loss'], label='Tr Loss', marker='o', markersize=3, alpha=0.7)
        ax_loss.plot(epochs, hist['val_loss'], label='Val Loss', marker='s', markersize=3, alpha=0.7)
        ax_loss.set_title(f"{s_f_title} - Loss", fontsize=10)
        ax_loss.grid(True, linestyle='--', alpha=0.5)

        ax_auc = ax_row[1]
        ax_auc.plot(epochs, hist['val_auc'], label='Val AUC', color='orange', marker='^', markersize=3)
        ax_auc.set_title(f"{s_f_title} - AUC", fontsize=10)
        ax_auc.grid(True, linestyle='--', alpha=0.5)

        ax_auprc = ax_row[2]
        ax_auprc.plot(epochs, hist['val_auprc'], label='Val AUPRC', color='green', marker='d', markersize=3)
        ax_auprc.set_title(f"{s_f_title} - AUPRC", fontsize=10)
        ax_auprc.grid(True, linestyle='--', alpha=0.5)

        if idx == 0:
            ax_loss.legend(fontsize=9, loc='upper right')
            ax_auc.legend(fontsize=9, loc='lower right')
            ax_auprc.legend(fontsize=9, loc='lower right')

        if idx == num_folds_total - 1:
            ax_loss.set_xlabel('Epochs')
            ax_auc.set_xlabel('Epochs')
            ax_auprc.set_xlabel('Epochs')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# load datasets
data_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/EHR_{datasets}.csv"
label_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/label_{datasets}.csv"
gpt_response_file = f"HiRMD/TimeAttention/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}_GPT-4o.jsonl"
icu_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/icu_score_{datasets}.csv"

df = pd.read_csv(data_file)
labels = pd.read_csv(label_file)

if len(df) != len(labels):
    raise ValueError("Feature data and label data size do not match!")

labels['Outcome'] = labels['Outcome'].astype(int)
df['Outcome'] = labels['Outcome']
df = df.sort_values(by=['PatientID', 'RecordTime'])

gpt_dict = {}
with open(gpt_response_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        pid = data['PatientID']
        gpt_seq = [int(x) for x in data['response'].split(',')]
        gpt_dict[pid] = gpt_seq

icu_df = pd.read_csv(icu_file).set_index('PatientID')
icu_df = icu_df.drop(columns=['APSIIIProb', 'SAPSIIProb', 'apsiii_prob', 'sapsii_prob'], errors='ignore')
icu_features = list(icu_df.columns)
icu_scores_dict = icu_df.to_dict(orient='index')

input_features = Input_Features
label_column = 'Outcome'

X_list_global_raw, y_list_global, pids_global = [], [], []
grouped = df.groupby('PatientID')
for patient_id, group in grouped:
    group = group.sort_values(by='RecordTime')
    raw_feat = group[input_features].values
    X_list_global_raw.append(raw_feat)
    y_val = int(group[label_column].values[-1])
    y_list_global.append(y_val)
    pids_global.append(patient_id)


class EHRDataset(Dataset):
    def __init__(self, X_list, y_list, pids, gpt_dict, icu_dict, icu_features):
        self.X = X_list
        self.y = y_list
        self.pids = pids
        self.gpt_dict = gpt_dict
        self.icu_dict = icu_dict
        self.icu_features = icu_features

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        pid = self.pids[idx]

        gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32) if pid in self.gpt_dict else torch.zeros(14,
                                                                                                                     dtype=torch.float32)

        if pid in self.icu_dict:
            icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
            icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
        else:
            icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)

        return x, y, gpt_feature, icu_feature


def custom_collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = torch.stack([item[1] for item in batch])
    gpts = torch.stack([item[2] for item in batch])
    icus = torch.stack([item[3] for item in batch])
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.int64)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return xs_padded, ys, gpts, icus, lengths

class LSTMAblationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 gpt_seq_len=14, icu_feature_dim=len(icu_features),
                 dropout=Dropout, embed_dim=32, num_heads=4):
        super(LSTMAblationModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
        self.icu_mlp = nn.Sequential(
            nn.Linear(icu_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        self.static_proj = nn.Linear(embed_dim * 2, hidden_size * 2)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout)

        final_fusion_dim = 4 * (hidden_size * 2) + (embed_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(final_fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_padded, lengths, gpt_feature, icu_feature):
        packed_x = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_out, _ = self.lstm(packed_x)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        batch_size, max_seq_len, _ = lstm_out.size()

        mask = torch.arange(max_seq_len, device=lstm_out.device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(
            1).to(lstm_out.device)

        gpt_embed = self.gpt_linear(gpt_feature)
        icu_embed = self.icu_mlp(icu_feature)
        static_embed = torch.cat([gpt_embed, icu_embed], dim=-1)
        query = self.static_proj(static_embed).unsqueeze(0)
        key_value = lstm_out.transpose(0, 1)

        attn_out, _ = self.cross_attention(query, key_value, key_value, key_padding_mask=mask)
        attn_out = attn_out.squeeze(0)

        last_valid_idx = lengths - 1
        last_valid_features = lstm_out[torch.arange(batch_size, device=lstm_out.device), last_valid_idx, :]

        seq_mask_float = (~mask).unsqueeze(-1).float()
        sum_hidden = torch.sum(lstm_out * seq_mask_float, dim=1)
        mean_pool = sum_hidden / lengths.unsqueeze(1).float().to(lstm_out.device)

        lstm_out_masked_for_max = lstm_out.masked_fill(mask.unsqueeze(-1), -1e9)
        max_pool, _ = torch.max(lstm_out_masked_for_max, dim=1)

        final_fusion = torch.cat([attn_out, last_valid_features, mean_pool, max_pool, static_embed], dim=-1)
        out = self.mlp(final_fusion)
        return out


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


def evaluate_model(model, loader, criterion, device, return_probs=False):
    model.eval()
    total_loss = 0
    all_outputs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch, gpt_batch, icu_batch, lengths in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            gpt_batch, icu_batch, lengths = gpt_batch.to(device), icu_batch.to(device), lengths.to(device)
            outputs = model(X_batch, lengths, gpt_batch, icu_batch)

            loss = criterion(outputs.squeeze(-1), y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_outputs.append(outputs.squeeze(-1).cpu())
            all_labels.append(y_batch.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)

    if return_probs:
        probs = torch.sigmoid(all_outputs)
        return avg_loss, acc, auc, auprc, f1, probs, all_labels
    return avg_loss, acc, auc, auprc, f1


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for X_batch, y_batch, gpt_batch, icu_batch, lengths in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        gpt_batch, icu_batch, lengths = gpt_batch.to(device), icu_batch.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch, lengths, gpt_batch, icu_batch)

        loss = criterion(outputs.squeeze(-1), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        prob = torch.sigmoid(outputs.squeeze(-1))
        predictions = (prob > 0.5).float()
        total_correct += (predictions == y_batch).sum().item()
        total_samples += X_batch.size(0)
    return total_loss / total_samples, total_correct / total_samples


seeds = [42, 1024, 2023, 8888, 9999]
final_seed_results = []
all_folds_historical_data = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = len(input_features)
hidden_size = Hidden_Size
num_layers = Num_Layers
output_size = 1
learning_rate = Learning_Rate
num_epochs = Num_Epochs
icu_feature_dim = len(icu_features)
num_folds = 5

model_save_dir = "saved_models_lstm"
os.makedirs(model_save_dir, exist_ok=True)

print(f"========== Starting Ablation (LSTM) over {len(seeds)} Seeds ==========\n")

for seed in seeds:
    print(f"\n==================== Running for SEED: {seed} ====================")
    set_seed(seed)

    X_cv_raw, X_test_raw, y_cv, y_test, pids_cv, pids_test = train_test_split(
        X_list_global_raw, y_list_global, pids_global, test_size=0.15, random_state=seed, stratify=y_list_global
    )

    flat_cv_features = np.vstack(X_cv_raw)
    scaler = StandardScaler().fit(flat_cv_features)

    X_cv = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_cv_raw]
    X_test = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_test_raw]

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_results = []
    fold_best_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"  Starting Fold {fold + 1}/{num_folds} for Seed {seed}...")
        try:
            X_train_fold = [X_cv[i] for i in train_idx]
            y_train_fold = [y_cv[i] for i in train_idx]
            pids_train_fold = [pids_cv[i] for i in train_idx]

            X_val_fold = [X_cv[i] for i in val_idx]
            y_val_fold = [y_cv[i] for i in val_idx]
            pids_val_fold = [pids_cv[i] for i in val_idx]

            num_pos = sum(y_train_fold)
            num_neg = len(y_train_fold) - num_pos
            pos_weight = torch.tensor([(num_neg / num_pos) * 0.7], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            train_dataset = EHRDataset(X_train_fold, y_train_fold, pids_train_fold, gpt_dict, icu_scores_dict,
                                       icu_features)
            val_dataset = EHRDataset(X_val_fold, y_val_fold, pids_val_fold, gpt_dict, icu_scores_dict, icu_features)

            train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)

            model = LSTMAblationModel(input_size, hidden_size, num_layers, output_size,
                                      gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

            best_val_auc = 0.0
            best_threshold_for_fold = 0.5
            best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_seed_{seed}_fold_{fold + 1}.pth")

            history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_auprc': []}

            for epoch in range(num_epochs):
                train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

                val_loss, val_acc, val_auc, val_auprc, val_f1, val_probs, val_labels = evaluate_model(
                    model, val_loader, criterion, device, return_probs=True
                )
                scheduler.step(val_auc)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)
                history['val_auprc'].append(val_auprc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc

                    best_threshold, _ = find_best_threshold(val_labels.numpy(), val_probs.numpy())
                    best_threshold_for_fold = best_threshold

                    torch.save({'model_state_dict': model.state_dict(), 'val_auc': best_val_auc}, best_model_path)

            all_folds_historical_data.append({
                'seed': seed,
                'fold': fold + 1,
                'history': history
            })

            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)

            fold_results.append(val_auc)
            fold_best_thresholds.append(best_threshold_for_fold)
            print(f"  Fold {fold + 1} Best Val AUC: {val_auc:.4f} | Dynamic Threshold: {best_threshold_for_fold:.4f}")

        except Exception as e:
            print(f"  Error in Fold {fold + 1}: {e}")
            fold_results.append(None)
            fold_best_thresholds.append(None)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"  Ensembling 5 folds for Seed {seed} on its specific Test Set...")
    test_dataset = EHRDataset(X_test, y_test, pids_test, gpt_dict, icu_scores_dict, icu_features)
    test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)

    all_fold_probs = []
    final_labels = None

    for fold in range(num_folds):
        if fold_results[fold] is None:
            continue
        model_path = os.path.join(model_save_dir, f"{datasets}_best_model_seed_{seed}_fold_{fold + 1}.pth")
        if os.path.exists(model_path):

            model = LSTMAblationModel(input_size, hidden_size, num_layers, output_size,
                                      gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            _, _, _, _, _, probs, labels = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device,
                                                          return_probs=True)
            all_fold_probs.append(probs)
            if final_labels is None:
                final_labels = labels

    ensemble_probs = torch.mean(torch.stack(all_fold_probs), dim=0)
    final_labels_np = final_labels.numpy()
    ensemble_probs_np = ensemble_probs.numpy()

    valid_thresholds = [t for t in fold_best_thresholds if t is not None]
    avg_best_threshold = np.mean(valid_thresholds) if valid_thresholds else 0.5
    print(f"  --> Applied Ensembled Threshold: {avg_best_threshold:.4f} (instead of default 0.5)")

    ensemble_preds = (ensemble_probs_np > avg_best_threshold).astype(float)

    seed_acc = (ensemble_preds == final_labels_np).mean()
    seed_auc = roc_auc_score(final_labels_np, ensemble_probs_np)
    seed_auprc = average_precision_score(final_labels_np, ensemble_probs_np)
    seed_f1 = f1_score(final_labels_np, ensemble_preds)

    print(
        f"  Seed {seed} Test Results -> Acc: {seed_acc:.4f}, AUC: {seed_auc:.4f}, AUPRC: {seed_auprc:.4f}, F1: {seed_f1:.4f}")

    final_seed_results.append({
        'seed': seed,
        'acc': seed_acc,
        'auc': seed_auc,
        'auprc': seed_auprc,
        'f1': seed_f1
    })

print("\n========== Rendering Combined Training Histories (Scrolling View) ==========")
plot_all_histories_combined(all_folds_historical_data)

print("\n\n" + "=" * 60)
print("datasets:" + datasets)
print("       FINAL ROBUSTNESS EVALUATION RESULTS (LSTM ABLATION)       ")
print("=" * 60)

acc_list = [res['acc'] for res in final_seed_results]
auc_list = [res['auc'] for res in final_seed_results]
auprc_list = [res['auprc'] for res in final_seed_results]
f1_list = [res['f1'] for res in final_seed_results]

print(f"Across {len(seeds)} different random seeds (Ensemble evaluated):")
print(f"Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"AUROC    : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"AUPRC    : {np.mean(auprc_list):.4f} ± {np.std(auprc_list):.4f}")
print(f"F1 Score : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print("=" * 60)

# Transformer
# import gc
# import logging
# import os
# import json
# import random
# import math
# import pandas as pd
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
# from config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, Dropout, \
#     Input_Features
# from torch.optim.lr_scheduler import ReduceLROnPlateau
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# log_dir = 'log'
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'training_transformer_ablation.log')
# logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# def find_best_threshold(y_true, y_prob):
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
#     f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
#     best_idx = np.argmax(f1_scores)
#
#     if best_idx < len(thresholds):
#         best_threshold = thresholds[best_idx]
#     else:
#         best_threshold = 0.5
#
#     best_f1 = f1_scores[best_idx]
#     return best_threshold, best_f1
#
# def plot_all_histories_combined(all_collected_histories):
#     num_folds_total = len(all_collected_histories)
#     if num_folds_total == 0:
#         print("No training histories to plot.")
#         return
#
#     fig, axs = plt.subplots(nrows=num_folds_total, ncols=3, figsize=(18, num_folds_total * 3), sharex='col')
#     fig.suptitle('Ablation Evaluation (Transformer): All 25 Folds Training History', fontsize=18, y=1.0)
#
#     for idx, meta in enumerate(all_collected_histories):
#         hist = meta['history']
#         s_f_title = f"S:{meta['seed']} F:{meta['fold']}"
#         epochs = range(1, len(hist['train_loss']) + 1)
#
#         if num_folds_total == 1:
#             ax_row = axs
#         else:
#             ax_row = axs[idx]
#
#         ax_loss = ax_row[0]
#         ax_loss.plot(epochs, hist['train_loss'], label='Tr Loss', marker='o', markersize=3, alpha=0.7)
#         ax_loss.plot(epochs, hist['val_loss'], label='Val Loss', marker='s', markersize=3, alpha=0.7)
#         ax_loss.set_title(f"{s_f_title} - Loss", fontsize=10)
#         ax_loss.grid(True, linestyle='--', alpha=0.5)
#
#         ax_auc = ax_row[1]
#         ax_auc.plot(epochs, hist['val_auc'], label='Val AUC', color='orange', marker='^', markersize=3)
#         ax_auc.set_title(f"{s_f_title} - AUC", fontsize=10)
#         ax_auc.grid(True, linestyle='--', alpha=0.5)
#
#         ax_auprc = ax_row[2]
#         ax_auprc.plot(epochs, hist['val_auprc'], label='Val AUPRC', color='green', marker='d', markersize=3)
#         ax_auprc.set_title(f"{s_f_title} - AUPRC", fontsize=10)
#         ax_auprc.grid(True, linestyle='--', alpha=0.5)
#
#         if idx == 0:
#             ax_loss.legend(fontsize=9, loc='upper right')
#             ax_auc.legend(fontsize=9, loc='lower right')
#             ax_auprc.legend(fontsize=9, loc='lower right')
#
#         if idx == num_folds_total - 1:
#             ax_loss.set_xlabel('Epochs')
#             ax_auc.set_xlabel('Epochs')
#             ax_auprc.set_xlabel('Epochs')
#
#     plt.tight_layout(rect=[0, 0, 1, 0.98])
#     plt.show()
#
#
# data_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/EHR_{datasets}.csv"
# label_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/label_{datasets}.csv"
# gpt_response_file = f"HiRMD/TimeAttention/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}_GPT-4o.jsonl"
# icu_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/icu_score_{datasets}.csv"
#
# df = pd.read_csv(data_file)
# labels = pd.read_csv(label_file)
#
# if len(df) != len(labels):
#     raise ValueError("Feature data and label data size do not match!")
#
# labels['Outcome'] = labels['Outcome'].astype(int)
# df['Outcome'] = labels['Outcome']
# df = df.sort_values(by=['PatientID', 'RecordTime'])
#
# gpt_dict = {}
# with open(gpt_response_file, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         pid = data['PatientID']
#         gpt_seq = [int(x) for x in data['response'].split(',')]
#         gpt_dict[pid] = gpt_seq
#
# icu_df = pd.read_csv(icu_file).set_index('PatientID')
# icu_df = icu_df.drop(columns=['APSIIIProb', 'SAPSIIProb', 'apsiii_prob', 'sapsii_prob'], errors='ignore')
# icu_features = list(icu_df.columns)
# icu_scores_dict = icu_df.to_dict(orient='index')
#
# input_features = Input_Features
# label_column = 'Outcome'
#
# X_list_global_raw, y_list_global, pids_global = [], [], []
# grouped = df.groupby('PatientID')
# for patient_id, group in grouped:
#     group = group.sort_values(by='RecordTime')
#     raw_feat = group[input_features].values
#     X_list_global_raw.append(raw_feat)
#     y_val = int(group[label_column].values[-1])
#     y_list_global.append(y_val)
#     pids_global.append(patient_id)
#
# class EHRDataset(Dataset):
#     def __init__(self, X_list, y_list, pids, gpt_dict, icu_dict, icu_features):
#         self.X = X_list
#         self.y = y_list
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
#         y = torch.tensor(self.y[idx], dtype=torch.float32)
#         pid = self.pids[idx]
#
#         gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32) if pid in self.gpt_dict else torch.zeros(14,
#                                                                                                                      dtype=torch.float32)
#
#         if pid in self.icu_dict:
#             icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
#             icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
#         else:
#             icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)
#
#         return x, y, gpt_feature, icu_feature
#
#
# def custom_collate_fn(batch):
#     xs = [item[0] for item in batch]
#     ys = torch.stack([item[1] for item in batch])
#     gpts = torch.stack([item[2] for item in batch])
#     icus = torch.stack([item[3] for item in batch])
#     lengths = torch.tensor([len(x) for x in xs], dtype=torch.int64)
#     xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
#     return xs_padded, ys, gpts, icus, lengths
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # batch_first = True
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#
# class TransformerAblationModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size,
#                  gpt_seq_len=14, icu_feature_dim=len(icu_features),
#                  dropout=Dropout, embed_dim=32, num_heads=8):
#         super(TransformerAblationModel, self).__init__()
#
#         self.d_model = hidden_size * 2
#
#         self.input_proj = nn.Linear(input_size, self.d_model)
#
#         self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout)
#
#         encoder_layers = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=num_heads,
#             dim_feedforward=self.d_model * 2,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
#
#         self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
#         self.icu_mlp = nn.Sequential(
#             nn.Linear(icu_feature_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, embed_dim)
#         )
#         self.static_proj = nn.Linear(embed_dim * 2, self.d_model)
#
#         self.cross_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_heads, dropout=dropout)
#
#         final_fusion_dim = 4 * self.d_model + (embed_dim * 2)
#         self.mlp = nn.Sequential(
#             nn.Linear(final_fusion_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_size)
#         )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x_padded, lengths, gpt_feature, icu_feature):
#         batch_size, max_seq_len, _ = x_padded.size()
#
#         mask = torch.arange(max_seq_len, device=x_padded.device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(
#             1).to(x_padded.device)
#
#         x_proj = self.input_proj(x_padded)
#         x_pe = self.pos_encoder(x_proj)
#
#         transformer_out = self.transformer_encoder(x_pe, src_key_padding_mask=mask)
#
#         gpt_embed = self.gpt_linear(gpt_feature)
#         icu_embed = self.icu_mlp(icu_feature)
#         static_embed = torch.cat([gpt_embed, icu_embed], dim=-1)
#
#         query = self.static_proj(static_embed).unsqueeze(0)
#
#         key_value = transformer_out.transpose(0, 1)
#
#         attn_out, _ = self.cross_attention(query, key_value, key_value, key_padding_mask=mask)
#         attn_out = attn_out.squeeze(0)
#
#         last_valid_idx = lengths - 1
#         last_valid_features = transformer_out[torch.arange(batch_size, device=transformer_out.device), last_valid_idx,
#                               :]
#
#         seq_mask_float = (~mask).unsqueeze(-1).float()
#         sum_hidden = torch.sum(transformer_out * seq_mask_float, dim=1)
#         mean_pool = sum_hidden / lengths.unsqueeze(1).float().to(transformer_out.device)
#
#         out_masked_for_max = transformer_out.masked_fill(mask.unsqueeze(-1), -1e9)
#         max_pool, _ = torch.max(out_masked_for_max, dim=1)
#
#         final_fusion = torch.cat([attn_out, last_valid_features, mean_pool, max_pool, static_embed], dim=-1)
#         out = self.mlp(final_fusion)
#         return out
#
#
# def evaluate_metrics(outputs, labels):
#     prob = torch.sigmoid(outputs).cpu().numpy()
#     labels = labels.cpu().numpy()
#     preds = (prob > 0.5).astype(float)
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
# def evaluate_model(model, loader, criterion, device, return_probs=False):
#     model.eval()
#     total_loss = 0
#     all_outputs, all_labels = [], []
#     with torch.no_grad():
#         for X_batch, y_batch, gpt_batch, icu_batch, lengths in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             gpt_batch, icu_batch, lengths = gpt_batch.to(device), icu_batch.to(device), lengths.to(device)
#             outputs = model(X_batch, lengths, gpt_batch, icu_batch)
#
#             loss = criterion(outputs.squeeze(-1), y_batch)
#             total_loss += loss.item() * X_batch.size(0)
#             all_outputs.append(outputs.squeeze(-1).cpu())
#             all_labels.append(y_batch.cpu())
#
#     all_outputs = torch.cat(all_outputs)
#     all_labels = torch.cat(all_labels)
#     avg_loss = total_loss / len(loader.dataset)
#     acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)
#
#     if return_probs:
#         probs = torch.sigmoid(all_outputs)
#         return avg_loss, acc, auc, auprc, f1, probs, all_labels
#     return avg_loss, acc, auc, auprc, f1
#
#
# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss, total_correct, total_samples = 0, 0, 0
#     for X_batch, y_batch, gpt_batch, icu_batch, lengths in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         gpt_batch, icu_batch, lengths = gpt_batch.to(device), icu_batch.to(device), lengths.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(X_batch, lengths, gpt_batch, icu_batch)
#
#         loss = criterion(outputs.squeeze(-1), y_batch)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
#         optimizer.step()
#
#         total_loss += loss.item() * X_batch.size(0)
#         prob = torch.sigmoid(outputs.squeeze(-1))
#         predictions = (prob > 0.5).float()
#         total_correct += (predictions == y_batch).sum().item()
#         total_samples += X_batch.size(0)
#     return total_loss / total_samples, total_correct / total_samples
#
#
# seeds = [42, 1024, 2023, 8888, 9999]
# final_seed_results = []
# all_folds_historical_data = []
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = len(input_features)
# hidden_size = Hidden_Size
# num_layers = Num_Layers
# output_size = 1
# learning_rate = Learning_Rate
# num_epochs = Num_Epochs
# icu_feature_dim = len(icu_features)
# num_folds = 5
#
# model_save_dir = "saved_models_transformer"
# os.makedirs(model_save_dir, exist_ok=True)
#
# print(f"========== Starting Ablation (Transformer) over {len(seeds)} Seeds ==========\n")
#
# for seed in seeds:
#     print(f"\n==================== Running for SEED: {seed} ====================")
#     set_seed(seed)
#
#     X_cv_raw, X_test_raw, y_cv, y_test, pids_cv, pids_test = train_test_split(
#         X_list_global_raw, y_list_global, pids_global, test_size=0.15, random_state=seed, stratify=y_list_global
#     )
#
#     flat_cv_features = np.vstack(X_cv_raw)
#     scaler = StandardScaler().fit(flat_cv_features)
#
#     X_cv = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_cv_raw]
#     X_test = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_test_raw]
#
#     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
#
#     fold_results = []
#     fold_best_thresholds = []
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
#         print(f"  Starting Fold {fold + 1}/{num_folds} for Seed {seed}...")
#         try:
#             X_train_fold = [X_cv[i] for i in train_idx]
#             y_train_fold = [y_cv[i] for i in train_idx]
#             pids_train_fold = [pids_cv[i] for i in train_idx]
#
#             X_val_fold = [X_cv[i] for i in val_idx]
#             y_val_fold = [y_cv[i] for i in val_idx]
#             pids_val_fold = [pids_cv[i] for i in val_idx]
#
#             num_pos = sum(y_train_fold)
#             num_neg = len(y_train_fold) - num_pos
#             pos_weight = torch.tensor([(num_neg / num_pos) * 0.7], dtype=torch.float32).to(device)
#             criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
#             train_dataset = EHRDataset(X_train_fold, y_train_fold, pids_train_fold, gpt_dict, icu_scores_dict,
#                                        icu_features)
#             val_dataset = EHRDataset(X_val_fold, y_val_fold, pids_val_fold, gpt_dict, icu_scores_dict, icu_features)
#
#             train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, collate_fn=custom_collate_fn)
#             val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)
#
#             model = TransformerAblationModel(input_size, hidden_size, num_layers, output_size,
#                                              gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)
#
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#             scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
#
#             best_val_auc = 0.0
#             best_threshold_for_fold = 0.5
#             best_model_path = os.path.join(model_save_dir, f"{datasets}_best_model_seed_{seed}_fold_{fold + 1}.pth")
#
#             history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_auprc': []}
#
#             for epoch in range(num_epochs):
#                 train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
#
#                 val_loss, val_acc, val_auc, val_auprc, val_f1, val_probs, val_labels = evaluate_model(
#                     model, val_loader, criterion, device, return_probs=True
#                 )
#                 scheduler.step(val_auc)
#
#                 history['train_loss'].append(train_loss)
#                 history['val_loss'].append(val_loss)
#                 history['val_auc'].append(val_auc)
#                 history['val_auprc'].append(val_auprc)
#
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#
#                     best_threshold, _ = find_best_threshold(val_labels.numpy(), val_probs.numpy())
#                     best_threshold_for_fold = best_threshold
#
#                     torch.save({'model_state_dict': model.state_dict(), 'val_auc': best_val_auc}, best_model_path)
#
#             all_folds_historical_data.append({
#                 'seed': seed,
#                 'fold': fold + 1,
#                 'history': history
#             })
#
#             checkpoint = torch.load(best_model_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
#
#             fold_results.append(val_auc)
#             fold_best_thresholds.append(best_threshold_for_fold)
#             print(f"  Fold {fold + 1} Best Val AUC: {val_auc:.4f} | Dynamic Threshold: {best_threshold_for_fold:.4f}")
#
#         except Exception as e:
#             print(f"  Error in Fold {fold + 1}: {e}")
#             fold_results.append(None)
#             fold_best_thresholds.append(None)
#
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     print(f"  Ensembling 5 folds for Seed {seed} on its specific Test Set...")
#     test_dataset = EHRDataset(X_test, y_test, pids_test, gpt_dict, icu_scores_dict, icu_features)
#     test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)
#
#     all_fold_probs = []
#     final_labels = None
#
#     for fold in range(num_folds):
#         if fold_results[fold] is None:
#             continue
#         model_path = os.path.join(model_save_dir, f"{datasets}_best_model_seed_{seed}_fold_{fold + 1}.pth")
#         if os.path.exists(model_path):
#             model = TransformerAblationModel(input_size, hidden_size, num_layers, output_size,
#                                              gpt_seq_len=14, icu_feature_dim=icu_feature_dim).to(device)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             _, _, _, _, _, probs, labels = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device,
#                                                           return_probs=True)
#             all_fold_probs.append(probs)
#             if final_labels is None:
#                 final_labels = labels
#
#     ensemble_probs = torch.mean(torch.stack(all_fold_probs), dim=0)
#     final_labels_np = final_labels.numpy()
#     ensemble_probs_np = ensemble_probs.numpy()
#
#     valid_thresholds = [t for t in fold_best_thresholds if t is not None]
#     avg_best_threshold = np.mean(valid_thresholds) if valid_thresholds else 0.5
#     print(f"  --> Applied Ensembled Threshold: {avg_best_threshold:.4f} (instead of default 0.5)")
#
#     ensemble_preds = (ensemble_probs_np > avg_best_threshold).astype(float)
#
#     seed_acc = (ensemble_preds == final_labels_np).mean()
#     seed_auc = roc_auc_score(final_labels_np, ensemble_probs_np)
#     seed_auprc = average_precision_score(final_labels_np, ensemble_probs_np)
#     seed_f1 = f1_score(final_labels_np, ensemble_preds)
#
#     print(
#         f"  Seed {seed} Test Results -> Acc: {seed_acc:.4f}, AUC: {seed_auc:.4f}, AUPRC: {seed_auprc:.4f}, F1: {seed_f1:.4f}")
#
#     final_seed_results.append({
#         'seed': seed,
#         'acc': seed_acc,
#         'auc': seed_auc,
#         'auprc': seed_auprc,
#         'f1': seed_f1
#     })
#
# print("\n========== Rendering Combined Training Histories (Scrolling View) ==========")
# plot_all_histories_combined(all_folds_historical_data)
#
# print("\n\n" + "=" * 60)
# print("datasets:" + datasets)
# print("     FINAL ROBUSTNESS EVALUATION RESULTS (TRANSFORMER ABLATION)     ")
# print("=" * 60)
#
# acc_list = [res['acc'] for res in final_seed_results]
# auc_list = [res['auc'] for res in final_seed_results]
# auprc_list = [res['auprc'] for res in final_seed_results]
# f1_list = [res['f1'] for res in final_seed_results]
#
# print(f"Across {len(seeds)} different random seeds (Ensemble evaluated):")
# print(f"Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
# print(f"AUROC    : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
# print(f"AUPRC    : {np.mean(auprc_list):.4f} ± {np.std(auprc_list):.4f}")
# print(f"F1 Score : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
# print("=" * 60)





# # w o ICU Score
# import gc
# import logging
# import os
# import json
# import random
# import pandas as pd
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
# from config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, Dropout, \
#     Input_Features
# from torch.optim.lr_scheduler import ReduceLROnPlateau
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# log_dir = 'log'
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'training_ablation_no_icu.log')
# logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# def find_best_threshold(y_true, y_prob):
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
#     f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
#     best_idx = np.argmax(f1_scores)
#
#     if best_idx < len(thresholds):
#         best_threshold = thresholds[best_idx]
#     else:
#         best_threshold = 0.5
#
#     best_f1 = f1_scores[best_idx]
#     return best_threshold, best_f1
#
#
# def plot_all_histories_combined(all_collected_histories):
#
#     num_folds_total = len(all_collected_histories)
#     if num_folds_total == 0:
#         print("No training histories to plot.")
#         return
#
#     fig, axs = plt.subplots(nrows=num_folds_total, ncols=3, figsize=(18, num_folds_total * 3), sharex='col')
#
#     fig.suptitle('Robustness Evaluation (Ablation: No ICU): All 25 Folds Training History', fontsize=18, y=1.0)
#
#     for idx, meta in enumerate(all_collected_histories):
#         hist = meta['history']
#         # 为了节省空间，标题用简写 Seed:Fold
#         s_f_title = f"S:{meta['seed']} F:{meta['fold']}"
#         epochs = range(1, len(hist['train_loss']) + 1)
#
#         if num_folds_total == 1:
#             ax_row = axs
#         else:
#             ax_row = axs[idx]
#
#         ax_loss = ax_row[0]
#         ax_loss.plot(epochs, hist['train_loss'], label='Tr Loss', marker='o', markersize=3, alpha=0.7)
#         ax_loss.plot(epochs, hist['val_loss'], label='Val Loss', marker='s', markersize=3, alpha=0.7)
#         ax_loss.set_title(f"{s_f_title} - Loss", fontsize=10)
#         ax_loss.grid(True, linestyle='--', alpha=0.5)
#
#         ax_auc = ax_row[1]
#         ax_auc.plot(epochs, hist['val_auc'], label='Val AUC', color='orange', marker='^', markersize=3)
#         ax_auc.set_title(f"{s_f_title} - AUC", fontsize=10)
#         ax_auc.grid(True, linestyle='--', alpha=0.5)
#
#         ax_auprc = ax_row[2]
#         ax_auprc.plot(epochs, hist['val_auprc'], label='Val AUPRC', color='green', marker='d', markersize=3)
#         ax_auprc.set_title(f"{s_f_title} - AUPRC", fontsize=10)
#         ax_auprc.grid(True, linestyle='--', alpha=0.5)
#
#         if idx == 0:
#             ax_loss.legend(fontsize=9, loc='upper right')
#             ax_auc.legend(fontsize=9, loc='lower right')
#             ax_auprc.legend(fontsize=9, loc='lower right')
#
#         if idx == num_folds_total - 1:
#             ax_loss.set_xlabel('Epochs')
#             ax_auc.set_xlabel('Epochs')
#             ax_auprc.set_xlabel('Epochs')
#
#     plt.tight_layout(rect=[0, 0, 1, 0.98])
#     plt.show()
#
# data_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/EHR_{datasets}.csv"
# label_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/label_{datasets}.csv"
# gpt_response_file = f"HiRMD/TimeAttention/LLM_medical_diagnosis/outputs/LLM_Diagnosis_{datasets}_GPT-4o.jsonl"
#
# df = pd.read_csv(data_file)
# labels = pd.read_csv(label_file)
#
# if len(df) != len(labels):
#     raise ValueError("Feature data and label data size do not match!")
#
# labels['Outcome'] = labels['Outcome'].astype(int)
# df['Outcome'] = labels['Outcome']
# df = df.sort_values(by=['PatientID', 'RecordTime'])
#
# gpt_dict = {}
# with open(gpt_response_file, 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         pid = data['PatientID']
#         gpt_seq = [int(x) for x in data['response'].split(',')]
#         gpt_dict[pid] = gpt_seq
#
# input_features = Input_Features
# label_column = 'Outcome'
#
# X_list_global_raw, y_list_global, pids_global = [], [], []
# grouped = df.groupby('PatientID')
# for patient_id, group in grouped:
#     group = group.sort_values(by='RecordTime')
#     raw_feat = group[input_features].values
#     X_list_global_raw.append(raw_feat)
#     y_val = int(group[label_column].values[-1])
#     y_list_global.append(y_val)
#     pids_global.append(patient_id)
#
# class EHRDataset(Dataset):
#     def __init__(self, X_list, y_list, pids, gpt_dict):
#         self.X = X_list
#         self.y = y_list
#         self.pids = pids
#         self.gpt_dict = gpt_dict
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         y = torch.tensor(self.y[idx], dtype=torch.float32)
#         pid = self.pids[idx]
#
#         gpt_feature = torch.tensor(self.gpt_dict[pid], dtype=torch.float32) if pid in self.gpt_dict else torch.zeros(14,
#                                                                                                                      dtype=torch.float32)
#
#         return x, y, gpt_feature
#
#
# def custom_collate_fn(batch):
#     xs = [item[0] for item in batch]
#     ys = torch.stack([item[1] for item in batch])
#     gpts = torch.stack([item[2] for item in batch])
#     lengths = torch.tensor([len(x) for x in xs], dtype=torch.int64)
#     xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
#     return xs_padded, ys, gpts, lengths
#
#
# class MultiHeadAttentionModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size,
#                  gpt_seq_len=14, dropout=Dropout, embed_dim=32, num_heads=4):
#         super(MultiHeadAttentionModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
#
#         self.gpt_linear = nn.Linear(gpt_seq_len, embed_dim)
#
#         self.static_proj = nn.Linear(embed_dim, hidden_size * 2)
#         self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout)
#
#         final_fusion_dim = 4 * (hidden_size * 2) + embed_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(final_fusion_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_size)
#         )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x_padded, lengths, gpt_feature):
#         packed_x = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         packed_out, _ = self.gru(packed_x)
#         gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)
#         batch_size, max_seq_len, _ = gru_out.size()
#
#         mask = torch.arange(max_seq_len, device=gru_out.device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(
#             1).to(gru_out.device)
#
#         gpt_embed = self.gpt_linear(gpt_feature)
#         static_embed = gpt_embed
#
#         query = self.static_proj(static_embed).unsqueeze(0)
#         key_value = gru_out.transpose(0, 1)
#
#         attn_out, _ = self.cross_attention(query, key_value, key_value, key_padding_mask=mask)
#         attn_out = attn_out.squeeze(0)
#
#         last_valid_idx = lengths - 1
#         last_valid_features = gru_out[torch.arange(batch_size, device=gru_out.device), last_valid_idx, :]
#
#         seq_mask_float = (~mask).unsqueeze(-1).float()
#         sum_hidden = torch.sum(gru_out * seq_mask_float, dim=1)
#         mean_pool = sum_hidden / lengths.unsqueeze(1).float().to(gru_out.device)
#
#         gru_out_masked_for_max = gru_out.masked_fill(mask.unsqueeze(-1), -1e9)
#         max_pool, _ = torch.max(gru_out_masked_for_max, dim=1)
#
#         final_fusion = torch.cat([attn_out, last_valid_features, mean_pool, max_pool, static_embed], dim=-1)
#         out = self.mlp(final_fusion)
#         return out
#
#
# def evaluate_metrics(outputs, labels):
#     prob = torch.sigmoid(outputs).cpu().numpy()
#     labels = labels.cpu().numpy()
#     preds = (prob > 0.5).astype(float)
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
# def evaluate_model(model, loader, criterion, device, return_probs=False):
#     model.eval()
#     total_loss = 0
#     all_outputs, all_labels = [], []
#     with torch.no_grad():
#         for X_batch, y_batch, gpt_batch, lengths in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             gpt_batch, lengths = gpt_batch.to(device), lengths.to(device)
#             outputs = model(X_batch, lengths, gpt_batch)
#
#             loss = criterion(outputs.squeeze(-1), y_batch)
#             total_loss += loss.item() * X_batch.size(0)
#             all_outputs.append(outputs.squeeze(-1).cpu())
#             all_labels.append(y_batch.cpu())
#
#     all_outputs = torch.cat(all_outputs)
#     all_labels = torch.cat(all_labels)
#     avg_loss = total_loss / len(loader.dataset)
#     acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)
#
#     if return_probs:
#         probs = torch.sigmoid(all_outputs)
#         return avg_loss, acc, auc, auprc, f1, probs, all_labels
#     return avg_loss, acc, auc, auprc, f1
#
#
# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss, total_correct, total_samples = 0, 0, 0
#     for X_batch, y_batch, gpt_batch, lengths in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         gpt_batch, lengths = gpt_batch.to(device), lengths.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(X_batch, lengths, gpt_batch)
#
#         loss = criterion(outputs.squeeze(-1), y_batch)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
#         optimizer.step()
#
#         total_loss += loss.item() * X_batch.size(0)
#         prob = torch.sigmoid(outputs.squeeze(-1))
#         predictions = (prob > 0.5).float()
#         total_correct += (predictions == y_batch).sum().item()
#         total_samples += X_batch.size(0)
#     return total_loss / total_samples, total_correct / total_samples
#
#
# seeds = [42, 1024, 2023, 8888, 9999]
# final_seed_results = []
# all_folds_historical_data = []
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = len(input_features)
# hidden_size = Hidden_Size
# num_layers = Num_Layers
# output_size = 1
# learning_rate = Learning_Rate
# num_epochs = Num_Epochs
# num_folds = 5
#
# model_save_dir = "saved_models_ablation"
# os.makedirs(model_save_dir, exist_ok=True)
#
# print(f"========== Starting Ablation Evaluation (No ICU) over {len(seeds)} Seeds ==========\n")
#
# for seed in seeds:
#     print(f"\n==================== Running for SEED: {seed} ====================")
#     set_seed(seed)
#
#     X_cv_raw, X_test_raw, y_cv, y_test, pids_cv, pids_test = train_test_split(
#         X_list_global_raw, y_list_global, pids_global, test_size=0.15, random_state=seed, stratify=y_list_global
#     )
#
#     flat_cv_features = np.vstack(X_cv_raw)
#     scaler = StandardScaler().fit(flat_cv_features)
#
#     X_cv = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_cv_raw]
#     X_test = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_test_raw]
#
#     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
#
#     fold_results = []
#     fold_best_thresholds = []
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
#         print(f"  Starting Fold {fold + 1}/{num_folds} for Seed {seed}...")
#         try:
#             X_train_fold = [X_cv[i] for i in train_idx]
#             y_train_fold = [y_cv[i] for i in train_idx]
#             pids_train_fold = [pids_cv[i] for i in train_idx]
#
#             X_val_fold = [X_cv[i] for i in val_idx]
#             y_val_fold = [y_cv[i] for i in val_idx]
#             pids_val_fold = [pids_cv[i] for i in val_idx]
#
#             num_pos = sum(y_train_fold)
#             num_neg = len(y_train_fold) - num_pos
#             pos_weight = torch.tensor([(num_neg / num_pos) * 0.7], dtype=torch.float32).to(device)
#             criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
#             train_dataset = EHRDataset(X_train_fold, y_train_fold, pids_train_fold, gpt_dict)
#             val_dataset = EHRDataset(X_val_fold, y_val_fold, pids_val_fold, gpt_dict)
#
#             train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, collate_fn=custom_collate_fn)
#             val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)
#
#             model = MultiHeadAttentionModel(input_size, hidden_size, num_layers, output_size,
#                                             gpt_seq_len=14).to(device)
#
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#             scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
#
#             best_val_auc = 0.0
#             best_threshold_for_fold = 0.5
#             best_model_path = os.path.join(model_save_dir,
#                                            f"{datasets}_ablation_no_icu_model_seed_{seed}_fold_{fold + 1}.pth")
#
#             history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_auprc': []}
#
#             for epoch in range(num_epochs):
#                 train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
#
#                 val_loss, val_acc, val_auc, val_auprc, val_f1, val_probs, val_labels = evaluate_model(
#                     model, val_loader, criterion, device, return_probs=True
#                 )
#                 scheduler.step(val_auc)
#
#                 history['train_loss'].append(train_loss)
#                 history['val_loss'].append(val_loss)
#                 history['val_auc'].append(val_auc)
#                 history['val_auprc'].append(val_auprc)
#
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#
#                     best_threshold, _ = find_best_threshold(val_labels.numpy(), val_probs.numpy())
#                     best_threshold_for_fold = best_threshold
#
#                     torch.save({'model_state_dict': model.state_dict(), 'val_auc': best_val_auc}, best_model_path)
#
#             all_folds_historical_data.append({
#                 'seed': seed,
#                 'fold': fold + 1,
#                 'history': history
#             })
#
#             checkpoint = torch.load(best_model_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
#
#             fold_results.append(val_auc)
#             fold_best_thresholds.append(best_threshold_for_fold)
#             print(f"  Fold {fold + 1} Best Val AUC: {val_auc:.4f} | Dynamic Threshold: {best_threshold_for_fold:.4f}")
#
#         except Exception as e:
#             print(f"  Error in Fold {fold + 1}: {e}")
#             fold_results.append(None)
#             fold_best_thresholds.append(None)
#
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     print(f"  Ensembling 5 folds for Seed {seed} on its specific Test Set...")
#     test_dataset = EHRDataset(X_test, y_test, pids_test, gpt_dict)
#     test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)
#
#     all_fold_probs = []
#     final_labels = None
#
#     for fold in range(num_folds):
#         if fold_results[fold] is None:
#             continue
#         model_path = os.path.join(model_save_dir, f"{datasets}_ablation_no_icu_model_seed_{seed}_fold_{fold + 1}.pth")
#         if os.path.exists(model_path):
#             model = MultiHeadAttentionModel(input_size, hidden_size, num_layers, output_size,
#                                             gpt_seq_len=14).to(device)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             _, _, _, _, _, probs, labels = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device,
#                                                           return_probs=True)
#             all_fold_probs.append(probs)
#             if final_labels is None:
#                 final_labels = labels
#
#     ensemble_probs = torch.mean(torch.stack(all_fold_probs), dim=0)
#     final_labels_np = final_labels.numpy()
#     ensemble_probs_np = ensemble_probs.numpy()
#
#     valid_thresholds = [t for t in fold_best_thresholds if t is not None]
#     avg_best_threshold = np.mean(valid_thresholds) if valid_thresholds else 0.5
#     print(f"  --> Applied Ensembled Threshold: {avg_best_threshold:.4f} (instead of default 0.5)")
#
#     ensemble_preds = (ensemble_probs_np > avg_best_threshold).astype(float)
#
#     seed_acc = (ensemble_preds == final_labels_np).mean()
#     seed_auc = roc_auc_score(final_labels_np, ensemble_probs_np)
#     seed_auprc = average_precision_score(final_labels_np, ensemble_probs_np)
#     seed_f1 = f1_score(final_labels_np, ensemble_preds)
#
#     print(
#         f"  Seed {seed} Test Results -> Acc: {seed_acc:.4f}, AUC: {seed_auc:.4f}, AUPRC: {seed_auprc:.4f}, F1: {seed_f1:.4f}")
#
#     final_seed_results.append({
#         'seed': seed,
#         'acc': seed_acc,
#         'auc': seed_auc,
#         'auprc': seed_auprc,
#         'f1': seed_f1
#     })
#
# print("\n========== Rendering Combined Training Histories (Scrolling View) ==========")
# plot_all_histories_combined(all_folds_historical_data)
#
#
# print("\n\n" + "=" * 60)
# print("datasets:" + datasets)
# print("         FINAL ABLATION (NO ICU) EVALUATION RESULTS         ")
# print("=" * 60)
#
# acc_list = [res['acc'] for res in final_seed_results]
# auc_list = [res['auc'] for res in final_seed_results]
# auprc_list = [res['auprc'] for res in final_seed_results]
# f1_list = [res['f1'] for res in final_seed_results]
#
# print(f"Across {len(seeds)} different random seeds (Ensemble evaluated):")
# print(f"Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
# print(f"AUROC    : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
# print(f"AUPRC    : {np.mean(auprc_list):.4f} ± {np.std(auprc_list):.4f}")
# print(f"F1 Score : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
# print("=" * 60)





# # wo LLM
# import gc
# import logging
# import os
# import json
# import random
# import pandas as pd
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# import torch.optim as optim
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
# from config.config import datasets, Hidden_Size, Num_Layers, Learning_Rate, Num_Epochs, Batch_Size, Dropout, \
#     Input_Features
# from torch.optim.lr_scheduler import ReduceLROnPlateau
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# log_dir = 'log'
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'training_ablation_no_llm.log')
# logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')
#
#
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# def find_best_threshold(y_true, y_prob):
#
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
#     f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
#     best_idx = np.argmax(f1_scores)
#
#     if best_idx < len(thresholds):
#         best_threshold = thresholds[best_idx]
#     else:
#         best_threshold = 0.5
#
#     best_f1 = f1_scores[best_idx]
#     return best_threshold, best_f1
#
#
# def plot_all_histories_combined(all_collected_histories):
#
#     num_folds_total = len(all_collected_histories)
#     if num_folds_total == 0:
#         print("No training histories to plot.")
#         return
#
#     fig, axs = plt.subplots(nrows=num_folds_total, ncols=3, figsize=(18, num_folds_total * 3), sharex='col')
#
#     fig.suptitle('Robustness Evaluation (Ablation: No LLM): All 25 Folds Training History', fontsize=18, y=1.0)
#
#     for idx, meta in enumerate(all_collected_histories):
#         hist = meta['history']
#         s_f_title = f"S:{meta['seed']} F:{meta['fold']}"
#         epochs = range(1, len(hist['train_loss']) + 1)
#
#         if num_folds_total == 1:
#             ax_row = axs
#         else:
#             ax_row = axs[idx]
#
#         ax_loss = ax_row[0]
#         ax_loss.plot(epochs, hist['train_loss'], label='Tr Loss', marker='o', markersize=3, alpha=0.7)
#         ax_loss.plot(epochs, hist['val_loss'], label='Val Loss', marker='s', markersize=3, alpha=0.7)
#         ax_loss.set_title(f"{s_f_title} - Loss", fontsize=10)
#         ax_loss.grid(True, linestyle='--', alpha=0.5)
#
#         ax_auc = ax_row[1]
#         ax_auc.plot(epochs, hist['val_auc'], label='Val AUC', color='orange', marker='^', markersize=3)
#         ax_auc.set_title(f"{s_f_title} - AUC", fontsize=10)
#         ax_auc.grid(True, linestyle='--', alpha=0.5)
#
#         ax_auprc = ax_row[2]
#         ax_auprc.plot(epochs, hist['val_auprc'], label='Val AUPRC', color='green', marker='d', markersize=3)
#         ax_auprc.set_title(f"{s_f_title} - AUPRC", fontsize=10)
#         ax_auprc.grid(True, linestyle='--', alpha=0.5)
#
#         if idx == 0:
#             ax_loss.legend(fontsize=9, loc='upper right')
#             ax_auc.legend(fontsize=9, loc='lower right')
#             ax_auprc.legend(fontsize=9, loc='lower right')
#
#         if idx == num_folds_total - 1:
#             ax_loss.set_xlabel('Epochs')
#             ax_auc.set_xlabel('Epochs')
#             ax_auprc.set_xlabel('Epochs')
#
#     plt.tight_layout(rect=[0, 0, 1, 0.98])
#     plt.show()
#
#
# data_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/EHR_{datasets}.csv"
# label_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/label_{datasets}.csv"
# icu_file = f"HiRMD/TimeAttention/datasets/{datasets}/processed/icu_score_{datasets}.csv"
#
# df = pd.read_csv(data_file)
# labels = pd.read_csv(label_file)
#
# if len(df) != len(labels):
#     raise ValueError("Feature data and label data size do not match!")
#
# labels['Outcome'] = labels['Outcome'].astype(int)
# df['Outcome'] = labels['Outcome']
# df = df.sort_values(by=['PatientID', 'RecordTime'])
#
# icu_df = pd.read_csv(icu_file).set_index('PatientID')
# icu_df = icu_df.drop(columns=['APSIIIProb', 'SAPSIIProb', 'apsiii_prob', 'sapsii_prob'], errors='ignore')
# icu_features = list(icu_df.columns)
# icu_scores_dict = icu_df.to_dict(orient='index')
#
# input_features = Input_Features
# label_column = 'Outcome'
#
# X_list_global_raw, y_list_global, pids_global = [], [], []
# grouped = df.groupby('PatientID')
# for patient_id, group in grouped:
#     group = group.sort_values(by='RecordTime')
#     raw_feat = group[input_features].values
#     X_list_global_raw.append(raw_feat)
#     y_val = int(group[label_column].values[-1])
#     y_list_global.append(y_val)
#     pids_global.append(patient_id)
#
# class EHRDataset(Dataset):
#     def __init__(self, X_list, y_list, pids, icu_dict, icu_features):
#         self.X = X_list
#         self.y = y_list
#         self.pids = pids
#         self.icu_dict = icu_dict
#         self.icu_features = icu_features
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         x = self.X[idx]
#         y = torch.tensor(self.y[idx], dtype=torch.float32)
#         pid = self.pids[idx]
#
#         if pid in self.icu_dict:
#             icu_vals = [self.icu_dict[pid][col] for col in self.icu_features]
#             icu_feature = torch.tensor(icu_vals, dtype=torch.float32)
#         else:
#             icu_feature = torch.zeros(len(self.icu_features), dtype=torch.float32)
#
#         return x, y, icu_feature
#
#
# def custom_collate_fn(batch):
#     xs = [item[0] for item in batch]
#     ys = torch.stack([item[1] for item in batch])
#     icus = torch.stack([item[2] for item in batch])
#     lengths = torch.tensor([len(x) for x in xs], dtype=torch.int64)
#     xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
#     return xs_padded, ys, icus, lengths
#
#
# class MultiHeadAttentionModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size,
#                  icu_feature_dim=len(icu_features),
#                  dropout=Dropout, embed_dim=32, num_heads=4):
#         super(MultiHeadAttentionModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
#
#         self.icu_mlp = nn.Sequential(
#             nn.Linear(icu_feature_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, embed_dim)
#         )
#
#         self.static_proj = nn.Linear(embed_dim, hidden_size * 2)
#         self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout)
#
#         final_fusion_dim = 4 * (hidden_size * 2) + embed_dim
#         self.mlp = nn.Sequential(
#             nn.Linear(final_fusion_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, output_size)
#         )
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x_padded, lengths, icu_feature):
#         packed_x = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         packed_out, _ = self.gru(packed_x)
#         gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)
#         batch_size, max_seq_len, _ = gru_out.size()
#
#         mask = torch.arange(max_seq_len, device=gru_out.device).expand(batch_size, max_seq_len) >= lengths.unsqueeze(
#             1).to(gru_out.device)
#
#         icu_embed = self.icu_mlp(icu_feature)
#         static_embed = icu_embed
#
#         query = self.static_proj(static_embed).unsqueeze(0)
#         key_value = gru_out.transpose(0, 1)
#
#         attn_out, _ = self.cross_attention(query, key_value, key_value, key_padding_mask=mask)
#         attn_out = attn_out.squeeze(0)
#
#         last_valid_idx = lengths - 1
#         last_valid_features = gru_out[torch.arange(batch_size, device=gru_out.device), last_valid_idx, :]
#
#         seq_mask_float = (~mask).unsqueeze(-1).float()
#         sum_hidden = torch.sum(gru_out * seq_mask_float, dim=1)
#         mean_pool = sum_hidden / lengths.unsqueeze(1).float().to(gru_out.device)
#
#         gru_out_masked_for_max = gru_out.masked_fill(mask.unsqueeze(-1), -1e9)
#         max_pool, _ = torch.max(gru_out_masked_for_max, dim=1)
#
#         final_fusion = torch.cat([attn_out, last_valid_features, mean_pool, max_pool, static_embed], dim=-1)
#         out = self.mlp(final_fusion)
#         return out
#
#
# def evaluate_metrics(outputs, labels):
#     prob = torch.sigmoid(outputs).cpu().numpy()
#     labels = labels.cpu().numpy()
#     preds = (prob > 0.5).astype(float)
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
# def evaluate_model(model, loader, criterion, device, return_probs=False):
#     model.eval()
#     total_loss = 0
#     all_outputs, all_labels = [], []
#     with torch.no_grad():
#         for X_batch, y_batch, icu_batch, lengths in loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#             icu_batch, lengths = icu_batch.to(device), lengths.to(device)
#             outputs = model(X_batch, lengths, icu_batch)
#
#             loss = criterion(outputs.squeeze(-1), y_batch)
#             total_loss += loss.item() * X_batch.size(0)
#             all_outputs.append(outputs.squeeze(-1).cpu())
#             all_labels.append(y_batch.cpu())
#
#     all_outputs = torch.cat(all_outputs)
#     all_labels = torch.cat(all_labels)
#     avg_loss = total_loss / len(loader.dataset)
#     acc, auc, auprc, f1 = evaluate_metrics(all_outputs, all_labels)
#
#     if return_probs:
#         probs = torch.sigmoid(all_outputs)
#         return avg_loss, acc, auc, auprc, f1, probs, all_labels
#     return avg_loss, acc, auc, auprc, f1
#
#
# def train_model(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_loss, total_correct, total_samples = 0, 0, 0
#     for X_batch, y_batch, icu_batch, lengths in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         icu_batch, lengths = icu_batch.to(device), lengths.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(X_batch, lengths, icu_batch)
#
#         loss = criterion(outputs.squeeze(-1), y_batch)
#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
#         optimizer.step()
#
#         total_loss += loss.item() * X_batch.size(0)
#         prob = torch.sigmoid(outputs.squeeze(-1))
#         predictions = (prob > 0.5).float()
#         total_correct += (predictions == y_batch).sum().item()
#         total_samples += X_batch.size(0)
#     return total_loss / total_samples, total_correct / total_samples
#
# seeds = [42, 1024, 2023, 8888, 9999]
# final_seed_results = []
# all_folds_historical_data = []
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = len(input_features)
# hidden_size = Hidden_Size
# num_layers = Num_Layers
# output_size = 1
# learning_rate = Learning_Rate
# num_epochs = Num_Epochs
# icu_feature_dim = len(icu_features)
# num_folds = 5
#
# model_save_dir = "saved_models_ablation"
# os.makedirs(model_save_dir, exist_ok=True)
#
# print(f"========== Starting Ablation Evaluation over {len(seeds)} Seeds ==========\n")
#
# for seed in seeds:
#     print(f"\n==================== Running for SEED: {seed} ====================")
#     set_seed(seed)
#
#     X_cv_raw, X_test_raw, y_cv, y_test, pids_cv, pids_test = train_test_split(
#         X_list_global_raw, y_list_global, pids_global, test_size=0.15, random_state=seed, stratify=y_list_global
#     )
#
#     flat_cv_features = np.vstack(X_cv_raw)
#     scaler = StandardScaler().fit(flat_cv_features)
#
#     X_cv = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_cv_raw]
#     X_test = [torch.tensor(scaler.transform(x), dtype=torch.float32) for x in X_test_raw]
#
#     skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
#
#     fold_results = []
#     fold_best_thresholds = []
#
#     # 5折交叉验证
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
#         print(f"  Starting Fold {fold + 1}/{num_folds} for Seed {seed}...")
#         try:
#             X_train_fold = [X_cv[i] for i in train_idx]
#             y_train_fold = [y_cv[i] for i in train_idx]
#             pids_train_fold = [pids_cv[i] for i in train_idx]
#
#             X_val_fold = [X_cv[i] for i in val_idx]
#             y_val_fold = [y_cv[i] for i in val_idx]
#             pids_val_fold = [pids_cv[i] for i in val_idx]
#
#             num_pos = sum(y_train_fold)
#             num_neg = len(y_train_fold) - num_pos
#             pos_weight = torch.tensor([(num_neg / num_pos) * 0.7], dtype=torch.float32).to(device)
#             criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#
#             train_dataset = EHRDataset(X_train_fold, y_train_fold, pids_train_fold, icu_scores_dict, icu_features)
#             val_dataset = EHRDataset(X_val_fold, y_val_fold, pids_val_fold, icu_scores_dict, icu_features)
#
#             train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, collate_fn=custom_collate_fn)
#             val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)
#
#             model = MultiHeadAttentionModel(input_size, hidden_size, num_layers, output_size,
#                                             icu_feature_dim=icu_feature_dim).to(device)
#
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#             scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
#
#             best_val_auc = 0.0
#             best_threshold_for_fold = 0.5
#             best_model_path = os.path.join(model_save_dir, f"{datasets}_ablation_model_seed_{seed}_fold_{fold + 1}.pth")
#
#             history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_auprc': []}
#
#             for epoch in range(num_epochs):
#                 train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
#
#                 val_loss, val_acc, val_auc, val_auprc, val_f1, val_probs, val_labels = evaluate_model(
#                     model, val_loader, criterion, device, return_probs=True
#                 )
#                 scheduler.step(val_auc)
#
#                 history['train_loss'].append(train_loss)
#                 history['val_loss'].append(val_loss)
#                 history['val_auc'].append(val_auc)
#                 history['val_auprc'].append(val_auprc)
#
#                 if val_auc > best_val_auc:
#                     best_val_auc = val_auc
#
#                     best_threshold, _ = find_best_threshold(val_labels.numpy(), val_probs.numpy())
#                     best_threshold_for_fold = best_threshold
#
#                     torch.save({'model_state_dict': model.state_dict(), 'val_auc': best_val_auc}, best_model_path)
#
#             all_folds_historical_data.append({
#                 'seed': seed,
#                 'fold': fold + 1,
#                 'history': history
#             })
#
#             checkpoint = torch.load(best_model_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             _, val_acc, val_auc, val_auprc, val_f1 = evaluate_model(model, val_loader, criterion, device)
#
#             fold_results.append(val_auc)
#             fold_best_thresholds.append(best_threshold_for_fold)
#             print(f"  Fold {fold + 1} Best Val AUC: {val_auc:.4f} | Dynamic Threshold: {best_threshold_for_fold:.4f}")
#
#         except Exception as e:
#             print(f"  Error in Fold {fold + 1}: {e}")
#             fold_results.append(None)
#             fold_best_thresholds.append(None)
#
#         torch.cuda.empty_cache()
#         gc.collect()
#
#     print(f"  Ensembling 5 folds for Seed {seed} on its specific Test Set...")
#     test_dataset = EHRDataset(X_test, y_test, pids_test, icu_scores_dict, icu_features)
#     test_loader = DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, collate_fn=custom_collate_fn)
#
#     all_fold_probs = []
#     final_labels = None
#
#     for fold in range(num_folds):
#         if fold_results[fold] is None:
#             continue
#         model_path = os.path.join(model_save_dir, f"{datasets}_ablation_model_seed_{seed}_fold_{fold + 1}.pth")
#         if os.path.exists(model_path):
#             model = MultiHeadAttentionModel(input_size, hidden_size, num_layers, output_size,
#                                             icu_feature_dim=icu_feature_dim).to(device)
#             checkpoint = torch.load(model_path)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             _, _, _, _, _, probs, labels = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device,
#                                                           return_probs=True)
#             all_fold_probs.append(probs)
#             if final_labels is None:
#                 final_labels = labels
#
#     ensemble_probs = torch.mean(torch.stack(all_fold_probs), dim=0)
#     final_labels_np = final_labels.numpy()
#     ensemble_probs_np = ensemble_probs.numpy()
#
#     valid_thresholds = [t for t in fold_best_thresholds if t is not None]
#     avg_best_threshold = np.mean(valid_thresholds) if valid_thresholds else 0.5
#     print(f"  --> Applied Ensembled Threshold: {avg_best_threshold:.4f} (instead of default 0.5)")
#
#     ensemble_preds = (ensemble_probs_np > avg_best_threshold).astype(float)
#
#     seed_acc = (ensemble_preds == final_labels_np).mean()
#     seed_auc = roc_auc_score(final_labels_np, ensemble_probs_np)
#     seed_auprc = average_precision_score(final_labels_np, ensemble_probs_np)
#     seed_f1 = f1_score(final_labels_np, ensemble_preds)
#
#     print(
#         f"  Seed {seed} Test Results -> Acc: {seed_acc:.4f}, AUC: {seed_auc:.4f}, AUPRC: {seed_auprc:.4f}, F1: {seed_f1:.4f}")
#
#     final_seed_results.append({
#         'seed': seed,
#         'acc': seed_acc,
#         'auc': seed_auc,
#         'auprc': seed_auprc,
#         'f1': seed_f1
#     })
#
# print("\n========== Rendering Combined Training Histories (Scrolling View) ==========")
# plot_all_histories_combined(all_folds_historical_data)
#
# print("\n\n" + "=" * 60)
# print("datasets:" + datasets)
# print("         FINAL ABLATION (NO LLM) EVALUATION RESULTS         ")
# print("=" * 60)
#
# acc_list = [res['acc'] for res in final_seed_results]
# auc_list = [res['auc'] for res in final_seed_results]
# auprc_list = [res['auprc'] for res in final_seed_results]
# f1_list = [res['f1'] for res in final_seed_results]
#
# print(f"Across {len(seeds)} different random seeds (Ensemble evaluated):")
# print(f"Accuracy : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
# print(f"AUROC    : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
# print(f"AUPRC    : {np.mean(auprc_list):.4f} ± {np.std(auprc_list):.4f}")
# print(f"F1 Score : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
# print("=" * 60)
