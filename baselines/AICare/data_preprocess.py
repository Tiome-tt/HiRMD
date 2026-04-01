import pandas as pd
import numpy as np
import torch


def read_partial_csv_by_header(file_path, needed_columns):
    """
    手工读取 CSV，只取 needed_columns 对应的列。
    这样可以绕过文件后半部分坏引号、坏逗号的问题。
    假设表头本身是正常的。
    """
    rows = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        header_line = f.readline().strip('\n').strip('\r')
        header_cols = [x.strip() for x in header_line.split(',')]

        missing = [c for c in needed_columns if c not in header_cols]
        if missing:
            raise ValueError(f'表头缺少这些列: {missing}')

        col_idx = {col: header_cols.index(col) for col in needed_columns}
        max_idx = max(col_idx.values())

        for line_num, line in enumerate(f, start=2):
            line = line.strip('\n').strip('\r')
            if not line:
                continue

            # 只切到需要覆盖的最大列位置
            parts = line.split(',', max_idx)

            # 如果列数不足，跳过
            if len(parts) <= max_idx:
                continue

            row = {}
            for col in needed_columns:
                idx = col_idx[col]
                row[col] = parts[idx].strip() if idx < len(parts) else ''
            rows.append(row)

    df = pd.DataFrame(rows, columns=needed_columns)
    return df


def preprocess_patient_sequence_for_aicare(
    file_path,
    save_path='aicare_patient_seq.pt'
):
    """
    按 patientid 聚合：
    - 每个 patientid = 一个样本
    - 每次住院记录 = 一个时间步
    - 标签规则：只要任意一次 outcome=1，则该患者标签=1，否则=0
    """

    # 原始生理特征
    base_time_features = [
        'glucose',
        'hematocrit',
        'hemoglobin',
        'ph',
        'temperature',
        'plateletmin',
        'plateletmax',
        'wbcmin',
        'wbcmax',
        'heartratemean',
        'sbpmean',
        'dbpmean',
        'mbpmean',
        'respiratoryratemean',
        'spo2mean'
    ]

    # ICU 评分相关特征
    icu_score_features = [
        'apsiii',
        'apsiii_prob',
        'apsiii_hr_score',
        'apsiii_meanbp_score',
        'apsiii_temp_score',
        'apsiii_resprate_score',
        'apsiii_pao2_aado2_score',
        'apsiii_hematocrit_score',
        'apsiii_wbc_score',
        'apsiii_creatinine_score',
        'apsiii_uo_score',
        'apsiii_bun_score',
        'apsiii_sodium_score',
        'apsiii_albumin_score',
        'apsiii_bilirubin_score',
        'apsiii_glucose_score',
        'apsiii_acidbase_score',
        'apsiii_gcs_score',
        'sapsii',
        'sapsii_prob',
        'sapsii_age_score',
        'sapsii_hr_score',
        'sapsii_sysbp_score',
        'sapsii_temp_score',
        'sapsii_pao2fio2_score',
        'sapsii_uo_score',
        'sapsii_bun_score',
        'sapsii_wbc_score',
        'sapsii_potassium_score',
        'sapsii_sodium_score',
        'sapsii_bicarbonate_score',
        'sapsii_bilirubin_score',
        'sapsii_gcs_score',
        'sapsii_comorbidity_score',
        'sapsii_admissiontype_score'
    ]

    # 最终作为序列输入的特征
    time_features = base_time_features + icu_score_features

    # 静态特征
    demo_features = ['age', 'weight', 'height', 'sex']
    label_col = 'outcome'

    needed_columns = ['patientid', 'icustayid', 'icuintime'] + demo_features + [label_col] + time_features
    needed_columns = list(dict.fromkeys(needed_columns))  # 去重并保持顺序

    # 手工读取需要的列
    df = read_partial_csv_by_header(file_path, needed_columns)

    # 清理缺失值标记
    df = df.replace(['', ' ', 'NA', 'N/A', 'nan', 'None', '#####'], np.nan)

    # 时间列
    df['icuintime'] = pd.to_datetime(df['icuintime'], errors='coerce')

    # 数值列转换
    numeric_cols = demo_features + [label_col] + time_features
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 去掉 patientid 缺失
    df = df.dropna(subset=['patientid']).copy()

    df['patientid'] = df['patientid'].astype(str)
    df['icustayid'] = df['icustayid'].astype(str)

    grouped = df.groupby('patientid', sort=False)

    all_sequences = []
    all_demos = []
    all_lens = []
    all_labels = []
    all_patientids = []

    for patientid, group in grouped:
        group = group.sort_values('icuintime', na_position='last').copy()

        seq = group[time_features].fillna(0.0).values.astype(np.float32)
        if len(seq) == 0:
            continue

        demo = []
        for col in demo_features:
            vals = group[col].dropna()
            demo.append(float(vals.iloc[0]) if len(vals) > 0 else 0.0)
        demo = np.array(demo, dtype=np.float32)

        label_series = pd.to_numeric(group[label_col], errors='coerce').fillna(0.0)
        label = np.float32(1.0 if (label_series == 1).any() else 0.0)

        all_sequences.append(seq)
        all_demos.append(demo)
        all_lens.append(len(seq))
        all_labels.append(label)
        all_patientids.append(patientid)

    if len(all_sequences) == 0:
        raise ValueError('没有构造出任何患者序列，请检查数据内容。')

    max_len = max(all_lens)
    feature_dim = len(time_features)

    padded_sequences = []
    for seq in all_sequences:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, feature_dim), dtype=np.float32)
            seq = np.vstack([seq, pad])
        padded_sequences.append(seq)

    input_tensor = torch.tensor(np.stack(padded_sequences), dtype=torch.float32)
    demo_tensor = torch.tensor(np.stack(all_demos), dtype=torch.float32)
    lens_tensor = torch.tensor(all_lens, dtype=torch.long)
    outcome_tensor = torch.tensor(np.array(all_labels).reshape(-1, 1), dtype=torch.float32)

    save_dict = {
        'input_tensor': input_tensor,
        'demo_tensor': demo_tensor,
        'lens_tensor': lens_tensor,
        'outcome_tensor': outcome_tensor,
        'patientids': all_patientids,
        'time_features': time_features,
        'demo_features': demo_features,
        'label_col': label_col
    }

    torch.save(save_dict, save_path)

    print(f'处理完成，保存到: {save_path}')
    print(f'患者数: {len(all_patientids)}')
    print(f'最大序列长度: {max_len}')
    print(f'feature_dim: {feature_dim}')
    print(f'input_tensor shape: {input_tensor.shape}')
    print(f'demo_tensor shape: {demo_tensor.shape}')
    print(f'lens_tensor shape: {lens_tensor.shape}')
    print(f'outcome_tensor shape: {outcome_tensor.shape}')
    print(f'阳性患者数: {int(outcome_tensor.sum().item())}')

    return save_dict


if __name__ == '__main__':
    file_path = '/home/user1/MXY/TimeAttention/datasets/mimic-iii/mimiciii_format.csv'
    save_path = '/home/user1/MXY/TimeAttention/baselines/AICare/data/mimic3_aicare_input.pt'

    preprocess_patient_sequence_for_aicare(
        file_path=file_path,
        save_path=save_path
    )