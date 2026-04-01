import torch
import torch.nn as nn


class iTransformer_Fusion(nn.Module):
    """
    适配四种模态输入的 iTransformer (ICLR 2024) 扩展模型 (Late-Fusion 版)
    保留了 iTransformer 最核心的 Inverted Embedding 和 变量级自注意力机制。
    """

    def __init__(
            self,
            input_dim,
            max_seq_len=100,  # iTransformer 需要固定长度投影，超过截断，不足补0
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_ff=256,
            demo_dim=4,
            gpt_dim=14,
            icu_dim=35,
            output_dim=1,
            dropout=0.1
    ):
        super(iTransformer_Fusion, self).__init__()
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim

        # 1. 核心创新：Inverted Embedding (倒置嵌入)
        # 将一个特征在所有时间点的历史记录，映射为 d_model 维度的 Token
        self.inverted_embedding = nn.Linear(max_seq_len, d_model)
        self.dropout_emb = nn.Dropout(dropout)

        # 2. iTransformer Encoder (跨变量注意力)
        # 官方论文明确指出：倒置后直接使用最原生的 Transformer Encoder 即可
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # 3. EHR 上下文提取器
        self.ehr_proj = nn.Linear(input_dim * d_model, d_model)

        # 4. 额外特征的投影分支 (Late-Fusion 准备)
        self.demo_proj = nn.Linear(demo_dim, d_model)
        self.gpt_proj = nn.Linear(gpt_dim, d_model)
        self.icu_proj = nn.Linear(icu_dim, d_model)

        # 5. 融合输出层
        fusion_dim = d_model * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x, demo_input, lens, gpt_input, icu_input):
        B, T, F = x.size()

        # --- 适配 iTransformer 的固定长度要求 ---
        # 如果当前序列短于 max_seq_len，在时间维度(dim=1)补 0
        if T < self.max_seq_len:
            pad_len = self.max_seq_len - T
            padding = torch.zeros(B, pad_len, F, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        # 如果长于 max_seq_len，截取最近的记录
        elif T > self.max_seq_len:
            x_padded = x[:, -self.max_seq_len:, :]
        else:
            x_padded = x

        # --- 1. Inverted Embedding ---
        # [B, T_max, F] -> [B, F, T_max]  (这是 iTransformer 最灵魂的一步转置)
        x_inv = x_padded.permute(0, 2, 1)

        # [B, F, T_max] -> [B, F, d_model]
        enc_out = self.dropout_emb(self.inverted_embedding(x_inv))

        # --- 2. Transformer Encoder ---
        # 此时的 Sequence Length 变成了 F (变量数)
        # 注意力机制是在探讨：“心率”、“血压”、“体温”这几个变量之间的关系
        enc_out = self.encoder(enc_out)  # [B, F, d_model]

        # --- 3. 提取 EHR Context ---
        # 根据官方代码对分类任务的处理，展平后过线性层
        enc_out = enc_out.reshape(B, -1)  # Flatten: [B, F * d_model]
        ehr_context = torch.relu(self.ehr_proj(enc_out))  # [B, d_model]

        # --- 4. 额外模态融合分支 ---
        demo_embed = torch.tanh(self.demo_proj(demo_input))  # [B, d_model]
        gpt_embed = torch.tanh(self.gpt_proj(gpt_input))  # [B, d_model]
        icu_embed = torch.tanh(self.icu_proj(icu_input))  # [B, d_model]

        # --- 5. 拼接 (Late-Fusion) ---
        fusion = torch.cat([ehr_context, demo_embed, gpt_embed, icu_embed], dim=-1)

        # 输出 Logits
        logits = self.classifier(fusion)
        return logits