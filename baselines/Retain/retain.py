import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RETAIN_Fusion(nn.Module):
    def __init__(
            self,
            input_dim,
            demo_dim=4,
            gpt_dim=14,
            icu_dim=35,
            emb_dim=64,
            alpha_hidden_dim=64,
            beta_hidden_dim=64,
            output_dim=1,
            dropout=0.2
    ):
        super(RETAIN_Fusion, self).__init__()

        # 1. 基础 Embedding 层 (官方代码的 W_emb)
        self.emb = nn.Linear(input_dim, emb_dim)
        self.dropout_emb = nn.Dropout(p=dropout)

        # 2. RETAIN 核心双轨 GRU (处理倒序序列)
        self.alpha_gru = nn.GRU(emb_dim, alpha_hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, beta_hidden_dim, batch_first=True)

        self.alpha_fc = nn.Linear(alpha_hidden_dim, 1)
        self.beta_fc = nn.Linear(beta_hidden_dim, emb_dim)

        # 3. 额外特征的投影分支
        self.demo_proj = nn.Linear(demo_dim, emb_dim)
        self.gpt_proj = nn.Linear(gpt_dim, emb_dim)
        self.icu_proj = nn.Linear(icu_dim, emb_dim)

        # 4. 融合输出层 (官方代码包含 dropout_context)
        self.dropout_context = nn.Dropout(p=dropout)
        fusion_dim = emb_dim * 4
        self.classifier = nn.Linear(fusion_dim, output_dim)

    def _reverse_padded_sequence(self, x, lens):
        """
        核心函数：在 PyTorch 中安全地反转带有 Padding 的变长序列
        模拟官方 Theano 代码中的 [::-1] 倒序操作
        """
        batch_size, max_len, feature_dim = x.size()

        # 构造反转索引
        idx = torch.arange(max_len, device=x.device).unsqueeze(0).expand(batch_size, max_len)
        rev_idx = lens.unsqueeze(1) - 1 - idx

        # 创建有效长度 Mask，防止索引越界
        mask = (rev_idx >= 0)
        rev_idx = rev_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, feature_dim)

        # 提取反转数据，并将无效区域清零
        rev_x = torch.gather(x, 1, rev_idx)
        rev_x = rev_x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return rev_x

    def forward(self, x, demo_input, lens, gpt_input, icu_input):
        batch_size, max_len, _ = x.size()

        # [B, T, emb_dim]
        emb = self.dropout_emb(self.emb(x))

        # --- 第一步：倒序 (Reverse Time) ---
        rev_emb = self._reverse_padded_sequence(emb, lens)

        # 打包倒序后的序列，交给 GRU 提特征
        packed_rev_emb = pack_padded_sequence(rev_emb, lens.cpu(), batch_first=True, enforce_sorted=False)

        rev_alpha_out, _ = self.alpha_gru(packed_rev_emb)
        rev_beta_out, _ = self.beta_gru(packed_rev_emb)

        # 解包
        rev_alpha_unpacked, _ = pad_packed_sequence(rev_alpha_out, batch_first=True, total_length=max_len)
        rev_beta_unpacked, _ = pad_packed_sequence(rev_beta_out, batch_first=True, total_length=max_len)

        # 官方代码特性：隐藏状态乘以 0.5 (防止数值饱和)
        rev_alpha_unpacked = rev_alpha_unpacked * 0.5
        rev_beta_unpacked = rev_beta_unpacked * 0.5

        # --- 第二步：正序回来 (Reverse back to original order) ---
        alpha_unpacked = self._reverse_padded_sequence(rev_alpha_unpacked, lens)
        beta_unpacked = self._reverse_padded_sequence(rev_beta_unpacked, lens)

        # 计算 Alpha (时间步注意力)
        e_alpha = self.alpha_fc(alpha_unpacked).squeeze(-1)  # [B, T]
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lens.unsqueeze(1)
        e_alpha = e_alpha.masked_fill(~mask, -1e9)
        alpha = torch.softmax(e_alpha, dim=1).unsqueeze(-1)  # [B, T, 1]

        # 计算 Beta (特征注意力)
        beta = torch.tanh(self.beta_fc(beta_unpacked))  # [B, T, emb_dim]

        # 计算 Context Vector
        ehr_context = torch.sum(alpha * beta * emb, dim=1)  # [B, emb_dim]
        ehr_context = self.dropout_context(ehr_context)

        # --- 额外模态融合分支 ---
        demo_embed = torch.tanh(self.demo_proj(demo_input))  # [B, emb_dim]
        gpt_embed = torch.tanh(self.gpt_proj(gpt_input))  # [B, emb_dim]
        icu_embed = torch.tanh(self.icu_proj(icu_input))  # [B, emb_dim]

        # 拼接 (Late-Fusion)
        fusion = torch.cat([ehr_context, demo_embed, gpt_embed, icu_embed], dim=-1)

        # 输出 Logits
        logits = self.classifier(fusion)
        return logits