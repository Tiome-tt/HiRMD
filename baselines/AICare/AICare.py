import copy
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class FinalAttentionQKV(nn.Module):
    """
    对 feature-level hidden states 做 attention 聚合
    输入:  [B, N, H]
    输出:  [B, H]
    """
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='mul', dropout=0.3):
        super().__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1,))
        self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [B, N, H]
        batch_size, num_tokens, hidden_dim = x.size()

        q = self.W_q(torch.mean(x, dim=1))   # [B, H]
        k = self.W_k(x)                      # [B, N, H]
        v = self.W_v(x)                      # [B, N, H]

        if self.attention_type == 'add':
            q = q.view(batch_size, 1, self.attention_hidden_dim)
            h = self.tanh(q + k + self.b_in)
            e = self.W_out(h).view(batch_size, num_tokens)

        elif self.attention_type == 'mul':
            q = q.view(batch_size, self.attention_hidden_dim, 1)
            e = torch.matmul(k, q).squeeze(-1)   # [B, N]

        elif self.attention_type == 'concat':
            q = q.unsqueeze(1).repeat(1, num_tokens, 1)
            c = torch.cat((q, k), dim=-1)
            h = self.tanh(torch.matmul(c, self.Wh))
            e = (torch.matmul(h, self.Wa) + self.ba).view(batch_size, num_tokens)

        else:
            raise ValueError(f"Unsupported attention_type: {self.attention_type}")

        a = self.softmax(e)                  # [B, N]
        a = self.dropout(a)
        context = torch.matmul(a.unsqueeze(1), v).squeeze(1)  # [B, H]

        return context, a


class AICare(nn.Module):
    """
    适配你当前四文件输入的 AICare 模型

    输入:
        x:           [B, T, F_ehr]
        demo_input:  [B, 4]           -> Sex, Age, Height, Weight
        lens:        [B]
        gpt_input:   [B, gpt_dim]
        icu_input:   [B, icu_dim]

    输出:
        logits:      [B, 1]
    """
    def __init__(
        self,
        input_dim,
        demo_dim=4,
        gpt_dim=14,
        icu_dim=35,
        hidden_dim=32,
        output_dim=1,
        keep_prob=0.5,
        rnn_type='GRU'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.demo_dim = demo_dim
        self.gpt_dim = gpt_dim
        self.icu_dim = icu_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.rnn_type = rnn_type.upper()

        if self.rnn_type == 'GRU':
            base_rnn = nn.GRU(1, hidden_dim, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'RNN':
            base_rnn = nn.RNN(1, hidden_dim, batch_first=True, bidirectional=True)
        else:
            raise ValueError("rnn_type must be 'GRU' or 'RNN'")

        # 每个 EHR feature 一个独立 RNN
        self.feature_rnns = nn.ModuleList([copy.deepcopy(base_rnn) for _ in range(input_dim)])

        # 双向 RNN 输出维度
        self.feature_hidden_dim = hidden_dim * 2

        # 三类额外输入投影
        self.demo_proj = nn.Linear(demo_dim, self.feature_hidden_dim)
        self.gpt_proj = nn.Linear(gpt_dim, self.feature_hidden_dim)
        self.icu_proj = nn.Linear(icu_dim, self.feature_hidden_dim)

        # feature-level attention
        self.final_attention = FinalAttentionQKV(
            attention_input_dim=self.feature_hidden_dim,
            attention_hidden_dim=self.feature_hidden_dim,
            attention_type='mul',
            dropout=1 - keep_prob
        )

        self.dropout = nn.Dropout(p=1 - keep_prob)
        self.tanh = nn.Tanh()

        # 融合输出层
        fusion_dim = self.feature_hidden_dim * 4
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, self.feature_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=1 - keep_prob),
            nn.Linear(self.feature_hidden_dim, output_dim)
        )

    def _encode_single_feature(self, x_feature, lens, rnn):
        # x_feature: [B, T]
        packed = pack_padded_sequence(
            x_feature.unsqueeze(-1),     # [B, T, 1]
            lengths=lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, h_n = rnn(packed)            # [2, B, H]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)   # [B, 2H]
        return h

    def forward(self, x, demo_input, lens, gpt_input, icu_input):
        """
        x:          [B, T, F]
        demo_input: [B, 4]
        lens:       [B]
        gpt_input:  [B, gpt_dim]
        icu_input:  [B, icu_dim]
        """
        batch_size, seq_len, feature_dim = x.size()
        assert feature_dim == self.input_dim, f"Expected input_dim={self.input_dim}, got {feature_dim}"

        # 1) 每个 EHR 特征单独编码
        feature_embeddings = []
        for i in range(feature_dim):
            h_i = self._encode_single_feature(x[:, :, i], lens, self.feature_rnns[i])  # [B, 2H]
            feature_embeddings.append(h_i.unsqueeze(1))

        feature_embeddings = torch.cat(feature_embeddings, dim=1)   # [B, F, 2H]

        # 2) attention 聚合 EHR 序列特征
        ehr_context, attn_weights = self.final_attention(self.dropout(feature_embeddings))  # [B, 2H]

        # 3) 额外分支
        demo_embed = self.tanh(self.demo_proj(demo_input))   # [B, 2H]
        gpt_embed = self.tanh(self.gpt_proj(gpt_input))      # [B, 2H]
        icu_embed = self.tanh(self.icu_proj(icu_input))      # [B, 2H]

        # 4) 融合
        fusion = torch.cat([ehr_context, demo_embed, gpt_embed, icu_embed], dim=-1)  # [B, 8H]
        logits = self.output_layer(fusion)   # [B, 1]

        return logits


if __name__ == "__main__":
    # quick test
    B, T, F = 8, 42, 20
    x = torch.randn(B, T, F)
    demo = torch.randn(B, 4)
    lens = torch.tensor([42, 40, 39, 35, 30, 28, 21, 12], dtype=torch.long)
    gpt = torch.randn(B, 14)
    icu = torch.randn(B, 35)

    model = AICare(input_dim=F, demo_dim=4, gpt_dim=14, icu_dim=35)
    logits = model(x, demo, lens, gpt, icu)
    print(logits.shape)  # [8, 1]