import torch
import torch.nn as nn


class StageNet_Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, conv_size=3, output_dim=1, levels=4,
                 dropconnect=0., dropout=0.5, dropres=0.3,
                 demo_dim=4, gpt_dim=14, icu_dim=35):
        super(StageNet_Fusion, self).__init__()

        # --- StageNet 官方核心参数 ---
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = hidden_dim // levels

        # --- 官方的自定义底层门控和卷积组件 ---
        self.kernel = nn.Linear(int(input_dim + 1), int(hidden_dim * 4 + levels * 2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(int(hidden_dim + 1), int(hidden_dim * 4 + levels * 2))
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

        # --- 为四模态增加的分支 (Demo, GPT, ICU) ---
        self.demo_proj = nn.Linear(demo_dim, hidden_dim)
        self.gpt_proj = nn.Linear(gpt_dim, hidden_dim)
        self.icu_proj = nn.Linear(icu_dim, hidden_dim)

        # --- Late-Fusion 分类器 (去掉原版 Sigmoid，输出 Logits) ---
        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    # 官方的 cumax 机制
    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    # 官方的自定义步进 Cell
    def step(self, inputs, c_last, h_last, interval):
        x_in = inputs
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r').unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels * 2], 'r2l').unsqueeze(2)
        x_out = x_out[:, self.levels * 2:].reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels * 2])
        o_gate = torch.sigmoid(x_out[:, self.levels * 2:self.levels * 3])
        c_in = torch.tanh(x_out[:, self.levels * 3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (
                    i_master_gate - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, x, demo_input, lens, gpt_input, icu_input):
        batch_size, time_step, feature_dim = x.size()
        device = x.device

        # 构造 Dummy 时间间隔 (因为当前 dataloader 没提供，默认为 1)
        time_interval = torch.ones(batch_size, time_step, device=device)

        c_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_out = torch.zeros(batch_size, self.hidden_dim).to(device)

        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size,
                                                                                                  batch_size,
                                                                                                  self.hidden_dim).to(
            device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(device)
        h = []
        origin_h = []

        # 官方的逐步推断逻辑
        for t in range(time_step):
            out, c_out, h_out = self.step(x[:, t, :], c_out, h_out, time_interval[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim + self.levels], -1)
            origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)

            # Re-weighted convolution operation
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.cumsum(local_dis, dim=1)
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)
            local_h = local_h * local_dis.unsqueeze(1)

            # Re-calibrate Progression patterns
            local_theme = torch.mean(local_h, dim=-1)
            local_theme = self.nn_scale(local_theme)
            local_theme = torch.relu(local_theme)
            local_theme = self.nn_rescale(local_theme)
            local_theme = torch.sigmoid(local_theme)

            local_h = self.nn_conv(local_h).squeeze(-1)
            local_h = local_theme * local_h
            h.append(local_h)

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)

        if self.dropres > 0.0:
            origin_h = self.nn_dropres(origin_h)
        rnn_outputs = rnn_outputs + origin_h  # [B, T, hidden_dim]

        # 【核心适配】：提取每个样本真实序列长度的最后一步输出 (因为有 Padding)
        last_step_indices = (lens - 1).clamp(min=0)
        ehr_context = rnn_outputs[torch.arange(batch_size, device=device), last_step_indices, :]  # [B, hidden_dim]

        if self.dropout > 0.0:
            ehr_context = self.nn_dropout(ehr_context)

        # --- 处理额外模态的分支 ---
        demo_embed = torch.tanh(self.demo_proj(demo_input))  # [B, hidden_dim]
        gpt_embed = torch.tanh(self.gpt_proj(gpt_input))  # [B, hidden_dim]
        icu_embed = torch.tanh(self.icu_proj(icu_input))  # [B, hidden_dim]

        # --- 拼接 (Late-Fusion) ---
        fusion = torch.cat([ehr_context, demo_embed, gpt_embed, icu_embed], dim=-1)  # [B, hidden_dim * 4]

        # 输出 Logits
        logits = self.classifier(fusion)
        return logits