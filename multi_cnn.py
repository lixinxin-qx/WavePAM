import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# 1. 基础组件：门控空间-通道聚合 (Intra-scale)
# ====================================================================

class ChannelAttentionBlock(nn.Module):
    """纯通道注意力（SE风格）"""

    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // reduction, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, L, C]
        residual = x
        x_in = x.transpose(1, 2)  # [B, C, L]

        y = self.avg_pool(x_in).squeeze(-1)  # [B, C]
        weights = self.channel_attention(y)  # [B, C]
        out = x * weights.unsqueeze(1)  # [B, L, C]

        return self.norm(residual + self.dropout(out))


# ====================================================================
# 2. 核心：微分引导的交互式动态融合 (Interaction-based Fusion)
# ====================================================================

class DiffGuidedInteractiveFusion(nn.Module):
    """
    微分引导的交互式融合模块（简化版）

    公式：
        D = LayerNorm(D_raw)                   -- 微分归一化
        Q = W_q · D                            -- 微分投影
        V_i = W_v · F̂_i                       -- 尺度投影
        α_i = Softmax_i(W_g · tanh(Q ⊙ V_i))  -- 交互式权重
        Output = Σ α_i · F̂_i                  -- 加权融合
    """

    def __init__(self, d_model, n_scales, d_hidden=None, dropout=0.1):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model // 2

        # A. 微分归一化（仅做数值稳定）
        self.diff_norm = nn.LayerNorm(d_model)

        # B. 共享空间投影
        self.W_q = nn.Linear(d_model, self.d_hidden)  # 微分 → Query
        self.W_v = nn.Linear(d_model, self.d_hidden)  # 尺度 → Value

        # C. 交互权重生成器
        self.W_g = nn.Linear(self.d_hidden, 1)

        # D. 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def forward(self, scale_features, diff_feature_raw):
        """
        Args:
            scale_features: List of [B, L, C], 长度为 n_scales
            diff_feature_raw: [B, L, C], 原始一阶差分特征
        Returns:
            fused_output: [B, L, C]
        """
        # Step 1: 微分归一化
        D = self.diff_norm(diff_feature_raw)  # [B, L, C]

        # Step 2: 投影到共享空间
        Q = self.W_q(D)  # [B, L, d_hidden]

        # Step 3: 计算各尺度的交互权重
        interaction_scores = []
        for i in range(self.n_scales):
            V_i = self.W_v(scale_features[i])  # [B, L, d_hidden]
            interaction = torch.tanh(Q * V_i)  # [B, L, d_hidden]
            score_i = self.W_g(interaction)    # [B, L, 1]
            interaction_scores.append(score_i)

        # Step 4: Softmax 归一化
        scores = torch.cat(interaction_scores, dim=-1)  # [B, L, n_scales]
        scale_weights = F.softmax(scores, dim=-1)       # [B, L, n_scales]

        # Step 5: 加权融合
        stacked_scales = torch.stack(scale_features, dim=2)  # [B, L, n_scales, C]
        scale_weights = scale_weights.unsqueeze(-1)          # [B, L, n_scales, 1]
        fused_content = (scale_weights * stacked_scales).sum(dim=2)  # [B, L, C]

        return self.out_proj(fused_content)


# ====================================================================
# 3. 完整模块：Inception + 变量交互 + 交互式微分融合
# ====================================================================

class Inception_Conv_Attention_Block(nn.Module):
    def __init__(self, d_model, kernel_list=[3, 7, 15, 20], use_diff_mask=True):
        super().__init__()
        self.d_model = d_model
        self.n_scales = len(kernel_list)
        self.use_diff_mask = use_diff_mask

        # 1. Pointwise Conv: 变量间线性重组
        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )

        # 2. Depthwise Multi-scale Conv: 时间尺度特征提取
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=(k - 1) // 2,
                          groups=d_model, padding_mode='zeros'),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True)
            ) for k in kernel_list
        ])

        # 3. Differential Conv (Fixed): 物理先验提取
        self.diff_conv = nn.Conv1d(d_model, d_model, kernel_size=2, padding=0, bias=False)
        with torch.no_grad():
            self.diff_conv.weight.fill_(0)
            self.diff_conv.weight[:, :, 0] = -1
            self.diff_conv.weight[:, :, 1] = 1
        self.diff_conv.weight.requires_grad = False

        # 4. Intra-Scale Module: 尺度内变量交互
        self.intra_scale_modules = nn.ModuleList([
            ChannelAttentionBlock(d_model)
            for _ in range(self.n_scales)
        ])

        # 5. Fusion: 微分引导的交互式融合
        self.diff_fusion = DiffGuidedInteractiveFusion(d_model, self.n_scales)

        self.final_norm = nn.LayerNorm(d_model)
        self.residual_proj = nn.Identity()

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.size()
        residual = x
        x_in = x.transpose(1, 2)  # [B, C, L]

        # A. 基础变换
        x_pw = self.pointwise_conv(x_in)

        # B. 多尺度提取
        scale_features = []
        for conv in self.conv_layers:
            out = conv(x_pw)
            if out.shape[2] != L:
                if out.shape[2] > L:
                    out = out[:, :, :L]
                else:
                    out = F.pad(out, (0, L - out.shape[2]))
            scale_features.append(out.transpose(1, 2))  # List of [B, L, C]

        # C. 微分特征提取
        diff_feat = torch.zeros_like(x)
        if self.use_diff_mask:
            diff_out = self.diff_conv(x_in)       # [B, C, L-1]
            diff_out = F.pad(diff_out, (0, 1))    # [B, C, L]
            diff_feat = diff_out.transpose(1, 2)  # [B, L, C]

        # D. 尺度内变量增强
        enhanced_scale_features = []
        for feat, intra_mod in zip(scale_features, self.intra_scale_modules):
            enhanced_scale_features.append(intra_mod(feat))

        # E. 微分引导交互式融合
        output = self.diff_fusion(enhanced_scale_features, diff_feat)

        # F. 残差输出
        return self.final_norm(output + self.residual_proj(residual))
