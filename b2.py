import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from model.b1 import WaveletTransform  # ✅ 复用 Branch1 的小波变换


# ==================== Branch2 专用小波变换 ====================
class FrequencyAwareWaveletTransform(nn.Module):
    """
    频率感知的小波变换 - Branch2 使用
    保持各尺度分离，便于层次化处理
    """

    def __init__(self, wavelet='db4', mode='symmetric', level=4):
        super(FrequencyAwareWaveletTransform, self).__init__()
        # ✅ 复用 Branch1 的 WaveletTransform
        self.base_transform = WaveletTransform(wavelet, mode, level)
        self.wavelet = wavelet
        self.mode = mode
        self.level = level

    def forward(self, x):
        """
        前向小波变换（保持各尺度分离）

        Args:
            x: [B, L, C] - 时域序列

        Returns:
            List of [B, C, scale_length] - 每个尺度的系数
        """
        batch_size, seq_len, n_channels = x.shape
        scales_coeffs = [[] for _ in range(self.level + 1)]

        for i in range(batch_size):
            batch_scales = [[] for _ in range(self.level + 1)]

            for j in range(n_channels):
                signal = x[i, :, j].detach().cpu().numpy()
                coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=self.level)

                for scale_idx, coeff in enumerate(coeffs):
                    batch_scales[scale_idx].append(torch.from_numpy(coeff).float())

            for scale_idx in range(self.level + 1):
                scales_coeffs[scale_idx].append(torch.stack(batch_scales[scale_idx]))

        result = [torch.stack(scale).to(x.device) for scale in scales_coeffs]
        return result

    def inverse(self, scales_coeffs, original_length):
        """
        逆小波变换

        Args:
            scales_coeffs: List of [B, C, scale_length]
            original_length: int

        Returns:
            [B, L, C] - 时域序列
        """
        batch_size = scales_coeffs[0].shape[0]
        n_channels = scales_coeffs[0].shape[1]

        reconstructed = []
        for i in range(batch_size):
            batch_recon = []

            for j in range(n_channels):
                coeffs_list = [scales_coeffs[s][i, j].detach().cpu().numpy()
                               for s in range(len(scales_coeffs))]

                recon = pywt.waverec(coeffs_list, self.wavelet, mode=self.mode)

                if len(recon) > original_length:
                    recon = recon[:original_length]
                elif len(recon) < original_length:
                    recon = torch.nn.functional.pad(
                        torch.from_numpy(recon).float(),
                        (0, original_length - len(recon))
                    )
                    recon = recon.numpy()

                batch_recon.append(torch.from_numpy(recon).float())

            reconstructed.append(torch.stack(batch_recon))

        result = torch.stack(reconstructed).to(scales_coeffs[0].device)
        result = result.permute(0, 2, 1)  # [B, L, C]
        return result


# ==================== 层次化通道注意力 ====================
class HierarchicalChannelAttention(nn.Module):
    """
    层次化通道注意力：局部聚合 → 全局精炼
    简化版：局部注意力输出直接送入全局注意力
    """

    def __init__(self, scale_length, n_heads=4, dropout=0.1, local_k=5):
        super(HierarchicalChannelAttention, self).__init__()

        self.scale_length = scale_length
        self.n_heads = n_heads
        self.local_k = local_k

        # 确保 scale_length 能被 n_heads 整除
        self.d_k = scale_length // n_heads
        if scale_length % n_heads != 0:
            for nh in [4, 2, 1]:
                if scale_length % nh == 0:
                    self.n_heads = nh
                    self.d_k = scale_length // nh
                    break
            else:
                self.n_heads = 1
                self.d_k = scale_length

        # Stage 1: 局部注意力
        self.local_q_linear = nn.Linear(scale_length, scale_length)
        self.local_k_linear = nn.Linear(scale_length, scale_length)
        self.local_v_linear = nn.Linear(scale_length, scale_length)
        self.local_out_linear = nn.Linear(scale_length, scale_length)

        # Stage 2: 全局注意力
        self.global_q_linear = nn.Linear(scale_length, scale_length)
        self.global_k_linear = nn.Linear(scale_length, scale_length)
        self.global_v_linear = nn.Linear(scale_length, scale_length)
        self.global_out_linear = nn.Linear(scale_length, scale_length)

        self.norm = nn.LayerNorm(scale_length)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([self.d_k])))

    def _reshape_for_attention(self, x, linear_q, linear_k, linear_v):
        """重塑为多头注意力格式"""
        B, C, F = x.shape

        Q = linear_q(x)
        K = linear_k(x)
        V = linear_v(x)

        Q = Q.view(B, C, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(B, C, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(B, C, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        return Q, K, V

    def _reshape_from_attention(self, x):
        """从多头注意力格式恢复"""
        B, n_heads, C, d_k = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, C, -1)

    def _compute_local_attention(self, Q, K, V):
        """Stage 1: 局部注意力 - Top-k 稀疏"""
        B, n_heads, C, d_k = Q.shape

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        k = min(self.local_k, C)
        _, topk_indices = torch.topk(scores, k=k, dim=-1)

        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, False)
        scores = scores.masked_fill(mask, float('-inf'))

        local_attn = F.softmax(scores, dim=-1)
        local_attn = torch.nan_to_num(local_attn, nan=0.0)
        local_attn = self.dropout(local_attn)

        output = torch.matmul(local_attn, V)
        return output, local_attn

    def _compute_global_attention(self, Q, K, V):
        """Stage 2: 全局注意力 - 密集"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        global_attn = F.softmax(scores, dim=-1)
        global_attn = self.dropout(global_attn)

        output = torch.matmul(global_attn, V)
        return output, global_attn

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, C, F]

        Returns:
            [B, C, F], attn_info
        """
        B, C, F = x.shape

        # Stage 1: 局部注意力
        Q1, K1, V1 = self._reshape_for_attention(
            x, self.local_q_linear, self.local_k_linear, self.local_v_linear
        )
        local_output, local_attn = self._compute_local_attention(Q1, K1, V1)
        local_output = self._reshape_from_attention(local_output)
        local_output = self.local_out_linear(local_output)

        # Stage 2: 全局注意力
        Q2, K2, V2 = self._reshape_for_attention(
            local_output, self.global_q_linear, self.global_k_linear, self.global_v_linear
        )
        global_output, global_attn = self._compute_global_attention(Q2, K2, V2)
        global_output = self._reshape_from_attention(global_output)
        global_output = self.global_out_linear(global_output)

        # 残差 + 归一化
        output = self.norm(x + global_output)

        attn_info = {
            'local_attn': local_attn,
            'global_attn': global_attn
        }

        return output, attn_info


# ==================== 多尺度通道注意力 ====================
class MultiScaleChannelAttention(nn.Module):
    """
    多尺度通道注意力模块
    对每个频带独立进行层次化通道注意力建模
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1, local_k=5, level=4):
        super(MultiScaleChannelAttention, self).__init__()

        self.level = level
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.local_k = local_k

        self.scale_attentions = nn.ModuleList()
        self.scale_projections = nn.ModuleList()
        self.scale_inv_projections = nn.ModuleList()
        self.initialized = False

    def _lazy_init(self, scales_coeffs):
        """延迟初始化"""
        if self.initialized:
            return

        device = scales_coeffs[0].device

        for scale_idx, scale_coeff in enumerate(scales_coeffs):
            scale_length = scale_coeff.shape[2]

            self.scale_projections.append(
                nn.Linear(scale_length, self.d_model).to(device)
            )

            self.scale_inv_projections.append(
                nn.Linear(self.d_model, scale_length).to(device)
            )

            self.scale_attentions.append(
                HierarchicalChannelAttention(
                    scale_length=self.d_model,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    local_k=self.local_k
                ).to(device)
            )

        self.initialized = True

    def forward(self, scales_coeffs):
        """
        前向传播

        Args:
            scales_coeffs: List of [B, C, scale_length]

        Returns:
            List of [B, C, scale_length], all_attn_weights
        """
        self._lazy_init(scales_coeffs)

        processed_scales = []
        all_attn_weights = []

        for scale_idx, scale_coeff in enumerate(scales_coeffs):
            # 1. 投影到 d_model
            x = self.scale_projections[scale_idx](scale_coeff)

            # 2. 层次化通道注意力
            x_attn, attn_info = self.scale_attentions[scale_idx](x)

            # 3. 投影回原始维度
            x_out = self.scale_inv_projections[scale_idx](x_attn)

            # 4. 残差连接
            x_out = x_out + scale_coeff

            processed_scales.append(x_out)
            all_attn_weights.append(attn_info)

        return processed_scales, all_attn_weights


# ==================== Token Embedding ====================
class TokenEmbeddingBranch2(nn.Module):
    """Branch2 专用的 Token Embedding"""

    def __init__(self, in_dim, d_model, n_window=100, n_layers=1,
                 branch_layers=['multiscale_conv1d', 'inter'],
                 group_embedding='False', match_dimension='first', kernel_size=[1],
                 init_type='normal', gain=0.02, dropout=0.1,
                 wavelet='db4', wavelet_level=4, local_k=5):
        super(TokenEmbeddingBranch2, self).__init__()

        self.window_size = n_window
        self.d_model = d_model
        self.branch_layers = branch_layers
        self.wavelet_level = wavelet_level
        self.match_dimension = match_dimension
        self.group_embedding = group_embedding

        # ✅ 使用 Branch2 专用的小波变换
        self.wavelet_transform = FrequencyAwareWaveletTransform(
            wavelet=wavelet, level=wavelet_level
        )

        self.encoder_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        self.residual_projection = None
        self.residual_alpha = nn.Parameter(torch.tensor([0.5]))

        current_channel_dim = in_dim

        for i, e_layer in enumerate(branch_layers):
            # 维度计算逻辑
            if self.match_dimension == 'none':
                updated_in_dim = current_channel_dim
                extended_dim = current_channel_dim
            elif (i == 0 and self.match_dimension == 'first') or (len(branch_layers) < 2):
                updated_in_dim = current_channel_dim
                extended_dim = d_model
            elif (i == 0) and (not self.match_dimension == 'first'):
                updated_in_dim = current_channel_dim
                extended_dim = current_channel_dim
            elif (i + 1 < len(branch_layers)) and (self.match_dimension == 'middle'):
                updated_in_dim = current_channel_dim
                extended_dim = d_model
            elif i + 1 == len(branch_layers):
                updated_in_dim = current_channel_dim
                extended_dim = d_model
            else:
                updated_in_dim = current_channel_dim
                extended_dim = current_channel_dim

            # Groups 计算
            if 'conv1d' in e_layer:
                if self.group_embedding == 'False':
                    groups = 1
                else:
                    if extended_dim >= updated_in_dim and extended_dim % updated_in_dim == 0:
                        groups = updated_in_dim
                    elif extended_dim < updated_in_dim and updated_in_dim % extended_dim == 0:
                        groups = extended_dim
                    else:
                        groups = 1

            # 构建各层
            if e_layer == 'dropout':
                self.encoder_layers.append(nn.Dropout(p=dropout))
                self.norm_layers.append(nn.Identity())

            elif e_layer == 'multiscale_conv1d':
                self.encoder_layers.append(
                    Inception_Block(
                        in_channels=updated_in_dim,
                        out_channels=extended_dim,
                        kernel_list=kernel_size,
                        groups=groups
                    )
                )
                self.norm_layers.append(nn.LayerNorm(extended_dim))
                current_channel_dim = extended_dim

            elif e_layer == 'inter':
                n_heads = self._calculate_n_heads(current_channel_dim)

                attention_layer = MultiScaleChannelAttention(
                    d_model=current_channel_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    local_k=local_k,
                    level=wavelet_level
                )
                self.encoder_layers.append(attention_layer)
                self.norm_layers.append(nn.LayerNorm(current_channel_dim))

            else:
                raise ValueError(f'不支持的层类型: {e_layer}')

        if in_dim != current_channel_dim:
            self.residual_projection = nn.Linear(in_dim, current_channel_dim)

        self.dropout = nn.Dropout(p=dropout)
        self._initialize_weights(init_type, gain)

    def _calculate_n_heads(self, d_model):
        """计算合适的注意力头数"""
        for n_heads in [8, 4, 2, 1]:
            if d_model % n_heads == 0:
                return n_heads
        return 1

    def _initialize_weights(self, init_type, gain):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, L, C]

        Returns:
            x: [B, L, C], latent_list
        """
        B, L, C = x.size()
        latent_list = []
        original_length = L

        x_original = x

        for i, (embedding_layer, norm_layer) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            current_layer = self.branch_layers[i]

            if current_layer == 'multiscale_conv1d':
                x = x.permute(0, 2, 1)
                x = embedding_layer(x)
                x = x.permute(0, 2, 1)
                x = norm_layer(x)

            elif current_layer == 'inter':
                # 1. 小波变换
                scales_coeffs = self.wavelet_transform.forward(x)

                # 2. 多尺度层次化通道注意力
                scales_coeffs, attn_weights = embedding_layer(scales_coeffs)

                # 3. 逆小波变换
                x_wavelet = self.wavelet_transform.inverse(scales_coeffs, original_length)

                # 4. 归一化
                x_wavelet = norm_layer(x_wavelet)

                # 5. 自适应融合
                if self.residual_projection is not None:
                    x_original_proj = self.residual_projection(x_original)
                else:
                    x_original_proj = x_original

                alpha = torch.sigmoid(self.residual_alpha)
                x = alpha * x_original_proj + (1 - alpha) * x_wavelet

            elif current_layer == 'dropout':
                x = embedding_layer(x)

            latent_list.append(x.clone())

        return x, latent_list

    def get_fusion_ratio(self):
        """获取小波融合比例"""
        alpha = torch.sigmoid(self.residual_alpha).item()
        return {
            'original_ratio': alpha,
            'wavelet_ratio': 1 - alpha,
            'raw_alpha': self.residual_alpha.item()
        }


# ==================== Branch2 Embedding ====================
class Branch2Embedding(nn.Module):
    """Branch2 输入嵌入层 - 主入口"""

    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=4,
                 group_embedding='False', kernel_size=5,
                 init_type='kaiming', match_dimension='first',
                 branch_layers=['multiscale_conv1d', 'inter'],
                 wavelet='db4', wavelet_level=4, local_k=5):
        super(Branch2Embedding, self).__init__()

        self.device = device
        self.token_embedding = TokenEmbeddingBranch2(
            in_dim=in_dim, d_model=d_model, n_window=n_window,
            n_layers=n_layers, branch_layers=branch_layers,
            group_embedding=group_embedding, match_dimension=match_dimension,
            init_type=init_type, kernel_size=kernel_size,
            dropout=dropout, wavelet=wavelet, wavelet_level=wavelet_level,
            local_k=local_k
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, L, C]

        Returns:
            output: [B, L, C], latent_list
        """
        x = x.to(self.device)
        x, latent_list = self.token_embedding(x)
        return self.dropout(x), latent_list

    def get_fusion_ratio(self):
        """获取融合比例信息"""
        return self.token_embedding.get_fusion_ratio()
