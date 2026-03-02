import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from model.Conv_Blocks import Inception_Block
from model.multi_cnn import Inception_Conv_Attention_Block


# ==================== 辅助函数 ====================
def complex_dropout(dropout_layer, x):
    """对复数张量应用dropout"""
    if isinstance(x, list):
        return [dropout_layer(item) for item in x]
    else:
        return dropout_layer(x)


def complex_operator(net_layer, x):
    """对复数张量应用网络层"""
    return net_layer[0](x) if isinstance(net_layer, nn.ModuleList) else net_layer(x)


# ==================== 位置编码 ====================
class PositionalEmbedding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


# ==================== 小波变换 ====================
class WaveletTransform(nn.Module):
    """
    小波变换模块 - Branch1使用
    支持前向变换、逆变换和长度计算
    """

    def __init__(self, wavelet='db4', mode='symmetric', level=4):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        # 缓存每个长度对应的系数结构
        self.coeffs_structure_cache = {}

    def forward(self, x):
        """
        前向小波变换

        Args:
            x: [B, C, L] - 批量时域信号

        Returns:
            [B, C, wavelet_len] - 拼接后的小波系数
        """
        batch_size, channels, seq_len = x.shape

        coeffs_list = []
        for i in range(batch_size):
            batch_coeffs = []
            for j in range(channels):
                signal = x[i, j].detach().cpu().numpy()
                # 小波分解：得到 [cA_n, cD_n, ..., cD_1]
                coeffs = pywt.wavedec(signal, self.wavelet, mode=self.mode, level=self.level)

                # 缓存系数结构（第一次处理时）
                if seq_len not in self.coeffs_structure_cache:
                    self.coeffs_structure_cache[seq_len] = [len(c) for c in coeffs]

                # 拼接所有系数
                concat_coeffs = torch.cat([torch.from_numpy(c).float() for c in coeffs], dim=0)
                batch_coeffs.append(concat_coeffs)
            coeffs_list.append(torch.stack(batch_coeffs))

        result = torch.stack(coeffs_list).to(x.device)
        return result  # [B, C, wavelet_len]

    def inverse(self, coeffs, original_length):
        """
        逆小波变换

        Args:
            coeffs: [B, C, wavelet_len] - 拼接的小波系数
            original_length: int - 原始信号长度

        Returns:
            [B, C, original_length] - 重构的时域信号
        """
        batch_size, channels, _ = coeffs.shape

        reconstructed = []
        for i in range(batch_size):
            batch_recon = []
            for j in range(channels):
                coeff_signal = coeffs[i, j].detach().cpu().numpy()

                # 使用缓存的系数结构分割
                if original_length in self.coeffs_structure_cache:
                    coeffs_split = self._split_coeffs_cached(coeff_signal, original_length)
                else:
                    # 如果没有缓存，使用临时信号获取结构
                    coeffs_split = self._split_coeffs_auto(coeff_signal, original_length)

                # 重构信号
                recon = pywt.waverec(coeffs_split, self.wavelet, mode=self.mode)

                # 精确恢复原始长度
                if len(recon) > original_length:
                    recon = recon[:original_length]
                elif len(recon) < original_length:
                    recon = torch.nn.functional.pad(
                        torch.from_numpy(recon).float(),
                        (0, original_length - len(recon))
                    ).numpy()

                batch_recon.append(torch.from_numpy(recon).float())
            reconstructed.append(torch.stack(batch_recon))

        result = torch.stack(reconstructed).to(coeffs.device)
        return result

    def _split_coeffs_cached(self, coeffs, original_length):
        """使用缓存的结构分割系数"""
        lengths = self.coeffs_structure_cache[original_length]

        coeffs_list = []
        start = 0
        for length in lengths:
            end = start + length
            coeffs_list.append(coeffs[start:end])
            start = end

        return coeffs_list

    def _split_coeffs_auto(self, coeffs, original_length):
        """自动获取系数结构并分割"""
        # 创建一个临时信号以获取正确的系数结构
        temp_signal = torch.randn(original_length).numpy()
        temp_coeffs = pywt.wavedec(temp_signal, self.wavelet, mode=self.mode, level=self.level)
        lengths = [len(c) for c in temp_coeffs]

        # 缓存结构
        self.coeffs_structure_cache[original_length] = lengths

        # 分割系数
        coeffs_list = []
        start = 0
        for length in lengths:
            end = start + length
            if end > len(coeffs):
                # 如果系数不够，用零填充
                pad_length = end - len(coeffs)
                coeffs_list.append(
                    torch.nn.functional.pad(
                        torch.from_numpy(coeffs[start:]).float(),
                        (0, pad_length)
                    ).numpy()
                )
            else:
                coeffs_list.append(coeffs[start:end])
            start = end

        return coeffs_list

    def get_wavelet_length(self, signal_length):
        """
        计算小波变换后的总长度

        Args:
            signal_length: int - 原始信号长度

        Returns:
            int - 小波系数总长度
        """
        # 使用临时信号获取真实的系数长度
        temp_signal = torch.randn(signal_length).numpy()
        temp_coeffs = pywt.wavedec(temp_signal, self.wavelet, mode=self.mode, level=self.level)

        # 缓存结构
        lengths = [len(c) for c in temp_coeffs]
        self.coeffs_structure_cache[signal_length] = lengths

        return sum(lengths)

    def get_split_point(self, signal_length):
        """
        获取小波分解的低频/高频分割点

        Args:
            signal_length: int - 原始信号长度

        Returns:
            int - 近似系数（低频）的长度
        """
        if signal_length not in self.coeffs_structure_cache:
            self.get_wavelet_length(signal_length)

        # 第一个系数是近似系数（低频）
        return self.coeffs_structure_cache[signal_length][0]


# ==================== 注意力机制 ====================
class LocalAttention(nn.Module):
    """局部窗口注意力 - 捕获高频细节"""

    def __init__(self, d_model, n_heads=4, window_size=30, dropout=0.1):
        super(LocalAttention, self).__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape

        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        attn_mask = self._create_local_mask(L).to(x.device)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_O(out)

        return out, attn_weights.mean(dim=1)

    def _create_local_mask(self, length):
        """创建局部窗口掩码"""
        mask = torch.zeros(length, length)
        for i in range(length):
            start = max(0, i - self.window_size // 2)
            end = min(length, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask


class SparseGlobalAttention(nn.Module):
    """稀疏全局注意力 - 捕获低频趋势"""

    def __init__(self, d_model, n_heads=4, stride=2, dropout=0.1):
        super(SparseGlobalAttention, self).__init__()

        assert d_model % n_heads == 0, f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.stride = stride

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape

        sparse_indices = torch.arange(0, L, self.stride, device=x.device)
        x_sparse = x[:, sparse_indices, :]
        L_sparse = x_sparse.shape[1]

        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x_sparse).view(B, L_sparse, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x_sparse).view(B, L_sparse, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.W_O(out)

        return out, attn_weights.mean(dim=1)


class FrequencyAwareAttention(nn.Module):
    """频率感知注意力 - Branch1核心模块"""

    def __init__(self, d_model, n_heads=4, window_size=30, stride=2, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # 低频：稀疏全局注意力
        self.low_freq_attn = SparseGlobalAttention(
            d_model, n_heads, stride=stride, dropout=dropout
        )

        # 高频：局部注意力
        self.high_freq_attn = LocalAttention(
            d_model, n_heads, window_size=window_size, dropout=dropout
        )

        # 跨频率交互
        self.cross_freq_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, split_point=None):
        """
        Args:
            x: [B, L, C] - 小波域拼接特征
            split_point: int - 低频/高频分割点

        Returns:
            [B, L, C], attention_dict
        """
        B, L, C = x.shape
        split_point = split_point or L // 2

        # 分离低频和高频
        x_low = x[:, :split_point, :]
        x_high = x[:, split_point:, :]

        # 1. 低频：稀疏全局注意力
        low_out, low_attn = self.low_freq_attn(x_low)

        # 2. 高频：局部注意力
        high_out, high_attn = self.high_freq_attn(x_high)

        # 3. 跨频率交互
        cross_out, _ = self.cross_freq_attn(
            query=x_high,
            key=x_low,
            value=x_low
        )

        # 4. 融合
        high_fused = torch.cat([high_out, cross_out], dim=-1)
        high_fused = self.fusion(high_fused)

        # 5. 重新拼接
        out = torch.cat([low_out, high_fused], dim=1)
        out = self.layer_norm(out + x)

        return out, {
            'low_freq': low_attn,
            'high_freq': high_attn,
            'cross_freq': cross_out
        }


class HybridAttentionLayer(nn.Module):
    """混合注意力层 - 封装频率感知注意力"""

    def __init__(self, w_size, d_model, n_heads=4, window_size=30, stride=2, dropout=0.1):
        super(HybridAttentionLayer, self).__init__()

        self.w_size = w_size
        self.d_model = d_model

        # 自动调整 n_heads
        n_heads = self._auto_adjust_heads(d_model, n_heads)

        self.attention = FrequencyAwareAttention(
            d_model, n_heads, window_size,
            stride=stride,
            dropout=dropout
        )

    def _auto_adjust_heads(self, d_model, desired_heads):
        """自动调整 n_heads 使其能整除 d_model"""
        if d_model % desired_heads == 0:
            return desired_heads

        # 尝试更小的头数
        for heads in range(desired_heads - 1, 0, -1):
            if d_model % heads == 0:
                print(f"⚠️ 调整: n_heads {desired_heads} → {heads} (d_model={d_model})")
                return heads

        # 尝试更大的头数
        for heads in range(desired_heads + 1, d_model + 1):
            if d_model % heads == 0:
                print(f"⚠️ 调整: n_heads {desired_heads} → {heads} (d_model={d_model})")
                return heads

        print(f"⚠️ 警告: d_model={d_model} 过小，使用 n_heads=1")
        return 1

    def forward(self, x, split_point=None):
        """
        Args:
            x: [B, L, C] 或 [B, C, L]
            split_point: int - 低频/高频分割点

        Returns:
            [B, L, C] 或 [B, C, L], attention_dict
        """
        # 检测输入格式并转换
        need_transpose = False
        if x.dim() == 3 and x.shape[1] == self.d_model:
            x = x.transpose(1, 2)
            need_transpose = True

        # 应用频率感知注意力
        out, attn_dict = self.attention(x, split_point)

        # 如果需要，转回原格式
        if need_transpose:
            out = out.transpose(1, 2)

        return out, attn_dict


# ==================== 编码器层 ====================
class EncoderLayer(nn.Module):
    """编码器层包装器"""

    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        self.attn_layer = attn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, split_point=None):
        """
        Args:
            x: [B, L, C]
            split_point: int - 传递给注意力层

        Returns:
            [B, L, C]
        """
        if split_point is not None:
            out, attn = self.attn_layer(x, split_point=split_point)
        else:
            out, attn = self.attn_layer(x)

        y = complex_dropout(self.dropout, out)
        return y


# ==================== Token Embedding ====================
class TokenEmbeddingBranch1(nn.Module):
    """Branch1 Token Embedding - 时域 + 小波域混合处理"""

    def __init__(self, in_dim, d_model, n_window=100, n_layers=1,
                 branch_layers=['intra_HY', 'multiscale_ts_attention'],
                 group_embedding='False', match_dimension='first',
                 multiscale_patch_size=[10, 20, 50], kernel_size=[1],
                 init_type='normal', gain=0.02, dropout=0.1,
                 wavelet='db4', wavelet_level=4, stride=2, window_size=30):
        super(TokenEmbeddingBranch1, self).__init__()

        self.window_size = n_window
        self.d_model = d_model
        self.n_layers = n_layers
        self.branch_layers = branch_layers
        self.group_embedding = group_embedding
        self.match_dimension = match_dimension
        self.multiscale_patch_size = multiscale_patch_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.local_window_size = window_size

        # 输入投影层
        self.input_proj = nn.Linear(in_dim, d_model, bias=False)
        self.input_norm = nn.LayerNorm(d_model)

        # 小波变换
        self.wavelet_transform = WaveletTransform(wavelet=wavelet, level=wavelet_level)
        self.wavelet_length = self.wavelet_transform.get_wavelet_length(n_window)
        self.wavelet_split_point = self.wavelet_transform.get_split_point(n_window)

        self.encoder_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        # 构建网络层
        for i, e_layer in enumerate(branch_layers):
            if e_layer == 'dropout':
                self.encoder_layers.append(nn.Dropout(p=dropout))
                self.norm_layers.append(nn.Identity())

            elif e_layer == 'intra_HY':
                attention_layer = HybridAttentionLayer(
                    w_size=self.wavelet_length,
                    d_model=d_model,
                    n_heads=4,
                    window_size=self.local_window_size,
                    stride=self.stride,
                    dropout=dropout
                )
                self.encoder_layers.append(
                    EncoderLayer(
                        attn=attention_layer,
                        d_model=d_model,
                        d_ff=128,
                        dropout=dropout,
                        activation='gelu'
                    )
                )
                self.norm_layers.append(nn.LayerNorm(d_model))

            elif e_layer == 'multiscale_ts_attention':
                self.encoder_layers.append(
                    Inception_Conv_Attention_Block(
                        d_model=d_model,
                        kernel_list=multiscale_patch_size,
                        use_diff_mask=True,
                    )
                )
                self.norm_layers.append(nn.Identity())

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

        self._initialize_weights(init_type, gain)

    def _initialize_weights(self, init_type, gain):
        """权重初始化"""
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

        # 1. 时域投影
        x = self.input_proj(x)
        x = self.input_norm(x)
        latent_list.append(x)

        is_in_wavelet_domain = False

        # 2. 逐层处理
        for i, (embedding_layer, norm_layer) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            current_layer = self.branch_layers[i]
            next_layer = self.branch_layers[i + 1] if i + 1 < len(self.branch_layers) else None
            next_needs_wavelet = next_layer == 'intra_HY'

            if current_layer == 'intra_HY':
                # 转到小波域
                if not is_in_wavelet_domain:
                    x = x.permute(0, 2, 1)
                    x = self.wavelet_transform.forward(x)
                    x = x.permute(0, 2, 1)
                    is_in_wavelet_domain = True

                # 频率感知注意力
                x = embedding_layer(x, split_point=self.wavelet_split_point)
                x = complex_operator(norm_layer, x)

                # 如果需要，转回时域
                if not next_needs_wavelet:
                    x = x.permute(0, 2, 1)
                    x = self.wavelet_transform.inverse(x, original_length)
                    x = x.permute(0, 2, 1)
                    is_in_wavelet_domain = False

            elif current_layer == 'multiscale_ts_attention':
                # 确保在时域
                if is_in_wavelet_domain:
                    x = x.permute(0, 2, 1)
                    x = self.wavelet_transform.inverse(x, original_length)
                    x = x.permute(0, 2, 1)
                    is_in_wavelet_domain = False

                x = embedding_layer(x)
                x = complex_operator(norm_layer, x)

            elif current_layer == 'dropout':
                if is_in_wavelet_domain:
                    x = x.permute(0, 2, 1)
                    x = self.wavelet_transform.inverse(x, original_length)
                    x = x.permute(0, 2, 1)
                    is_in_wavelet_domain = False

                x = embedding_layer(x)
                x = complex_operator(norm_layer, x)

            # 保存中间特征（确保在时域）
            if is_in_wavelet_domain:
                x_time = x.permute(0, 2, 1)
                x_time = self.wavelet_transform.inverse(x_time, original_length)
                x_time = x_time.permute(0, 2, 1)
                latent_list.append(x_time)
            else:
                latent_list.append(x)

        # 3. 确保最终输出在时域
        if is_in_wavelet_domain:
            x = x.permute(0, 2, 1)
            x = self.wavelet_transform.inverse(x, original_length)
            x = x.permute(0, 2, 1)

        return x, latent_list


# ==================== Branch1 Embedding ====================
class Branch1Embedding(nn.Module):
    """Branch1 输入嵌入层 - 主入口"""

    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=4,
                 use_pos_embedding='False', group_embedding='False', kernel_size=5,
                 init_type='kaiming', match_dimension='first',
                 branch_layers=['intra_HY', 'multiscale_ts_attention'],
                 multiscale_patch_size=[5, 15, 25, 50],
                 wavelet='db4', wavelet_level=4, stride=2, window_size=30):
        super(Branch1Embedding, self).__init__()

        self.device = device
        self.use_pos_embedding = use_pos_embedding

        self.token_embedding = TokenEmbeddingBranch1(
            in_dim=in_dim,
            d_model=d_model,
            n_window=n_window,
            n_layers=n_layers,
            branch_layers=branch_layers,
            group_embedding=group_embedding,
            match_dimension=match_dimension,
            multiscale_patch_size=multiscale_patch_size,
            init_type=init_type,
            kernel_size=kernel_size,
            dropout=dropout,
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            stride=stride,
            window_size=window_size
        )

        self.pos_embedding = PositionalEmbedding(d_model=d_model)
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

        if self.use_pos_embedding == 'True':
            x = x + self.pos_embedding(x).to(self.device)


