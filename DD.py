import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# 保留你原有的导入
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.Autoformer_EncDec import series_decomp


class CapacityRecoveryDetector(nn.Module):
    """专门检测容量回升现象的可变形卷积模块"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 可变形卷积的偏移预测
        self.offset_conv = nn.Conv1d(channels, 2 * kernel_size, kernel_size=3, padding=1)

        # 权重预测（调制）
        self.modulator_conv = nn.Conv1d(channels, kernel_size, kernel_size=3, padding=1)

        # 主卷积
        self.weight = nn.Parameter(torch.randn(channels, channels, kernel_size))

        # 容量回升强度预测
        self.recovery_predictor = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape

        # 预测偏移量
        offset = self.offset_conv(x)  # [B, 2*K, T]

        # 预测调制权重
        modulator = torch.sigmoid(self.modulator_conv(x))  # [B, K, T]

        # 应用可变形卷积（简化版实现）
        deformed_features = self.deformable_conv1d(x, offset, modulator)

        # 预测容量回升强度
        recovery_strength = self.recovery_predictor(deformed_features)

        return deformed_features, recovery_strength.squeeze(1)  # [B, C, T], [B, T]

    def deformable_conv1d(self, x, offset, modulator):
        """简化的一维可变形卷积实现"""
        B, C, T = x.shape
        K = self.kernel_size

        # 重塑偏移量
        offset = offset.view(B, 2, K, T).permute(0, 3, 1, 2)  # [B, T, 2, K]
        modulator = modulator.view(B, K, T).permute(0, 2, 1)  # [B, T, K]

        # 生成基础采样位置
        base_pos = torch.arange(T, device=x.device).float().unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        kernel_pos = torch.arange(K, device=x.device).float() - K // 2  # [K]

        # 计算采样位置
        sampling_pos = base_pos + kernel_pos.unsqueeze(0).unsqueeze(0) + offset[:, :, 0, :]  # [B, T, K]

        # 限制采样位置在有效范围内
        sampling_pos = torch.clamp(sampling_pos, 0, T - 1)

        # 双线性插值采样
        output = torch.zeros_like(x)
        for c in range(C):
            for k in range(K):
                # 简化的插值实现
                pos = sampling_pos[:, :, k]  # [B, T]
                pos_floor = torch.floor(pos).long()
                pos_ceil = torch.ceil(pos).long()

                weight_floor = pos_ceil.float() - pos
                weight_ceil = pos - pos_floor.float()

                # 安全索引
                pos_floor = torch.clamp(pos_floor, 0, T - 1)
                pos_ceil = torch.clamp(pos_ceil, 0, T - 1)

                # 插值
                sampled_floor = x[torch.arange(B).unsqueeze(1), c, pos_floor]
                sampled_ceil = x[torch.arange(B).unsqueeze(1), c, pos_ceil]
                sampled = sampled_floor * weight_floor + sampled_ceil * weight_ceil

                # 应用权重和调制
                output[:, c, :] += sampled * self.weight[c, c, k] * modulator[:, :, k]

        return output


class AdaptiveKANMixer(nn.Module):
    """精简版KAN混合器"""

    def __init__(self, d_model, component_type='trend'):
        super().__init__()
        order_map = {'trend': 6, 'seasonal': 4, 'recovery': 8}
        order = order_map.get(component_type, 4)

        self.kan_layer = ChebyKANLinear(d_model, d_model, order)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)
        return self.norm(x + x_kan)


class GlobalSTAR(nn.Module):
    """保留你的GlobalSTAR模块（精简版）"""

    def __init__(self, seq_len, channels, d_core=None):
        super().__init__()
        self.d_core = d_core if d_core is not None else channels // 2

        self.core_gen = nn.Linear(channels, self.d_core)
        self.fusion_net = nn.Linear(channels + self.d_core, channels)

    def forward(self, x):
        B, T, C = x.shape

        # 生成核心表示
        core_candidates = self.core_gen(x)  # [B, T, d_core]

        # 全局池化
        if self.training:
            # 训练时使用随机池化
            probs = F.softmax(core_candidates, dim=2)
            indices = torch.multinomial(probs.view(-1, self.d_core), 1).view(B, T, 1)
            global_core = torch.gather(core_candidates, 2, indices.expand(-1, -1, 1))
        else:
            # 测试时使用平均池化
            global_core = core_candidates.mean(dim=2, keepdim=True)

        # 扩展并融合
        global_core_expanded = global_core.expand(B, T, self.d_core)
        fused_input = torch.cat([x, global_core_expanded], dim=-1)
        fused_output = self.fusion_net(fused_input)

        return x + fused_output


class Model(nn.Module):
    """精简版Enhanced STAR模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # GlobalSTAR模块
        if getattr(configs, 'use_global_star', True):
            self.global_star = GlobalSTAR(
                seq_len=configs.seq_len,
                channels=configs.enc_in,
                d_core=int(configs.enc_in * getattr(configs, 'star_core_ratio', 0.5))
            )
        else:
            self.global_star = None

        # 时序分解
        self.decomposition = series_decomp(configs.moving_avg)

        # 嵌入层
        if getattr(configs, 'channel_independence', 0) == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 容量回升检测器（可变形卷积）
        if getattr(configs, 'use_deformable_conv', True):
            self.recovery_detector = CapacityRecoveryDetector(configs.d_model)
        else:
            self.recovery_detector = None

        # KAN处理器
        self.trend_processor = AdaptiveKANMixer(configs.d_model, 'trend')
        self.seasonal_processor = AdaptiveKANMixer(configs.d_model, 'seasonal')
        self.recovery_processor = AdaptiveKANMixer(configs.d_model, 'recovery')

        # 归一化
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if getattr(configs, 'use_norm', 1) == 0 else False)
            for i in range(getattr(configs, 'down_sampling_layers', 0) + 1)
        ])

        # 预测层
        self.trend_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.seasonal_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.recovery_predictor = nn.Linear(configs.seq_len, configs.pred_len)

        # 输出投影
        if getattr(configs, 'channel_independence', 0) == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 融合权重
        initial_weights = torch.tensor([
            getattr(configs, 'trend_weight', 0.4),
            getattr(configs, 'seasonal_weight', 0.2),
            getattr(configs, 'recovery_weight', 0.4)
        ])
        self.fusion_weights = nn.Parameter(initial_weights)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # GlobalSTAR信息聚合
        if self.global_star is not None:
            x_star_enhanced = self.global_star(x_enc)
        else:
            x_star_enhanced = x_enc

        # 通道独立性处理
        if getattr(self.configs, 'channel_independence', 0) == 1:
            x_star_enhanced = x_star_enhanced.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc.repeat(N, 1, 1)

        # 时序分解
        seasonal, trend = self.decomposition(x_star_enhanced)
        residual = x_star_enhanced - seasonal - trend

        # 嵌入
        trend_emb = self.enc_embedding(trend, x_mark_enc)
        seasonal_emb = self.enc_embedding(seasonal, x_mark_enc)
        residual_emb = self.enc_embedding(residual, x_mark_enc)

        # 容量回升检测和处理
        if self.recovery_detector is not None:
            # 转换为卷积格式 [B, C, T]
            residual_conv = residual_emb.transpose(1, 2)
            recovery_features, recovery_strength = self.recovery_detector(residual_conv)
            recovery_features = recovery_features.transpose(1, 2)  # 转回 [B, T, C]

            # 使用回升强度来调制残差
            recovery_mask = torch.sigmoid(recovery_strength).unsqueeze(-1)  # [B, T, 1]
            residual_emb = residual_emb * (1 + recovery_mask) + recovery_features * recovery_mask

        # KAN处理
        trend_out = self.trend_processor(trend_emb)
        seasonal_out = self.seasonal_processor(seasonal_emb)
        recovery_out = self.recovery_processor(residual_emb)

        # 时序预测
        trend_pred = self.trend_predictor(trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_pred = self.seasonal_predictor(seasonal_out.permute(0, 2, 1)).permute(0, 2, 1)
        recovery_pred = self.recovery_predictor(recovery_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 投影
        trend_pred = self.projection_layer(trend_pred)
        seasonal_pred = self.projection_layer(seasonal_pred)
        recovery_pred = self.projection_layer(recovery_pred)

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        dec_out = (weights[0] * trend_pred +
                   weights[1] * seasonal_pred +
                   weights[2] * recovery_pred)

        # 输出重塑
        if getattr(self.configs, 'channel_independence', 0) == 1:
            dec_out = dec_out.reshape(B, N, self.pred_len, -1)
            if dec_out.shape[-1] == 1:
                dec_out = dec_out.squeeze(-1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

        if dec_out.shape[-1] > self.configs.c_out:
            dec_out = dec_out[..., :self.configs.c_out]

        # 反归一化
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out


class BarronAdaptiveLoss(nn.Module):
    """Barron自适应损失函数"""

    def __init__(self, alpha_init=2.0, scale_init=1.0):
        super().__init__()
        # 形状参数（限制在0以上避免发散）
        self._alpha = nn.Parameter(torch.tensor(alpha_init))
        # 尺度参数
        self._scale = nn.Parameter(torch.tensor(scale_init))

    @property
    def alpha(self):
        return F.softplus(self._alpha) + 1e-8  # 确保 > 0

    @property
    def scale(self):
        return F.softplus(self._scale) + 1e-8  # 确保 > 0

    def forward(self, pred, target):
        """
        Barron自适应损失函数
        """
        residual = pred - target
        alpha = self.alpha
        scale = self.scale

        # 避免除零
        scaled_residual = residual / (scale + 1e-8)

        # Barron损失核心公式
        if torch.abs(alpha - 2.0) < 1e-6:
            # 当alpha接近2时，近似为L2损失
            loss = 0.5 * scaled_residual.pow(2)
        else:
            # 通用Barron损失
            abs_alpha_minus_2 = torch.abs(alpha - 2.0)
            inner_term = (scaled_residual.pow(2) / abs_alpha_minus_2 + 1).pow(alpha / 2) - 1
            loss = abs_alpha_minus_2 / alpha * inner_term

        return loss.mean()


def capacity_aware_loss(pred, true, recovery_strength=None, alpha=0.2):
    """
    结合容量特性的损失函数
    """
    # 基础Barron损失
    barron_loss = BarronAdaptiveLoss()
    base_loss = barron_loss(pred, true)

    # 容量回升段额外惩罚
    capacity_diff = torch.diff(true, dim=1)
    recovery_mask = (capacity_diff > 0.001).float()  # 检测回升

    if recovery_mask.sum() > 0:
        recovery_loss = F.mse_loss(
            pred[:, 1:] * recovery_mask,
            true[:, 1:] * recovery_mask
        )
        total_loss = base_loss + alpha * recovery_loss
    else:
        total_loss = base_loss

    # 如果有回升强度信息，额外利用
    if recovery_strength is not None:
        # 鼓励模型在实际回升段预测更高的回升强度
        strength_loss = F.mse_loss(
            recovery_strength[:, 1:] * recovery_mask.squeeze(-1) if recovery_mask.dim() > 2 else recovery_strength[:,
                                                                                                 1:] * recovery_mask,
            recovery_mask.squeeze(-1) if recovery_mask.dim() > 2 else recovery_mask
        )
        total_loss += 0.1 * strength_loss

    return total_loss