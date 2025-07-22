import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion
from utils.RevIN import RevIN


class ChannelSTAR(nn.Module):
    """Channel-level STAR模块 - 基于SOFTS论文设计"""

    def __init__(self, d_model, d_core=None):
        super().__init__()
        self.d_model = d_model
        self.d_core = d_core if d_core is not None else d_model // 2

        # 核心表示生成网络 - 参考SOFTS的MLP设计
        self.core_gen = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.d_core)
        )

        # 融合网络 - 参考SOFTS的fusion设计
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model + self.d_core, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def stochastic_pooling(self, x):
        """
        SOFTS风格的随机池化 - 在channel维度聚合
        x: [B, N, d_core] - N是channel数
        """
        batch_size, n_channels, core_dim = x.shape

        if self.training:
            # 训练时：按概率随机采样channel
            probs = F.softmax(x, dim=1)  # [B, N, d_core] - 在channel维度计算概率

            # 重新塑形便于采样：[B*d_core, N]
            probs_reshaped = probs.permute(0, 2, 1).contiguous().view(-1, n_channels)  # [B*d_core, N]
            x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, n_channels)  # [B*d_core, N]

            # 为每个(batch, feature)对采样一个channel
            sampled_indices = torch.multinomial(probs_reshaped, 1).squeeze(-1)  # [B*d_core]

            # 收集采样结果
            batch_indices = torch.arange(batch_size * core_dim, device=x.device)
            sampled_values = x_reshaped[batch_indices, sampled_indices]  # [B*d_core]

            # 重新塑形为核心表示：[B, 1, d_core]
            core = sampled_values.view(batch_size, core_dim).unsqueeze(1)  # [B, 1, d_core]
        else:
            # 测试时：加权平均
            weights = F.softmax(x, dim=1)  # [B, N, d_core]
            core = torch.sum(x * weights, dim=1, keepdim=True)  # [B, 1, d_core]

        return core

    def forward(self, x):
        """
        x: [B, N, d_model] - N是channel数，d_model是series embedding维度
        输出: [B, N, d_model] - 增强后的channel特征
        """
        B, N, D = x.shape

        # 生成每个channel的核心表示候选
        core_candidates = self.core_gen(x)  # [B, N, d_core]

        # 随机池化生成全局核心（在channel维度聚合）
        global_core = self.stochastic_pooling(core_candidates)  # [B, 1, d_core]

        # 将全局核心分发到每个channel
        global_core_expanded = global_core.expand(B, N, self.d_core)  # [B, N, d_core]

        # 融合原始channel特征和全局核心
        fused_input = torch.cat([x, global_core_expanded], dim=-1)  # [B, N, d_model + d_core]
        fused_output = self.fusion_net(fused_input)  # [B, N, d_model]

        # 残差连接
        return x + fused_output


class LightweightDiffusion(nn.Module):
    """轻量级扩散模块"""

    def __init__(self, time_steps=20, device='cuda', scheduler='linear'):
        super().__init__()
        self.diffusion = Diffusion(time_steps=time_steps, device=device, scheduler=scheduler)

    def forward(self, x, apply_noise=True):
        if apply_noise and self.training:
            return self.diffusion(x)
        else:
            return x, None, None


class AdaptiveKANMixer(nn.Module):
    """自适应KAN混合器"""

    def __init__(self, d_model, component_type='trend'):
        super().__init__()
        # 根据分量类型选择KAN阶数
        order_map = {'trend': 3, 'seasonal': 5, 'residual': 4}
        order = order_map.get(component_type, 4)

        self.kan_layer = ChebyKANLinear(d_model, d_model, order)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        x_kan = self.kan_layer(x.reshape(B * T, C)).reshape(B, T, C)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + x_kan + x_conv)


class ComponentProcessor(nn.Module):
    """分量处理器 - seasonal分量使用Channel-level STAR"""

    def __init__(self, configs, component_type):
        super().__init__()
        self.component_type = component_type

        if component_type == 'trend':
            self.processor = nn.Sequential(
                AdaptiveKANMixer(configs.d_model, 'trend'),
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Dropout(configs.dropout)
            )
        elif component_type == 'seasonal':
            # 🔥 使用Channel-level STAR + Diffusion + KAN
            self.channel_star = ChannelSTAR(configs.d_model, configs.d_model // 2)
            self.diffusion = LightweightDiffusion(time_steps=20, device=configs.device)
            self.processor = AdaptiveKANMixer(configs.d_model, 'seasonal')
        else:  # residual
            self.processor = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
                nn.Dropout(configs.dropout)
            )

    def forward(self, x):
        if self.component_type == 'seasonal':
            # 🔥 注意：这里需要处理维度
            B, T, D = x.shape

            # 转换为channel维度处理：假设我们有多个channel
            # 如果是单变量，我们需要创造channel维度
            if self.training:
                # Channel-level STAR处理
                # 这里需要根据你的具体数据格式调整
                # 假设x是[B*N, T, d_model]格式（通道独立处理后）

                # 应用Channel STAR（需要重新组织数据）
                # 暂时跳过Channel STAR，因为在分量级别channel已经分离
                x_processed = x

                # 应用扩散
                x_noise, noise, t = self.diffusion(x_processed, apply_noise=True)
                return self.processor(x_noise)
            else:
                x_processed = x
                return self.processor(x_processed)
        else:
            return self.processor(x)

class Model(nn.Module):
    """Channel-level STAR模型 - 基于SOFTS设计思想"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 分解模块
        self.decomposition = series_decomp(configs.moving_avg)

        # 🔥 SOFTS风格的Series Embedding - 每个channel独立embedding
        self.series_embedding = nn.Linear(configs.seq_len, configs.d_model)

        # 🔥 Channel-level STAR模块 - 多层堆叠
        self.channel_star_layers = nn.ModuleList([
            ChannelSTAR(configs.d_model, configs.d_model // 2)
            for _ in range(getattr(configs, 'star_layers', 2))
        ])

        # 分量处理器
        self.trend_processor = ComponentProcessor(configs, 'trend')
        self.seasonal_processor = ComponentProcessor(configs, 'seasonal')
        self.residual_processor = ComponentProcessor(configs, 'residual')

        # 归一化
        self.revin_layer = RevIN(configs.enc_in, affine=True)

        # 🔥 SOFTS风格的预测层 - 直接从series representation预测
        self.trend_predictor = nn.Linear(configs.d_model, configs.pred_len)
        self.seasonal_predictor = nn.Linear(configs.d_model, configs.pred_len)
        self.residual_predictor = nn.Linear(configs.d_model, configs.pred_len)

        # 可学习融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([0.25, 0.5, 0.25]))  # [trend, seasonal, residual]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.revin_layer(x_enc, 'norm')

        # 分解
        seasonal, trend = self.decomposition(x_enc)
        residual = x_enc - seasonal - trend

        # 🔥 SOFTS风格的Series Embedding - 每个channel的时序映射到embedding空间
        # [B, T, N] -> [B, N, T] -> [B, N, d_model]
        seasonal_series = self.series_embedding(seasonal.transpose(1, 2))  # [B, N, d_model]
        trend_series = self.series_embedding(trend.transpose(1, 2))  # [B, N, d_model]
        residual_series = self.series_embedding(residual.transpose(1, 2))  # [B, N, d_model]

        # 🔥 Channel-level STAR处理 - 在channel维度交互
        # 对每个分量分别应用Channel STAR
        for star_layer in self.channel_star_layers:
            seasonal_series = star_layer(seasonal_series)
            trend_series = star_layer(trend_series)
            residual_series = star_layer(residual_series)

        # 直接从series representation预测（SOFTS风格）
        seasonal_pred = self.seasonal_predictor(seasonal_series)  # [B, N, pred_len]
        trend_pred = self.trend_predictor(trend_series)  # [B, N, pred_len]
        residual_pred = self.residual_predictor(residual_series)  # [B, N, pred_len]

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        final_pred = (weights[0] * trend_pred +
                      weights[1] * seasonal_pred +
                      weights[2] * residual_pred)  # [B, N, pred_len]

        # 转换回时序格式：[B, N, pred_len] -> [B, pred_len, N]
        final_pred = final_pred.transpose(1, 2)

        # 反归一化
        final_pred = self.revin_layer(final_pred, 'denorm')
        return final_pred