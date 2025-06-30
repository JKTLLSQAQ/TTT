import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TimeDART_EncDec import Diffusion


class EmbeddingSTAR(nn.Module):
    """在embedding空间应用的STAR模块 - 更符合SOFTS设计"""

    def __init__(self, d_model, d_core=None):
        super().__init__()
        self.d_model = d_model
        self.d_core = d_core if d_core is not None else d_model // 2

        # 核心表示生成网络 - 在embedding维度操作
        self.core_gen = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.d_core)
        )

        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model + self.d_core, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def stochastic_pooling(self, x):
        """SOFTS风格的随机池化 - 在时间维度聚合"""
        # x: [B, T, d_core] 或 [B*N, T, d_core]
        if self.training:
            # 训练时：按概率随机采样
            probs = F.softmax(x, dim=1)  # [B, T, d_core] - 在时间维度计算概率

            # 对每个特征维度和batch独立采样
            batch_size, seq_len, core_dim = x.shape

            # 重新塑形便于采样：[B*d_core, T]
            probs_reshaped = probs.permute(0, 2, 1).contiguous().view(-1, seq_len)  # [B*d_core, T]
            x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, seq_len)  # [B*d_core, T]

            # 为每个(batch, feature)对采样一个时间点
            sampled_indices = torch.multinomial(probs_reshaped, 1).squeeze(-1)  # [B*d_core]

            # 收集采样结果
            batch_indices = torch.arange(batch_size * core_dim, device=x.device)
            sampled_values = x_reshaped[batch_indices, sampled_indices]  # [B*d_core]

            # 重新塑形为核心表示：[B, 1, d_core]
            core = sampled_values.view(batch_size, core_dim).unsqueeze(1)  # [B, 1, d_core]
        else:
            # 测试时：加权平均
            weights = F.softmax(x, dim=1)  # [B, T, d_core]
            core = torch.sum(x * weights, dim=1, keepdim=True)  # [B, 1, d_core]

        return core

    def forward(self, x):
        """
        x: [B, T, d_model] 或 [B*N, T, d_model] - embedding后的特征
        输出: [B, T, d_model] - 增强后的特征
        """
        B, T, D = x.shape

        # 生成核心表示候选
        core_candidates = self.core_gen(x)  # [B, T, d_core]

        # 随机池化生成全局核心（在时间维度聚合）
        global_core = self.stochastic_pooling(core_candidates)  # [B, 1, d_core]

        # 将全局核心分发到每个时间步
        global_core_expanded = global_core.expand(B, T, self.d_core)  # [B, T, d_core]

        # 融合原始特征和全局核心
        fused_input = torch.cat([x, global_core_expanded], dim=-1)  # [B, T, d_model + d_core]
        fused_output = self.fusion_net(fused_input)  # [B, T, d_model]

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
    """分量处理器 - 在seasonal embedding后应用STAR"""

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
            # 为seasonal分量添加embedding后的STAR模块
            self.embedding_star = EmbeddingSTAR(configs.d_model, configs.d_model // 2)
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
            # 先应用embedding级别的STAR模块
            x_star = self.embedding_star(x)

            # 然后应用扩散和处理
            if self.training:
                x_noise, noise, t = self.diffusion(x_star, apply_noise=True)
                return self.processor(x_noise)
            else:
                return self.processor(x_star)
        else:
            return self.processor(x)


class Model(nn.Module):
    """embedding后STAR + 分解KAN融合模型"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # 分解模块
        self.decomposition = series_decomp(configs.moving_avg)

        # 嵌入层
        if configs.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        # 分量处理器（seasonal处理器包含embedding后的STAR模块）
        self.trend_processor = ComponentProcessor(configs, 'trend')
        self.seasonal_processor = ComponentProcessor(configs, 'seasonal')  # 包含EmbeddingSTAR
        self.residual_processor = ComponentProcessor(configs, 'residual')

        # 归一化
        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for i in range(configs.down_sampling_layers + 1)
        ])

        # 预测层
        self.trend_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.seasonal_predictor = nn.Linear(configs.seq_len, configs.pred_len)
        self.residual_predictor = nn.Linear(configs.seq_len, configs.pred_len)

        # 输出投影
        if configs.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

        # 可学习融合权重（继续关注seasonal分量）
        self.fusion_weights = nn.Parameter(torch.tensor([0.25, 0.5, 0.25]))  # [trend, seasonal, residual]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise ValueError('Only long_term_forecast implemented')

    def forecast(self, x_enc, x_mark_enc=None):
        B, T, N = x_enc.size()

        # 归一化
        x_enc = self.normalize_layers[0](x_enc, 'norm')

        # 分解（不再使用GlobalSTAR）
        seasonal, trend = self.decomposition(x_enc)
        residual = x_enc - seasonal - trend

        # 通道独立性处理（现在应该是0，所以这部分不会执行）
        if self.configs.channel_independence == 1:
            seasonal = seasonal.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            trend = trend.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            residual = residual.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 嵌入
        if self.configs.channel_independence == 1 and x_mark_enc is not None:
            x_mark_enc_expanded = x_mark_enc.repeat(N, 1, 1)  # [B*N, T, mark_dim]
        else:
            x_mark_enc_expanded = x_mark_enc

        seasonal_emb = self.enc_embedding(seasonal, x_mark_enc_expanded)
        trend_emb = self.enc_embedding(trend, x_mark_enc_expanded)
        residual_emb = self.enc_embedding(residual, x_mark_enc_expanded)

        # 分量处理（seasonal现在会在embedding后通过STAR模块）
        seasonal_out = self.seasonal_processor(seasonal_emb)  # 内部包含EmbeddingSTAR
        trend_out = self.trend_processor(trend_emb)
        residual_out = self.residual_processor(residual_emb)

        # 时序预测
        seasonal_pred = self.seasonal_predictor(seasonal_out.permute(0, 2, 1)).permute(0, 2, 1)
        trend_pred = self.trend_predictor(trend_out.permute(0, 2, 1)).permute(0, 2, 1)
        residual_pred = self.residual_predictor(residual_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 投影
        seasonal_pred = self.projection_layer(seasonal_pred)
        trend_pred = self.projection_layer(trend_pred)
        residual_pred = self.projection_layer(residual_pred)

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        dec_out = (weights[0] * trend_pred +
                   weights[1] * seasonal_pred +
                   weights[2] * residual_pred)

        # 输出重塑
        if self.configs.channel_independence == 1:
            dec_out = dec_out.reshape(B, N, self.pred_len, -1)
            if dec_out.shape[-1] == 1:
                dec_out = dec_out.squeeze(-1)
            dec_out = dec_out.permute(0, 2, 1).contiguous()

        if dec_out.shape[-1] > self.configs.c_out:
            dec_out = dec_out[..., :self.configs.c_out]

        # 反归一化
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out