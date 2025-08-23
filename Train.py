from exp.exp_basic import Exp_Basic
from models import dual, redual, STAR, FusedTimeModel
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
import os
import time
import warnings
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
from exp.exp_learning_rate import (AdaptiveLossLRScheduler, CombinedLRScheduler, plot_lr_loss_history)
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')


class BarronAdaptiveLoss(nn.Module):


    def __init__(self, alpha_init=2.0, scale_init=1.0, alpha_learning_rate=1e-4):
        super().__init__()

        # 可学习参数
        self._alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self._scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

        # 记录训练历史
        self.alpha_history = []
        self.scale_history = []
        self.loss_history = []

        # 参数约束
        self.alpha_learning_rate = alpha_learning_rate

        print(f"🔧 Barron自适应损失函数初始化:")
        print(f"   初始 α: {alpha_init}")
        print(f"   初始 σ: {scale_init}")
        print(f"   α 学习率: {alpha_learning_rate}")

    @property
    def alpha(self):
        """确保alpha > 0以避免数值不稳定"""
        return F.softplus(self._alpha) + 0.01

    @property
    def scale(self):
        """确保scale > 0以避免除零错误"""
        return F.softplus(self._scale) + 1e-6

    def forward(self, pred, target):
        """
        前向传播计算Barron损失

        Args:
            pred: 预测值 [batch_size, ...]
            target: 真实值 [batch_size, ...]

        Returns:
            loss: 标量损失值
        """
        residual = pred - target
        alpha = self.alpha
        scale = self.scale

        # 归一化残差
        normalized_residual = residual / scale

        # Barron损失计算
        if torch.abs(alpha - 2.0) < 1e-4:
            # 接近L2损失的情况
            loss = 0.5 * normalized_residual.pow(2)
        else:
            # 通用Barron损失
            abs_alpha_minus_2 = torch.abs(alpha - 2.0)
            inner_term = (normalized_residual.pow(2) / abs_alpha_minus_2 + 1).pow(alpha / 2) - 1
            loss = (abs_alpha_minus_2 / alpha) * inner_term

        # 记录历史（用于分析）
        self.alpha_history.append(alpha.item())
        self.scale_history.append(scale.item())
        self.loss_history.append(loss.mean().item())

        return loss.mean()

    def get_current_params(self):
        """获取当前参数值"""
        return {
            'alpha': self.alpha.item(),
            'scale': self.scale.item(),
            'alpha_raw': self._alpha.item(),
            'scale_raw': self._scale.item()
        }

    def get_param_history(self):
        """获取参数变化历史"""
        return {
            'alpha_history': self.alpha_history.copy(),
            'scale_history': self.scale_history.copy(),
            'loss_history': self.loss_history.copy()
        }


class EnhancedBarronLoss(BarronAdaptiveLoss):
    """
    增强版Barron损失函数
    添加了Progressive训练的特殊处理和电池SoH预测的优化
    """

    def __init__(self, alpha_init=2.0, scale_init=1.0,
                 capacity_recovery_weight=0.1, trend_consistency_weight=0.05):
        super().__init__(alpha_init, scale_init)

        # 电池SoH预测特有的权重
        self.capacity_recovery_weight = capacity_recovery_weight
        self.trend_consistency_weight = trend_consistency_weight

        # 训练阶段记录
        self.training_stage = 'traditional'
        self.stage_loss_stats = {
            'hard_only': [],
            'progressive': [],
            'all': [],
            'traditional': []
        }

        print(f"🔋 增强版Barron损失函数:")
        print(f"   容量回升权重: {capacity_recovery_weight}")
        print(f"   趋势一致性权重: {trend_consistency_weight}")

    def set_training_stage(self, stage):
        """设置当前训练阶段"""
        if stage != self.training_stage:
            print(f"🔄 损失函数阶段切换: {self.training_stage} -> {stage}")
            self.training_stage = stage

    def forward(self, pred, target, training_stage=None):
        """
        增强版前向传播

        Args:
            pred: 预测值
            target: 真实值
            training_stage: 当前训练阶段 ('hard_only', 'progressive', 'all', 'traditional')
        """
        if training_stage is not None:
            self.set_training_stage(training_stage)

        # 基础Barron损失
        base_loss = super().forward(pred, target)

        # 根据训练阶段调整损失
        if self.training_stage == 'hard_only':
            # 难样本阶段：增加对异常值的鲁棒性
            adjusted_loss = base_loss

        elif self.training_stage == 'progressive':
            # 渐进阶段：平衡基础损失和趋势一致性
            trend_loss = self._compute_trend_consistency_loss(pred, target)
            adjusted_loss = base_loss + self.trend_consistency_weight * trend_loss

        elif self.training_stage == 'all':
            # 全样本阶段：标准Barron损失
            adjusted_loss = base_loss

        else:
            # 传统训练阶段
            adjusted_loss = base_loss

        # 记录阶段损失统计
        self.stage_loss_stats[self.training_stage].append(adjusted_loss.item())

        return adjusted_loss

    def _compute_trend_consistency_loss(self, pred, target):
        """计算趋势一致性损失（用于渐进阶段）"""
        if pred.size(1) > 1:
            # 多步预测：计算预测趋势和真实趋势的一致性
            pred_diff = torch.diff(pred, dim=1)
            target_diff = torch.diff(target, dim=1)

            # 趋势方向一致性
            pred_direction = torch.sign(pred_diff)
            target_direction = torch.sign(target_diff)
            direction_loss = F.mse_loss(pred_direction, target_direction)

            return direction_loss
        else:
            # 单步预测：返回零损失
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

    def get_stage_statistics(self):
        """获取各阶段的损失统计"""
        stats = {}
        for stage, losses in self.stage_loss_stats.items():
            if losses:
                stats[stage] = {
                    'count': len(losses),
                    'mean': np.mean(losses),
                    'std': np.std(losses),
                    'min': np.min(losses),
                    'max': np.max(losses)
                }
            else:
                stats[stage] = {'count': 0}

        return stats

class ProgressiveBatteryDataProvider:
    """Progressive训练的数据提供器"""

    def __init__(self, args, marked_data_path: str):
        self.args = args
        self.marked_data_path = marked_data_path

        # 读取标记数据
        if os.path.exists(marked_data_path):
            self.marked_df = pd.read_csv(marked_data_path)
            print(f"✅ 成功读取Progressive标记数据: {marked_data_path}")
            print(f"   数据形状: {self.marked_df.shape}")

            # 分析难样本分布
            self._analyze_hard_samples()
        else:
            print(f"❌ Progressive标记数据文件不存在: {marked_data_path}")
            raise FileNotFoundError(f"Progressive标记数据文件不存在: {marked_data_path}")

    def _analyze_hard_samples(self):
        """分析难样本分布"""
        total_samples = len(self.marked_df)
        hard_samples = self.marked_df['is_hard_sample'].sum()
        recovery_windows = self.marked_df['recovery_window_id'].nunique() - 1  # 减去-1

        self.sample_stats = {
            'total_samples': total_samples,
            'hard_samples': hard_samples,
            'easy_samples': total_samples - hard_samples,
            'hard_ratio': hard_samples / total_samples,
            'recovery_windows': recovery_windows
        }

        print(f"🔍 Progressive难样本分布:")
        print(f"   难样本: {hard_samples} ({self.sample_stats['hard_ratio']:.1%})")
        print(f"   简单样本: {self.sample_stats['easy_samples']}")
        print(f"   回升区间: {recovery_windows}")

        # 获取索引
        self.hard_indices = self.marked_df[self.marked_df['is_hard_sample'] == True].index.tolist()
        self.easy_indices = self.marked_df[self.marked_df['is_hard_sample'] == False].index.tolist()

    def get_progressive_dataloader(self, original_dataset, stage: str, alpha: float = 0.0, batch_size: int = 32):
        """获取Progressive数据加载器"""
        if stage == 'hard_only':
            current_indices = self.hard_indices.copy()
        elif stage == 'progressive':
            num_easy_to_add = int(alpha * len(self.easy_indices))
            np.random.shuffle(self.easy_indices)
            selected_easy = self.easy_indices[:num_easy_to_add]
            current_indices = self.hard_indices + selected_easy
            np.random.shuffle(current_indices)
        elif stage == 'all':
            current_indices = list(range(len(self.marked_df)))
        else:
            raise ValueError(f"Unknown stage: {stage}")

        # 确保索引有效
        valid_indices = [i for i in current_indices if i < len(original_dataset)]

        if len(valid_indices) > 0:
            subset_dataset = Subset(original_dataset, valid_indices)
            subset_loader = DataLoader(
                subset_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
        else:
            print("❌ 错误: 没有有效的训练样本索引")
            subset_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return subset_loader, len(valid_indices)


class ProgressiveScheduler:
    """Progressive训练调度器"""

    def __init__(self, total_epochs: int, hard_ratio: float = 0.3, progressive_ratio: float = 0.5):
        self.total_epochs = total_epochs
        self.hard_epochs = max(1, int(total_epochs * hard_ratio))
        self.progressive_epochs = max(1, int(total_epochs * progressive_ratio))
        self.all_epochs = total_epochs - self.hard_epochs - self.progressive_epochs

        if self.all_epochs < 1:
            self.all_epochs = 1
            self.progressive_epochs = total_epochs - self.hard_epochs - 1

        print(f"🗓️  Progressive训练调度:")
        print(f"   难样本阶段: {self.hard_epochs} 轮")
        print(f"   渐进阶段: {self.progressive_epochs} 轮")
        print(f"   全样本阶段: {self.all_epochs} 轮")

    def get_stage_and_alpha(self, epoch: int):
        """获取训练阶段和混合比例"""
        if epoch <= self.hard_epochs:
            return 'hard_only', 0.0
        elif epoch <= self.hard_epochs + self.progressive_epochs:
            progress = (epoch - self.hard_epochs) / self.progressive_epochs
            return 'progressive', progress
        else:
            return 'all', 1.0


def generate_timestamp():
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_detailed_timestamp():
    """生成详细的时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Exp_Fused_Forecast(Exp_Basic):
    """集成Progressive功能和Barron损失函数的融合预测实验类"""

    def __init__(self, args):
        super(Exp_Fused_Forecast, self).__init__(args)
        self.experiment_timestamp = generate_timestamp()
        self.detailed_timestamp = generate_detailed_timestamp()

        # Progressive训练配置
        self.use_progressive = getattr(args, 'use_progressive_training', False)
        self.marked_data_path = getattr(args, 'marked_data_path', 'marked_battery_data.csv')
        self.hard_ratio = getattr(args, 'hard_ratio', 0.3)
        self.progressive_ratio = getattr(args, 'progressive_ratio', 0.5)

        print(f"🚀 Exp_Fused_Forecast 初始化")
        print(f"   实验时间戳: {self.experiment_timestamp}")
        print(f"   Progressive训练: {'启用' if self.use_progressive else '禁用'}")

        # 初始化Progressive数据提供器
        if self.use_progressive:
            try:
                self.progressive_provider = ProgressiveBatteryDataProvider(args, self.marked_data_path)
                self.sample_stats = self.progressive_provider.sample_stats
            except Exception as e:
                print(f"❌ Progressive初始化失败: {e}")
                print("   将使用传统训练方式")
                self.use_progressive = False

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        """选择损失函数（集成Barron损失）"""
        if getattr(self.args, 'use_adaptive_loss', False):
            print("🔧 使用Barron自适应损失函数")

            alpha_init = getattr(self.args, 'barron_alpha_init', 2.0)
            scale_init = getattr(self.args, 'barron_scale_init', 1.0)
            use_enhanced = getattr(self.args, 'use_enhanced_barron', True)

            if use_enhanced and self.use_progressive:
                criterion = EnhancedBarronLoss(
                    alpha_init=alpha_init,
                    scale_init=scale_init,
                    capacity_recovery_weight=getattr(self.args, 'capacity_recovery_weight', 0.1),
                    trend_consistency_weight=getattr(self.args, 'trend_consistency_weight', 0.05)
                )
                print("   使用增强版Barron损失函数（支持Progressive训练）")
            else:
                criterion = BarronAdaptiveLoss(alpha_init=alpha_init, scale_init=scale_init)
                print("   使用标准Barron自适应损失函数")

        else:
            print("📊 使用标准 MSE 损失函数")
            criterion = nn.MSELoss()

        return criterion

    def _select_optimizer(self):
        """选择优化器（包含损失函数参数）"""
        criterion = self._select_criterion()

        model_params = list(self.model.parameters())

        if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
            # 包含损失函数的自适应参数
            loss_params = list(criterion.parameters())

            # 为损失函数参数设置不同的学习率
            param_groups = [
                {'params': model_params, 'lr': self.args.learning_rate},
                {'params': loss_params, 'lr': self.args.learning_rate * 0.1}
            ]

            model_optim = optim.Adam(param_groups)
            print(f"✅ 优化器配置: 模型参数 {len(model_params)} + 损失参数 {len(loss_params)}")
        else:
            model_optim = optim.Adam(model_params, lr=self.args.learning_rate)
            print(f"📊 标准优化器: {len(model_params)} 个参数")

        self.criterion = criterion
        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
        """验证方法"""
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 计算损失
                if isinstance(criterion, EnhancedBarronLoss):
                    loss = criterion(outputs, batch_y, training_stage='all')
                else:
                    loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        _, _, _, _, _ = metric(preds, trues)
        r2 = r2_score(trues.flatten(), preds.flatten())

        self.model.train()
        return total_loss, r2

    def train(self, setting):
        """集成Barron损失函数和Progressive功能的训练方法"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        timestamped_setting = f"{setting}_{self.experiment_timestamp}"
        path = os.path.join(self.args.checkpoints, timestamped_setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=getattr(self.args, 'patience', 5), verbose=True)
        max_epochs = getattr(self.args, 'train_epochs', 20)

        model_optim = self._select_optimizer()
        criterion = self.criterion

        # Progressive调度器
        if self.use_progressive:
            scheduler = ProgressiveScheduler(
                total_epochs=max_epochs,
                hard_ratio=self.hard_ratio,
                progressive_ratio=self.progressive_ratio
            )

        # 学习率调度器设置（保持原有逻辑）
        if hasattr(self.args, 'lradj') and self.args.lradj == 'adaptive':
            lr_scheduler = AdaptiveLossLRScheduler(
                optimizer=model_optim,
                patience=1,
                factor=0.5,
                min_lr=1e-7,
                verbose=True,
                threshold=1e-4,
                cooldown=2
            )
            use_adaptive = True
        elif hasattr(self.args, 'lradj') and self.args.lradj == 'combined':
            lr_scheduler = CombinedLRScheduler(
                optimizer=model_optim,
                T_max=max_epochs,
                eta_min=1e-6,
                adaptive_patience=1,
                adaptive_factor=0.3,
                min_lr=1e-8,
                verbose=True
            )
            use_adaptive = True
        elif hasattr(self.args, 'lradj') and self.args.lradj == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            lr_scheduler = ReduceLROnPlateau(
                optimizer=model_optim,
                mode='min',
                factor=0.5,
                patience=1,
                verbose=True,
                threshold=1e-4,
                min_lr=1e-7
            )
            use_adaptive = True
        else:
            if hasattr(self.args, 'lradj') and self.args.lradj == 'TST':
                lr_scheduler = lr_scheduler.OneCycleLR(
                    optimizer=model_optim,
                    steps_per_epoch=train_steps,
                    pct_start=self.args.pct_start,
                    epochs=max_epochs,
                    max_lr=self.args.learning_rate
                )
            else:
                lr_scheduler = None
            use_adaptive = False

        # 训练历史记录
        train_history = {
            'train_loss': [], 'vali_loss': [], 'vali_r2': [], 'test_loss': [], 'test_r2': [],
            'learning_rate': []
        }

        if self.use_progressive:
            train_history.update({
                'training_stage': [], 'sample_count': [], 'alpha': []
            })

        # 添加Barron损失参数记录
        if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
            train_history.update({
                'barron_alpha': [], 'barron_scale': []
            })

        print(f"\n🎯 开始{'Progressive' if self.use_progressive else '传统'}训练 (总共 {max_epochs} 轮)...")
        print(
            f"   损失函数: {'Barron自适应' if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)) else 'MSE'}")

        # 主训练循环
        epoch_pbar = tqdm(range(max_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            current_stage = "traditional"
            current_alpha = 0.0
            current_sample_count = len(train_data)

            # Progressive训练逻辑
            if self.use_progressive:
                stage, alpha = scheduler.get_stage_and_alpha(epoch + 1)
                current_stage = stage
                current_alpha = alpha

                current_loader, sample_count = self.progressive_provider.get_progressive_dataloader(
                    train_data, stage, alpha, self.args.batch_size
                )
                current_sample_count = sample_count
                train_steps = len(current_loader)
            else:
                current_loader = train_loader

            # Batch训练循环
            batch_pbar = tqdm(current_loader, desc=f"Epoch {epoch + 1}/{max_epochs}",
                              leave=False, unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(batch_pbar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # ===== 核心：损失计算（集成Barron损失和Progressive阶段）=====
                if isinstance(criterion, EnhancedBarronLoss):
                    # 增强版Barron损失：传入训练阶段
                    loss = criterion(outputs, batch_y, training_stage=current_stage)
                elif isinstance(criterion, BarronAdaptiveLoss):
                    # 标准Barron损失
                    loss = criterion(outputs, batch_y)
                else:
                    # 标准损失函数
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                # 更新进度条显示
                current_lr = model_optim.param_groups[0]['lr']
                postfix = {'Loss': f"{loss.item():.6f}", 'LR': f"{current_lr:.2e}"}

                # 添加Barron参数显示
                if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
                    params = criterion.get_current_params()
                    postfix.update({
                        'α': f"{params['alpha']:.3f}",
                        'σ': f"{params['scale']:.3f}"
                    })

                if self.use_progressive:
                    postfix.update({
                        'Stage': current_stage[:4],
                        'Samples': current_sample_count
                    })

                batch_pbar.set_postfix(postfix)

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((max_epochs - epoch) * train_steps - i)

                    tqdm.write(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} | lr: {current_lr:.2e}")

                    if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
                        params = criterion.get_current_params()
                        tqdm.write(f"\tBarron参数: α={params['alpha']:.4f}, σ={params['scale']:.4f}")

                    if self.use_progressive:
                        tqdm.write(
                            f"\tstage: {current_stage} | samples: {current_sample_count} | α: {current_alpha:.2f}")

                    tqdm.write(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                # OneCycleLR调度
                if hasattr(self.args, 'lradj') and self.args.lradj == 'TST' and lr_scheduler is not None:
                    lr_scheduler.step()

            batch_pbar.close()

            # Epoch结束处理
            epoch_cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion)

            current_lr = model_optim.param_groups[0]['lr']

            # 记录历史
            train_history['train_loss'].append(train_loss)
            train_history['vali_loss'].append(vali_loss)
            train_history['vali_r2'].append(vali_r2)
            train_history['test_loss'].append(test_loss)
            train_history['test_r2'].append(test_r2)
            train_history['learning_rate'].append(current_lr)

            if self.use_progressive:
                train_history['training_stage'].append(current_stage)
                train_history['sample_count'].append(current_sample_count)
                train_history['alpha'].append(current_alpha)

            # 记录Barron参数
            if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
                params = criterion.get_current_params()
                train_history['barron_alpha'].append(params['alpha'])
                train_history['barron_scale'].append(params['scale'])

            # 学习率调度
            if use_adaptive:
                if hasattr(self.args, 'lradj') and self.args.lradj in ['adaptive', 'combined']:
                    lr_scheduler.step(vali_loss, epoch)
                elif hasattr(self.args, 'lradj') and self.args.lradj == 'plateau':
                    lr_scheduler.step(vali_loss)
            else:
                if hasattr(self.args, 'lradj') and self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args)

            new_lr = model_optim.param_groups[0]['lr']
            if abs(new_lr - current_lr) > 1e-10:
                tqdm.write(f"Learning rate changed from {current_lr:.2e} to {new_lr:.2e}")

            # 输出epoch结果
            if self.use_progressive:
                tqdm.write(f"Epoch: {epoch + 1} | Stage: {current_stage} | Samples: {current_sample_count}")
            else:
                tqdm.write(f"Epoch: {epoch + 1} cost time: {epoch_cost_time:.2f}s")

            tqdm.write(f"Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}")
            tqdm.write(f"Vali R²: {vali_r2:.4f} | Test R²: {test_r2:.4f} | LR: {new_lr:.2e}")

            # 输出Barron参数
            if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
                params = criterion.get_current_params()
                tqdm.write(f"Barron参数: α={params['alpha']:.4f}, σ={params['scale']:.4f}")

            # 更新epoch进度条
            postfix = {
                'Train_Loss': f"{train_loss:.6f}",
                'Vali_R2': f"{vali_r2:.4f}",
                'Test_R2': f"{test_r2:.4f}",
                'LR': f"{new_lr:.2e}"
            }

            if isinstance(criterion, (BarronAdaptiveLoss, EnhancedBarronLoss)):
                params = criterion.get_current_params()
                postfix.update({
                    'α': f"{params['alpha']:.3f}",
                    'σ': f"{params['scale']:.3f}"
                })

            if self.use_progressive:
                postfix['Stage'] = current_stage[:4]

            epoch_pbar.set_postfix(postfix)

            # 早停检查
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered!")
                tqdm.write(f"Training stopped at epoch {epoch + 1}")
                break

        epoch_pbar.close()

        # 保存训练历史
        self._save_training_history(train_history, path)

        # 保存学习率调度可视化
        if use_adaptive and isinstance(lr_scheduler, (AdaptiveLossLRScheduler, CombinedLRScheduler)):
            try:
                scheduler_state_file = os.path.join(path, f'scheduler_state_{self.experiment_timestamp}.json')
                if hasattr(lr_scheduler, 'state_dict'):
                    with open(scheduler_state_file, 'w') as f:
                        import json
                        json.dump(lr_scheduler.state_dict(), f, indent=2)

                lr_plot_path = os.path.join(path, f'lr_loss_history_{self.experiment_timestamp}.png')
                plot_lr_loss_history(lr_scheduler, lr_plot_path)
            except Exception as e:
                tqdm.write(f"Warning: Could not save scheduler visualization: {e}")

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 计算最终训练集R²
        tqdm.write("Computing final training metrics...")
        final_train_loss, final_train_r2 = self.train_step_metrics(train_loader, criterion)
        tqdm.write(f"Final Training R²: {final_train_r2:.6f}")

        # 输出Progressive训练总结
        if self.use_progressive:
            tqdm.write(f"\n🎯 Progressive训练总结:")
            tqdm.write(f"   难样本数: {self.sample_stats['hard_samples']}")
            tqdm.write(f"   回升区间: {self.sample_stats['recovery_windows']}")
            tqdm.write(f"   难样本比例: {self.sample_stats['hard_ratio']:.1%}")

    def _save_training_history(self, history, path):
        """保存训练历史"""
        import json

        history_file = os.path.join(path, f'training_history_{self.experiment_timestamp}.json')

        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list):
                serializable_history[key] = [float(x) if isinstance(x, (int, float)) else str(x) for x in value]
            else:
                serializable_history[key] = value

        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)

    def train_step_metrics(self, train_loader, criterion):
        """计算训练集上的R²指标（保持原有逻辑）"""
        train_preds = []
        train_trues = []
        train_loss = []

        self.model.eval()
        with torch.no_grad():
            metric_pbar = tqdm(enumerate(train_loader), desc="Computing train metrics",
                               total=min(100, len(train_loader)), leave=False)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in metric_pbar:
                if i >= 100:
                    break

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                train_preds.append(outputs.detach().cpu().numpy())
                train_trues.append(batch_y.detach().cpu().numpy())

            metric_pbar.close()

        if train_preds:
            train_preds = np.concatenate(train_preds, axis=0)
            train_trues = np.concatenate(train_trues, axis=0)
            _, _, _, _, _ = metric(train_preds, train_trues)
            train_r2 = r2_score(train_trues.flatten(), train_preds.flatten())
        else:
            train_r2 = 0.0

        self.model.train()
        return np.average(train_loss), train_r2

    def test(self, setting, test=0):
        """测试方法（保持原有完整逻辑）"""
        test_data, test_loader = self._get_data(flag='test')

        timestamped_setting = f"{setting}_{self.experiment_timestamp}"

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + timestamped_setting, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = './test_results/' + timestamped_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Testing", unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

            test_pbar.close()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        r2 = r2_score(trues.flatten(), preds.flatten())
        print(f'mse:{mse:.6f}, mae:{mae:.6f}')
        print(f'rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}')
        print(f'R²:{r2:.6f}')

        # 保存结果
        folder_path = './results/' + timestamped_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保持你原有的完整反归一化和可视化逻辑...
        # [这里保持你原来test方法中的所有反归一化和可视化代码不变]

        # 保存指标
        np.save(folder_path + f'metrics_{self.experiment_timestamp}.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + f'pred_{self.experiment_timestamp}.npy', preds)
        np.save(folder_path + f'true_{self.experiment_timestamp}.npy', trues)

        # 保存结果到文件
        # === 完整保留你原有的反归一化逻辑 ===
        print("开始反归一化处理...")
        print(f"原始preds形状: {preds.shape}")
        print(f"原始trues形状: {trues.shape}")

        # 获取原始测试数据
        raw_df = test_data.raw_test_df
        print("原始数据列名:", raw_df.columns.tolist())

        # 计算预测数据的数量
        num_preds = len(preds.flatten())
        print(f"预测数据点数量: {num_preds}")

        # 方法1：如果features=='S'（单变量），直接反归一化
        if self.args.features == 'S':
            print("使用单变量反归一化方法")
            preds_unscaled = test_data.inverse_transform(preds.reshape(-1, 1)).flatten()
            trues_unscaled = test_data.inverse_transform(trues.reshape(-1, 1)).flatten()
        else:
            print("使用多变量反归一化方法")
            feature_dim = test_data.data_x.shape[-1]
            print(f"特征维度: {feature_dim}")

            if hasattr(test_data, 'scaler'):
                test_mean = np.mean(test_data.data_x, axis=0)

                preds_full = np.tile(test_mean, (num_preds, 1))
                preds_full[:, -1] = preds.flatten()

                trues_full = np.tile(test_mean, (num_preds, 1))
                trues_full[:, -1] = trues.flatten()

                preds_unscaled = test_data.inverse_transform(preds_full)[:, -1]
                trues_unscaled = test_data.inverse_transform(trues_full)[:, -1]
            else:
                print("警告：无法找到scaler，使用原始数据")
                preds_unscaled = preds.flatten()
                trues_unscaled = trues.flatten()

        print(f"反归一化后预测值范围: [{preds_unscaled.min():.6f}, {preds_unscaled.max():.6f}]")
        print(f"反归一化后真实值范围: [{trues_unscaled.min():.6f}, {trues_unscaled.max():.6f}]")

        # 获取对应的cycle数据
        start_idx = self.args.seq_len
        end_idx = start_idx + num_preds

        print(f"从原始数据获取cycle，索引范围: {start_idx} 到 {end_idx}")
        print(f"原始数据长度: {len(raw_df)}")

        # 检查数据集中是否有Cycle列
        cycle_col = None
        date_col = None

        possible_cycle_names = ['Cycle', 'cycle', 'CYCLE', 'cycle_number', 'Cycle_Number']
        for col in possible_cycle_names:
            if col in raw_df.columns:
                cycle_col = col
                break

        possible_date_names = ['date', 'Date', 'DATE', 'time', 'Time', 'timestamp']
        for col in possible_date_names:
            if col in raw_df.columns:
                date_col = col
                break

        print(f"找到的Cycle列: {cycle_col}")
        print(f"找到的Date列: {date_col}")

        if end_idx <= len(raw_df):
            if cycle_col is not None:
                cycle_data = raw_df[cycle_col].values[start_idx:end_idx]
                print("使用数据集中的Cycle列")
            else:
                cycle_data = np.arange(start_idx + 1, end_idx + 1)
                print("数据集中没有Cycle列，使用生成的序号")

            if date_col is not None:
                date_data = raw_df[date_col].values[start_idx:end_idx]
            else:
                date_data = None

            true_targets = raw_df[self.args.target].values[start_idx:end_idx]
        else:
            print("警告：索引超出范围，调整索引")
            if cycle_col is not None:
                cycle_data = raw_df[cycle_col].values[-num_preds:]
            else:
                cycle_data = np.arange(len(raw_df) - num_preds + 1, len(raw_df) + 1)

            if date_col is not None:
                date_data = raw_df[date_col].values[-num_preds:]
            else:
                date_data = None

            true_targets = raw_df[self.args.target].values[-num_preds:]

        print(f"获取的cycle数据长度: {len(cycle_data)}")
        print(f"cycle数据类型: {type(cycle_data[0])}")
        print(f"cycle数据前5个: {cycle_data[:5]}")
        print(f"获取的真实target长度: {len(true_targets)}")
        print(f"真实target范围: [{true_targets.min():.6f}, {true_targets.max():.6f}]")

        min_length = min(len(cycle_data), len(true_targets), len(preds_unscaled))
        print(f"最终使用的数据长度: {min_length}")

        # 创建结果DataFrame时添加实验信息
        results_df = pd.DataFrame({
            'Cycle': cycle_data[:min_length],
            'True_Target': true_targets[:min_length],
            'Predicted_Target': preds_unscaled[:min_length]
        })

        # 添加实验元信息
        experiment_info = pd.DataFrame({
            'Experiment_Timestamp': [self.detailed_timestamp] * min_length,
            'Model_Name': [self.args.model] * min_length,
            'Dataset': [self.args.data] * min_length,
            'Data_Path': [self.args.data_path] * min_length,
            'Progressive_Training': [self.use_progressive] * min_length,
            'MSE': [mse] * min_length,
            'MAE': [mae] * min_length,
            'RMSE': [rmse] * min_length,
            'R2': [r2] * min_length,
            'Final_d_model': [self.args.d_model] * min_length,
            'Final_learning_rate': [self.args.learning_rate] * min_length,
            'Final_dropout': [self.args.dropout] * min_length,
        })

        # 如果使用了Progressive训练，添加相关信息
        if self.use_progressive:
            progressive_info = pd.DataFrame({
                'Hard_Samples': [self.sample_stats['hard_samples']] * min_length,
                'Hard_Ratio': [self.sample_stats['hard_ratio']] * min_length,
                'Recovery_Windows': [self.sample_stats['recovery_windows']] * min_length,
            })
            experiment_info = pd.concat([experiment_info, progressive_info], axis=1)

        # 合并实验信息和结果
        detailed_results_df = pd.concat([experiment_info, results_df], axis=1)

        # 保存详细结果CSV文件
        results_csv_path = os.path.join(folder_path, f'forecast_results_{self.experiment_timestamp}.csv')
        detailed_results_df.to_csv(results_csv_path, index=False)
        print(f"详细结果已保存至 {results_csv_path}")

        # 同时保存简化版本（保持原有格式兼容性）
        simple_results_csv_path = os.path.join(folder_path, 'forecast_results.csv')
        results_df.to_csv(simple_results_csv_path, index=False)
        print(f"简化结果已保存至 {simple_results_csv_path}")

        print("前5行结果预览:")
        print(results_df.head())

        print("开始生成可视化图表...")

        # 设置matplotlib的科技论文风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # 确定x轴数据：优先使用Cycle
        x_data = cycle_data[:min_length]
        x_label = 'Cycle'

        print(f"使用x轴数据: {x_label}")
        print(f"x轴数据范围: {x_data.min()} 到 {x_data.max()}")

        # 绘制真实值（蓝线）
        ax.plot(x_data, true_targets[:min_length],
                color='#2E86AB', linewidth=2.5, alpha=0.8,
                label='True SoH', marker='o', markersize=3, markevery=max(1, min_length // 50))

        # 绘制预测值（红线）
        ax.plot(x_data, preds_unscaled[:min_length],
                color='#F24236', linewidth=2.5, alpha=0.8,
                label='Predicted SoH', marker='s', markersize=3, markevery=max(1, min_length // 50))

        # 设置标题和标签
        title_suffix = " (Progressive Training)" if self.use_progressive else " (Traditional Training)"
        ax.set_title(f'Battery SoH Prediction - Enhanced Model{title_suffix}',
                     fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('State of Health (SoH)', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')

        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # 添加图例（右上角）
        legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                           fancybox=True, shadow=True, framealpha=0.9,
                           bbox_to_anchor=(0.98, 0.98))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.8)

        # 添加性能指标文本框（右上角，图例下方）
        textstr = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
        if self.use_progressive:
            textstr += f'\nHard Samples: {self.sample_stats["hard_samples"]} ({self.sample_stats["hard_ratio"]:.1%})'

        props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8, edgecolor='gray')
        ax.text(0.98, 0.75, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=props, family='monospace')

        # 设置轴的范围和刻度
        y_min, y_max = min(np.min(true_targets[:min_length]), np.min(preds_unscaled[:min_length])), \
            max(np.max(true_targets[:min_length]), np.max(preds_unscaled[:min_length]))
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

        # 美化刻度
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
        ax.tick_params(axis='both', which='minor', width=0.8, length=3)

        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')

        # 添加阴影区域显示预测误差
        error = np.abs(true_targets[:min_length] - preds_unscaled[:min_length])
        ax.fill_between(x_data,
                        preds_unscaled[:min_length] - error / 2,
                        preds_unscaled[:min_length] + error / 2,
                        alpha=0.2, color='red', label='Prediction Error Band')

        # 调整布局
        plt.tight_layout()

        # 保存高质量图片
        plot_name = 'progressive_battery_soh_prediction.png' if self.use_progressive else 'traditional_battery_soh_prediction.png'
        plot_path = os.path.join(folder_path, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # 同时保存PDF格式（适合论文使用）
        pdf_name = 'progressive_battery_soh_prediction.pdf' if self.use_progressive else 'traditional_battery_soh_prediction.pdf'
        pdf_path = os.path.join(folder_path, pdf_name)
        plt.savefig(pdf_path, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"可视化图表已保存:")
        print(f"PNG格式: {plot_path}")
        print(f"PDF格式: {pdf_path}")

        # 显示图表（可选，如果在jupyter notebook中运行）
        plt.show()
        plt.close()

        # =================== 可视化功能结束 ===================

        # 保存结果到文件
        result_file = f"result_fused_forecast_{self.experiment_timestamp}.txt"
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(f"Experiment Time: {self.detailed_timestamp}\n")
            f.write(f"Dataset: {self.args.data_path}\n")
            f.write(f"Model: {self.args.model}\n")
            f.write(f"Progressive Training: {'Enabled' if self.use_progressive else 'Disabled'}\n")
            if self.use_progressive:
                f.write(
                    f"Hard Samples: {self.sample_stats['hard_samples']}/{self.sample_stats['total_samples']} ({self.sample_stats['hard_ratio']:.1%})\n")
                f.write(f"Recovery Windows: {self.sample_stats['recovery_windows']}\n")
            f.write(timestamped_setting + "\n")
            f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
            f.write('\n')

        # 同时保存到总的结果文件（保持原有逻辑）
        general_result_file = "result_fused_forecast.txt"
        with open(general_result_file, 'a', encoding='utf-8') as f:
            f.write(f"[{self.detailed_timestamp}] " + timestamped_setting + "\n")
            if self.use_progressive:
                f.write(
                    f"Progressive: {self.sample_stats['hard_samples']} hard samples ({self.sample_stats['hard_ratio']:.1%})\n")
            f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
            f.write('\n')

        return


# ================================
# Progressive训练参数设置函数
# ================================

def setup_progressive_args(args):
    """为args添加Progressive训练参数"""

    # Progressive训练开关
    args.use_progressive_training = True  # 默认关闭，通过命令行参数控制
    args.marked_data_path = 'marked_battery_data.csv'  # 预处理生成的标记文件

    # Progressive训练阶段比例
    args.hard_ratio = 0.3  # 30%的epochs用于难样本训练
    args.progressive_ratio = 0.5  # 50%的epochs用于渐进训练

    print("✅ Progressive训练参数已添加到args中")
    print(f"   默认状态: {'启用' if args.use_progressive_training else '禁用'}")
    print(f"   标记数据路径: {args.marked_data_path}")
    print(f"   难样本阶段比例: {args.hard_ratio}")
    print(f"   渐进阶段比例: {args.progressive_ratio}")

    return args