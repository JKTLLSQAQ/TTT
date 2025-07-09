from exp.exp_basic import Exp_Basic
from models import dual,redual,STAR,FusedTimeModel  # 修改：导入新的双分支模型
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
from exp.learning_rate import (AdaptiveLossLRScheduler,CombinedLRScheduler,plot_lr_loss_history,)
import math
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F


class BarronAdaptiveLoss(nn.Module):
    """Barron自适应损失函数核心"""

    def __init__(self, alpha_init=2.0, scale_init=1.0):
        super().__init__()
        self._alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self._scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    @property
    def alpha(self):
        return F.softplus(self._alpha) + 0.01

    @property
    def scale(self):
        return F.softplus(self._scale) + 1e-6

    def forward(self, pred, target):
        residual = pred - target
        alpha = self.alpha
        scale = self.scale

        normalized_residual = residual / scale

        if torch.abs(alpha - 2.0) < 1e-4:
            loss = 0.5 * normalized_residual.pow(2)
        else:
            abs_alpha_minus_2 = torch.abs(alpha - 2.0)
            inner_term = (normalized_residual.pow(2) / abs_alpha_minus_2 + 1).pow(alpha / 2) - 1
            loss = (abs_alpha_minus_2 / alpha) * inner_term

        return loss.mean()


class DebugOptimizedCompleteLoss(nn.Module):
    """调试版自适应损失函数"""

    def __init__(self, alpha_init=2.0, scale_init=1.0, recovery_weight=0.3, trend_weight=0.1):
        super().__init__()
        self.barron_loss = BarronAdaptiveLoss(alpha_init, scale_init)
        self.recovery_weight = recovery_weight
        self.trend_weight = trend_weight

        # 🔥 基于你的数据分析的优化阈值
        self.recovery_thresholds = {
            'conservative': 0.000008,  # 89.81% 回升检测率
            'balanced': 0.000049,  # 74.93% 回升检测率
            'strict': 0.000470,  # 49.86% 回升检测率
            'very_strict': 0.001000,  # 39.94% 回升检测率
            'debug': 0.000001  # 超保守策略，用于调试
        }

        # 默认使用调试策略
        self.current_strategy = 'debug'
        self.recovery_threshold = self.recovery_thresholds[self.current_strategy]

        # 统计计数器
        self.total_calls = 0
        self.recovery_detections = 0
        self.debug_counter = 0
        self.strategy_stats = {strategy: 0 for strategy in self.recovery_thresholds.keys()}

        print(f"🚀 初始化调试版自适应损失函数")
        print(f"   策略: {self.current_strategy}")
        print(f"   阈值: {self.recovery_threshold:.6f}")
        print(f"   权重: recovery={recovery_weight}, trend={trend_weight}")

    def set_detection_strategy(self, strategy='debug'):
        """设置检测策略"""
        if strategy in self.recovery_thresholds:
            self.current_strategy = strategy
            self.recovery_threshold = self.recovery_thresholds[strategy]
            print(f"🔄 切换检测策略: {strategy} (阈值: {self.recovery_threshold:.6f})")
        else:
            print(f"❌ 未知策略: {strategy}")
            print(f"   可用策略: {list(self.recovery_thresholds.keys())}")

    def set_custom_threshold(self, threshold):
        """设置自定义阈值"""
        self.recovery_threshold = threshold
        self.current_strategy = 'custom'
        print(f"🎯 设置自定义阈值: {threshold:.6f}")

    def forward(self, pred, true):
        self.total_calls += 1
        self.debug_counter += 1

        # 基础自适应损失
        base_loss = self.barron_loss(pred, true)
        total_loss = base_loss

        # 🔍 详细调试信息
        debug_info = {
            'pred_shape': pred.shape,
            'true_shape': true.shape,
            'pred_device': pred.device,
            'true_device': true.device,
            'pred_range': (pred.min().item(), pred.max().item()),
            'true_range': (true.min().item(), true.max().item()),
            'has_time_dim': pred.size(1) > 1,
            'detected': False,
            'positive_changes': 0,
            'max_change': 0.0,
            'min_change': 0.0,
            'total_changes': 0
        }

        # 容量回升检测和额外惩罚
        if pred.size(1) > 1:
            # 计算真实值的变化率
            true_diff = torch.diff(true, dim=1)
            pred_diff = torch.diff(pred, dim=1)

            # 更新调试信息
            debug_info['diff_shape'] = true_diff.shape
            debug_info['diff_device'] = true_diff.device
            debug_info['diff_range'] = (true_diff.min().item(), true_diff.max().item())
            debug_info['total_changes'] = true_diff.numel()

            # 统计正变化
            positive_changes = true_diff[true_diff > 0]
            debug_info['positive_changes'] = len(positive_changes)

            if len(positive_changes) > 0:
                debug_info['max_change'] = positive_changes.max().item()
                debug_info['min_change'] = positive_changes.min().item()
                debug_info['avg_change'] = positive_changes.mean().item()

                # 测试不同阈值
                for test_threshold in [0.000001, 0.000008, 0.000049, 0.001]:
                    count = (positive_changes > test_threshold).sum().item()
                    debug_info[f'threshold_{test_threshold:.6f}'] = count

            # 使用当前阈值检测回升
            recovery_mask = (true_diff > self.recovery_threshold).float()
            debug_info['recovery_mask_sum'] = recovery_mask.sum().item()
            debug_info['threshold_used'] = self.recovery_threshold

            if recovery_mask.sum() > 0:
                debug_info['detected'] = True
                self.recovery_detections += 1
                self.strategy_stats[self.current_strategy] += 1

                # 回升损失计算
                recovery_error = F.mse_loss(
                    pred[:, 1:] * recovery_mask,
                    true[:, 1:] * recovery_mask,
                    reduction='mean'
                )

                pred_trend = torch.sign(pred_diff)
                true_trend = torch.sign(true_diff)
                trend_consistency = F.mse_loss(
                    pred_trend * recovery_mask,
                    true_trend * recovery_mask,
                    reduction='mean'
                )

                recovery_magnitude_error = F.l1_loss(
                    torch.abs(pred_diff) * recovery_mask,
                    torch.abs(true_diff) * recovery_mask,
                    reduction='mean'
                )

                total_loss = (base_loss +
                              self.recovery_weight * recovery_error +
                              self.trend_weight * trend_consistency +
                              0.1 * recovery_magnitude_error)

                debug_info['recovery_loss'] = recovery_error.item()
                debug_info['trend_loss'] = trend_consistency.item()
                debug_info['magnitude_loss'] = recovery_magnitude_error.item()

        # 每10次调用输出一次详细调试信息
        if self.debug_counter % 10 == 0:
            self._print_debug_info(debug_info)

        # 每100次调用输出统计信息
        if self.total_calls % 100 == 0:
            self._print_statistics()

        return total_loss

    def _print_debug_info(self, debug_info):
        """输出详细调试信息"""
        print(f"\n🔬 调试信息 (调用: {self.total_calls})")
        print(f"   📊 数据形状: pred={debug_info['pred_shape']}, true={debug_info['true_shape']}")
        print(f"   💾 设备: pred={debug_info['pred_device']}, true={debug_info['true_device']}")
        print(f"   📈 数据范围: pred=[{debug_info['pred_range'][0]:.6f}, {debug_info['pred_range'][1]:.6f}]")
        print(f"               true=[{debug_info['true_range'][0]:.6f}, {debug_info['true_range'][1]:.6f}]")

        if debug_info['has_time_dim']:
            print(f"   🔍 差分分析:")
            print(f"      差分形状: {debug_info['diff_shape']}")
            print(f"      差分范围: [{debug_info['diff_range'][0]:.6f}, {debug_info['diff_range'][1]:.6f}]")
            print(f"      总变化数: {debug_info['total_changes']}")
            print(f"      正变化数: {debug_info['positive_changes']}")

            if debug_info['positive_changes'] > 0:
                print(f"      正变化范围: [{debug_info['min_change']:.6f}, {debug_info['max_change']:.6f}]")
                print(f"      平均正变化: {debug_info.get('avg_change', 0):.6f}")

                print(f"   🎯 不同阈值测试:")
                for threshold in [0.000001, 0.000008, 0.000049, 0.001]:
                    count = debug_info.get(f'threshold_{threshold:.6f}', 0)
                    print(f"      阈值 {threshold:.6f}: {count} 个检测点")

            print(f"   🚨 当前检测结果:")
            print(f"      使用阈值: {debug_info['threshold_used']:.6f}")
            print(f"      检测到回升: {'是' if debug_info['detected'] else '否'}")
            print(f"      回升掩码总和: {debug_info['recovery_mask_sum']}")

            if debug_info['detected']:
                print(f"      回升损失: {debug_info.get('recovery_loss', 0):.6f}")
                print(f"      趋势损失: {debug_info.get('trend_loss', 0):.6f}")
        else:
            print(f"   ⚠️  序列长度不足，无法计算差分")

    def _print_statistics(self):
        """输出统计信息"""
        alpha_val = self.barron_loss.alpha.item()
        scale_val = self.barron_loss.scale.item()
        detection_rate = self.recovery_detections / self.total_calls * 100

        print(f"\n📊 累计统计 (调用: {self.total_calls})")
        print(f"   自适应参数: α={alpha_val:.4f}, σ={scale_val:.4f}")
        print(f"   回升检测: {self.recovery_detections}次 ({detection_rate:.2f}%)")
        print(f"   策略统计: {dict(self.strategy_stats)}")

    def get_comprehensive_stats(self):
        """获取全面的统计信息"""
        return {
            'loss_params': {
                'alpha': self.barron_loss.alpha.item(),
                'scale': self.barron_loss.scale.item(),
                'alpha_raw': self.barron_loss._alpha.item(),
                'scale_raw': self.barron_loss._scale.item()
            },
            'detection_stats': {
                'total_calls': self.total_calls,
                'recovery_detections': self.recovery_detections,
                'detection_rate': self.recovery_detections / max(1, self.total_calls) * 100,
                'current_strategy': self.current_strategy,
                'current_threshold': self.recovery_threshold,
                'strategy_stats': dict(self.strategy_stats)
            },
            'thresholds': dict(self.recovery_thresholds)
        }

    def get_params(self):
        """获取当前参数"""
        return {
            'alpha': self.barron_loss.alpha.item(),
            'scale': self.barron_loss.scale.item(),
            'alpha_raw': self.barron_loss._alpha.item(),
            'scale_raw': self.barron_loss._scale.item(),
            'recovery_detections': self.recovery_detections,
            'total_calls': self.total_calls
        }

    def reset_stats(self):
        """重置统计"""
        self.total_calls = 0
        self.recovery_detections = 0
        self.debug_counter = 0
        self.strategy_stats = {strategy: 0 for strategy in self.recovery_thresholds.keys()}


def print_final_recovery_report(criterion):
    """打印最终的回升检测报告"""
    if isinstance(criterion, OptimizedCompleteLoss):
        stats = criterion.get_comprehensive_stats()

        print(f"\n" + "=" * 60)
        print(f"🎯 最终回升检测报告")
        print(f"=" * 60)

        # 损失参数变化
        loss_params = stats['loss_params']
        print(f"📊 自适应损失参数:")
        print(f"   最终 α: {loss_params['alpha']:.4f}")
        print(f"   最终 σ: {loss_params['scale']:.4f}")

        # 检测统计
        detection_stats = stats['detection_stats']
        print(f"\n🔍 回升检测统计:")
        print(f"   总调用次数: {detection_stats['total_calls']}")
        print(f"   回升检测次数: {detection_stats['recovery_detections']}")
        print(f"   检测率: {detection_stats['detection_rate']:.2f}%")
        print(f"   最终策略: {detection_stats['current_strategy']}")
        print(f"   最终阈值: {detection_stats['current_threshold']:.6f}")

        # 策略使用统计
        print(f"\n📈 策略使用统计:")
        for strategy, count in detection_stats['strategy_stats'].items():
            percentage = count / max(1, detection_stats['recovery_detections']) * 100
            print(f"   {strategy:12s}: {count:6d} 次 ({percentage:5.1f}%)")

        print(f"=" * 60)


def generate_timestamp():
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_detailed_timestamp():
    """生成详细的时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Exp_Fused_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Fused_Forecast, self).__init__(args)
        self.experiment_timestamp = generate_timestamp()
        self.detailed_timestamp = generate_detailed_timestamp()
        print(f"实验时间戳: {self.experiment_timestamp}")
        print(f"详细时间戳: {self.detailed_timestamp}")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        """调试版损失函数选择"""
        if getattr(self.args, 'use_adaptive_loss', False):
            print("🔍 使用调试版自适应损失函数")

            strategy = getattr(self.args, 'recovery_strategy', 'debug')  # 默认使用debug策略
            custom_threshold = getattr(self.args, 'custom_recovery_threshold', None)

            debug_loss = DebugOptimizedCompleteLoss(
                alpha_init=2.0,
                scale_init=1.0,
                recovery_weight=getattr(self.args, 'recovery_weight', 0.3),
                trend_weight=getattr(self.args, 'trend_weight', 0.1)
            )

            # 设置检测策略
            if custom_threshold is not None:
                debug_loss.set_custom_threshold(custom_threshold)
            else:
                debug_loss.set_detection_strategy(strategy)

            if hasattr(self, 'device'):
                debug_loss = debug_loss.to(self.device)
            elif torch.cuda.is_available():
                debug_loss = debug_loss.cuda()

            return debug_loss
        else:
            print("📊 使用标准 MSE 损失函数")
            return nn.MSELoss()

    def _select_optimizer(self):
        """修改优化器选择，包含损失函数参数"""
        # 先获取损失函数
        criterion = self._select_criterion()

        # 收集所有需要优化的参数
        model_params = list(self.model.parameters())

        if isinstance(criterion, DebugOptimizedCompleteLoss):
            # 包含损失函数的参数
            loss_params = list(criterion.parameters())
            all_params = model_params + loss_params
            print(f"✅ 优化器包含: 模型参数 {len(model_params)} + 损失参数 {len(loss_params)} = {len(all_params)}")
        else:
            all_params = model_params
            print(f"📊 优化器包含: 模型参数 {len(model_params)}")

        model_optim = optim.Adam(all_params, lr=self.args.learning_rate)

        # 保存criterion以供后续使用
        self.criterion = criterion

        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
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

                # 模型前向
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 直接调用损失函数
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                # 收集预测和真实值用于计算R²
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        total_loss = np.average(total_loss)

        # 计算验证集上的R²
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        _, _, _, _, _ = metric(preds, trues)
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())

        self.model.train()
        return total_loss, r2

    def train(self, setting):
        """修改后的训练方法，完整集成自适应损失函数"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        timestamped_setting = f"{setting}_{self.experiment_timestamp}"
        path = os.path.join(self.args.checkpoints, timestamped_setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=5, verbose=True)
        max_epochs = 20

        # 重要：先选择优化器（内部会调用_select_criterion）
        model_optim = self._select_optimizer()
        criterion = self.criterion  # 使用保存的criterion

        # 输出损失函数信息
        if isinstance(criterion, DebugOptimizedCompleteLoss):
            print("✅ 启用完整自适应损失函数，包含容量回升检测")
            print(f"📋 初始参数: α={criterion.get_params()['alpha']:.3f}, σ={criterion.get_params()['scale']:.3f}")
        else:
            print("📌 使用标准MSE损失函数")

        # ===== 学习率调度器设置 =====
        if hasattr(self.args, 'lradj') and self.args.lradj == 'adaptive':
            scheduler = AdaptiveLossLRScheduler(
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
            scheduler = CombinedLRScheduler(
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
            scheduler = ReduceLROnPlateau(
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
                scheduler = lr_scheduler.OneCycleLR(
                    optimizer=model_optim,
                    steps_per_epoch=train_steps,
                    pct_start=self.args.pct_start,
                    epochs=max_epochs,
                    max_lr=self.args.learning_rate
                )
            else:
                scheduler = None
            use_adaptive = False

        # 记录训练历史
        train_history = {
            'train_loss': [],
            'vali_loss': [],
            'vali_r2': [],
            'test_loss': [],
            'test_r2': [],
            'learning_rate': []
        }

        # 如果使用自适应损失，记录损失参数变化
        if isinstance(criterion, DebugOptimizedCompleteLoss):
            train_history['loss_alpha'] = []
            train_history['loss_scale'] = []
            train_history['recovery_detections'] = []

        # 创建总的epoch进度条
        epoch_pbar = tqdm(range(max_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # 重置损失函数统计（每个epoch开始）
            if isinstance(criterion, DebugOptimizedCompleteLoss):
                criterion.reset_stats()

            # 为每个epoch的batch创建进度条
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}",
                              leave=False, unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(batch_pbar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 前向传播
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 计算损失
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # 更新进度条显示
                current_lr = model_optim.param_groups[0]['lr']

                if isinstance(criterion, DebugOptimizedCompleteLoss):
                    params = criterion.get_params()
                    batch_pbar.set_postfix({
                        'Loss': f"{loss.item():.6f}",
                        'LR': f"{current_lr:.2e}",
                        'α': f"{params['alpha']:.3f}",
                        'σ': f"{params['scale']:.3f}"
                    })
                else:
                    batch_pbar.set_postfix({
                        'Loss': f"{loss.item():.6f}",
                        'LR': f"{current_lr:.2e}"
                    })

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((max_epochs - epoch) * train_steps - i)
                    tqdm.write(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f} | lr: {current_lr:.2e}")
                    tqdm.write(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

                    # 输出自适应损失参数
                    if isinstance(criterion, DebugOptimizedCompleteLoss):
                        params = criterion.get_params()
                        tqdm.write(
                            f'\t自适应损失: α={params["alpha"]:.4f}, σ={params["scale"]:.4f}, 回升检测={params["recovery_detections"]}次')

                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                # 对于OneCycleLR，在每个batch后调用
                if hasattr(self.args, 'lradj') and self.args.lradj == 'TST' and scheduler is not None:
                    scheduler.step()

            batch_pbar.close()

            epoch_cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion)

            # 记录当前学习率
            current_lr = model_optim.param_groups[0]['lr']

            # 记录历史
            train_history['train_loss'].append(train_loss)
            train_history['vali_loss'].append(vali_loss)
            train_history['vali_r2'].append(vali_r2)
            train_history['test_loss'].append(test_loss)
            train_history['test_r2'].append(test_r2)
            train_history['learning_rate'].append(current_lr)

            # 如果使用自适应损失，记录损失参数
            if isinstance(criterion, DebugOptimizedCompleteLoss):
                params = criterion.get_params()
                train_history['loss_alpha'].append(params['alpha'])
                train_history['loss_scale'].append(params['scale'])
                train_history['recovery_detections'].append(params['recovery_detections'])

            # ===== 学习率调度 =====
            if use_adaptive:
                if hasattr(self.args, 'lradj') and self.args.lradj in ['adaptive', 'combined']:
                    scheduler.step(vali_loss, epoch)
                elif hasattr(self.args, 'lradj') and self.args.lradj == 'plateau':
                    scheduler.step(vali_loss)
            else:
                if hasattr(self.args, 'lradj') and self.args.lradj != 'TST':
                    adjust_learning_rate(model_optim, epoch + 1, self.args)

            # 检查学习率变化
            new_lr = model_optim.param_groups[0]['lr']
            if abs(new_lr - current_lr) > 1e-10:
                tqdm.write(f"Learning rate changed from {current_lr:.2e} to {new_lr:.2e}")

            # 输出epoch结果
            tqdm.write(f"Epoch: {epoch + 1} cost time: {epoch_cost_time:.2f}s")
            tqdm.write(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            tqdm.write(f"Vali R²: {vali_r2:.4f} Test R²: {test_r2:.4f} | LR: {new_lr:.2e}")

            # 输出自适应损失参数变化
            if isinstance(criterion, DebugOptimizedCompleteLoss):
                params = criterion.get_params()
                tqdm.write(
                    f"自适应损失参数: α={params['alpha']:.4f}, σ={params['scale']:.4f}, 本epoch回升检测={params['recovery_detections']}次")

            # 更新epoch进度条
            if isinstance(criterion, DebugOptimizedCompleteLoss):
                params = criterion.get_params()
                epoch_pbar.set_postfix({
                    'Train_Loss': f"{train_loss:.6f}",
                    'Vali_R2': f"{vali_r2:.4f}",
                    'Test_R2': f"{test_r2:.4f}",
                    'α': f"{params['alpha']:.3f}",
                    'σ': f"{params['scale']:.3f}"
                })
            else:
                epoch_pbar.set_postfix({
                    'Train_Loss': f"{train_loss:.6f}",
                    'Vali_R2': f"{vali_r2:.4f}",
                    'Test_R2': f"{test_r2:.4f}",
                    'LR': f"{new_lr:.2e}"
                })

            # 早停检查
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered!")
                tqdm.write(f"Training stopped at epoch {epoch + 1}")
                break

        epoch_pbar.close()

        # 保存训练历史
        history_file = os.path.join(path, f'training_history_{self.experiment_timestamp}.json')
        train_history_serializable = {}
        for key, value in train_history.items():
            if isinstance(value[0], (int, float)):
                train_history_serializable[key] = [float(x) for x in value]
            else:
                train_history_serializable[key] = value

        import json
        with open(history_file, 'w') as f:
            json.dump(train_history_serializable, f, indent=2)

        # 如果使用自适应调度器，保存可视化
        if use_adaptive and isinstance(scheduler, (AdaptiveLossLRScheduler, CombinedLRScheduler)):
            try:
                scheduler_state_file = os.path.join(path, f'scheduler_state_{self.experiment_timestamp}.json')
                if hasattr(scheduler, 'state_dict'):
                    with open(scheduler_state_file, 'w') as f:
                        json.dump(scheduler.state_dict(), f, indent=2)

                lr_plot_path = os.path.join(path, f'lr_loss_history_{self.experiment_timestamp}.png')
                plot_lr_loss_history(scheduler, lr_plot_path)
            except Exception as e:
                tqdm.write(f"Warning: Could not save scheduler visualization: {e}")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 最终训练完成后计算训练集R²
        tqdm.write("Computing final training metrics...")
        final_train_loss, final_train_r2 = self.train_step_metrics(train_loader, criterion)
        tqdm.write(f"Final Training R²: {final_train_r2:.6f}")

        # 输出学习率调度总结
        if use_adaptive:
            final_lr = model_optim.param_groups[0]['lr']
            initial_lr = self.args.learning_rate
            tqdm.write(f"Learning rate summary: Initial: {initial_lr:.2e}, Final: {final_lr:.2e}")
            if hasattr(scheduler, 'lr_history') and scheduler.lr_history:
                min_lr = min(scheduler.lr_history)
                tqdm.write(f"Minimum learning rate reached: {min_lr:.2e}")

        # 输出自适应损失总结
        if isinstance(criterion, DebugOptimizedCompleteLoss):
            print_final_recovery_report(criterion)

    def train_step_metrics(self, train_loader, criterion):
        """计算训练集上的R²指标"""
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
            from sklearn.metrics import r2_score
            train_r2 = r2_score(train_trues.flatten(), train_preds.flatten())
        else:
            train_r2 = 0.0

        self.model.train()
        return np.average(train_loss), train_r2

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # 修改：添加时间戳到设置
        timestamped_setting = f"{setting}_{self.experiment_timestamp}"

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + timestamped_setting, 'checkpoint.pth')))

        preds = []
        trues = []

        # 修改：创建带时间戳的测试结果文件夹
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
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())
        print(f'mse:{mse:.6f}, mae:{mae:.6f}')
        print(f'rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}')
        print(f'R²:{r2:.6f}')

        # 修改：保存结果到带时间戳的文件夹
        folder_path = './results/' + timestamped_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # === 反归一化逻辑保持不变 ===
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

        # 修改：创建结果DataFrame时添加实验信息
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
            'MSE': [mse] * min_length,
            'MAE': [mae] * min_length,
            'RMSE': [rmse] * min_length,
            'R2': [r2] * min_length,
            'Final_d_model': [self.args.d_model] * min_length,
            'Final_learning_rate': [self.args.learning_rate] * min_length,
            'Final_dropout': [self.args.dropout] * min_length,
        })

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

        # 确定x轴数据：优先使用Cycle，如果用户明确需要date则使用date
        # 这里我们使用Cycle作为x轴，因为这是电池研究的标准做法
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
        ax.set_title('Battery SoH Prediction - Dual Branch Model', fontsize=20, fontweight='bold', pad=20)
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
        plot_path = os.path.join(folder_path, 'dual_branch_battery_soh_prediction.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # 同时保存PDF格式（适合论文使用）
        pdf_path = os.path.join(folder_path, 'dual_branch_battery_soh_prediction.pdf')
        plt.savefig(pdf_path, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"可视化图表已保存:")
        print(f"PNG格式: {plot_path}")
        print(f"PDF格式: {pdf_path}")

        # 显示图表（可选，如果在jupyter notebook中运行）
        plt.show()
        plt.close()

        # =================== 可视化功能结束 ===================

        # 保存指标
        np.save(folder_path + f'metrics_{self.experiment_timestamp}.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + f'pred_{self.experiment_timestamp}.npy', preds)
        np.save(folder_path + f'true_{self.experiment_timestamp}.npy', trues)

        # 修改：保存到带时间戳的文本文件
        result_file = f"result_dual_branch_forecast_{self.experiment_timestamp}.txt"
        f = open(result_file, 'a', encoding='utf-8')
        f.write(f"Experiment Time: {self.detailed_timestamp}\n")
        f.write(f"Dataset: {self.args.data_path}\n")
        f.write(f"Model: {self.args.model}\n")
        f.write(timestamped_setting + "\n")
        f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        # 同时保存到总的结果文件（保持原有逻辑）
        f = open("result_dual_branch_forecast.txt", 'a', encoding='utf-8')
        f.write(f"[{self.detailed_timestamp}] " + timestamped_setting + "\n")
        f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        return