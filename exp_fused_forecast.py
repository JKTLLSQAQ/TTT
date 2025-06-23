from exp.exp_basic import Exp_Basic
from models import dual  # 修改：导入新的双分支模型
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
import copy
import json

warnings.filterwarnings('ignore')


def generate_timestamp():
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_detailed_timestamp():
    """生成详细的时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class HarrisHawkOptimizer:
    """哈里斯鹰优化算法用于超参数优化"""

    def __init__(self, dim, pop_size=10, max_iter=20, bounds=None):
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds if bounds else [(-1, 1)] * dim

        # 初始化种群
        self.positions = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for j in range(dim):
                self.positions[i][j] = np.random.uniform(self.bounds[j][0], self.bounds[j][1])

        self.fitness = np.full(pop_size, float('inf'))
        self.best_position = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

    def optimize(self, objective_func):
        """执行优化过程"""
        print("开始哈里斯鹰优化...")

        for t in range(self.max_iter):
            print(f"优化迭代 {t + 1}/{self.max_iter}")

            # 评估当前种群
            for i in range(self.pop_size):
                fitness = objective_func(self.positions[i])
                if fitness < self.fitness[i]:
                    self.fitness[i] = fitness
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_position = self.positions[i].copy()

            self.convergence_curve.append(self.best_fitness)
            print(f"当前最佳适应度: {self.best_fitness:.6f}")

            # 更新位置
            E0 = 2 * np.random.random() - 1  # 能量参数
            E = 2 * E0 * (1 - t / self.max_iter)  # 动态能量

            for i in range(self.pop_size):
                if abs(E) >= 1:  # 探索阶段
                    # 随机选择一个鹰
                    rand_idx = np.random.randint(0, self.pop_size)
                    r1, r2 = np.random.random(), np.random.random()

                    if r1 >= 0.5:
                        # 策略1
                        self.positions[i] = self.positions[rand_idx] - r1 * abs(
                            self.positions[rand_idx] - 2 * r2 * self.positions[i])
                    else:
                        # 策略2
                        mean_pos = np.mean(self.positions, axis=0)
                        self.positions[i] = mean_pos - r1 * abs(
                            mean_pos - 2 * r2 * self.positions[i])

                else:  # 开发阶段
                    r = np.random.random()
                    if r >= 0.5 and abs(E) >= 0.5:  # 软围攻
                        delta_X = self.best_position - self.positions[i]
                        self.positions[i] = delta_X - E * abs(
                            np.random.random() * self.best_position - self.positions[i])
                    elif r >= 0.5 and abs(E) < 0.5:  # 硬围攻
                        self.positions[i] = self.best_position - E * abs(
                            self.best_position - self.positions[i])
                    elif r < 0.5 and abs(E) >= 0.5:  # 软围攻（更复杂）
                        S = np.random.random(self.dim) * 2 - 1
                        self.positions[i] = self.best_position - E * abs(
                            np.random.random() * self.best_position - self.positions[i]) + \
                                            np.random.random() * S
                    else:  # 硬围攻（更复杂）
                        S = np.random.random(self.dim) * 2 - 1
                        self.positions[i] = self.best_position - E * abs(
                            self.best_position - self.positions[i]) - np.random.random() * S

                # 边界处理
                for j in range(self.dim):
                    if self.positions[i][j] < self.bounds[j][0]:
                        self.positions[i][j] = self.bounds[j][0]
                    elif self.positions[i][j] > self.bounds[j][1]:
                        self.positions[i][j] = self.bounds[j][1]

        print(f"优化完成！最佳适应度: {self.best_fitness:.6f}")
        return self.best_position, self.best_fitness


class Exp_Enhanced_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Enhanced_Forecast, self).__init__(args)
        self.experiment_timestamp = generate_timestamp()
        self.detailed_timestamp = generate_detailed_timestamp()
        self.optimized_params = None
        self.use_optimization = getattr(args, 'use_hho_optimization', False)
        self.optimization_epochs = getattr(args, 'optimization_epochs', 5)
        print(f"实验时间戳: {self.experiment_timestamp}")
        print(f"详细时间戳: {self.detailed_timestamp}")

        # 哈里斯鹰优化相关参数
        self.use_optimization = getattr(args, 'use_hho_optimization', True)
        self.optimization_epochs = getattr(args, 'optimization_epochs', 5)  # 优化时使用的训练轮数

    def _build_model(self):
        # 如果有优化的参数，应用它们
        if hasattr(self, 'optimized_params') and self.optimized_params is not None:
            self._apply_optimized_params()

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _apply_optimized_params(self):
        """应用优化后的超参数"""
        if self.optimized_params is None:
            return

        params = self.optimized_params

        # 应用优化的超参数
        self.args.d_model = max(64, min(512, int(params[0] * 256 + 128)))  # 64-384
        self.args.learning_rate = max(0.0001, min(0.01, params[1] * 0.009 + 0.001))  # 0.001-0.01
        self.args.dropout = max(0.05, min(0.4, params[2] * 0.3 + 0.1))  # 0.1-0.4
        self.args.moving_avg = max(3, min(25, int(params[3] * 20 + 5)))  # 5-25

        print(f"应用优化参数:")
        print(f"  d_model: {self.args.d_model}")
        print(f"  learning_rate: {self.args.learning_rate:.6f}")
        print(f"  dropout: {self.args.dropout:.3f}")
        print(f"  moving_avg: {self.args.moving_avg}")

        # 新增：双分支相关参数
        if len(params) > 4:
            # 频域注意力头数
            freq_heads = max(1, min(16, int(params[4] * 15 + 1)))  # 1-16
            # 通道注意力reduction比例
            channel_reduction = max(2, min(16, int(params[5] * 14 + 2)))  # 2-16
            # 分支融合权重初始化
            branch_weight_time = max(0.1, min(0.8, params[6] * 0.7 + 0.1))  # 0.1-0.8

            # 将这些参数保存到args中，供模型使用
            self.args.freq_attention_heads = freq_heads
            self.args.channel_attention_reduction = channel_reduction
            self.args.initial_branch_weight_time = branch_weight_time

        print(f"应用优化参数:")
        print(f"  d_model: {self.args.d_model}")
        print(f"  learning_rate: {self.args.learning_rate:.6f}")
        print(f"  dropout: {self.args.dropout:.3f}")
        print(f"  moving_avg: {self.args.moving_avg}")
        if hasattr(self.args, 'freq_attention_heads'):
            print(f"  freq_attention_heads: {self.args.freq_attention_heads}")
            print(f"  channel_attention_reduction: {self.args.channel_attention_reduction}")
            print(f"  initial_branch_weight_time: {self.args.initial_branch_weight_time:.3f}")

    def _objective_function(self, params):
        """哈里斯鹰优化的目标函数"""
        try:
            # 临时保存原始参数
            original_d_model = self.args.d_model
            original_lr = self.args.learning_rate
            original_dropout = self.args.dropout
            original_moving_avg = self.args.moving_avg

            # 应用测试参数
            self.args.d_model = max(64, min(512, int(params[0] * 256 + 128)))
            self.args.learning_rate = max(0.0001, min(0.01, params[1] * 0.009 + 0.001))
            self.args.dropout = max(0.05, min(0.4, params[2] * 0.3 + 0.1))
            self.args.moving_avg = max(3, min(25, int(params[3] * 20 + 5)))

            # 构建临时模型
            temp_model = self.model_dict[self.args.model].Model(self.args).float()
            if self.args.use_multi_gpu and self.args.use_gpu:
                temp_model = nn.DataParallel(temp_model, device_ids=self.args.device_ids)
            temp_model = temp_model.to(self.device)

            # 获取训练和验证数据
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')

            # 简化训练（少量epoch）
            optimizer = optim.Adam(temp_model.parameters(), lr=self.args.learning_rate)
            criterion = nn.MSELoss()

            temp_model.train()
            train_losses = []

            # 只训练指定轮数用于评估
            for epoch in range(self.optimization_epochs):
                epoch_loss = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    if i >= 10:  # 只训练前10个batch
                        break

                    optimizer.zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs = temp_model(batch_x, batch_x_mark, None, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss.append(loss.item())

                train_losses.append(np.mean(epoch_loss))

            # 评估验证集
            temp_model.eval()
            vali_losses = []
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    if i >= 5:  # 只评估前5个batch
                        break

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs = temp_model(batch_x, batch_x_mark, None, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    loss = criterion(outputs, batch_y)
                    vali_losses.append(loss.item())

            final_vali_loss = np.mean(vali_losses)

            # 清理内存
            del temp_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 恢复原始参数
            self.args.d_model = original_d_model
            self.args.learning_rate = original_lr
            self.args.dropout = original_dropout
            self.args.moving_avg = original_moving_avg

            return final_vali_loss

        except Exception as e:
            print(f"优化过程中出现错误: {e}")
            return float('inf')

    def optimize_hyperparameters(self):
        """使用哈里斯鹰算法优化超参数"""
        if not self.use_optimization:
            print("跳过超参数优化")
            return

        print("开始使用哈里斯鹰算法优化超参数...")

        # 定义参数边界（归一化到[0,1]）
        bounds = [
            (0, 1),  # d_model (映射到64-384)
            (0, 1),  # learning_rate (映射到0.001-0.01)
            (0, 1),  # dropout (映射到0.1-0.4)
            (0, 1),  # moving_avg (映射到5-25)
            (0, 1),  # freq_attention_heads (映射到1-16)
            (0, 1),  # channel_attention_reduction (映射到2-16)
            (0, 1),  # branch_weight_time (映射到0.1-0.8)
        ]

        # 创建优化器
        optimizer = HarrisHawkOptimizer(
            dim=len(bounds),
            pop_size=8,  # 较小的种群，加快速度
            max_iter=10,  # 较少的迭代次数
            bounds=bounds
        )

        # 执行优化
        best_params, best_fitness = optimizer.optimize(self._objective_function)

        self.optimized_params = best_params
        print(f"优化完成！最佳验证损失: {best_fitness:.6f}")

        # 保存优化结果
        optimization_results = {
            'timestamp': self.detailed_timestamp,
            'best_params': best_params.tolist(),
            'best_fitness': float(best_fitness),
            'convergence_curve': optimizer.convergence_curve
        }

        optimization_file = f"optimization_results_{self.experiment_timestamp}.json"
        with open(optimization_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        print(f"优化结果已保存至: {optimization_file}")

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

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

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        total_loss = np.average(total_loss)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        _, _, _, _, _ = metric(preds, trues)
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())

        self.model.train()
        return total_loss, r2

    def train(self, setting):
        # 1. 首先执行超参数优化
        if self.use_optimization:
            self.optimize_hyperparameters()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        timestamped_setting = f"{setting}_{self.experiment_timestamp}"
        path = os.path.join(self.args.checkpoints, timestamped_setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        # 增加早停patience，最大epoch保持20
        early_stopping = EarlyStopping(patience=5, verbose=True)  # 增加patience
        max_epochs = 25  # 稍微增加最大epoch

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=max_epochs,
            max_lr=self.args.learning_rate
        )

        # 记录训练历史
        train_history = {
            'train_loss': [],
            'vali_loss': [],
            'vali_r2': [],
            'test_loss': [],
            'test_r2': []
        }

        epoch_pbar = tqdm(range(max_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}",
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

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                batch_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((max_epochs - epoch) * train_steps - i)
                    tqdm.write(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    tqdm.write(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    scheduler.step()

            batch_pbar.close()

            epoch_cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion)

            # 记录历史
            train_history['train_loss'].append(train_loss)
            train_history['vali_loss'].append(vali_loss)
            train_history['vali_r2'].append(vali_r2)
            train_history['test_loss'].append(test_loss)
            train_history['test_r2'].append(test_r2)

            tqdm.write(f"Epoch: {epoch + 1} cost time: {epoch_cost_time:.2f}s")
            tqdm.write(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            tqdm.write(f"Vali R²: {vali_r2:.4f} Test R²: {test_r2:.4f}")

            epoch_pbar.set_postfix({
                'Train_Loss': f"{train_loss:.6f}",
                'Vali_R2': f"{vali_r2:.4f}",
                'Test_R2': f"{test_r2:.4f}"
            })

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered!")
                tqdm.write(f"Training stopped at epoch {epoch + 1}")
                break

        epoch_pbar.close()

        # 保存训练历史
        history_file = os.path.join(path, f'training_history_{self.experiment_timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump(train_history, f, indent=2)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        tqdm.write("Computing final training metrics...")
        final_train_loss, final_train_r2 = self.train_step_metrics(train_loader, criterion)
        tqdm.write(f"Final Training R²: {final_train_r2:.6f}")

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
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())
        print(f'mse:{mse:.6f}, mae:{mae:.6f}')
        print(f'rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}')
        print(f'R²:{r2:.6f}')

        folder_path = './results/' + timestamped_setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 反归一化处理（保持原有逻辑）
        print("开始反归一化处理...")
        print(f"原始preds形状: {preds.shape}")
        print(f"原始trues形状: {trues.shape}")

        raw_df = test_data.raw_test_df
        print("原始数据列名:", raw_df.columns.tolist())

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

        # 创建结果DataFrame时添加实验信息和优化信息
        results_df = pd.DataFrame({
            'Cycle': cycle_data[:min_length],
            'True_Target': true_targets[:min_length],
            'Predicted_Target': preds_unscaled[:min_length]
        })

        # 添加实验元信息（包括优化信息）
        experiment_info = pd.DataFrame({
            'Experiment_Timestamp': [self.detailed_timestamp] * min_length,
            'Model_Name': [self.args.model] * min_length,
            'Dataset': [self.args.data] * min_length,
            'Data_Path': [self.args.data_path] * min_length,
            'MSE': [mse] * min_length,
            'MAE': [mae] * min_length,
            'RMSE': [rmse] * min_length,
            'R2': [r2] * min_length,
            'Used_HHO_Optimization': [self.use_optimization] * min_length,
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)

        # 确定x轴数据
        x_data = cycle_data[:min_length]
        x_label = 'Cycle'

        print(f"使用x轴数据: {x_label}")
        print(f"x轴数据范围: {x_data.min()} 到 {x_data.max()}")

        # 第一个子图：预测结果对比
        ax1.plot(x_data, true_targets[:min_length],
                 color='#2E86AB', linewidth=2.5, alpha=0.8,
                 label='True SoH', marker='o', markersize=3, markevery=max(1, min_length // 50))

        ax1.plot(x_data, preds_unscaled[:min_length],
                 color='#F24236', linewidth=2.5, alpha=0.8,
                 label='Predicted SoH', marker='s', markersize=3, markevery=max(1, min_length // 50))

        # 添加阴影区域显示预测误差
        error = np.abs(true_targets[:min_length] - preds_unscaled[:min_length])
        ax1.fill_between(x_data,
                         preds_unscaled[:min_length] - error / 2,
                         preds_unscaled[:min_length] + error / 2,
                         alpha=0.2, color='red', label='Prediction Error Band')

        ax1.set_title('Battery SoH Prediction with Harris Hawks Optimization', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('State of Health (SoH)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)

        # 添加图例
        legend1 = ax1.legend(loc='upper right', fontsize=10, frameon=True,
                             fancybox=True, shadow=True, framealpha=0.9)
        legend1.get_frame().set_facecolor('white')

        # 添加性能指标文本框
        textstr = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
        if self.use_optimization:
            textstr += f'\nOptimized with HHO'
        props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8, edgecolor='gray')
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=props, family='monospace')

        # 第二个子图：误差分析
        residuals = true_targets[:min_length] - preds_unscaled[:min_length]
        ax2.scatter(x_data, residuals, alpha=0.6, color='#A23B72', s=20)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8)
        ax2.axhline(y=np.mean(residuals), color='red', linestyle='-', alpha=0.8,
                    label=f'Mean Error: {np.mean(residuals):.4f}')
        ax2.axhline(y=np.mean(residuals) + np.std(residuals), color='orange', linestyle=':', alpha=0.8,
                    label=f'±1σ: {np.std(residuals):.4f}')
        ax2.axhline(y=np.mean(residuals) - np.std(residuals), color='orange', linestyle=':', alpha=0.8)

        ax2.set_title('Prediction Residuals Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (True - Predicted)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.legend(loc='upper right', fontsize=9)

        # 调整子图间距
        plt.tight_layout()

        # 保存高质量图片
        plot_path = os.path.join(folder_path, 'enhanced_battery_soh_prediction.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        pdf_path = os.path.join(folder_path, 'enhanced_battery_soh_prediction.pdf')
        plt.savefig(pdf_path, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        print(f"增强版可视化图表已保存:")
        print(f"PNG格式: {plot_path}")
        print(f"PDF格式: {pdf_path}")

        plt.show()
        plt.close()

        # 如果使用了优化，生成优化过程可视化
        if self.use_optimization and hasattr(self, 'optimized_params'):
            self._plot_optimization_results(folder_path)

        # 保存指标
        np.save(folder_path + f'metrics_{self.experiment_timestamp}.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + f'pred_{self.experiment_timestamp}.npy', preds)
        np.save(folder_path + f'true_{self.experiment_timestamp}.npy', trues)

        # 保存结果到文件
        result_file = f"result_enhanced_forecast_{self.experiment_timestamp}.txt"
        f = open(result_file, 'a', encoding='utf-8')
        f.write(f"Experiment Time: {self.detailed_timestamp}\n")
        f.write(f"Dataset: {self.args.data_path}\n")
        f.write(f"Used HHO Optimization: {self.use_optimization}\n")
        if self.use_optimization:
            f.write(f"Optimized Parameters:\n")
            f.write(f"  d_model: {self.args.d_model}\n")
            f.write(f"  learning_rate: {self.args.learning_rate:.6f}\n")
            f.write(f"  dropout: {self.args.dropout:.3f}\n")
            f.write(f"  moving_avg: {self.args.moving_avg}\n")
        f.write(timestamped_setting + "\n")
        f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        # 同时保存到总的结果文件
        f = open("result_enhanced_forecast.txt", 'a', encoding='utf-8')
        f.write(f"[{self.detailed_timestamp}] " + timestamped_setting + "\n")
        f.write(
            f'HHO: {self.use_optimization} | mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        return

    def _plot_optimization_results(self, folder_path):
        """绘制优化过程可视化"""
        try:
            optimization_file = f"optimization_results_{self.experiment_timestamp}.json"
            if os.path.exists(optimization_file):
                with open(optimization_file, 'r') as f:
                    opt_results = json.load(f)

                convergence_curve = opt_results['convergence_curve']

                plt.figure(figsize=(10, 6), dpi=300)
                plt.plot(range(1, len(convergence_curve) + 1), convergence_curve,
                         'b-', linewidth=2, marker='o', markersize=6)
                plt.title('Harris Hawks Optimization Convergence', fontsize=14, fontweight='bold')
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel('Best Fitness (Validation Loss)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                opt_plot_path = os.path.join(folder_path, 'optimization_convergence.png')
                plt.savefig(opt_plot_path, dpi=300, bbox_inches='tight')
                print(f"优化过程图表已保存至: {opt_plot_path}")
                plt.close()
        except Exception as e:
            print(f"生成优化过程图表时出错: {e}")
