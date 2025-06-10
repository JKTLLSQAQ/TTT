import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def visualize_predictions(preds, trues, setting, save_path=None):
    """
    为电池SOH预测创建专业的可视化图表

    参数:
    - preds: 预测值，形状为 [samples, pred_len, features]
    - trues: 真实值，形状为 [samples, pred_len, features]
    - setting: 实验设置名称
    - save_path: 保存路径，默认为 './results/' + setting + '/visualization/'
    """
    if save_path is None:
        save_path = './results/' + setting + '/visualization/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取结果文件夹路径
    results_folder = './results/' + setting + '/'

    # 设置更好看的图表样式
    plt.style.use('seaborn-v0_8-whitegrid')

    # 尝试读取prediction_results.csv文件获取反归一化的值
    try:
        # 读取CSV文件
        results_df = pd.read_csv(results_folder + 'prediction_results.csv')

        # 确保CSV文件包含所需的列
        if 'date' in results_df.columns and 'Prediction' in results_df.columns:
            # 找到目标列（可能是'Target'或类似名称）
            target_col = None
            for col in results_df.columns:
                if col.lower() == 'target':
                    target_col = col
                    break

            if target_col is None:
                # 尝试其他可能的目标列名
                potential_targets = ['target', 'value', 'actual', 'true', 'soh', 'battery_soh']
                for col in results_df.columns:
                    if any(pt in col.lower() for pt in potential_targets) and col != 'Prediction':
                        target_col = col
                        break

            if target_col is None:
                # 如果仍然找不到目标列，选择第一个不是date和不是Prediction的列
                for col in results_df.columns:
                    if col != 'date' and col != 'Prediction' and not col.startswith('Prediction_'):
                        target_col = col
                        break

            if target_col is not None:
                # 创建主预测图
                fig, ax = plt.subplots(figsize=(14, 7))

                # 尝试解析日期
                x_values = None
                try:
                    # 转换日期列为datetime
                    results_df['date'] = pd.to_datetime(results_df['date'])
                    x_values = results_df['date']

                    # 格式化日期轴
                    date_format = DateFormatter('%Y-%m-%d')
                    ax.xaxis.set_major_formatter(date_format)
                    fig.autofmt_xdate()  # 自动旋转日期标签

                    # 设置适当的日期间隔
                    if len(results_df) > 30:
                        ax.xaxis.set_major_locator(mdates.MonthLocator())
                    else:
                        ax.xaxis.set_major_locator(mdates.WeekdayLocator())

                except:
                    # 如果无法解析为日期，使用序号并显示部分日期标签
                    x_values = np.arange(len(results_df))
                    step = max(1, len(results_df) // 10)
                    ax.set_xticks(np.arange(0, len(results_df), step))
                    ax.set_xticklabels(results_df['date'].iloc[::step], rotation=45)

                # 绘制真实值和预测值
                line1, = ax.plot(x_values, results_df[target_col], 'b-', label='True Value', linewidth=2, marker='o',
                                 markersize=3)
                line2, = ax.plot(x_values, results_df['Prediction'], 'r--', label='Predicted Value', linewidth=2,
                                 marker='x', markersize=3)

                # 添加标题和标签
                ax.set_title('Battery capacity Prediction', fontsize=18, fontweight='bold')
                ax.set_xlabel('Time', fontsize=14)
                ax.set_ylabel('RUL Value', fontsize=14)

                # 添加图例到右上角
                ax.legend(handles=[line1, line2], loc='upper right', fontsize=12, frameon=True, framealpha=0.9)

                # 设置y轴适当的范围
                min_y = min(results_df[target_col].min(), results_df['Prediction'].min())
                max_y = max(results_df[target_col].max(), results_df['Prediction'].max())
                padding = (max_y - min_y) * 0.1
                ax.set_ylim([min_y - padding, max_y + padding])

                # 添加网格
                ax.grid(True, linestyle='--', alpha=0.7)

                # 计算并显示性能指标
                mse = mean_squared_error(results_df[target_col], results_df['Prediction'])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(results_df[target_col], results_df['Prediction'])
                r2 = r2_score(results_df[target_col], results_df['Prediction'])

                # 添加文本框显示性能指标
                textstr = f'MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='bottom', bbox=props)

                # 优化布局
                plt.tight_layout()

                # 保存图像
                plt.savefig(f'{save_path}battery_soh_prediction.png', dpi=300, bbox_inches='tight')
                plt.close()

                # 创建误差分析图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # 计算预测误差
                errors = results_df['Prediction'] - results_df[target_col]

                # 绘制误差随时间变化图
                ax1.plot(x_values, errors, 'g-', linewidth=1.5)
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                ax1.fill_between(x_values, errors, 0, where=(errors > 0), color='salmon', alpha=0.5)
                ax1.fill_between(x_values, errors, 0, where=(errors <= 0), color='skyblue', alpha=0.5)

                # 添加标题和标签
                ax1.set_title('Prediction Error Over Time', fontsize=15)
                ax1.set_xlabel('Time', fontsize=12)
                ax1.set_ylabel('Error (Predicted - True)', fontsize=12)
                ax1.grid(True, linestyle='--', alpha=0.7)

                # 如果x_values是日期类型，格式化x轴
                if isinstance(x_values, pd.DatetimeIndex):
                    date_format = DateFormatter('%Y-%m-%d')
                    ax1.xaxis.set_major_formatter(date_format)
                    fig.autofmt_xdate()

                # 绘制散点图
                ax2.scatter(results_df[target_col], results_df['Prediction'], alpha=0.7, c='blue')

                # 添加45度参考线（完美预测）
                min_val = min(results_df[target_col].min(), results_df['Prediction'].min())
                max_val = max(results_df[target_col].max(), results_df['Prediction'].max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

                # 添加标题和标签
                ax2.set_title(f'True vs Predicted Values (R²: {r2:.4f})', fontsize=15)
                ax2.set_xlabel('True SOH', fontsize=12)
                ax2.set_ylabel('Predicted SOH', fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.7)

                # 设置相同的轴范围
                ax2.set_xlim([min_val - padding, max_val + padding])
                ax2.set_ylim([min_val - padding, max_val + padding])
                ax2.set_aspect('equal')

                # 优化布局
                plt.tight_layout()

                # 保存图像
                plt.savefig(f'{save_path}error_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

                # 如果有多个预测步长，创建额外的可视化
                multi_step_cols = [col for col in results_df.columns if col.startswith('Prediction_Step')]
                if multi_step_cols:
                    # 创建多步预测图
                    fig, ax = plt.subplots(figsize=(14, 7))

                    # 绘制真实值
                    ax.plot(x_values, results_df[target_col], 'b-', label='True Value', linewidth=2)

                    # 绘制多步预测值
                    colors = plt.cm.viridis(np.linspace(0, 0.8, len(multi_step_cols) + 1))
                    ax.plot(x_values, results_df['Prediction'], color=colors[0], linestyle='--',
                            label='Step 1 Prediction', linewidth=1.5)

                    for i, col in enumerate(multi_step_cols):
                        ax.plot(x_values, results_df[col], color=colors[i + 1], linestyle='--',
                                label=f'Step {i + 2} Prediction', linewidth=1.5)

                    # 添加标题和标签
                    ax.set_title('Multi-step Battery SOH Prediction', fontsize=18, fontweight='bold')
                    ax.set_xlabel('Time', fontsize=14)
                    ax.set_ylabel('SOH Value', fontsize=14)

                    # 添加图例
                    ax.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)

                    # 添加网格
                    ax.grid(True, linestyle='--', alpha=0.7)

                    # 优化布局
                    plt.tight_layout()

                    # 保存图像
                    plt.savefig(f'{save_path}multi_step_prediction.png', dpi=300, bbox_inches='tight')
                    plt.close()

                print(f"Battery SOH prediction visualizations saved to {save_path}")
                return
    except Exception as e:
        print(f"Could not read or process prediction_results.csv: {e}")

    # 如果无法从CSV读取数据，使用传入的原始预测值和真实值创建可视化
    print("Creating visualization using provided prediction arrays...")

    # 获取样本数量和特征数量
    n_samples, pred_len, n_features = preds.shape
    feature_idx = 0  # 默认使用第一个特征

    # 创建时间索引（由于没有实际日期信息，使用序号）
    x_values = np.arange(n_samples)

    # 创建主预测图
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制真实值和预测值
    line1, = ax.plot(x_values, trues[:, 0, feature_idx], 'b-', label='True Value', linewidth=2, marker='o',
                     markersize=3)
    line2, = ax.plot(x_values, preds[:, 0, feature_idx], 'r--', label='Predicted Value', linewidth=2, marker='x',
                     markersize=3)

    # 添加标题和标签
    ax.set_title('Battery SOH Prediction', fontsize=18, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('SOH Value', fontsize=14)

    # 添加图例到右上角
    ax.legend(handles=[line1, line2], loc='upper right', fontsize=12, frameon=True, framealpha=0.9)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 计算并显示性能指标
    true_vals = trues[:, 0, feature_idx]
    pred_vals = preds[:, 0, feature_idx]
    mse = np.mean((true_vals - pred_vals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_vals - pred_vals))
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    # 添加文本框显示性能指标
    textstr = f'MSE: {mse:.6f}\nRMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)

    # 优化布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'{save_path}battery_soh_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 创建误差分析图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 计算预测误差
    errors = pred_vals - true_vals

    # 绘制误差随时间变化图
    ax1.plot(x_values, errors, 'g-', linewidth=1.5)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax1.fill_between(x_values, errors, 0, where=(errors > 0), color='salmon', alpha=0.5)
    ax1.fill_between(x_values, errors, 0, where=(errors <= 0), color='skyblue', alpha=0.5)

    # 添加标题和标签
    ax1.set_title('Prediction Error Over Time', fontsize=15)
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Error (Predicted - True)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 绘制散点图
    ax2.scatter(true_vals, pred_vals, alpha=0.7, c='blue')

    # 添加45度参考线（完美预测）
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    padding = (max_val - min_val) * 0.1
    ax2.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'r--', alpha=0.7)

    # 添加标题和标签
    ax2.set_title(f'True vs Predicted Values (R²: {r2:.4f})', fontsize=15)
    ax2.set_xlabel('True SOH', fontsize=12)
    ax2.set_ylabel('Predicted SOH', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 设置相同的轴范围
    ax2.set_xlim([min_val - padding, max_val + padding])
    ax2.set_ylim([min_val - padding, max_val + padding])
    ax2.set_aspect('equal')

    # 优化布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'{save_path}error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 如果预测步长大于1，创建多步预测可视化
    if pred_len > 1:
        fig, ax = plt.subplots(figsize=(14, 7))

        # 对于多步预测，我们选择第一个样本进行可视化
        sample_idx = 0

        # 绘制真实值和预测值
        time_steps = np.arange(pred_len)
        ax.plot(time_steps, trues[sample_idx, :, feature_idx], 'b-', label='True Value',
                marker='o', markersize=6, linewidth=2)
        ax.plot(time_steps, preds[sample_idx, :, feature_idx], 'r--', label='Predicted Value',
                marker='x', markersize=6, linewidth=2)

        # 添加标题和标签
        ax.set_title('Multi-step Battery SOH Prediction', fontsize=18, fontweight='bold')
        ax.set_xlabel('Prediction Steps', fontsize=14)
        ax.set_ylabel('SOH Value', fontsize=14)

        # 添加图例
        ax.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 优化布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(f'{save_path}multi_step_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 创建热力图显示所有样本的多步预测误差
        plt.figure(figsize=(12, 8))

        # 计算每个样本每个预测步骤的绝对误差
        multi_step_errors = np.abs(preds[:, :, feature_idx] - trues[:, :, feature_idx])

        # 使用热力图展示
        plt.imshow(multi_step_errors, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Absolute Error')

        # 添加标题和标签
        plt.title('Multi-step Prediction Error Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Prediction Step', fontsize=12)
        plt.ylabel('Sample Index', fontsize=12)

        # 保存图像
        plt.savefig(f'{save_path}error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Battery SOH prediction visualizations saved to {save_path}")

    return