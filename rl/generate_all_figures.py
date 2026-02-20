#!/usr/bin/env python3
"""
自动生成所有论文图表

用法：
    python scripts/generate_all_figures.py handle-press-v3
    
    或者指定完整路径：
    python scripts/generate_all_figures.py logs/parallel_runs_vtrace_20260112_154851/handle-press-v3

功能：
1. 自动查找指定路径下的 tensorboard_all 目录
2. 从 tensorboard_all 中读取所有 run 的数据
3. 生成所有需要的图表：
   - Figure 1: Return Curves (Fresh & Stale)
   - Figure 2: Matched-Stability 三合一图
   - Figure 3: Utilization Bar Charts
   - Figure 4: σ Sensitivity (如果适用)
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard not installed")
    print("Install with: pip install tensorboard")
    exit(1)

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  Warning: scipy not available, using simple moving average for smoothing")


def extract_metrics_from_tb(log_dir, metrics_of_interest):
    """从 TensorBoard 目录提取指标"""
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    
    results = {}
    steps_dict = {}  # 存储每个指标的 step 信息
    
    for metric in metrics_of_interest:
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            values = [e.value for e in events]
            steps = [e.step for e in events]
            if len(values) > 0:
                results[metric] = np.array(values)
                steps_dict[metric] = np.array(steps)
            else:
                results[metric] = None
                steps_dict[metric] = None
        else:
            results[metric] = None
            steps_dict[metric] = None
    
    # 将 steps 信息也添加到结果中（使用特殊键）
    results['_steps'] = steps_dict
    
    return results


def get_stable_mean(metric_data, stable_ratio=0.2):
    """获取稳定后的平均值"""
    if metric_data is None or len(metric_data) == 0:
        return np.nan
    stable_start = int(len(metric_data) * (1 - stable_ratio))
    return np.mean(metric_data[stable_start:])


def smooth_curve(data, window_size=None, method='moving_avg'):
    """
    平滑曲线数据
    
    Args:
        data: 1D numpy array
        window_size: 平滑窗口大小（如果为 None，自动计算）
        method: 'moving_avg' 或 'savgol'
    
    Returns:
        平滑后的数据
    """
    if data is None or len(data) == 0:
        return data
    
    data = np.array(data)
    
    # 如果数据点太少，不进行平滑
    if len(data) < 5:
        return data
    
    # 自动计算窗口大小（约为数据长度的 5%，但至少为 5，最多为 50）
    if window_size is None:
        window_size = max(5, min(50, int(len(data) * 0.05)))
        # 确保窗口大小为奇数（对于 savgol）
        if window_size % 2 == 0:
            window_size += 1
    
    if method == 'savgol' and SCIPY_AVAILABLE:
        # 使用 Savitzky-Golay 滤波器（更平滑，保持峰值）
        try:
            # window_size 必须是奇数且小于数据长度
            window_size = min(window_size, len(data))
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < 3:
                window_size = 3
            return savgol_filter(data, window_size, 3)
        except Exception:
            # 如果失败，回退到移动平均
            method = 'moving_avg'
    
    # 移动平均平滑
    if method == 'moving_avg':
        # 使用卷积进行移动平均
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='same')
        # 边界处理：使用原始值
        half_window = window_size // 2
        smoothed[:half_window] = data[:half_window]
        smoothed[-half_window:] = data[-half_window:]
        return smoothed
    
    return data


def parse_run_name(run_name):
    """解析 run 目录名，提取方法、参数等信息"""
    run_name_lower = run_name.lower()
    
    # 识别方法
    # 首先检查是否是 GIPO（必须是 log_gauss_clip，不能是 soft_clip）
    if 'log_gauss_clip' in run_name_lower and 'soft_clip' not in run_name_lower:
        method = 'GIPO'
        # 解析 sigma: 支持多种格式（按实际格式优先级）
        # 1. sigma0d5 (d 作为小数点，如 run_log_gauss_clip_eps1e-9_sigma0d5_...)
        # 2. sigma1, sigma2 (整数，如 run_log_gauss_clip_eps1e-9_sigma1_...)
        # 3. sigma-0.5, sigma-1, sigma-2 (标准格式，一个连字符)
        # 4. sigma-0-5 (两个连字符，需要转换为 0.5)
        param = None
        param_name = None
        
        # 首先尝试匹配 d 作为小数点的格式：sigma0d5 -> 0.5
        match = re.search(r'sigma(\d+)d(\d+)', run_name_lower)
        if match:
            int_part = match.group(1)
            frac_part = match.group(2)
            try:
                param = float(f"{int_part}.{frac_part}")
            except ValueError:
                param = None
        
        # 如果没匹配到，尝试匹配整数格式：sigma1 -> 1.0, sigma2 -> 2.0
        if param is None:
            match = re.search(r'sigma(\d+)(?![d.])', run_name_lower)  # 确保后面不是 d 或小数点
            if match:
                int_val = match.group(1)
                try:
                    param = float(int_val)
                except ValueError:
                    param = None
        
        # 如果还没匹配到，尝试标准格式：sigma-0.5, sigma-1, sigma-2
        if param is None:
            match = re.search(r'sigma-([\d.]+)', run_name_lower)
            if match:
                param_str = match.group(1)
                try:
                    param = float(param_str)
                except ValueError:
                    param = None
        
        # 如果还没匹配到，尝试两个连字符的格式：sigma-0-5 -> 0.5
        if param is None:
            match = re.search(r'sigma-(\d+)-(\d+)', run_name_lower)
            if match:
                int_part = match.group(1)
                frac_part = match.group(2)
                try:
                    param = float(f"{int_part}.{frac_part}")
                except ValueError:
                    param = None
        
        # 如果都没匹配到，使用默认值 1.0
        if param is None:
            param = 1.0
            param_name = 'σ=1'
        else:
            # 明确排除 sigma=0（如果解析出 0，使用默认值 1.0）
            if abs(param) < 0.01:
                param = 1.0
                param_name = 'σ=1'
            # 标准化显示：0.5 -> σ=0.5, 1.0 -> σ=1, 2.0 -> σ=2
            elif abs(param - 0.5) < 0.01:
                param = 0.5
                param_name = 'σ=0.5'
            elif abs(param - 1.0) < 0.01:
                param = 1.0
                param_name = 'σ=1'
            elif abs(param - 2.0) < 0.01:
                param = 2.0
                param_name = 'σ=2'
            else:
                param_name = f'σ={param}'
        
    elif 'sapo' in run_name_lower or 'sapo_soft_clip' in run_name_lower:
        method = 'SAPO'
        # 解析 tau_pos: taup1 -> 1.0 或 tau-pos-1
        match = re.search(r'taup-?([\d.]+)', run_name_lower)
        if match:
            param_str = match.group(1)
            if 'd' in param_str:
                param_str = param_str.replace('d', '.')
            param = float(param_str)
        else:
            param = 1.0
        param_name = f'τ={param}'
        
    elif 'clip' in run_name_lower and 'log_gauss' not in run_name_lower and 'soft_clip' not in run_name_lower:
        method = 'PPO-Clip'
        # 解析 eps（如果有）
        match = re.search(r'eps-?([\d.e-]+)', run_name_lower)
        if match:
            param = float(match.group(1))
        else:
            param = 0.2
        param_name = f'ε={param}'
        
    elif 'soft_clip' in run_name_lower and 'sapo' not in run_name_lower:
        # 检查是否是 soft_clip_alpha-0（no clip）
        match_alpha = re.search(r'alpha-?([\d.]+)', run_name_lower)
        if match_alpha:
            alpha_str = match_alpha.group(1)
            if 'd' in alpha_str:
                alpha_str = alpha_str.replace('d', '.')
            alpha_val = float(alpha_str)
            # 如果 alpha=0，这是 "no clip"
            if abs(alpha_val) < 0.01:
                method = 'No-Clip'
                param = 0.0
                param_name = 'α=0'
            else:
                method = 'Soft-Clip'
                param = alpha_val
                param_name = f'α={param}'
        else:
            method = 'Soft-Clip'
            param = 1.0
            param_name = f'α={param}'
    else:
        return None
    
    # 解析 seed
    match = re.search(r'seed(\d+)', run_name_lower)
    seed = int(match.group(1)) if match else 0
    
    return {
        'method': method,
        'param': param,
        'param_name': param_name,
        'seed': seed
    }


def find_task_logs(input_path):
    """查找指定路径下的 tensorboard_all 目录"""
    input_path = Path(input_path)
    task_logs = []
    
    # 如果输入路径不存在，尝试在 logs 目录下搜索
    if not input_path.exists():
        # 假设输入是任务名，在 logs 目录下搜索
        log_base = Path('logs')
        if log_base.exists():
            for log_dir in log_base.glob('**/{}'.format(input_path.name)):
                if log_dir.is_dir():
                    tb_all_dir = log_dir / 'tensorboard_all'
                    if tb_all_dir.exists():
                        task_logs.append(tb_all_dir)
    else:
        # 输入路径存在，直接使用
        if input_path.is_dir():
            # 如果直接是 tensorboard_all 目录
            if input_path.name == 'tensorboard_all':
                task_logs.append(input_path)
            # 如果是任务目录，查找 tensorboard_all
            elif (input_path / 'tensorboard_all').exists():
                task_logs.append(input_path / 'tensorboard_all')
            # 如果是日志根目录，搜索所有 tensorboard_all
            else:
                for tb_all_dir in input_path.rglob('tensorboard_all'):
                    if tb_all_dir.is_dir():
                        task_logs.append(tb_all_dir)
    
    return task_logs


def collect_task_data(task_logs, task_name=None):
    """收集任务的所有运行数据"""
    metrics_of_interest = [
        'Metrics/KL_Divergence',           # 稳定性指标（KL）
        'Ratio/AbsLogRho_P95',             # 稳定性指标（D0.95）
        'ESS/ESS_Eff_Norm',                # 利用率指标（ESS_eff）
        'ESS/ESS_Norm',                    # 利用率指标（ESS，备用）
        'ESS/ESS_Eff_Norm_Old',            # 利用率指标（旧数据 ESS_eff，关键）
        'ESS/ESS_Eff_Norm_Old_Abs',        # 利用率指标（旧数据 ESS_eff，绝对阈值版本）
        'Contribution/OldUShare_Abs',      # 利用率指标（旧数据贡献占比）
        'Contribution/OldUShare_AbsGradProxy',  # 利用率指标（旧数据梯度贡献占比）
        'Contribution/NearZero_U_Frac',    # 利用率指标（低贡献样本占比）
        'Contribution/NearZero_U_Frac_Old', # 利用率指标（旧数据低贡献样本占比）
        'Staleness/OldFrac_Abs',           # 调试：旧数据占比
        'Staleness/Version_Mean',           # 调试：平均版本差
        'Metrics/Grad_Norm',                # 稳定性验证：梯度范数
        'Metrics/ExplainedVariance',        # Critic 验证：解释方差
        'Eval/Average_Return',             # 性能指标（Eval return）
        'Rollout/Average_Return',          # 性能指标（Training return，备用）
        'Eval/Average_Episode_Length',     # 性能指标（Episode length）
        # Soft clip 特定指标
        'Soft/Outside_Clip_Frac_Old',      # Soft clip 旧数据未 clip 比例
        'Suppressed_Frac_Old',              # Soft clip 旧数据被抑制比例（备用名称）
    ]
    
    all_data = []
    
    # 如果没有提供任务名，从路径中提取
    if task_name is None and task_logs:
        # 从第一个 tensorboard_all 的父目录提取任务名
        task_name = task_logs[0].parent.name
    
    for tb_all_dir in task_logs:
        # 遍历所有 run 目录
        for run_dir in tb_all_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # 跳过名称以 "broke" 结尾的文件夹
            if run_dir.name.endswith('_broke') or run_dir.name.endswith('broke'):
                print(f"⏭️  Skipping broken run: {run_dir.name}")
                continue
            
            # 解析 run 名称
            run_info = parse_run_name(run_dir.name)
            if run_info is None:
                print(f"⚠️  Warning: Could not parse run name: {run_dir.name}")
                continue
            
            # 提取指标
            metrics = extract_metrics_from_tb(run_dir, metrics_of_interest)
            
            if metrics:
                data_point = {
                    'task': task_name,
                    'run_dir': str(run_dir),
                    **run_info,
                    **metrics
                }
                all_data.append(data_point)
    
    return all_data


def plot_return_curves(data, output_dir, task_name, regime=None):
    """生成 Return Curves 图
    
    Args:
        data: 数据列表
        output_dir: 输出目录
        task_name: 任务名
        regime: 'fresh' 或 'stale'，如果指定则只绘制该 regime 的子图
    """
    # 尝试多个 return 指标（按优先级）
    return_metrics = ['Eval/Average_Return', 'Rollout/Average_Return']
    
    # 准备方法列表：PPO-Clip, SAPO, GIPO-σ=0.5, GIPO-σ=1, GIPO-σ=2
    methods_to_plot = []
    method_data_map = {}
    
    # PPO-Clip
    ppo_data = [d for d in data if d['method'] == 'PPO-Clip']
    if ppo_data:
        methods_to_plot.append('PPO-Clip')
        method_data_map['PPO-Clip'] = ppo_data
    
    # SAPO
    sapo_data = [d for d in data if d['method'] == 'SAPO']
    if sapo_data:
        methods_to_plot.append('SAPO')
        method_data_map['SAPO'] = sapo_data
    
    # GIPO 按 sigma 分组
    gipo_data = [d for d in data if d['method'] == 'GIPO']
    for d in gipo_data:
        sigma = d.get('param', 1.0)
        # 只保留 sigma=0.5, 1, 2
        if abs(sigma - 0.5) < 0.01:
            if 'GIPO-σ=0.5' not in methods_to_plot:
                methods_to_plot.append('GIPO-σ=0.5')
            if 'GIPO-σ=0.5' not in method_data_map:
                method_data_map['GIPO-σ=0.5'] = []
            method_data_map['GIPO-σ=0.5'].append(d)
        elif abs(sigma - 1.0) < 0.01:
            if 'GIPO-σ=1' not in methods_to_plot:
                methods_to_plot.append('GIPO-σ=1')
            if 'GIPO-σ=1' not in method_data_map:
                method_data_map['GIPO-σ=1'] = []
            method_data_map['GIPO-σ=1'].append(d)
        elif abs(sigma - 2.0) < 0.01:
            if 'GIPO-σ=2' not in methods_to_plot:
                methods_to_plot.append('GIPO-σ=2')
            if 'GIPO-σ=2' not in method_data_map:
                method_data_map['GIPO-σ=2'] = []
            method_data_map['GIPO-σ=2'].append(d)
    
    if len(methods_to_plot) == 0:
        print("⚠️  Warning: No data found for return curves!")
        return
    
    print(f"📊 Methods for return curves: {methods_to_plot}")
    
    # 根据 regime 参数决定绘制几个子图
    if regime:
        # 只绘制指定 regime 的子图
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
        regimes_to_plot = [regime.capitalize()]
    else:
        # 绘制两个子图（Fresh 和 Stale）
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        regimes_to_plot = ['Fresh', 'Stale']
    
    for regime_idx, regime_name in enumerate(regimes_to_plot):
        ax = axes[regime_idx]
        has_data = False
        
        for method in methods_to_plot:
            method_data = method_data_map.get(method, [])
            if not method_data:
                continue
            
            # 收集所有 runs 的 return 曲线和对应的 steps
            all_curves = []
            all_steps = []
            used_metric = None
            
            for d in method_data:
                # 尝试找到可用的 return 指标
                return_data = None
                return_steps = None
                
                for metric in return_metrics:
                    if d.get(metric) is not None and len(d[metric]) > 0:
                        return_data = d[metric]
                        # 尝试获取对应的 steps
                        steps_dict = d.get('_steps', {})
                        if steps_dict and metric in steps_dict and steps_dict[metric] is not None:
                            return_steps = steps_dict[metric]
                        else:
                            # 如果没有 step 信息，使用索引
                            return_steps = np.arange(len(return_data))
                        used_metric = metric
                        break
                
                if return_data is not None:
                    all_curves.append(return_data)
                    all_steps.append(return_steps)
            
            if not all_curves:
                print(f"⚠️  Warning: No return data found for {method}")
                continue
            
            # 计算平均值和标准差
            min_len = min(len(c) for c in all_curves)
            if min_len == 0:
                print(f"⚠️  Warning: Empty curves for {method}")
                continue
            
            # 对齐所有曲线到相同长度
            aligned_curves = [c[:min_len] for c in all_curves]
            aligned_steps = [s[:min_len] for s in all_steps]
            
            # 使用第一个 run 的 steps（假设所有 runs 的 step 序列相同）
            if len(aligned_steps) > 0:
                steps = aligned_steps[0]
            else:
                steps = np.arange(min_len)
            
            mean_curve = np.mean(aligned_curves, axis=0)
            std_curve = np.std(aligned_curves, axis=0)
            
            # 如果数据点太多，进行降采样以提高绘图性能和平滑度
            max_points = 2000  # 最多保留 2000 个数据点
            if len(mean_curve) > max_points:
                step_indices = np.linspace(0, len(mean_curve) - 1, max_points, dtype=int)
                steps = steps[step_indices]
                mean_curve = mean_curve[step_indices]
                std_curve = std_curve[step_indices]
            
            # 平滑处理：对平均值和标准差都进行平滑
            # 使用较大的窗口以获得更平滑的效果
            smooth_window = max(10, int(len(mean_curve) * 0.05))  # 约 5% 的数据点
            if smooth_window % 2 == 0:
                smooth_window += 1
            
            mean_curve_smooth = smooth_curve(mean_curve, window_size=smooth_window, method='savgol' if SCIPY_AVAILABLE else 'moving_avg')
            std_curve_smooth = smooth_curve(std_curve, window_size=smooth_window, method='moving_avg')
            
            # 绘制
            color_map = {
                'PPO-Clip': '#1f77b4',
                'SAPO': '#ff7f0e',
                'GIPO-σ=0.5': '#90EE90',  # 浅绿
                'GIPO-σ=1': '#2ca02c',     # 中绿
                'GIPO-σ=2': '#006400'     # 深绿
            }
            linestyle_map = {
                'PPO-Clip': '-',
                'SAPO': '--',
                'GIPO-σ=0.5': '-',
                'GIPO-σ=1': '-',
                'GIPO-σ=2': '-'
            }
            
            # 简化标签显示（去掉 GIPO- 前缀，只显示 σ 值）
            display_label = method
            if method.startswith('GIPO-'):
                display_label = method.replace('GIPO-', 'GIPO ')
            
            # 绘制平滑后的曲线（稍微加粗以提高可见性）
            ax.plot(steps, mean_curve_smooth, 
                   label=display_label, 
                   color=color_map.get(method, 'black'),
                   linestyle=linestyle_map.get(method, '-'),
                   linewidth=2.5,
                   alpha=0.9)
            
            # 绘制误差带（使用平滑后的标准差，但透明度降低）
            if len(all_curves) > 1:
                ax.fill_between(steps, 
                               mean_curve_smooth - std_curve_smooth,
                               mean_curve_smooth + std_curve_smooth,
                               alpha=0.15,  # 降低透明度，使图更清晰
                               color=color_map.get(method, 'black'),
                               linewidth=0)
            has_data = True
        
        if not has_data:
            # 显示警告
            ax.text(0.5, 0.5, f'No return data found\nfor {regime_name} regime', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
            print(f"⚠️  Warning: No return data found for {regime_name} regime")
            print(f"   Tried metrics: {return_metrics}")
        
        ax.set_xlabel('Environment Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel('Episodic Return', fontsize=12, fontweight='bold')
        ax.set_title(f'{regime_name} Regime', fontsize=14, fontweight='bold')
        if has_data:
            # 优化图例：放在右上角，使用较小的字体，避免遮挡
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=1)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    filename = f'{task_name}_return_curves'
    if regime:
        filename += f'_{regime}'
    filename += '.pdf'
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_matched_stability(data, output_dir, task_name, regime=None):
    """生成 Matched-Stability 三合一图（生成三种版本：ESS_Eff_Norm, OldUShare, ESS_Eff_Norm_Old）
    
    Args:
        data: 数据列表
        output_dir: 输出目录
        task_name: 任务名
        regime: 'fresh' 或 'stale'，用于在标题中显示
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # 分离不同方法的数据
    ppo_data = [d for d in data if d['method'] == 'PPO-Clip']
    
    # 先获取所有 GIPO 数据，用于调试（确保只包含 log_gauss_clip）
    all_gipo_data = [d for d in data if d['method'] == 'GIPO']
    if len(all_gipo_data) > 0:
        all_gipo_params = sorted(set(d['param'] for d in all_gipo_data))
        print(f"📊 All GIPO params found: {all_gipo_params}")
    
    # 过滤 GIPO 数据：只保留 sigma=0.5, 1, 2 的数据，排除 sigma=0
    gipo_data = [d for d in all_gipo_data 
                  if abs(d['param']) > 0.01  # 排除 sigma=0
                  and (abs(d['param'] - 0.5) < 0.01 or 
                       abs(d['param'] - 1.0) < 0.01 or 
                       abs(d['param'] - 2.0) < 0.01)]  # 只保留 0.5, 1, 2
    
    if len(gipo_data) == 0:
        print("⚠️  Warning: No GIPO data found with σ in {0.5, 1, 2}")
        if len(all_gipo_data) > 0:
            print(f"   All GIPO params found: {all_gipo_params}")
            print(f"   Filtered out params: {[p for p in all_gipo_params if not (abs(p) > 0.01 and (abs(p - 0.5) < 0.01 or abs(p - 1.0) < 0.01 or abs(p - 2.0) < 0.01))]}")
    else:
        filtered_params = sorted(set(d['param'] for d in gipo_data))
        print(f"✅ Filtered GIPO params: {filtered_params}")
    
    sapo_data = [d for d in data if d['method'] == 'SAPO']
    
    # 包含 No-Clip 数据
    no_clip_data = [d for d in data if d['method'] == 'No-Clip']
    if len(no_clip_data) > 0:
        print(f"✅ Found No-Clip data: {len(no_clip_data)} runs")
    
    # 稳定性指标：优先使用 KL，如果不存在则使用 D0.95
    x_metric_primary = 'Metrics/KL_Divergence'
    x_metric_fallback = 'Ratio/AbsLogRho_P95'
    
    # 性能指标：优先使用 Eval，如果不存在则使用 Training
    color_metric_primary = 'Eval/Average_Return'
    color_metric_fallback = 'Rollout/Average_Return'
    
    # 定义三种利用率指标配置
    y_metric_configs = [
        {
            'name': 'ESS_Eff_Norm',
            'primary': 'ESS/ESS_Eff_Norm',
            'fallback': 'ESS/ESS_Norm',
            'ylabel': 'ESS_eff (Normalized)',
            'filename_suffix': 'ess_eff_norm'
        },
        {
            'name': 'OldUShare',
            'primary': 'Contribution/OldUShare_AbsGradProxy',
            'fallback': 'Contribution/OldUShare_Abs',
            'ylabel': 'Old Data Gradient Share',
            'filename_suffix': 'old_ushare'
        },
        {
            'name': 'ESS_Eff_Norm_Old',
            'primary': 'ESS/ESS_Eff_Norm_Old',
            'fallback': 'ESS/ESS_Eff_Norm_Old_Abs',  # 如果 Old 不存在，尝试 Old_Abs
            'ylabel': 'ESS_eff_Old (Normalized)',
            'filename_suffix': 'ess_eff_norm_old'
        }
    ]
    
    def get_coords(data_list, y_metric_primary, y_metric_fallback):
        """获取坐标数据"""
        x = []
        y = []
        c = []
        labels = []
        for d in data_list:
            # 尝试主指标，如果不存在则使用备选
            kl_data = d.get(x_metric_primary)
            if kl_data is None or len(kl_data) == 0:
                kl_data = d.get(x_metric_fallback)
            
            # 尝试 y 指标的主指标和备选
            y_data = d.get(y_metric_primary)
            if y_data is None or len(y_data) == 0:
                y_data = d.get(y_metric_fallback)
            
            return_data = d.get(color_metric_primary)
            if return_data is None or len(return_data) == 0:
                return_data = d.get(color_metric_fallback)
            
            # 检查数据是否存在且非空
            if (kl_data is not None and len(kl_data) > 0 and
                y_data is not None and len(y_data) > 0 and
                return_data is not None and len(return_data) > 0):
                
                x_val = get_stable_mean(kl_data)
                y_val = get_stable_mean(y_data)
                c_val = get_stable_mean(return_data)
                
                # 只添加非 NaN 的值
                if not (np.isnan(x_val) or np.isnan(y_val) or np.isnan(c_val)):
                    x.append(x_val)
                    y.append(y_val)
                    c.append(c_val)
                    labels.append(d['param_name'])
        
        return np.array(x), np.array(y), np.array(c), labels
    
    # 为每种 y 指标生成一个图
    for y_config in y_metric_configs:
        y_metric = y_config['primary']
        y_metric_fallback = y_config['fallback']
        ylabel = y_config['ylabel']
        filename_suffix = y_config['filename_suffix']
        
        print(f"\n📊 Generating matched-stability plot with Y-axis: {y_config['name']}")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 获取坐标数据
        ppo_x, ppo_y, ppo_c, ppo_labels = get_coords(ppo_data, y_metric, y_metric_fallback)
        gipo_x, gipo_y, gipo_c, gipo_labels = get_coords(gipo_data, y_metric, y_metric_fallback)
        sapo_x, sapo_y, sapo_c, sapo_labels = get_coords(sapo_data, y_metric, y_metric_fallback)
        no_clip_x, no_clip_y, no_clip_c, no_clip_labels = get_coords(no_clip_data, y_metric, y_metric_fallback)
        
        # 检查是否有有效数据
        total_points = len(ppo_x) + len(gipo_x) + len(sapo_x) + len(no_clip_x)
        if total_points == 0:
            print(f"⚠️  Warning: No valid data points found for {y_config['name']}!")
            print(f"   Check if the following metrics exist in TensorBoard:")
            print(f"   - {x_metric_primary} or {x_metric_fallback}")
            print(f"   - {y_metric} or {y_metric_fallback}")
            print(f"   - {color_metric_primary} or {color_metric_fallback}")
            # 仍然保存空图，但添加警告文本
            ax.text(0.5, 0.5, f'No data available\nfor {y_config["name"]}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, color='red')
            plt.tight_layout()
            filename = f'{task_name}_matched_stability_{filename_suffix}'
            if regime:
                filename += f'_{regime}'
            filename += '.pdf'
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved (empty): {output_path}")
            plt.close()
            continue
        
        # 颜色映射
        all_colors = np.concatenate([ppo_c, gipo_c, sapo_c, no_clip_c])
        all_colors = all_colors[~np.isnan(all_colors)]
        if len(all_colors) > 0:
            vmin, vmax = all_colors.min(), all_colors.max()
        else:
            vmin, vmax = 0, 1
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('RdYlGn')
        
        # 绘制
        if len(ppo_x) > 0:
            ax.scatter(ppo_x, ppo_y, c=ppo_c,
                      s=200, marker='o',
                      cmap=cmap, norm=norm,
                      edgecolors='black', linewidths=1.5,
                      alpha=0.8, label='PPO-Clip')
            for i, label in enumerate(ppo_labels):
                if not np.isnan(ppo_x[i]) and not np.isnan(ppo_y[i]):
                    ax.annotate(label, (ppo_x[i], ppo_y[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.7)
        
        if len(gipo_x) > 0:
            ax.scatter(gipo_x, gipo_y, c=gipo_c,
                      s=200, marker='^',
                      cmap=cmap, norm=norm,
                      edgecolors='black', linewidths=1.5,
                      alpha=0.8, label='GIPO')
            for i, label in enumerate(gipo_labels):
                if not np.isnan(gipo_x[i]) and not np.isnan(gipo_y[i]):
                    ax.annotate(label, (gipo_x[i], gipo_y[i]),
                               xytext=(5, -15), textcoords='offset points',
                               fontsize=9, alpha=0.7)
        
        if len(sapo_x) > 0:
            ax.scatter(sapo_x, sapo_y, c=sapo_c,
                      s=200, marker='s',
                      cmap=cmap, norm=norm,
                      edgecolors='black', linewidths=1.5,
                      alpha=0.8, label='SAPO')
            for i, label in enumerate(sapo_labels):
                if not np.isnan(sapo_x[i]) and not np.isnan(sapo_y[i]):
                    ax.annotate(label, (sapo_x[i], sapo_y[i]),
                               xytext=(-15, 5), textcoords='offset points',
                               fontsize=9, alpha=0.7)
        
        if len(no_clip_x) > 0:
            ax.scatter(no_clip_x, no_clip_y, c=no_clip_c,
                      s=200, marker='D',  # 菱形标记
                      cmap=cmap, norm=norm,
                      edgecolors='black', linewidths=1.5,
                      alpha=0.8, label='No-Clip')
            for i, label in enumerate(no_clip_labels):
                if not np.isnan(no_clip_x[i]) and not np.isnan(no_clip_y[i]):
                    ax.annotate(label, (no_clip_x[i], no_clip_y[i]),
                               xytext=(5, 15), textcoords='offset points',
                               fontsize=9, alpha=0.7)
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        color_label_map = {
            'Eval/Average_Return': 'Eval Average Return',
            'Rollout/Average_Return': 'Training Average Return'
        }
        cbar.set_label(color_label_map.get(color_metric_primary, 'Return'), fontsize=12)
        
        # 动态设置标签
        xlabel_map = {
            'Metrics/KL_Divergence': 'KL Divergence (Stability)',
            'Ratio/AbsLogRho_P95': 'D₀.₉₅ (Policy Drift)'
        }
        
        ax.set_xlabel(xlabel_map.get(x_metric_primary, x_metric_primary), fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        title = 'Matched-Stability Analysis'
        if regime:
            title += f' ({regime.capitalize()} Regime)'
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'{task_name}_matched_stability_{filename_suffix}'
        if regime:
            filename += f'_{regime}'
        filename += '.pdf'
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()


def plot_utilization_bars(data, output_dir, task_name, regime=None):
    """生成 Utilization Bar Charts
    
    Args:
        data: 数据列表
        output_dir: 输出目录
        task_name: 任务名
        regime: 'fresh' 或 'stale'，用于在标题中显示
    """
    # 使用备选指标，如果主指标不存在
    metrics = {
        'Share_old': {
            'primary': 'Contribution/OldUShare_AbsGradProxy',
            'fallback': 'Contribution/OldUShare_Abs',  # 如果 AbsGradProxy 不存在
            'fallback2': 'Contribution/OldUShare'  # 最后的备选
        },
        'ESS_eff': {
            'primary': 'ESS/ESS_Eff_Norm',
            'fallback': 'ESS/ESS_Norm'  # 如果 ESS_Eff_Norm 不存在
        },
        'NearZeroFrac': {
            'primary': 'Contribution/NearZero_U_Frac',
            'fallback': None  # 如果不存在，显示为 0
        },
        'D0.95': {
            'primary': 'Ratio/AbsLogRho_P95',
            'fallback': None
        }
    }
    
    # 准备方法列表：PPO-Clip, SAPO, GIPO-σ=0.5, GIPO-σ=1, GIPO-σ=2
    methods_to_plot = []
    method_data_map = {}
    
    # PPO-Clip
    ppo_data = [d for d in data if d['method'] == 'PPO-Clip']
    if ppo_data:
        methods_to_plot.append('PPO-Clip')
        method_data_map['PPO-Clip'] = ppo_data
    
    # SAPO
    sapo_data = [d for d in data if d['method'] == 'SAPO']
    if sapo_data:
        methods_to_plot.append('SAPO')
        method_data_map['SAPO'] = sapo_data
    
    # GIPO 按 sigma 分组
    gipo_data = [d for d in data if d['method'] == 'GIPO']
    gipo_by_sigma = {}
    for d in gipo_data:
        sigma = d.get('param', 1.0)
        # 只保留 sigma=0.5, 1, 2
        if abs(sigma - 0.5) < 0.01:
            if 'GIPO-σ=0.5' not in methods_to_plot:
                methods_to_plot.append('GIPO-σ=0.5')
            if 'GIPO-σ=0.5' not in method_data_map:
                method_data_map['GIPO-σ=0.5'] = []
            method_data_map['GIPO-σ=0.5'].append(d)
        elif abs(sigma - 1.0) < 0.01:
            if 'GIPO-σ=1' not in methods_to_plot:
                methods_to_plot.append('GIPO-σ=1')
            if 'GIPO-σ=1' not in method_data_map:
                method_data_map['GIPO-σ=1'] = []
            method_data_map['GIPO-σ=1'].append(d)
        elif abs(sigma - 2.0) < 0.01:
            if 'GIPO-σ=2' not in methods_to_plot:
                methods_to_plot.append('GIPO-σ=2')
            if 'GIPO-σ=2' not in method_data_map:
                method_data_map['GIPO-σ=2'] = []
            method_data_map['GIPO-σ=2'].append(d)
    
    if len(methods_to_plot) == 0:
        print("⚠️  Warning: No data found for utilization bars!")
        return
    
    print(f"📊 Methods for utilization bars: {methods_to_plot}")
    
    fig, axes = plt.subplots(1, 4, figsize=(max(16, len(methods_to_plot) * 1.5), 4))
    
    for idx, (metric_name, metric_config) in enumerate(metrics.items()):
        ax = axes[idx]
        
        values = []
        errors = []
        
        for method in methods_to_plot:
            method_data = method_data_map.get(method, [])
            all_values = []
            
            for d in method_data:
                # 尝试主指标，如果不存在则尝试备选
                metric_data = None
                tb_metric = metric_config['primary']
                
                # 检查主指标
                metric_data = d.get(tb_metric)
                if metric_data is not None and len(metric_data) > 0:
                    pass  # 使用主指标
                elif metric_config.get('fallback'):
                    # 尝试备选1
                    tb_metric = metric_config['fallback']
                    metric_data = d.get(tb_metric)
                    if metric_data is not None and len(metric_data) > 0:
                        pass  # 使用备选1
                    elif metric_config.get('fallback2'):
                        # 尝试备选2
                        tb_metric = metric_config['fallback2']
                        metric_data = d.get(tb_metric)
                
                if metric_data is not None and len(metric_data) > 0:
                    stable_mean = get_stable_mean(metric_data)
                    # 检查值是否合理
                    if not np.isnan(stable_mean):
                        # ESS_eff 应该在 [0, 1] 范围内，但如果接近 1 可能是异常
                        if metric_name == 'ESS_eff':
                            if stable_mean < 0 or stable_mean > 1.1:
                                print(f"⚠️  Warning: {method} {tb_metric}={stable_mean:.4f} out of range [0, 1]")
                            elif stable_mean > 0.99:
                                # 检查数据分布，看看是否所有值都相同
                                data_sample = metric_data[-100:] if len(metric_data) > 100 else metric_data
                                if len(np.unique(data_sample)) < 3:
                                    print(f"⚠️  Warning: {method} ESS_eff={stable_mean:.4f} seems suspicious (low variance)")
                                else:
                                    all_values.append(stable_mean)
                            else:
                                all_values.append(stable_mean)
                        # Share_old 应该在 [0, 1] 范围内
                        elif metric_name == 'Share_old':
                            if stable_mean < 0 or stable_mean > 1.1:
                                print(f"⚠️  Warning: {method} {tb_metric}={stable_mean:.4f} out of range [0, 1]")
                            elif stable_mean == 0.0:
                                # 检查是否真的没有旧数据
                                print(f"⚠️  Warning: {method} Share_old=0, checking if old data exists...")
                                # 尝试检查 Staleness/OldFrac_Abs
                                old_frac = d.get('Staleness/OldFrac_Abs')
                                if old_frac is not None and len(old_frac) > 0:
                                    old_frac_mean = get_stable_mean(old_frac)
                                    print(f"      Staleness/OldFrac_Abs={old_frac_mean:.4f} (should be > 0.7 for stale regime)")
                                all_values.append(stable_mean)  # 仍然添加，即使是 0
                            else:
                                all_values.append(stable_mean)
                        else:
                            all_values.append(stable_mean)
            
            if all_values:
                mean_val = np.mean(all_values)
                # 检查 ESS_eff 是否为异常值（接近 1.0 可能是计算错误）
                if metric_name == 'ESS_eff' and mean_val > 0.99:
                    print(f"⚠️  Warning: {method} ESS_eff={mean_val:.4f} seems too high, checking data...")
                values.append(mean_val)
                errors.append(np.std(all_values) if len(all_values) > 1 else 0)
            else:
                values.append(0)
                errors.append(0)
                # 调试信息（只在第一次遇到问题时打印）
                if method_data and idx == 0:  # 只在第一个指标时打印，避免重复
                    available_metrics = [k for k in method_data[0].keys() 
                                        if isinstance(k, str) and 
                                        ('OldUShare' in k or 'ESS' in k or 'NearZero' in k or 'Contribution' in k)]
                    if available_metrics:
                        print(f"\n⚠️  Warning: {method} has no data for {metric_name}")
                        print(f"   Tried: {metric_config['primary']}")
                        if metric_config.get('fallback'):
                            print(f"   Fallback: {metric_config['fallback']}")
                        print(f"   Available metrics: {available_metrics[:10]}")
        
        x_pos = np.arange(len(methods_to_plot))
        # 颜色映射：PPO-Clip=蓝色, SAPO=橙色, GIPO=不同深浅的绿色
        color_map = {
            'PPO-Clip': '#1f77b4',
            'SAPO': '#ff7f0e',
            'GIPO-σ=0.5': '#90EE90',  # 浅绿
            'GIPO-σ=1': '#2ca02c',     # 中绿
            'GIPO-σ=2': '#006400'       # 深绿
        }
        colors = [color_map.get(method, '#808080') for method in methods_to_plot]
        
        bars = ax.bar(x_pos, values, yerr=errors,
                     color=colors, alpha=0.8,
                     edgecolor='black', linewidth=1.5,
                     capsize=5)
        
        ax.set_xticks(x_pos)
        # 缩短标签，避免重叠
        labels = [method.replace('GIPO-', 'GIPO\n') for method in methods_to_plot]
        ax.set_xticklabels(labels, rotation=0, ha='center')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=10)
    
    # 添加总标题，显示 regime
    title = 'Utilization Metrics Comparison'
    if regime:
        title += f' ({regime.capitalize()} Regime)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filename = f'{task_name}_utilization_bars'
    if regime:
        filename += f'_{regime}'
    filename += '.pdf'
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_sigma_sensitivity(data, output_dir, task_name, regime=None):
    """生成 σ Sensitivity 图（仅 GIPO）
    
    Args:
        data: 数据列表
        output_dir: 输出目录
        task_name: 任务名
        regime: 'fresh' 或 'stale'，用于在文件名中显示
    """
    gipo_data = [d for d in data if d['method'] == 'GIPO']
    
    if len(gipo_data) < 2:
        print("⚠️  Not enough GIPO runs for σ sensitivity analysis")
        return
    
    # 先过滤 GIPO 数据：只保留 sigma=0.5, 1, 2 的数据，排除 sigma=0
    filtered_gipo_data = [d for d in gipo_data 
                          if abs(d['param']) > 0.01  # 排除 sigma=0
                          and (abs(d['param'] - 0.5) < 0.01 or 
                               abs(d['param'] - 1.0) < 0.01 or 
                               abs(d['param'] - 2.0) < 0.01)]  # 只保留 0.5, 1, 2
    
    if len(filtered_gipo_data) == 0:
        all_sigmas = sorted(set(d['param'] for d in gipo_data))
        print(f"⚠️  No GIPO data found with σ in {{0.5, 1, 2}}")
        print(f"   All σ values found: {all_sigmas}")
        return
    
    # 按 σ 分组，只处理 0.5, 1, 2 这三个值
    all_sigmas = sorted(set(d['param'] for d in filtered_gipo_data))
    print(f"📊 All σ values in filtered data: {all_sigmas}")
    
    # 过滤：只保留接近 0.5, 1, 2 的值（容差 0.01），明确排除 0
    target_sigmas = [0.5, 1.0, 2.0]
    sigmas = []
    for target in target_sigmas:
        for s in all_sigmas:
            # 明确排除 sigma=0 的情况
            if abs(s) < 0.01:  # sigma 接近 0
                continue
            if abs(s - target) < 0.01:
                sigmas.append(target)  # 使用目标值而不是实际值，确保顺序一致
                break
    
    # 确保 sigmas 列表按顺序排列（0.5, 1, 2）
    sigmas = sorted(sigmas)
    
    if len(sigmas) < 2:
        print(f"⚠️  Not enough GIPO runs with target σ values (0.5, 1, 2)")
        print(f"   Found σ values: {all_sigmas}")
        print(f"   Filtered σ values: {sigmas}")
        return
    
    print(f"✅ Processing σ sensitivity for values: {sigmas}")
    
    # 使用过滤后的数据
    gipo_data = filtered_gipo_data
    
    metrics = {
        'D0.95': 'Ratio/AbsLogRho_P95',
        'ESS_eff': 'ESS/ESS_Eff_Norm',
        'Return': 'Eval/Average_Return'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (metric_name, tb_metric) in enumerate(metrics.items()):
        ax = axes[idx]
        
        means = []
        stds = []
        
        for sigma in sigmas:
            # 明确排除 sigma=0 的情况（虽然已经过滤过了，但再加一层检查）
            if abs(sigma) < 0.01:
                print(f"⚠️  Skipping σ=0 (not in target set)")
                means.append(np.nan)
                stds.append(0)
                continue
                
            # 匹配该 sigma 值的数据（容差 0.01）
            sigma_data = [d for d in gipo_data if abs(d['param'] - sigma) < 0.01]
            all_values = []
            
            for d in sigma_data:
                metric_data = d.get(tb_metric)
                # 检查数据是否存在且非空
                if metric_data is not None and len(metric_data) > 0:
                    stable_mean = get_stable_mean(metric_data)
                    if not np.isnan(stable_mean):
                        all_values.append(stable_mean)
            
            if all_values:
                means.append(np.mean(all_values))
                stds.append(np.std(all_values) if len(all_values) > 1 else 0)
            else:
                means.append(np.nan)
                stds.append(0)
                print(f"⚠️  Warning: No data for σ={sigma}, metric={tb_metric}")
        
        ax.plot(sigmas, means, 'o-', linewidth=2, markersize=8, color='#2ca02c')
        ax.fill_between(sigmas,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, color='#2ca02c')
        
        # 标记最优 σ
        valid_means = [m for m in means if not np.isnan(m)]
        if valid_means:
            if metric_name in ['ESS_eff', 'Return']:
                optimal_idx = np.argmax(means)
            else:
                optimal_idx = np.argmin(means)
            
            if not np.isnan(means[optimal_idx]):
                ax.axvline(sigmas[optimal_idx], color='red', linestyle='--', alpha=0.5)
                ax.text(sigmas[optimal_idx], means[optimal_idx],
                       f'σ={sigmas[optimal_idx]}',
                       ha='center', va='bottom', fontsize=10, color='red')
        
        ax.set_xlabel('σ', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # 只设置实际存在的 sigma 值作为 x 轴刻度
        ax.set_xticks(sigmas)
        # 确保 x 轴标签正确显示（避免显示 0）
        ax.set_xticklabels([f'{s:.1f}' if s != 1.0 else '1' for s in sigmas])
    
    plt.tight_layout()
    filename = f'{task_name}_sigma_sensitivity'
    if regime:
        filename += f'_{regime}'
    filename += '.pdf'
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate all figures for a task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用任务名（自动在 logs 目录下搜索）
  python scripts/generate_all_figures.py handle-press-v3
  
  # 使用完整路径，指定 regime
  python scripts/generate_all_figures.py logs/.../handle-press-v3 --regime stale
  
  # 指定 fresh regime
  python scripts/generate_all_figures.py logs/.../handle-press-v3 --regime fresh
  
  # 直接指定 tensorboard_all 目录
  python scripts/generate_all_figures.py logs/.../handle-press-v3/tensorboard_all --regime stale
  
  # 只生成特定图表
  python scripts/generate_all_figures.py handle-press-v3 --regime stale --figures utilization return
        """
    )
    parser.add_argument(
        'path',
        help='Task name or path to task directory (e.g., handle-press-v3 or logs/.../handle-press-v3)'
    )
    parser.add_argument(
        '--output-dir',
        default='figures',
        help='Output directory (default: figures)'
    )
    parser.add_argument(
        '--figures',
        nargs='+',
        default=['all'],
        choices=['all', 'return', 'matched', 'utilization', 'sigma', 'additional'],
        help='Which figures to generate (default: all). "additional" includes diagnostic charts from clip_metrics.md'
    )
    parser.add_argument(
        '--regime',
        type=str,
        default=None,
        choices=['fresh', 'stale'],
        help='Regime type: fresh or stale (default: auto-detect from path or assume stale)'
    )
    
    args = parser.parse_args()
    
    # 从路径中提取任务名
    input_path = Path(args.path)
    if input_path.name == 'tensorboard_all':
        # 如果输入是 tensorboard_all，向上找任务名
        task_name = input_path.parent.name
    else:
        task_name = input_path.name
    
    print("=" * 80)
    print(f"Generating figures for: {args.path}")
    print(f"Task name: {task_name}")
    print("=" * 80)
    
    # 查找任务日志
    print("\n[1/4] Finding task logs...")
    task_logs = find_task_logs(args.path)
    
    if not task_logs:
        print(f"❌ No tensorboard_all directories found!")
        print(f"   Searched in: {args.path}")
        print(f"   Tip: Make sure the path contains a 'tensorboard_all' directory")
        return
    
    print(f"✅ Found {len(task_logs)} tensorboard_all directories")
    for log_dir in task_logs:
        print(f"   - {log_dir}")
    
    # 收集数据
    print("\n[2/4] Collecting data...")
    data = collect_task_data(task_logs, task_name)
    
    if not data:
        print("❌ No data found!")
        print("   Make sure tensorboard_all contains run directories with event files")
        return
    
    print(f"✅ Collected {len(data)} runs")
    method_counts = defaultdict(int)
    method_params = defaultdict(list)
    for d in data:
        method_counts[d['method']] += 1
        method_params[d['method']].append(d.get('param', 'N/A'))
    for method, count in method_counts.items():
        params = sorted(set(method_params[method]))
        print(f"   - {method}: {count} runs, params: {params}")
    
    # 调试：检查每个 run 的数据
    print("\n[Debug] Checking data availability...")
    key_metrics = [
        'Metrics/KL_Divergence', 
        'ESS/ESS_Eff_Norm', 
        'Eval/Average_Return',
        'Contribution/OldUShare_AbsGradProxy',
        'Contribution/NearZero_U_Frac'
    ]
    for d in data[:5]:  # 显示前5个
        print(f"   Run: {d['method']} {d['param_name']}")
        for metric in key_metrics:
            metric_data = d.get(metric)
            if metric_data is not None and len(metric_data) > 0:
                stable_val = get_stable_mean(metric_data)
                print(f"      {metric}: {len(metric_data)} points, stable_mean={stable_val:.4f}")
            else:
                # 检查是否有类似的指标
                similar_metrics = [k for k in d.keys() if isinstance(k, str) and metric.split('/')[-1] in k]
                if similar_metrics:
                    print(f"      {metric}: Not found, but found similar: {similar_metrics[:3]}")
                else:
                    print(f"      {metric}: Not found")
    
    # 确定 regime（如果未指定，尝试从路径推断）
    regime = args.regime
    if regime is None:
        # 尝试从路径推断：如果路径中包含 num_actors_16 或类似的关键词，可能是 fresh
        # 如果包含 num_actors_2，可能是 stale
        path_str = str(args.path).lower()
        if 'num_actors_16' in path_str or 'fresh' in path_str:
            regime = 'fresh'
            print(f"🔍 Auto-detected regime: {regime} (from path)")
        elif 'num_actors_2' in path_str or 'stale' in path_str:
            regime = 'stale'
            print(f"🔍 Auto-detected regime: {regime} (from path)")
        else:
            # 默认假设是 stale（因为大多数实验都是 stale）
            regime = 'stale'
            print(f"⚠️  Regime not specified, defaulting to: {regime}")
            print(f"   Use --regime fresh or --regime stale to specify explicitly")
    else:
        print(f"📊 Using specified regime: {regime}")
    
    # 生成图表
    print("\n[3/4] Generating figures...")
    
    figures_to_gen = args.figures
    if 'all' in figures_to_gen:
        figures_to_gen = ['return', 'matched', 'utilization', 'sigma', 'additional']
    
    if 'return' in figures_to_gen:
        plot_return_curves(data, args.output_dir, task_name, regime)
    
    if 'matched' in figures_to_gen:
        plot_matched_stability(data, args.output_dir, task_name, regime)
    
    if 'utilization' in figures_to_gen:
        plot_utilization_bars(data, args.output_dir, task_name, regime)
    
    if 'sigma' in figures_to_gen:
        plot_sigma_sensitivity(data, args.output_dir, task_name, regime)
    
    if 'additional' in figures_to_gen:
        plot_additional_diagnostic_charts(data, args.output_dir, task_name, regime)
    
    print("\n[4/4] Done!")
    print(f"✅ All figures saved to: {args.output_dir}/")


def plot_additional_diagnostic_charts(data, output_dir, task_name, regime=None):
    """生成额外的诊断图表（根据 clip_metrics.md 中的图表列表）
    
    Args:
        data: 数据列表
        output_dir: 输出目录
        task_name: 任务名
        regime: 'fresh' 或 'stale'，用于在文件名中显示
    """
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    
    # 分离不同方法的数据
    ppo_data = [d for d in data if d['method'] == 'PPO-Clip']
    all_gipo_data = [d for d in data if d['method'] == 'GIPO']
    gipo_data = [d for d in all_gipo_data 
                  if abs(d['param']) > 0.01 and 
                  (abs(d['param'] - 0.5) < 0.01 or abs(d['param'] - 1.0) < 0.01 or abs(d['param'] - 2.0) < 0.01)]
    sapo_data = [d for d in data if d['method'] == 'SAPO']
    no_clip_data = [d for d in data if d['method'] == 'No-Clip']
    
    # 性能指标：优先使用 Eval，如果不存在则使用 Training
    color_metric_primary = 'Eval/Average_Return'
    color_metric_fallback = 'Rollout/Average_Return'
    
    # 定义所有图表配置（根据 clip_metrics.md）
    chart_configs = [
        {
            'name': '图A_Mechanism',
            'chart_label': '图A',
            'x_metric': 'Soft/Outside_Clip_Frac_Old',
            'x_fallback': 'Suppressed_Frac_Old',
            'y_metric': 'Contribution/NearZero_U_Frac_Old',
            'y_fallback': 'Contribution/NearZero_U_Frac',
            'xlabel': 'Outside Clip Frac (Old)',
            'ylabel': 'NearZero Frac (Old)',
            'title': '图A: 机制一锤定音',
            'filename': 'figA_mechanism_proof',
            'is_scatter': False  # 曲线图（参数化：X(t) vs Y(t)）
        },
        {
            'name': '图B_OldContribution',
            'chart_label': '图B',
            'x_metric': 'Staleness/Version_Mean',
            'y_metric': 'Contribution/OldUShare_AbsGradProxy',
            'y_fallback': 'Contribution/OldUShare_Abs',
            'xlabel': 'Version Gap (Mean)',
            'ylabel': 'Old Data Gradient Share',
            'title': '图B: 旧数据真实贡献',
            'filename': 'figB_old_contribution',
            'is_scatter': False  # 曲线图（参数化：X(t) vs Y(t)）
        },
        {
            'name': '图C_EffectiveUtilization',
            'chart_label': '图C',
            'x_metric': 'ESS/ESS_Eff_Norm_Old',
            'x_fallback': 'ESS/ESS_Eff_Norm_Old_Abs',
            'y_metric': 'Contribution/NearZero_U_Frac_Old',
            'y_fallback': 'Contribution/NearZero_U_Frac',
            'y_transform': lambda y: 1 - y,  # Y = 1 - NearZero_U_Frac_Old
            'xlabel': 'ESS_eff_Old (Normalized)',
            'ylabel': '1 - NearZero Frac (Old)',
            'title': '图C: 有效利用率',
            'filename': 'figC_effective_utilization',
            'is_scatter': False  # 曲线图（参数化：X(t) vs Y(t)）
        },
        {
            'name': '图D_Pareto',
            'chart_label': '图D',
            'x_metric': 'Ratio/AbsLogRho_P95',
            'y_metric': 'ESS/ESS_Eff_Norm_Old',
            'xlabel': 'D₀.₉₅ (Policy Drift)',
            'ylabel': 'ESS_eff_Old (Normalized)',
            'title': '图D: Pareto 前沿（最强证据）',
            'filename': 'figD_pareto_frontier',
            'is_scatter': True,  # 三合一散点图
            'use_colorbar': True  # 使用颜色条表示Eval Average Return
        },
        {
            'name': '图E_StabilityComparison',
            'chart_label': '图E',
            'x_metric': 'Ratio/AbsLogRho_P95',
            'y_metric': 'Contribution/OldUShare_AbsGradProxy',
            'y_fallback': 'Contribution/OldUShare_Abs',
            'xlabel': 'D₀.₉₅ (Policy Drift)',
            'ylabel': 'Old Data Gradient Share',
            'title': '图E: 同等稳定性对比',
            'filename': 'figE_stability_comparison',
            'is_scatter': False  # 曲线图（参数化：X(t) vs Y(t)）
        },
        {
            'name': 'StabilityVerification',
            'chart_label': '稳定性验证',
            'y_metric': 'Metrics/Grad_Norm',
            'xlabel': 'Environment Steps',
            'ylabel': 'Gradient Norm',
            'title': '稳定性验证（排除暴力更新）',
            'filename': 'stability_verification',
            'is_scatter': False  # 时间序列图
        },
        {
            'name': 'CriticVerification',
            'chart_label': 'Critic验证',
            'y_metric': 'Metrics/ExplainedVariance',
            'xlabel': 'Environment Steps',
            'ylabel': 'Explained Variance',
            'title': 'Critic 验证（排除 Critic 崩坏）',
            'filename': 'critic_verification',
            'is_scatter': False  # 时间序列图
        }
    ]
    
    # 为每个图表配置生成图
    for chart_config in chart_configs:
        print(f"\n📊 Generating: {chart_config['name']} - {chart_config['title']}")
        
        if chart_config.get('is_scatter', False):
            # 判断是三合一散点图还是参数化曲线图
            if chart_config.get('use_colorbar', False):
                # 三合一散点图：X-Y散点，颜色表示Eval Average Return
                fig, ax = plt.subplots(figsize=(10, 7))
                
                x_config = {
                    'x_metric': chart_config['x_metric'],
                    'x_fallback': chart_config.get('x_fallback')
                }
                y_config = {
                    'y_metric': chart_config['y_metric'],
                    'y_fallback': chart_config.get('y_fallback'),
                    'y_transform': chart_config.get('y_transform')
                }
                
                # 性能指标：优先使用 Eval，如果不存在则使用 Training
                color_metric_primary = 'Eval/Average_Return'
                color_metric_fallback = 'Rollout/Average_Return'
                
                # 获取每个方法的坐标点（使用稳定后的平均值）
                def get_scatter_coords_with_color(data_list):
                    x_vals = []
                    y_vals = []
                    c_vals = []
                    labels = []
                    
                    for d in data_list:
                        # 获取X数据
                        x_data = d.get(x_config['x_metric'])
                        if (x_data is None or len(x_data) == 0) and x_config.get('x_fallback'):
                            x_data = d.get(x_config['x_fallback'])
                        
                        # 获取Y数据
                        y_data = d.get(y_config['y_metric'])
                        if (y_data is None or len(y_data) == 0) and y_config.get('y_fallback'):
                            y_data = d.get(y_config['y_fallback'])
                        
                        # 获取颜色数据（性能指标）
                        return_data = d.get(color_metric_primary)
                        if return_data is None or len(return_data) == 0:
                            return_data = d.get(color_metric_fallback)
                        
                        if x_data is not None and len(x_data) > 0 and y_data is not None and len(y_data) > 0:
                            x_val = get_stable_mean(x_data)
                            y_val = get_stable_mean(y_data)
                            
                            # 应用Y变换（如果有）
                            if y_config.get('y_transform'):
                                y_val = y_config['y_transform'](y_val)
                            
                            c_val = get_stable_mean(return_data) if return_data is not None and len(return_data) > 0 else np.nan
                            
                            if not (np.isnan(x_val) or np.isnan(y_val)):
                                x_vals.append(x_val)
                                y_vals.append(y_val)
                                c_vals.append(c_val if not np.isnan(c_val) else 0)
                                labels.append(d['param_name'])
                    
                    return np.array(x_vals), np.array(y_vals), np.array(c_vals), labels
                
                ppo_x, ppo_y, ppo_c, ppo_labels = get_scatter_coords_with_color(ppo_data)
                gipo_x, gipo_y, gipo_c, gipo_labels = get_scatter_coords_with_color(gipo_data)
                sapo_x, sapo_y, sapo_c, sapo_labels = get_scatter_coords_with_color(sapo_data)
                no_clip_x, no_clip_y, no_clip_c, no_clip_labels = get_scatter_coords_with_color(no_clip_data)
                
                total_points = len(ppo_x) + len(gipo_x) + len(sapo_x) + len(no_clip_x)
                if total_points == 0:
                    print(f"⚠️  Warning: No valid data points found for {chart_config['name']}!")
                    ax.text(0.5, 0.5, f'No data available\nfor {chart_config["name"]}',
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=14, color='red')
                    plt.tight_layout()
                    filename = f'{task_name}_{chart_config["filename"]}'
                    if regime:
                        filename += f'_{regime}'
                    filename += '.pdf'
                    output_path = Path(output_dir) / filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"✅ Saved (empty): {output_path}")
                    plt.close()
                    continue
                
                # 颜色映射
                all_colors = np.concatenate([ppo_c, gipo_c, sapo_c, no_clip_c])
                all_colors = all_colors[~np.isnan(all_colors)]
                if len(all_colors) > 0:
                    vmin, vmax = all_colors.min(), all_colors.max()
                else:
                    vmin, vmax = 0, 1
                
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = cm.get_cmap('RdYlGn')
                
                # 绘制散点
                if len(ppo_x) > 0:
                    ax.scatter(ppo_x, ppo_y, c=ppo_c, s=200, marker='o', cmap=cmap, norm=norm,
                              edgecolors='black', linewidths=1.5, alpha=0.8, label='PPO-Clip', zorder=3)
                    for i, label in enumerate(ppo_labels):
                        if not np.isnan(ppo_x[i]) and not np.isnan(ppo_y[i]):
                            ax.annotate(label, (ppo_x[i], ppo_y[i]), xytext=(5, 5),
                                       textcoords='offset points', fontsize=9, alpha=0.7)
                
                if len(gipo_x) > 0:
                    # GIPO需要按sigma分组显示
                    gipo_by_sigma = {}
                    for i, label in enumerate(gipo_labels):
                        sigma = None
                        if 'σ=0.5' in label or 'sigma-0.5' in label.lower():
                            sigma = 'GIPO-σ=0.5'
                        elif 'σ=1' in label or 'sigma-1' in label.lower() or 'sigma1' in label.lower():
                            sigma = 'GIPO-σ=1'
                        elif 'σ=2' in label or 'sigma-2' in label.lower() or 'sigma2' in label.lower():
                            sigma = 'GIPO-σ=2'
                        else:
                            sigma = 'GIPO-σ=1'  # 默认
                        
                        if sigma not in gipo_by_sigma:
                            gipo_by_sigma[sigma] = {'x': [], 'y': [], 'c': [], 'labels': []}
                        gipo_by_sigma[sigma]['x'].append(gipo_x[i])
                        gipo_by_sigma[sigma]['y'].append(gipo_y[i])
                        gipo_by_sigma[sigma]['c'].append(gipo_c[i])
                        gipo_by_sigma[sigma]['labels'].append(label)
                    
                    for sigma_label, sigma_data in gipo_by_sigma.items():
                        x_arr = np.array(sigma_data['x'])
                        y_arr = np.array(sigma_data['y'])
                        c_arr = np.array(sigma_data['c'])
                        ax.scatter(x_arr, y_arr, c=c_arr, s=200, marker='^', cmap=cmap, norm=norm,
                                  edgecolors='black', linewidths=1.5, alpha=0.8, 
                                  label=sigma_label.replace('GIPO-', 'GIPO '), zorder=3)
                        for i, label in enumerate(sigma_data['labels']):
                            if not np.isnan(x_arr[i]) and not np.isnan(y_arr[i]):
                                ax.annotate(label, (x_arr[i], y_arr[i]), xytext=(5, -15),
                                           textcoords='offset points', fontsize=9, alpha=0.7)
                
                if len(sapo_x) > 0:
                    ax.scatter(sapo_x, sapo_y, c=sapo_c, s=200, marker='s', cmap=cmap, norm=norm,
                              edgecolors='black', linewidths=1.5, alpha=0.8, label='SAPO', zorder=3)
                    for i, label in enumerate(sapo_labels):
                        if not np.isnan(sapo_x[i]) and not np.isnan(sapo_y[i]):
                            ax.annotate(label, (sapo_x[i], sapo_y[i]), xytext=(-15, 5),
                                       textcoords='offset points', fontsize=9, alpha=0.7)
                
                if len(no_clip_x) > 0:
                    ax.scatter(no_clip_x, no_clip_y, c=no_clip_c, s=200, marker='D', cmap=cmap, norm=norm,
                              edgecolors='black', linewidths=1.5, alpha=0.8, label='No-Clip', zorder=3)
                    for i, label in enumerate(no_clip_labels):
                        if not np.isnan(no_clip_x[i]) and not np.isnan(no_clip_y[i]):
                            ax.annotate(label, (no_clip_x[i], no_clip_y[i]), xytext=(5, 15),
                                       textcoords='offset points', fontsize=9, alpha=0.7)
                
                # Colorbar
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Eval Average Return', fontsize=12)
                
                ax.set_xlabel(chart_config['xlabel'], fontsize=14, fontweight='bold')
                ax.set_ylabel(chart_config['ylabel'], fontsize=14, fontweight='bold')
                title = chart_config['title']
                if regime:
                    title += f' ({regime.capitalize()} Regime)'
                ax.set_title(title, fontsize=16, fontweight='bold')
                ax.legend(loc='best', fontsize=12)
                ax.grid(True, alpha=0.3)
                
            else:
                # 参数化曲线图：X(t) vs Y(t)，其中X和Y都是随时间变化的指标
                fig, ax = plt.subplots(figsize=(10, 7))
                
                x_config = {
                    'x_metric': chart_config['x_metric'],
                    'x_fallback': chart_config.get('x_fallback')
                }
                y_config = {
                    'y_metric': chart_config['y_metric'],
                    'y_fallback': chart_config.get('y_fallback'),
                    'y_transform': chart_config.get('y_transform')
                }
                
                max_points = 2000  # 最多保留 2000 个数据点
            
            color_map = {
                'PPO-Clip': '#1f77b4',
                'SAPO': '#ff7f0e',
                'GIPO-σ=0.5': '#90EE90',
                'GIPO-σ=1': '#2ca02c',
                'GIPO-σ=2': '#006400',
                'No-Clip': '#808080'
            }
            
            # 准备数据组：GIPO 需要按 sigma 分组
            all_data_groups = [
                ('PPO-Clip', ppo_data),
                ('SAPO', sapo_data),
                ('No-Clip', no_clip_data)
            ]
            
            # GIPO 按 sigma 分组
            gipo_by_sigma = {}
            for d in gipo_data:
                sigma = d.get('param', 1.0)
                if abs(sigma - 0.5) < 0.01:
                    if 'GIPO-σ=0.5' not in gipo_by_sigma:
                        gipo_by_sigma['GIPO-σ=0.5'] = []
                    gipo_by_sigma['GIPO-σ=0.5'].append(d)
                elif abs(sigma - 1.0) < 0.01:
                    if 'GIPO-σ=1' not in gipo_by_sigma:
                        gipo_by_sigma['GIPO-σ=1'] = []
                    gipo_by_sigma['GIPO-σ=1'].append(d)
                elif abs(sigma - 2.0) < 0.01:
                    if 'GIPO-σ=2' not in gipo_by_sigma:
                        gipo_by_sigma['GIPO-σ=2'] = []
                    gipo_by_sigma['GIPO-σ=2'].append(d)
            
            # 添加 GIPO 分组
            for sigma_label, sigma_data in gipo_by_sigma.items():
                all_data_groups.append((sigma_label, sigma_data))
            
            has_data = False
            for method_name, method_data in all_data_groups:
                if not method_data:
                    continue
                
                # 收集所有 runs 的 X-Y 曲线
                all_x_curves = []
                all_y_curves = []
                all_steps = []
                
                for d in method_data:
                    # 获取X数据
                    x_data = d.get(x_config['x_metric'])
                    if (x_data is None or len(x_data) == 0) and x_config.get('x_fallback'):
                        x_data = d.get(x_config['x_fallback'])
                    
                    # 获取Y数据
                    y_data = d.get(y_config['y_metric'])
                    if (y_data is None or len(y_data) == 0) and y_config.get('y_fallback'):
                        y_data = d.get(y_config['y_fallback'])
                    
                    if x_data is not None and len(x_data) > 0 and y_data is not None and len(y_data) > 0:
                        # 应用Y变换（如果有）
                        if y_config.get('y_transform'):
                            y_data = np.array(y_data)
                            y_data = y_config['y_transform'](y_data)
                        
                        # 对齐X和Y的长度
                        min_len = min(len(x_data), len(y_data))
                        x_aligned = np.array(x_data[:min_len])
                        y_aligned = np.array(y_data[:min_len])
                        
                        # 获取steps（用于对齐）
                        steps_dict = d.get('_steps', {})
                        steps = None
                        for metric in [x_config['x_metric'], y_config['y_metric'], 'Eval/Average_Return', 'Rollout/Average_Return']:
                            if metric in steps_dict and steps_dict[metric] is not None:
                                steps = steps_dict[metric][:min_len]
                                break
                        
                        if steps is None or len(steps) != min_len:
                            steps = np.arange(min_len)
                        
                        all_x_curves.append(x_aligned)
                        all_y_curves.append(y_aligned)
                        all_steps.append(steps)
                
                if not all_x_curves:
                    continue
                
                # 对齐所有曲线到相同长度
                min_len = min(len(c) for c in all_x_curves)
                aligned_x_curves = [c[:min_len] for c in all_x_curves]
                aligned_y_curves = [c[:min_len] for c in all_y_curves]
                aligned_steps = [s[:min_len] for s in all_steps]
                
                # 计算均值和标准差
                mean_x_curve = np.mean(aligned_x_curves, axis=0)
                mean_y_curve = np.mean(aligned_y_curves, axis=0)
                std_x_curve = np.std(aligned_x_curves, axis=0)
                std_y_curve = np.std(aligned_y_curves, axis=0)
                
                # 平滑处理
                if len(mean_x_curve) > max_points:
                    step_indices = np.linspace(0, len(mean_x_curve) - 1, max_points, dtype=int)
                    mean_x_curve = mean_x_curve[step_indices]
                    mean_y_curve = mean_y_curve[step_indices]
                    std_x_curve = std_x_curve[step_indices]
                    std_y_curve = std_y_curve[step_indices]
                
                smooth_window = max(10, int(len(mean_x_curve) * 0.05))
                if smooth_window % 2 == 0:
                    smooth_window += 1
                
                mean_x_smooth = smooth_curve(mean_x_curve, window_size=smooth_window, method='savgol' if SCIPY_AVAILABLE else 'moving_avg')
                mean_y_smooth = smooth_curve(mean_y_curve, window_size=smooth_window, method='savgol' if SCIPY_AVAILABLE else 'moving_avg')
                std_x_smooth = smooth_curve(std_x_curve, window_size=smooth_window, method='moving_avg')
                std_y_smooth = smooth_curve(std_y_curve, window_size=smooth_window, method='moving_avg')
                
                # 确定颜色和标签
                if method_name.startswith('GIPO-'):
                    color = color_map.get(method_name, '#2ca02c')
                    label = method_name.replace('GIPO-', 'GIPO ')
                else:
                    color = color_map.get(method_name, '#808080')
                    label = method_name
                
                # 绘制参数化曲线 X(t) vs Y(t)
                ax.plot(mean_x_smooth, mean_y_smooth, label=label, color=color, linewidth=2.5, alpha=0.9, zorder=3)
                
                # 绘制误差带（使用简化的方法：在X和Y方向分别显示不确定性）
                if len(all_x_curves) > 1:
                    # 计算误差带的边界
                    x_lower = mean_x_smooth - std_x_smooth
                    x_upper = mean_x_smooth + std_x_smooth
                    y_lower = mean_y_smooth - std_y_smooth
                    y_upper = mean_y_smooth + std_y_smooth
                    
                    # 绘制填充区域（简化：使用矩形近似）
                    # 更精确的方法需要计算每个点的误差椭圆，但这里用简化版本
                    ax.fill_betweenx(mean_y_smooth, x_lower, x_upper, alpha=0.1, color=color, linewidth=0)
                    ax.fill_between(mean_x_smooth, y_lower, y_upper, alpha=0.1, color=color, linewidth=0)
                
                has_data = True
            
            if not has_data:
                ax.text(0.5, 0.5, f'No data available\nfor {chart_config["name"]}',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='red')
            
            ax.set_xlabel(chart_config['xlabel'], fontsize=14, fontweight='bold')
            ax.set_ylabel(chart_config['ylabel'], fontsize=14, fontweight='bold')
            title = chart_config['title']
            if regime:
                title += f' ({regime.capitalize()} Regime)'
            ax.set_title(title, fontsize=16, fontweight='bold')
            if has_data:
                ax.legend(loc='best', fontsize=12)
            ax.grid(True, alpha=0.3)
            
        else:
            # 时间序列图：绘制多条曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_config = {
                'y_metric': chart_config['y_metric'],
                'y_fallback': chart_config.get('y_fallback'),
                'y_transform': chart_config.get('y_transform')
            }
            
            max_points = 2000  # 最多保留 2000 个数据点
            
            color_map = {
                'PPO-Clip': '#1f77b4',
                'SAPO': '#ff7f0e',
                'GIPO-σ=0.5': '#90EE90',
                'GIPO-σ=1': '#2ca02c',
                'GIPO-σ=2': '#006400',
                'No-Clip': '#808080'
            }
            
            # 准备数据组：GIPO 需要按 sigma 分组
            all_data_groups = [
                ('PPO-Clip', ppo_data),
                ('SAPO', sapo_data),
                ('No-Clip', no_clip_data)
            ]
            
            # GIPO 按 sigma 分组
            gipo_by_sigma = {}
            for d in gipo_data:
                sigma = d.get('param', 1.0)
                if abs(sigma - 0.5) < 0.01:
                    if 'GIPO-σ=0.5' not in gipo_by_sigma:
                        gipo_by_sigma['GIPO-σ=0.5'] = []
                    gipo_by_sigma['GIPO-σ=0.5'].append(d)
                elif abs(sigma - 1.0) < 0.01:
                    if 'GIPO-σ=1' not in gipo_by_sigma:
                        gipo_by_sigma['GIPO-σ=1'] = []
                    gipo_by_sigma['GIPO-σ=1'].append(d)
                elif abs(sigma - 2.0) < 0.01:
                    if 'GIPO-σ=2' not in gipo_by_sigma:
                        gipo_by_sigma['GIPO-σ=2'] = []
                    gipo_by_sigma['GIPO-σ=2'].append(d)
            
            # 添加 GIPO 分组
            for sigma_label, sigma_data in gipo_by_sigma.items():
                all_data_groups.append((sigma_label, sigma_data))
            
            has_data = False
            for method_name, method_data in all_data_groups:
                if not method_data:
                    continue
                
                # 收集所有 runs 的曲线
                all_curves = []
                all_steps = []
                
                for d in method_data:
                    y_data = d.get(y_config['y_metric'])
                    if y_data is None or len(y_data) == 0:
                        if y_config.get('y_fallback'):
                            y_data = d.get(y_config['y_fallback'])
                    
                    if y_data is not None and len(y_data) > 0:
                        # 应用变换（如果有）
                        if y_config.get('y_transform'):
                            y_data = np.array(y_data)
                            y_data = y_config['y_transform'](y_data)
                        
                        steps_dict = d.get('_steps', {})
                        # 尝试找到对应的 steps
                        steps = None
                        for metric in ['Eval/Average_Return', 'Rollout/Average_Return', y_config['y_metric']]:
                            if metric in steps_dict and steps_dict[metric] is not None:
                                steps = steps_dict[metric]
                                break
                        
                        if steps is None or len(steps) != len(y_data):
                            steps = np.arange(len(y_data))
                        
                        all_curves.append(y_data)
                        all_steps.append(steps)
                
                if not all_curves:
                    continue
                
                # 对齐所有曲线到相同长度
                min_len = min(len(c) for c in all_curves)
                aligned_curves = [c[:min_len] for c in all_curves]
                aligned_steps = [s[:min_len] for s in all_steps]
                
                # 使用第一个 run 的 steps
                steps = aligned_steps[0] if aligned_steps else np.arange(min_len)
                mean_curve = np.mean(aligned_curves, axis=0)
                std_curve = np.std(aligned_curves, axis=0)
                
                # 平滑处理
                if len(mean_curve) > max_points:
                    step_indices = np.linspace(0, len(mean_curve) - 1, max_points, dtype=int)
                    steps = steps[step_indices]
                    mean_curve = mean_curve[step_indices]
                    std_curve = std_curve[step_indices]
                
                smooth_window = max(10, int(len(mean_curve) * 0.05))
                if smooth_window % 2 == 0:
                    smooth_window += 1
                mean_curve_smooth = smooth_curve(mean_curve, window_size=smooth_window, method='savgol' if SCIPY_AVAILABLE else 'moving_avg')
                std_curve_smooth = smooth_curve(std_curve, window_size=smooth_window, method='moving_avg')
                
                # 确定颜色和标签
                if method_name.startswith('GIPO-'):
                    color = color_map.get(method_name, '#2ca02c')
                    label = method_name.replace('GIPO-', 'GIPO ')
                else:
                    color = color_map.get(method_name, '#808080')
                    label = method_name
                
                ax.plot(steps, mean_curve_smooth, label=label, color=color, linewidth=2.5, alpha=0.9)
                if len(all_curves) > 1:
                    ax.fill_between(steps, mean_curve_smooth - std_curve_smooth,
                                   mean_curve_smooth + std_curve_smooth,
                                   alpha=0.15, color=color, linewidth=0)
                has_data = True
            
            if not has_data:
                ax.text(0.5, 0.5, f'No data available\nfor {chart_config["name"]}',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='red')
            
            ax.set_xlabel(chart_config['xlabel'], fontsize=12, fontweight='bold')
            ax.set_ylabel(chart_config['ylabel'], fontsize=12, fontweight='bold')
            title = chart_config['title']
            if regime:
                title += f' ({regime.capitalize()} Regime)'
            ax.set_title(title, fontsize=14, fontweight='bold')
            if has_data:
                ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        filename = f'{task_name}_{chart_config["filename"]}'
        if regime:
            filename += f'_{regime}'
        filename += '.pdf'
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()


if __name__ == '__main__':
    main()
