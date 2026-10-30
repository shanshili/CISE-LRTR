"""
核心特点: 全局标准化版本
流程: 全局StandardScaler拟合和应用
功能: 全局特征标准化，保持一致性
改进: 引入build_global_scaler函数，统一特征处理
掩码处理: 保持特征处理的一致性


1. Mask参数初始化
参数形状: [100, 100, 4] (original_nodes, original_nodes, tau_max+1)
存储位置: nn.Parameter(self.causal_mask_kernel)
初始化方式:
如果有先验掩码: init_logits = (prior_mask_kernel * 3.0) - 1.0
随机初始化: torch.randn(original_nodes, original_nodes, tau_max + 1) * 0.1
2. 温度控制的Sigmoid变换
函数: get_small_mask() → torch.sigmoid(logits / self.temperature)
形状: [100, 100, 4] → [100, 100, 4]
3. 时间维度扩展
函数: get_causal_mask()
扩展逻辑:
将[100, 100, 4]扩展为[400, 400]
按时间滞后块复制: (t, t+lag)位置复制对应[100, 100]块
最终形状: [400, 400]
4. 降维聚合
函数: get_causal_mask_reduced()
聚合方式: torch.mean(small_mask, dim=2) (沿时间滞后维度平均)
输出形状: [100, 100]


保存的Mask形状
1. 完整掩码矩阵 (full_mask_matrix_*.npy)
形状: [400, 400]
来源: model.get_causal_mask().detach().cpu().numpy()
用途: 保持时间扩展后的完整结构
2. 降维掩码矩阵 (reduced_mask_matrix_*.npy)
形状: [100, 100]
来源: model.get_causal_mask_reduced(original_nodes, tau_max).detach().cpu().numpy()
用途: 原始节点维度的聚合结果
3. 模型状态中的掩码
参数形状: [100, 100, 4] (存储在causal_mask_kernel)
完整形状: [400, 400] (通过get_causal_mask()获取)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os
from pathlib import Path
import sys
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import pickle
import re


sys.path.append(str(Path(__file__).parent.parent))

# 从CADGNN导入相关组件
from Module.CADGNN import CADGNN, load_data
from utils.get_data_file import get_project_root


# ==========================================
# 辅助函数：矩阵扩展
# ==========================================

def build_global_scaler(feature_file_path, target_dim=5085, original_nodes=100):
    """
    在原始特征（100×10000，或转置后）上拟合全局 StandardScaler。
    只用前 target_dim 列来拟合，确保 transform 维度匹配后续 5085。
    """
    import numpy as np
    import pandas as pd

    p = str(feature_file_path)
    if p.endswith('.npy'):
        raw = np.load(p)  # 可能是 [Time, Nodes] 或 [Nodes, Time]
    else:
        try:
            raw = np.loadtxt(p, delimiter=',')
        except ValueError:
            raw = np.loadtxt(p, delimiter=',', skiprows=1)
        except Exception:
            # 最稳妥：pandas 兜底
            df = pd.read_csv(p, header=None)
            # 若第一列是索引（字符串），去掉
            if df.dtypes[0] == 'object':
                df = df.iloc[:, 1:]
            raw = df.values

    # 保证形状为 [Nodes, Time]，即 [100, 10000]
    if raw.shape[0] > raw.shape[1] and raw.shape[1] == original_nodes:
        raw = raw.T  # 由 [Time, Nodes] 转到 [Nodes, Time]

    # 维度对齐（只保留 transform 会用到的前 target_dim 列）
    # 注意：你后面会把列裁到 target_dim=5085，这里拟合同样的 5085 列
    raw = raw[:, :target_dim]  # [100, 5085]

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(raw)  # 用全局数据拟合 μ、σ
    return scaler

def filter_pcmci_matrix_by_graph(val_matrix, graph):
    """
    根据 graph 中的连接关系过滤 val_matrix，只保留显著连接（-->）。
    """
    # 初始化全零矩阵，形状与 val_matrix 一致
    filtered_matrix = np.zeros_like(val_matrix)

    # 获取维度
    n_targets, n_sources, n_lags = val_matrix.shape

    for target in range(n_targets):
        for source in range(n_sources):
            for lag in range(n_lags):
                # 获取连接符号，处理可能的字节编码问题
                relation = graph[target, source, lag]
                if isinstance(relation, bytes):
                    relation = relation.decode('utf-8')

                # 只有当 graph 标记为确定的因果方向 '-->' 时，才保留 val_matrix 的值
                # 注意：'o-o' (不确定) 或 '<--' (反向) 通常被视为当前方向无因果或存疑
                if relation == '-->':
                    filtered_matrix[target, source, lag] = val_matrix[target, source, lag]

    return filtered_matrix

def normalize_matrix_with_sign(matrix):
    """
    标准化矩阵保留正负号信息
    对于有正负值的矩阵(如PCMCI)，使用最大绝对值进行标准化
    """
    # 获取非零元素
    non_zero_elements = matrix[matrix != 0]

    if len(non_zero_elements) == 0:
        # 如果所有元素都是0，返回原矩阵
        return matrix

    # 获取最大绝对值
    max_abs_val = np.max(np.abs(non_zero_elements))

    if max_abs_val == 0:
        return matrix

    # 标准化：除以最大绝对值
    normalized = np.where(matrix != 0, matrix / max_abs_val, 0.0)
    return normalized


def normalize_matrix_robust(matrix, percentile=95.0):
    """
    鲁棒归一化：解决 SCMI 稀疏问题的关键。
    使用分位数（如 95%）作为最大值，而不是使用绝对最大值。
    这样可以防止极个别的大值（如自回归强连接）把其他弱连接"压"成0。
    """
    if matrix is None: return None

    # 转换为浮点型并取绝对值（SCMI通常是非负的，但以防万一）
    matrix_abs = np.abs(np.array(matrix, dtype=float))

    # 如果全是0，直接返回
    if np.all(matrix_abs == 0):
        return matrix_abs

    # 1. 计算分位数 (Robust Max)
    # 我们只关心非零值的分布，避免大量的0拉低分位数
    non_zeros = matrix_abs[matrix_abs > 1e-6]
    if len(non_zeros) > 0:
        robust_max = np.percentile(non_zeros, percentile)
    else:
        robust_max = matrix_abs.max()

    # 防止除零（极罕见情况）
    if robust_max == 0:
        robust_max = 1.0

    # 2. 截断 (Clip)
    # 所有超过 robust_max 的值都被强行设为 robust_max
    # 比如 robust_max 是 0.5，原来是 10.0 的值现在变成 0.5
    matrix_clipped = np.clip(matrix_abs, 0, robust_max)

    # 3. 归一化
    # 现在的范围是 [0, 1]
    # 原来 0.5 的值现在变成了 1.0，原来 0.05 的值现在变成 0.1 (可见了！)
    normalized = matrix_clipped / robust_max

    return normalized


def expand_and_merge_perturbed_matrix(pcmci_3d, scmi_2d, n_nodes, tau_max,
                                      pcmci_weight=1.0, scmi_weight=1.0):
    """
    标准化 (Robust) -> 加权合并 -> 扩展
    """
    n_time_layers = tau_max + 1
    expanded_size = n_time_layers * n_nodes
    expanded_matrix = np.zeros((expanded_size, expanded_size))

    # 1. 预处理数据类型
    if isinstance(pcmci_3d, torch.Tensor): pcmci_3d = pcmci_3d.cpu().numpy()
    if isinstance(scmi_2d, torch.Tensor): scmi_2d = scmi_2d.cpu().numpy()

    pcmci_3d = np.array(pcmci_3d, dtype=float)
    scmi_2d = np.array(scmi_2d, dtype=float)

    # 维度修正
    if len(pcmci_3d.shape) == 2:
        pcmci_3d = pcmci_3d[:, :, np.newaxis]

    # 2. === 标准化 ===

    # PCMCI: 保持 [-1, 1] (PCMCI 通常分布比较好，MinMax 也可以，但这里保持原逻辑)
    # 注意：如果 PCMCI 也很稀疏，也可以考虑对 PCMCI 用 Robust，但通常 PCMCI 不需要
    pcmci_norm = normalize_matrix_with_sign(pcmci_3d)

    # SCMI: [关键修改] 使用 Robust 归一化
    # 这里的 scmi_2d 是 [N, N] 矩阵
    # 使用 95% 分位数，意味着把前 95% 强度的连接都拉伸显示出来
    scmi_norm = normalize_matrix_robust(scmi_2d, percentile=95.0)

    # 3. 填充扩展矩阵
    for t in range(n_time_layers):
        # 确定行列索引范围
        row_start = t * n_nodes
        row_end = (t + 1) * n_nodes
        col_start = t * n_nodes
        col_end = (t + 1) * n_nodes

        # --- Lag 0: PCMCI(0) + SCMI 的加权融合 ---
        # 两个矩阵都已经归一化到相似的量级 (~0到1)
        pcmci_lag0 = pcmci_norm[:, :, 0]

        # 加权求和
        combined_lag0 = (pcmci_weight * pcmci_lag0) + (scmi_weight * scmi_norm)

        # 填充 (注意转置 .T 以适配 GNN 的 source->target 流向)
        expanded_matrix[row_start:row_end, col_start:col_end] = combined_lag0.T

        # --- Lag k: 仅 PCMCI (加权处理) ---
        for lag in range(1, tau_max + 1):
            target_t = t + lag
            if target_t < n_time_layers:
                col_start_lag = target_t * n_nodes
                col_end_lag = (target_t + 1) * n_nodes

                if lag < pcmci_norm.shape[2]:
                    # 保持权重一致性
                    pcmci_lagk = pcmci_norm[:, :, lag] * pcmci_weight
                    expanded_matrix[row_start:row_end, col_start_lag:col_end_lag] = pcmci_lagk.T

    return expanded_matrix


def get_stationary_prior_mask(data_list, original_nodes=100, tau_max=3):
    """
    提取平稳的先验Mask。
    返回形状: [original_nodes, original_nodes, tau_max + 1]
    逻辑：无论边出现在哪个时间窗口，都将其映射回 Lag 维度并取最大值。
    """
    time_layers = tau_max + 1
    # 初始化核心 tensor [100, 100, 4]
    prior_kernel = torch.zeros(original_nodes, original_nodes, time_layers)

    print("Computing stationary prior mask kernel...")
    count = 0

    # 只需要遍历部分数据即可获取结构，全遍历也可以
    for _, edge_index, _ in data_list:
        # edge_index: [2, E]
        src, dst = edge_index[0], edge_index[1]

        # 计算节点所在的层 (0, 1, 2, 3)
        layer_src = src // original_nodes
        layer_dst = dst // original_nodes

        # 计算对应的原始节点 ID (0-99)
        node_src = src % original_nodes
        node_dst = dst % original_nodes

        # 计算滞后 (Lag)
        # 如果是 src(t) -> dst(t+k)，则 lag = k
        # 注意处理反向边或者同一层的情况
        lag = layer_dst - layer_src

        # 只要合法的滞后 (0 <= lag <= tau_max)
        valid_mask = (lag >= 0) & (lag <= tau_max)

        if valid_mask.any():
            valid_src = node_src[valid_mask]
            valid_dst = node_dst[valid_mask]
            valid_lag = lag[valid_mask]

            # 填充 Kernel (置为1)
            # 使用 index_put_ 或者简单的赋值
            prior_kernel[valid_src, valid_dst, valid_lag] = 1.0

        count += 1

    print(f"  Aggregated {count} graphs.")
    print(f"  Prior kernel density: {prior_kernel.mean():.4f}")
    return prior_kernel


def process_features_from_file(feature_file_path, target_nodes=400, target_dim=5085, original_nodes=100, tau_max=3,scaler=None):
    """
    加载并处理特征文件
    1. 加载 100 nodes x 10000 features (CSV 或 NPY)
    2. 扩展行: 100 -> 400 (按时间滞后复制)
    3. 裁剪列: 10000 -> 5085
    """
    print(f"Loading features from: {feature_file_path}")
    feature_file_path = str(feature_file_path)  # 确保是字符串路径

    try:
        features = None
        # 处理 .npy 文件
        if feature_file_path.endswith('.npy'):
            features = np.load(feature_file_path)
        # 处理 .csv 文件
        elif feature_file_path.endswith('.csv'):
            try:
                # 尝试直接读取，假设没有header
                features = np.loadtxt(feature_file_path, delimiter=',')
            except ValueError:
                print("  Warning: ValueError during loadtxt, trying with skiprows=1 (assuming header)...")
                try:
                    # 如果报错，尝试跳过第一行（假设有header）
                    features = np.loadtxt(feature_file_path, delimiter=',', skiprows=1)
                except Exception as e:
                    # 尝试使用pandas读取（如果安装了pandas），因为它更健壮
                    try:
                        import pandas as pd
                        df = pd.read_csv(feature_file_path, header=None if 'header' not in str(e) else 0)
                        # 如果第一列是索引（字符串），则删除
                        if df.dtypes[0] == 'object':
                            df = df.iloc[:, 1:]
                        features = df.values
                    except ImportError:
                        raise ValueError(f"Failed to load CSV and pandas is not installed. Error: {e}")

        if features is None:
            raise ValueError("Unsupported file format or load failure.")

        print(f"  Raw loaded shape: {features.shape}")

        # 维度检查与转置
        # 目标是 [100, 10000] (Nodes, Features)
        # 如果读入的是 [10000, 100] (Time, Nodes)，则转置
        if features.shape[0] > features.shape[1] and features.shape[1] == original_nodes:
            print("  Transposing features from [Time, Nodes] to [Nodes, Time]")
            features = features.T

        # 再次检查节点数
        if features.shape[0] != original_nodes:
            # 最后的容错：有时候CSV包含索引列导致 shape变成 (100, 10001)
            if features.shape[0] == original_nodes + 1:
                features = features[1:, :]  # 去头
            elif features.shape[1] == original_nodes + 1:
                features = features[:, 1:].T  # 去索引列并转置
            else:
                print(f"  Warning: Expected {original_nodes} nodes, got {features.shape[0]}. Check data format.")

        current_nodes, current_dim = features.shape

        # 1. 扩展节点维度 (行)
        # 目标是400个节点，即 (tau_max + 1) * original_nodes
        time_layers = tau_max + 1
        expected_rows = time_layers * original_nodes

        if features.shape[0] == original_nodes:
            # 复制扩展：将 100 个节点的特征复制 4 份
            print(f"  Expanding nodes from {original_nodes} to {expected_rows}...")
            expanded_features = np.tile(features, (time_layers, 1))
        elif features.shape[0] == expected_rows:
            expanded_features = features
        else:
            # 如果维度实在对不上，强行截断或补零（防止程序崩溃，但会警告）
            print(f"  Warning: Dimension mismatch. Resizing strictly to {expected_rows} rows.")
            if features.shape[0] > expected_rows:
                expanded_features = features[:expected_rows, :]
            else:
                padding = np.zeros((expected_rows - features.shape[0], features.shape[1]))
                expanded_features = np.vstack([features, padding])

        # 2. 调整特征维度 (列)
        # 目标是 5085 个特征
        print(f"  Adjusting feature dimension from {expanded_features.shape[1]} to {target_dim}...")
        if expanded_features.shape[1] >= target_dim:
            final_features = expanded_features[:, :target_dim]
        else:
            # 如果特征不足，补零
            padding = np.zeros((expanded_features.shape[0], target_dim - expanded_features.shape[1]))
            final_features = np.hstack([expanded_features, padding])
        if scaler is not None:
            # 要求 scaler 是用形状 [*, target_dim] 拟合的（上面①已保证）
            final_features = scaler.transform(final_features)

        print(f"  Final processed feature shape: {final_features.shape}")  # 应该是 [400, 5085]
        return torch.FloatTensor(final_features)

    except Exception as e:
        print(f"Error processing features: {e}")
        import traceback
        traceback.print_exc()
        # 兜底返回随机特征
        return torch.randn(target_nodes, target_dim)

def analyze_mask_distribution(mask_matrix, title="Mask Distribution Analysis"):
    """
    分析掩码分布并提供可视化
    """
    mask_np = mask_matrix.detach().cpu().numpy()

    print(f"\n{title}:")
    print(f"  Shape: {mask_np.shape}")
    print(f"  Min: {mask_np.min():.6f}")
    print(f"  Max: {mask_np.max():.6f}")
    print(f"  Mean: {mask_np.mean():.6f}")
    print(f"  Std: {mask_np.std():.6f}")
    print(f"  Median: {np.median(mask_np):.6f}")

    # 计算分位数
    q10, q25, q50, q75, q90 = np.percentile(mask_np, [10, 25, 50, 75, 90])
    print(f"  10th percentile: {q10:.6f}")
    print(f"  25th percentile: {q25:.6f}")
    print(f"  50th percentile: {q50:.6f}")
    print(f"  75th percentile: {q75:.6f}")
    print(f"  90th percentile: {q90:.6f}")

    # 计算接近0.5的值的比例
    near_half = np.sum((mask_np > 0.45) & (mask_np < 0.55))
    total_elements = mask_np.size
    print(f"  Values near 0.5 (0.45-0.55): {near_half}/{total_elements} ({100 * near_half / total_elements:.2f}%)")

    # 计算显著偏离0.5的值
    significant_deviation = np.sum((mask_np < 0.2) | (mask_np > 0.8))
    print(
        f"  Values with significant deviation (<0.2 or >0.8): {significant_deviation}/{total_elements} ({100 * significant_deviation / total_elements:.2f}%)")

    return {
        'min': mask_np.min(),
        'max': mask_np.max(),
        'mean': mask_np.mean(),
        'std': mask_np.std(),
        'median': np.median(mask_np),
        'q10': q10,
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'q90': q90,
        'near_half_ratio': 100 * near_half / total_elements,
        'significant_deviation_ratio': 100 * significant_deviation / total_elements
    }

def plot_losses(loss_history, save_path):
    """
    绘制损失曲线
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(loss_history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(loss_history['lrtr_loss'])  # 修改键名为 'causal_lrtr_loss'
    plt.title('Causal LRTR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(loss_history['binary_loss'])
    plt.title('binary_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def enhance_mask_contrast(mask_np, threshold_low=0.3, threshold_high=0.7):
    """
    增强掩码对比度的函数
    """
    # 方法1: Sigmoid变换增强对比度
    # 将数据映射到更极端的值
    enhanced = 1 / (1 + np.exp(-10 * (mask_np - 0.5)))  # Sigmoid增强

    # 方法2: 分段线性变换
    # enhanced = np.zeros_like(mask_np)
    # mask_low = mask_np <= threshold_low
    # mask_mid = (mask_np > threshold_low) & (mask_np < threshold_high)
    # mask_high = mask_np >= threshold_high

    # enhanced[mask_low] = mask_np[mask_low] * 0.5  # 压缩低值
    # enhanced[mask_mid] = 0.5 + (mask_np[mask_mid] - 0.5) * 0.2  # 压缩中间值
    # enhanced[mask_high] = 0.9 + (mask_np[mask_high] - 0.7) * 0.1  # 扩展高值

    # 方法3: 幂函数增强 (可选)
    # enhanced = np.power(mask_np, 2)  # 平方变换，增强高值

    return enhanced


def visualize_mask_heatmap(mask_matrix, save_path, title="Causal Mask Heatmap", original_nodes=100, tau_max=3,
                           threshold_low=0.3, threshold_high=0.7, colormap='Reds'):
    """
    可视化掩码矩阵热图，增强对比度
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 转换为numpy数组
    mask_np = mask_matrix.detach().cpu().numpy()

    # 如果是扩展维度，进行降维
    # if mask_np.shape[0] > original_nodes:
    #     print(f"Reducing mask from {mask_np.shape[0]}x{mask_np.shape[1]} to {original_nodes}x{original_nodes}")
    #     # 取原始节点部分
    #     mask_np = mask_np[:original_nodes, :original_nodes]

    # 数据增强处理 - 增强对比度
    enhanced_mask = enhance_mask_contrast(mask_np, threshold_low, threshold_high)

    plt.figure(figsize=(10, 8))

    # 使用更适合的色图和中心值
    sns.heatmap(
        enhanced_mask,
        cmap=colormap,  # 使用红蓝对比色图
        center=0.5,  # 中心值设为0.5
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Mask Value (Enhanced)'}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mask heatmap saved to: {save_path}")
    print(f"Enhanced mask stats - Min: {enhanced_mask.min():.4f}, Max: {enhanced_mask.max():.4f}, "
          f"Mean: {enhanced_mask.mean():.4f}, Std: {enhanced_mask.std():.4f}")


def save_results(model, loss_history, save_dir="causal_invariant_results", original_nodes=100, tau_max=4):
    """
    保存结果，包含时间戳和掩码热图（包括降维版本）
    """
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建带时间戳的保存目录
    save_path = Path(save_dir) / f"results_pems_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存模型 - 包含时间戳
    model_path = save_path / f"causal_invariant_subgraph_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'mask_matrix': model.get_causal_mask().detach().cpu().numpy(),
        'reduced_mask_matrix': model.get_causal_mask_reduced(original_nodes, tau_max).detach().cpu().numpy(),
        'loss_history': loss_history
    }, model_path)

    # 保存完整掩码矩阵 - 包含时间戳
    full_mask_path = save_path / f"full_mask_matrix_{timestamp}.npy"
    np.save(full_mask_path, model.get_causal_mask().detach().cpu().numpy())

    # 保存降维掩码矩阵 - 包含时间戳
    reduced_mask_path = save_path / f"reduced_mask_matrix_{timestamp}.npy"
    reduced_mask = model.get_causal_mask_reduced(original_nodes, tau_max)
    np.save(reduced_mask_path, reduced_mask.detach().cpu().numpy())

    # 绘制损失曲线 - 包含时间戳
    plot_losses(loss_history, save_path / f"training_loss_{timestamp}.png")

    # 绘制完整掩码热图
    full_heatmap_path = save_path / f"full_mask_heatmap_{timestamp}.png"
    visualize_mask_heatmap(
        model.get_causal_mask(),
        full_heatmap_path,
        f"Full Causal Mask Heatmap (Epoch {len(loss_history['total_loss'])})",
        original_nodes=400
    )

    # 绘制降维掩码热图
    reduced_heatmap_path = save_path / f"reduced_mask_heatmap_{timestamp}.png"
    visualize_mask_heatmap(
        reduced_mask,
        reduced_heatmap_path,
        f"Reduced Causal Mask Heatmap (Epoch {len(loss_history['total_loss'])})",
        original_nodes=original_nodes
    )

    print(f"Results saved to: {save_path}")
    return model_path, reduced_mask_path

# ==========================================
# 模型定义
# ==========================================

class CausalGatedBilinearProjection(nn.Module):
    def __init__(self, hidden_dim):
        super(CausalGatedBilinearProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, H, A):
        H_proj = self.W_p(H)  # [N, d]
        Phi = torch.matmul(H_proj, H.t())  # [N, N]
        Phi_filtered = A * Phi  # [N, N]
        return Phi_filtered


class CausalInvariantSubgraphExtractor(nn.Module):
    def __init__(self, cadgnn_model, input_dim, hidden_dim, num_nodes,
                 original_nodes=100, tau_max=3, prior_mask_kernel=None):  # 注意参数变化

        super(CausalInvariantSubgraphExtractor, self).__init__()
        self.cadgnn = cadgnn_model

        # 冻结 CADGNN
        for param in self.cadgnn.parameters():
            param.requires_grad = False

        self.original_nodes = original_nodes
        self.tau_max = tau_max
        self.target_nodes = num_nodes  # 400

        # === 核心修改：参数形状改为 [100, 100, 4] ===
        # 这样模型只能学习一套规则，不管是在 t=0 还是 t=3 应用
        if prior_mask_kernel is not None:
            print("Initializing mask kernel using Stationary Prior (Soft Mode)...")
            if isinstance(prior_mask_kernel, np.ndarray):
                prior_mask_kernel = torch.from_numpy(prior_mask_kernel).float()

            # === 修改处：温和的初始化 ===
            # Prior=1 -> 3-1 = 2 -> Sigmoid(2) ≈ 0.88 (保留高置信度)
            # Prior=0 -> 0-1 = -1 -> Sigmoid(-1) ≈ 0.27 (给予探索空间，而不是直接杀死)
            init_logits = (prior_mask_kernel * 3.0) - 1.0

            # 加入一点噪声防止完全对称
            noise = torch.randn_like(init_logits) * 0.1
            self.causal_mask_kernel = nn.Parameter(init_logits + noise)
        else:
            print("Initializing mask kernel randomly...")
            self.causal_mask_kernel = nn.Parameter(torch.randn(original_nodes, original_nodes, tau_max + 1) * 0.1)

        self.causal_bilinear = CausalGatedBilinearProjection(hidden_dim)
        self.temperature = 2.0

    def set_temperature(self, temp):
        self.temperature = temp

    def get_small_mask(self):
        """获取 100x100x4 的基础 Mask"""
        logits = self.causal_mask_kernel
        return torch.sigmoid(logits / self.temperature)

    def get_causal_mask(self):
        """
        核心逻辑：将 100x100x4 的核 扩展为 400x400 的全图 Mask
        强制保证平稳性：(0,1)块 和 (1,2)块 使用完全相同的参数
        """
        small_mask = self.get_small_mask()  # [N, N, Lags]

        device = small_mask.device
        N = self.original_nodes
        L = self.tau_max + 1  # Time layers (4)
        total_nodes = self.target_nodes

        # 初始化大图 [400, 400]
        # full_mask = torch.zeros(total_nodes, total_nodes, device=device)
        full_mask = 0.05*torch.ones(total_nodes, total_nodes, device=device)

        # 遍历每一个滞后 Lag (0, 1, 2, 3)
        for lag in range(L):
            # 获取该 Lag 对应的 100x100 权重矩阵
            # shape: [N, N]
            weight_matrix = small_mask[:, :, lag]

            # 我们需要把这个矩阵复制到 full_mask 的所有 (t, t+lag) 对角线上
            # 例如 Lag 0: (0,0), (1,1), (2,2), (3,3)
            # 例如 Lag 1: (0,1), (1,2), (2,3)

            # 我们可以用 block_diag 的变体或者循环切片来实现
            num_blocks = L - lag  # Lag 0有4个块，Lag 1有3个块...

            for t in range(num_blocks):
                row_start = t * N
                row_end = (t + 1) * N
                col_start = (t + lag) * N
                col_end = (t + lag + 1) * N

                # 关键：所有时间步都赋值同一个 weight_matrix！
                # 物理意义：规则在时间上是不变的
                full_mask[row_start:row_end, col_start:col_end] = weight_matrix

                # 注意：这里假设了数据流向是 row -> col (t -> t+lag)
                # 之前的代码似乎用过 .T，如果发现方向反了，在这里加 .t()

        return full_mask

    def get_causal_mask_reduced(self, original_nodes=100, tau_max=4):
        """
        现在的 reduced mask 变得非常简单，直接就是我们的参数 (经过sigmoid)
        """
        # 调整维度顺序以匹配之前的输出习惯 [N, N, Lags] -> [N, N] ?
        # 之前的 reduced_mask 似乎是聚合后的 [N, N]，这里我们可以取 max 或 mean
        small_mask = self.get_small_mask() # [N, N, Lags]
        # 聚合所有 Lag 变成一张总结图 (如果你需要的话)
        # 或者直接返回 [N, N] (Lag 1 + Lag 2...?)
        # 这里为了兼容旧的可视化逻辑，我们取 max 聚合所有 lag
        return torch.mean(small_mask, dim=2) # torch.max(small_mask, dim=2)[0]

    def _dense_adj_to_edge_index_weight(self, adj_matrix):
        """
        将稠密邻接矩阵转换为edge_index和edge_weight
        关键点：保留edge_weight的梯度，这允许Mask通过GNN学习
        """
        # 找到非零元素（或大于极小阈值）的索引
        # 注意：为了梯度传播，我们不能只选非零，通常要对Mask后的结果全量计算，
        # 但为了GNN效率，通常取大于极小值的边。
        # 如果Mask是Soft的且未完全二值化，这可能很稠密。

        # 策略：选择adj_matrix中值大于阈值的边构建结构，但把adj_matrix的值作为权重
        threshold = 1e-8
        mask_idx = torch.abs(adj_matrix) > threshold
        edge_index = torch.nonzero(mask_idx).t()

        if edge_index.size(1) == 0:
            # 防止空图
            edge_index = torch.tensor([[0], [0]], device=adj_matrix.device)
            edge_weight = torch.tensor([0.0], device=adj_matrix.device, requires_grad=True)
        else:
            # 关键：从adj_matrix中gather出数值，保留梯度
            edge_weight = adj_matrix[edge_index[0], edge_index[1]]

        return edge_index, edge_weight

    def forward(self, causal_perturbed_data_list):
        """
        修改后的前向传播流程：
        1. 获取全局掩码 Mask
        2. 遍历每个扰动图：
           b. 应用掩码: Adj_new = Adj_expanded * Mask
           c. 输入 GNN: Encoder(Feat_expanded, Adj_new) -> Z
           d. 双线性投影
        """
        causal_mask_matrix = self.get_causal_mask()  # [400, 400]
        causal_features_list = []

        target_nodes = self.target_nodes

        for i, (features, edge_index, edge_weight) in enumerate(causal_perturbed_data_list):
            current_nodes = features.size(0)

            # 1. 扩展邻接矩阵 (Structure Expansion)
            # 先转回稠密矩阵进行扩展
            adj_dense = torch.zeros(current_nodes, current_nodes, device=features.device)
            adj_dense[edge_index[0], edge_index[1]] = edge_weight

            # 特征归一化
            # feat_norm = (features - features.min()) / (features.max() - features.min() + 1e-8)
            feat_norm = features
            # 3. 应用掩码 (Masking BEFORE GNN)
            # 核心修改：在这里应用掩码，决定GNN看到什么结构
            # 这样被掩码掉的边不会传播特征，插零的节点保持孤立
            adj_masked = adj_dense * causal_mask_matrix

            # 4. 转换回稀疏格式输入GNN (Encoding)
            masked_edge_index, masked_edge_weight = self._dense_adj_to_edge_index_weight(adj_masked)

            # 输入GNN获取表示
            # 注意：这里使用的是经过扩展和掩码后的图
            z_c = self.cadgnn.causal_encoder(feat_norm, masked_edge_index, masked_edge_weight)

            # 5. 解码/投影 (Decoding)
            # 使用学到的表示和掩码后的图结构进行投影
            # Phi = self.causal_bilinear(z_c, adj_masked)
            # causal_features_list.append(Phi)
            causal_features_list.append(z_c)

        causal_features_tensor = torch.stack(causal_features_list, dim=2) if causal_features_list else torch.empty(0, 0,
                                                                                                                   0)
        return causal_features_tensor, causal_mask_matrix

# ==========================================
# 数据加载与主流程
# ==========================================

def load_causal_perturbed_graphs(pcmci_dir, scmi_dir, feature_file_path, num_nodes=100, tau_max=3,scaler=None):
    """
    加载并合并 PCMCI 和 SCMI 扰动图
    修复：检测 PCMCI 维度，如果是 (N-1, N-1) (即删除了扰动节点)，则在 node_id 处补零恢复为 (N, N)
    """
    import pickle
    import numpy as np
    import torch

    pcmci_dir = Path(pcmci_dir)
    scmi_dir = Path(scmi_dir)

    # 查找所有 PCMCI 结果文件
    pkl_files = []
    for sub_dir in pcmci_dir.glob('pcmci_results_node_*'):
        if sub_dir.is_dir():
            files = list(sub_dir.glob('*.pkl'))
            if files:
                try:
                    # 从文件夹名提取节点ID 'pcmci_results_node_5' -> 5
                    node_id = int(sub_dir.name.split('_')[-1])
                    pkl_files.append((files[0], node_id))
                except ValueError:
                    print(f"Skipping directory with invalid format: {sub_dir}")

    # 按节点ID排序
    pkl_files.sort(key=lambda x: x[1])

    # 加载统一的外部特征文件
    base_features = process_features_from_file(
        feature_file_path,
        target_nodes=num_nodes * (tau_max + 1),
        target_dim=5085,
        original_nodes=num_nodes,
        tau_max=tau_max,
        scaler=scaler
    )

    causal_perturbed_data_list = []

    print(f"Found {len(pkl_files)} PCMCI perturbation files. Matching with SCMI...")

    # 创建可视化目录
    vis_dir = Path("causal_invariant_results_pems") / "input_data_2601161538"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i, (pcmci_path, node_id) in enumerate(pkl_files):
        try:
            # 1. 加载 PCMCI 数据
            with open(pcmci_path, 'rb') as f:
                data = pickle.load(f)

            # 提取矩阵和图结构
            if isinstance(data, dict):
                # 优先获取 val_matrix，部分版本可能是 'val'
                val_matrix = data.get('val_matrix', data.get('val', None))
                graph = data.get('graph', None)
            else:
                # 兼容旧格式
                val_matrix = data
                # 如果没有 graph 结构，无法进行过滤，建议跳过或使用默认
                # 这里假设必须有 graph 用于过滤
                print(f"Node {node_id}: Data format warning (no dict), skipping.")
                continue

            if val_matrix is None or graph is None:
                continue

            # 确保转换为 numpy 数组
            val_matrix = np.array(val_matrix)
            graph = np.array(graph)

            # ============================================================
            # [关键修改] 维度修复：针对被扰动节点 (N-1 -> N) 进行补零
            # ============================================================
            # 检查维度：如果行数比预期少1，说明该节点在PCMCI结果中被移除了
            if val_matrix.shape[0] == num_nodes - 1:
                # print(f"  Restoring perturbed node {node_id} (Padding 0s)...")

                # 1. 补全数值矩阵 (val_matrix)
                # axis=0 (行/Target): 在 node_id 处插入一整行0
                val_matrix = np.insert(val_matrix, node_id, 0, axis=0)
                # axis=1 (列/Source): 在 node_id 处插入一整列0
                val_matrix = np.insert(val_matrix, node_id, 0, axis=1)

                # 2. 补全结构矩阵 (graph) - 插入空字符串或占位符
                # 注意：numpy 插入到字符串数组时会自动处理类型
                graph = np.insert(graph, node_id, '', axis=0)
                graph = np.insert(graph, node_id, '', axis=1)

            # ============================================================
            # [关键修改] 噪声过滤：只保留 graph 中有 '-->' 连接的数值
            # ============================================================
            filtered_pcmci = filter_pcmci_matrix_by_graph(val_matrix, graph)

            # ============================================================
            # SCMI 加载部分 (修正读取 + 自动插零)
            # ============================================================
            scmi_candidates = [
                scmi_dir / f"scmi_perturbed_graph_node_{node_id}.npy",
            ]

            scmi_matrix = None
            found_scmi_path = None

            # 1. 寻找文件
            for p in scmi_candidates:
                if p.exists():
                    found_scmi_path = p
                    break

            if found_scmi_path is None:
                print(f"  [SCMI Warning] No file found for Node {node_id}. Using zeros.")
                scmi_matrix = np.zeros((num_nodes, num_nodes))
            else:
                try:
                    # 2. 读取文件 (允许 pickle 以支持字典)
                    raw_data = np.load(found_scmi_path, allow_pickle=True)

                    # 3. 解析数据结构 (提取 'graph')
                    if raw_data.ndim == 0 and isinstance(raw_data.item(), dict):
                        # 情况: np.save 保存的字典，需用 .item() 取出
                        data_dict = raw_data.item()
                        scmi_matrix = data_dict.get('graph', None)
                    elif isinstance(raw_data, dict):
                        # 情况: 直接是字典
                        scmi_matrix = raw_data.get('graph', None)
                    else:
                        # 情况: 直接是矩阵
                        scmi_matrix = raw_data

                    if scmi_matrix is None:
                        print(f"  [SCMI Error] 'graph' key missing in {found_scmi_path.name}. Using zeros.")
                        scmi_matrix = np.zeros((num_nodes, num_nodes))
                    else:
                        # 转 numpy 确保操作兼容
                        scmi_matrix = np.array(scmi_matrix)

                        # 4. 打印原始读取尺寸 (调试用)
                        # print(f"  [SCMI Read] Node {node_id}: Raw Shape {scmi_matrix.shape} from {found_scmi_path.name}")

                        # 5. 维度降维 (如果是 3D [N, N, 1] -> 2D [N, N])
                        if scmi_matrix.ndim > 2:
                            scmi_matrix = scmi_matrix[:, :, 0]

                        # 6. 核心修改：检测是否需要插零 (N-1 -> N)
                        # 如果读入的是 99x99 (而目标是 100x100)，说明该节点被移除，需要插回去
                        if scmi_matrix.shape[0] == num_nodes - 1:
                            # print(f"  [SCMI Padding] Node {node_id}: Restoring dimension {scmi_matrix.shape} -> ({num_nodes}, {num_nodes})...")

                            # 在行 (axis=0) 的 node_id 位置插入 0
                            scmi_matrix = np.insert(scmi_matrix, node_id, 0, axis=0)
                            # 在列 (axis=1) 的 node_id 位置插入 0
                            scmi_matrix = np.insert(scmi_matrix, node_id, 0, axis=1)

                except Exception as e:
                    print(f"  [SCMI Error] Failed to load {found_scmi_path}: {e}")
                    scmi_matrix = np.zeros((num_nodes, num_nodes))

            # 7. 最终维度校验与兜底
            if scmi_matrix.shape != (num_nodes, num_nodes):
                print(
                    f"  [SCMI Warning] Shape Mismatch Node {node_id}: Got {scmi_matrix.shape}, expected ({num_nodes}, {num_nodes}). Force Zeroing.")
                scmi_matrix = np.zeros((num_nodes, num_nodes))
            # else:
            # if i < 3: # 只打印前几个成功的，避免刷屏
            # print(f"  [SCMI Success] Node {node_id} ready. Shape: {scmi_matrix.shape}")

            # ================= 3. 合并与扩展 =================
            expanded_adj = expand_and_merge_perturbed_matrix(
                filtered_pcmci,
                scmi_matrix,
                num_nodes,
                tau_max,
                pcmci_weight=1, # 可以根据置信度调整
                scmi_weight=1
            )
            # expanded_adj = expanded_adj * 10.0
            # === [新增] 可视化第一个融合图 ===
            if i == 0:
                print(f"  [Visualization] Plotting merged input graph for Node {node_id}...")
                plt.figure(figsize=(12, 10))
                # 使用 RdBu_r 色图因为 PCMCI可能有负值
                plt.imshow(expanded_adj, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
                plt.colorbar(label='Normalized Weighted Strength')
                plt.title(f"Merged Input Graph (Node {node_id})\nPCMCI(Norm)+SCMI(Norm) Expanded")

                # 画框区分时间层
                time_layers = tau_max + 1
                for tl in range(time_layers):
                    # 每一层的边界
                    plt.axhline(y=tl * num_nodes, color='black', linewidth=0.5, alpha=0.5)
                    plt.axvline(x=tl * num_nodes, color='black', linewidth=0.5, alpha=0.5)

                vis_path = vis_dir / "merged_input_sample_0.png"
                plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Merged input heatmap saved to: {vis_path}")

                # 打印统计
                print(
                    f"  Merged Stats - Min: {expanded_adj.min():.4f}, Max: {expanded_adj.max():.4f}, Mean: {expanded_adj.mean():.4f}")

            # 转为 Tensor (保持不变)
            adj_tensor = torch.FloatTensor(expanded_adj)
            threshold = 1e-6  # 归一化后值变小，阈值要足够小
            mask = torch.abs(adj_tensor) > threshold
            edges = torch.nonzero(mask).t()
            edge_weights = adj_tensor[mask]

            if edges.size(1) == 0:
                n = adj_tensor.size(0)
                edges = torch.arange(n).unsqueeze(0).repeat(2, 1)
                edge_weights = torch.ones(n)

            current_features = base_features.clone()
            causal_perturbed_data_list.append((current_features, edges, edge_weights))

            # if i >= 50: break  # 调试用

        except Exception as e:
            print(f"Error processing node {node_id}: {e}")
            continue

    return causal_perturbed_data_list


# ==========================================
# 训练与辅助函数 (保持大部分逻辑不变)
# ==========================================

def compute_lrtr_loss(Phi_list):
    if len(Phi_list) == 0: return torch.tensor(0.0, requires_grad=True)
    Phi_tensor = torch.stack(Phi_list, dim=2)
    Phi_mean = torch.mean(Phi_tensor, dim=2, keepdim=True).expand_as(Phi_tensor)
    lrtr_loss = torch.mean((Phi_tensor - Phi_mean) ** 2)
    return lrtr_loss


def train_causal_invariant_subgraph_extractor(model, causal_perturbed_data_list, num_epochs=100, lr=0.001,
                                              device='cuda', batch_size=4):
    model = model.to(device)

    # === 修复 1: 参数名称修改 ===
    # 旧代码: 'params': model.causal_mask_theta
    # 新代码: 'params': model.causal_mask_kernel
    optimizer = Adam([
        {'params': model.causal_mask_kernel, 'lr': lr * 5.0},  # 这里必须改名为 kernel
        {'params': model.causal_bilinear.parameters(), 'lr': lr},
    ], weight_decay=1e-5)

    data_list_cpu = causal_perturbed_data_list

    loss_history = {'total_loss': [], 'lrtr_loss': [], 'sparsity_loss': [], 'binary_loss': []}
    num_samples = len(data_list_cpu)

    print(f"Start training... Batch Size: {batch_size}, LRTR Scaler: 1e5, Sparsity Delay: 20 epochs")

    for epoch in range(num_epochs):
        # 温度退火
        # temp = 2.0 * (0.98 ** epoch)
        # if temp < 0.1: temp = 0.1
        # model.set_temperature(temp)
        if epoch < 60:
            temp = 1.0
        else:
            # 60~100 逐步退火到 ~0.2
            progress = (epoch - 60) / 40.0
            temp = 1.0 - 0.4 * progress
            temp = max(0.1, temp)
        model.set_temperature(temp)

        model.train()
        perm = torch.randperm(num_samples)

        epoch_stats = {'total': 0, 'lrtr': 0, 'spar': 0, 'bin': 0}
        steps = 0

        for i in range(0, num_samples, batch_size):
            optimizer.zero_grad()
            indices = perm[i: i + batch_size]
            batch_data = []
            for idx in indices:
                f, ei, ew = data_list_cpu[idx]
                batch_data.append((f.to(device), ei.to(device), ew.to(device)))
            if not batch_data: continue

            # Forward
            causal_features, mask = model(batch_data)

            # --- Loss Calculation ---

            # 1. LRTR (因果不变性)
            if causal_features.size(2) > 1:
                phi_list = [causal_features[:, :, j] for j in range(causal_features.size(2))]
                raw_lrtr = compute_lrtr_loss(phi_list)
            else:
                raw_lrtr = torch.tensor(0.0, device=device, requires_grad=True)

            weighted_lrtr = raw_lrtr * 10000.0

            # 2. Sparsity (稀疏惩罚) - 针对 kernel 进行惩罚
            # 注意：这里我们应该惩罚 mask (Sigmoid后的值)，而不是 raw_logits
            # 获取当前的小尺寸 mask [100, 100, 4] 用于计算稀疏度，这样更准确
            small_mask = model.get_small_mask()

            sparsity_coeff = 2e-6
            if epoch < 50:
                sparsity_loss = torch.tensor(0.0, device=device)
            else:
                ramp_up = min(1.0, (epoch - 20) / 10.0)
                sparsity_loss = sparsity_coeff * torch.norm(small_mask, p=1) * ramp_up

            # 3. Binary Loss (二值化)
            # binary_loss = 0.01 * torch.mean(small_mask * (1 - small_mask))
            # Binary 分段权重：再加强一些
            if epoch < 60:
                bin_w = 1e-4
            elif epoch < 80:
                bin_w = 8e-4
            else:
                bin_w = 2e-3
            binary_loss = bin_w * torch.mean(small_mask * (1 - small_mask))

            # === 修复 2: 取消 Loss 的注释 ===
            # 如果不加 sparsity 和 binary，Mask 不会变稀疏，只会保持初始化的状态
            total_loss = weighted_lrtr + sparsity_loss + binary_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Record
            epoch_stats['total'] += total_loss.item()
            epoch_stats['lrtr'] += raw_lrtr.item()
            epoch_stats['spar'] += sparsity_loss.item()
            epoch_stats['bin'] += binary_loss.item()
            steps += 1

            del batch_data, causal_features, total_loss
            torch.cuda.empty_cache()

        # Avg Logs
        avg_total = epoch_stats['total'] / max(1, steps)
        avg_lrtr = epoch_stats['lrtr'] / max(1, steps)
        avg_spar = epoch_stats['spar'] / max(1, steps)

        loss_history['total_loss'].append(avg_total)
        loss_history['lrtr_loss'].append(avg_lrtr)
        loss_history['sparsity_loss'].append(avg_spar)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Total {avg_total:.10f} | LRTR(raw) {avg_lrtr:.2e} | Sparsity {avg_spar:.10f} | Binary {epoch_stats['bin'] / max(1, steps):.10f}")

    return model, loss_history


# ==========================================
# Main
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    # 路径配置
    cadgnn_model_path = "./CADGNN_results/pems/N100_T5085_hid128_lay5_ladv0.13_lw0.02_20260116_171042_model.pth"  # 替换实际路径
    pcmci_dir = project_root / "dataset" / "Perturbed_Causal_Graphs_PEMS_N100_1h_standardize_id"
    scmi_dir = project_root / "dataset" / "Perturbed_SCMI_Causal_Graphs_PEMS_N100_1h_standardize_id_2/scmi_perturbations"
    # **指定特征文件路径**
    feature_file_path = project_root / "dataset/pems_spatial_kmeans_topK" / "id_standardize_Env_A_timeseries.csv"  # 替换为你的特征文件路径

    # 1. 加载 CADGNN
    print("Loading CADGNN...")
    # 模拟加载，实际需确保路径正确
    checkpoint = torch.load(cadgnn_model_path, map_location=device) if os.path.exists(cadgnn_model_path) else None
    if checkpoint:
        config = checkpoint.get('model_config', {'input_dim': 5085, 'hidden_dim': 128})
        cadgnn = CADGNN(config['input_dim'], config['hidden_dim'], 5, 0.1, 0.02).to(device)
        cadgnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: Model not found, creating new for demo.")
        cadgnn = CADGNN(5085, 128, 5, 0.1, 0.02).to(device)

    # 先拟合全局 scaler（只用会被实际用到的前 5085 列）
    global_scaler = build_global_scaler(feature_file_path, target_dim=5085, original_nodes=100)


    # 2. 加载数据 (传入新的 scmi_dir)
    print("Loading and Merging Data (PCMCI + SCMI)...")
    data_list = load_causal_perturbed_graphs(
        pcmci_dir=pcmci_dir,
        scmi_dir=scmi_dir,           # 传入 SCMI 路径
        feature_file_path=str(feature_file_path),
        num_nodes=100,
        tau_max=3,
        scaler=global_scaler
    )

    if not data_list:
        print("No data loaded.")
        return

    # === 修改处：使用新的 stationary 先验提取函数 ===
    prior_mask_kernel = get_stationary_prior_mask(data_list, original_nodes=100, tau_max=3)
    prior_mask_kernel = prior_mask_kernel.to(device) # [100, 100, 4]


    # 3. 初始化并训练提取器
    print("Initializing Extractor...")
    extractor = CausalInvariantSubgraphExtractor(
        cadgnn,
        input_dim=5085,
        hidden_dim=128,
        num_nodes=400, # target_nodes
        original_nodes=100,  # 新增
        tau_max=3 ,           # 新增
        prior_mask_kernel=prior_mask_kernel # 参数名变了
    )


    # ==========================================
    # 新增：可视化初始化 Mask (扩展后 & 降维版)
    # ==========================================
    print("\nVisualizing Initialized Mask...")
    init_save_dir = Path("causal_invariant_results") / "initialization_pems"
    init_save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 获取扩展后的 400x400 Mask (get_causal_mask 内部已做扩展)
    init_full_mask = extractor.get_causal_mask()
    visualize_mask_heatmap(
        init_full_mask,
        init_save_dir / "init_full_mask_heatmap2601161538.png",
        title="Initialized Full Causal Mask (400x400)",
        original_nodes=400
    )

    # 2. 获取降维后的 100x100 Mask
    init_reduced_mask = extractor.get_causal_mask_reduced(original_nodes=100, tau_max=3)
    visualize_mask_heatmap(
        init_reduced_mask,
        init_save_dir / "init_reduced_mask_heatmap.png",
        title="Initialized Reduced Causal Mask (100x100)",
        original_nodes=100
    )
    print(f"Initialization maps saved to {init_save_dir}\n")
    # ==========================================



    print("Training...")
    trained_model, loss_history = train_causal_invariant_subgraph_extractor(
        extractor,
        data_list,
        num_epochs=100,
        lr=0.005,
        device=device,
        batch_size=4
    )

    # 分析训练后的掩码分布
    final_causal_mask = trained_model.get_causal_mask()
    final_reduced_mask = trained_model.get_causal_mask_reduced(original_nodes=100, tau_max=4)

    print("\n=== Full Mask Analysis ===")
    full_analysis = analyze_mask_distribution(final_causal_mask, "Full Causal Mask Analysis")

    print("\n=== Reduced Mask Analysis ===")
    reduced_analysis = analyze_mask_distribution(final_reduced_mask, "Reduced Causal Mask Analysis")

    # 保存结果 - 包含时间戳
    print("Saving results...")
    model_path, reduced_mask_path = save_results(
        trained_model,
        loss_history,
        original_nodes=100,  # 原始节点数
        tau_max=3  # 时间滞后
    )

    # 输出最终的软掩码统计信息
    print(f"\nFinal causal soft mask statistics:")
    print(f"  Mean value: {final_causal_mask.mean().item():.4f}")
    print(f"  Std value: {final_causal_mask.std().item():.4f}")
    print(f"  Max value: {final_causal_mask.max().item():.4f}")
    print(f"  Min value: {final_causal_mask.min().item():.4f}")
    print(f"  Values > 0.5: {(final_causal_mask > 0.5).sum().item()}/{final_causal_mask.numel()}")

    print(f"\nFinal reduced causal soft mask statistics:")
    print(f"  Mean value: {final_reduced_mask.mean().item():.4f}")
    print(f"  Std value: {final_reduced_mask.std().item():.4f}")
    print(f"  Max value: {final_reduced_mask.max().item():.4f}")
    print(f"  Min value: {final_reduced_mask.min().item():.4f}")
    print(f"  Values > 0.5: {(final_reduced_mask > 0.5).sum().item()}/{final_reduced_mask.numel()}")

    print("Training completed successfully!")



if __name__ == "__main__":
    main()
