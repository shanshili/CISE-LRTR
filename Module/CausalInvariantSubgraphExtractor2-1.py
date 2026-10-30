"""
核心特点: PCMCI+SCMI融合版本
流程: 扩展并合并PCMCI和SCMI矩阵
功能: 支持先验掩码初始化，基于数据结构预设掩码
改进: PCMCI和SCMI矩阵合并策略，支持滞后关系融合
掩码处理: 基于先验数据的掩码初始化
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

def expand_and_merge_perturbed_matrix(pcmci_3d, scmi_2d, n_nodes, tau_max):
    """
    扩展并合并 PCMCI 和 SCMI 矩阵
    规则：
    1. PCMCI 是 3D [Nodes, Nodes, Lags] (包含 Lag 0, 1, 2...)
    2. SCMI 是 2D [Nodes, Nodes] (仅代表 Lag 0)
    3. 合并：Lag 0 位置 = PCMCI_Lag0 + SCMI (等权相加)
    4. 扩展：
       - (t, t) 对角块位置放置 Lag 0 的合并结果
       - (t, t+k) 偏对角块位置放置 PCMCI Lag k 的结果
    """
    n_time_layers = tau_max + 1
    expanded_size = n_time_layers * n_nodes
    expanded_matrix = np.zeros((expanded_size, expanded_size))

    # 确保输入为 numpy float
    if isinstance(pcmci_3d, torch.Tensor): pcmci_3d = pcmci_3d.cpu().numpy()
    if isinstance(scmi_2d, torch.Tensor): scmi_2d = scmi_2d.cpu().numpy()

    pcmci_3d = np.array(pcmci_3d, dtype=float)
    scmi_2d = np.array(scmi_2d, dtype=float)

    # 检查 PCMCI 维度，如果不足 tau_max，需要填充
    # 假设 pcmci_3d 形状为 [N, N, Lags]
    current_lags = pcmci_3d.shape[2] if len(pcmci_3d.shape) == 3 else 1

    # 如果 PCMCI 是 2D [N, N]，扩展为 [N, N, 1]
    if len(pcmci_3d.shape) == 2:
        pcmci_3d = pcmci_3d[:, :, np.newaxis]

    # 开始填充扩展矩阵
    # 这里的 rows 是 "Cause" (源时间步), cols 是 "Effect" (目标时间步)
    # 通常 A[i, j] 表示 i->j 或 j->i，这里沿用之前的 .T (转置) 逻辑，假设原始数据需要转置适配 GNN

    for t in range(n_time_layers):
        # --- 1. 处理对角块 (Lag 0, 瞬时关系) ---
        # 位置：(t, t)
        row_start = t * n_nodes
        row_end = (t + 1) * n_nodes
        col_start = t * n_nodes
        col_end = (t + 1) * n_nodes

        # 获取 PCMCI Lag 0
        pcmci_lag0 = pcmci_3d[:, :, 0]

        # 合并逻辑：等权相加
        combined_lag0 = pcmci_lag0 + scmi_2d

        # 填充 (注意转置，保持与之前代码一致)
        expanded_matrix[row_start:row_end, col_start:col_end] = combined_lag0.T

        # --- 2. 处理偏对角块 (Lag k, 滞后关系) ---
        # 位置：(t, t+k)
        for lag in range(1, tau_max + 1):
            # 目标时间步 t + lag
            target_t = t + lag
            if target_t < n_time_layers:
                col_start_lag = target_t * n_nodes
                col_end_lag = (target_t + 1) * n_nodes

                # 获取 PCMCI Lag k (如果数据中有)
                if lag < pcmci_3d.shape[2]:
                    pcmci_lagk = pcmci_3d[:, :, lag]
                    expanded_matrix[row_start:row_end, col_start_lag:col_end_lag] = pcmci_lagk.T
                # else: 数据中没有该滞后，保持为0

    return expanded_matrix


def get_prior_mask_from_data(data_list, num_nodes=400):
    """
    从加载的数据列表中提取先验结构。
    策略：计算所有扰动图的并集（Union）。
    如果某条边在任何一个扰动图中存在，初始时我们就认为它可能存在。
    """
    prior_adj = torch.zeros(num_nodes, num_nodes)

    print("Computing prior mask from data...")
    count = 0
    for _, edge_index, _ in data_list:
        # edge_index 是 [2, E]
        # 将这些边在 prior_adj 中置为 1
        prior_adj[edge_index[0], edge_index[1]] = 1.0
        count += 1

    print(f"  Aggregated {count} graphs.")
    print(f"  Prior mask density: {prior_adj.sum() / (num_nodes * num_nodes):.4f} (Edges: {prior_adj.sum().int()})")

    return prior_adj


def process_features_from_file(feature_file_path, target_nodes=400, target_dim=8000, original_nodes=100, tau_max=3):
    """
    加载并处理特征文件
    1. 加载 100 nodes x 10000 features (CSV 或 NPY)
    2. 扩展行: 100 -> 400 (按时间滞后复制)
    3. 裁剪列: 10000 -> 8000
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
        # 目标是 8000 个特征
        print(f"  Adjusting feature dimension from {expanded_features.shape[1]} to {target_dim}...")
        if expanded_features.shape[1] >= target_dim:
            final_features = expanded_features[:, :target_dim]
        else:
            # 如果特征不足，补零
            padding = np.zeros((expanded_features.shape[0], target_dim - expanded_features.shape[1]))
            final_features = np.hstack([expanded_features, padding])

        print(f"  Final processed feature shape: {final_features.shape}")  # 应该是 [400, 8000]
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
    save_path = Path(save_dir) / f"results_{timestamp}"
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
    def __init__(self, cadgnn_model, input_dim, hidden_dim, num_nodes, target_nodes=400, prior_mask=None):

        super(CausalInvariantSubgraphExtractor, self).__init__()
        self.cadgnn = cadgnn_model

        # 冻结CADGNN参数，避免在训练掩码时破坏预训练的编码器
        for param in self.cadgnn.parameters():
            param.requires_grad = False

        self.target_nodes = target_nodes
        # 参数初始化
        # self.causal_mask_theta = nn.Parameter(torch.randn(target_nodes, target_nodes) * 0.1 + 1.0)
        # === 修改：基于先验掩码初始化 ===
        if prior_mask is not None:
            print("Initializing mask using PCMCI+SCMI prior...")
            # 确保 prior_mask 是 Float Tensor
            if isinstance(prior_mask, np.ndarray):
                prior_mask = torch.from_numpy(prior_mask).float()

            # 逻辑：
            # 如果 prior=1 (有边) -> 设置 theta 为 +2.0 (Sigmoid后约 0.88)
            # 如果 prior=0 (无边) -> 设置 theta 为 -2.0 (Sigmoid后约 0.12)
            # 这样模型倾向于相信先验，但如果数据极力反对，梯度仍能将其改变

            # 基础值
            init_logits = (prior_mask * 7.0) - 5.0

            # 添加少量随机噪声 (±0.1)，防止完全死板
            noise = torch.randn_like(init_logits) * 0.05
            self.causal_mask_theta = nn.Parameter(init_logits + noise)
        else:
            # 回退到随机初始化 (均值设为0.5附近)
            print("Initializing mask randomly...")
            self.causal_mask_theta = nn.Parameter(torch.randn(target_nodes, target_nodes) * 0.1)

        self.causal_bilinear = CausalGatedBilinearProjection(hidden_dim)
        self.temperature = 2.0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_causal_mask(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        logits = self.causal_mask_theta
        return torch.sigmoid(logits / temperature)

    def get_causal_mask_reduced(self, original_nodes=100, tau_max=4):
        full_mask = self.get_causal_mask()
        if full_mask.shape[0] != original_nodes * tau_max:
            # 若维度不匹配，尝试自动推断
            tau_max = full_mask.shape[0] // original_nodes

        aggregated_mask = torch.zeros(original_nodes, original_nodes, device=full_mask.device)
        for i in range(original_nodes):
            for j in range(original_nodes):
                values = []
                for t in range(tau_max):
                    time_i = t * original_nodes + i
                    time_j = t * original_nodes + j
                    if time_i < full_mask.shape[0] and time_j < full_mask.shape[1]:
                        values.append(full_mask[time_i, time_j])
                values_tensor = torch.stack(values)
                # 聚合策略
                aggregated_mask[i, j] = torch.mean(values_tensor)
        return aggregated_mask

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
        threshold = 1e-4
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
           a. 扩展特征和邻接矩阵到 400x400 (插零)
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

            # 执行扩展/插零操作 (Expand)
            adj_expanded = self._extend_adjacency_matrix_ordered(adj_dense, current_nodes, target_nodes)

            # 2. 扩展特征矩阵 (Feature Expansion)
            # 如果特征已经是400维（在加载时处理过），则跳过；否则扩展
            if current_nodes != target_nodes:
                feat_expanded = self._extend_feature_matrix_ordered(features, current_nodes, target_nodes)
            else:
                feat_expanded = features

            # 特征归一化
            feat_norm = (feat_expanded - feat_expanded.min()) / (feat_expanded.max() - feat_expanded.min() + 1e-8)

            # 3. 应用掩码 (Masking BEFORE GNN)
            # 核心修改：在这里应用掩码，决定GNN看到什么结构
            # 这样被掩码掉的边不会传播特征，插零的节点保持孤立
            adj_masked = adj_expanded * causal_mask_matrix

            # 4. 转换回稀疏格式输入GNN (Encoding)
            masked_edge_index, masked_edge_weight = self._dense_adj_to_edge_index_weight(adj_masked)

            # 输入GNN获取表示
            # 注意：这里使用的是经过扩展和掩码后的图
            z_c = self.cadgnn.causal_encoder(feat_norm, masked_edge_index, masked_edge_weight)

            # 5. 解码/投影 (Decoding)
            # 使用学到的表示和掩码后的图结构进行投影
            Phi = self.causal_bilinear(z_c, adj_masked)
            causal_features_list.append(Phi)

        causal_features_tensor = torch.stack(causal_features_list, dim=2) if causal_features_list else torch.empty(0, 0,
                                                                                                                   0)
        return causal_features_tensor, causal_mask_matrix

    # 保留原有的扩展辅助函数
    def _extend_adjacency_matrix_ordered(self, adj_matrix, current_nodes, target_nodes):
        if current_nodes == target_nodes: return adj_matrix
        original_nodes = 100
        time_lags = 4
        extended_matrix = torch.zeros(target_nodes, target_nodes, device=adj_matrix.device, dtype=adj_matrix.dtype)
        current_original_nodes = current_nodes // time_lags if time_lags > 0 else current_nodes

        for t in range(time_lags):
            time_offset = t * original_nodes
            start_node = t * current_original_nodes
            end_node = min((t + 1) * current_original_nodes, current_nodes)
            length = end_node - start_node

            if length > 0:
                t_s, t_e = time_offset, time_offset + length
                s_s, s_e = start_node, end_node
                extended_matrix[t_s:t_e, t_s:t_e] = adj_matrix[s_s:s_e, s_s:s_e]  # 对角块

                # 跨时间块处理
                for t2 in range(time_lags):
                    if t2 == t: continue
                    time_offset2 = t2 * original_nodes
                    start_node2 = t2 * current_original_nodes
                    end_node2 = min((t2 + 1) * current_original_nodes, current_nodes)
                    length2 = end_node2 - start_node2
                    if length2 > 0:
                        t_s2, t_e2 = time_offset2, time_offset2 + length2
                        s_s2, s_e2 = start_node2, end_node2
                        extended_matrix[t_s:t_e, t_s2:t_e2] = adj_matrix[s_s:s_e, s_s2:s_e2]
        return extended_matrix

    def _extend_feature_matrix_ordered(self, feature_matrix, current_nodes, target_nodes):
        if current_nodes == target_nodes: return feature_matrix
        original_nodes = 100
        time_lags = 4
        feature_dim = feature_matrix.size(1)
        extended_features = torch.zeros(target_nodes, feature_dim, device=feature_matrix.device,
                                        dtype=feature_matrix.dtype)
        current_original_nodes = current_nodes // time_lags if time_lags > 0 else current_nodes

        for t in range(time_lags):
            time_offset = t * original_nodes
            start_node = t * current_original_nodes
            end_node = min((t + 1) * current_original_nodes, current_nodes)
            length = end_node - start_node
            if length > 0:
                extended_features[time_offset:time_offset + length, :] = feature_matrix[start_node:end_node, :]
        return extended_features


# ==========================================
# 数据加载与主流程
# ==========================================

def load_causal_perturbed_graphs(pcmci_dir, scmi_dir, feature_file_path, num_nodes=100, tau_max=3):
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
        target_dim=8000,
        original_nodes=num_nodes,
        tau_max=tau_max
    )

    causal_perturbed_data_list = []

    print(f"Found {len(pkl_files)} PCMCI perturbation files. Matching with SCMI...")

    for i, (pcmci_path, node_id) in enumerate(pkl_files):
        try:
            # ================= 1. 加载 PCMCI (3D) =================
            with open(pcmci_path, 'rb') as f:
                data = pickle.load(f)

            graph_pcmci = None
            if isinstance(data, dict):
                if 'val_matrix' in data:
                    graph_pcmci = data['val_matrix']
                elif 'val' in data:
                    graph_pcmci = data['val']
                elif 'graph' in data:
                    graph_pcmci = data['graph']
                else:
                    graph_pcmci = list(data.values())[0]
            else:
                graph_pcmci = data

            if hasattr(graph_pcmci, 'values'): graph_pcmci = graph_pcmci.values
            graph_pcmci = np.array(graph_pcmci)

            # PCMCI 符号清洗
            if graph_pcmci.dtype.kind in {'U', 'S', 'O'}:
                numeric_graph = np.zeros(graph_pcmci.shape, dtype=float)
                mask_edge = (graph_pcmci == '-->') | (graph_pcmci == '<--') | (graph_pcmci == 'o-o')
                numeric_graph[mask_edge] = 1.0
                graph_pcmci = numeric_graph

            graph_pcmci = graph_pcmci.astype(float)

            # 调整维度顺序 [Lags, N, N] -> [N, N, Lags] (如果需要)
            # 假设 N 接近 100，Lags 接近 4，根据大小判断
            if graph_pcmci.shape[0] < graph_pcmci.shape[1] and graph_pcmci.shape[0] <= tau_max + 1:
                graph_pcmci = np.transpose(graph_pcmci, (1, 2, 0))

            # 如果是 2D，增加 Lag 维度
            if len(graph_pcmci.shape) == 2:
                graph_pcmci = graph_pcmci[:, :, np.newaxis]

            # --- [关键修复] PCMCI 维度对齐 ---
            # 如果 PCMCI 返回的是 99x99 (删除了扰动节点)，需要补回 100x100
            current_pcmci_nodes = graph_pcmci.shape[0]

            if current_pcmci_nodes == num_nodes - 1:
                # print(f"  Fixing PCMCI dimensions for node {node_id}: {graph_pcmci.shape} -> ({num_nodes}, {num_nodes}, ...)")
                # 在 node_id 处插入全零行 (axis 0)
                graph_pcmci = np.insert(graph_pcmci, node_id, 0, axis=0)
                # 在 node_id 处插入全零列 (axis 1)
                graph_pcmci = np.insert(graph_pcmci, node_id, 0, axis=1)

            # 双重检查维度
            if graph_pcmci.shape[0] != num_nodes:
                print(
                    f"  Warning: PCMCI shape mismatch for node {node_id}. Expected {num_nodes}, got {graph_pcmci.shape[0]}. Skipping.")
                continue

            # ================= 2. 加载 SCMI (2D) =================
            scmi_candidates = [
                scmi_dir / f"scmi_causal_graph_node_{node_id}.npy",
                scmi_dir / f"scmi_causal_graph_{node_id}.npy",
                scmi_dir / f"node_{node_id}.npy"
            ]

            scmi_matrix = None
            for p in scmi_candidates:
                if p.exists():
                    scmi_matrix = np.load(p)
                    break

            if scmi_matrix is None:
                # 找不到文件时，创建全零矩阵
                scmi_matrix = np.zeros((num_nodes, num_nodes))
            else:
                if len(scmi_matrix.shape) > 2:
                    scmi_matrix = scmi_matrix[:, :, 0]

            # 确保 SCMI 尺寸正确，如果不正确则重置为零（防止报错）
            if scmi_matrix.shape[0] != num_nodes:
                # 尝试简单的补零或截断（视情况而定），这里简化为全零处理以防报错
                print(f"  Warning: SCMI shape mismatch {scmi_matrix.shape}. Using zeros.")
                scmi_matrix = np.zeros((num_nodes, num_nodes))

            # ================= 3. 合并与扩展 =================

            # 现在两个矩阵应该都是 100x100 了，可以安全合并
            expanded_adj = expand_and_merge_perturbed_matrix(
                graph_pcmci,
                scmi_matrix,
                num_nodes,
                tau_max
            )

            # 转为 Tensor
            adj_tensor = torch.FloatTensor(expanded_adj)

            # 稀疏转换
            threshold = 1e-6
            mask = torch.abs(adj_tensor) > threshold
            edges = torch.nonzero(mask).t()
            edge_weights = adj_tensor[mask]

            if edges.size(1) == 0:
                n = adj_tensor.size(0)
                edges = torch.arange(n).unsqueeze(0).repeat(2, 1)
                edge_weights = torch.ones(n)

            # 特征分配 (直接使用 base_features，因为扩展后的图已经是完整维度，其中扰动节点的连接已被处理)
            current_features = base_features.clone()

            causal_perturbed_data_list.append((current_features, edges, edge_weights))

            # 调试输出
            if i == 0:
                print(
                    f"  [Debug] Node {node_id}: PCMCI {graph_pcmci.shape} + SCMI {scmi_matrix.shape} -> Expanded {expanded_adj.shape}")

            if i >= 50: break

        except Exception as e:
            print(f"Error processing node {node_id}: {e}")
            import traceback
            # traceback.print_exc()
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

    # 稍微调大一点 Mask 的学习率，让它能动起来
    optimizer = Adam([
        {'params': model.causal_mask_theta, 'lr': lr * 5.0},  # Mask 学习率加大
        {'params': model.causal_bilinear.parameters(), 'lr': lr},
    ], weight_decay=1e-5)

    data_list_cpu = causal_perturbed_data_list

    loss_history = {'total_loss': [], 'lrtr_loss': [], 'sparsity_loss': [], 'binary_loss': []}
    num_samples = len(data_list_cpu)

    print(f"Start training... Batch Size: {batch_size}, LRTR Scaler: 1e5, Sparsity Delay: 20 epochs")

    for epoch in range(num_epochs):
        # 温度退火：从 2.0 慢慢降到 0.5
        # 前期温度高，梯度流更顺畅
        temp = 2.0 * (0.98 ** epoch)
        if temp < 0.1: temp = 0.1
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

            # --- Loss Calculation (Rebalanced) ---

            # 1. LRTR (因果不变性)
            # 原始值通常在 1e-6 到 1e-8 之间
            # 我们乘以 100,000 (1e5) 让它变成 0.1 ~ 1.0 的量级
            if causal_features.size(2) > 1:
                phi_list = [causal_features[:, :, j] for j in range(causal_features.size(2))]
                raw_lrtr = compute_lrtr_loss(phi_list)
            else:
                raw_lrtr = torch.tensor(0.0, device=device, requires_grad=True)

            weighted_lrtr = raw_lrtr * 100000.0

            # 2. Sparsity (稀疏惩罚)
            # 这里的 mask 已经在 [0,1] 之间
            # 400*400*0.6 = 96000 (L1 Norm)
            # 如果系数是 1e-5，Loss = 0.96。这比较合理。
            sparsity_coeff = 2e-5

            # === 关键策略：Sparsity Warmup ===
            # 前 20 个 Epoch 不加稀疏惩罚！
            # 让模型先学会如何利用现有的边来保持不变性，然后再开始剪枝。
            if epoch < 20:
                sparsity_loss = torch.tensor(0.0, device=device)
            else:
                # 慢慢增加惩罚力度
                ramp_up = min(1.0, (epoch - 20) / 10.0)
                sparsity_loss = sparsity_coeff * torch.norm(mask, p=1) * ramp_up

            # 3. Binary Loss (二值化)
            binary_loss = 0.05 * torch.mean(mask * (1 - mask))

            # Total Loss
            total_loss = weighted_lrtr # + sparsity_loss + binary_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Record
            epoch_stats['total'] += total_loss.item()
            epoch_stats['lrtr'] += raw_lrtr.item()  # 记录原始值
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

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Total {avg_total:.10f} | LRTR(raw) {avg_lrtr:.2e} | Sparsity {avg_spar:.10f}")

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
    cadgnn_model_path = "./CADGNN_results/N100_T10000_hid128_lay5_ladv0.1_lw0.02_20250902_203304_model.pth"  # 替换实际路径
    pcmci_dir = project_root / "dataset" / "Perturbed_Causal_Graphs_N100_T10000_Hubs15"
    scmi_dir = project_root / "dataset" / "Perturbed_SCMI_Causal_Graphs_N100_T10000_Hubs15/scmi_perturbations"
    # **指定特征文件路径**
    feature_file_path = project_root / "dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final" / "Env_A_timeseries.csv"  # 替换为你的特征文件路径

    # 1. 加载 CADGNN
    print("Loading CADGNN...")
    # 模拟加载，实际需确保路径正确
    checkpoint = torch.load(cadgnn_model_path, map_location=device) if os.path.exists(cadgnn_model_path) else None
    if checkpoint:
        config = checkpoint.get('model_config', {'input_dim': 8000, 'hidden_dim': 128})
        cadgnn = CADGNN(config['input_dim'], config['hidden_dim'], 5, 0.1, 0.02).to(device)
        cadgnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: Model not found, creating new for demo.")
        cadgnn = CADGNN(8000, 128, 5, 0.1, 0.02).to(device)

    # 2. 加载数据 (传入新的 scmi_dir)
    print("Loading and Merging Data (PCMCI + SCMI)...")
    data_list = load_causal_perturbed_graphs(
        pcmci_dir=pcmci_dir,
        scmi_dir=scmi_dir,           # 传入 SCMI 路径
        feature_file_path=str(feature_file_path),
        num_nodes=100,
        tau_max=3
    )

    if not data_list:
        print("No data loaded.")
        return

    # === 新增：获取先验掩码 ===
    prior_mask = get_prior_mask_from_data(data_list, num_nodes=400)
    # 移至 GPU (如果使用 cuda)
    prior_mask = prior_mask.to(device)


    # 3. 初始化并训练提取器
    print("Initializing Extractor...")
    extractor = CausalInvariantSubgraphExtractor(
        cadgnn,
        input_dim=8000,
        hidden_dim=128,
        num_nodes=400,
        target_nodes=400,
        prior_mask = prior_mask
    )

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
