"""
基于对比学习与低秩张量残差的因果不变子图提取
调用CADGNN已训练好的模型，使用扰动图进行因果不变子图提取

整体流程
1. 数据加载阶段
load_causal_perturbed_graphs 函数加载扰动图数据
从 ./dataset/Perturbed_Causal_Graphs 目录读取 pcmci_results_node_* 子目录下的 .pkl 文件
每个扰动图代表删除一个节点后的因果关系数据
2. 模型初始化阶段
load_cadgnn_model 加载预训练的CADGNN模型
CausalInvariantSubgraphExtractor 初始化，创建400×400的软掩码矩阵参数
3. 数据扩展处理阶段
expand_perturbed_matrix 将原始99×99矩阵扩展为396×396（考虑时间滞后）
_extend_adjacency_matrix_ordered 和 _extend_feature_matrix_ordered 将数据扩展到400个节点
4. 训练阶段
train_causal_invariant_subgraph_extractor 函数执行训练
使用低秩张量残差损失（LRTR Loss）保持因果不变性
优化**causal_mask_theta** 参数学习因果不变子图掩码
5. 掩码处理阶段
get_causal_mask 获取400×400的扩展掩码
get_causal_mask_reduced 将400×400掩码聚合为99×99的最终因果图
6. 结果保存阶段
save_results 保存模型、完整掩码、降维掩码和可视化结果
生成热图可视化最终的因果不变子图

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

sys.path.append(str(Path(__file__).parent.parent))

# 从CADGNN导入相关组件
from Module.CADGNN import CADGNN, load_data

from utils.get_data_file import get_project_root, find_causal_adj_file, find_corr_adj_file, find_features_file


def expand_perturbed_matrix(perturbed_matrix, n_nodes, tau_max):
    """
    扩展扰动图矩阵，考虑时间滞后

    Args:
        perturbed_matrix: 扰动图因果矩阵 [n_nodes, n_nodes]
        n_nodes: 原始节点数
        tau_max: 最大时间滞后

    Returns:
        expanded_matrix: 扩展后的矩阵 [(tau_max+1)*n_nodes, (tau_max+1)*n_nodes]
    """
    n_time_layers = tau_max + 1
    expanded_size = n_time_layers * n_nodes
    expanded_matrix = np.zeros((expanded_size, expanded_size))

    # 确保输入矩阵是数值类型
    if isinstance(perturbed_matrix, np.ndarray):
        # 转换为浮点数类型，处理非数值
        try:
            perturbed_matrix = np.array(perturbed_matrix, dtype=float)
        except ValueError:
            # 如果有非数值，尝试清理数据
            print(f"Warning: Converting non-numeric values in matrix")
            # 将非数值替换为0
            perturbed_matrix_clean = np.zeros_like(perturbed_matrix, dtype=float)
            for i in range(perturbed_matrix.shape[0]):
                for j in range(perturbed_matrix.shape[1]):
                    try:
                        val = perturbed_matrix[i, j]
                        if isinstance(val, (int, float)):
                            perturbed_matrix_clean[i, j] = float(val)
                        elif isinstance(val, str) and val.strip():
                            # 尝试转换字符串为浮点数
                            perturbed_matrix_clean[i, j] = float(val)
                        else:
                            # 非数值设为0
                            perturbed_matrix_clean[i, j] = 0.0
                    except (ValueError, TypeError):
                        perturbed_matrix_clean[i, j] = 0.0
            perturbed_matrix = perturbed_matrix_clean
    else:
        perturbed_matrix = np.array(perturbed_matrix, dtype=float)

    # 在对角块填充扰动矩阵
    for t in range(n_time_layers):
        row_start = t * n_nodes
        row_end = (t + 1) * n_nodes
        col_start = t * n_nodes
        col_end = (t + 1) * n_nodes
        expanded_matrix[row_start:row_end, col_start:col_end] = perturbed_matrix.T

    return expanded_matrix


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


class CausalGatedBilinearProjection(nn.Module):
    """
    因果门控双线性投影层 (Causal-Gated Bilinear Projection)
    注意：在实际实现中，我们复用CADGNN的CausalDecoder功能
    """

    def __init__(self, hidden_dim):
        super(CausalGatedBilinearProjection, self).__init__()
        self.hidden_dim = hidden_dim
        # 使用CADGNN的theta_weight作为投影矩阵W_p
        self.W_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, H, A):
        """
        Args:
            H: 节点表示矩阵 [N, d]
            A: 因果邻接矩阵 [N, N]
        Returns:
            Phi: 投影后的因果矩阵 [N, N]
        """
        # 双线性映射 H * W_p * H^T
        H_proj = self.W_p(H)  # [N, d]
        Phi = torch.matmul(H_proj, H.t())  # [N, N]

        # 因果过滤 A ⊙ (H * W_p * H^T)
        Phi_filtered = A * Phi  # [N, N]

        return Phi_filtered


class CausalInvariantSubgraphExtractor(nn.Module):
    """
    因果不变子图提取器 - 支持因果扰动图和相关性扰动图
    """

    def __init__(self, cadgnn_model, input_dim, hidden_dim, num_nodes, target_nodes=400):
        super(CausalInvariantSubgraphExtractor, self).__init__()

        # 保留CADGNN模型
        self.cadgnn = cadgnn_model

        # 软掩码矩阵参数 (扩展到目标节点数，考虑时间维度)
        self.target_nodes = target_nodes  # 400 = 100 * 4 (时间滞后)
        # self.causal_mask_theta = nn.Parameter(torch.randn(target_nodes, target_nodes) * 0.1)
        # 使用正态分布初始化，均值0，标准差较小
        self.causal_mask_theta = nn.Parameter(torch.randn(target_nodes, target_nodes) * 0.3)

        # 因果门控双线性投影层
        self.causal_bilinear = CausalGatedBilinearProjection(hidden_dim)

        # 节点数量
        self.num_nodes = num_nodes

        self.temperature = 2.0  # 添加温度参数

    # def get_causal_mask(self):
    #     """获取因果软掩码矩阵 M_causal = sigmoid(Theta_causal)"""
    #     return torch.sigmoid(self.causal_mask_theta)

    # def get_causal_mask(self, temperature=None):
    #     """使用Gumbel-Softmax获取近似二值掩码"""
    #     if temperature is None:
    #         temperature = self.temperature
    #
    #     logits = self.causal_mask_theta
    #     return F.gumbel_softmax(
    #         torch.stack([logits, -logits], dim=-1),
    #         tau=temperature,  # 使用tau而不是temperature
    #         dim=-1
    #     )[..., 0]
    def set_temperature(self, temperature):
        """设置Gumbel-Softmax温度参数"""
        self.temperature = temperature

    def get_causal_mask(self, temperature=None):
        """使用Sigmoid获取连续分布掩码"""
        if temperature is None:
            temperature = self.temperature

        logits = self.causal_mask_theta
        # 使用温度控制的sigmoid，温度高则更接近0.5
        return torch.sigmoid(logits / temperature)


    def get_causal_mask_reduced(self, original_nodes=100, tau_max=4):
        """
        获取降维后的因果软掩码矩阵 [n×n]
        从400×400降维到100×100
        按时间滞后维度聚合（取对应时间滞后位置的均值）
        """
        full_mask = self.get_causal_mask()  # [400×400]

        # 检查掩码的实际形状并调整参数
        # if full_mask.shape[0] == 396:  # 99×4
        #     original_nodes = 99
        #     tau_max = 4
        # elif full_mask.shape[0] == 400:  # 100×4
        if full_mask.shape[0] == 400:
            original_nodes = 100
            tau_max = 4
        else:
            raise ValueError(f"Unexpected mask shape: {full_mask.shape}")

        # 创建聚合后的掩码
        aggregated_mask = torch.zeros(original_nodes, original_nodes, device=full_mask.device, dtype=full_mask.dtype)

        # 按时间滞后维度聚合 - 对应位置求均值
        for i in range(original_nodes):
            for j in range(original_nodes):
                # 获取对应时间滞后位置的值
                values = []
                for t in range(tau_max):
                    time_i = t * original_nodes + i
                    time_j = t * original_nodes + j
                    if time_i < full_mask.shape[0] and time_j < full_mask.shape[1]:
                        values.append(full_mask[time_i, time_j])
                # 计算均值
                # aggregated_mask[i, j] = torch.stack(values).mean()
                values_tensor = torch.stack(values)
                aggregated_mask[i, j] = torch.std(values_tensor) + torch.mean(values_tensor)

        return aggregated_mask

    def get_temporal_aggregated_mask(self, original_nodes=100, tau_max=4):
        """
        获取时间维度聚合的因果掩码矩阵 [n×n]
        将扩展的时间维度聚合回原始维度
        """
        full_mask = self.get_causal_mask()  # [400×400]

        # 将扩展矩阵重塑为 [tau+1, n, tau+1, n]
        reshaped_mask = full_mask.view(tau_max, original_nodes, tau_max, original_nodes)

        # 沿时间维度聚合（取平均值）
        aggregated_mask = reshaped_mask.mean(dim=0).mean(dim=1)  # [n, n] -> [100, 100]

        return aggregated_mask

    def get_causal_representations(self, features, edge_index, edge_weight):
        """使用CADGNN获取因果表示"""
        # 特征标准化 - 参考MGC-RM3
        features_normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)

        with torch.no_grad():
            z_c = self.cadgnn.causal_encoder(features_normalized, edge_index, edge_weight)
        return z_c

    def forward(self, causal_perturbed_data_list, target_nodes=400):
        """
        前向传播 - 处理因果扰动图和相关性扰动图，扩展到统一节点数（考虑时间维度）

        Args:
            causal_perturbed_data_list: 因果扰动图数据列表 [(features, edge_index, edge_weight), ...]
            target_nodes: 目标节点数（400，即100*4，考虑时间维度）

        Returns:
            causal_features_tensor: 因果扰动图特征矩阵张量 [target_nodes, target_nodes, num_causal_perturbed]
            causal_mask_matrix: 因果软掩码矩阵 [target_nodes, target_nodes]
        """
        # 获取软掩码矩阵（扩展到目标节点数）
        causal_mask_matrix = self.get_causal_mask()  # [target_nodes, target_nodes]

        # 处理因果扰动图
        causal_features_list = []
        for i, (perturbed_features, perturbed_edge_index, perturbed_edge_weight) in enumerate(
                causal_perturbed_data_list):
            # 获取当前扰动图的节点数
            current_nodes = perturbed_features.size(0)

            # 获取因果扰动图的因果表示
            H_perturbed = self.get_causal_representations(
                perturbed_features, perturbed_edge_index, perturbed_edge_weight
            )

            # 获取扰动图的邻接矩阵
            A_perturbed = self._edge_index_to_adj_matrix(
                perturbed_edge_index, perturbed_edge_weight, H_perturbed.size(0)
            )

            # 检查当前节点数是否与目标节点数一致，如果不一致则进行扩展
            if current_nodes != target_nodes:
                # 扩展A_perturbed到目标节点数（按照原始节点顺序在对应位置插入零）
                A_perturbed_extended = self._extend_adjacency_matrix_ordered(A_perturbed, current_nodes, target_nodes)

                # 扩展H_perturbed到目标节点数（按照原始节点顺序在对应位置插入零向量）
                H_perturbed_extended = self._extend_feature_matrix_ordered(H_perturbed, current_nodes, target_nodes)
            else:
                # 如果节点数已经匹配，直接使用
                A_perturbed_extended = A_perturbed
                H_perturbed_extended = H_perturbed

            # 获取对应位置的掩码子矩阵
            mask_sub = causal_mask_matrix[:target_nodes, :target_nodes]

            # 应用因果软掩码得到扰动子图
            A_sub_perturbed = A_perturbed_extended * mask_sub

            # 应用因果门控双线性投影
            Phi = self.causal_bilinear(H_perturbed_extended, A_sub_perturbed)
            causal_features_list.append(Phi)

        # 将特征列表转换为张量
        causal_features_tensor = torch.stack(causal_features_list, dim=2) if causal_features_list else torch.empty(0, 0,
                                                                                                                   0)

        return causal_features_tensor, causal_mask_matrix

    def _extend_adjacency_matrix_ordered(self, adj_matrix, current_nodes, target_nodes):
        """
        按照时间滞后结构扩展邻接矩阵
        在指定的时间滞后位置插入零行/零列

        Args:
            adj_matrix: 原始邻接矩阵 [current_nodes, current_nodes]
            current_nodes: 当前节点数
            target_nodes: 目标节点数 (如400)

        Returns:
            extended_matrix: 扩展后的邻接矩阵 [target_nodes, target_nodes]
        """
        if current_nodes == target_nodes:
            return adj_matrix

        # 计算原始节点数和时间滞后数
        original_nodes = 100  # 原始节点数
        time_lags = 4  # 时间滞后数 (tau_max + 1)

        # 创建目标大小的扩展矩阵
        extended_matrix = torch.zeros(target_nodes, target_nodes,
                                      device=adj_matrix.device, dtype=adj_matrix.dtype)

        # 计算当前包含的原始节点数
        current_original_nodes = current_nodes // time_lags if time_lags > 0 else current_nodes

        # 按时间滞后维度处理，将原始邻接矩阵的块复制到扩展矩阵的对应位置
        for t in range(time_lags):
            # 计算时间滞后t对应的时间索引
            time_offset = t * original_nodes

            # 计算当前时间滞后中实际存在的节点数
            start_node = t * current_original_nodes
            end_node = min((t + 1) * current_original_nodes, current_nodes)
            actual_nodes_in_time_lag = end_node - start_node

            # 如果当前时间滞后有节点数据，则复制
            if actual_nodes_in_time_lag > 0:
                # 计算在扩展矩阵中的位置
                target_start = time_offset
                target_end = time_offset + actual_nodes_in_time_lag

                # 从原始邻接矩阵中提取当前时间滞后块
                source_start = start_node
                source_end = end_node

                # 复制对应块
                extended_matrix[target_start:target_end, target_start:target_end] = \
                    adj_matrix[source_start:source_end, source_start:source_end]

                # 复制跨时间滞后块（如果有跨时间滞后的连接）
                for t2 in range(time_lags):
                    if t2 == t:
                        continue  # 跳过自身块，已在上面处理
                    time_offset2 = t2 * original_nodes

                    # 计算第二个时间滞后的节点范围
                    start_node2 = t2 * current_original_nodes
                    end_node2 = min((t2 + 1) * current_original_nodes, current_nodes)
                    actual_nodes_in_time_lag2 = end_node2 - start_node2

                    if actual_nodes_in_time_lag2 > 0:
                        target_start2 = time_offset2
                        target_end2 = time_offset2 + actual_nodes_in_time_lag2

                        # 复制跨时间滞后的连接
                        extended_matrix[target_start:target_end, target_start2:target_end2] = \
                            adj_matrix[source_start:source_end, start_node2:end_node2]

        return extended_matrix

    def _extend_feature_matrix_ordered(self, feature_matrix, current_nodes, target_nodes):
        """
        按照时间滞后结构扩展特征矩阵到目标节点数
        在指定的时间滞后位置插入零向量

        Args:
            feature_matrix: 原始特征矩阵 [current_nodes, feature_dim]
            current_nodes: 当前节点数
            target_nodes: 目标节点数 (如400)

        Returns:
            extended_features: 扩展后的特征矩阵 [target_nodes, feature_dim]
        """
        if current_nodes == target_nodes:
            return feature_matrix

        # 计算原始节点数和时间滞后数
        original_nodes = 100  # 原始节点数
        time_lags = 4  # 时间滞后数 (tau_max + 1)

        feature_dim = feature_matrix.size(1)
        extended_features = torch.zeros(target_nodes, feature_dim, device=feature_matrix.device,
                                        dtype=feature_matrix.dtype)

        # 计算当前包含的原始节点数
        current_original_nodes = current_nodes // time_lags if time_lags > 0 else current_nodes

        # 按时间滞后维度处理，将原始特征矩阵的行复制到扩展矩阵的对应位置
        for t in range(time_lags):
            # 计算时间滞后t对应的时间索引
            time_offset = t * original_nodes

            # 计算当前时间滞后中实际存在的节点数
            start_node = t * current_original_nodes
            end_node = min((t + 1) * current_original_nodes, current_nodes)
            actual_nodes_in_time_lag = end_node - start_node

            # 如果当前时间滞后有节点数据，则复制
            if actual_nodes_in_time_lag > 0:
                # 计算在扩展矩阵中的位置
                target_start = time_offset
                target_end = time_offset + actual_nodes_in_time_lag

                # 从原始特征矩阵中提取当前时间滞后块
                source_start = start_node
                source_end = end_node

                # 复制对应块
                extended_features[target_start:target_end, :] = \
                    feature_matrix[source_start:source_end, :]

        return extended_features

    def _edge_index_to_adj_matrix(self, edge_index, edge_weight, num_nodes):
        """将边索引和边权重转换为邻接矩阵"""
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj_matrix[edge_index[0], edge_index[1]] = edge_weight
        return adj_matrix


def load_causal_perturbed_graphs(causal_perturbed_dir, num_nodes=100, tau_max=3, time_steps=10000):
    """
    加载因果扰动图数据并进行维度扩展
    """
    from utils.feature_expansion import expand_features_for_causal_graph
    import pickle

    causal_perturbed_dir = Path(causal_perturbed_dir)
    print(f"Looking for perturbed graph files in: {causal_perturbed_dir}")

    # 检查目录是否存在
    if not causal_perturbed_dir.exists():
        print(f"Warning: Directory {causal_perturbed_dir} does not exist!")
        return []

    # 修改文件匹配模式：查找 pcmci_results_node_* 目录下的 .pkl 文件
    pkl_files = []
    for sub_dir in causal_perturbed_dir.glob('pcmci_results_node_*'):
        if sub_dir.is_dir():
            pkl_files.extend(list(sub_dir.glob('*.pkl')))

    print(f"Found {len(pkl_files)} .pkl files in subdirectories")

    causal_perturbed_data_list = []

    for i, file_path in enumerate(pkl_files):
        print(f"Processing file {i + 1}/{len(pkl_files)}: {file_path.name}")
        try:
            # 加载 .pkl 文件
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # 假设 data 包含因果邻接矩阵，根据实际数据结构调整
            if isinstance(data, dict):
                if 'graph' in data:
                    graph = data['graph']
                elif 'adjacency' in data:
                    graph = data['adjacency']
                elif 'causal_matrix' in data:
                    graph = data['causal_matrix']
                elif 'selected_links' in data:
                    # 如果PCMCI结果包含selected_links，需要转换为邻接矩阵
                    selected_links = data['selected_links']
                    # 创建邻接矩阵
                    graph = np.zeros((num_nodes, num_nodes))
                    if hasattr(selected_links, 'items'):
                        for (i, j), value in selected_links.items():
                            if i < num_nodes and j < num_nodes:
                                graph[i, j] = value
                    elif isinstance(selected_links, (list, tuple)):
                        for link in selected_links:
                            if len(link) >= 3:
                                i, j, value = link[0], link[1], link[2]
                                if i < num_nodes and j < num_nodes:
                                    graph[i, j] = value
                else:
                    graph = list(data.values())[0] if data else np.zeros((num_nodes, num_nodes))
            else:
                graph = data

            # 确保是 numpy 数组格式
            if torch.is_tensor(graph):
                graph = graph.numpy()

            print(f"  Raw graph shape: {graph.shape}")

            # 处理不同维度的数据格式
            if len(graph.shape) == 3:
                if graph.shape[2] == tau_max + 1:  # 4 = tau_max + 1
                    graph = graph[:, :, 0]  # 取第一个时间滞后
                    print(f"  Extracted adjacency matrix shape: {graph.shape}")
                elif graph.shape[0] == graph.shape[1]:
                    graph = np.mean(graph, axis=2)  # 沿时间维度平均
                    print(f"  Averaged adjacency matrix shape: {graph.shape}")
            elif len(graph.shape) == 2:
                print(f"  2D adjacency matrix shape: {graph.shape}")

            # 获取实际节点数
            actual_nodes = graph.shape[0]
            print(f"  Actual nodes: {actual_nodes}, Expected: {num_nodes}")

            # 检查矩阵维度并进行扩展
            if actual_nodes != num_nodes:
                print(f"  Adjusting num_nodes from {num_nodes} to {actual_nodes}")
                num_nodes = actual_nodes

            expected_size = num_nodes * (tau_max + 1)
            if graph.shape[0] != expected_size:
                print(f"  Graph shape {graph.shape} does not match expected size {expected_size}")
                if graph.shape[0] == num_nodes:
                    graph = expand_perturbed_matrix(graph, num_nodes, tau_max)
                    print(f"  Expanded graph shape: {graph.shape}")
                else:
                    print(f"  Skipping file due to dimension mismatch")
                    continue

            print(f"  Final graph shape: {graph.shape}")

            # 数据类型检查和清理
            if graph.dtype == object:
                print("  Converting object matrix to float...")
                graph_float = np.zeros_like(graph, dtype=float)
                for i in range(graph.shape[0]):
                    for j in range(graph.shape[1]):
                        try:
                            val = graph[i, j]
                            if isinstance(val, (int, float, np.number)):
                                graph_float[i, j] = float(val)
                            elif isinstance(val, str) and val.strip():
                                graph_float[i, j] = float(val)
                            else:
                                graph_float[i, j] = 0.0
                        except (ValueError, TypeError):
                            graph_float[i, j] = 0.0
                graph = graph_float

            # 确保是数值类型
            graph = np.array(graph, dtype=float)

            # 数值稳定性处理 - 参考CADGNN
            graph = np.clip(graph, -10, 10)  # 限制数值范围

            # 创建边权重矩阵
            val_matrix = graph

            # 转换为PyTorch张量
            graph_tensor = torch.FloatTensor(graph)
            val_tensor = torch.FloatTensor(val_matrix)

            # 创建边索引和边权重 - 参考CADGNN的策略
            # 只选择非零边，避免空图
            threshold = 1e-6  # 设置阈值过滤弱连接
            mask = torch.abs(graph_tensor) > threshold
            edges = torch.nonzero(mask).t().contiguous()
            edge_weights = val_tensor[mask]

            # 如果没有找到边，创建自环以确保图连通
            if edges.size(1) == 0:
                print(f"  Warning: No edges found with threshold {threshold}, creating self-loops")
                n_nodes = graph_tensor.size(0)
                self_loop_edges = torch.arange(n_nodes).unsqueeze(0).repeat(2, 1)
                self_loop_weights = torch.ones(n_nodes)
                edges = self_loop_edges
                edge_weights = self_loop_weights

            print(f"  Final edges: {edges.shape[1]} edges found")

            # 使用特征扩展函数处理扰动图特征
            original_features = torch.randn(num_nodes, time_steps)

            # 扩展特征维度
            expanded_features = expand_features_for_causal_graph(
                original_features.numpy(),
                causal_nodes=(tau_max + 1) * num_nodes,
                original_nodes=num_nodes,
                tau_max=tau_max
            )

            # 调整特征维度以匹配CADGNN的输入要求 (8000维)
            features = torch.FloatTensor(expanded_features)

            # 如果特征维度不匹配CADGNN的要求，需要调整
            cadgnn_input_dim = 8000
            current_feature_dim = features.shape[1]

            if current_feature_dim != cadgnn_input_dim:
                print(f"  Adjusting feature dimension from {current_feature_dim} to {cadgnn_input_dim}")
                if current_feature_dim > cadgnn_input_dim:
                    features = features[:, :cadgnn_input_dim]
                else:
                    padded_features = torch.zeros(features.shape[0], cadgnn_input_dim)
                    padded_features[:, :current_feature_dim] = features
                    features = padded_features

            print(f"  Features shape: {features.shape}, Edges shape: {edges.shape}")

            causal_perturbed_data_list.append((features, edges, edge_weights))

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Successfully loaded {len(causal_perturbed_data_list)} perturbed graphs")
    return causal_perturbed_data_list


def load_cadgnn_model(model_path, device):
    """
    加载训练好的CADGNN模型
    """
    checkpoint = torch.load(model_path, map_location=device)

    # 检查模型配置
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # 如果没有model_config，使用默认配置
        model_config = {
            'input_dim': 8000,
            'hidden_dim': 128,
            'num_layers': 5,
            'lambda_adv': 0.1,
            'lambda_weight': 0.02
        }

    # 创建CADGNN模型
    model = CADGNN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        lambda_adv=model_config['lambda_adv'],
        lambda_weight=model_config['lambda_weight']
    )

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, model_config


def compute_lrtr_loss(Phi_list):
    """
    计算低秩张量残差损失 (LRTR Loss)

    Args:
        Phi_list: 所有扰动图的因果投影矩阵列表

    Returns:
        lrtr_loss: 低秩张量残差损失
    """
    if len(Phi_list) == 0:
        return torch.tensor(0.0, requires_grad=True)

    # 将所有Phi堆叠成张量 [N, N, M]，其中M是扰动图数量
    Phi_tensor = torch.stack(Phi_list, dim=2)  # [N, N, M]

    # 计算平均因果模式 (秩-1基底)
    Phi_mean = torch.mean(Phi_tensor, dim=2, keepdim=True)  # [N, N, 1]
    Phi_mean = Phi_mean.expand_as(Phi_tensor)  # [N, N, M]

    # 计算LRTR损失
    lrtr_loss = torch.mean((Phi_tensor - Phi_mean) ** 2)

    return lrtr_loss


def compute_contrastive_loss(H_original, H_perturbed_list, temperature=0.1):
    """
    计算对比学习损失

    Args:
        H_original: 原始图的因果表示 [N, d]
        H_perturbed_list: 扰动图的因果表示列表
        temperature: 对比学习温度参数

    Returns:
        contrastive_loss: 对比学习损失
    """
    if len(H_perturbed_list) == 0:
        return torch.tensor(0.0, requires_grad=True)

    total_loss = 0.0

    for H_perturbed in H_perturbed_list:
        # 计算相似度
        sim_original = F.cosine_similarity(H_original.unsqueeze(1), H_original.unsqueeze(0), dim=-1) / temperature
        sim_perturbed = F.cosine_similarity(H_original.unsqueeze(1), H_perturbed.unsqueeze(0), dim=-1) / temperature

        # 对角线元素应该是正样本
        pos_sim = torch.diag(sim_perturbed)

        # 计算对比损失
        logits = torch.cat([pos_sim.unsqueeze(1), sim_original], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        contrastive_loss = F.cross_entropy(logits, labels)
        total_loss += contrastive_loss

    return total_loss / len(H_perturbed_list) if len(H_perturbed_list) > 0 else torch.tensor(0.0, requires_grad=True)


def train_causal_invariant_subgraph_extractor(
        model,
        causal_perturbed_data_list,
        num_epochs=100,
        lr=0.001,
        device='cuda'
):
    """
    训练因果不变子图提取器 - 支持因果扰动图和相关性扰动图
    """
    model = model.to(device)
    optimizer = Adam([
        {'params': model.causal_mask_theta, 'lr': lr},  # 学习因果软掩码
        {'params': model.causal_bilinear.parameters(), 'lr': lr * 0.1},  # 因果投影层
    ], weight_decay=1e-5)


    # 移动扰动数据到设备
    causal_perturbed_data_on_device = []
    for features, edge_index, edge_weight in causal_perturbed_data_list:
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        causal_perturbed_data_on_device.append((features, edge_index, edge_weight))

    loss_history = {'total_loss': [], 'causal_lrtr_loss': [], 'sparsity_loss': [], 'binary_reg_loss': []}

    for epoch in range(num_epochs):
        # 在训练中动态调整温度
        # current_temp = max(0.1, 1.0 * (0.95 ** epoch))  # 温度退火
        # model.set_temperature(current_temp)  # 设置当前温度

        # 温度策略：保持适中温度，避免过度二值化
        current_temp = 0.3 + 0.2 * max(0.1, (0.99 ** epoch))  # 保持在1.0-3.0之间
        model.set_temperature(current_temp)

        model.train()
        optimizer.zero_grad()

        # 前向传播
        causal_features_tensor, causal_mask_matrix = model(
            causal_perturbed_data_on_device,
        )

        # 计算因果扰动图的LRTR损失
        causal_lrtr_loss = compute_lrtr_loss([causal_features_tensor[:, :, i] for i in range(causal_features_tensor.size(2))]) if causal_features_tensor.size(2) > 0 else torch.tensor(0.0, device=device)

        # 计算对比学习损失（可选，这里简化处理）
        contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 总损失
        # total_loss = causal_lrtr_loss + 0.1 * contrastive_loss

        # def entropy_reg_loss(mask_matrix, beta=0.01):
        #     """最大化掩码的熵，促进二值化"""
        #     p = torch.clamp(mask_matrix, 1e-6, 1 - 1e-6)
        #     entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))  # 负号改变优化方向
        #     return -beta * torch.mean(entropy)  # 再次取负以最小化负熵（即最大化熵）

        # def gaussian_reg_loss(mask_matrix, alpha=0.01):
        #     """促进掩码值接近0.5的正则化项"""
        #     # 计算偏离0.5的程度，使用二次损失
        #     deviation = torch.abs(mask_matrix - 0.5)
        #     return alpha * torch.mean(deviation ** 2)
        def binary_regularization_loss(mask_matrix, alpha=0.01):
            """
            促进掩码二值化的正则化损失
            使掩码值趋向于0或1，增强对比度
            """
            # 计算接近0.5的惩罚项，鼓励值远离0.5（趋向0或1）
            deviation_from_binary = mask_matrix * (1 - mask_matrix)  # 这个值在0.5时最大，在0和1时为0
            return alpha * torch.mean(deviation_from_binary)
        sparsity_loss = 0.00001 * torch.norm(causal_mask_matrix, p=1)
        # sparsity_loss = entropy_reg_loss(causal_mask_matrix)
        # center_loss = 0.1 * torch.mean((causal_mask_matrix - 0.5) ** 2)  # 控制中心
        binary_reg_loss = binary_regularization_loss(causal_mask_matrix, alpha=0.01)
        total_loss = causal_lrtr_loss + sparsity_loss + binary_reg_loss

        # 反向传播
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录损失
        loss_history['total_loss'].append(total_loss.item())
        loss_history['causal_lrtr_loss'].append(causal_lrtr_loss.item())
        loss_history['sparsity_loss'].append(sparsity_loss.item())
        loss_history['binary_reg_loss'].append(binary_reg_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Total Loss: {total_loss.item():.6f}, "
                  f"Causal LRTR Loss: {causal_lrtr_loss.item():.6f}, "
                  f"sparsity_loss: {sparsity_loss.item():.6f},"
                  f"binary_reg_loss: {binary_reg_loss.item():.6f}")

    return model, loss_history


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
    plt.plot(loss_history['causal_lrtr_loss'])  # 修改键名为 'causal_lrtr_loss'
    plt.title('Causal LRTR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(loss_history['binary_reg_loss'])
    plt.title('binary_reg_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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


def main():
    """
    主函数 - 支持因果扰动图和相关性扰动图
    """
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 获取项目根目录
    project_root = Path(__file__).parent.parent  # Module的父目录
    print(f"Project root: {project_root}")

    # 配置参数 - 使用实际的节点数
    params = {
        'N': 99,  # 实际节点数
        'T': 10000,
        't': 3
    }

    # CADGNN模型路径
    cadgnn_model_path = "./CADGNN_results/N100_T10000_hid128_lay5_ladv0.1_lw0.02_20250902_203304_model.pth"

    # 扰动图目录
    causal_perturbed_dir = project_root / "dataset" / "Perturbed_Causal_Graphs_N100_T10000_Hubs15"

    try:
        # 加载CADGNN模型
        print("Loading CADGNN model...")
        cadgnn_model, cadgnn_config = load_cadgnn_model(cadgnn_model_path, device)
        print(f"Loaded CADGNN model with config: {cadgnn_config}")

        # 加载因果扰动图数据
        print("Loading causal perturbed graphs...")
        print(f"Looking in directory: {causal_perturbed_dir}")
        causal_perturbed_data_list = load_causal_perturbed_graphs(
            causal_perturbed_dir,
            num_nodes=99,  # 使用实际的99个节点
            tau_max=3,  # 时间滞后
            time_steps=10000  # 时间步数
        )

        print(f"Loaded {len(causal_perturbed_data_list)} causal perturbed graphs")

        if len(causal_perturbed_data_list) == 0:
            print("Warning: No perturbed graphs loaded. Check file format and paths.")
            print(
                "Expected file structure: ./dataset/Perturbed_Causal_Graphs/pcmci_results_node_*/pcmci_results_tau3_alpha0p0001.pkl")
            return

        # 创建因果不变子图提取器 - 使用实际扩展后的节点数
        print("Creating causal invariant subgraph extractor...")
        extractor_model = CausalInvariantSubgraphExtractor(
            cadgnn_model=cadgnn_model,
            input_dim=cadgnn_config['input_dim'],  # 应该是8000
            hidden_dim=cadgnn_config['hidden_dim'],  # 应该是128
            num_nodes=396,  # 修改为400，对应100个节点×4个时间滞后
            target_nodes=400  # 目标节点数也设为400
        )

        # 训练模型
        print("Training causal invariant subgraph extractor...")
        trained_model, loss_history = train_causal_invariant_subgraph_extractor(
            model=extractor_model,
            causal_perturbed_data_list=causal_perturbed_data_list,
            num_epochs=200,
            lr=0.005,
            device=device
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

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

