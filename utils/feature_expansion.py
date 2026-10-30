"""
概述
该文件提供特征扩展功能，将原始时间序列数据扩展为包含时间滞后的特征矩阵。
用于因果发现算法中的时间序列数据预处理。

函数说明

1. expand_features_for_causal_graph(features, causal_nodes=400, original_nodes=100, tau_max=3)
功能: 统一的特征扩展方法（适用于因果图和相关图）
参数:
  - features: 原始特征，形状为(N, T)，其中N是原始节点数，T是时间步数
  - causal_nodes: 扩展后的节点数（默认为400）
  - original_nodes: 原始节点数（默认为100）
  - tau_max: 最大时间滞后（默认为3）
返回: 扩展后的特征矩阵，形状为(causal_nodes, T-tau_max)

2. verify_temporal_alignment(features_expanded, original_features, tau_max=3)
功能: 验证时间滞后对齐是否正确
参数:
  - features_expanded: 扩展后的特征矩阵
  - original_features: 原始特征矩阵
  - tau_max: 最大时间滞后
返回: 无，但会在控制台输出验证结果

时间滞后对齐规则
- τ=0: 从第4个值开始取（索引3）
- τ=1: 从第3个值开始取（索引2）
- τ=2: 从第2个值开始取（索引1）
- τ=3: 从第1个值开始取（索引0）

尾部删减规则
- τ=0: 不删减
- τ=1: 删减1个时间步
- τ=2: 删减2个时间步
- τ=3: 删减3个时间步
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def expand_features_for_causal_graph(features, causal_nodes=400, original_nodes=100, tau_max=3):
    """
    统一的特征扩展方法（适用于因果图和相关图）

    时间滞后对齐规则：
    - τ=0: 从第4个值开始取 (索引3)
    - τ=1: 从第3个值开始取 (索引2)
    - τ=2: 从第2个值开始取 (索引1)
    - τ=3: 从第1个值开始取 (索引0)

    尾部删减规则：
    - τ=0: 不删减
    - τ=1: 删减1个时间步
    - τ=2: 删减2个时间步
    - τ=3: 删减3个时间步

    Args:
        features: 原始特征 (N, T) 格式，其中N是原始节点数，T是时间步数
        causal_nodes: 扩展后的节点数 (400)
        original_nodes: 原始节点数 (100)
        tau_max: 最大时间滞后 (3)

    Returns:
        expanded_features: 扩展后的特征 (causal_nodes, T-tau_max) 格式
    """
    if features.shape[0] != original_nodes:
        raise ValueError(f"Expected {original_nodes} nodes, but got {features.shape[0]}")

    T = features.shape[1]
    if T <= tau_max:
        raise ValueError(f"Time steps {T} must be greater than tau_max {tau_max}")

    # 计算扩展后的特征维度
    n_time_layers = tau_max + 1  # 4层 (t, t-1, t-2, t-3)
    expanded_T = T - tau_max  # 10000 - 3 = 9997

    # 创建扩展后的特征矩阵
    expanded_features = np.zeros((causal_nodes, expanded_T), dtype=features.dtype)

    # 每个时间层按规则扩展特征
    for t in range(n_time_layers):  # t=0,1,2,3 对应 τ=3,2,1,0
        start_idx = t * original_nodes
        end_idx = (t + 1) * original_nodes

        # 计算时间步偏移
        start_offset = tau_max - t  # τ=3,2,1,0 对应偏移 0,1,2,3
        end_offset = t  # 尾部删减偏移

        # 提取对应的特征片段
        if end_offset == 0:
            feature_slice = features[:original_nodes, start_offset:]
        else:
            feature_slice = features[:original_nodes, start_offset:-end_offset]

        expanded_features[start_idx:end_idx, :] = feature_slice

    return expanded_features


def verify_temporal_alignment(features_expanded, original_features, tau_max=3):
    """
    验证时间滞后对齐

    Args:
        features_expanded: 扩展后的特征
        original_features: 原始特征
        tau_max: 最大时间滞后
    """
    print("时间滞后对齐验证:")
    for t in range(tau_max + 1):
        start_idx = t * 100
        end_idx = (t + 1) * 100

        # 计算理论偏移
        start_offset = tau_max - t
        print(f"τ={tau_max - t}: 时间层{t} 应从索引{start_offset}开始")

        # 验证前5个时间步
        for i in range(5):
            original_idx = start_offset + i
            expanded_idx = i
            print(f"  原始特征[{original_idx}] = {original_features[0, original_idx]:.4f}, "
                  f"扩展特征[{expanded_idx}] = {features_expanded[start_idx, expanded_idx]:.4f}")
