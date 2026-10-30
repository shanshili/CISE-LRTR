# Module/visualize_mask_subgraph2.py
"""
基于训练好的掩码矩阵可视化因果不变子图
将400x400掩码矩阵还原为原始PCMCI格式进行可视化，并融合SCMI因果发现结果

功能说明：
- 将扩展的时间维度掩码矩阵（如400x400）重塑为原始PCMCI格式（[N, N, tau_max+1]）
- 同时处理PCMCI和SCMI两种因果发现方法的结果
- 根据阈值筛选重要的因果连接，生成融合多种方法的因果不变子图
- 可视化筛选后的因果关系网络（包含有向和无向连接）
- 支持多阈值分析以探索不同强度的因果连接

代码流程：
1. 加载400x400大小的掩码矩阵（对应100个节点，最大滞后3步）
2. 使用reshape_mask_to_original_format函数将矩阵重塑为[N, N, tau_max+1]格式
   - 遍历滞后时间层（0,1,2,3），提取对应的时间块
   - 对相同滞后时间的多个块取平均值，得到最终的因果掩码
3. 加载PCMCI和SCMI两种因果发现结果
4. 应用阈值过滤，生成二值化掩码
5. 使用create_subgraph_from_mask函数根据掩码筛选PCMCI和SCMI结果中的连接
   - PCMCI结果：时序因果关系（有向边）
   - SCMI结果：瞬时因果关系（无向边，用双向箭头表示）
6. 构建NetworkX MultiDiGraph对象，整合两种因果关系
7. 使用add_scmi_connections函数添加SCMI无向连接到图中
8. 使用plot_causal_graph3和plot_causal_matrix2进行综合可视化
9. 保存筛选后的子图数据到pickle文件

输入文件：
- mask_path: 训练好的掩码矩阵文件(.npy格式)
- original_pcmci_path: 原始PCMCI因果发现结果(.pkl格式)
- scmi_path: 原始SCMI因果发现结果(.npy格式)
- position_path: 节点位置坐标文件(.csv格式)

输出文件：
- 可视化图像文件（融合因果网络图和因果矩阵热力图）
- 子图数据文件(.pkl格式，包含PCMCI和SCMI结果)
- 统计信息打印输出

参数说明：
- threshold: 掩码阈值，控制因果连接的选择强度
- original_nodes: 原始节点数量，默认100
- tau_max: 最大时间滞后，默认3

依赖模块：
- numpy, torch, pickle, matplotlib, networkx, pandas, seaborn
- Module.pcmci_draw_MultiDiGraph2中的高级绘图函数
"""

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys
import pandas as pd
from pcmci_draw_MultiDiGraph2 import plot_causal_graph3, plot_causal_matrix2
from Module.CausalInvariantSubgraphExtractor2_3_1_1 import normalize_matrix_robust, normalize_matrix_with_sign
import os
import seaborn as sns



def reshape_mask_to_original_format(mask_matrix, original_nodes=100, tau_max=3):
    """
    修正版：正确提取滞后时间步的掩码
    """
    # 确保输入是numpy数组
    if torch.is_tensor(mask_matrix):
        mask_matrix = mask_matrix.detach().cpu().numpy()

    # 获取实际尺寸
    total_nodes = mask_matrix.shape[0]
    time_layers = tau_max + 1  # 4层 (t=0,1,2,3)

    # 自动修正 original_nodes
    if total_nodes != original_nodes * time_layers:
        original_nodes = total_nodes // time_layers
        print(f"Auto-corrected original_nodes to: {original_nodes}")

    # 初始化重塑后的掩码: [Nodes, Nodes, Lags] (对应 PCMCI 格式)
    # Lags = 0, 1, 2, 3
    reshaped_mask = np.zeros((original_nodes, original_nodes, time_layers))

    # 遍历所有可能的起始时间层 t
    # 我们想要提取 Lag k:  Block(t, t+k)
    # 扩展矩阵结构：行是源(Source/Cause)，列是目标(Target/Effect) -> 这里假设 mask[i, j] 是 i->j
    # 注意：PCMCI通常是 graph[i, j, lag] 表示 i(lag) -> j(0)

    # 我们采取一种聚合策略：对于 Lag k，在扩展矩阵中有多个块代表这个 Lag (例如 0->1, 1->2, 2->3 都是 Lag 1)
    # 我们将它们取平均，作为该 Lag 的最终掩码值

    for lag in range(time_layers):  # lag = 0, 1, 2, 3
        blocks = []

        # 寻找所有代表 Lag k 的块
        # 也就是从 t 时刻 指向 t + lag 时刻
        for t in range(time_layers - lag):
            t_source = t
            t_target = t + lag

            row_start = t_source * original_nodes
            row_end = (t_source + 1) * original_nodes
            col_start = t_target * original_nodes
            col_end = (t_target + 1) * original_nodes

            # 提取块
            block = mask_matrix[row_start:row_end, col_start:col_end]

            # PCMCI 格式通常需要转置？
            # 我们的 mask_matrix[i, j] 意味着 i -> j
            # PCMCI data[i, j, lag] 意味着 j(t) depends on i(t-lag) (通常 i 是 cause)
            # 这里先保持原样，如果方向反了再加 .T
            blocks.append(block)

        if blocks:
            # 堆叠并取平均，得到该 Lag 的共识掩码
            avg_block = np.mean(np.stack(blocks, axis=0), axis=0)
            reshaped_mask[:, :, lag] = avg_block

    # 这里的 reshaped_mask[i, j, lag] 表示 节点i (滞后lag) -> 节点j (当前)
    # 如果此时全是 0.5，说明训练确实有问题；如果能看到结构，说明只是可视化提取错了位置。
    return reshaped_mask


def apply_threshold_to_mask(mask_matrix, threshold=0.5, mode='greater'):
    """
    对掩码应用阈值，生成二值化掩码

    Args:
        mask_matrix: 掩码矩阵
        threshold: 阈值
        mode: 'greater' (> threshold), 'less' (< threshold), 'both' (|x-threshold| > delta)

    Returns:
        binary_mask: 二值化掩码
    """
    if mode == 'greater':
        binary_mask = (mask_matrix > threshold).astype(float)
    elif mode == 'less':
        binary_mask = (mask_matrix < threshold).astype(float)
    elif mode == 'both':
        binary_mask = np.abs(mask_matrix - threshold) > (0.5 - threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return binary_mask


def create_subgraph_from_mask(graph, val_matrix, scmi_matrix, mask_reshaped, threshold=0.5,normalize_effects=True):
    """
    根据掩码创建子图，同时考虑PCMCI和SCMI结果

    Args:
        graph: 原始PCMCI因果图 [N, N, tau_max+1]
        val_matrix: 原始因果效应矩阵 [N, N, tau_max+1]
        scmi_matrix: 原始SCMI因果矩阵 [N, N]
        mask_reshaped: 重塑后的掩码矩阵 [N, N, tau_max+1]
        threshold: 掩码阈值

    Returns:
        subgraph: 子图的因果图矩阵
        subval_matrix: 子图的因果效应矩阵
        subscmi_matrix: 子图的SCMI矩阵
    """
    # 应用阈值掩码
    binary_mask = apply_threshold_to_mask(mask_reshaped, threshold, mode='greater')

    # 创建子图
    subgraph = np.zeros_like(graph)
    subval_matrix = np.zeros_like(val_matrix)
    subscmi_matrix = np.zeros_like(scmi_matrix)

    # 只保留掩码值大于阈值的连接
    mask_condition = binary_mask > 0
    # 复制原始PCMCI连接，但只保留被掩码选中的
    subgraph[mask_condition & (graph == '-->')] = '-->'
    subval_matrix[mask_condition] = val_matrix[mask_condition]
    # 保留非零的因果效应
    subgraph[(subgraph == '-->') & (np.abs(subval_matrix) > 1e-6)] = '-->'

    # 处理SCMI结果：SCMI是N*N矩阵，对应Lag 0
    # 对于SCMI，使用掩码的第0层（Lag 0）来筛选连接
    scmi_mask = binary_mask[:, :, 0]  # 取Lag 0的掩码
    # 应用SCMI掩码
    scmi_condition = scmi_mask > 0
    subscmi_matrix[scmi_condition] = scmi_matrix[scmi_condition]
    # 4. 仅对选中的连接进行标准化（用于可视化）
    if normalize_effects:
        # 对选中的PCMCI效应值使用带符号标准化
        if np.any(subval_matrix):
            # 只对被掩码选中的位置进行标准化
            selected_values = subval_matrix[mask_condition]
            normalized_selected = normalize_matrix_with_sign(selected_values)
            subval_matrix[mask_condition] = normalized_selected

        # 对选中的SCMI效应值使用鲁棒标准化
        if np.any(subscmi_matrix):
            selected_scmi_values = subscmi_matrix[scmi_condition]
            normalized_scmi_selected = normalize_matrix_robust(selected_scmi_values)
            subscmi_matrix[scmi_condition] = normalized_scmi_selected

    return subgraph, subval_matrix, subscmi_matrix


def load_original_pcmci_results(pickle_path):
    """
    加载原始PCMCI结果
    """
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    return results['graph'], results['val_matrix']


def load_original_scmi_result(scmi_path):
    """
    加载原始SCMI结果

    Args:
        scmi_path: SCMI结果文件路径

    Returns:
        scmi_matrix: SCMI因果矩阵 [N, N]
    """
    scmi_data = np.load(scmi_path, allow_pickle=True)

    # 处理不同的数据格式
    if isinstance(scmi_data, dict):
        # 如果是字典格式，提取graph键
        scmi_matrix = scmi_data.get('graph', scmi_data.get('scmi_matrix', None))
    else:
        # 如果是直接的矩阵
        scmi_matrix = scmi_data

    if scmi_matrix is None:
        raise ValueError(f"No 'graph' or 'scmi_matrix' key found in {scmi_path}")

    # 确保是numpy数组
    scmi_matrix = np.array(scmi_matrix)

    print(f"SCMI matrix shape: {scmi_matrix.shape}")
    return scmi_matrix


def add_scmi_connections(G, scmi_matrix, positions):
    """
    向网络图中添加SCMI无向连接

    Args:
        G: NetworkX MultiDiGraph
        scmi_matrix: SCMI因果矩阵 [N, N]
        positions: 节点位置数组
    """
    N = scmi_matrix.shape[0]

    # 遍历SCMI矩阵，添加无向连接
    for i in range(N):
        for j in range(i + 1, N):  # 只遍历上三角，避免重复
            if scmi_matrix[i, j] != 0:
                # 添加无向连接（在MultiDiGraph中用双向箭头表示无向连接）
                # 使用特殊属性标识这是SCMI连接
                G.add_edge(i, j,
                           connection_type='scmi',
                           weight=abs(scmi_matrix[i, j]),
                           causal_effect=scmi_matrix[i, j],
                           relation='---',  # 用虚线表示无向连接
                           color='blue',
                           style='solid',  # 实线
                           arrows=False)

                # 同时添加反向连接，使可视化时表现为无向连接
                G.add_edge(j, i,
                           connection_type='scmi',
                           weight=abs(scmi_matrix[j, i]),
                           causal_effect=scmi_matrix[j, i],
                           relation='---',  # 用虚线表示无向连接
                           color='blue',
                           style='solid',  # 实线
                           arrows=False)


def visualize_mask_subgraph(
        mask_path,
        original_pcmci_path,
        scmi_path,  # 添加SCMI路径参数
        position_path,
        output_dir,
        threshold=0.5,
        original_nodes=100,
        tau_max=3
):
    """
    可视化基于掩码的因果子图，同时包含PCMCI和SCMI结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 直接加载.npy文件 - 400×400未降维掩码
    mask_path_str = str(mask_path)
    if mask_path_str.endswith('.npy'):
        mask_matrix = np.load(mask_path)
    else:
        raise ValueError(f"Expected .npy file, got: {mask_path}")

    print(f"Loaded full mask matrix shape: {mask_matrix.shape}")

    # 加载原始PCMCI结果
    original_graph, original_val_matrix = load_original_pcmci_results(original_pcmci_path)
    print(f"Original graph shape: {original_graph.shape}")
    print(f"Original val_matrix shape: {original_val_matrix.shape}")

    # 加载原始SCMI结果
    original_scmi_matrix = load_original_scmi_result(scmi_path)
    print(f"Original SCMI shape: {original_scmi_matrix.shape}")

    # 加载节点位置
    positions_df = pd.read_csv(position_path)
    positions = positions_df[['x', 'y']].values
    print(f"Positions shape: {positions.shape}")

    # 重塑掩码矩阵 - 从400×400转换为原始格式
    mask_reshaped = reshape_mask_to_original_format(
        mask_matrix,
        original_nodes=original_nodes,  # 100
        tau_max=tau_max  # 3
    )

    # 创建子图 - 现在包括SCMI
    subgraph, subval_matrix, subscmi_matrix = create_subgraph_from_mask(
        original_graph,
        original_val_matrix,
        original_scmi_matrix,
        mask_reshaped,
        threshold=threshold
    )

    # 统计子图信息
    selected_edges = np.sum((subgraph == '-->').astype(int))
    total_edges = np.sum((original_graph == '-->').astype(int))
    selected_scmi_edges = np.count_nonzero(subscmi_matrix)
    total_scmi_edges = np.count_nonzero(original_scmi_matrix)

    print(f"Selected PCMCI edges: {selected_edges}, Total PCMCI edges: {total_edges}")
    print(f"Selected SCMI edges: {selected_scmi_edges}, Total SCMI edges: {total_scmi_edges}")
    print(f"PCMCI edge selection ratio: {selected_edges / total_edges if total_edges > 0 else 0:.4f}")
    print(f"SCMI edge selection ratio: {selected_scmi_edges / total_scmi_edges if total_scmi_edges > 0 else 0:.4f}")

    # 使用pcmci_draw_MultiDiGraph2的函数进行可视化
    N = original_graph.shape[0]

    # 构建MultiDiGraph进行可视化 - 现在包含SCMI无向连接
    from Module.pcmci_draw_MultiDiGraph2 import build_multi_di_graph

    # 首先使用PCMCI结果构建图
    G, edge_counts, node_self_counts = build_multi_di_graph(
        subgraph, subval_matrix, N, tau_max, positions
    )

    # 然后添加SCMI无向连接
    add_scmi_connections(G, subscmi_matrix, positions)

    print(f"Subgraph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # 生成输出文件名
    tau_str = f'tau{tau_max}'
    alpha_str = 'mask'  # 表示这是掩码子图

    # 可视化子图 - 现在包含SCMI连接
    plot_causal_graph3(
        G,
        dict(enumerate(positions)),
        edge_counts,
        node_self_counts,
        output_dir,
        tau_str,
        alpha_str,
        title=f"Mask-Based Causal Subgraph (threshold={threshold}, N={N})",
        scmi_matrix=subscmi_matrix  # 传递SCMI矩阵用于可视化
    )

    # 可视化因果矩阵热力图
    plot_causal_matrix2(
        edge_counts,
        N,
        output_dir,
        tau_str,
        alpha_str,
        scmi_matrix=subscmi_matrix  # 传递SCMI矩阵用于可视化
    )

    # 保存子图数据 - 包含SCMI结果
    subgraph_data = {
        'graph': subgraph,
        'val_matrix': subval_matrix,
        'scmi_matrix': subscmi_matrix,
        'mask_threshold': threshold,
        'selected_pcmci_edges_count': selected_edges,
        'total_pcmci_edges_count': total_edges,
        'selected_scmi_edges_count': selected_scmi_edges,
        'total_scmi_edges_count': total_scmi_edges
    }

    subgraph_path = os.path.join(output_dir, f'subgraph_threshold_{threshold}.pkl')
    with open(subgraph_path, 'wb') as f:
        pickle.dump(subgraph_data, f)

    print(f"Subgraph saved to: {subgraph_path}")
    print("Visualization completed!")



def analyze_and_visualize_multiple_thresholds(
        mask_path,
        original_pcmci_path,
        scmi_path,  # 添加SCMI路径参数
        position_path,
        output_base_dir,
        thresholds=[0.4, 0.5, 0.6],
        original_nodes=100,
        tau_max=3
):
    """
    使用多个阈值分析和可视化子图，同时包含PCMCI和SCMI
    """
    for threshold in thresholds:
        output_dir = os.path.join(output_base_dir, f'threshold_{threshold}')
        print(f"\nProcessing threshold: {threshold}")

        try:
            visualize_mask_subgraph(
                mask_path=mask_path,
                original_pcmci_path=original_pcmci_path,
                scmi_path=scmi_path,  # 传递SCMI路径
                position_path=position_path,
                output_dir=output_dir,
                threshold=threshold,
                original_nodes=original_nodes,
                tau_max=tau_max
            )
        except Exception as e:
            print(f"Error processing threshold {threshold}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    主函数 - 可视化掩码子图，同时包含PCMCI和SCMI结果
    """
    # 配置路径
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "Module/causal_invariant_results" / "results_20260114_173506"

    # 直接使用未降维的掩码文件
    full_mask_files = list(results_dir.glob("full_mask_matrix_20260114_173506.npy"))
    if not full_mask_files:
        print("No full mask files found in results directory")
        return

    mask_path = full_mask_files[0]  # 使用400×400未降维掩码
    print(f"Using full mask: {mask_path}")

    # 原始PCMCI结果路径
    pcmci_results_dir = project_root / "Module" / "PCMCI_" / "N100_T10000_Hubs15_Viz_Final"
    pcmci_file = pcmci_results_dir / "pcmci_results_tau3_alpha0p0001.pkl"

    # SCMI结果路径
    scmi_results_dir = project_root / "Module" / "SCMI_N100_T10000_Hubs15_delta20.0_k10_bins5"
    scmi_file = scmi_results_dir / "scmi_causal_graph_delta20.0_k10_bins5_threshold1e-05.npy"

    # 节点位置文件
    dataset_dir = project_root / "dataset" / "simulation_hubs_core" / "N100_T10000_Hubs15_Viz_Final"
    position_file = dataset_dir / "node_positions.csv"

    # 输出目录
    output_base_dir = project_root / "mask_subgraph_visualization/N100_T10000_Hubs15_Viz_Final_20260114_173506"

    # 检查文件是否存在
    if not mask_path.exists():
        print(f"Mask file not found: {mask_path}")
        return

    if not pcmci_file.exists():
        print(f"PCMCI file not found: {pcmci_file}")
        return

    if not scmi_file.exists():
        print(f"SCMI file not found: {scmi_file}")
        return

    if not position_file.exists():
        print(f"Position file not found: {position_file}")
        return

    print(f"Full mask path: {mask_path}")
    print(f"PCMCI path: {pcmci_file}")
    print(f"SCMI path: {scmi_file}")
    print(f"Position path: {position_file}")
    print(f"Output base dir: {output_base_dir}")

    # 分析和可视化多个阈值 - 针对400×400掩码，同时包含PCMCI和SCMI
    analyze_and_visualize_multiple_thresholds(
        mask_path=mask_path,
        original_pcmci_path=pcmci_file,
        scmi_path=scmi_file,  # 传递SCMI路径
        position_path=position_file,
        output_base_dir=output_base_dir,
        thresholds=[0.3, 0.5, 0.8, 0.9, 0.95, 0.99, 0.86],
        original_nodes=100,  # 因为是400=100×4，所以原始节点数是100
        tau_max=3
    )

if __name__ == "__main__":
    main()
