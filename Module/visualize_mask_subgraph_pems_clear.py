# Module/visualize_mask_subgraph.py
"""
基于训练好的掩码矩阵可视化因果不变子图
将400x400掩码矩阵还原为原始PCMCI格式进行可视化

功能说明：
- 将扩展的时间维度掩码矩阵（如400x400）重塑为原始PCMCI格式（[N, N, tau_max+1]）
- 根据阈值筛选重要的因果连接，生成因果不变子图
- 可视化筛选后的因果关系网络
- 支持多阈值分析以探索不同强度的因果连接

代码流程：
1. 加载400x400大小的掩码矩阵（对应100个节点，最大滞后3步）
2. 使用reshape_mask_to_original_format函数将矩阵重塑为[N, N, tau_max+1]格式
   - 遍历滞后时间层（0,1,2,3），提取对应的时间块
   - 对相同滞后时间的多个块取平均值，得到最终的因果掩码
3. 应用阈值过滤，生成二值化掩码
4. 使用create_subgraph_from_mask函数根据掩码筛选原始PCMCI结果中的连接
5. 构建NetworkX MultiDiGraph对象
6. 使用plot_causal_graph2和plot_causal_matrix进行可视化
7. 保存筛选后的子图数据到pickle文件

输入文件：
- mask_path: 训练好的掩码矩阵文件(.npy格式)
- original_pcmci_path: 原始PCMCI因果发现结果(.pkl格式)
- position_path: 节点位置坐标文件(.csv格式)

输出文件：
- 可视化图像文件（因果网络图和因果矩阵热力图）
- 子图数据文件(.pkl格式)
- 统计信息打印输出

参数说明：
- threshold: 掩码阈值，控制因果连接的选择强度
- original_nodes: 原始节点数量，默认100
- tau_max: 最大时间滞后，默认3

依赖模块：
- numpy, torch, pickle, matplotlib, networkx, pandas
- Module.pcmci_draw_MultiDiGraph2中的绘图函数
"""

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys
import pandas as pd
from pcmci_draw_MultiDiGraph2 import plot_causal_graph2_clear,plot_causal_graph2_pems_clear,plot_causal_graph2,plot_causal_graph2_pems,plot_causal_graph2_pems_o,plot_causal_matrix
import os


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


def create_subgraph_from_mask(graph, val_matrix, mask_reshaped, threshold=0.5):
    """
    根据掩码创建子图

    Args:
        graph: 原始PCMCI因果图 [N, N, tau_max+1]
        val_matrix: 原始因果效应矩阵 [N, N, tau_max+1]
        mask_reshaped: 重塑后的掩码矩阵 [N, N, tau_max+1]
        threshold: 掩码阈值

    Returns:
        subgraph: 子图的因果图矩阵
        subval_matrix: 子图的因果效应矩阵
    """
    # 应用阈值掩码
    binary_mask = apply_threshold_to_mask(mask_reshaped, threshold, mode='greater')

    # 创建子图
    subgraph = np.zeros_like(graph)
    subval_matrix = np.zeros_like(val_matrix)

    # 只保留掩码值大于阈值的连接
    mask_condition = binary_mask > 0

    # 复制原始连接，但只保留被掩码选中的
    subgraph[mask_condition & (graph == '-->')] = '-->'
    subval_matrix[mask_condition] = val_matrix[mask_condition]

    # 保留非零的因果效应
    subgraph[(subgraph == '-->') & (np.abs(subval_matrix) > 1e-6)] = '-->'

    return subgraph, subval_matrix


def node_scores_from_masked_effect(mask_reshaped, val_matrix, mode="inout"):
    """
    计算每个节点的“叠加掩码后的因果效应”重要性分数。
    S = |val| * mask
    mode:
      - "out": sum over outgoing (i -> j, lag)
      - "in":  sum over incoming (j -> i, lag)
      - "inout": out + in
    """
    S = np.abs(val_matrix) * mask_reshaped  # [N,N,L]
    out_score = S.sum(axis=(1, 2))         # sum over (dst, lag) for each src
    in_score  = S.sum(axis=(0, 2))         # sum over (src, lag) for each dst

    if mode == "out":
        return out_score
    if mode == "in":
        return in_score
    if mode == "inout":
        return out_score + in_score
    raise ValueError(f"Unknown mode: {mode}")


def create_subgraph_from_mask_top_nodes(
    graph,
    val_matrix,
    mask_reshaped,
    mask_threshold=0.5,
    top_ratio=0.30,
    node_score_mode="inout",
    edge_keep_mode="or",   # "or": 边两端任意一端在top节点就保留；"and": 两端都在top节点才保留
):
    """
    1) mask_threshold: 先把 mask 小于阈值的部分置零（只考虑强掩码区域）
    2) 用 |val|*mask 计算节点分数，选 top_ratio 节点
    3) 只保留：mask过阈值 & 原图有边 & (src/dst满足edge_keep_mode) 的边
    """

    # 先把弱mask过滤掉（只用强mask区域算节点分数）
    strong_mask = (mask_reshaped > mask_threshold).astype(float)

    # 节点分数（基于强mask区域的叠加效应）
    node_scores = node_scores_from_masked_effect(strong_mask, val_matrix, mode=node_score_mode)

    N = graph.shape[0]
    k = max(1, int(np.ceil(N * top_ratio)))
    top_nodes = np.argsort(-node_scores)[:k]
    top_set = set(top_nodes.tolist())

    # 构造子图：在强mask AND 原图有边 的基础上，再做节点门控
    subgraph = np.full_like(graph, '', dtype=object)  # 或 dtype=graph.dtype
    subval = np.zeros_like(val_matrix, dtype=float)

    N, _, L = graph.shape
    for u in range(N):
        for v in range(N):
            # 节点门控（先在节点级过滤，减少循环内判断）
            if edge_keep_mode == "and":
                if (u not in top_set) or (v not in top_set):
                    continue
            elif edge_keep_mode == "or":
                if (u not in top_set) and (v not in top_set):
                    continue
            else:
                raise ValueError(f"Unknown edge_keep_mode: {edge_keep_mode}")

            for lag in range(L):
                if strong_mask[u, v, lag] > 0 and graph[u, v, lag] == "-->":
                    subgraph[u, v, lag] = "-->"
                    subval[u, v, lag] = val_matrix[u, v, lag]

    # 去掉数值几乎为0的边
    mask_small = (subgraph == '-->') & (np.abs(subval) <= 1e-6)
    subgraph[mask_small] = ''
    subval[mask_small] = 0.0

    info = {
        "mask_threshold": float(mask_threshold),
        "top_ratio": float(top_ratio),
        "node_score_mode": node_score_mode,
        "edge_keep_mode": edge_keep_mode,
        "top_nodes": top_nodes.tolist(),
        "top_nodes_count": int(len(top_nodes)),
        "node_scores": node_scores.tolist(),  # 可选：如果太大你也可以不存
    }

    return subgraph, subval, info



def node_strength_from_val(val_matrix: np.ndarray) -> np.ndarray:
    """节点因果效应强度：入+出 的 |val| 总和，得到 shape [N]"""
    V = np.abs(val_matrix)
    out_strength = V.sum(axis=(1, 2))  # i -> *
    in_strength  = V.sum(axis=(0, 2))  # * -> i
    return out_strength + in_strength


def load_original_pcmci_results(pickle_path):
    """
    加载原始PCMCI结果
    """
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    return results['graph'], results['val_matrix']


def visualize_mask_subgraph(
        mask_path,
        original_pcmci_path,
        position_path,
        output_dir,
        threshold=0.5,
        original_nodes=100,  # 修改默认值为100
        tau_max=3
):
    """
    可视化基于掩码的因果子图
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

    # 创建子图
    # subgraph, subval_matrix = create_subgraph_from_mask(
    #     original_graph,
    #     original_val_matrix,
    #     mask_reshaped,
    #     threshold=threshold
    # )

    subgraph, subval_matrix, node_gate_info = create_subgraph_from_mask_top_nodes(
        graph=original_graph,
        val_matrix=original_val_matrix,
        mask_reshaped=mask_reshaped,
        mask_threshold=threshold,  # 你原来的 threshold
        top_ratio=0.30,  # 前30%节点
        node_score_mode="inout",  # in/out/inout 都可
        edge_keep_mode="or",  # ✅ 按你描述：top节点 + 该节点的连接（OR更像）
    )

    # 统计子图信息
    selected_edges = np.sum((subgraph == '-->').astype(int))
    total_edges = np.sum((original_graph == '-->').astype(int))
    print(f"Selected edges: {selected_edges}, Total edges: {total_edges}")
    print(f"Edge selection ratio: {selected_edges / total_edges if total_edges > 0 else 0:.4f}")

    # 使用pcmci_draw_MultiDiGraph2的函数进行可视化
    N = original_graph.shape[0]

    # 构建MultiDiGraph进行可视化
    from Module.pcmci_draw_MultiDiGraph2 import build_multi_di_graph

    G, edge_counts, node_self_counts = build_multi_di_graph(
        subgraph, subval_matrix, N, tau_max, positions
    )

    print(f"Subgraph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # 生成输出文件名
    tau_str = f'tau{tau_max}'
    alpha_str = 'mask'  # 表示这是掩码子图

    # 可视化子图
    plot_causal_graph2_clear(
        G,
        dict(enumerate(positions)),
        edge_counts,
        node_self_counts,
        output_dir,
        tau_str,
        alpha_str,
        title=f"Mask-Based Causal Subgraph (threshold={threshold}, N={N})"

    )

    # 可视化因果矩阵热力图
    plot_causal_matrix(
        edge_counts,
        N,
        output_dir,
        tau_str,
        alpha_str
    )

    # 保存子图数据
    # subgraph_data = {
    #     'graph': subgraph,
    #     'val_matrix': subval_matrix,
    #     'mask_threshold': threshold,
    #     'selected_edges_count': selected_edges,
    #     'total_edges_count': total_edges
    # }

    # subgraph_data = {
    #     'graph': subgraph,
    #     'val_matrix': subval_matrix,
    #     'mask_threshold': threshold,
    #     'selected_edges_count': int(np.sum((subgraph == '-->').astype(int))),
    #     'total_edges_count': int(np.sum((original_graph == '-->').astype(int))),
    #     **node_gate_info
    # }
    subgraph_data = {
        'graph': subgraph,
        'val_matrix': subval_matrix,
        "meta": {
            "mask_threshold": float(threshold),
            "selected_edges_count": int(np.sum(subgraph == "-->").astype(int)),
            "total_edges_count": int(np.sum((original_graph == "-->").astype(int))),

            # ✅ top_nodes 信息（全局节点编号 0..99）
            "top_nodes": node_gate_info["top_nodes"],
            "top_nodes_count": int(node_gate_info["top_nodes_count"]),
            "top_ratio": float(node_gate_info["top_ratio"]),
            "node_score_mode": node_gate_info["node_score_mode"],
            "edge_keep_mode": node_gate_info["edge_keep_mode"],

            # 可选：节点分数（如果你觉得太大可以不存）
            "node_scores": node_gate_info["node_scores"],
        }
    }

    subgraph_path = os.path.join(output_dir, f'subgraph_threshold_{threshold}.pkl')
    with open(subgraph_path, 'wb') as f:
        pickle.dump(subgraph_data, f)

    print(f"Subgraph saved to: {subgraph_path}")
    print("Visualization completed!")


def analyze_and_visualize_multiple_thresholds(
        mask_path,
        original_pcmci_path,
        position_path,
        output_base_dir,
        thresholds=[0.5],
        original_nodes=100,
        tau_max=3
):
    """
    使用多个阈值分析和可视化子图
    """
    for threshold in thresholds:
        output_dir = os.path.join(output_base_dir, f'threshold_{threshold}')
        print(f"\nProcessing threshold: {threshold}")

        try:
            visualize_mask_subgraph(
                mask_path=mask_path,
                original_pcmci_path=original_pcmci_path,
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
    主函数 - 可视化掩码子图
    """
    # 配置路径
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "Module/causal_invariant_results_pems/results_pems_20260116_172104"

    # 直接使用未降维的掩码文件
    full_mask_files = list(results_dir.glob("full_mask_matrix_20260116_172104.npy"))
    if not full_mask_files:
        print("No full mask files found in results directory")
        return

    mask_path = full_mask_files[0]  # 使用400×400未降维掩码
    print(f"Using full mask: {mask_path}")

    # 原始PCMCI结果路径
    pcmci_results_dir = project_root / "Module" / "PCMCI_PEMS/N100_1h_standardize_id_4"
    pcmci_file = pcmci_results_dir / "pcmci_results_tau3_alpha0p0500.pkl"

    # 节点位置文件
    dataset_dir = project_root / "dataset" / "pems_spatial_kmeans_topK"
    position_file = dataset_dir / "id_node_positions.csv"

    # 输出目录
    output_base_dir = project_root / "mask_subgraph_visualization/results_pems_20260116_172104_clear2"

    # 检查文件是否存在
    if not mask_path.exists():
        print(f"Mask file not found: {mask_path}")
        return

    if not pcmci_file.exists():
        print(f"PCMCI file not found: {pcmci_file}")
        return

    if not position_file.exists():
        print(f"Position file not found: {position_file}")
        return

    print(f"Full mask path: {mask_path}")
    print(f"PCMCI path: {pcmci_file}")
    print(f"Position path: {position_file}")
    print(f"Output base dir: {output_base_dir}")

    # 分析和可视化多个阈值 - 针对400×400掩码
    analyze_and_visualize_multiple_thresholds(
        mask_path=mask_path,
        original_pcmci_path=pcmci_file,
        position_path=position_file,
        output_base_dir=output_base_dir,
        #thresholds=[0.3,0.5,0.8,0.9,0.95,0.99,0.86],
        thresholds=[0.5],
        original_nodes=100,  # 因为是400=100×4，所以原始节点数是100
        tau_max=3
    )


if __name__ == "__main__":
    main()
