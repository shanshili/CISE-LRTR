# visualize_perturbation_comparison.py
"""
该文件用于可视化扰动因果图并与原始因果图进行对比
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import pickle
from PCMCI_2 import load_pcmci_results

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入 pcmci_draw_MultiDiGraph2 模块
from pcmci_draw_MultiDiGraph2 import draw_causal_graph


def load_original_causal_graph(pickle_file_path):
    """
    加载原始因果图

    Parameters:
    - pickle_file_path: 原始因果图pickle文件路径

    Returns:
    - graph, val_matrix
    """
    with open(pickle_file_path, 'rb') as f:
        loaded_results = pickle.load(f)

    graph = loaded_results['graph']
    val_matrix = loaded_results['val_matrix']

    return graph, val_matrix


def load_perturbed_causal_graph(npy_file_path):
    """
    加载扰动因果图

    Parameters:
    - npy_file_path: 扰动因果图npy文件路径

    Returns:
    - graph, val_matrix
    """
    data = np.load(npy_file_path, allow_pickle=True).item()
    graph = data['graph']
    val_matrix = data['val_matrix']

    return graph, val_matrix


def visualize_causal_graph_comparison(original_graph, original_val_matrix,
                                      perturbed_graph, perturbed_val_matrix,
                                      removed_node, output_dir,
                                      node_positions_path, tau_max=3):
    """
    可视化原始因果图与扰动因果图的对比

    Parameters:
    - original_graph: 原始因果图
    - original_val_matrix: 原始因果效应矩阵
    - perturbed_graph: 扰动因果图
    - perturbed_val_matrix: 扰动因果效应矩阵
    - removed_node: 被移除的节点
    - output_dir: 输出目录
    - node_positions_path: 节点位置文件路径
    - tau_max: 最大时间滞后
    """
    # 加载节点位置
    positions_df = pd.read_csv(node_positions_path)
    positions = positions_df[['x', 'y']].values

    # 获取节点数量
    original_N = original_graph.shape[0]
    perturbed_N = perturbed_graph.shape[0]

    print(f"原始节点数: {original_N}, 扰动后节点数: {perturbed_N}")

    # 调整扰动图的位置矩阵（移除被删除的节点）
    adjusted_positions = np.delete(positions, removed_node, axis=0)

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 使用 pcmci_draw_MultiDiGraph2.py 中的函数绘制原始因果图
    draw_causal_graph(
        graph=original_graph,
        val_matrix=original_val_matrix,
        positions=positions,
        ax=axes[0],
        title=f"Original Causal Graph (N={original_N})",
        tau_max=tau_max
    )

    # 使用 pcmci_draw_MultiDiGraph2.py 中的函数绘制扰动因果图
    draw_causal_graph(
        graph=perturbed_graph,
        val_matrix=perturbed_val_matrix,
        positions=adjusted_positions,
        ax=axes[1],
        title=f"Perturbed Causal Graph (Node {removed_node} removed, N={perturbed_N})",
        tau_max=tau_max
    )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    output_path = os.path.join(output_dir, f'comparison_causal_graph_node_{removed_node}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"对比图像已保存: {output_path}")
    return output_path


def visualize_comparison_for_nodes(original_pickle_path, perturbed_dir,
                                   node_positions_path, output_dir,
                                   nodes_to_compare=[0, 1, 5, 10, 20]):
    """
    为指定节点可视化原始因果图与扰动因果图的对比

    Parameters:
    - original_pickle_path: 原始因果图pickle文件路径
    - perturbed_dir: 扰动因果图目录
    - node_positions_path: 节点位置文件路径
    - output_dir: 输出目录
    - nodes_to_compare: 要比较的节点列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始因果图
    original_graph, original_val_matrix = load_original_causal_graph(original_pickle_path)

    print(f"原始因果图已加载，节点数: {original_graph.shape[0]}")

    # 为每个指定节点生成对比图像
    saved_files = []
    for node in nodes_to_compare:
        print(f"正在处理节点 {node} 的对比...")

        # 构建扰动因果图文件路径
        perturbed_file_path = os.path.join(perturbed_dir, f'perturbed_graph_node_{node}.npy')

        if not os.path.exists(perturbed_file_path):
            print(f"警告: 找不到扰动文件 {perturbed_file_path}")
            continue

        try:
            # 加载扰动因果图
            perturbed_graph, perturbed_val_matrix = load_perturbed_causal_graph(perturbed_file_path)

            # 可视化对比
            saved_path = visualize_causal_graph_comparison(
                original_graph, original_val_matrix,
                perturbed_graph, perturbed_val_matrix,
                node, output_dir, node_positions_path
            )
            saved_files.append(saved_path)

        except Exception as e:
            print(f"处理节点 {node} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print(f"共保存了 {len(saved_files)} 个对比图像")
    return saved_files


def visualize_single_perturbed_graph(perturbed_file_path, node_positions_path,
                                     output_dir, removed_node, tau_max=3):
    """
    可视化单个扰动因果图

    Parameters:
    - perturbed_file_path: 扰动因果图文件路径
    - node_positions_path: 节点位置文件路径
    - output_dir: 输出目录
    - removed_node: 被移除的节点
    - tau_max: 最大时间滞后
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载扰动因果图
    perturbed_graph, perturbed_val_matrix = load_perturbed_causal_graph(perturbed_file_path)

    # 加载节点位置
    positions_df = pd.read_csv(node_positions_path)
    positions = positions_df[['x', 'y']].values

    # 调整扰动图的位置矩阵（移除被删除的节点）
    adjusted_positions = np.delete(positions, removed_node, axis=0)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用 pcmci_draw_MultiDiGraph2.py 中的函数绘制扰动因果图
    draw_causal_graph(
        graph=perturbed_graph,
        val_matrix=perturbed_val_matrix,
        positions=adjusted_positions,
        ax=ax,
        title=f"Perturbed Causal Graph (Node {removed_node} removed, N={perturbed_graph.shape[0]})",
        tau_max=tau_max
    )

    # 保存图像
    output_path = os.path.join(output_dir, f'perturbed_causal_graph_node_{removed_node}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"扰动因果图已保存: {output_path}")
    return output_path


def visualize_single_perturbed_graph_direct(perturbed_graph, perturbed_val_matrix,
                                            positions, output_dir, removed_node,
                                            tau_max=3, title=None):
    """
    直接可视化单个扰动因果图（不需要从文件加载）

    Parameters:
    - perturbed_graph: 扰动因果图
    - perturbed_val_matrix: 扰动因果效应矩阵
    - positions: 节点位置
    - output_dir: 输出目录
    - removed_node: 被移除的节点
    - tau_max: 最大时间滞后
    - title: 图标题
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    if title is None:
        title = f"Perturbed Causal Graph (Node {removed_node} removed, N={perturbed_graph.shape[0]})"

    # 使用 pcmci_draw_MultiDiGraph2.py 中的函数绘制扰动因果图
    draw_causal_graph(
        graph=perturbed_graph,
        val_matrix=perturbed_val_matrix,
        positions=positions,
        ax=ax,
        title=title,
        tau_max=tau_max
    )

    # 保存图像
    output_path = os.path.join(output_dir, f'perturbed_causal_graph_node_{removed_node}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"扰动因果图已保存: {output_path}")
    return output_path


def visualize_selected_nodes_comparison(original_pickle_path, perturbed_dir,
                                        node_positions_path, output_dir,
                                        selected_nodes=None):
    """
    选择性可视化特定节点的扰动图对比

    Parameters:
    - original_pickle_path: 原始因果图pickle文件路径
    - perturbed_dir: 扰动因果图目录
    - node_positions_path: 节点位置文件路径
    - output_dir: 输出目录
    - selected_nodes: 要可视化的节点列表，如果为None则使用默认列表
    """
    # 设置默认选择的节点
    if selected_nodes is None:
        selected_nodes = [0, 1, 5, 10, 20]  # 可以根据需要调整

    print(f"选择性可视化节点: {selected_nodes}")

    # 调用对比可视化函数
    saved_files = visualize_comparison_for_nodes(
        original_pickle_path=original_pickle_path,
        perturbed_dir=perturbed_dir,
        node_positions_path=node_positions_path,
        output_dir=output_dir,
        nodes_to_compare=selected_nodes
    )

    return saved_files


def main():
    """
    主函数
    """
    # 配置路径
    dataset_dir = os.path.join(project_root, 'dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final')
    original_pickle_path = os.path.join(project_root, 'Module/PCMCI_/N100_T10000_Hubs15_Viz_Final/pcmci_results_tau3_alpha0p0001.pkl')
    perturbed_dir = os.path.join(project_root, 'dataset/Perturbed_Causal_Graphs_N100_T10000_Hubs15')
    node_positions_path = os.path.join(dataset_dir, 'node_positions.csv')
    output_dir = os.path.join(project_root, 'visualization/Perturbed_Causal_Graphs_N100_T10000_Hubs15_Perturbation_Comparison')

    # 检查必要文件
    if not os.path.exists(original_pickle_path):
        print(f"错误: 找不到原始因果图文件 {original_pickle_path}")
        return
    if not os.path.exists(perturbed_dir):
        print(f"错误: 找不到扰动因果图目录 {perturbed_dir}")
        return
    if not os.path.exists(node_positions_path):
        print(f"错误: 找不到节点位置文件 {node_positions_path}")
        return

    # 选择性可视化 - 可以根据需要修改要比较的节点
    # 例如：只可视化节点0, 5, 10
    selected_nodes = [12, 41, 7, 38, 11]  # 可以根据需要调整

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    saved_files = visualize_selected_nodes_comparison(
        original_pickle_path=original_pickle_path,
        perturbed_dir=perturbed_dir,
        node_positions_path=node_positions_path,
        output_dir=output_dir,
        selected_nodes=selected_nodes
    )

    print(f"选择性对比可视化完成！已保存到: {output_dir}")
    print(f"保存的文件列表: {saved_files}")


if __name__ == "__main__":
    main()
