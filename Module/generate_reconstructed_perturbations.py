import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 路径设置 ===
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# === 导入模块 ===
from PCMCI_2 import create_tigramite_dataframe, run_pcmci_analysis
from SCMI2 import create_scmi_algorithm, run_scmi_analysis
from utils.Causal_dataset_process import (
    expand_pcmci_matrix,
    expand_scmi_matrix,
    weighted_sum_matrices_normalized,
    filter_pcmci_matrix_by_graph,
    expand_features_for_causal_graph
)

# === 配置参数 ===
CONFIG = {
    'N': 100,
    'T': 10000,
    'tau_max': 3,
    'dataset_dir': os.path.join(project_root, 'dataset/Causal_simulation_data_2/N100_T10000_t3'),
    'output_dir': os.path.join(project_root, 'dataset/Perturbed_Reconstructed_N100'),
    # PCMCI 参数
    'pc_alpha': 0.0001,
    # SCMI 参数
    'scmi_delta': 15.0,
    'scmi_k': 5,
    'scmi_bins': 10,
    'scmi_threshold': 0.01,
    # 融合权重
    'pcmci_weight': 0.5,
    'scmi_weight': 0.5
}


def map_subset_to_global(subset_matrix, removed_idx, global_n, matrix_type='adj'):
    """
    将 N-1 x N-1 的矩阵映射回 N x N 的矩阵，被移除的行/列填0。

    Parameters:
    - subset_matrix: (N-1, N-1) 或 (N-1, N-1, tau)
    - removed_idx: 被移除的节点索引 (0 to N-1)
    - global_n: 原始节点数 N
    - matrix_type: 'adj' (2D) 或 'pcmci' (3D)
    """
    # 创建索引映射: subset_idx -> global_idx
    # 如果 subset_idx < removed_idx, global_idx = subset_idx
    # 如果 subset_idx >= removed_idx, global_idx = subset_idx + 1
    mapping = []
    curr = 0
    for i in range(global_n):
        if i == removed_idx:
            mapping.append(-1)  # 标记为被移除
        else:
            mapping.append(curr)
            curr += 1

    # 反向映射：0~N-2 -> 0~N-1 (跳过 removed_idx)
    reverse_map = [i for i in range(global_n) if i != removed_idx]

    if matrix_type == 'adj':
        # SCMI (2D)
        global_matrix = np.zeros((global_n, global_n))
        for r_sub in range(global_n - 1):
            for c_sub in range(global_n - 1):
                r_glob = reverse_map[r_sub]
                c_glob = reverse_map[c_sub]
                global_matrix[r_glob, c_glob] = subset_matrix[r_sub, c_sub]
        return global_matrix

    elif matrix_type == 'pcmci':
        # PCMCI (3D: target, source, tau)
        tau_dim = subset_matrix.shape[2]
        global_matrix = np.zeros((global_n, global_n, tau_dim), dtype=subset_matrix.dtype)
        # 如果是字符串矩阵(graph)，初始化为空字符串或其他占位符
        if subset_matrix.dtype.kind in {'U', 'S'}:
            global_matrix[:] = ""

        for r_sub in range(global_n - 1):
            for c_sub in range(global_n - 1):
                r_glob = reverse_map[r_sub]
                c_glob = reverse_map[c_sub]
                global_matrix[r_glob, c_glob, :] = subset_matrix[r_sub, c_sub, :]
        return global_matrix


def run_reconstruction():
    # 1. 初始化
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    vis_dir = os.path.join(CONFIG['output_dir'], 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # 2. 加载原始全量数据
    data_path = os.path.join(CONFIG['dataset_dir'], 'simulated_data.csv')
    pos_path = os.path.join(CONFIG['dataset_dir'], 'node_positions.csv')

    df_data = pd.read_csv(data_path)  # (T, N)
    df_pos = pd.read_csv(pos_path)  # (N, 2)

    data_values = df_data.values  # (T, N)
    pos_values = df_pos.values  # (N, 2)

    # 保存特征数据的扩展版本 (Features)
    # 因为特征是不变的（除了被Mask掉），我们只需要生成一次全量的扩展特征，后续训练时动态mask即可
    # 或者为了方便，我们也可以生成N个特征文件，每个文件对应的列全为0
    print("Pre-processing features...")
    features_expanded_orig = expand_features_for_causal_graph(
        data_values.T,  # 需要 (N, T)
        causal_nodes=400,
        original_nodes=100,
        tau_max=CONFIG['tau_max']
    )
    np.save(os.path.join(CONFIG['output_dir'], 'features_expanded_original.npy'), features_expanded_orig)

    # 3. 开始循环扰动
    total_nodes = CONFIG['N']

    # 创建临时文件夹用于存放SCMI需要的中间文件
    temp_dir = os.path.join(CONFIG['output_dir'], 'temp_files')
    os.makedirs(temp_dir, exist_ok=True)
    temp_data_path = os.path.join(temp_dir, 'temp_data.csv')
    temp_pos_path = os.path.join(temp_dir, 'temp_pos.csv')

    print(f"Starting reconstruction for {total_nodes} perturbations...")

    for i in tqdm(range(total_nodes), desc="Node Perturbation"):
        try:
            # === Step A: 构造子集数据 ===
            # 移除第 i 列数据
            subset_data = np.delete(data_values, i, axis=1)  # (T, N-1)
            # 移除第 i 行位置
            subset_pos = np.delete(pos_values, i, axis=0)  # (N-1, 2)

            subset_node_names = [f"Node_{k}" for k in range(total_nodes) if k != i]
            N_subset = total_nodes - 1

            # === Step B: 运行 PCMCI+ ===
            # 创建 tigramite dataframe
            dataframe = create_tigramite_dataframe(subset_data, subset_node_names)
            # 运行分析
            pcmci_results = run_pcmci_analysis(
                dataframe,
                tau_max=CONFIG['tau_max'],
                pc_alpha=CONFIG['pc_alpha']
            )
            # 提取结果 (N-1, N-1, tau)
            val_matrix_sub = pcmci_results['val_matrix']
            graph_sub = pcmci_results['graph']

            # === Step C: 运行 SCMI ===
            # SCMI需要读取文件，所以保存临时CSV
            pd.DataFrame(subset_data, columns=subset_node_names).to_csv(temp_data_path, index=False)
            pd.DataFrame(subset_pos, columns=['x', 'y']).to_csv(temp_pos_path, index=False)

            scmi_instance = create_scmi_algorithm(
                temp_data_path,
                temp_pos_path,
                delta=CONFIG['scmi_delta'],
                k_neighbors=CONFIG['scmi_k'],
                n_bins=CONFIG['scmi_bins']
            )
            # 计算因果图 (N-1, N-1)
            scmi_adj_sub = run_scmi_analysis(scmi_instance, threshold=CONFIG['scmi_threshold'])

            # === Step D: 映射回全局维度 (Restore to 100x100) ===
            # 1. 恢复 PCMCI
            global_val_matrix = map_subset_to_global(val_matrix_sub, i, total_nodes, 'pcmci')
            global_graph = map_subset_to_global(graph_sub, i, total_nodes, 'pcmci')

            # 2. 恢复 SCMI
            global_scmi_adj = map_subset_to_global(scmi_adj_sub, i, total_nodes, 'adj')

            # === Step E: 过滤、扩展与融合 ===
            # 1. 过滤 PCMCI (仅保留显著边)
            filtered_pcmci = filter_pcmci_matrix_by_graph(global_val_matrix, global_graph)

            # 2. 扩展矩阵 (100 -> 400)
            # 注意：expand 函数接受的是 (N, N, tau) 或 (N, N)
            pcmci_expanded = expand_pcmci_matrix(filtered_pcmci, total_nodes, CONFIG['tau_max'])
            scmi_expanded = expand_scmi_matrix(global_scmi_adj, total_nodes, CONFIG['tau_max'])

            # 3. 加权融合
            combined_matrix, _, _ = weighted_sum_matrices_normalized(
                pcmci_expanded,
                scmi_expanded,
                pcmci_weight=CONFIG['pcmci_weight'],
                scmi_weight=CONFIG['scmi_weight']
            )

            # === Step F: 保存结果 ===
            save_name = f'pert_node_{i}.npy'
            np.save(os.path.join(CONFIG['output_dir'], save_name), combined_matrix)

            # === Step G: 可视化 (可选，前几个) ===
            if i < 3:
                plt.figure(figsize=(10, 8))
                plt.imshow(combined_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                plt.colorbar(label='Causal Strength')
                plt.title(f'Reconstructed Causal Graph (Removed Node {i})')
                plt.xlabel('Target Node (Expanded)')
                plt.ylabel('Source Node (Expanded)')
                plt.savefig(os.path.join(vis_dir, f'heatmap_pert_{i}.png'))
                plt.close()

        except Exception as e:
            print(f"Error processing perturbation for node {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 清理临时文件
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("All perturbations generated successfully.")


if __name__ == "__main__":
    run_reconstruction()
