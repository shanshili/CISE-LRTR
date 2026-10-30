# generate_perturbation_corr_graphs.py
"""
该文件用于生成扰动相关性图，通过删除第N个节点的数据来构造扰动数据集，
然后计算扰动相关性图，并保存结果。
新增功能：支持随机扰动多个节点
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def compute_correlation_matrix(data):
    """
    计算相关性矩阵

    Parameters:
    - data: 数据矩阵 (T, N)

    Returns:
    - corr_matrix: 相关性矩阵 (N, N)
    - p_values: p值矩阵 (N, N)
    """
    T, N = data.shape
    corr_matrix = np.zeros((N, N))
    p_values = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_values[i, j] = 0.0
            else:
                corr, p_val = pearsonr(data[:, i], data[:, j])
                corr_matrix[i, j] = corr
                p_values[i, j] = p_val

    return corr_matrix, p_values


def generate_perturbation_corr_graphs(original_data_path, output_dir,
                                      num_nodes, threshold=0.5):
    """
    生成扰动相关性图 - 单节点扰动

    Parameters:
    - original_data_path: 原始数据文件路径
    - output_dir: 输出目录
    - num_nodes: 节点总数
    - threshold: 相关性阈值，用于创建稀疏图
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始数据
    data_df = pd.read_csv(original_data_path)  # shape: (T, N)
    print(f"原始数据形状: {data_df.shape}")

    # 创建临时目录用于保存扰动数据
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 为每个节点生成扰动图
    for i in range(num_nodes):
        print(f"处理节点 {i + 1}/{num_nodes} 的扰动...")

        # 删除第i个节点的数据
        perturbed_data = data_df.drop(columns=[data_df.columns[i]])
        perturbed_array = perturbed_data.values

        # 计算扰动后的相关性矩阵
        corr_matrix, p_values = compute_correlation_matrix(perturbed_array)

        # 应用阈值创建稀疏图
        corr_thresholded = np.abs(corr_matrix) > threshold
        graph = corr_matrix * corr_thresholded.astype(float)
        val_matrix = np.abs(corr_matrix)  # 使用绝对值作为边权重

        # 保存扰动相关性图
        perturbed_graph_path = os.path.join(output_dir, f'perturbed_corr_graph_node_{i}.npy')
        np.save(perturbed_graph_path, {
            'graph': graph,
            'val_matrix': val_matrix,
            'p_values': p_values,
            'threshold': threshold
        })

        print(f"节点 {i} 的扰动相关性图已保存: {perturbed_graph_path}")

    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

    print(f"所有单节点扰动相关性图已生成并保存到: {output_dir}")


def generate_multiple_perturbation_corr_graphs(original_data_path, output_dir,
                                               num_nodes, num_perturbations_list,
                                               num_samples=10, threshold=0.5):
    """
    生成多个节点扰动相关性图 - 随机扰动多个节点

    Parameters:
    - original_data_path: 原始数据文件路径
    - output_dir: 输出目录
    - num_nodes: 节点总数
    - num_perturbations_list: 要扰动的节点数量列表
    - num_samples: 每种扰动数量的样本数
    - threshold: 相关性阈值
    """
    # 创建输出目录
    multiple_output_dir = os.path.join(output_dir, 'multiple_perturbations_corr')
    os.makedirs(multiple_output_dir, exist_ok=True)

    # 读取原始数据
    data_df = pd.read_csv(original_data_path)  # shape: (T, N)
    print(f"原始数据形状: {data_df.shape}")

    # 为每个扰动数量生成多个样本
    for num_perturb in num_perturbations_list:
        print(f"开始生成扰动 {num_perturb} 个节点的相关性图样本...")

        for sample_idx in range(num_samples):
            print(f"  生成样本 {sample_idx + 1}/{num_samples} (扰动 {num_perturb} 个节点)...")

            # 随机选择要扰动的节点
            perturbed_nodes = random.sample(range(num_nodes), num_perturb)
            perturbed_nodes.sort()  # 排序以便于识别
            print(f"    随机选择的扰动节点: {perturbed_nodes}")

            # 删除选中的节点数据
            perturbed_data = data_df.copy()

            # 按降序删除列，避免索引变化影响后续删除
            for node_idx in sorted(perturbed_nodes, reverse=True):
                perturbed_data = perturbed_data.drop(columns=[perturbed_data.columns[node_idx]])

            # 转换为数组并计算相关性矩阵
            perturbed_array = perturbed_data.values
            corr_matrix, p_values = compute_correlation_matrix(perturbed_array)

            # 应用阈值创建稀疏图
            corr_thresholded = np.abs(corr_matrix) > threshold
            graph = corr_matrix * corr_thresholded.astype(float)
            val_matrix = np.abs(corr_matrix)  # 使用绝对值作为边权重

            # 保存扰动相关性图
            perturbed_graph_path = os.path.join(multiple_output_dir,
                                                f'perturbed_corr_graph_multiple_{num_perturb}_sample_{sample_idx}.npy')
            np.save(perturbed_graph_path, {
                'graph': graph,
                'val_matrix': val_matrix,
                'p_values': p_values,
                'threshold': threshold,
                'perturbed_nodes': perturbed_nodes  # 记录被扰动的节点
            })

            print(f"  扰动 {num_perturb} 个节点的样本 {sample_idx} 的相关性图已保存: {perturbed_graph_path}")

    print(f"多节点扰动相关性图已生成并保存到: {multiple_output_dir}")


def main():
    """
    主函数
    """
    # 配置参数
    dataset_dir = os.path.join(project_root, 'dataset/simulation_hubs_core/N500_T10000_Hubs40')
    data_path = os.path.join(dataset_dir, 'Env_A_timeseries.csv')

    output_dir = os.path.join(project_root, 'dataset/Perturbed_corr_Graphs_N500')

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return

    # 生成单节点扰动相关性图
    generate_perturbation_corr_graphs(
        original_data_path=data_path,
        output_dir=output_dir,
        num_nodes=500,  # 根据实际情况调整
        threshold=0.5
    )

    # 生成多节点扰动相关性图
    # generate_multiple_perturbation_corr_graphs(
    #     original_data_path=data_path,
    #     output_dir=output_dir,
    #     num_nodes=100,
    #     num_perturbations_list=[2, 3, 5],  # 要扰动的节点数量
    #     num_samples=5,  # 每种扰动数量的样本数
    #     threshold=0.5
    # )


if __name__ == "__main__":
    main()
