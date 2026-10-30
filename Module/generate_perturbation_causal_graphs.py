# generate_perturbation_causal_graphs.py
"""
该文件用于生成扰动因果图，通过删除第N个节点的数据来构造扰动数据集，
然后使用PCMCI算法生成扰动因果图，并保存结果。
新增功能：支持随机扰动多个节点
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PCMCI_2_pems import run_pcmci_causal_analysis
import random

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def generate_perturbation_graphs(original_data_path, original_positions_path,
                                 output_dir, num_nodes, tau_max=3, pc_alpha=0.0001):
    """
    生成扰动因果图 - 单节点扰动

    Parameters:
    - original_data_path: 原始数据文件路径
    - original_positions_path: 原始位置文件路径
    - output_dir: 输出目录
    - num_nodes: 节点总数
    - tau_max: 最大时间滞后
    - pc_alpha: 显著性阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取原始数据
    data_df = pd.read_csv(original_data_path)  # shape: (T, N)
    positions_df = pd.read_csv(original_positions_path)  # shape: (N, 2)

    print(f"原始数据形状: {data_df.shape}")
    print(f"原始位置数据形状: {positions_df.shape}")

    # 创建临时目录用于保存扰动数据
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 为每个节点生成扰动图
    for i in range(num_nodes):
        print(f"处理节点 {i + 1}/{num_nodes} 的扰动...")

        # 删除第i个节点的数据
        perturbed_data = data_df.drop(columns=[data_df.columns[i]])
        perturbed_positions = positions_df.drop(index=[i])

        # 重置索引
        perturbed_data = perturbed_data.reset_index(drop=True)
        perturbed_positions = perturbed_positions.reset_index(drop=True)

        # 创建临时文件路径
        temp_data_path = os.path.join(temp_dir, f'temp_data_node_{i}.csv')
        temp_positions_path = os.path.join(temp_dir, f'temp_positions_node_{i}.csv')

        # 保存临时扰动数据
        perturbed_data.to_csv(temp_data_path, index=False)
        perturbed_positions.to_csv(temp_positions_path, index=False)

        # 创建临时数据目录
        temp_dataset_dir = os.path.join(temp_dir, f'temp_dataset_{i}')
        os.makedirs(temp_dataset_dir, exist_ok=True)  # 修复：这里应该是 temp_dataset_dir

        # 为临时数据创建目录结构
        temp_simulated_data_path = os.path.join(temp_dataset_dir, 'id_standardize_Env_A_timeseries.csv')
        temp_node_positions_path = os.path.join(temp_dataset_dir, 'id_node_positions.csv')

        # 复制临时文件到模拟目录结构
        perturbed_data.to_csv(temp_simulated_data_path, index=False)
        perturbed_positions.to_csv(temp_node_positions_path, index=False)

        # 为扰动数据运行PCMCI分析
        try:
            graph, val_matrix, node_self_counts, edge_counts, pickle_file = run_pcmci_causal_analysis(
                dataset_dir=temp_dataset_dir,
                output_dir=os.path.join(output_dir, f'pcmci_results_node_{i}'),
                tau_max=tau_max,
                pc_alpha=pc_alpha
            )

            # 保存扰动因果图
            perturbed_graph_path = os.path.join(output_dir, f'perturbed_graph_node_{i}.npy')
            np.save(perturbed_graph_path, {
                'graph': graph,
                'val_matrix': val_matrix,
                'node_self_counts': node_self_counts,
                'edge_counts': edge_counts
            })

            print(f"节点 {i} 的扰动因果图已保存: {perturbed_graph_path}")

        except Exception as e:
            print(f"处理节点 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()

        # 清理临时文件
        try:
            os.remove(temp_data_path)
            os.remove(temp_positions_path)
            if os.path.exists(temp_simulated_data_path):
                os.remove(temp_simulated_data_path)
            if os.path.exists(temp_node_positions_path):
                os.remove(temp_node_positions_path)
            if os.path.exists(temp_dataset_dir):
                import shutil
                shutil.rmtree(temp_dataset_dir)  # 使用 shutil.rmtree 删除整个目录
        except:
            pass  # 忽略清理错误

    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

    print(f"所有单节点扰动因果图已生成并保存到: {output_dir}")


def generate_multiple_perturbation_graphs(original_data_path, original_positions_path,
                                          output_dir, num_nodes, num_perturbations_list,
                                          num_samples=10, tau_max=3, pc_alpha=0.0001):
    """
    生成多个节点扰动因果图 - 随机扰动多个节点

    Parameters:
    - original_data_path: 原始数据文件路径
    - original_positions_path: 原始位置文件路径
    - output_dir: 输出目录
    - num_nodes: 节点总数
    - num_perturbations_list: 要扰动的节点数量列表（例如 [2, 3, 5]）
    - num_samples: 每种扰动数量的样本数
    - tau_max: 最大时间滞后
    - pc_alpha: 显著性阈值

    num_perturbations_list=[2, 3, 5]
功能：定义要同时扰动的节点数量列表
含义：程序将分别生成扰动2个节点、3个节点和5个节点的因果图
用途：允许用户指定多种扰动强度，以便研究不同数量节点被移除时对因果关系的影响
num_samples=5
功能：指定每种扰动数量的样本数量
含义：对于每种扰动数量（如扰动2个节点），将随机生成5个不同的样本
用途：增加统计可靠性，因为每次随机选择的节点组合不同，可以更好地分析多节点扰动的模式
实际运行效果
当这些参数被使用时（假设在调用generate_multiple_perturbation_graphs函数时）：
扰动2个节点：生成5个不同的随机组合样本
扰动3个节点：生成5个不同的随机组合样本
扰动5个节点：生成5个不同的随机组合样本
总共会生成 3种扰动数量 × 5个样本 = 15 个不同的多节点扰动因果图，这样可以更全面地研究多节点扰动对因果关系网络的影响。
    """
    # 创建输出目录
    multiple_output_dir = os.path.join(output_dir, 'multiple_perturbations')
    os.makedirs(multiple_output_dir, exist_ok=True)

    # 读取原始数据
    data_df = pd.read_csv(original_data_path)  # shape: (T, N)
    positions_df = pd.read_csv(original_positions_path)  # shape: (N, 2)

    print(f"原始数据形状: {data_df.shape}")
    print(f"原始位置数据形状: {positions_df.shape}")

    # 创建临时目录用于保存扰动数据
    temp_dir = os.path.join(multiple_output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # 为每个扰动数量生成多个样本
    for num_perturb in num_perturbations_list:
        print(f"开始生成扰动 {num_perturb} 个节点的样本...")

        for sample_idx in range(num_samples):
            print(f"  生成样本 {sample_idx + 1}/{num_samples} (扰动 {num_perturb} 个节点)...")

            # 随机选择要扰动的节点
            perturbed_nodes = random.sample(range(num_nodes), num_perturb)
            perturbed_nodes.sort()  # 排序以便于识别
            print(f"    随机选择的扰动节点: {perturbed_nodes}")

            # 删除选中的节点数据
            perturbed_data = data_df.copy()
            perturbed_positions = positions_df.copy()

            # 按降序删除列和行，避免索引变化影响后续删除
            for node_idx in sorted(perturbed_nodes, reverse=True):
                perturbed_data = perturbed_data.drop(columns=[perturbed_data.columns[node_idx]])
                perturbed_positions = perturbed_positions.drop(index=[node_idx])

            # 重置索引
            perturbed_data = perturbed_data.reset_index(drop=True)
            perturbed_positions = perturbed_positions.reset_index(drop=True)

            # 创建临时文件路径
            temp_data_path = os.path.join(temp_dir, f'temp_data_multiple_{num_perturb}_sample_{sample_idx}.csv')
            temp_positions_path = os.path.join(temp_dir,
                                               f'temp_positions_multiple_{num_perturb}_sample_{sample_idx}.csv')

            # 保存临时扰动数据
            perturbed_data.to_csv(temp_data_path, index=False)
            perturbed_positions.to_csv(temp_positions_path, index=False)

            # 创建临时数据目录
            temp_dataset_dir = os.path.join(temp_dir, f'temp_dataset_multiple_{num_perturb}_sample_{sample_idx}')
            os.makedirs(temp_dataset_dir, exist_ok=True)

            # 为临时数据创建目录结构
            temp_simulated_data_path = os.path.join(temp_dataset_dir, 'simulated_data.csv')
            temp_node_positions_path = os.path.join(temp_dataset_dir, 'node_positions.csv')

            # 复制临时文件到模拟目录结构
            perturbed_data.to_csv(temp_simulated_data_path, index=False)
            perturbed_positions.to_csv(temp_node_positions_path, index=False)

            # 为扰动数据运行PCMCI分析
            try:
                graph, val_matrix, node_self_counts, edge_counts, pickle_file = run_pcmci_causal_analysis(
                    dataset_dir=temp_dataset_dir,
                    output_dir=os.path.join(multiple_output_dir,
                                            f'pcmci_results_multiple_{num_perturb}_sample_{sample_idx}'),
                    tau_max=tau_max,
                    pc_alpha=pc_alpha
                )

                # 保存扰动因果图
                perturbed_graph_path = os.path.join(multiple_output_dir,
                                                    f'perturbed_graph_multiple_{num_perturb}_sample_{sample_idx}.npy')
                np.save(perturbed_graph_path, {
                    'graph': graph,
                    'val_matrix': val_matrix,
                    'node_self_counts': node_self_counts,
                    'edge_counts': edge_counts,
                    'perturbed_nodes': perturbed_nodes  # 记录被扰动的节点
                })

                print(f"  扰动 {num_perturb} 个节点的样本 {sample_idx} 的扰动因果图已保存: {perturbed_graph_path}")

            except Exception as e:
                print(f"  处理扰动 {num_perturb} 个节点的样本 {sample_idx} 时出错: {e}")
                import traceback
                traceback.print_exc()

            # 清理临时文件
            try:
                os.remove(temp_data_path)
                os.remove(temp_positions_path)
                if os.path.exists(temp_simulated_data_path):
                    os.remove(temp_simulated_data_path)
                if os.path.exists(temp_node_positions_path):
                    os.remove(temp_node_positions_path)
                if os.path.exists(temp_dataset_dir):
                    import shutil
                    shutil.rmtree(temp_dataset_dir)
            except:
                pass  # 忽略清理错误

    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

    print(f"多节点扰动因果图已生成并保存到: {multiple_output_dir}")


def main():
    """
    主函数
    """
    # 配置参数
    dataset_dir = os.path.join(project_root, 'dataset/simulation_hubs_core/N1000_T10000_Hubs80')
    data_path = os.path.join(dataset_dir, 'Env_A_timeseries.csv')
    positions_path = os.path.join(dataset_dir, 'node_positions.csv')

    output_dir = os.path.join(project_root, 'dataset/Perturbed_Causal_Graphs_N1000_4')

    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        return
    if not os.path.exists(positions_path):
        print(f"错误: 找不到位置文件 {positions_path}")
        return

    # 生成单节点扰动因果图
    generate_perturbation_graphs(
        original_data_path=data_path,
        original_positions_path=positions_path,
        output_dir=output_dir,
        num_nodes=1000,  # 根据实际情况调整
        tau_max=3,
        pc_alpha=0.001
    )

    # 生成多节点扰动因果图
    # generate_multiple_perturbation_graphs(
    #     original_data_path=data_path,
    #     original_positions_path=positions_path,
    #     output_dir=output_dir,
    #     num_nodes=100,
    #     num_perturbations_list=[2, 3, 5],  # 要扰动的节点数量
    #     num_samples=5,  # 每种扰动数量的样本数
    #     tau_max=3,
    #     pc_alpha=0.0001
    # )


if __name__ == "__main__":
    main()
