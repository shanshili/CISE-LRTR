"""
概述
该文件提供了一个完整的因果分析流水线，用于为韧性预测模型准备输入数据。
通过整合PCMCI和SCMI算法的结果，为Resilience Prediction模型生成因果图特征。

函数说明

1. build_causal_analysis_pipeline(dataset_dir, selected_features, selected_positions, candidate_feature, candidate_position, pcmci_params=None, scmi_params=None)
功能: 构建因果分析流水线，为Resilience Prediction提供输入
参数:
  - dataset_dir: 数据集目录路径
  - selected_features: 已选节点特征列表
  - selected_positions: 已选节点位置列表
  - candidate_feature: 候选节点特征
  - candidate_position: 候选节点位置
  - pcmci_params: PCMCI算法参数（可选）
  - scmi_params: SCMI算法参数（可选）
返回: 包含处理后的因果图和相关数据的字典

2. prepare_cadgnn_input(causal_result)
功能: 准备CADGNN的输入数据格式
参数:
  - causal_result: 因果分析流水线的输出结果
返回: CADGNN可以直接使用的输入数据字典

3. run_resilience_causal_analysis(dataset_dir, selected_node_features, selected_node_positions, unselected_node_features, unselected_node_positions, pcmci_params=None, scmi_params=None)
功能: 为韧性预测运行完整的因果分析
参数:
  - dataset_dir: 数据集目录
  - selected_node_features: 已选节点特征列表
  - selected_node_positions: 已选节点位置列表
  - unselected_node_features: 候选节点特征列表
  - unselected_node_positions: 候选节点位置列表
  - pcmci_params: PCMCI算法参数（可选）
  - scmi_params: SCMI算法参数（可选）
返回: 每个候选节点对应的CADGNN输入数据列表

流水线流程：
1. 构建完整节点集（已选节点 + 候选节点）
2. 运行PCMCI分析（针对特定节点）
3. 运行SCMI分析（针对特定节点）
4. 处理PCMCI结果（过滤和展开矩阵）
5. 处理SCMI结果（展开矩阵）
6. 组合因果矩阵（标准化和加权求和）
7. 准备CADGNN输入格式

主要特点：
- 支持动态节点集分析
- 整合多种因果发现算法
- 为下游机器学习模型提供标准化输入
- 支持批量处理候选节点
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入必要的模块
from PCMCI_2 import run_pcmci_causal_analysis, load_causal_data, create_tigramite_dataframe
from SCMI2 import SCMIAlgorithm
from utils.Causal_dataset_process import (
    filter_pcmci_matrix_by_graph,
    expand_pcmci_matrix,
    expand_scmi_matrix,
    normalize_matrix_with_sign,
    normalize_matrix_positive_only,
    weighted_sum_matrices_normalized
)


def build_causal_analysis_pipeline(dataset_dir, selected_features, selected_positions,
                                   candidate_feature, candidate_position,
                                   pcmci_params=None, scmi_params=None):
    """
    构建因果分析流水线，为Resilience Prediction提供输入

    Parameters:
    dataset_dir (str): 数据集目录路径
    selected_features (list): 已选节点特征列表
    selected_positions (list): 已选节点位置列表
    candidate_feature (np.array): 候选节点特征
    candidate_position (np.array): 候选节点位置
    pcmci_params (dict): PCMCI参数
    scmi_params (dict): SCMI参数

    Returns:
    dict: 包含处理后的因果图和相关数据的字典
    """

    # 默认参数设置
    if pcmci_params is None:
        pcmci_params = {
            'tau_max': 3,
            'pc_alpha': 0.0001
        }

    if scmi_params is None:
        scmi_params = {
            'delta': 15.0,
            'k_neighbors': 5,
            'n_bins': 10,
            'threshold': 0.01
        }

    # 构建完整的节点集（已选节点 + 候选节点）
    all_features = selected_features + [candidate_feature]
    all_positions = np.vstack([selected_positions, candidate_position])
    n_nodes = len(all_features)

    print(f"构建包含 {n_nodes} 个节点的因果图")

    # 1. 运行PCMCI分析（针对特定节点）
    print("1. 运行PCMCI分析...")
    try:
        # 创建tigramite DataFrame
        dataframe = create_tigramite_dataframe(all_features, [f'Node_{i}' for i in range(n_nodes)])

        # 运行完整的PCMCI分析获取完整结果
        PCMCI_results = run_pcmci_causal_analysis(
            dataframe,
            tau_max=pcmci_params['tau_max'],
            pc_alpha=pcmci_params['pc_alpha']
        )

        # 提取我们关心的节点的子集结果
        pcmci_graph = PCMCI_results['graph']
        pcmci_val_matrix = PCMCI_results['val_matrix']

        print("PCMCI分析完成")
    except Exception as e:
        print(f"PCMCI分析失败: {e}")
        pcmci_graph, pcmci_val_matrix = None, None

    # 2. 运行SCMI分析（针对特定节点）
    print("2. 运行SCMI分析...")
    try:
        # 创建临时数据文件用于SCMI分析
        temp_data_path = os.path.join(dataset_dir, 'temp_causal_data.csv')
        temp_positions_path = os.path.join(dataset_dir, 'temp_node_positions.csv')

        # 准备时间序列数据
        # 将特征列表转换为时间序列格式 (时间步数 × 节点数)
        time_steps = len(all_features[0]) if len(all_features) > 0 and len(all_features[0]) > 0 else 0
        if time_steps > 0:
            temp_data = np.array(all_features).T  # 转置为 (时间步数 × 节点数)
            pd.DataFrame(temp_data, columns=[f'Node_{i}' for i in range(n_nodes)]).to_csv(
                temp_data_path, index=False)

            # 准备位置数据
            positions_df = pd.DataFrame(all_positions, columns=['x', 'y'])
            positions_df.to_csv(temp_positions_path, index=False)

            # 运行SCMI
            scmi = SCMIAlgorithm(
                data_path=temp_data_path,
                positions_path=temp_positions_path,
                delta=scmi_params['delta'],
                k_neighbors=scmi_params['k_neighbors'],
                n_bins=scmi_params['n_bins']
            )

            scmi_causal_graph = scmi.compute_causal_graph(threshold=scmi_params['threshold'])
            print("SCMI分析完成")

            # 清理临时文件
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            if os.path.exists(temp_positions_path):
                os.remove(temp_positions_path)
        else:
            scmi_causal_graph = None

    except Exception as e:
        print(f"SCMI分析失败: {e}")
        scmi_causal_graph = None

    # 3. 处理PCMCI结果
    pcmci_expanded = None
    if pcmci_val_matrix is not None and pcmci_graph is not None:
        print("3. 处理PCMCI结果...")
        try:
            # 过滤PCMCI矩阵
            filtered_pcmci = filter_pcmci_matrix_by_graph(pcmci_val_matrix, pcmci_graph)
            # 展开PCMCI矩阵
            pcmci_expanded = expand_pcmci_matrix(
                filtered_pcmci, n_nodes, pcmci_params['tau_max'])
            print(f"PCMCI展开矩阵形状: {pcmci_expanded.shape}")
        except Exception as e:
            print(f"处理PCMCI结果失败: {e}")
            pcmci_expanded = None

    # 4. 处理SCMI结果
    scmi_expanded = None
    if scmi_causal_graph is not None:
        print("4. 处理SCMI结果...")
        try:
            # 展开SCMI矩阵
            scmi_expanded = expand_scmi_matrix(
                scmi_causal_graph, n_nodes, scmi_params.get('tau_max', 3))
            print(f"SCMI展开矩阵形状: {scmi_expanded.shape}")
        except Exception as e:
            print(f"处理SCMI结果失败: {e}")
            scmi_expanded = None

    # 5. 组合结果
    combined_causal_matrix = None
    if pcmci_expanded is not None and scmi_expanded is not None:
        print("5. 组合因果矩阵...")
        try:
            # 加权求和标准化后的矩阵
            combined_causal_matrix, pcmci_normalized, scmi_normalized = weighted_sum_matrices_normalized(
                pcmci_expanded, scmi_expanded,
                pcmci_weight=0.5, scmi_weight=0.5
            )
            print(f"组合矩阵形状: {combined_causal_matrix.shape}")
        except Exception as e:
            print(f"组合因果矩阵失败: {e}")
            combined_causal_matrix = None
    elif pcmci_expanded is not None:
        print("5. 使用PCMCI结果...")
        try:
            combined_causal_matrix = normalize_matrix_with_sign(pcmci_expanded)
        except Exception as e:
            print(f"标准化PCMCI结果失败: {e}")
            combined_causal_matrix = None
    elif scmi_expanded is not None:
        print("5. 使用SCMI结果...")
        try:
            combined_causal_matrix = normalize_matrix_positive_only(scmi_expanded)
        except Exception as e:
            print(f"标准化SCMI结果失败: {e}")
            combined_causal_matrix = None

    # 6. 准备CADGNN输入格式
    result = {
        'causal_matrix': combined_causal_matrix,
        'n_nodes': n_nodes,
        'tau_max': pcmci_params['tau_max'] if pcmci_expanded is not None else scmi_params.get('tau_max', 3),
        'all_features': all_features,
        'all_positions': all_positions,
        'pcmci_available': pcmci_expanded is not None,
        'scmi_available': scmi_expanded is not None
    }

    return result


def prepare_cadgnn_input(causal_result):
    """
    准备CADGNN的输入数据格式

    Parameters:
    causal_result (dict): 因果分析流水线的输出结果

    Returns:
    dict: CADGNN可以直接使用的输入数据
    """
    if causal_result['causal_matrix'] is None:
        print("警告: 没有可用的因果矩阵")
        return None

    # 构建CADGNN输入格式
    cadgnn_input = {
        'adjacency_matrix': causal_result['causal_matrix'],
        'node_features': np.array(causal_result['all_features']),
        'node_positions': causal_result['all_positions'],
        'n_nodes': causal_result['n_nodes'],
        'tau_max': causal_result['tau_max']
    }

    return cadgnn_input


def run_resilience_causal_analysis(dataset_dir, selected_node_features, selected_node_positions,
                                   unselected_node_features, unselected_node_positions,
                                   pcmci_params=None, scmi_params=None):
    """
    为韧性预测运行完整的因果分析

    Parameters:
    dataset_dir (str): 数据集目录
    selected_node_features (list): 已选节点特征列表
    selected_node_positions (list): 已选节点位置列表
    unselected_node_features (list): 候选节点特征列表
    unselected_node_positions (list): 候选节点位置列表
    pcmci_params (dict): PCMCI参数
    scmi_params (dict): SCMI参数

    Returns:
    list: 每个候选节点对应的CADGNN输入数据列表
    """

    cadgnn_inputs = []

    print(f"开始为 {len(unselected_node_features)} 个候选节点进行因果分析...")

    # 为每个候选节点单独构建因果图
    for i, (candidate_feature, candidate_position) in enumerate(
            zip(unselected_node_features, unselected_node_positions)
    ):
        print(f"\n处理候选节点 {i + 1}/{len(unselected_node_features)}")
        print(f"已选节点数: {len(selected_node_features)}, 候选节点数: {len(unselected_node_features)}")

        # 运行因果分析流水线 - 每次只处理一个候选节点
        causal_result = build_causal_analysis_pipeline(
            dataset_dir=dataset_dir,
            selected_features=selected_node_features,  # 已选节点
            selected_positions=selected_node_positions,  # 已选节点位置
            candidate_feature=candidate_feature,  # 当前候选节点
            candidate_position=candidate_position,  # 当前候选节点位置
            pcmci_params=pcmci_params,
            scmi_params=scmi_params
        )

        # 准备CADGNN输入
        cadgnn_input = prepare_cadgnn_input(causal_result)
        cadgnn_inputs.append(cadgnn_input)

        print(f"候选节点 {i + 1} 处理完成")

    return cadgnn_inputs


# 使用示例
if __name__ == "__main__":
    # 示例使用方式
    print("因果分析流水线模块已准备就绪")
    print("请在 Resilience_prediction.py 中调用 run_resilience_causal_analysis 函数")

    # 示例参数设置
    example_pcmci_params = {
        'tau_max': 3,
        'pc_alpha': 0.0001
    }

    example_scmi_params = {
        'delta': 15.0,
        'k_neighbors': 5,
        'n_bins': 10,
        'threshold': 0.01,
        'tau_max': 3
    }

    print("\n推荐参数设置:")
    print("PCMCI参数:", example_pcmci_params)
    print("SCMI参数:", example_scmi_params)
