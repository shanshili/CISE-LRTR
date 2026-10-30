"""
概述
该文件实现了扰动图生成功能，用于评估节点在因果图中的重要性。
通过对基准图进行节点隔离扰动，生成一系列扰动图用于下游分析。

函数说明

1. Read_data_CSV(dataset_in_project_dir)
功能: 读取基准图数据（保留函数，实际已改用npy文件读取）
参数: dataset_in_project_dir - 数据集在项目中的目录
返回: (None, None) - 占位符，实际数据从npy文件读取

2. memory_over(tem_data, data_location, node, data_num)
功能: 处理扩展后的特征数据，限制内存使用
参数:
  - tem_data: 原始特征数据
  - data_location: 位置信息（不再需要）
  - node: 节点数
  - data_num: 时间步数
返回: (fea_ori, None) - 处理后的特征数据和空位置信息

3. Construct_base_graph_from_adj(causal_weight)
功能: 从权重邻接矩阵构造图
参数: causal_weight - 因果图权重矩阵（浮点数）
返回: (G, causal_weight) - 图对象和权重矩阵

4. Construct_perturbation_graph(G_0, fea_ori)
功能: 构造扰动图（保留权重信息）
参数:
  - G_0: 基准图对象
  - fea_ori: 原始特征数据
返回: (perturbed_weight_set, perturbed_graph_label, perturbed_graph_fea)

5. load_causal_data_and_features()
功能: 加载因果图邻接矩阵和特征数据
返回: (causal_weight, features_expanded) - 因果权重矩阵和扩展后的特征

算法流程：
1. 加载因果图权重矩阵和特征数据
2. 使用expand_features_for_causal_graph扩展特征
3. 从权重矩阵构造基准图
4. 对每个节点执行扰动操作：
   a. 创建基准图副本
   b. 删除与目标节点相连的所有边
   c. 生成权重矩阵（保留浮点数权重）
   d. 计算节点重要性标签（加权度/节点总数）
   e. 掩蔽目标节点特征
5. 保存所有扰动图和相关信息

主要特点：
- 保留因果图的权重信息
- 支持扩展后的特征数据格式
- 生成带重要性标签的扰动图
- 适用于因果图的重要性评估

应用场景：
- 节点重要性分析
- 因果图的鲁棒性评估
- 图神经网络的扰动训练
- 因果结构敏感性分析

"""
import numpy as np
import networkx as nx
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.feature_expansion import expand_features_for_causal_graph, verify_temporal_alignment
from utils.get_data_file import get_project_root,find_causal_adj_file,find_corr_adj_file,find_features_file
import pandas as pd
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))



def Read_data_CSV(dataset_in_project_dir):
    # 读取基准图数据 并打包为文件
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    Path1 = project_dir + dataset_in_project_dir
    return None, None  # 占位符，因为我们现在从 npy 文件读取


def memory_over(tem_data, data_location, node, data_num):
    # 这个函数现在将处理扩展后的特征
    if tem_data is not None:
        tem = tem_data[:node, :data_num]
        # location = data_location[:node,:]  # 位置信息不再需要
        fea_ori = tem.astype(np.float32)
        return fea_ori, None  # 不再需要位置信息
    return None, None


# def Construct_base_graph_from_adj(adj_matrix):
#     """
#     从邻接矩阵构造图
#     """
#     G = nx.from_numpy_array(adj_matrix)
#     print("G_0.number_of_nodes:", G.number_of_nodes())  # 节点数：
#     print("G_0.number_of_edges:", G.number_of_edges())  # 边数：
#     return G, adj_matrix

def Construct_base_graph_from_adj(causal_weight):
    """
    从权重邻接矩阵构造图
    """
    # 确保输入是浮点数矩阵
    causal_weight = causal_weight.astype(np.float32)

    # 使用浮点数邻接矩阵构造图（保留权重）
    G = nx.from_numpy_array(causal_weight, create_using=nx.Graph())

    # 添加权重检查逻辑
    edge_weights = [G[u][v].get('weight', 0.0) for u, v in G.edges()]
    print(f"Constructed graph with weight range: {min(edge_weights):.4f} - {max(edge_weights):.4f}")

    return G, causal_weight


# def Construct_perturbation_graph(G_0, fea_ori):
#     # 对所有节点执行
#     # perturbed_edge_set = []
#     perturbed_adj_set = []
#     perturbed_graph_label = []
#     perturbed_graph_fea = []
#     for i in range(G_0.number_of_nodes()):  # 450 node
#
#         # 找到与中心节点相邻的所有边
#         edge_list_0 = G_0.edges(i)
#         # print("edge_list_0:\n",edge_list_0)
#
#         # copy基准图（防止直接删了基准图上的边）
#         G = G_0.copy()
#         # 删除与中心节点相邻的所有边
#         G.remove_edges_from(edge_list_0)
#
#         # copy原始特征
#         if fea_ori is not None:
#             fea = fea_ori.copy()
#
#             # 判断中心节点是否孤立
#             if (nx.is_isolate(G, i) == True):
#                 # edge_index = G.edges
#                 # perturbed_edge_set.append(edge_index)
#                 adj_perturbed = nx.to_pandas_adjacency(G)  # 已孤立，则汇总起来邻接矩阵
#                 perturbed_adj_set.append(adj_perturbed)
#                 # 获取标签 节点度/总节点数 （重要性特征？？）
#                 perturbed_graph_label.append(G_0.degree[i] / G_0.number_of_nodes())
#
#                 # 掩蔽特征
#                 fea[i] = 1e-20
#                 perturbed_graph_fea.append(fea)  # 掩蔽点特征为0
#
#             else:
#                 print('node {} is not isolated'.format(i))
#
#     print("perturbed_adj_set", len(perturbed_adj_set))
#     print("perturbed_graph_label", len(perturbed_graph_label))
#     print("perturbed_graph_fea", len(perturbed_graph_fea))
#
#     return perturbed_adj_set, perturbed_graph_label, perturbed_graph_fea

def Construct_perturbation_graph(G_0, fea_ori):
    """
    构造扰动图（保留权重信息）
    """
    perturbed_weight_set = []
    perturbed_graph_label = []
    perturbed_graph_fea = []

    for i in range(G_0.number_of_nodes()):
        # 创建扰动图副本
        G = G_0.copy()
        edge_list_0 = list(G.edges(i))

        # 删除与节点i相连的边
        G.remove_edges_from(edge_list_0)

        # 保留权重信息（使用浮点数邻接矩阵）
        if edge_list_0:
            # 使用浮点数邻接矩阵（保留权重）
            weight_matrix = nx.adjacency_matrix(G, weight='weight').toarray().astype(np.float32)
        else:
            # 孤立节点的权重矩阵全0
            weight_matrix = np.zeros((G_0.number_of_nodes(), G_0.number_of_nodes()), dtype=np.float32)


        # 存储扰动图权重矩阵
        perturbed_weight_set.append(weight_matrix)
        perturbed_graph_label.append(G_0.degree(i, weight='weight') / G_0.number_of_nodes())

        # 特征掩蔽
        fea = fea_ori.copy()
        fea[i] = 1e-20
        perturbed_graph_fea.append(fea)

    print("perturbed_weight_set", len(perturbed_weight_set))
    print("perturbed_graph_label", len(perturbed_graph_label))
    print("perturbed_graph_fea", len(perturbed_graph_fea))

    # 在扰动图生成后检查权重
    for i, weight_matrix in enumerate(perturbed_weight_set):
        print(f"Weight matrix {i} - min={np.min(weight_matrix):.4f}, max={np.max(weight_matrix):.4f}")
        if np.allclose(weight_matrix, weight_matrix.astype(bool).astype(np.float32)):
            print("  Warning: Binary weight matrix, weights are missing")

    return perturbed_weight_set, perturbed_graph_label, perturbed_graph_fea



def load_causal_data_and_features():
    """
    加载因果图邻接矩阵和特征数据
    """
    # 获取项目根目录
    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # 配置参数
    params = {
        'N': 100,
        'T': 10000,
        't': 3,
        'Hubs':15
    }

    try:
        # 查找因果图权重文件
        causal_weight_path = find_causal_adj_file(project_root, params)
        print(f"Found causal weight matrix at: {causal_weight_path}")

        # 查找特征文件
        features_path = find_features_file(project_root, params)
        print(f"Found features file at: {features_path}")

        # 加载因果图权重矩阵（浮点数）
        causal_weight = np.load(causal_weight_path)
        print(f"Causal weight matrix shape: {causal_weight.shape}")

        # 加载特征数据 (CSV格式)
        features_df = pd.read_csv(features_path, header=0, low_memory=False)

        # 提取数值数据
        features_values = features_df.iloc[:, :].values.T

        # 处理特征形状，确保是 (N, T) 格式，其中N是节点数，T是时间步数
        try:
            features = features_values.astype(np.float32)
        except ValueError:
            features = np.zeros(features_values.shape, dtype=np.float32)
            for i in range(features_values.shape[0]):
                for j in range(features_values.shape[1]):
                    try:
                        features[i, j] = float(features_values[i, j])
                    except ValueError:
                        features[i, j] = 0.0

        print(f"Original features shape: {features.shape}")

        # 使用 expand_features_for_causal_graph 扩展特征
        features_expanded = expand_features_for_causal_graph(
            features,
            causal_nodes=400,
            original_nodes=100,
            tau_max=3
        )

        print(f"Features shape after expansion: {features_expanded.shape}")

        return causal_weight, features_expanded

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the data files exist in the expected directories.")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 主执行部分
if __name__ == "__main__":
    # 加载因果图数据和特征
    causal_weight, features_expanded = load_causal_data_and_features()

    if causal_weight is not None and features_expanded is not None:
        # 构造基准图
        G_0, adj_matrix = Construct_base_graph_from_adj(causal_weight)  # adj_matrix 仍用于边索引

        # 使用扩展后的特征
        fea_ori = features_expanded
        print(f"Using expanded features with shape: {fea_ori.shape}")

        # 构造扰动图
        perturbed_weight_set, perturbed_graph_label, perturbed_graph_fea = Construct_perturbation_graph(G_0, fea_ori)

        '''
        保存到文件
        '''
        # 保存数据
        node = causal_weight.shape[0]
        data_num = features_expanded.shape[1]

        os.makedirs('./perturbation_data', exist_ok=True)

        # 保存权重矩阵文件
        np.savez(f'./perturbation_data/origin_data_weight_ori_node_{node}_data_{data_num}.npz', causal_weight)
        np.savez(f'./perturbation_data/origin_data_fea_ori_node_{node}_data_{data_num}.npz', features_expanded)

        # 保存扰动图权重矩阵
        np.savez(f'./perturbation_data/perturbed_graph_weight_node_{node}_data_{data_num}.npz', perturbed_weight_set)
        np.savez(f'./perturbation_data/perturbed_label_node_{node}_data_{data_num}.npz', perturbed_graph_label)
        np.savez(f'./perturbation_data/perturbed_fea_node_{node}_data_{data_num}.npz', perturbed_graph_fea)

        print("Data saved successfully!")
    else:
        print("Failed to load data. Exiting.")
