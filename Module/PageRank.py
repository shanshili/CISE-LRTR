"""
PageRank.py - 基于时空因果关系的加权PageRank算法实现

本代码实现了用于因果推断场景的加权PageRank算法，主要功能包括：

1. **数据预处理**：
   - 从PCMCI和SCMI结果文件中加载时空因果关系数据
   - 使用utils.Causal_dataset_process中的标准化方法处理因果矩阵
   - 读取节点位置信息和相似度权重

2. **图构建**：
   - 构建包含时空因果关系的有向多重图
   - 支持时间滞后因果边（来自PCMCI）和空间因果边（来自SCMI）
   - 集成节点位置信息用于可视化

3. **加权PageRank算法**：
   - 实现带权重的PageRank算法(WeightedPageRank)
   - 在转移概率矩阵中集成外部权重信息
   - 提供标准PageRank算法对比

4. **权重聚合**：
   - 将扩展节点(400个)的贡献度聚合为原始节点(100个)
   - 生成节点重要性排序列表

5. **可视化功能**：
   - 提供多种可视化方法展示因果关系和节点权重
   - 包含散点图、网络图等多种展示方式
   - 支持因果效应的颜色映射和边类型区分

6. **结果输出**：
   - 生成优化节点子集
   - 保存节点排序结果和可视化图表
   - 提供详细的权重分析报告

参数配置：
- Gpn: 图编号(62)
- node: 扩展节点数(400)
- original_node: 原始节点数(100)
- data_num: 数据量(9997)
- select_node: 选择节点数(18)
"""

import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 找到项目的根目录（假设 utils 和 MGC-RM 是同级目录）
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录加入 sys.path
sys.path.append(project_root)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime
import pickle
# 在文件开头的导入部分添加
sys.path.append(project_root)
from utils.Causal_dataset_process import filter_pcmci_matrix_by_graph, normalize_matrix_with_sign, normalize_matrix_positive_only


# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式：年月日_时分秒

# 全局修改字体
config = {
    "font.family": 'Times New Roman',
    "font.size": 10.5,
    # "mathtext.fontset": 'stix',
    # "font.serif": ['SimSun'],
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

# 参数设置
Gpn = 62
node = 400  # 扩展后的节点数
original_node = 100  # 原始节点数
data_num = 9997
select_node = 18
# 在参数设置部分后添加PCMCI和SCMI文件路径
pcmci_file_path = os.path.join(current_dir, 'PCMCI_/N100_T10000_t3/pcmci_results_tau3_alpha0p0001.pkl')
scmi_file_path = os.path.join(current_dir, 'SCMI_N100_T10000_t3_delta15.0_k5_bins10/scmi_causal_graph_delta15.0_k5_bins10_threshold0.01.npy')
weight_ori = np.load('perturbation_data/origin_data_weight_ori' + '_node_' + str(node) + '_data_' + str(data_num) + '.npz', allow_pickle=True)
weight_matrix = weight_ori['arr_0']

# 读取相似度分数文件
weights = []
similarity_file = os.path.join(current_dir, 'MGC-RM_result', 'similarity score',
                               'zGP_62_node_400_data_9997model_20250907_153259.txt')

# 检查文件是否存在
if os.path.exists(similarity_file):
    with open(similarity_file, 'r') as f:
        for line in f:
            weights.append(float(list(line.strip('\n').split(','))[0]))
else:
    print(f"警告：文件 {similarity_file} 不存在，使用默认权重")
    weights = np.ones(node) / node

weights = np.array(weights)
weights = weights / np.sum(weights)  # 归一化

# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/Causal_simulation_data_2/N100_T10000_t3')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)

position_file = os.path.join(dataset_dir, 'node_positions.csv')
positions_df = pd.read_csv(position_file)
positions = positions_df[['x', 'y']].values
lon = positions[:, 0]
lat = positions[:, 1]


# 创建一个简单的图用于可视化（基于节点位置）
def create_simple_graph_from_positions(positions):
    """根据节点位置创建一个简单的图"""
    G = nx.Graph()
    n_nodes = len(positions)

    # 添加节点
    for i in range(n_nodes):
        G.add_node(i)

    # 简单的连接策略：连接距离较近的节点
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # 计算节点间距离
            dist = np.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                           (positions[i][1] - positions[j][1]) ** 2)
            # 如果距离小于某个阈值，则连接
            if dist < 0.2:  # 阈值可以根据需要调整
                G.add_edge(i, j, weight=1.0 / dist if dist > 0 else 1.0)

    return G


def create_causal_graph_from_matrices(pcmci_path, scmi_path, positions, n_nodes=100, tau_max=3):
    """
    根据PCMCI和SCMI因果矩阵构建时空因果图，并使用utils.Causal_dataset_process中的标准化方法

    Parameters:
    - pcmci_path: PCMCI结果文件路径
    - scmi_path: SCMI结果文件路径
    - positions: 节点位置数组
    - n_nodes: 节点数
    - tau_max: 最大时间延迟

    Returns:
    - G: 构建的因果图 (包含时空因果关系)
    """
    # 加载PCMCI结果
    with open(pcmci_path, 'rb') as f:
        pcmci_results = pickle.load(f)
    pcmci_graph = pcmci_results['graph']
    pcmci_val_matrix = pcmci_results['val_matrix']

    # 使用utils.Causal_dataset_process中的filter_pcmci_matrix_by_graph函数过滤PCMCI结果
    filtered_pcmci_val_matrix = filter_pcmci_matrix_by_graph(pcmci_val_matrix, pcmci_graph)

    # 使用utils.Causal_dataset_process中的normalize_matrix_with_sign函数标准化PCMCI结果
    normalized_pcmci = normalize_matrix_with_sign(filtered_pcmci_val_matrix)

    # 加载SCMI结果
    scmi_matrix = np.load(scmi_path)

    # 使用utils.Causal_dataset_process中的normalize_matrix_positive_only函数标准化SCMI结果
    normalized_scmi = normalize_matrix_positive_only(scmi_matrix)

    # 创建MultiDiGraph以支持多重边和自循环
    G = nx.MultiDiGraph()

    # 添加节点并设置位置信息
    for i in range(n_nodes):
        G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

    # 添加时间因果边 (来自标准化后的PCMCI)
    for target in range(n_nodes):
        for source in range(n_nodes):
            for tau in range(1, tau_max + 1):
                if pcmci_graph[target, source, tau] == '-->':
                    causal_effect = normalized_pcmci[target, source, tau]
                    weight = abs(causal_effect)
                    # 添加时间因果边: source -> target (带时间滞后)
                    G.add_edge(source, target, key=f'time_lag_{tau}',
                               weight=weight, lag=tau, causal_effect=causal_effect,
                               edge_type='temporal')

    # 添加空间因果边 (来自标准化后的SCMI)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if scmi_matrix[i, j] > 0:  # 如果存在空间因果关系
                # 添加无向空间因果边 (使用两个方向表示无向)
                causal_effect = normalized_scmi[i, j]
                G.add_edge(i, j, key='spatial', weight=causal_effect,
                           edge_type='spatial', lag=0, causal_effect=causal_effect)
                G.add_edge(j, i, key='spatial', weight=causal_effect,
                           edge_type='spatial', lag=0, causal_effect=causal_effect)

    return G


# # 创建用于可视化的图
# G_viz = create_simple_graph_from_positions(positions)
# 替换原来的图构建代码
# 创建用于可视化的因果图 (替换原来的 create_simple_graph_from_positions 调用)
if os.path.exists(pcmci_file_path) and os.path.exists(scmi_file_path):
    G_causal = create_causal_graph_from_matrices(pcmci_file_path, scmi_file_path, positions)
    print(f"因果图构建完成: {G_causal.number_of_nodes()} 节点, {G_causal.number_of_edges()} 边")
else:
    print("警告: 无法找到因果数据文件，使用简单图代替")
    G_causal = create_simple_graph_from_positions(positions)

def WeightedPageRank(M, weights, N, T=300, eps=1e-6, beta=0.85):
    """
    R : rank 每次迭代
    N : 节点个数
    M : 概率转移矩阵
    beta : 阻尼因子，控制随机跳转的概率
    T : 迭代次数
    eps ： 迭代收敛限度
    """
    R = np.ones(N) / N  # 向量
    teleport = np.ones(N) / N
    for time in range(T):
        R_new = beta * np.dot(M, weights * R) + (1 - beta) * teleport   #weights * R 扩展矩阵 且1/N
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new


def PageRank(M, N, T=300, eps=1e-6, beta=0.85):
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for time in range(T):
        R_new = beta * np.dot(M, R) + (1 - beta) * teleport
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new


# def data_color_graph(score, locationgraph, location, color, labels):
#     norm1 = mcolors.Normalize(vmin=np.min(score), vmax=np.max(score))
#     fig, ax = plt.subplots()
#     plt.title('Contribution weights')
#     nx.draw(locationgraph, pos=location, with_labels=labels, alpha=0.8, node_size=8,
#             node_color=score, cmap=color, width=0.8, edge_color='#9585bd', font_size=0)
#
#     ax = plt.gca()
#     ax.set_axis_on()
#     ax.spines['bottom'].set_color('0')
#     ax.spines['top'].set_color('0')
#     ax.spines['right'].set_color('0')
#     ax.spines['left'].set_color('0')
#     ax.tick_params(which='both', width=1, length=1, colors='black')
#
#     sm = cm.ScalarMappable(cmap=color, norm=norm1)
#     sm.set_array([])
#     plt.colorbar(sm, ax=ax)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.savefig('WPR_result/'+timestamp+'.svg', format='svg')
#     plt.show()


# 创建一个随机的概率转移矩阵 (使用400节点)


# def data_color_graph(score, locationgraph, location, color, labels):
#     """
#     绘制带有时空因果关系的图
#     """
#     norm1 = mcolors.Normalize(vmin=np.min(score), vmax=np.max(score))
#     fig, ax = plt.subplots(figsize=(14, 12))
#     plt.title('Contribution weights with Causal Relationships')
#
#     # 获取节点位置
#     pos = {i: (location[i, 0], location[i, 1]) for i in range(len(location))}
#
#     # 绘制节点，使用黄色系颜色映射
#     nodes = nx.draw_networkx_nodes(locationgraph, pos, node_size=80,
#                                    node_color=score, cmap='YlGn', alpha=0.8)  # 改为黄色系
#
#     # 绘制节点编号
#     node_labels = {i: str(i) for i in range(len(location))}
#     nx.draw_networkx_labels(locationgraph, pos, labels=node_labels, font_size=6)
#
#     # 分离不同类型和时间滞后的边
#     temporal_edges = []
#     spatial_edges = []
#
#     for source, target, key, data in locationgraph.edges(keys=True, data=True):
#         if data.get('edge_type') == 'temporal':
#             temporal_edges.append((source, target, key, data))
#         elif data.get('edge_type') == 'spatial':
#             spatial_edges.append((source, target, key, data))
#
#     # 获取所有因果效应值用于颜色映射
#     all_effects = [data['causal_effect'] for _, _, _, data in temporal_edges]
#     all_effects.extend([data['causal_effect'] for _, _, _, data in spatial_edges])
#
#     if all_effects:
#         min_effect = min(all_effects)
#         max_effect = max(all_effects)
#         abs_max = max(abs(min_effect), abs(max_effect))
#         vmin, vmax = -abs_max, abs_max
#     else:
#         vmin, vmax = -1, 1
#
#     # 创建颜色映射
#     cmap = plt.cm.coolwarm  # 红蓝表示正负因果效应
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
#
#     # 绘制时间因果边 (带箭头)
#     for source, target, key, data in temporal_edges:
#         causal_effect = data['causal_effect']
#         edge_color = cmap(norm(causal_effect))
#         width = 0.5 + abs(causal_effect) * 3  # 根据效应强度调整宽度
#         lag = data.get('lag', 0)
#
#         # 绘制带箭头的边
#         nx.draw_networkx_edges(locationgraph, pos,
#                                edgelist=[(source, target, key)],
#                                width=width, edge_color=[edge_color],
#                                alpha=0.7, arrows=True, arrowsize=10,
#                                arrowstyle='->')
#
#         # 添加时间滞后标签（调整位置和大小以避免覆盖节点）
#         if lag > 0:
#             # 计算边的中点位置
#             x1, y1 = pos[source]
#             x2, y2 = pos[target]
#             mid_x = (x1 + x2) / 2
#             mid_y = (y1 + y2) / 2
#
#             # 对于自环边，调整标签位置避免与节点重叠和节点编号
#             if source == target:
#                 # 自环边标签位置调整到节点左侧，避免挡住节点编号
#                 mid_x = x1 - 1.2  # 向左偏移
#                 mid_y = y1  # 稍微向上偏移
#
#             # 添加滞后标签，调整字体大小
#             plt.text(mid_x, mid_y, f'τ{lag}',
#                      fontsize=5, ha='center', va='center',
#                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
#
#     # 绘制空间因果边 (无箭头，表示无向)
#     for source, target, key, data in spatial_edges:
#         causal_effect = data['causal_effect']
#         edge_color = cmap(norm(causal_effect))
#         width = 0.3 + abs(causal_effect) * 2  # 根据效应强度调整宽度
#
#         # 绘制无向边
#         nx.draw_networkx_edges(locationgraph, pos,
#                                edgelist=[(source, target, key)],
#                                width=width, edge_color=[edge_color],
#                                alpha=0.5, arrows=False)
#
#     # 添加颜色条 (因果效应)
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
#     cbar.set_label('Causal Effect', rotation=270, labelpad=20)
#
#     # 添加节点权重颜色条（使用黄色系）
#     sm_nodes = cm.ScalarMappable(norm=norm1, cmap='YlGn')  # 改为黄色系
#     sm_nodes.set_array([])
#     cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.8, location='left')
#     cbar_nodes.set_label('Node Weight', rotation=270, labelpad=20)
#
#     ax = plt.gca()
#     ax.set_axis_on()
#     ax.spines['bottom'].set_color('0')
#     ax.spines['top'].set_color('0')
#     ax.spines['right'].set_color('0')
#     ax.spines['left'].set_color('0')
#     ax.tick_params(which='both', width=1, length=1, colors='black')
#
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.savefig(f'WPR_result/{timestamp}.svg', format='svg')
#     # plt.show()


def get_enhanced_color_mapping(score):
    """
    获取增强对比度的颜色映射参数

    Parameters:
    - score: 权重分数数组

    Returns:
    - vmin, vmax: 颜色映射的范围
    """
    # 使用分位数来设置颜色映射范围，增强视觉对比度
    q_low = np.percentile(score, 10)
    q_high = np.percentile(score, 90)
    range_expand = (q_high - q_low) * 0.1
    vmin = q_low - range_expand
    vmax = q_high + range_expand

    return vmin, vmax

def data_color_graph(score, locationgraph, location, color, labels):
    """
    绘制带有时空因果关系的网络图
    """
    # 获取增强的颜色映射参数
    vmin, vmax = get_enhanced_color_mapping(score)

    norm1 = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(14, 12))
    plt.title('Contribution weights with Causal Relationships')

    # 获取节点位置
    pos = {i: (location[i, 0], location[i, 1]) for i in range(len(location))}

    # 绘制节点，使用增强对比度的颜色映射
    # 移除norm参数，使用vmin和vmax直接传入
    nodes = nx.draw_networkx_nodes(locationgraph, pos, node_size=80,
                                   node_color=score, cmap='YlGn', alpha=0.8,
                                   vmin=vmin, vmax=vmax)

    # 绘制节点编号
    node_labels = {i: str(i) for i in range(len(location))}
    nx.draw_networkx_labels(locationgraph, pos, labels=node_labels, font_size=6)

    # 分离不同类型和时间滞后的边
    temporal_edges = []
    spatial_edges = []

    for source, target, key, data in locationgraph.edges(keys=True, data=True):
        if data.get('edge_type') == 'temporal':
            temporal_edges.append((source, target, key, data))
        elif data.get('edge_type') == 'spatial':
            spatial_edges.append((source, target, key, data))

    # 获取所有因果效应值用于颜色映射
    all_effects = [data['causal_effect'] for _, _, _, data in temporal_edges]
    all_effects.extend([data['causal_effect'] for _, _, _, data in spatial_edges])

    if all_effects:
        min_effect = min(all_effects)
        max_effect = max(all_effects)
        abs_max = max(abs(min_effect), abs(max_effect))
        vmin_edges, vmax_edges = -abs_max, abs_max
    else:
        vmin_edges, vmax_edges = -1, 1

    # 创建颜色映射
    cmap = plt.cm.coolwarm  # 红蓝表示正负因果效应
    norm_edges = plt.Normalize(vmin=vmin_edges, vmax=vmax_edges)

    # 绘制时间因果边 (带箭头)
    for source, target, key, data in temporal_edges:
        causal_effect = data['causal_effect']
        edge_color = cmap(norm_edges(causal_effect))
        width = 0.5 + abs(causal_effect) * 3  # 根据效应强度调整宽度
        lag = data.get('lag', 0)

        # 绘制带箭头的边
        nx.draw_networkx_edges(locationgraph, pos,
                               edgelist=[(source, target, key)],
                               width=width, edge_color=[edge_color],
                               alpha=0.7, arrows=True, arrowsize=10,
                               arrowstyle='->')

        # 添加时间滞后标签（调整位置和大小以避免覆盖节点）
        if lag > 0:
            # 计算边的中点位置
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # 对于自环边，调整标签位置避免与节点重叠和节点编号
            if source == target:
                # 自环边标签位置调整到节点左侧，避免挡住节点编号
                mid_x = x1 - 2  # 向左偏移
                mid_y = y1  # 稍微向上偏移

            # 添加滞后标签，调整字体大小
            plt.text(mid_x, mid_y, f'τ{lag}',
                     fontsize=5, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))

    # 绘制空间因果边 (无箭头，表示无向)
    for source, target, key, data in spatial_edges:
        causal_effect = data['causal_effect']
        edge_color = cmap(norm_edges(causal_effect))
        width = 0.3 + abs(causal_effect) * 2  # 根据效应强度调整宽度

        # 绘制无向边
        nx.draw_networkx_edges(locationgraph, pos,
                               edgelist=[(source, target, key)],
                               width=width, edge_color=[edge_color],
                               alpha=0.5, arrows=False)

    # 添加颜色条 (因果效应)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_edges)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Causal Effect', rotation=270, labelpad=20)

    # 添加节点权重颜色条（使用增强对比度的颜色映射）
    sm_nodes = cm.ScalarMappable(norm=norm1, cmap='YlGn')
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.8, location='left')
    cbar_nodes.set_label('Node Weight', rotation=270, labelpad=20)

    # 在颜色条上标注极值点
    cbar_nodes.set_ticks([vmin, np.mean([vmin, vmax]), vmax])
    cbar_nodes.set_ticklabels([f'{vmin:.6f}', f'{np.mean([vmin, vmax]):.6f}', f'{vmax:.6f}'])

    ax = plt.gca()
    ax.set_axis_on()
    ax.spines['bottom'].set_color('0')
    ax.spines['top'].set_color('0')
    ax.spines['right'].set_color('0')
    ax.spines['left'].set_color('0')
    ax.tick_params(which='both', width=1, length=1, colors='black')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'WPR_result/{timestamp}.svg', format='svg')
    # plt.show()


def plot_scatter_weights(score, location, title_suffix=""):
    """
    绘制权重散点图

    Parameters:
    - score: 权重分数数组
    - location: 节点位置数组
    - title_suffix: 图标题后缀
    """
    lon = location[:, 0]
    lat = location[:, 1]

    # 使用相同的增强颜色映射
    vmin, vmax = get_enhanced_color_mapping(score)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(lon, lat, s=50, c=score, cmap='plasma_r',
                          vmin=vmin, vmax=vmax)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Aggregated Contribution weights {title_suffix}')

    # 标注极值点
    max_idx = np.argmax(score)
    min_idx = np.argmin(score)
    plt.annotate(f'Max({max_idx}): {score[max_idx]:.6f}',
                 (lon[max_idx], lat[max_idx]),
                 xytext=(lon[max_idx] + 0.1, lat[max_idx] + 0.1),
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.annotate(f'Min({min_idx}): {score[min_idx]:.6f}',
                 (lon[min_idx], lat[min_idx]),
                 xytext=(lon[min_idx] + 0.1, lat[min_idx] - 0.1),
                 arrowprops=dict(arrowstyle='->', color='blue'))

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Node Weight', rotation=270, labelpad=20)

    plt.savefig(f'WPR_result/{timestamp}_scatter.svg', format='svg')
    plt.show()


D = np.diag(np.power(np.sum(weight_matrix, axis=1), -1))
M = np.dot(weight_matrix, D)
contribution_weights = WeightedPageRank(M, weights, N=node, T=300)

"""
需要对贡献度权重进行错位加权
恢复原图节点数目
获得加权后的贡献权重
"""
# 将400个节点的贡献度聚合为100个原始节点
aggregated_weights = np.zeros(original_node)
for i in range(original_node):
    indices = [i + j * original_node for j in range(4)]  # [i, i+100, i+200, i+300]
    aggregated_weights[i] = np.sum(contribution_weights[indices])
    # aggregated_weights[i] = contribution_weights[i]

# 在聚合权重之后添加调试信息
print("=== 权重分析 ===")
print(f"原始贡献权重统计:")
print(f"  最小值: {np.min(contribution_weights):.8f}")
print(f"  最大值: {np.max(contribution_weights):.8f}")
print(f"  平均值: {np.mean(contribution_weights):.8f}")
print(f"  标准差: {np.std(contribution_weights):.8f}")

print(f"\n聚合后权重统计:")
print(f"  最小值: {np.min(aggregated_weights):.8f}")
print(f"  最大值: {np.max(aggregated_weights):.8f}")
print(f"  平均值: {np.mean(aggregated_weights):.8f}")
print(f"  标准差: {np.std(aggregated_weights):.8f}")

# 查找异常值节点
print(f"\n=== 异常值分析 ===")
sorted_indices = np.argsort(-aggregated_weights)
print(f"前10个最高权重节点: {sorted_indices[:10]}")
print(f"对应的权重值: {aggregated_weights[sorted_indices[:10]]}")

print(f"后10个最低权重节点: {sorted_indices[-10:]}")
print(f"对应的权重值: {aggregated_weights[sorted_indices[-10:]]}")

# 特别检查42号和45号节点
print(f"\n=== 特定节点分析 ===")
print(f"42号节点聚合权重: {aggregated_weights[42]:.8f}")
print(f"45号节点聚合权重: {aggregated_weights[45]:.8f}")

# 检查构成这些节点的4个扩展节点
indices_42 = [42 + j * original_node for j in range(4)]
indices_45 = [45 + j * original_node for j in range(4)]

print(f"42号节点由以下扩展节点构成: {indices_42}")
print(f"  对应权重: {contribution_weights[indices_42]}")
print(f"  各权重总和: {np.sum(contribution_weights[indices_42]):.8f}")

print(f"45号节点由以下扩展节点构成: {indices_45}")
print(f"  对应权重: {contribution_weights[indices_45]}")
print(f"  各权重总和: {np.sum(contribution_weights[indices_45]):.8f}")

# 检查权重分布
print(f"\n=== 权重分布分析 ===")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"{p}百分位数: {np.percentile(aggregated_weights, p):.8f}")

# 归一化处理
aggregated_weights = aggregated_weights / np.sum(aggregated_weights)

node_list = np.argsort(-aggregated_weights)

os.makedirs('./WPR_result', exist_ok=True)
# 重要性从大到小排序
np.savetxt('./WPR_result/_node_list_' + timestamp +'_GP_' + str(Gpn) + '_node_' + str(node) + '_data_' + str(data_num) + '.txt',
           node_list, fmt='%d')

plt.figure()
plt.plot(range(1, original_node + 1), aggregated_weights, '-o')
plt.xlabel('Sensor node')
plt.ylabel('Aggregated Contribution weights')
# 添加极值标注
max_idx = np.argmax(aggregated_weights)
min_idx = np.argmin(aggregated_weights)
plt.annotate(f'Max: {aggregated_weights[max_idx]:.6f}',
             xy=(max_idx, aggregated_weights[max_idx]),
             xytext=(max_idx+5, aggregated_weights[max_idx]),
             arrowprops=dict(arrowstyle='->'))
plt.annotate(f'Min: {aggregated_weights[min_idx]:.6f}',
             xy=(min_idx, aggregated_weights[min_idx]),
             xytext=(min_idx+5, aggregated_weights[min_idx]-0.00002),
             arrowprops=dict(arrowstyle='->'))
plt.savefig('WPR_result/Gp' + str(Gpn) + '_Aggregated_Contribution_weights_line' + timestamp+ '.svg', format='svg')
plt.show()


selected_node = node_list[:select_node]
plt.scatter(lon, lat, s=15, c='gainsboro', label='All nodes')
plt.scatter(lon[selected_node], lat[selected_node], s=30, c='#7e2f8e', label='Optimized nodes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Optimized nodes subset')
plt.legend()
plt.savefig('WPR_result/Gp' + str(Gpn) + '_MGC-RM' + 'select' + str(select_node) +'_'+ timestamp+ '.svg', format='svg')
plt.show()

plt.scatter(lon, lat, s=15, c='gainsboro', label='All nodes')
plt.scatter(lon[selected_node], lat[selected_node], s=30, c='#7e2f8e', label='Optimized nodes')
# 只有当还有剩余节点时才绘制备选节点
if select_node < original_node:
    plt.scatter(lon[node_list[select_node]], lat[node_list[select_node]], s=30, c='#d400fe',
                label='Contribution weights backup node')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('')
plt.legend(loc='lower left')
plt.savefig('WPR_result/Gp' + str(Gpn) + '_MGC-RM' + 'select' + str(select_node) + '_'+ timestamp + 'backup.svg', format='svg')
plt.show()

# 使用创建的简单图进行可视化，注意这里使用的是聚合后的权重
# data_color_graph(aggregated_weights, G_viz, positions, 'plasma_r', False)
# 在最后的可视化调用中使用新的图
data_color_graph(aggregated_weights, G_causal, positions, 'plasma_r', False)
# 绘制散点图
plot_scatter_weights(aggregated_weights, positions, "")