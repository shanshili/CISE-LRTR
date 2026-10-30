"""
使用 nx.MultiDiGraph()，能处理多个时间滞后的边
包含自关联边的可视化（绘制自循环）
保存详细信息如lag、causal_effect、weight
从 .pkl 文件加载结果

根据因果效应的正负值使用颜色编码（红蓝表示）
边的宽度根据因果效应强度变化
显示时间滞后信息（τ值）
双侧颜色条：节点出边数量和边的因果效应

本脚本用于可视化PCMCI因果发现算法得到的时序因果关系图。主要特点：
1. 使用MultiDiGraph处理多个时间滞后的边
2. 包含自关联边的可视化（绘制自循环）
3. 根据因果效应正负值进行红蓝颜色编码
4. 边宽根据因果效应强度变化
5. 显示时间滞后信息（τ值）
6. 双侧颜色条：节点出边数量和边的因果效应

主要功能模块：
---------------
1. 数据加载与处理
   - 从.pkl文件加载PCMCI结果
   - 加载节点位置信息
   - 提取因果图和效应矩阵

2. 因果图统计与分析
   - 统计节点自关联数量
   - 构建节点间连接统计字典
   - 计算每个节点的出边数量

3. 网络图构建与可视化
   - 使用nx.MultiDiGraph构建多向有向图
   - 添加带时间滞后信息的边
   - 分离处理自循环边和普通边
   - 基于因果效应值设置边颜色和宽度
   - 添加时间滞后标签

4. 矩阵热力图生成
   - 生成源节点→目标节点因果连接数量矩阵
   - 可视化节点间的连接强度

5. 可视化输出
   - 生成高分辨率因果网络图
   - 生成因果矩阵热力图
   - 保存图像到指定目录

函数说明：
---------
主流程函数：
1. main() - 脚本主入口，协调整个可视化流程
   参数：无
   功能：加载数据、构建图、可视化、保存结果

数据处理函数：
2. load_pcmci_results(pickle_file)
   参数：pickle_file - PCMCI结果文件路径
   返回：graph, val_matrix - 因果图和效应矩阵
   功能：从.pkl文件加载PCMCI分析结果

3. load_node_positions(position_file)
   参数：position_file - 节点位置CSV文件路径
   返回：positions - 节点坐标数组(N, 2)
   功能：加载节点的空间坐标信息

图构建函数：
4. build_multi_di_graph(graph, val_matrix, N, tau_max, positions)
   参数：graph - PCMCI因果图
         val_matrix - 因果效应矩阵
         N - 节点数量
         tau_max - 最大时间滞后
         positions - 节点位置
   返回：G - NetworkX MultiDiGraph对象
         edge_counts - 边统计字典
         node_self_counts - 自关联计数数组
   功能：构建带时间滞后信息的因果图

5. calculate_edge_statistics(graph, N, tau_max)
   参数：graph - PCMCI因果图
         N - 节点数量
         tau_max - 最大时间滞后
   返回：edge_counts - 边统计字典
         node_self_counts - 自关联计数数组
   功能：统计节点间的因果连接数量

可视化函数：
6. plot_causal_graph(G, positions, edge_counts, node_self_counts,
                     output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information")
   参数：G - NetworkX MultiDiGraph对象
         positions - 节点位置
         edge_counts - 边统计字典
         node_self_counts - 自关联计数数组
         output_dir - 输出目录
         tau_str - 时间滞后字符串
         alpha_str - 显著性阈值字符串
         title - 图标题
   功能：绘制因果网络图，包括节点、边、自循环、颜色条等

7. plot_causal_matrix(edge_counts, N, output_dir, tau_str, alpha_str)
   参数：edge_counts - 边统计字典
         N - 节点数量
         output_dir - 输出目录
         tau_str - 时间滞后字符串
         alpha_str - 显著性阈值字符串
   功能：生成因果连接数量矩阵热力图

辅助函数：
8. separate_self_edges(G)
   参数：G - NetworkX MultiDiGraph对象
   返回：self_edges, normal_edges - 自循环边列表和普通边列表
   功能：分离自循环边和节点间边

9. calculate_visualization_parameters(G, N)
   参数：G - NetworkX MultiDiGraph对象
         N - 节点数量
   返回：out_degree, vmax - 节点出度数组和最大值
         all_weights, vmin, vmax_edges - 边权重范围和归一化参数
   功能：计算可视化所需的参数

使用说明：
---------
1. 确保已安装所需库：networkx, matplotlib, numpy, pandas, tigramite
2. 准备PCMCI结果文件(.pkl)和节点位置文件(.csv)
3. 根据需要调整参数：tau_max, pc_alpha
4. 运行脚本，结果将保存到PCMCI_/N100_T10000_t3目录

输出文件：
---------
1. network_causal_graph_tau{tau}_alpha{alpha}_green_new.png
   - 因果网络图，包含节点、边、自循环、颜色条
   - 分辨率：600 DPI

2. causal_matrix_tau{tau}_alpha{alpha}_new.png
   - 因果连接数量矩阵热力图
   - 分辨率：1800 DPI

颜色编码：
---------
- 节点颜色：YlGn渐变色，表示节点出边数量
- 边颜色：seismic渐变色（红-白-蓝）
  * 蓝色：负因果效应（抑制作用）
  * 红色：正因果效应（促进作用）
- 边宽度：1 + |因果效应| × 5，效应越强边越粗

标签说明：
---------
- 节点标签：节点ID
- 边标签：符号τ滞后值（如：+τ1表示滞后1的正效应）
- 颜色条：
  * 左侧：节点出边数量
  * 右侧：边因果效应值

注意事项：
---------
1. 自循环边使用小箭头表示，位于节点右上方
2. MultiDiGraph允许同一节点对间有多条边（不同时间滞后）
3. 效应矩阵中的值可能为NaN，需在可视化前处理
4. 大量节点时，建议调整节点大小和标签字体
"""
import sys
import os
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（GNN-Causal-Inference）
project_root = os.path.dirname(current_dir)
# 将项目根目录加入 sys.path
sys.path.append(project_root)
import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
import matplotlib.pyplot as plt
from utils.euclidean_distance import  compute_euclidean_matrix
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pickle
import torch

tau_max = 3  # 可根据需要调整
pc_alpha = 0.0001  # 显著性阈值

# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/Causal_simulation_data_2/N100_T10000_t3')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
# 构建 PCMCI 结果保存文件夹路径
output_dir = os.path.join(current_dir, 'PCMCI_/N100_T10000_t3')
# 如果文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 参数格式化，用于文件名
tau_str = f'tau{tau_max}'
alpha_str = f'alpha{pc_alpha:.4f}'.replace('.', 'p')  # 0.001 → alpha0p001
filename3 = f'pcmci_results_{tau_str}_{alpha_str}.pkl'
pickle_file = os.path.join(output_dir, filename3)

# 加载数据
position_file = os.path.join(dataset_dir, 'node_positions.csv')
positions_df = pd.read_csv(position_file)
positions = positions_df[['x', 'y']].values  # shape: (N, 2)
N = positions.shape[0]

with open(pickle_file, 'rb') as f:
    loaded_results = pickle.load(f)

graph = loaded_results['graph']
val_matrix = loaded_results['val_matrix']

# 统计每个节点的自关联数量（节点到自己的因果联系）
node_self_counts = np.zeros(N)
for target in range(N):
    for tau in range(1, tau_max + 1):
        if graph[target, target, tau] == '-->':  # 检查节点target到节点target的关联
            node_self_counts[target] += 1

# 构建连接统计字典（排除自关联）
edge_counts = {}
for target in range(N):  # target是目标节点
    print(f"节点{target}(目标)")
    for source in range(N):  # source是源节点
        # if target == source:  # 排除自关联
        #     continue
        count = 0
        for tau in range(1, tau_max + 1):
            # graph[target, source, tau] == '-->' 表示 source 在 t-τ 时刻对 target 在 t 时刻有因果影响
            if graph[target, source, tau] == '-->':
                count += 1
                print(graph[target, source, tau])
                print(f"节点{source}(源)对节点{target}(目标)因果效应{val_matrix[target, source, tau]}")
        if count > 0:
            # edge_counts[(source, target)] 表示从source到target的连接
            edge_counts[(source, target)] = count
            print(f"节点{source}链接到节点{target}共{count}条")


"""
标绘制因果图
:param links_coeffs: 因果关系字典
:param node_positions: 节点坐标数组 (N, 2)
:param output_dir: 输出目录
:param title: 图标题
:param show: 是否显示图像
"""

G = nx.MultiDiGraph()  # 改为MultiDiGraph

# 添加节点并设置位置信息
for i in range(N):
    G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

# 添加边并保留所有时间滞后信息
edge_count = 0
for target in range(N):  # target是目标节点
    for source in range(N):  # source是源节点
        # if target == source:  # 排除自关联
        #     continue
        for tau in range(1, tau_max + 1):  # 时间滞后 (通常从1开始)
            # 检查是否存在显著的因果关系
            # graph[target, source, tau] == '-->' 表示 source 在 t-τ 时刻对 target 在 t 时刻有因果影响
            if graph[target, source, tau] == '-->':
                # 获取对应的因果效应权重值
                causal_effect = val_matrix[target, source, tau]
                weight = abs(causal_effect)  # 使用绝对值作为边的权重

                # 添加边到MultiDiGraph: source -> target
                G.add_edge(source, target, key=tau, weight=weight, lag=tau, causal_effect=causal_effect)
                edge_count += 1
                print(f"添加边: 节点{source}(源) -> 节点{target}(目标), 滞后{tau}, 因果效应{causal_effect:.4f}")

print(f"总共添加了 {edge_count} 条边")

# 打印图信息以调试
print(f"图中节点数: {G.number_of_nodes()}")
print(f"图中边数: {G.number_of_edges()}")

pos = nx.get_node_attributes(G, 'pos')

# 计算每个节点的出边数量（出边表示该节点影响其他节点的数量）
out_degree = np.zeros(N)
for source, target, key, data in G.edges(keys=True, data=True):
    out_degree[source] += 1  # 节点source有一条出边
vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1

# 设置绘图参数
plt.figure(figsize=(14, 12))
nx.draw_networkx_nodes(G, pos, node_size=60, node_color=out_degree,
                       cmap=plt.cm.YlGn, alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=6)

# 分别处理自相关边（自循环）和普通边
self_edges = []  # 自相关边
normal_edges = []  # 普通边

for source, target, key, data in G.edges(keys=True, data=True):
    if source == target:
        self_edges.append((source, target, key, data))
    else:
        normal_edges.append((source, target, key, data))

# 获取所有权重用于颜色映射
all_weights = [abs(data['weight']) for source, target, key, data in G.edges(keys=True, data=True)]
if all_weights:
    min_weight = min(all_weights)
    max_weight = max(all_weights)

    # 确保颜色映射是对称的，以便更好地表示正负关系
    abs_max = max(abs(min_weight), abs(max_weight))
    vmin, vmax_edges = -abs_max, abs_max
else:
    vmin, vmax_edges = -1, 1

# 创建颜色映射（基于权重实际值，包括正负）
cmap = plt.cm.seismic  # 红-黄-蓝反转，负值为蓝色，正值为红色
norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

# 绘制普通边（节点间连接）
for source, target, key, data in normal_edges:
    lag = data['lag']
    causal_effect = data['causal_effect']

    # 根据因果效应大小设置颜色
    color = cmap(norm(causal_effect))

    # 根据因果效应绝对值设置边宽度
    width = 1 + abs(causal_effect) * 5  # 基础宽度1，根据因果效应绝对值调整

    # 绘制单条边: source -> target
    nx.draw_networkx_edges(G, pos, edgelist=[(source, target, key)],
                           width=width,
                           edge_color=[color],
                           alpha=0.7, arrows=True, arrowsize=10)

# 绘制自相关边（自循环）
for source, target, key, data in self_edges:
    lag = data['lag']
    causal_effect = data['causal_effect']

    # 根据权重大小设置颜色
    color = cmap(norm(causal_effect))

    # 根据权重绝对值设置边宽度
    width = 1 + abs(causal_effect) * 5

    # 绘制自循环边
    x, y = pos[source]
    # 手动绘制一个小箭头表示自循环
    plt.annotate('', xy=(x, y+0.005), xytext=(x-0.005, y),
                 arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7, mutation_scale=width*3+10))

# 添加边标签（包含时间滞后和效应符号）
edge_labels = {}
for source, target, key, data in G.edges(keys=True, data=True):
    lag = data['lag']
    causal_effect = data['causal_effect']
    sign = '+' if causal_effect >= 0 else '-'
    edge_labels[(source, target, key)] = f"{sign}τ{lag}"

# 添加颜色图例（因果效应）
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
cbar.set_label('Causal Effect', rotation=270, labelpad=20)

# 添加节点出边数量颜色图例
sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
sm_nodes.set_array([])
cbar_nodes = plt.colorbar(sm_nodes, ax=plt.gca(), shrink=0.8, location='left')
cbar_nodes.set_label('Number of Outgoing Edges', rotation=270, labelpad=20)

# 设置标题和坐标轴
plt.title("Causal Graph with Time Lag Information")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存图像
filename1 = f'network_causal_graph_{tau_str}_{alpha_str}_green_new.png'
plt.savefig(os.path.join(output_dir, filename1), dpi=600, bbox_inches='tight')
print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
plt.close()

"""
因果矩阵热力图
"""

# 创建 N×N 矩阵
causal_matrix = np.zeros((N, N))

# 填充矩阵（source → target 的因果联系数量）
for (source, target), count in edge_counts.items():
    causal_matrix[source, target] = count  # source → target

# 创建绘图对象
fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8), dpi=1800)

# 绘制热力图
cax = ax_matrix.imshow(causal_matrix, cmap='YlOrRd', interpolation='none', aspect='auto')

# 添加颜色条
cbar = fig_matrix.colorbar(cax, ax=ax_matrix)
cbar.set_label('Causal Link Count (τ)')

# 设置标题和坐标轴
ax_matrix.set_title("Node Causal Link Count Matrix (source → target)", fontsize=14)
ax_matrix.set_xlabel("Target Node", fontsize=12)
ax_matrix.set_ylabel("Source Node", fontsize=12)

# 设置坐标轴刻度
ax_matrix.set_xticks(np.arange(N))
ax_matrix.set_yticks(np.arange(N))
ax_matrix.tick_params(axis='both', which='major', labelsize=5)

plt.tight_layout()

filename2 = f'causal_matrix_{tau_str}_{alpha_str}_new.png'
plt.savefig(os.path.join(output_dir, filename2), dpi=1800, bbox_inches='tight')
print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")
