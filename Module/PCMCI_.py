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

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

"""
数据读取
"""
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/simulation_hubs_core/N100_T10000_Hubs15_Viz_Final')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
# 构建 PCMCI 结果保存文件夹路径
output_dir = os.path.join(current_dir, 'PCMCI_')
# 如果文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 加载数据
position_file = os.path.join(dataset_dir, 'node_positions.csv')
positions_df = pd.read_csv(position_file)

data_path = os.path.join(dataset_dir, 'Env_A_timeseries.csv')
data = pd.read_csv(data_path)
print(f"data: {data.shape} 时间点×节点数")
data = pd.read_csv(data_path).values


# 定义变量名
var_names = [f'Node_{i}' for i in range(data.shape[1])]

# 创建tigramite DataFrame (必须使用numpy数组)
dataframe = pp.DataFrame(data,
                        var_names=var_names,
                        datatime=np.arange(data.shape[1]))  # 添加时间索引
"""
初始化 PCMCI 对象
"""
# 初始化条件独立性检验
cond_ind_test = ParCorr(significance='analytic')
# 初始化 PCMCI 对象
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)


# 获取节点坐标（假设列名为 'x', 'y'）
positions = positions_df[['x', 'y']].values  # shape: (N, 2)
N = positions.shape[0]
# # 计算 欧式 距离矩阵
# distance_matrix = compute_euclidean_matrix(positions)  # 单位：米
# # 设置距离阈值（根据你的数据调整）
# distance_threshold =10.0  # 例如：只保留 _ km 以内的潜在因果关系

tau_max = 5  # 可根据需要调整
pc_alpha = 0.0001  # 显著性阈值

# # 手动构造 link_assumptions（等效于 make_link_assumptions）
# link_assumptions = {}
#
# # 初始化统计变量
# k_counts = {}  # 每个节点 i 的有链接边数（考虑时间）
# d_counts = {}  # 每个节点 i 的有链接的源点数（不考虑时间）
#
# for i in range(N):  # i 是目标变量（被影响变量）
#     k = 0
#     d_set = set()  # 用于记录不重复的源节点 j
#     link_assumptions[i] = {}
#     for j in range(N):  # j 是源变量（影响变量）
#         for tau in range(1, tau_max + 1):
#             if distance_matrix[i, j] > distance_threshold:
#                 # 不搜索该连接 → 不添加该键值对即可
#                 continue
#             else:
#                 # 允许连接，并假设方向为 'o-o'
#                 link_assumptions[i][(j, -tau)] = 'o-o'
#                 k += 1
#                 d_set.add(j)  # 将源节点 j 加入集合（自动去重）
#     # 保存统计结果
#     k_counts[i] = k
#     d_counts[i] = len(d_set)
#     print(f"Node {i}: 共有 {k_counts[i]} 条链接（含时间），来自 {d_counts[i]} 个不同节点")
#

results = pcmci.run_pcmciplus(
    tau_max=tau_max,
    pc_alpha=pc_alpha
    # link_assumptions=link_assumptions,
)

# 参数格式化，用于文件名
tau_str = f'tau{tau_max}'
alpha_str = f'alpha{pc_alpha:.4f}'.replace('.', 'p')  # 0.001 → alpha0p001
# dist_str = f'dist{distance_threshold:.1f}km'.replace('.', 'd')  # 5.0 → dist5d0km
# 保存 PCMCI 的结果到 .npz 文件
# filename3 = f'pcmci_results_{tau_str}_{alpha_str}_{dist_str}.npz'
filename3 = f'pcmci_results_{tau_str}_{alpha_str}.pkl'
pickle_file = os.path.join(output_dir, filename3)
with open(pickle_file, 'wb') as f:
    pickle.dump(results, f)
print(f"完整 PCMCI 结果已保存为 pickle 文件: {pickle_file}")


with open(pickle_file, 'rb') as f:
    loaded_results = pickle.load(f)

# 打印 loaded_results 的详细信息
# print("\n=== loaded_results 详细信息 ===")
# print(f"loaded_results 类型: {type(loaded_results)}")
#
# if isinstance(loaded_results, dict):
#     print(f"loaded_results 键: {list(loaded_results.keys())}")

graph = loaded_results['graph']
val_matrix = loaded_results['val_matrix']

"""
loaded_results['graph']
类型：三维数组，元素为字符串
内容：因果关系的存在性和方向
值：
'-->'：表示存在因果关系，从源节点指向目标节点
'' 或其他值：表示不存在显著的因果关系

loaded_results['val_matrix']
类型：三维数组，元素为数值
内容：因果关系的强度（因果效应值）
值：具体的数值，表示因果效应的大小和方向（正负）
"""


# 统计每个节点的自关联数量（节点到自己的因果联系）
node_self_counts = np.zeros(N)
for i in range(N):
    for tau in range(1, tau_max + 1):
        if graph[i, i, tau] == '-->':  # 检查节点i到节点i的关联
            node_self_counts[i] += 1

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
因果图
"""
# # 创建 NetworkX 图对象
# G = nx.DiGraph()
#
# # 添加节点（仅有效位置），记录自关联数量
# for i in range(N):
#     G.add_node(i, pos=(positions[i, 0], positions[i, 1]),self_count=node_self_counts[i])
#
# # 添加边（仅两个节点都有效的边），排除自关联
# for (i, j), count in edge_counts.items():
#     if  i != j and count > 0:  # 确保不是自关联
#         G.add_edge(j, i, weight=count)  # 从j到i的有向边
#
#
# # 获取节点位置
# pos = nx.get_node_attributes(G, 'pos')
#
# # 打印图信息以调试
# print(f"图中节点数: {G.number_of_nodes()}")
# print(f"图中边数: {G.number_of_edges()}")
#
# # 打印统计信息
# print("=== 节点自关联、入边、出边统计 ===")
# total_tau = sum(edge_counts.values())
# # 创建统计表
# stats = []
# outcoming_node_count = []
# for i in G.nodes:
#     # 自关联数量
#     self_count = node_self_counts[i]
#     # 入边统计
#     incoming_edges = [u for u, v in G.edges if v == i]  # 指向该节点的源节点列表
#     incoming_edge_weights = [G[u][v]['weight'] for u, v in G.edges if v == i]
#     total_incoming_tau = sum(incoming_edge_weights)
#     # 出边统计
#     outgoing_edges = [v for u, v in G.edges if u == i]  # 该节点指向的目标节点列表
#     outgoing_edge_weights = [G[i][v]['weight'] for v in outgoing_edges]
#     total_outgoing_tau = sum(outgoing_edge_weights)
#
#     outcoming_node_count.append(len(outgoing_edges))
#
#     # 保存统计
#     stats.append({
#         'Node': i,
#         'SelfLinkCount': self_count,
#         'IncomingNodeCount': len(incoming_edges),
#         'TotalIncomingTau': total_incoming_tau,
#         'OutgoingNodeCount': len(outgoing_edges),
#         'TotalOutgoingTau': total_outgoing_tau
#     })
#
#     # 打印
#     print(f"Node {i}:")
#     print(f"  自关联数量: {self_count}")
#     print(f"  指向它的节点数: {len(incoming_edges)}")
#     print(f"  入边因果联系总数（τ）: {total_incoming_tau}")
#     print(f"  被它指向的节点数: {len(outgoing_edges)}")
#     print(f"  出边因果联系总数（τ）: {total_outgoing_tau}")
#     print("-" * 40)
#
# print("=============================")
#
# # 设置绘图参数
# fig, ax = plt.subplots(figsize=(12, 10), dpi=600)
#
# # 设置节点的颜色映射
# if outcoming_node_count:
#     max_incoming = max(outcoming_node_count)
#     node_norm = Normalize(vmin=0, vmax=max_incoming)
#     node_cmap = cm.plasma_r
# else:
#     node_norm = Normalize(vmin=0, vmax=1)
#     node_cmap = cm.plasma_r
#
# # 获取节点颜色
# node_colors = [node_cmap(node_norm(c)) for c in outcoming_node_count]
#
# # 获取边的权重（因果联系数量）
# edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
#
# # 设置边的颜色映射（基于入边因果联系总数）
# if edge_weights:
#     max_edge_weight = max(edge_weights)
#     edge_norm = Normalize(vmin=0, vmax=max_edge_weight)
#     edge_cmap = cm.plasma_r
# else:
#     edge_norm = Normalize(vmin=0, vmax=1)
#     edge_cmap = cm.plasma_r
#
# # 边颜色
# edge_colors = [edge_cmap(edge_norm(w)) for w in edge_weights]
#
# # 可选：添加节点编号
# labels = {i: str(moteid) for i, moteid in enumerate(positions_df['node_id'].values.flatten())}
#
# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, ax=ax)
# # 绘制边
# nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1, alpha=0.7, ax=ax,
#                               arrowstyle='->', arrowsize=10)
# # 绘制标签
# nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='black', ax=ax)
#
# # ✅ 添加颜色条 - 边因果联系数量
# sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
# sm_edges.set_array([])
# cbar_edges = fig.colorbar(sm_edges, ax=ax, orientation='vertical', shrink=0.3, pad=0.05)
# cbar_edges.set_label('Edge Causal Link Count')
#
# # ✅ 添加颜色条 - 节点出边数量
# sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
# sm_nodes.set_array([])
# cbar_nodes = fig.colorbar(sm_nodes, ax=ax, orientation='vertical', shrink=0.3, pad=0.05)
# cbar_nodes.set_label('Number of Outgoing Nodes')
#
# nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='black', ax=ax)
#
# # 设置标题和坐标轴
# ax.set_title("Causal Graph")
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.grid(True, linestyle='--', alpha=0.3)
# plt.axis('on')
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# plt.tight_layout()
# # 保存图像
# # 拼接文件名
# # filename1 = f'network_causal_graph_{tau_str}_{alpha_str}_{dist_str}.png'
# filename1 = f'network_causal_graph_{tau_str}_{alpha_str}.png'
#
#
# # 保存图像
# plt.savefig(os.path.join(output_dir, filename1), dpi=600, bbox_inches='tight')
# print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")

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
for i in range(N):  # 源节点
    for j in range(N):  # 目标节点
        # if i == j:  # 排除自关联
        #     continue
        for tau in range(1, tau_max + 1):  # 时间滞后 (通常从1开始)
            # 检查是否存在显著的因果关系
            if graph[i, j, tau] == '-->':
                # 获取对应的因果效应权重值
                causal_effect = val_matrix[i, j, tau]
                weight = abs(causal_effect)  # 使用绝对值作为边的权重

                # 添加边到MultiDiGraph，使用tau作为key保留时间滞后信息
                G.add_edge(i, j, key=tau, weight=weight, lag=tau, causal_effect=causal_effect)
                edge_count += 1
                print(f"添加边: 节点{i} -> 节点{j}, 滞后{tau}, 因果效应{causal_effect:.4f}")

print(f"总共添加了 {edge_count} 条边")

# 打印图信息以调试
print(f"图中节点数: {G.number_of_nodes()}")
print(f"图中边数: {G.number_of_edges()}")

pos = nx.get_node_attributes(G, 'pos')



# 设置绘图参数
fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
ax.set_aspect('equal', adjustable='box')

# 计算节点的度数
degrees = np.array([G.degree(i) for i in range(N)])
vmax = np.max(degrees) if np.max(degrees) > 0 else 1
node_colors = degrees
node_sizes = 200 + degrees * 10
cmap_nodes = plt.cm.YlGn

nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=node_colors, cmap=cmap_nodes,
                       edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)

# 分别处理自相关边（自循环）和普通边
self_edges = []  # 自相关边
normal_edges = []  # 普通边

for u, v, key, data in G.edges(keys=True, data=True):
    if u == v:
        self_edges.append((u, v, key, data))
    else:
        normal_edges.append((u, v, key, data))

# 获取所有权重用于颜色映射
all_weights = [abs(data['weight']) for u, v, key, data in G.edges(keys=True, data=True)]
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
for u, v, key, data in normal_edges:
    lag = data['lag']
    causal_effect = data['causal_effect']

    # 根据因果效应大小设置颜色
    color = cmap(norm(causal_effect))

    # 根据因果效应绝对值设置边宽度
    width = 1 + abs(causal_effect) * 5  # 基础宽度1，根据因果效应绝对值调整

    # 绘制单条边
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v, key)],
                           width=width,
                           edge_color=[color],
                           alpha=0.7, arrows=True, arrowsize=10, ax=ax)

# 绘制自相关边（自循环）
for u, v, key, data in self_edges:
    lag = data['lag']
    causal_effect = data['causal_effect']

    # 根据权重大小设置颜色
    color = cmap(norm(causal_effect))

    # 根据权重绝对值设置边宽度
    width = 1 + abs(causal_effect) * 8

    # 绘制自循环边
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v, key)],
                           width=width,
                           edge_color=[color],
                           alpha=0.7, arrows=True, arrowsize=20,
                           connectionstyle=f'arc3,rad=0.3', ax=ax)  # 弯曲的自循环

# 添加边标签（包含时间滞后和效应符号）
edge_labels = {}
for u, v, key, data in G.edges(keys=True, data=True):
    lag = data['lag']
    causal_effect = data['causal_effect']
    sign = '+' if causal_effect >= 0 else '-'
    edge_labels[(u, v, key)] = f"{sign}τ{lag}"

# 绘制边滞后标签
for (u, v, key), label in edge_labels.items():
    # 计算边的中点位置
    if u == v:  # 自循环
        # 自循环标签位置
        x, y = pos[u]
        label_pos = (x + 1.5, y + 1.5)
    else:  # 普通边
        # 普通边中点位置
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        label_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

        # 如果有多条边连接同一对节点，稍微偏移标签位置
        edge_count = len([k for uu, vv, k, d in G.edges(keys=True, data=True) if (uu, vv) == (u, v)])
        if edge_count > 1:
            label_pos = (label_pos[0] + 0.5, label_pos[1] + 0.5)

    # 根据因果效应符号设置标签颜色
    causal_effect = None
    for uu, vv, kk, dd in G.edges(keys=True, data=True):
        if (uu, vv, kk) == (u, v, key):
            causal_effect = dd['causal_effect']
            break

    label_color = 'red' if causal_effect is not None and causal_effect >= 0 else 'blue'

    plt.text(label_pos[0], label_pos[1], label,
             fontsize=4, ha='center', va='center', color=label_color,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 添加颜色图例（因果效应）
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
cbar.set_label('Causal Effect', rotation=270, labelpad=10)

# 添加节点出边数量颜色图例
sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
sm_nodes.set_array([])
cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.040, pad=0.05, location='left')
cbar_nodes.set_label('Number of Outgoing Edges', rotation=90, labelpad=7)

# 设置标题和坐标轴
plt.title("Causal Graph with Time Lag Information")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存图像
filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
# 1. 保存高清 SVG (矢量图，无损缩放)
svg_path = os.path.join(output_dir, filename1 + '.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
# 2. 保存 PNG (预览用)
png_path = os.path.join(output_dir, filename1 + '.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')  # 高DPI

print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
plt.close()


"""
因果矩阵热力图
"""
# 创建 N×N 矩阵
N = positions.shape[0]
causal_matrix = np.zeros((N, N))

# 填充矩阵（i → j 的因果联系数量）
for (i, j), count in edge_counts.items():
    causal_matrix[i, j] = count  # 注意：G.add_edge(j, i) 表示 j → i
# # 添加自关联（i → i）
# for i in range(N):
#     causal_matrix[i, i] = node_self_counts[i]  # node_self_counts[i] 是 i → i 的因果联系数量

# 创建绘图对象
fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8), dpi=1800)

# 绘制热力图
cax = ax_matrix.imshow(causal_matrix, cmap='YlOrRd', interpolation='none', aspect='auto')

# 添加颜色条
cbar = fig_matrix.colorbar(cax, ax=ax_matrix)
cbar.set_label('Causal Link Count (τ)')

# 设置标题和坐标轴
ax_matrix.set_title("Node Causal Link Count Matrix (j → i)", fontsize=14)
ax_matrix.set_xlabel("Target Node (i)", fontsize=12)
ax_matrix.set_ylabel("Source Node (j)", fontsize=12)

# 设置坐标轴刻度
ax_matrix.set_xticks(np.arange(N))
ax_matrix.set_yticks(np.arange(N))
ax_matrix.tick_params(axis='both', which='major', labelsize=5)

plt.tight_layout()

# filename2 = f'causal_matrix_{tau_str}_{alpha_str}_{dist_str}.png'
filename2 = f'causal_matrix_{tau_str}_{alpha_str}.png'
plt.savefig(os.path.join(output_dir, filename2), dpi=1800, bbox_inches='tight')
print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")

