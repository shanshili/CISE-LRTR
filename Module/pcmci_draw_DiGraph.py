"""
使用 nx.DiGraph()，只处理简单有向图
统计自关联但未在图中特殊表示
只保存简单的边权重
从 .npz 文件加载结果

使用单一颜色映射表示边权重
节点颜色表示出边数量
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import networkx as nx

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 构建路径
dataset_dir = os.path.join(current_dir, '..', 'dataset')
output_dir = os.path.join(current_dir, 'PCMCI')
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 加载保存的 PCMCI 结果
result_file = os.path.join(output_dir, 'pcmci_results_tau32_alpha0p001_dist8d0km.npz')
results = np.load(result_file, allow_pickle=True)

# 加载位置数据
position_file = os.path.join(dataset_dir, 'TJ_position.csv')
positions_df = pd.read_csv(position_file)
positions = positions_df[['lat', 'lon']].values
N = positions.shape[0]


distance_threshold =8.0  # 例如：只保留 _ km 以内的潜在因果关系
tau_max = 32  # 可根据需要调整
pc_alpha = 0.001  # 显著性阈值
graph = results['graph']
# 参数格式化，用于文件名
tau_str = f'tau{tau_max}'
alpha_str = f'alpha{pc_alpha:.3f}'.replace('.', 'p')  # 0.001 → alpha0p001
dist_str = f'dist{distance_threshold:.1f}km'.replace('.', 'd')  # 5.0 → dist5d0km


# 统计每个节点的自关联数量（节点到自己的因果联系）
node_self_counts = np.zeros(N)
for i in range(N):
    for tau in range(1, tau_max + 1):
        if graph[i, i, tau] == '-->':  # 检查节点i到节点i的关联
            node_self_counts[i] += 1

# 构建连接统计字典（排除自关联）
edge_counts = {}
for i in range(N):
    print(f"节点{i}")
    for j in range(N):
        if i == j:  # 排除自关联
            continue
        count = 0
        for tau in range(1, tau_max + 1):
            if graph[i, j, tau] == '-->':
                count += 1
        if count > 0:
            edge_counts[(i, j)] = count
            print(f"与{j}链接{count}")


"""
因果图
"""
# 创建 NetworkX 图对象
G = nx.DiGraph()

# 添加节点（仅有效位置），记录自关联数量
for i in range(N):
    G.add_node(i, pos=(positions[i, 0], positions[i, 1]),self_count=node_self_counts[i])

# 添加边（仅两个节点都有效的边），排除自关联
for (i, j), count in edge_counts.items():
    if  i != j:  # 确保不是自关联
        G.add_edge(j, i, weight=count)  # 从j到i的有向边

# 获取节点位置
pos = nx.get_node_attributes(G, 'pos')
# 打印统计信息
# 打印统计信息
print("=== 节点自关联、入边、出边统计 ===")
total_tau = sum(edge_counts.values())
# 创建统计表
stats = []
outcoming_node_count = []
for i in G.nodes:
    # 自关联数量
    self_count = G.nodes[i]['self_count']
    # 入边统计
    incoming_edges = [u for u, v in G.edges if v == i]  # 指向该节点的源节点列表
    incoming_edge_weights = [G[u][v]['weight'] for u, v in G.edges if v == i]
    total_incoming_tau = sum(incoming_edge_weights)
    # 出边统计
    outgoing_edges = [v for u, v in G.edges if u == i]  # 该节点指向的目标节点列表
    outgoing_edge_weights = [G[i][v]['weight'] for v in outgoing_edges]
    total_outgoing_tau = sum(outgoing_edge_weights)

    outcoming_node_count.append(len(outgoing_edges))

    # 保存统计
    stats.append({
        'Node': i,
        'SelfLinkCount': self_count,
        'IncomingNodeCount': len(incoming_edges),
        'TotalIncomingTau': total_incoming_tau,
        'OutgoingNodeCount': len(outgoing_edges),
        'TotalOutgoingTau': total_outgoing_tau
    })

    # 打印
    print(f"Node {i}:")
    print(f"  自关联数量: {self_count}")
    print(f"  指向它的节点数: {len(incoming_edges)}")
    print(f"  入边因果联系总数（τ）: {total_incoming_tau}")
    print(f"  被它指向的节点数: {len(outgoing_edges)}")
    print(f"  出边因果联系总数（τ）: {total_outgoing_tau}")
    print("-" * 40)

print("=============================")

# 设置绘图参数
fig, ax = plt.subplots(figsize=(12, 10), dpi=600)

# 设置节点的颜色映射（基于“指向它的节点数”）
if outcoming_node_count:
    max_incoming = max(outcoming_node_count)
    node_norm = Normalize(vmin=0, vmax=max_incoming)
    node_cmap = cm.plasma_r
else:
    node_norm = Normalize(vmin=0, vmax=1)
    node_cmap = cm.plasma_r

# 获取节点颜色
node_colors = [node_cmap(node_norm(c)) for c in outcoming_node_count]

# 获取边的权重（因果联系数量）
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# 设置边的颜色映射（基于入边因果联系总数）
if edge_weights:
    max_edge_weight = max(edge_weights)
    edge_norm = Normalize(vmin=0, vmax=max_edge_weight)
    edge_cmap = cm.plasma_r
else:
    edge_norm = Normalize(vmin=0, vmax=1)
    edge_cmap = cm.plasma_r

# 边颜色
edge_colors = [edge_cmap(edge_norm(w)) for w in edge_weights]


# ✅ 绘制节点（颜色表示自关联数量）
node_colors = [node_cmap(node_norm(c)) for c in outcoming_node_count]
nodes = nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8, ax=ax)

# ✅ 绘制边（颜色表示因果联系数量）
edge_colors = [edge_cmap(edge_norm(w)) for w in edge_weights]
edges = nx.draw_networkx_edges(
    G,
    pos,
    edge_color=[edge_cmap(edge_norm(w)) for w in edge_weights],
    width=0.8,
    alpha=0.8,
    arrows=True,
    #arrowstyle='simple, head_length=0.2, head_width=0.3, tail_width=0.05',
    node_size=50,
    arrowstyle="->",
    arrowsize=5,
    connectionstyle='arc3,rad=0.2',
    ax=ax
)

# ✅ 添加颜色条 - 边因果联系数量
sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm_edges.set_array([])
cbar_edges = fig.colorbar(sm_edges, ax=ax, orientation='vertical', shrink=0.3, pad=0.05)
cbar_edges.set_label('Edge Causal Link Count')

# ✅ 添加颜色条 - 节点出边数量
sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
sm_nodes.set_array([])
cbar_nodes = fig.colorbar(sm_nodes, ax=ax, orientation='vertical', shrink=0.3, pad=0.05)
cbar_nodes.set_label('Number of Outgoing Nodes')

# 可选：添加节点编号
# labels = {i: str(i) for i in G.nodes}
# nx.draw_networkx_labels(G, pos, labels, font_size=6, font_color='black', ax=ax)

# 设置标题和坐标轴
ax.set_title("Causal Graph")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle='--', alpha=0.3)
plt.axis('on')
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.tight_layout()
# 保存图像
# 拼接文件名
filename1 = f'network_causal_graph_{tau_str}_{alpha_str}_{dist_str}.png'

# 保存图像
plt.savefig(os.path.join(output_dir, filename1), dpi=600, bbox_inches='tight')
print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")

"""
因果矩阵热力图
"""
# 创建 N×N 矩阵
N = positions.shape[0]
causal_matrix = np.zeros((N, N))

# 填充矩阵（i → j 的因果联系数量）
for (i, j), count in edge_counts.items():
    causal_matrix[i, j] = count  # 注意：G.add_edge(j, i) 表示 j ← i，即 i → j
# 添加自关联（i → i）
for i in range(N):
    causal_matrix[i, i] = node_self_counts[i]  # node_self_counts[i] 是 i → i 的因果联系数量

# 创建绘图对象
fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8), dpi=1800)

# 绘制热力图
cax = ax_matrix.imshow(causal_matrix, cmap=cm.plasma_r, interpolation='none', aspect='auto')

# 添加颜色条
cbar = fig_matrix.colorbar(cax, ax=ax_matrix)
cbar.set_label('Causal Link Count (τ)')

# 设置标题和坐标轴
ax_matrix.set_title("Node Causal Link Count Matrix (i → j)", fontsize=14)
ax_matrix.set_xlabel("Target Node (j)", fontsize=12)
ax_matrix.set_ylabel("Source Node (i)", fontsize=12)

# 设置坐标轴刻度
ax_matrix.set_xticks(np.arange(N))
ax_matrix.set_yticks(np.arange(N))
ax_matrix.tick_params(axis='both', which='major', labelsize=1)

plt.tight_layout()

filename2 = f'causal_matrix_{tau_str}_{alpha_str}_{dist_str}.png'
plt.savefig(os.path.join(output_dir, filename2), dpi=1800, bbox_inches='tight')
print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")

