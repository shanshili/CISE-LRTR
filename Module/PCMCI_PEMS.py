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
from utils.haversine_distance import  compute_haversine_matrix
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pickle

"""
数据读取 (PEMSD7)
"""
# 构建 dataset 文件夹的路径
dataset_dir = os.path.join(current_dir, '..', 'dataset/pemsd7')
# 确保目标路径存在
os.makedirs(dataset_dir, exist_ok=True)
# 构建 PCMCI 结果保存文件夹路径
output_dir = os.path.join(current_dir, 'PCMCI_PEMS_hour_100_FLOW')
# 如果文件夹不存在，则创建
os.makedirs(output_dir, exist_ok=True)

# 加载传感器位置数据
location_file = os.path.join(dataset_dir, 'location2_sampled100.csv')
locations_df = pd.read_csv(location_file)

# 加载传感器流量数据
data_file = os.path.join(dataset_dir, 'hour_sampled100_sensor_data.csv')
sensor_data_df = pd.read_csv(data_file)

# 提取 Total Flow 值并按 Station 分组
flow_data = sensor_data_df.pivot(index='Timestamp', columns='Station', values='Total Flow')
# 处理 NaN 值，用 0 填充
flow_data = flow_data.fillna(0.0)
# 转换为 numpy 数组
data_raw = flow_data.values  # shape: (时间步数, 传感器数量)

# 转置数据使其成为 (传感器数量, 时间步数)
data_transposed = data_raw.T  # shape: (传感器数量, 时间步数)

# 如果需要进一步处理成 (时间步数, 传感器数量) 格式用于 PCMCI
data = data_raw  # shape: (时间步数, 传感器数量)

# 定义变量名（传感器名）
var_names = [f'Sensor_{col}' for col in flow_data.columns]  # N个传感器名

# 创建 DataFrame
dataframe = pp.DataFrame(data, var_names=var_names)

"""
初始化 PCMCI 对象
"""
# 初始化条件独立性检验
cond_ind_test = ParCorr(significance='analytic')
# 初始化 PCMCI 对象
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=1)

# 获取传感器坐标（假设列名为 'latitude', 'longitude'）
positions = locations_df[['Latitude', 'Longitude']].values  # shape: (N, 2)
N = positions.shape[0]

tau_max = 30  # 可根据需要调整
pc_alpha = 0.01  # 显著性阈值

results = pcmci.run_pcmciplus(
    tau_max=tau_max,
    pc_alpha=pc_alpha,
)

# 参数格式化，用于文件名
tau_str = f'tau{tau_max}'
alpha_str = f'alpha{pc_alpha:.3f}'.replace('.', 'p')  # 0.001 → alpha0p001
filename3 = f'pcmci_results_{tau_str}_{alpha_str}.pkl'
pickle_file = os.path.join(output_dir, filename3)

with open(pickle_file, 'wb') as f:
    pickle.dump(results, f)
print(f"完整 PCMCI 结果已保存为 pickle 文件: {pickle_file}")

with open(pickle_file, 'rb') as f:
    loaded_results = pickle.load(f)

graph = results['graph']
val_matrix = loaded_results['val_matrix']

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
        count = 0
        for tau in range(1, tau_max + 1):
            if graph[i, j, tau] == '-->':  # 节点 i 在时间 t-τ 时刻对节点 j 在时间 t 时刻有因果影响
                count += 1
                print(graph[i, j, tau])
                print(f"节点{i}与{j}因果效应{val_matrix[i, j, tau]}")
        if count > 0:
            edge_counts[(i, j)] = count
            print(f"与{j}链接{count}")

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

# 计算每个节点的出边数量（出边表示该节点影响其他节点的数量）
out_degree = np.zeros(N)
for u, v, key, data in G.edges(keys=True, data=True):
    out_degree[u] += 1  # 节点u有一条出边
vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1

# 设置绘图参数
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=out_degree,
                       cmap=plt.cm.autumn, alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=8)

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
                           alpha=0.7, arrows=True, arrowsize=10)

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
                           connectionstyle=f'arc3,rad=0.3')  # 弯曲的自循环

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
cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
cbar.set_label('Causal Effect', rotation=270, labelpad=20)

# 添加节点出边数量颜色图例
sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.autumn, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
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
filename1 = f'network_causal_graph_{tau_str}_{alpha_str}.png'
plt.savefig(os.path.join(output_dir, filename1), dpi=600, bbox_inches='tight')
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

filename2 = f'causal_matrix_{tau_str}_{alpha_str}.png'
plt.savefig(os.path.join(output_dir, filename2), dpi=1800, bbox_inches='tight')
print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")
