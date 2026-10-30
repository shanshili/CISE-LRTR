"""
# 流程特点：简单直接，基础功能实现
- 地理位置生成 → 拓扑构建 → 参数生成 → 数据仿真 → 基础可视化
- 节点数：50，时间步：3000，核心枢纽：5个
- 使用固定半径区分core/variant连接
- 内置基础绘图函数
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite import data_processing as pp

# ================= 配置与文件管理 =================
N = 50  # 节点总数
T = 3000  # 时间步长
num_hubs = 5  # 核心枢纽节点数量
transient_frac = 0.2  # 预热比例

# 设置随机种子
seed = 42
np.random.seed(seed)

# 动态生成文件夹名称
folder_name = f"simulation_hubs_core/N{N}_T{T}_Hubs{num_hubs}_Spatial_CoreVariant"

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
output_dir = os.path.join(project_root, 'dataset', folder_name)
os.makedirs(output_dir, exist_ok=True)

print(f"输出目录已创建: {output_dir}")


# ================= 0. 核心函数 =================
def saturation_func(x):
    """
    使用 tanh 模拟物理饱和，防止数值爆炸。
    输入 x 为线性组合结果，输出被限制在 (-1, 1) 之间 (类似流量归一化)
    """
    return np.tanh(x)


# ================= 1. 生成地理位置与识别 Hubs =================
print("1. 生成地理分布...")
# 生成随机坐标 (100 x 100)
node_positions = np.random.rand(N, 2) * 100
pos_dict = {i: node_positions[i] for i in range(N)}

# 保存坐标
pos_df = pd.DataFrame(node_positions, columns=['x', 'y'])
pos_df['node_id'] = range(N)
pos_df.to_csv(os.path.join(output_dir, 'node_positions.csv'), index=False)

# 计算距离矩阵
dist_matrix = squareform(pdist(node_positions))

# 随机选定 Hubs (Core Nodes)
hub_nodes = np.random.choice(N, num_hubs, replace=False).tolist()
peripheral_nodes = [i for i in range(N) if i not in hub_nodes]
print(f"Core Hubs (Highways): {hub_nodes}")

# ================= 2. 基于空间距离构建拓扑 =================
print("2. 构建空间受限的因果拓扑...")
G_structure = nx.DiGraph()
G_structure.add_nodes_from(range(N))

# 参数设定
core_radius = 80.0  # Hub 之间连接的最大距离 (长程)
local_radius = 20.0  # 普通节点连接的最大距离 (短程)
prob_connect_local = 0.5  # 近距离节点产生因果的概率 (稀疏化)
prob_connect_core = 0.8  # Hub 之间如果有通路，概率较高

# --- 构建边 (Skeleton) ---
for i in range(N):
    for j in range(N):
        if i == j: continue

        dist = dist_matrix[i, j]
        is_i_hub = i in hub_nodes
        is_j_hub = j in hub_nodes

        # A. Core Links (Hub -> Hub)
        if is_i_hub and is_j_hub:
            if dist < core_radius:
                # 只有 80% 的长程 Hub 对会连接 (避免全连接)
                if np.random.rand() < prob_connect_core:
                    G_structure.add_edge(i, j, type='core')

        # B. Hub -> Leaf (辐射)
        elif is_i_hub and not is_j_hub:
            if dist < local_radius * 1.5:  # Hub 辐射范围稍大
                if np.random.rand() < prob_connect_local:
                    G_structure.add_edge(i, j, type='variant')

        # C. Leaf -> Leaf (局部微循环)
        elif not is_i_hub and not is_j_hub:
            if dist < local_radius:
                if np.random.rand() < prob_connect_local:
                    G_structure.add_edge(i, j, type='variant')

print(f"总边数: {G_structure.number_of_edges()}")


# ================= 3. 定义因果连接参数生成器 =================

def generate_causal_params(env_type):
    links = {i: [] for i in range(N)}

    # 1. 自回归 (Inertia) - 始终存在且强力
    for i in range(N):
        # Hub 的自回归系数更高 (惯性大)
        coeff = 0.8 if i in hub_nodes else 0.5
        # 也可以添加多阶滞后
        links[i].append(((i, -1), coeff, saturation_func))

    active_edge_count = 0

    for u, v, data in G_structure.edges(data=True):
        edge_type = data['type']

        # --- Core Links (始终强力) ---
        if edge_type == 'core':
            # Core 连接系数高，且在所有环境中稳定
            links[v].append(((u, -1), 0.6, saturation_func))
            active_edge_count += 1

        # --- Variant Links (环境调制) ---
        else:
            # 使用地理区域模拟环境变化 (West vs East)
            target_x = pos_dict[v][0]
            is_west = target_x < 50

            active = False
            weight = 0.0
            lag = 1

            if env_type == 'Env_A':
                # 环境A：西边活跃，东边拥堵/断连
                if is_west:
                    active = True
                    weight = 0.3  # 正常连接
                    lag = 1
                else:
                    active = True
                    weight = 0.02  # 极弱连接 (几乎断开)
                    lag = 3  # 延迟变大

            elif env_type == 'Env_B':
                # 环境B：东边活跃，西边抑制
                if not is_west:
                    active = True
                    weight = 0.3
                    lag = 1
                else:
                    active = True
                    weight = 0.02
                    lag = 3

            if active:
                # 随机增加一点正负号，模拟抑制或促进
                final_weight = weight if np.random.rand() > 0.3 else -weight
                links[v].append(((u, -lag), final_weight, saturation_func))
                active_edge_count += 1

    return links


# ================= 4. 生成数据并保存 =================
total_steps = int((1 + transient_frac) * T)
print(f"\n生成数据中... (目标长度 T={T}, 实际生成 {total_steps})")

# --- Env A ---
links_A = generate_causal_params('Env_A')
noises_A = np.zeros((total_steps, N))
for i in range(N):
    # Hub 噪声小，边缘噪声大
    scale = 0.1 if i in hub_nodes else 0.3
    noises_A[:, i] = np.random.normal(0, scale, total_steps)

data_A, _ = toys.structural_causal_process(
    links=links_A, T=T, noises=noises_A, seed=10, transient_fraction=transient_frac
)
pd.DataFrame(data_A).to_csv(os.path.join(output_dir, 'env_A.csv'), index=False)

# --- Env B ---
links_B = generate_causal_params('Env_B')
noises_B = np.zeros((total_steps, N))
for i in range(N):
    scale = 0.1 if i in hub_nodes else 0.3
    noises_B[:, i] = np.random.normal(0, scale, total_steps)

data_B, _ = toys.structural_causal_process(
    links=links_B, T=T, noises=noises_B, seed=20, transient_fraction=transient_frac
)
pd.DataFrame(data_B).to_csv(os.path.join(output_dir, 'env_B.csv'), index=False)

print("数据生成完毕。")


# ================= 5. 可视化模块 (修复版) =================

def plot_timeseries_check(data, title, filename):
    """绘制时间序列检查是否有爆炸 (修复版)"""

    # 1. 选定要画的节点索引 (2个Hub + 3个普通节点)
    sample_nodes = hub_nodes[:2] + [n for n in range(N) if n not in hub_nodes][:3]
    sample_nodes = [int(i) for i in sample_nodes]  # 确保是整数索引

    # 2. 提取子集数据
    # data shape is (T, N), we slice columns
    subset_data = data[:, sample_nodes]

    # 3. 准备对应的变量名
    all_var_names = [f'Node_{i}' for i in range(N)]
    subset_var_names = [all_var_names[i] for i in sample_nodes]

    # 4. 创建仅包含子集的 Tigramite DataFrame
    # 这样 plot_timeseries 就会只画这几个变量，不需要额外的参数
    dataframe_subset = pp.DataFrame(
        subset_data,
        var_names=subset_var_names,
        datatime=np.arange(subset_data.shape[0])
    )

    plt.figure(figsize=(15, 6))

    # 5. 调用绘图 (移除 val_keys 参数)
    tp.plot_timeseries(dataframe=dataframe_subset)

    plt.suptitle(title, y=1.02)  # 使用 suptitle 防止遮挡
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

    # 统计信息
    max_val = np.max(np.abs(data))
    print(f"[{title}] Max Value: {max_val:.4f}")
    if max_val > 100:
        print("  !!! 警告: 数据似乎数值爆炸，请检查自回归系数或饱和函数 !!!")


def plot_spatial_causal_graph():
    """绘制带有真实坐标的因果图，区分 Core 和 Variant"""
    plt.figure(figsize=(12, 10))

    # 1. 绘制节点
    # Hub 为大红点，Leaf 为小蓝点
    node_colors = ['#FF4136' if n in hub_nodes else '#7FDBFF' for n in G_structure.nodes()]
    node_sizes = [500 if n in hub_nodes else 100 for n in G_structure.nodes()]
    node_edge_colors = ['black'] * N

    nx.draw_networkx_nodes(G_structure, pos_dict,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors=node_edge_colors, linewidths=1.5)

    nx.draw_networkx_labels(G_structure, pos_dict, font_size=8, font_color='black')

    # 2. 绘制边
    # 分离 Core 和 Variant 边
    core_edges = [(u, v) for u, v, d in G_structure.edges(data=True) if d['type'] == 'core']
    variant_edges = [(u, v) for u, v, d in G_structure.edges(data=True) if d['type'] != 'core']

    # 绘制 Variant (背景，灰色，细)
    nx.draw_networkx_edges(G_structure, pos_dict, edgelist=variant_edges,
                           edge_color='gray', alpha=0.3, width=1.0,
                           arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')

    # 绘制 Core (前景，红色，粗)
    nx.draw_networkx_edges(G_structure, pos_dict, edgelist=core_edges,
                           edge_color='#FF4136', alpha=0.9, width=2.5,
                           arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')

    # 3. 绘制环境分割线
    plt.axvline(x=50, color='green', linestyle='--', alpha=0.6, label='Env Split (West vs East)')

    # 图例构造
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FF4136', lw=2.5, label='Core Link (Invariant)'),
        Line2D([0], [0], color='gray', lw=1, label='Variant Link (Env-dependent)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4136', markersize=10, label='Hub Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7FDBFF', markersize=8, label='Local Node'),
        Line2D([0], [0], color='green', linestyle='--', label='Env Boundary')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f"Spatial Causal Structure (N={N})\nRed=Hubs/Core, Gray=Local/Variant", fontsize=14)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle=':', alpha=0.5)

    save_path = os.path.join(output_dir, 'spatial_structure_ground_truth.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"空间因果图已保存: {save_path}")


# ================= 6. 执行可视化 =================

# 1. 检查数据稳定性
plot_timeseries_check(data_A, 'Env A Time Series (Sample)', 'timeseries_check_A.png')
plot_timeseries_check(data_B, 'Env B Time Series (Sample)', 'timeseries_check_B.png')

# 2. 绘制漂亮的因果图
plot_spatial_causal_graph()

# 3. 导出 Ground Truth 矩阵 (Core Only) 用于评测
gt_matrix = np.zeros((N, N))
for u, v, d in G_structure.edges(data=True):
    if d['type'] == 'core':
        gt_matrix[u, v] = 1  # 标记为1
np.savetxt(os.path.join(output_dir, 'core_ground_truth.csv'), gt_matrix, fmt='%d')

print("\n=== 仿真全部完成 ===")
print(f"Core 定义: 节点 {hub_nodes} 之间的连接。")
print(f"Variant 定义: 局部的短程连接，受地理位置(Env A/B)影响。")
