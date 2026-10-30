"""
# 流程特点：功能完整，可视化丰富，数据持久化
- 拓扑构建 → 高级参数生成 → 数据仿真 → 多样可视化 → Ground Truth保存
- 节点数：100，时间步：10000，核心枢纽：15个
- 引入PCMCI格式Ground Truth保存
- 支持SVG/PNG双重输出格式
- 包含core子图提取和评估功能

"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import data_processing as pp
from tigramite import plotting as tp
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 16            # 全局基础字号
mpl.rcParams['axes.titlesize'] = 16       # 标题
mpl.rcParams['axes.labelsize'] = 16       # 坐标轴标签
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] =12
mpl.rcParams['figure.titlesize'] = 12


# ==============================================================================
# 0. 配置与路径
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
module_dir = os.path.join(project_root, 'Module')

# 确保能找到 helper 模块 (如果有的话，本脚本目前设计为独立运行)
if module_dir not in sys.path:
    sys.path.append(module_dir)

# 参数设置
N = 100
T = 10000
num_hubs = 15
transient_frac = 0.2
TAU_MAX = 5

seed = 42
np.random.seed(seed)

folder_name = f"simulation_hubs_core/N{N}_T{T}_Hubs{num_hubs}_Viz_Final5"
output_dir = os.path.join(project_root, 'dataset', folder_name)
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")


# ================= 1. 核心工具函数 =================
def saturation_func(x):
    return np.tanh(x)


import pickle

def build_and_save_core_subgraph_pcmci(
    links: dict,
    G_structure: nx.DiGraph,
    N: int,
    tau_max: int,
    hub_nodes: list,
    env_name: str,
    output_dir: str,
    filename: str = None
):
    """
    从真值 links（tigramite toys 的 causal params）中提取 core 子图，
    保存为 PCMCI-like 格式：{'graph': [N,N,L], 'val_matrix': [N,N,L], ...}

    core 判定：G_structure 边属性 type == 'core'
    lag：从 links 里解析（core 一般是 lag=1）
    """
    L = tau_max + 1  # 你评估代码用的是 [0..tau_max]
    graph_core = np.full((N, N, L), "", dtype=object)
    val_core = np.zeros((N, N, L), dtype=float)

    # 方便：先把 core 边集合拿出来
    core_edge_set = set()
    for u, v, data in G_structure.edges(data=True):
        if data.get("type") == "core":
            core_edge_set.add((u, v))

    # 从 links（按 target 汇总 parents）中提取对应 core 边的 weight/lag
    # links[v] 里是 ((u, -lag), coeff, func)
    for v in range(N):
        for (u, lag_neg), coeff, _ in links.get(v, []):
            lag = abs(int(lag_neg))
            if (u, v) in core_edge_set:
                if lag < L:
                    graph_core[u, v, lag] = "-->"
                    val_core[u, v, lag] = float(coeff)

    # 输出文件名
    if filename is None:
        filename = f"{env_name}_gt_core_pcmci_tau{tau_max}.pkl"
    out_path = os.path.join(output_dir, filename)

    data = {
        "graph": graph_core,
        "val_matrix": val_core,
        "core_nodes": list(hub_nodes),
        "meta": {
            "env": env_name,
            "N": N,
            "T": T,
            "tau_max": tau_max,
            "seed": seed,
            "core_radius": core_radius,
            "local_radius": local_radius,
            "definition": "Edges with G_structure.edge['type']=='core'; weights/lags from links",
        }
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    # 也可顺带保存一份可读的 edge list CSV
    rows = []
    for u in range(N):
        for v in range(N):
            for lag in range(L):
                if graph_core[u, v, lag] == "-->":
                    rows.append({"src": u, "dst": v, "lag": lag, "weight": val_core[u, v, lag], "type": "core"})
    df = pd.DataFrame(rows)
    csv_path = out_path.replace(".pkl", ".csv")
    df.to_csv(csv_path, index=False)

    print(f"  ✅ Core GT 子图已保存: {out_path}")
    print(f"  ✅ Core GT 边列表已保存: {csv_path} (edges={len(rows)})")



# ================= 2. 拓扑结构生成 =================
print("1. 构建拓扑结构...")
node_positions = np.random.rand(N, 2) * 100
pos_dict = {i: (node_positions[i, 0], node_positions[i, 1]) for i in range(N)}

# 保存坐标
pd.DataFrame(node_positions, columns=['x', 'y']).assign(node_id=range(N)) \
    .to_csv(os.path.join(output_dir, 'node_positions.csv'), index=False)

dist_matrix = squareform(pdist(node_positions))
hub_nodes = np.random.choice(N, num_hubs, replace=False).tolist()
print(f"Core Hubs: {hub_nodes}")

# 构建结构图 (用于记录 core/variant 类型)
G_structure = nx.DiGraph()
G_structure.add_nodes_from(range(N))

core_radius = 20.0
local_radius = 20.0

for i in range(N):
    for j in range(N):
        if i == j: continue
        dist = dist_matrix[i, j]
        is_i_hub = i in hub_nodes
        is_j_hub = j in hub_nodes

        # Core Links
        if is_i_hub and is_j_hub and dist < core_radius:
            if np.random.rand() < 0.8:
                G_structure.add_edge(i, j, type='core')
        # Variant Links
        elif dist < local_radius:
            if np.random.rand() < 0.5:
                G_structure.add_edge(i, j, type='variant')


# ================= 3. 高级绘图函数 (完全重构) =================

# def draw_custom_causal_graph(links_dict, node_positions, hub_list, g_struct,
#                              title, filename,
#                              mode='full',  # 'full' or 'core_highlight'
#                              show_labels=True):
#     """
#     通用绘图函数，支持全图展示和核心骨干高亮展示，支持标签开关
#     """
#     N = node_positions.shape[0]
#     G = nx.MultiDiGraph()
#
#     # 1. 构建图数据结构
#     G.add_nodes_from(range(N))
#
#     # 解析 links_dict 到 NetworkX
#     for target, parents in links_dict.items():
#         for (source, lag_neg), coeff, _ in parents:
#             # Tigramite返回的是负lag，转为正
#             lag = abs(lag_neg)
#             # 判断连接类型 (从 G_structure 获取，如果没有则默认为 variant)
#             etype = 'variant'
#             if g_struct.has_edge(source, target):
#                 etype = g_struct.edges[source, target].get('type', 'variant')
#             # 只有自回归不在 G_structure 中，视情况处理
#             if source == target: etype = 'self'
#
#             G.add_edge(source, target, key=lag, weight=coeff, lag=lag, type=etype)
#
#     # 2. 节点样式
#     # 计算出度用于颜色
#     out_degree = np.zeros(N)
#     for u, v, k, d in G.edges(keys=True, data=True):
#         if u != v: out_degree[u] += 1
#
#     plt.figure(figsize=(14, 12), dpi=300)
#     ax = plt.gca()
#
#     # 绘制节点
#     if mode == 'core_highlight':
#         # 高亮模式：Hub是红色，普通是蓝色
#         node_colors = ['#d62728' if i in hub_list else '#1f77b4' for i in range(N)]
#         node_sizes = [400 if i in hub_list else 100 for i in range(N)]
#     else:
#         # 全图模式：使用出度热力图
#         vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
#         node_colors = out_degree
#         node_sizes = 100 + out_degree * 10
#
#     # 绘制节点
#     nx.draw_networkx_nodes(G, pos_dict, node_size=node_sizes,
#                            node_color=node_colors,
#                            cmap=plt.cm.YlGn if mode == 'full' else None,
#                            edgecolors='white', linewidths=1.0, alpha=0.9)
#
#     nx.draw_networkx_labels(G, pos_dict, font_size=8, font_color='black')
#
#     # 3. 边的数据准备
#     # 颜色映射
#     all_weights = [d['weight'] for u, v, k, d in G.edges(keys=True, data=True)]
#     if all_weights:
#         max_val = max(abs(min(all_weights)), abs(max(all_weights)))
#         norm = Normalize(vmin=-max_val, vmax=max_val)
#     else:
#         norm = Normalize(vmin=-1, vmax=1)
#     cmap = plt.cm.seismic
#
#     # 分离边
#     edges_bg = []  # 背景边 (灰色)
#     edges_fg = []  # 前景边 (彩色)
#     edges_self = []  # 自环
#
#     for u, v, k, d in G.edges(keys=True, data=True):
#         if u == v:
#             edges_self.append((u, v, k, d))
#             continue
#
#         if mode == 'core_highlight':
#             if d['type'] == 'core':
#                 edges_fg.append((u, v, k, d))
#             else:
#                 edges_bg.append((u, v, k, d))
#         else:
#             # 全图模式，所有非自环都是前景
#             edges_fg.append((u, v, k, d))
#
#     # 4. 绘制边
#
#     # (A) 绘制背景边 (Variant in highlight mode) -> 灰色，细，无标签
#     if edges_bg:
#         for u, v, k, d in edges_bg:
#             width = 0.5 + abs(d['weight'])  # 稍微细一点
#             alpha = 0.2  # 很淡
#             nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
#                                    edge_color='gray', width=width, alpha=alpha,
#                                    arrowsize=10)
#
#     # (B) 绘制前景边 (Core or All) -> 彩色，粗
#     for u, v, k, d in edges_fg:
#         weight = d['weight']
#         color = cmap(norm(weight))
#         width = 1.5 + abs(weight) * 6.0  # 粗线条
#         alpha = 0.8
#
#         nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
#                                edge_color=[color], width=width, alpha=alpha,
#                                arrowsize=15)
#
#     # (C) 绘制自环
#     for u, v, k, d in edges_self:
#         weight = d['weight']
#         color = cmap(norm(weight))
#         width = 1.0 + abs(weight) * 4.0
#         # 自环特殊绘制 (虽然 nx 对自环支持一般，这里用 arrow 模拟)
#         x, y = pos_dict[u]
#         plt.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
#                      arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
#                                      mutation_scale=width * 3 + 10))
#
#     # 5. 绘制标签 (仅针对前景边)
#     if show_labels:
#         processed_edges = edges_fg + edges_self
#         for u, v, k, d in processed_edges:
#             weight = d['weight']
#             lag = d['lag']
#             sign = '+' if weight >= 0 else '-'
#             label_text = f"{sign}τ{lag}"
#             label_color = 'red' if weight >= 0 else 'blue'
#
#             # 计算位置
#             if u == v:
#                 x, y = pos_dict[u]
#                 lbl_x, lbl_y = x - 1, y - 1
#             else:
#                 x1, y1 = pos_dict[u]
#                 x2, y2 = pos_dict[v]
#                 # 线性中点
#                 mid_x = (x1 + x2) / 2
#                 mid_y = (y1 + y2) / 2
#
#                 # 针对多 Lag 的偏移处理 (简易版: 根据 lag 沿法线偏移)
#                 # 计算向量
#                 dx, dy = x2 - x1, y2 - y1
#                 dist = np.sqrt(dx * dx + dy * dy)
#                 if dist > 0:
#                     # 法向量 (-dy, dx)
#                     perp_x, perp_y = -dy / dist, dx / dist
#                     # 偏移量: rad * dist * 0.5 大概是弧顶位置
#                     offset = 0.01 * k * dist * 0.5
#                     lbl_x = mid_x + perp_x * offset
#                     lbl_y = mid_y + perp_y * offset
#                 else:
#                     lbl_x, lbl_y = mid_x, mid_y
#
#             # 绘制带背景框的文字 (防止被线遮挡)
#             plt.text(lbl_x, lbl_y, label_text,
#                      fontsize=6, ha='center', va='center', color=label_color,
#                      fontweight='bold',
#                      bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))
#
#     # 6. 图例
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
#     cbar.set_label('Connection Weight', rotation=270, labelpad=15)
#
#     if mode == 'full':
#         sm_node = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=np.max(out_degree)))
#         sm_node.set_array([])
#         cbar2 = plt.colorbar(sm_node, ax=ax, shrink=0.7, location='left')
#         cbar2.set_label('Node Out-Degree', rotation=90, labelpad=15)
#
#     ax.set_title(title)
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, filename), dpi=300)
#     plt.close()
#     print(f"  --> 图表已保存: {filename}")

def draw_custom_causal_graph(links_dict, node_positions, hub_list, g_struct,
                             title, filename_base,
                             mode='full',  # 'full' or 'core_highlight'
                             show_labels=True):
    """
    通用绘图函数，支持 SVG 输出，1:1 比例，紧凑颜色条
    """
    N = node_positions.shape[0]
    G = nx.MultiDiGraph()

    # 1. 构建图数据结构 (保持不变)
    G.add_nodes_from(range(N))
    for target, parents in links_dict.items():
        for (source, lag_neg), coeff, _ in parents:
            lag = abs(lag_neg)
            etype = 'variant'
            if g_struct.has_edge(source, target):
                etype = g_struct.edges[source, target].get('type', 'variant')
            if source == target: etype = 'self'
            G.add_edge(source, target, key=lag, weight=coeff, lag=lag, type=etype)

    # 2. 节点样式 (保持不变)
    out_degree = np.zeros(N)
    for u, v, k, d in G.edges(keys=True, data=True):
        if u != v: out_degree[u] += 1

    # --- 绘图初始化 ---
    # 使用正方形画布，预留右侧给 Colorbar
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # !!! 关键修改：强制 1:1 比例 !!!
    # 'box' 意味着坐标轴的变化比例是相同的，且调整 box 形状适应
    ax.set_aspect('equal', adjustable='box')

    # 绘制节点
    if mode == 'core_highlight':
        node_colors = ['#d62728' if i in hub_list else '#1f77b4' for i in range(N)]
        node_sizes = [400 if i in hub_list else 100 for i in range(N)]
        cmap_nodes = None
    else:
        vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
        node_colors = out_degree
        node_sizes = 100 + out_degree * 10
        cmap_nodes = plt.cm.YlGn

    # 绘制
    nx.draw_networkx_nodes(G, pos_dict, node_size=node_sizes,
                           node_color=node_colors, cmap=cmap_nodes,
                           edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos_dict, font_size=8, font_color='black', ax=ax)

    # 3. 边的数据准备 (保持不变)
    all_weights = [d['weight'] for u, v, k, d in G.edges(keys=True, data=True)]
    if all_weights:
        max_val = max(abs(min(all_weights)), abs(max(all_weights)))
        norm = Normalize(vmin=-max_val, vmax=max_val)
    else:
        norm = Normalize(vmin=-1, vmax=1)
    cmap_edges = plt.cm.seismic

    # 分离边
    edges_bg, edges_fg, edges_self = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            edges_self.append((u, v, k, d))
        elif mode == 'core_highlight':
            if d['type'] == 'core':
                edges_fg.append((u, v, k, d))
            else:
                edges_bg.append((u, v, k, d))
        else:
            edges_fg.append((u, v, k, d))

    # 4. 绘制边
    # 背景边
    if edges_bg:
        for u, v, k, d in edges_bg:
            nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                                   edge_color='gray', width=0.5 + abs(d['weight']), alpha=0.2,
                                   arrowsize=10, ax=ax)

    # 前景边
    for u, v, k, d in edges_fg:
        weight = d['weight']
        color = cmap_edges(norm(weight))
        width = 1.5 + abs(weight) * 2.0
        nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                               edge_color=[color], width=width, alpha=0.8,
                               arrowsize=15, ax=ax)

    # 自环
    for u, v, k, d in edges_self:
        weight = d['weight']
        color = cmap_edges(norm(weight))
        width = 0.5 + abs(weight) * 2.0
        x, y = pos_dict[u]
        ax.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
                     arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
                                     mutation_scale=width * 3 + 10))
    # 5. 绘制标签
    if show_labels:
        processed_edges = edges_fg + edges_self
        for u, v, k, d in processed_edges:
            weight = d['weight']
            lag = d['lag']
            sign = '+' if weight >= 0 else '-'
            label_text = f"{sign}τ{lag}"
            label_color = 'red' if weight >= 0 else 'blue'

            if u == v:
                x, y = pos_dict[u]
                lbl_x, lbl_y = x - 1, y - 1
            else:
                x1, y1 = pos_dict[u]
                x2, y2 = pos_dict[v]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    perp_x, perp_y = -dy / dist, dx / dist
                    offset = 0.01 * k * dist * 0.5
                    lbl_x, lbl_y = mid_x + perp_x * offset, mid_y + perp_y * offset
                else:
                    lbl_x, lbl_y = mid_x, mid_y

            ax.text(lbl_x, lbl_y, label_text, fontsize=5, ha='center', va='center',
                    color=label_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))

    # 6. 图例优化 (Colorbars)
    # 这里的 fraction 和 pad 是控制位置的关键
    # fraction: colorbar 长度占轴的比例 (0.046 是个魔法数字，通常配合正方形轴很好看)
    # pad: colorbar 和轴之间的距离

    sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Edge Weight', rotation=270, labelpad=10)

    if mode == 'full':
        sm_node = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=np.max(out_degree)))
        sm_node.set_array([])
        # 放在左边
        cbar2 = plt.colorbar(sm_node, ax=ax, location='left', fraction=0.040, pad=0.05)
        cbar2.set_label('Node Out-Degree', rotation=90, labelpad=7)

    ax.set_title(title, pad=15)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # 设置坐标轴范围略微宽一点，防止标签被切
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    ax.grid(True, linestyle='--', alpha=0.2)

    # 保存逻辑
    plt.tight_layout()

    # 保存 SVG (矢量)
    svg_path = os.path.join(output_dir, filename_base + ".svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    # 保存 PNG (预览)
    png_path = os.path.join(output_dir, filename_base + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"  --> 图表已保存: {filename_base} (.svg + .png)")
def draw_custom_causal_graph4(links_dict, node_positions, hub_list, g_struct,
                             title, filename_base,
                             mode='full',  # 'full' or 'core_highlight'
                             show_labels=True):
    """
    通用绘图函数，支持 SVG 输出，1:1 比例，紧凑颜色条
    参数:
        links_dict: 包含节点间连接关系的字典
        node_positions: 节点位置的坐标数组
        hub_list: 核心节点的列表
        g_struct: 图的数据结构
        title: 图表的标题
        filename_base: 输出文件的基础名称
        mode: 显示模式，'full' 或 'core_highlight'
        show_labels: 是否显示边的标签
    返回:
        无返回值，直接保存图表文件
    """
    N = node_positions.shape[0]  # 获取节点数量
    G = nx.MultiDiGraph()  # 创建有向多重图

    # 1. 构建图数据结构 (保持不变)
    G.add_nodes_from(range(N))  # 添加所有节点
    for target, parents in links_dict.items():  # 遍历所有连接关系
        for (source, lag_neg), coeff, _ in parents:  # 获取源节点、延迟和系数
            lag = abs(lag_neg)  # 获取延迟的绝对值
            etype = 'variant'  # 默认边类型
            if g_struct.has_edge(source, target):  # 检查在结构图中是否存在该边
                etype = g_struct.edges[source, target].get('type', 'variant')  # 获取边类型
            if source == target: etype = 'self'  # 自环边类型
            # 添加边到图中，包含延迟、权重和类型信息
            G.add_edge(source, target, key=lag, weight=coeff, lag=lag, type=etype)

    # 2. 节点样式 (保持不变)
    out_degree = np.zeros(N)  # 初始化出度数组
    for u, v, k, d in G.edges(keys=True, data=True):  # 遍历所有边
        if u != v: out_degree[u] += 1  # 计算节点的出度

    # --- 绘图初始化 ---
    # 使用正方形画布，预留右侧给 Colorbar
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # !!! 关键修改：强制 1:1 比例 !!!
    # 'box' 意味着坐标轴的变化比例是相同的，且调整 box 形状适应
    ax.set_aspect('equal', adjustable='box')

    # 绘制节点
    if mode == 'core_highlight':
        node_colors = ['#d62728' if i in hub_list else '#1f77b4' for i in range(N)]
        node_sizes = [400 if i in hub_list else 100 for i in range(N)]
        cmap_nodes = None
    else:
        vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
        node_colors = out_degree
        node_sizes = 300 + out_degree * 10
        cmap_nodes = plt.cm.YlGn

    # 3. 边的数据准备 (保持不变)
    all_weights = [d['weight'] for u, v, k, d in G.edges(keys=True, data=True)]
    if all_weights:
        max_val = max(abs(min(all_weights)), abs(max(all_weights)))
        norm = Normalize(vmin=-max_val, vmax=max_val)
    else:
        norm = Normalize(vmin=-1, vmax=1)
    cmap_edges = plt.cm.seismic

    # 分离边
    edges_bg, edges_fg, edges_self = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            edges_self.append((u, v, k, d))
        elif mode == 'core_highlight':
            if d['type'] == 'core':
                edges_fg.append((u, v, k, d))
            else:
                edges_bg.append((u, v, k, d))
        else:
            edges_fg.append((u, v, k, d))

    # 4. 绘制边
    # 背景边
    if edges_bg:
        for u, v, k, d in edges_bg:
            nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                                   edge_color='gray', width=0.5 + abs(d['weight']), alpha=0.2,
                                   arrowsize=10, ax=ax)

    # 前景边
    for u, v, k, d in edges_fg:
        weight = d['weight']
        color = cmap_edges(norm(weight))
        width = 1.5 + abs(weight) * 2.0
        nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                               edge_color=[color], width=width, alpha=0.8,
                               arrowsize=15, ax=ax)

    # 自环
    for u, v, k, d in edges_self:
        weight = d['weight']
        color = cmap_edges(norm(weight))
        width = 0.5 + abs(weight) * 2.0
        x, y = pos_dict[u]
        ax.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
                     arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
                                     mutation_scale=width * 3 + 10))
    # 5. 绘制标签
    if show_labels:
        processed_edges = edges_fg + edges_self
        for u, v, k, d in processed_edges:
            weight = d['weight']
            lag = d['lag']
            sign = '+' if weight >= 0 else '-'
            label_text = f"{sign}τ{lag}"
            label_color = 'red' if weight >= 0 else 'blue'

            if u == v:
                x, y = pos_dict[u]
                lbl_x, lbl_y = x - 1, y - 1
            else:
                x1, y1 = pos_dict[u]
                x2, y2 = pos_dict[v]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    perp_x, perp_y = -dy / dist, dx / dist
                    offset = 0.01 * k * dist * 0.5
                    lbl_x, lbl_y = mid_x + perp_x * offset, mid_y + perp_y * offset
                else:
                    lbl_x, lbl_y = mid_x, mid_y

            ax.text(lbl_x, lbl_y, label_text, fontsize=5, ha='center', va='center',
                    color=label_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))

    # 绘制
    nx.draw_networkx_nodes(G, pos_dict, node_size=node_sizes,
                           node_color=node_colors, cmap=cmap_nodes,
                           edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos_dict, font_size=12, font_color='black', font_family='Times New Roman',ax=ax)

    # 6. 图例优化 (Colorbars)
    # 这里的 fraction 和 pad 是控制位置的关键
    # fraction: colorbar 长度占轴的比例 (0.046 是个魔法数字，通常配合正方形轴很好看)
    # pad: colorbar 和轴之间的距离

    sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Edge Weight', fontsize=16,rotation=270, labelpad=10)

    if mode == 'full':
        sm_node = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=np.max(out_degree)))
        sm_node.set_array([])
        # 放在左边
        cbar2 = plt.colorbar(sm_node, ax=ax, location='left', fraction=0.040, pad=0.05)
        cbar2.set_label('Node Out-Degree',fontsize=16, rotation=90, labelpad=7)

    ax.set_title(title, pad=15)
    ax.set_xlabel("X Coordinate", fontsize=16, fontname='Times New Roman')
    ax.set_ylabel("Y Coordinate", fontsize=16, fontname='Times New Roman')

    # 设置坐标轴范围略微宽一点，防止标签被切
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    ax.grid(True, linestyle='--', alpha=0.2)

    # 保存逻辑
    plt.tight_layout()

    # 保存 SVG (矢量)
    svg_path = os.path.join(output_dir, filename_base + ".svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    # 保存 PNG (预览)
    png_path = os.path.join(output_dir, filename_base + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"  --> 图表已保存: {filename_base} (.svg + .png)")


def draw_custom_causal_graph2(links_dict, node_positions, hub_list, g_struct,
                             title, filename_base,
                             mode='full',  # 'full' or 'core_highlight'
                             show_labels=True):
    """
    通用绘图函数，支持 SVG 输出，1:1 比例，紧凑颜色条
    在 core_highlight 模式下不绘制自环
    """
    N = node_positions.shape[0]
    G = nx.MultiDiGraph()

    # 1. 构建图数据结构 (保持不变)
    G.add_nodes_from(range(N))
    for target, parents in links_dict.items():
        for (source, lag_neg), coeff, _ in parents:
            lag = abs(lag_neg)
            etype = 'variant'
            if g_struct.has_edge(source, target):
                etype = g_struct.edges[source, target].get('type', 'variant')
            if source == target: etype = 'self'
            G.add_edge(source, target, key=lag, weight=coeff, lag=lag, type=etype)

    # 2. 节点样式 (保持不变)
    out_degree = np.zeros(N)
    for u, v, k, d in G.edges(keys=True, data=True):
        if u != v: out_degree[u] += 1

    # --- 绘图初始化 ---
    # 使用正方形画布，预留右侧给 Colorbar
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # !!! 关键修改：强制 1:1 比例 !!!
    # 'box' 意味着坐标轴的变化比例是相同的，且调整 box 形状适应
    ax.set_aspect('equal', adjustable='box')

    # 绘制节点
    if mode == 'core_highlight':
        node_colors = ['#d62728' if i in hub_list else '#1f77b4' for i in range(N)]
        node_sizes = [400 if i in hub_list else 100 for i in range(N)]
        cmap_nodes = None
    else:
        vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
        node_colors = out_degree
        node_sizes = 100 + out_degree * 10
        cmap_nodes = plt.cm.YlGn

    # 绘制
    nx.draw_networkx_nodes(G, pos_dict, node_size=node_sizes,
                           node_color=node_colors, cmap=cmap_nodes,
                           edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos_dict, font_size=8, font_color='black', ax=ax)

    # 3. 边的数据准备 (保持不变)
    all_weights = [d['weight'] for u, v, k, d in G.edges(keys=True, data=True)]
    if all_weights:
        max_val = max(abs(min(all_weights)), abs(max(all_weights)))
        norm = Normalize(vmin=-max_val, vmax=max_val)
    else:
        norm = Normalize(vmin=-1, vmax=1)
    cmap_edges = plt.cm.seismic

    # 分离边 - 修改：在 core_highlight 模式下不包含自环
    edges_bg, edges_fg, edges_self = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            if mode != 'core_highlight':  # 只在非 core_highlight 模式下添加自环
                edges_self.append((u, v, k, d))
        elif mode == 'core_highlight':
            if d['type'] == 'core':
                edges_fg.append((u, v, k, d))
            else:
                edges_bg.append((u, v, k, d))
        else:
            edges_fg.append((u, v, k, d))

    # 4. 绘制边
    # 背景边
    if edges_bg:
        for u, v, k, d in edges_bg:
            nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                                   edge_color='gray', width=0.5 + abs(d['weight']), alpha=0.2,
                                   arrowsize=10, ax=ax)

    # 前景边
    for u, v, k, d in edges_fg:
        weight = d['weight']
        color = cmap_edges(norm(weight))
        width = 1.5 + abs(weight) * 2.0
        nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                               edge_color=[color], width=width, alpha=0.8,
                               arrowsize=15, ax=ax)

    # 自环 - 在 core_highlight 模式下不绘制
    if mode != 'core_highlight':  # 仅在非 core_highlight 模式下绘制自环
        for u, v, k, d in edges_self:
            weight = d['weight']
            color = cmap_edges(norm(weight))
            width = 0.5 + abs(weight) * 2.0
            x, y = pos_dict[u]
            ax.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
                        arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
                                        mutation_scale=width * 3 + 10))

    # 5. 绘制标签 - 修改：在 core_highlight 模式下不包含自环标签
    if show_labels:
        processed_edges = edges_fg  # 在 core_highlight 模式下不包含自环
        if mode != 'core_highlight':
            processed_edges = edges_fg + edges_self

        for u, v, k, d in processed_edges:
            weight = d['weight']
            lag = d['lag']
            sign = '+' if weight >= 0 else '-'
            label_text = f"{sign}τ{lag}"
            label_color = 'red' if weight >= 0 else 'blue'

            if u == v:
                x, y = pos_dict[u]
                lbl_x, lbl_y = x - 1, y - 1
            else:
                x1, y1 = pos_dict[u]
                x2, y2 = pos_dict[v]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    perp_x, perp_y = -dy / dist, dx / dist
                    offset = 0.01 * k * dist * 0.5
                    lbl_x, lbl_y = mid_x + perp_x * offset, mid_y + perp_y * offset
                else:
                    lbl_x, lbl_y = mid_x, mid_y

            ax.text(lbl_x, lbl_y, label_text, fontsize=5, ha='center', va='center',
                    color=label_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))

    # 6. 图例优化 (Colorbars)
    # 这里的 fraction 和 pad 是控制位置的关键
    # fraction: colorbar 长度占轴的比例 (0.046 是个魔法数字，通常配合正方形轴很好看)
    # pad: colorbar 和轴之间的距离

    sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Edge Weight', rotation=270, labelpad=10)

    if mode == 'full':
        sm_node = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=np.max(out_degree)))
        sm_node.set_array([])
        # 放在左边
        cbar2 = plt.colorbar(sm_node, ax=ax, location='left', fraction=0.040, pad=0.05)
        cbar2.set_label('Node Out-Degree', rotation=90, labelpad=7)

    ax.set_title(title, pad=15)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # 设置坐标轴范围略微宽一点，防止标签被切
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    ax.grid(True, linestyle='--', alpha=0.2)

    # 保存逻辑
    plt.tight_layout()

    # 保存 SVG (矢量)
    svg_path = os.path.join(output_dir, filename_base + ".svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    # 保存 PNG (预览)
    png_path = os.path.join(output_dir, filename_base + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"  --> 图表已保存: {filename_base} (.svg + .png)")

def draw_custom_causal_graph3(links_dict, node_positions, hub_list, g_struct,
                             title, filename_base,
                             mode='full',  # 'full' or 'core_highlight'
                             show_labels=True):
    """
    通用绘图函数，支持 SVG 输出，1:1 比例，紧凑颜色条
    在 core_highlight 模式下不绘制自环
    """
    N = node_positions.shape[0]
    G = nx.MultiDiGraph()

    # 1. 构建图数据结构 (保持不变)
    G.add_nodes_from(range(N))
    for target, parents in links_dict.items():
        for (source, lag_neg), coeff, _ in parents:
            lag = abs(lag_neg)
            etype = 'variant'
            if g_struct.has_edge(source, target):
                etype = g_struct.edges[source, target].get('type', 'variant')
            if source == target: etype = 'self'
            G.add_edge(source, target, key=lag, weight=coeff, lag=lag, type=etype)

    # 2. 节点样式 (保持不变)
    out_degree = np.zeros(N)
    for u, v, k, d in G.edges(keys=True, data=True):
        if u != v: out_degree[u] += 1

    # --- 绘图初始化 ---
    # 使用正方形画布，预留右侧给 Colorbar
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # !!! 关键修改：强制 1:1 比例 !!!
    # 'box' 意味着坐标轴的变化比例是相同的，且调整 box 形状适应
    ax.set_aspect('equal', adjustable='box')

    # 绘制节点
    if mode == 'core_highlight':
        node_colors = ['#d62728' if i in hub_list else '#1f77b4' for i in range(N)]
        node_sizes = [500 if i in hub_list else 200 for i in range(N)]
        cmap_nodes = None
    else:
        vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
        node_colors = out_degree
        node_sizes = 200 + out_degree * 10
        cmap_nodes = plt.cm.YlGn

    # 3. 边的数据准备 (保持不变)
    all_weights = [d['weight'] for u, v, k, d in G.edges(keys=True, data=True)]
    if all_weights:
        max_val = max(abs(min(all_weights)), abs(max(all_weights)))
        norm = Normalize(vmin=-max_val, vmax=max_val)
    else:
        norm = Normalize(vmin=-1, vmax=1)
    cmap_edges = plt.cm.seismic

    # 分离边 - 修改：在 core_highlight 模式下不包含自环
    edges_bg, edges_fg, edges_self = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            if mode != 'core_highlight':  # 只在非 core_highlight 模式下添加自环
                edges_self.append((u, v, k, d))
        elif mode == 'core_highlight':
            if d['type'] == 'core':
                edges_fg.append((u, v, k, d))
            else:
                edges_bg.append((u, v, k, d))
        else:
            edges_fg.append((u, v, k, d))

    # 4. 绘制边
    # 背景边
    if edges_bg:
        for u, v, k, d in edges_bg:
            nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                                   edge_color='gray', width=1.5 + abs(d['weight']), alpha=0.2,
                                   arrowsize=10, ax=ax)

    # 前景边
    for u, v, k, d in edges_fg:
        weight = d['weight']
        color = cmap_edges(norm(weight))
        width = 3 + abs(weight) * 2.0
        nx.draw_networkx_edges(G, pos_dict, edgelist=[(u, v)],
                               edge_color=[color], width=width, alpha=0.8,
                               arrowsize=15, ax=ax)

    # 自环 - 在 core_highlight 模式下不绘制
    if mode != 'core_highlight':  # 仅在非 core_highlight 模式下绘制自环
        for u, v, k, d in edges_self:
            weight = d['weight']
            color = cmap_edges(norm(weight))
            width = 0.5 + abs(weight) * 2.0
            x, y = pos_dict[u]
            ax.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
                        arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
                                        mutation_scale=width * 3 + 10))

    # 5. 绘制标签 - 修改：在 core_highlight 模式下不包含自环标签
    if show_labels:
        processed_edges = edges_fg  # 在 core_highlight 模式下不包含自环
        if mode != 'core_highlight':
            processed_edges = edges_fg + edges_self

        for u, v, k, d in processed_edges:
            weight = d['weight']
            lag = d['lag']
            sign = '+' if weight >= 0 else '-'
            label_text = f"{sign}τ{lag}"
            label_color = 'red' if weight >= 0 else 'blue'

            if u == v:
                x, y = pos_dict[u]
                lbl_x, lbl_y = x - 1, y - 1
            else:
                x1, y1 = pos_dict[u]
                x2, y2 = pos_dict[v]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    perp_x, perp_y = -dy / dist, dx / dist
                    offset = 0.01 * k * dist * 0.5
                    lbl_x, lbl_y = mid_x + perp_x * offset, mid_y + perp_y * offset
                else:
                    lbl_x, lbl_y = mid_x, mid_y

            ax.text(lbl_x, lbl_y, label_text, fontsize=5, ha='center', va='center',
                    color=label_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.9, edgecolor='none'))

    # 绘制
    nx.draw_networkx_nodes(G, pos_dict, node_size=node_sizes,
                           node_color=node_colors, cmap=cmap_nodes,
                           edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos_dict, font_size=20, font_color='black', font_family='Times New Roman',ax=ax)


    # 6. 图例优化 (Colorbars)
    # 这里的 fraction 和 pad 是控制位置的关键
    # fraction: colorbar 长度占轴的比例 (0.046 是个魔法数字，通常配合正方形轴很好看)
    # pad: colorbar 和轴之间的距离

    sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Edge Weight',fontsize=16, rotation=270, labelpad=10)

    if mode == 'full':
        sm_node = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=np.max(out_degree)))
        sm_node.set_array([])
        # 放在左边
        cbar2 = plt.colorbar(sm_node, ax=ax, location='left', fraction=0.040, pad=0.05)
        cbar2.set_label('Node Out-Degree',fontsize=16, rotation=90, labelpad=7)

    ax.set_title(title, pad=15)
    ax.set_xlabel("X Coordinate", fontsize=16, fontname='Times New Roman')
    ax.set_ylabel("Y Coordinate", fontsize=16, fontname='Times New Roman')

    # 设置坐标轴范围略微宽一点，防止标签被切
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    ax.grid(True, linestyle='--', alpha=0.2)

    # 保存逻辑
    plt.tight_layout()

    # 保存 SVG (矢量)
    svg_path = os.path.join(output_dir, filename_base + ".svg")
    plt.savefig(svg_path, format='svg', bbox_inches='tight')

    # 保存 PNG (预览)
    png_path = os.path.join(output_dir, filename_base + ".png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    plt.close()
    print(f"  --> 图表已保存: {filename_base} (.svg + .png)")

# ================= 4. 参数生成与仿真 =================
def generate_causal_params(env_type):
    links = {i: [] for i in range(N)}
    # 自回归
    for i in range(N):
        coeff = 0.8 if i in hub_nodes else 0.5
        links[i].append(((i, -1), coeff, saturation_func))

    # 交互连接
    for u, v, data in G_structure.edges(data=True):
        edge_type = data['type']
        weight, lag, active = 0.0, 1, False

        if edge_type == 'core':
            active, weight, lag = True, 0.6, 1
        else:
            target_x = node_positions[v, 0]
            is_west = target_x < 50

            if env_type == 'Env_A':  # 西边强
                if is_west:
                    active, weight, lag = True, 0.3, 1
                else:
                    active, weight, lag = True, 0.02, 3
            elif env_type == 'Env_B':  # 东边强
                if not is_west:
                    active, weight, lag = True, 0.3, 1
                else:
                    active, weight, lag = True, 0.02, 3

        if active:
            # 固定随机种子以保证复现
            local_seed = (u * N + v + (1 if env_type == 'Env_A' else 2))
            rng = np.random.RandomState(local_seed)
            final_weight = weight if rng.rand() > 0.3 else -weight
            links[v].append(((u, -lag), final_weight, saturation_func))

    return links


# ================= 5. 主流程 =================

def process_environment(env_name, seed_val):
    print(f"\n🚀 处理环境: {env_name} ...")

    links = generate_causal_params(env_name)
    # 0. 保存 groundtruth 的核心子图（PCMCI-like）
    build_and_save_core_subgraph_pcmci(
        links=links,
        G_structure=G_structure,
        N=N,
        tau_max=TAU_MAX,
        hub_nodes=hub_nodes,
        env_name=env_name,
        output_dir=output_dir
    )


    # 1. 生成数据 (含噪声)
    # 这里的 noises 就是您问的 "包含噪声吗" -> 是的，这里加入了高斯白噪声
    total_steps = int((1 + transient_frac) * T)
    noises = np.zeros((total_steps, N))
    for i in range(N):
        scale = 0.1 if i in hub_nodes else 0.3
        noises[:, i] = np.random.normal(0, scale, total_steps)

    data, _ = toys.structural_causal_process(
        links=links, T=T, noises=noises, seed=seed_val, transient_fraction=transient_frac
    )

    # 2. 保存时序数据
    var_names = [f"Node_{i}" for i in range(N)]
    df_data = pd.DataFrame(data, columns=var_names)
    df_data.to_csv(os.path.join(output_dir, f'{env_name}_timeseries.csv'), index=False)

    # 3. 绘制时间序列 (Feedback 3)
    print("  📈 绘制时间序列...")

    # 3.1 全部节点 (超高清, 字体极小)
    dataframe_all = pp.DataFrame(data=data, var_names=var_names, datatime=np.arange(T))
    plt.figure(figsize=(22, 12))
    tp.plot_timeseries(
        dataframe=dataframe_all,
        figsize=(22, 12),
        label_fontsize=4
    )
    plt.suptitle(f'{env_name} Time Series (All Nodes)', fontsize=10, y=0.98)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.savefig(os.path.join(output_dir, f'{env_name}_timeseries_all.png'), dpi=600)
    plt.close()

    # 3.2 前10个节点 (清晰版)
    subset_data = pp.DataFrame(data=data[:, :10], var_names=var_names[:10], datatime=np.arange(T))
    plt.figure(figsize=(14, 8))
    tp.plot_timeseries(dataframe=subset_data)
    plt.suptitle(f'{env_name} Time Series (First 10 Nodes)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{env_name}_timeseries_first10.png'), dpi=300)
    plt.close()

    # # 4. 绘制因果图 (Feedback 1 & 2)
    # print("  🎨 绘制因果图...")
    #
    # # 4.1 全图 - 带标签
    # draw_custom_causal_graph(links, node_positions, hub_nodes, G_structure,
    #                          title=f"{env_name}: Full Causal Graph (With Labels)",
    #                          filename=f"{env_name}_network_full_labels.png",
    #                          mode='full', show_labels=True)
    #
    # # 4.2 全图 - 无标签
    # draw_custom_causal_graph(links, node_positions, hub_nodes, G_structure,
    #                          title=f"{env_name}: Full Causal Graph (No Labels)",
    #                          filename=f"{env_name}_network_full_clean.png",
    #                          mode='full', show_labels=False)
    #
    # # 4.3 核心骨干高亮 - 带标签 (Core Highlight)
    # draw_custom_causal_graph(links, node_positions, hub_nodes, G_structure,
    #                          title=f"{env_name}: Core Highlight (With Labels)\nGray=Variant, Color=Core",
    #                          filename=f"{env_name}_core_highlight_labels.png",
    #                          mode='core_highlight', show_labels=True)
    #
    # # 4.4 核心骨干高亮 - 无标签
    # draw_custom_causal_graph(links, node_positions, hub_nodes, G_structure,
    #                          title=f"{env_name}: Core Highlight (No Labels)",
    #                          filename=f"{env_name}_core_highlight_clean.png",
    #                          mode='core_highlight', show_labels=False)

    # 4. 绘制因果图 (调用新函数)
    print("  🎨 绘制因果图 (SVG + PNG)...")

    # 注意：这里传入 filename_base (不带后缀)，函数内部会自动添加 .svg 和 .png

    # # 4.1 全图 - 带标签
    # draw_custom_causal_graph(links, node_positions, hub_nodes, G_structure,
    #                          title=f"{env_name}: Full Causal Graph",
    #                          filename_base=f"{env_name}_network_full_labels",
    #                          mode='full', show_labels=True)
    #
    # 4.2 全图 - 无标签
    draw_custom_causal_graph(links, node_positions, hub_nodes, G_structure,
                             title=f"{env_name}: Full Causal Graph (Clean)",
                             filename_base=f"{env_name}_network_full_clean",
                             mode='full', show_labels=False)
    # 4.2 全图 - 无标签
    draw_custom_causal_graph4(links, node_positions, hub_nodes, G_structure,
                             title=f"{env_name}: Full Causal Graph (Clean)",
                             filename_base=f"{env_name}_network_full_clean",
                             mode='full', show_labels=False)


    # # 4.3 核心骨干高亮 - 带标签
    # draw_custom_causal_graph2(links, node_positions, hub_nodes, G_structure,
    #                          title=f"{env_name}: Core Highlight",
    #                          filename_base=f"{env_name}_core_highlight_labels",
    #                          mode='core_highlight', show_labels=True)

    # 4.4 核心骨干高亮 - 无标签
    draw_custom_causal_graph2(links, node_positions, hub_nodes, G_structure,
                             title=f"{env_name}: Core Highlight (Clean)",
                             filename_base=f"{env_name}_core_highlight_clean",
                             mode='core_highlight', show_labels=False)
    # 4.4 核心骨干高亮 - 无标签-加粗
    draw_custom_causal_graph3(links, node_positions, hub_nodes, G_structure,
                             title=f"{env_name}: Core Highlight (Clean)",
                             filename_base=f"{env_name}_core_highlight_clean",
                             mode='core_highlight', show_labels=False)


if __name__ == "__main__":
    process_environment('Env_A', 10)
    # process_environment('Env_B', 20)

    print("\n✅ 全部任务完成！")
    print(f"数据是否包含噪声? 是 (高斯噪声, Hubs=0.1, Others=0.3)")
    print(f"结果保存在: {output_dir}")
