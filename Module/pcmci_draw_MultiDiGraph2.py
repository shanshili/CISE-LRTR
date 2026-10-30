# pcmci_draw_MultiDiGraph2.py
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
6. draw_causal_graph(graph, val_matrix, positions, ax, title, tau_max=3)
   参数：graph - PCMCI因果图
         val_matrix - 因果效应矩阵
         positions - 节点位置
         ax - matplotlib轴对象
         title - 图标题
         tau_max - 最大时间滞后
   功能：在指定轴上绘制因果网络图

7. plot_causal_graph(G, positions, edge_counts, node_self_counts,
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

8. plot_causal_matrix(edge_counts, N, output_dir, tau_str, alpha_str)
   参数：edge_counts - 边统计字典
         N - 节点数量
         output_dir - 输出目录
         tau_str - 时间滞后字符串
         alpha_str - 显著性阈值字符串
   功能：生成因果连接数量矩阵热力图

辅助函数：
9. separate_self_edges(G)
   参数：G - NetworkX MultiDiGraph对象
   返回：self_edges, normal_edges - 自循环边列表和普通边列表
   功能：分离自循环边和节点间边

10. calculate_visualization_parameters(G, N)
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
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

from matplotlib.font_manager import font_family_aliases

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
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pickle
import torch


def load_pcmci_results(pickle_file):
    """
    从.pkl文件加载PCMCI结果

    参数:
        pickle_file: PCMCI结果文件路径

    返回:
        graph, val_matrix - 因果图和效应矩阵
    """
    with open(pickle_file, 'rb') as f:
        loaded_results = pickle.load(f)

    graph = loaded_results['graph']
    val_matrix = loaded_results['val_matrix']

    return graph, val_matrix


def load_node_positions(position_file):
    """
    加载节点位置信息

    参数:
        position_file: 节点位置CSV文件路径

    返回:
        positions - 节点坐标数组(N, 2)
    """
    positions_df = pd.read_csv(position_file)
    positions = positions_df[['x', 'y']].values  # shape: (N, 2)
    return positions


def build_multi_di_graph(graph, val_matrix, N, tau_max, positions):
    """
    构建带时间滞后信息的因果图

    参数:
        graph: PCMCI因果图
        val_matrix: 因果效应矩阵
        N: 节点数量
        tau_max: 最大时间滞后
        positions: 节点位置

    返回:
        G: NetworkX MultiDiGraph对象
        edge_counts: 边统计字典
        node_self_counts: 自关联计数数组
    """
    # 统计每个节点的自关联数量（节点到自己的因果联系）
    node_self_counts = np.zeros(N)
    for target in range(N):
        for tau in range(1, tau_max + 1):
            if graph[target, target, tau] == '-->':  # 检查节点target到节点target的关联
                node_self_counts[target] += 1

    # 构建连接统计字典（排除自关联）
    edge_counts = {}
    for target in range(N):  # target是目标节点
        for source in range(N):  # source是源节点
            count = 0
            for tau in range(1, tau_max + 1):
                # graph[target, source, tau] == '-->' 表示 source 在 t-τ 时刻对 target 在 t 时刻有因果影响
                if graph[target, source, tau] == '-->':
                    count += 1
            if count > 0:
                # edge_counts[(source, target)] 表示从source到target的连接
                edge_counts[(source, target)] = count

    # 构建MultiDiGraph
    G = nx.MultiDiGraph()  # 改为MultiDiGraph

    # 添加节点并设置位置信息
    for i in range(N):
        G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

    # 添加边并保留所有时间滞后信息
    edge_count = 0
    for target in range(N):  # target是目标节点
        for source in range(N):  # source是源节点
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

    return G, edge_counts, node_self_counts


def draw_causal_graph(graph, val_matrix, positions, ax, title, tau_max=3, node_size=60, font_size=6):
    """
    在指定的轴上绘制因果图

    参数:
        graph: PCMCI因果图矩阵
        val_matrix: 因果效应矩阵
        positions: 节点位置
        ax: matplotlib 轴对象
        title: 图标题
        tau_max: 最大时间滞后
        node_size: 节点大小
        font_size: 标签字体大小
    """
    N = graph.shape[0]

    # 构建MultiDiGraph
    G = nx.MultiDiGraph()

    # 添加节点并设置位置信息
    for i in range(N):
        G.add_node(i, pos=(positions[i, 0], positions[i, 1]))

    # 添加边并保留所有时间滞后信息
    for target in range(N):  # target是目标节点
        for source in range(N):  # source是源节点
            for tau in range(1, tau_max + 1):  # 时间滞后 (通常从1开始)
                # 检查是否存在显著的因果关系
                # graph[target, source, tau] == '-->' 表示 source 在 t-τ 时刻对 target 在 t 时刻有因果影响
                if graph[target, source, tau] == '-->':
                    # 获取对应的因果效应权重值
                    causal_effect = val_matrix[target, source, tau]
                    weight = abs(causal_effect)  # 使用绝对值作为边的权重

                    # 添加边到MultiDiGraph: source -> target
                    G.add_edge(source, target, key=tau, weight=weight, lag=tau, causal_effect=causal_effect)

    pos = nx.get_node_attributes(G, 'pos')

    # 计算每个节点的出边数量（出边表示该节点影响其他节点的数量）
    out_degree = np.zeros(N)
    for source, target, key, data in G.edges(keys=True, data=True):
        out_degree[source] += 1  # 节点source有一条出边

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

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=out_degree,
                           cmap=plt.cm.YlGn, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=font_size, ax=ax)

    # 绘制普通边（节点间连接）
    for source, target, key, data in normal_edges:
        lag = data['lag']
        causal_effect = data['causal_effect']

        # 根据因果效应大小设置颜色
        color = cmap(norm(causal_effect))

        # 根据因果效应绝对值设置边宽度
        width = 1 + abs(causal_effect) * 5  # 基础宽度1，根据因果效应绝对值调整

        # 绘制单条边: source -> target
        nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                               width=width,
                               edge_color=[color],
                               alpha=0.7, arrows=True, arrowsize=10, ax=ax)

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
        ax.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
                    arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
                                    mutation_scale=width * 3 + 10))

    # 添加颜色图例（因果效应）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Causal Effect', rotation=270, labelpad=20)

    # 添加节点出边数量颜色图例
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=np.max(out_degree) if np.max(
        out_degree) > 0 else 1))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.8, location='left')
    cbar_nodes.set_label('Number of Outgoing Edges', rotation=270, labelpad=20)

    # 设置标题和坐标轴
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_causal_graph(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information"):
    """
    绘制因果网络图，包括节点、边、自循环、颜色条等

    参数:
        G: NetworkX MultiDiGraph对象
        positions: 节点位置
        edge_counts: 边统计字典
        node_self_counts: 自关联计数数组
        output_dir: 输出目录
        tau_str: 时间滞后字符串
        alpha_str: 显著性阈值字符串
        title: 图标题
    """
    # 获取图的节点数
    N = len(G.nodes())

    # # 计算每个节点的出边数量（出边表示该节点影响其他节点的数量）
    # out_degree = np.zeros(N)
    # for source, target, key, data in G.edges(keys=True, data=True):
    #     out_degree[source] += 1  # 节点source有一条出边
    # vmax = np.max(out_degree) if np.max(out_degree) > 0 else 1
    # 计算节点的度数
    degrees = np.array([G.degree(i) for i in range(N)])
    vmax = np.max(degrees) if np.max(degrees) > 0 else 1
    node_colors = degrees
    node_sizes = 100 + degrees * 10
    cmap_nodes = plt.cm.YlGn

    # 设置绘图参数
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect('equal', adjustable='box')

    nx.draw_networkx_nodes(G, positions, node_size=node_sizes, node_color=node_colors,
                           cmap=cmap_nodes, edgecolors='white', linewidths=1.0, alpha=0.9,ax=ax)
    nx.draw_networkx_labels(G, positions, font_size=8, font_color='black',ax=ax)

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
        nx.draw_networkx_edges(G, positions, edgelist=[(target,source)],
                               width=width,
                               edge_color=[color],
                               alpha=0.7, arrows=True, arrowsize=10, ax=ax)

    # 绘制自相关边（自循环）
    for source, target, key, data in self_edges:
        lag = data['lag']
        causal_effect = data['causal_effect']

        # 根据权重大小设置颜色
        color = cmap(norm(causal_effect))

        # 根据权重绝对值设置边宽度
        width = 1 + abs(causal_effect) * 5

        # 绘制自循环边
        x, y = positions[source]
        # 手动绘制一个小箭头表示自循环
        plt.annotate('', xy=(x, y + 0.005), xytext=(x - 0.005, y),
                     arrowprops=dict(arrowstyle='simple', color=color, lw=width, alpha=0.7,
                                     mutation_scale=width * 3 + 10))

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
    plt.title(title)
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
    plt.savefig(png_path, dpi=300,
                bbox_inches='tight')  # 高DPI    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()

"""
绘制增强版因果网络图，改进节点颜色编码方式：
节点颜色基于节点相连因果强度绝对值之和（入边+出边）
只显示强度前30%的节点标签，提高可视化清晰度
自循环边的绘制位置有所调整
"""
def plot_causal_graph2(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information"):
    """
    绘制因果网络图，包括节点、边、自循环、颜色条等
    """
    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 关键修改：节点强度（node_strength） =====================
    # 节点颜色：与该节点相连的因果强度绝对值之和（入+出；自环按自身强度计一次）
    node_strength = np.zeros(N, dtype=float)

    for u, v, k, data in G.edges(keys=True, data=True):
        # 统一用 causal_effect（你边宽/边色本来也用它）；如果没有就退回 weight
        ce = data.get('causal_effect', data.get('weight', 0.0))
        w = abs(float(ce))

        if u == v:  # 自环：按该自环自己的强度计一次
            node_strength[u] += w
        else:  # 普通边：对两端节点都算“相连强度”
            node_strength[u] += w
            node_strength[v] += w

    vmax_nodes = node_strength.max() if node_strength.max() > 0 else 1.0
    node_colors = node_strength
    node_sizes = 100 + (node_strength / vmax_nodes) * 300  # 不想改大小就保持你原来的
    cmap_nodes = plt.cm.YlGn

    # ============================================================================

    # 设置绘图参数
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect('equal', adjustable='box')

    nx.draw_networkx_nodes(
        G, positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap_nodes,
        edgecolors='white',
        linewidths=1.0,
        alpha=0.9,
        ax=ax
    )
    label_threshold = np.quantile(node_strength, 0.70)  # 只标注前30%

    # 只给强节点画编号
    labels_strong = {
        i: i
        for i in range(N)
        if node_strength[i] >= label_threshold
    }

    nx.draw_networkx_labels(
        G,
        positions,
        labels=labels_strong,
        font_size=8,
        font_color='black',
        ax=ax
    )

    # 分别处理自相关边（自循环）和普通边
    self_edges = []
    normal_edges = []

    for source, target, key, data in G.edges(keys=True, data=True):
        if source == target:
            self_edges.append((source, target, key, data))
        else:
            normal_edges.append((source, target, key, data))

    # 获取所有权重用于颜色映射（边的颜色仍然按因果效应正负）
    all_weights = [abs(data.get('weight', data.get('causal_effect', 0.0)))
                   for source, target, key, data in G.edges(keys=True, data=True)]

    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        abs_max = max(abs(min_weight), abs(max_weight))
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # 绘制普通边（节点间连接）
    for source, target, key, data in normal_edges:
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = 3 + abs(causal_effect) * 5

        nx.draw_networkx_edges(
            G, positions,
            edgelist=[(target, source)],
            width=width,
            edge_color=[color],
            alpha=1,
            arrows=True,
            arrowsize=10,
            ax=ax
        )

    # 绘制自相关边（自循环）
    for source, target, key, data in self_edges:
        # ❗ 只给“显示 label 的节点”画自环
        if source not in labels_strong:
            continue
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = abs(causal_effect) * 5+5
        x, y = positions[source]
        dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        plt.annotate(
            '',
            xy=(x - dx, y - dx),
            xytext=(x - dy, y - dy),
            arrowprops=dict(
                arrowstyle='simple',
                color=color,
                lw=1,
                alpha=0.7,
                mutation_scale=width * 1.1 + 5
            )
        )

        # plt.annotate(
        #     '',
        #     xy=(x-1, y - 0.995),
        #     xytext=(x - 1.005, y-1),
        #     arrowprops=dict(
        #         arrowstyle='simple',
        #         color=color,
        #         lw=width,
        #         alpha=0.7,
        #         mutation_scale=width * 3 + 10
        #     )
        # )

    # 添加颜色图例（边：因果效应）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label('Causal Effect', rotation=270, labelpad=10)

    # 添加节点颜色图例（节点：连接因果强度绝对值之和）
    sm_nodes = plt.cm.ScalarMappable(
        cmap=cmap_nodes,
        norm=plt.Normalize(vmin=0, vmax=vmax_nodes if vmax_nodes > 0 else 1.0)
    )
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.05, location='left')
    cbar_nodes.set_label('Sum(|Causal Effect|) of Connected Edges', rotation=90, labelpad=7)

    # 设置标题和坐标轴
    plt.title(title + "\n(Node labels shown for top 30% Σ|Causal Effect|)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存图像
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()
def plot_causal_graph2_clear2(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information"):
    """
    绘制因果网络图，包括节点、边、自循环、颜色条等
    """
    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 关键修改：节点强度（node_strength） =====================
    # 节点颜色：与该节点相连的因果强度绝对值之和（入+出；自环按自身强度计一次）
    node_strength = np.zeros(N, dtype=float)

    for u, v, k, data in G.edges(keys=True, data=True):
        # 统一用 causal_effect（你边宽/边色本来也用它）；如果没有就退回 weight
        ce = data.get('causal_effect', data.get('weight', 0.0))
        w = abs(float(ce))

        if u == v:  # 自环：按该自环自己的强度计一次
            node_strength[u] += w
        else:  # 普通边：对两端节点都算“相连强度”
            node_strength[u] += w
            node_strength[v] += w

    vmax_nodes = node_strength.max() if node_strength.max() > 0 else 1.0
    node_colors = node_strength
    node_sizes = 100 + (node_strength / vmax_nodes) * 300  # 不想改大小就保持你原来的
    cmap_nodes = plt.cm.YlGn

    # ===================== 新增：节点分类 =====================
    # 前30%的节点为红色，其余为灰色
    label_threshold = np.quantile(node_strength, 0.70)  # 前30%的阈值
    core_nodes = {i for i in range(N) if node_strength[i] >= label_threshold}
    non_core_nodes = {i for i in range(N) if node_strength[i] < label_threshold}

    # 设置绘图参数
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_aspect('equal', adjustable='box')

    # 绘制核心节点（红色）
    core_lons = [positions[i][0] for i in core_nodes]
    core_lats = [positions[i][1] for i in core_nodes]
    core_strengths = [node_strength[i] for i in core_nodes]
    core_sizes = [100 + (s / vmax_nodes) * 300 for s in core_strengths]

    ax.scatter(core_lons, core_lats, s=core_sizes, c='red', edgecolors='white', linewidths=1.0, alpha=0.9, zorder=5)

    # 绘制非核心节点（灰色）
    non_core_lons = [positions[i][0] for i in non_core_nodes]
    non_core_lats = [positions[i][1] for i in non_core_nodes]
    non_core_strengths = [node_strength[i] for i in non_core_nodes]
    non_core_sizes = [100 + (s / vmax_nodes) * 300 for s in non_core_strengths]

    ax.scatter(non_core_lons, non_core_lats, s=non_core_sizes, c='gray', edgecolors='white', linewidths=1.0, alpha=0.9, zorder=5)

    # 只给强节点画编号
    for i in core_nodes:
        x, y = positions[i]
        ax.text(x, y, str(i), fontsize=12, color='black', ha='center', va='center', zorder=6)

    # ===================== 边绘制逻辑保持不变 =====================
    self_edges = []
    normal_edges = []

    for source, target, key, data in G.edges(keys=True, data=True):
        if source == target:
            self_edges.append((source, target, key, data))
        else:
            normal_edges.append((source, target, key, data))

    all_weights = [abs(data.get('weight', data.get('causal_effect', 0.0)))
                   for source, target, key, data in G.edges(keys=True, data=True)]

    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        abs_max = max(abs(min_weight), abs(max_weight))
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # 绘制普通边（节点间连接）
    for source, target, key, data in normal_edges:
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = 3 + abs(causal_effect) * 5

        ax.annotate(
            '',
            xy=positions[target], xytext=positions[source],
            arrowprops=dict(arrowstyle='-|>', color=color, lw=width, alpha=0.7, mutation_scale=10),
            zorder=4
        )

    # 绘制自相关边（自循环）
    for source, target, key, data in self_edges:
        if source not in core_nodes:
            continue
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = abs(causal_effect) * 5 + 5
        x, y = positions[source]
        dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(
            '',
            xy=(x - dx, y - dx),
            xytext=(x - dy, y - dy),
            arrowprops=dict(arrowstyle='simple', color=color, lw=1, alpha=0.7, mutation_scale=width * 1.1 + 5),
            zorder=4
        )

    # ===================== 添加颜色图例 =====================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Causal Effect', fontsize=14,rotation=270, labelpad=10)

    sm_nodes = plt.cm.ScalarMappable(
        cmap=cmap_nodes,
        norm=plt.Normalize(vmin=0, vmax=vmax_nodes if vmax_nodes > 0 else 1.0)
    )
    sm_nodes.set_array([])
    # cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.05, location='left')
    # cbar_nodes.set_label('Sum(|Causal Effect|) of Connected Edges', rotation=90, labelpad=7)

    # ===================== 新增：图例 =====================
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Core Causal Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='Non-Core Nodes')
    ]
    # 设置图例字体大小
    font_prop = FontProperties(size=14)  # 字体大小设为14
    # 添加图例并应用字体设置
    ax.legend(handles=legend_elements, prop=font_prop, loc='upper right')
    # 设置标题和坐标轴
    # plt.title(title + "\n(Red nodes: Top 30% Σ|Causal Effect|, Gray nodes: Others)")
    plt.xlabel("Longitude",fontsize=14)
    plt.ylabel("Latitude",fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存图像
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # 保存为 EPS 格式
    eps_path = os.path.join(output_dir, filename1 + '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')

    # 保存为 PDF 格式
    pdf_path = os.path.join(output_dir, filename1 + '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # 保存为 TIFF 格式
    tiff_path = os.path.join(output_dir, filename1 + '.tiff')
    plt.savefig(tiff_path, format='tiff', dpi=300, bbox_inches='tight')

    # 保存为 JPEG 格式
    jpeg_path = os.path.join(output_dir, filename1 + '.jpg')
    plt.savefig(jpeg_path, format='jpg', dpi=300, bbox_inches='tight')

    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()

def plot_causal_graph2_clear(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information"):
    """
    绘制因果网络图，包括节点、边、自循环、颜色条等
    """
    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 关键修改：节点强度（node_strength） =====================
    # 节点颜色：与该节点相连的因果强度绝对值之和（入+出；自环按自身强度计一次）
    node_strength = np.zeros(N, dtype=float)

    for u, v, k, data in G.edges(keys=True, data=True):
        # 统一用 causal_effect（你边宽/边色本来也用它）；如果没有就退回 weight
        ce = data.get('causal_effect', data.get('weight', 0.0))
        w = abs(float(ce))

        if u == v:  # 自环：按该自环自己的强度计一次
            node_strength[u] += w
        else:  # 普通边：对两端节点都算“相连强度”
            node_strength[u] += w
            node_strength[v] += w

    vmax_nodes = node_strength.max() if node_strength.max() > 0 else 1.0
    node_colors = node_strength
    node_sizes = 100 + (node_strength / vmax_nodes) * 300  # 不想改大小就保持你原来的
    cmap_nodes = plt.cm.YlGn

    # ===================== 新增：节点分类 =====================
    # 前30%的节点为红色，其余为灰色
    label_threshold = np.quantile(node_strength, 0.70)  # 前30%的阈值
    core_nodes = {i for i in range(N) if node_strength[i] >= label_threshold}
    non_core_nodes = {i for i in range(N) if node_strength[i] < label_threshold}

    # 设置绘图参数
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect('equal', adjustable='box')

    # 绘制核心节点（红色）
    core_lons = [positions[i][0] for i in core_nodes]
    core_lats = [positions[i][1] for i in core_nodes]
    core_strengths = [node_strength[i] for i in core_nodes]
    core_sizes = [100 + (s / vmax_nodes) * 300 for s in core_strengths]

    ax.scatter(core_lons, core_lats, s=core_sizes, c='red', edgecolors='white', linewidths=1.0, alpha=0.9, zorder=5)

    # 绘制非核心节点（灰色）
    non_core_lons = [positions[i][0] for i in non_core_nodes]
    non_core_lats = [positions[i][1] for i in non_core_nodes]
    non_core_strengths = [node_strength[i] for i in non_core_nodes]
    non_core_sizes = [100 + (s / vmax_nodes) * 300 for s in non_core_strengths]

    ax.scatter(non_core_lons, non_core_lats, s=non_core_sizes, c='gray', edgecolors='white', linewidths=1.0, alpha=0.9, zorder=5)

    # 只给强节点画编号
    for i in core_nodes:
        x, y = positions[i]
        ax.text(x, y, str(i), fontsize=12, color='black', ha='center', va='center', zorder=6)

    # ===================== 边绘制逻辑保持不变 =====================
    self_edges = []
    normal_edges = []

    for source, target, key, data in G.edges(keys=True, data=True):
        if source == target:
            self_edges.append((source, target, key, data))
        else:
            normal_edges.append((source, target, key, data))

    all_weights = [abs(data.get('weight', data.get('causal_effect', 0.0)))
                   for source, target, key, data in G.edges(keys=True, data=True)]

    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        abs_max = max(abs(min_weight), abs(max_weight))
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # 绘制普通边（节点间连接）
    for source, target, key, data in normal_edges:
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = 3 + abs(causal_effect) * 5

        ax.annotate(
            '',
            xy=positions[target], xytext=positions[source],
            arrowprops=dict(arrowstyle='-|>', color=color, lw=width, alpha=0.7, mutation_scale=10),
            zorder=4
        )

    # 绘制自相关边（自循环）
    for source, target, key, data in self_edges:
        if source not in core_nodes:
            continue
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = abs(causal_effect) * 5 + 5
        x, y = positions[source]
        dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.annotate(
            '',
            xy=(x - dx, y - dx),
            xytext=(x - dy, y - dy),
            arrowprops=dict(arrowstyle='simple', color=color, lw=1, alpha=0.7, mutation_scale=width * 1.1 + 5),
            zorder=4
        )

    # ===================== 添加颜色图例 =====================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label('Causal Effect', fontsize=14,rotation=270, labelpad=10)

    sm_nodes = plt.cm.ScalarMappable(
        cmap=cmap_nodes,
        norm=plt.Normalize(vmin=0, vmax=vmax_nodes if vmax_nodes > 0 else 1.0)
    )
    sm_nodes.set_array([])
    # cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.05, location='left')
    # cbar_nodes.set_label('Sum(|Causal Effect|) of Connected Edges', rotation=90, labelpad=7)

    # ===================== 新增：图例 =====================
    from matplotlib.lines import Line2D
    from matplotlib.font_manager import FontProperties
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='Core Causal Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, label='Non-Core Nodes')
    ]
    # 设置图例字体大小
    font_prop = FontProperties(size=14)  # 字体大小设为14
    # 添加图例并应用字体设置
    ax.legend(handles=legend_elements, prop=font_prop, loc='lower left')
    # 设置标题和坐标轴
    # plt.title(title + "\n(Red nodes: Top 30% Σ|Causal Effect|, Gray nodes: Others)")
    plt.xlabel("Longitude",fontsize=14)
    plt.ylabel("Latitude",fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存图像
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # 保存为 EPS 格式
    eps_path = os.path.join(output_dir, filename1 + '.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')

    # 保存为 PDF 格式
    pdf_path = os.path.join(output_dir, filename1 + '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # 保存为 TIFF 格式
    tiff_path = os.path.join(output_dir, filename1 + '.tiff')
    plt.savefig(tiff_path, format='tiff', dpi=300, bbox_inches='tight')

    # 保存为 JPEG 格式
    jpeg_path = os.path.join(output_dir, filename1 + '.jpg')
    plt.savefig(jpeg_path, format='jpg', dpi=300, bbox_inches='tight')

    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()


def plot_causal_graph2_pems(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information"):
    """
    绘制因果网络图：底图为地图（WGS84/EPSG:4326），节点坐标为经纬度 (lon, lat)
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    from cartopy.io.img_tiles import Stamen

    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 节点强度（node_strength） =====================
    node_strength = np.zeros(N, dtype=float)
    for u, v, k, data in G.edges(keys=True, data=True):
        ce = data.get('causal_effect', data.get('weight', 0.0))
        w = abs(float(ce))
        if u == v:
            node_strength[u] += w
        else:
            node_strength[u] += w
            node_strength[v] += w

    vmax_nodes = node_strength.max() if node_strength.max() > 0 else 1.0
    node_colors = node_strength
    node_sizes = 100 + (node_strength / vmax_nodes) * 300
    cmap_nodes = plt.cm.YlGn


    # ===================== 地图底图设置（EPSG:4326） =====================
    data_crs = ccrs.PlateCarree()     # 数据坐标系：WGS84 lon/lat (EPSG:4326)
    map_crs  = ccrs.PlateCarree()     # 显示投影：城市尺度用它足够；也可换 ccrs.Mercator()

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.axes(projection=map_crs)

    # 从 positions 自动计算范围，保证底图与节点范围对齐
    lons = np.array([positions[i][0] for i in range(N)], dtype=float)
    lats = np.array([positions[i][1] for i in range(N)], dtype=float)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # 城市尺度：固定最小边距 + 比例边距（你这个 LA 范围很合适）
    pad_lon = max(0.02, (lon_max - lon_min) * 0.12)
    pad_lat = max(0.02, (lat_max - lat_min) * 0.12)

    ax.set_extent([lon_min - pad_lon, lon_max + pad_lon,
                   lat_min - pad_lat, lat_max + pad_lat],
                  crs=data_crs)

    # 底图要素（国外也通用；城市尺度建议 10m 更清晰）
    # ax.add_feature(cfeature.OCEAN.with_scale('10m'), alpha=0.25)
    # ax.add_feature(cfeature.LAND.with_scale('10m'), alpha=0.15)
    # ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, alpha=0.8)
    # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.5, alpha=0.6)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.2)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), alpha=0.2)

    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.4, alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # ===================== 街道瓦片底图（OSM） =====================
    tiler = cimgt.OSM()

    # 这个范围（约 0.4° x 0.3°，LA 城市尺度）推荐 12~15
    # 你可以动态估一个 zoom，或直接固定 13/14
    zoom = 13
    ax.add_image(tiler, zoom,alpha=0.55)

    # ===================== 画节点（scatter，确保 transform=data_crs） =====================
    sc = ax.scatter(
        lons, lats,
        s=node_sizes,
        c=node_colors,
        cmap=cmap_nodes,
        edgecolors='white',
        linewidths=1.0,
        alpha=0.9,
        transform=data_crs,
        zorder=5
    )

    # 标签：只标注前 30%
    label_threshold = np.quantile(node_strength, 0.70)
    labels_strong = {i: i for i in range(N) if node_strength[i] >= label_threshold}

    for i in labels_strong:
        x, y = positions[i]
        ax.text(x, y, str(i),
                fontsize=8, color='black',
                ha='center', va='center',
                transform=data_crs,
                zorder=6)

    # ===================== 边颜色映射（仍按 causal_effect 正负） =====================
    self_edges = []
    normal_edges = []
    for source, target, key, data in G.edges(keys=True, data=True):
        if source == target:
            self_edges.append((source, target, key, data))
        else:
            normal_edges.append((source, target, key, data))

    all_weights = [abs(data.get('weight', data.get('causal_effect', 0.0)))
                   for _, _, _, data in G.edges(keys=True, data=True)]
    if all_weights:
        abs_max = max(all_weights)
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # ===================== 画普通边（用 annotate + EPSG:4326 transform） =====================
    for source, target, key, data in normal_edges:
        causal_effect = float(data['causal_effect'])
        color = cmap(norm(causal_effect))
        width = 3 + abs(causal_effect) * 8

        x1, y1 = positions[source]
        x2, y2 = positions[target]

        ax.annotate(
            '',
            xy=(x2, y2), xytext=(x1, y1),
            xycoords=data_crs._as_mpl_transform(ax),
            textcoords=data_crs._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle='-|>',
                color=color,
                lw=width,
                alpha=0.7,
                mutation_scale=10
            ),
            zorder=4
        )

    # ===================== 画自环（只给强节点画） =====================
    d_lon = max(0.001, (lon_max - lon_min) * 0.01)
    d_lat = max(0.001, (lat_max - lat_min) * 0.01)

    for source, target, key, data in self_edges:
        if source not in labels_strong:
            continue

        causal_effect = float(data['causal_effect'])
        color = cmap(norm(causal_effect))
        width = abs(causal_effect) * 2+5

        x, y = positions[source]

        ax.annotate(
            '',
            xy=(x - d_lon, y - d_lat),
            xytext=(x - 2*d_lon, y - 2*d_lat),
            xycoords=data_crs._as_mpl_transform(ax),
            textcoords=data_crs._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle='simple',
                color=color,
                lw=width * 0.5,
                alpha=0.7,
                mutation_scale=width * 1.1 + 2
            ),
            zorder=4
        )

    # ===================== 颜色条（用 fig.colorbar 更稳） =====================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label('Causal Effect', rotation=270, labelpad=10)

    sm_nodes = plt.cm.ScalarMappable(
        cmap=cmap_nodes,
        norm=plt.Normalize(vmin=0, vmax=vmax_nodes if vmax_nodes > 0 else 1.0)
    )
    sm_nodes.set_array([])
    cbar_nodes = fig.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.06, location='left')
    cbar_nodes.set_label('Sum(|Causal Effect|) of Connected Edges', rotation=90, labelpad=7)

    # 标题
    ax.set_title(title + "\n(Node labels shown for top 30% Σ|Causal Effect|)")

    plt.tight_layout()

    # 保存
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()

def plot_causal_graph2_pems_clear(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information"):
    """
    绘制因果网络图：底图为地图（WGS84/EPSG:4326），节点坐标为经纬度 (lon, lat)
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    from cartopy.io.img_tiles import Stamen

    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 节点强度（node_strength） =====================
    node_strength = np.zeros(N, dtype=float)
    for u, v, k, data in G.edges(keys=True, data=True):
        ce = data.get('causal_effect', data.get('weight', 0.0))
        w = abs(float(ce))
        if u == v:
            node_strength[u] += w
        else:
            node_strength[u] += w
            node_strength[v] += w

    vmax_nodes = node_strength.max() if node_strength.max() > 0 else 1.0
    node_colors = node_strength
    node_sizes = 100 + (node_strength / vmax_nodes) * 300
    cmap_nodes = plt.cm.YlGn

    # ===================== 节点分类：前30%为红色，其余为灰色 =====================
    label_threshold = np.quantile(node_strength, 0.70)  # 前30%的阈值
    core_nodes = {i for i in range(N) if node_strength[i] >= label_threshold}
    non_core_nodes = {i for i in range(N) if node_strength[i] < label_threshold}

    # ===================== 地图底图设置（EPSG:4326） =====================
    data_crs = ccrs.PlateCarree()     # 数据坐标系：WGS84 lon/lat (EPSG:4326)
    map_crs  = ccrs.PlateCarree()     # 显示投影：城市尺度用它足够；也可换 ccrs.Mercator()

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.axes(projection=map_crs)

    # 从 positions 自动计算范围，保证底图与节点范围对齐
    lons = np.array([positions[i][0] for i in range(N)], dtype=float)
    lats = np.array([positions[i][1] for i in range(N)], dtype=float)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # 城市尺度：固定最小边距 + 比例边距（你这个 LA 范围很合适）
    pad_lon = max(0.02, (lon_max - lon_min) * 0.12)
    pad_lat = max(0.02, (lat_max - lat_min) * 0.12)

    ax.set_extent([lon_min - pad_lon, lon_max + pad_lon,
                   lat_min - pad_lat, lat_max + pad_lat],
                  crs=data_crs)

    # 底图要素（国外也通用；城市尺度建议 10m 更清晰）
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.2)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), alpha=0.2)

    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.4, alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # ===================== 街道瓦片底图（OSM） =====================
    tiler = cimgt.OSM()
    zoom = 13
    ax.add_image(tiler, zoom, alpha=0.55)

    # ===================== 绘制节点 =====================
    # 核心节点（红色）
    core_lons = [positions[i][0] for i in core_nodes]
    core_lats = [positions[i][1] for i in core_nodes]
    core_strengths = [node_strength[i] for i in core_nodes]
    core_sizes = [100 + (s / vmax_nodes) * 300 for s in core_strengths]

    ax.scatter(core_lons, core_lats, s=core_sizes, c='red', edgecolors='white', linewidths=1.0, alpha=0.9, transform=data_crs, zorder=5)

    # 非核心节点（灰色）
    non_core_lons = [positions[i][0] for i in non_core_nodes]
    non_core_lats = [positions[i][1] for i in non_core_nodes]
    non_core_strengths = [node_strength[i] for i in non_core_nodes]
    non_core_sizes = [100 + (s / vmax_nodes) * 300 for s in non_core_strengths]

    ax.scatter(non_core_lons, non_core_lats, s=non_core_sizes, c='gray', edgecolors='white', linewidths=1.0, alpha=0.9, transform=data_crs, zorder=5)

    # ===================== 标注节点序号 =====================
    for i in core_nodes:
        x, y = positions[i]
        ax.text(x, y, str(i), fontsize=8, color='white', ha='center', va='center', transform=data_crs, zorder=6)

    # ===================== 边颜色映射（仍按 causal_effect 正负） =====================
    self_edges = []
    normal_edges = []
    for source, target, key, data in G.edges(keys=True, data=True):
        if source == target:
            self_edges.append((source, target, key, data))
        else:
            normal_edges.append((source, target, key, data))

    all_weights = [abs(data.get('weight', data.get('causal_effect', 0.0)))
                   for _, _, _, data in G.edges(keys=True, data=True)]
    if all_weights:
        abs_max = max(all_weights)
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # ===================== 画普通边 =====================
    for source, target, key, data in normal_edges:
        causal_effect = float(data['causal_effect'])
        color = cmap(norm(causal_effect))
        width = 3 + abs(causal_effect) * 8

        x1, y1 = positions[source]
        x2, y2 = positions[target]

        ax.annotate(
            '',
            xy=(x2, y2), xytext=(x1, y1),
            xycoords=data_crs._as_mpl_transform(ax),
            textcoords=data_crs._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle='-|>',
                color=color,
                lw=width,
                alpha=0.7,
                mutation_scale=10
            ),
            zorder=4
        )

    # ===================== 画自环 =====================
    d_lon = max(0.001, (lon_max - lon_min) * 0.01)
    d_lat = max(0.001, (lat_max - lat_min) * 0.01)

    for source, target, key, data in self_edges:
        if source not in core_nodes:
            continue

        causal_effect = float(data['causal_effect'])
        color = cmap(norm(causal_effect))
        width = abs(causal_effect) * 2 + 5

        x, y = positions[source]

        ax.annotate(
            '',
            xy=(x - d_lon, y - d_lat),
            xytext=(x - 2*d_lon, y - 2*d_lat),
            xycoords=data_crs._as_mpl_transform(ax),
            textcoords=data_crs._as_mpl_transform(ax),
            arrowprops=dict(
                arrowstyle='simple',
                color=color,
                lw=width * 0.5,
                alpha=0.7,
                mutation_scale=width * 1.1 + 2
            ),
            zorder=4
        )

    # ===================== 颜色条 =====================
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label('Causal Effect', rotation=270, labelpad=10)

    sm_nodes = plt.cm.ScalarMappable(
        cmap=cmap_nodes,
        norm=plt.Normalize(vmin=0, vmax=vmax_nodes if vmax_nodes > 0 else 1.0)
    )
    sm_nodes.set_array([])
    cbar_nodes = fig.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.06, location='left')
    cbar_nodes.set_label('Sum(|Causal Effect|) of Connected Edges', rotation=90, labelpad=7)

    # ===================== 图例 =====================
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Core Causal Nodes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Non-Core Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 标题
    ax.set_title(title + "\n(Red nodes: Top 30% Σ|Causal Effect|, Gray nodes: Others)")

    plt.tight_layout()

    # 保存
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()


def plot_causal_graph2_pems_o(G, positions, edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str, title="PeMS Sensor Node Spatial Distribution"
):
    """
    绘制因果网络图：底图为地图（WGS84/EPSG:4326），节点坐标为经纬度 (lon, lat)
    """

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    from cartopy.io.img_tiles import Stamen

    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 地图底图设置（EPSG:4326） =====================
    data_crs = ccrs.PlateCarree()     # 数据坐标系：WGS84 lon/lat (EPSG:4326)
    map_crs  = ccrs.PlateCarree()     # 显示投影：城市尺度用它足够；也可换 ccrs.Mercator()

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.axes(projection=map_crs)

    # 从 positions 自动计算范围，保证底图与节点范围对齐
    lons = np.array([positions[i][0] for i in range(N)], dtype=float)
    lats = np.array([positions[i][1] for i in range(N)], dtype=float)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    # 城市尺度：固定最小边距 + 比例边距（你这个 LA 范围很合适）
    pad_lon = max(0.02, (lon_max - lon_min) * 0.12)
    pad_lat = max(0.02, (lat_max - lat_min) * 0.12)

    ax.set_extent([lon_min - pad_lon, lon_max + pad_lon,
                   lat_min - pad_lat, lat_max + pad_lat],
                  crs=data_crs)

    # 底图要素（国外也通用；城市尺度建议 10m 更清晰）
    ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.2)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), alpha=0.2)

    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=0.4, alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 设置 x 轴刻度
    ax.tick_params(axis='x', labelsize=12, labelcolor='black', labelrotation=0)  # 可以设置旋转角度

    # 设置 y 轴刻度
    ax.tick_params(axis='y', labelsize=12, labelcolor='black', labelrotation=0)

    # ===================== 街道瓦片底图（OSM） =====================
    tiler = cimgt.OSM()

    # 这个范围（约 0.4° x 0.3°，LA 城市尺度）推荐 12~15
    # 你可以动态估一个 zoom，或直接固定 13/14
    zoom = 13
    ax.add_image(tiler, zoom)

    # ===================== 画节点（scatter，确保 transform=data_crs） =====================
    sc = ax.scatter(
        lons, lats,
        s=300,
        c="#0E4CAB",
        edgecolors="#0E4CAB",
        linewidths=1.0,
        alpha=0.9,
        transform=data_crs,
        zorder=5
    )

    for i in range(N):
        x, y = positions[i]
        ax.text(x, y, str(i),
                fontsize=12, fontfamily='Times New Roman',color='white',
                ha='center', va='center',
                transform=data_crs,
                zorder=6)


    # 标题
    ax.set_title(title)

    plt.tight_layout()

    # 保存
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()

"""
新增函数：绘制融合 PCMCI 和 SCMI 结果的因果网络图
核心特性：
支持SCMI 无向连接的可视化（使用虚线表示）
区分 PCMCI 有向边（实线）和 SCMI 无向边（虚线）
同时考虑 PCMCI 和 SCMI 结果计算节点强度
添加图例说明不同类型的边
自循环边只绘制在有标签的节点上
"""
def plot_causal_graph3(G, positions, edge_counts, node_self_counts,
                       output_dir, tau_str, alpha_str, title="Causal Graph with Time Lag Information",
                       scmi_matrix=None):
    """
    绘制因果网络图，包括节点、边、自循环、颜色条等，现在支持SCMI无向连接
    """
    # 获取图的节点数
    N = len(G.nodes())

    # ===================== 关键修改：节点强度（node_strength） =====================
    # 节点颜色：与该节点相连的因果强度绝对值之和（入+出；自环按自身强度计一次）
    node_strength = np.zeros(N, dtype=float)

    for u, v, k, data in G.edges(keys=True, data=True):
        # 统一用 causal_effect（你边宽/边色本来也用它）；如果没有就退回 weight
        ce = data.get('causal_effect', data.get('weight', 0.0))
        w = abs(float(ce))

        if u == v:  # 自环：按该自环自己的强度计一次
            node_strength[u] += w
        else:  # 普通边：对两端节点都算"相连强度"
            node_strength[u] += w
            node_strength[v] += w

    # 如果提供了SCMI矩阵，也计入节点强度
    if scmi_matrix is not None:
        scmi_strength = np.sum(np.abs(scmi_matrix), axis=0) + np.sum(np.abs(scmi_matrix), axis=1)
        node_strength += scmi_strength

    vmax_nodes = node_strength.max() if node_strength.max() > 0 else 1.0
    node_colors = node_strength
    node_sizes = 100 + (node_strength / vmax_nodes) * 300  # 不想改大小就保持你原来的
    cmap_nodes = plt.cm.YlGn

    # ============================================================================

    # 设置绘图参数
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_aspect('equal', adjustable='box')

    nx.draw_networkx_nodes(
        G, positions,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap_nodes,
        edgecolors='white',
        linewidths=1.0,
        alpha=0.9,
        ax=ax
    )
    label_threshold = np.quantile(node_strength, 0.70)  # 只标注前30%

    # 只给强节点画编号
    labels_strong = {
        i: i
        for i in range(N)
        if node_strength[i] >= label_threshold
    }

    nx.draw_networkx_labels(
        G,
        positions,
        labels=labels_strong,
        font_size=8,
        font_color='black',
        ax=ax
    )

    # 分离不同类型边
    pcmci_edges = []
    scmi_edges = []

    for source, target, key, data in G.edges(keys=True, data=True):
        if data.get('connection_type') == 'scmi':
            scmi_edges.append((source, target, key, data))
        else:
            # PCMCI边（有向）
            if source == target:
                pcmci_edges.append(('self', (source, target, key, data)))
            else:
                pcmci_edges.append(('normal', (source, target, key, data)))

    # 获取所有权重用于颜色映射（边的颜色仍然按因果效应正负）
    all_weights = [abs(data.get('weight', data.get('causal_effect', 0.0)))
                   for source, target, key, data in G.edges(keys=True, data=True)]

    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        abs_max = max(abs(min_weight), abs(max_weight))
        vmin, vmax_edges = -abs_max, abs_max
    else:
        vmin, vmax_edges = -1, 1

    cmap = plt.cm.seismic
    norm = plt.Normalize(vmin=vmin, vmax=vmax_edges)

    # 绘制SCMI无向边
    if scmi_edges:
        scmi_edge_list = [(source, target) for source, target, key, data in scmi_edges]
        scmi_edge_colors = []
        scmi_edge_widths = []

        for source, target, key, data in scmi_edges:
            causal_effect = data.get('causal_effect', 0.0)
            color = cmap(norm(causal_effect))
            width = 0.5 + abs(causal_effect) * 3  # 调整SCMI边的宽度
            scmi_edge_colors.append(color)
            scmi_edge_widths.append(width)

        nx.draw_networkx_edges(
            G, positions,
            edgelist=scmi_edge_list,
            edge_color=scmi_edge_colors,
            width=scmi_edge_widths,
            alpha=0.6,
            style='solid',  # SCMI边使用虚线
            arrows=False,  # SCMI边无方向
            ax=ax
        )

    # 绘制PCMCI普通边（节点间连接）
    normal_edges_data = [(data[0], data[1]) for data in pcmci_edges if data[0] == 'normal']
    normal_edges_info = [data[1] for data in pcmci_edges if data[0] == 'normal']

    if normal_edges_info:
        normal_edge_colors = []
        normal_edge_widths = []

        for source, target, key, data in normal_edges_info:
            causal_effect = data['causal_effect']
            color = cmap(norm(causal_effect))
            width = 1 + abs(causal_effect) * 5  # 基础宽度1，根据因果效应绝对值调整
            normal_edge_colors.append(color)
            normal_edge_widths.append(width)

        nx.draw_networkx_edges(
            G, positions,
            edgelist=[(target, source) for source, target, key, data in normal_edges_info],  # 注意这里的方向
            edge_color=normal_edge_colors,
            width=normal_edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=10,
            ax=ax
        )

    # 绘制PCMCI自相关边（自循环）
    self_edges_info = [data[1] for data in pcmci_edges if data[0] == 'self']

    for source, target, key, data in self_edges_info:
        # ❗ 只给"显示 label 的节点"画自环
        if source not in labels_strong:
            continue
        causal_effect = data['causal_effect']
        color = cmap(norm(causal_effect))
        width = abs(causal_effect) * 2

        x, y = positions[source]
        plt.annotate(
            '',
            xy=(x - 1, y - 0.995),
            xytext=(x - 1.005, y - 1),
            arrowprops=dict(
                arrowstyle='simple',
                color=color,
                lw=width,
                alpha=0.7,
                mutation_scale=width * 3 + 10
            )
        )

    # 添加颜色图例（边：因果效应）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Causal Effect', rotation=270, labelpad=10)

    # 添加节点颜色图例（节点：连接因果强度绝对值之和）
    sm_nodes = plt.cm.ScalarMappable(
        cmap=cmap_nodes,
        norm=plt.Normalize(vmin=0, vmax=vmax_nodes if vmax_nodes > 0 else 1.0)
    )
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.040, pad=0.05, location='left')
    cbar_nodes.set_label('Sum(|Causal Effect|) of Connected Edges', rotation=90, labelpad=7)

    # 添加图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='PCMCI Directed Edges'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='SCMI Undirected Edges')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 设置标题和坐标轴
    plt.title(title + "\n(Node labels shown for top 30% Σ|Causal Effect|)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存图像
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()


def plot_causal_matrix(edge_counts, N, output_dir, tau_str, alpha_str):
    """
    生成因果连接数量矩阵热力图

    参数:
        edge_counts: 边统计字典
        N: 节点数量
        output_dir: 输出目录
        tau_str: 时间滞后字符串
        alpha_str: 显著性阈值字符串
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
    plt.savefig(os.path.join(output_dir, filename2), dpi=600, bbox_inches='tight')
    print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")
    plt.close()


"""
新增函数：绘制融合 PCMCI 和 SCMI 信息的因果矩阵热力图
核心特性：
支持 SCMI 矩阵信息的集成
当提供 scmi_matrix 时，创建综合矩阵（PCMCI 计数 + SCMI 绝对值）
标题和颜色条标签会根据是否有 SCMI 数据自动调整
保持与 plot_causal_graph3 的一致性
"""
def plot_causal_matrix2(edge_counts, N, output_dir, tau_str, alpha_str, scmi_matrix=None):
    """
    绘制因果矩阵热力图，现在包含SCMI信息

    Args:
        edge_counts: 边计数字典
        N: 节点数量
        output_dir: 输出目录
        tau_str: 滞后字符串
        alpha_str: 显著性阈值字符串
        scmi_matrix: SCMI矩阵（可选）
    """
    # 创建 N×N 矩阵
    causal_matrix = np.zeros((N, N))

    # 填充矩阵（source → target 的因果联系数量）
    for (source, target), count in edge_counts.items():
        causal_matrix[source, target] = count  # source → target

    # 如果有SCMI结果，将其叠加到PCMCI结果中
    if scmi_matrix is not None:
        # 将SCMI矩阵与PCMCI矩阵结合
        # 创建综合矩阵
        combined_matrix = causal_matrix.copy()

        # SCMI矩阵通常是N×N，表示无向关联
        # 将SCMI结果与PCMCI结果结合
        scmi_abs = np.abs(scmi_matrix)
        # 将SCMI信息叠加到综合矩阵中，可以使用不同的方式
        # 这里使用将PCMCI计数和SCMI绝对值都考虑的方式
        combined_matrix = causal_matrix + scmi_abs
    else:
        # 如果没有SCMI数据，只使用PCMCI结果
        combined_matrix = causal_matrix

    # 创建绘图对象
    fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8), dpi=300)

    # 绘制热力图 - 使用YlOrRd颜色映射保持一致性，或者使用RdBu_r来显示正负值
    cax = ax_matrix.imshow(combined_matrix, cmap='YlOrRd', interpolation='none', aspect='auto')

    # 添加颜色条
    cbar = fig_matrix.colorbar(cax, ax=ax_matrix)
    if scmi_matrix is not None:
        cbar.set_label('Combined Causal Link Count (PCMCI + SCMI)')
    else:
        cbar.set_label('Causal Link Count (τ)')

    # 设置标题和坐标轴
    if scmi_matrix is not None:
        ax_matrix.set_title("Combined Node Causal Link Matrix (PCMCI + SCMI)", fontsize=14)
    else:
        ax_matrix.set_title("Node Causal Link Count Matrix (source → target)", fontsize=14)

    ax_matrix.set_xlabel("Target Node", fontsize=12)
    ax_matrix.set_ylabel("Source Node", fontsize=12)

    # 设置坐标轴刻度
    ax_matrix.set_xticks(np.arange(N))
    ax_matrix.set_yticks(np.arange(N))
    ax_matrix.tick_params(axis='both', which='major', labelsize=5)

    plt.tight_layout()

    filename2 = f'causal_matrix_{tau_str}_{alpha_str}_new.png'
    plt.savefig(os.path.join(output_dir, filename2), dpi=300, bbox_inches='tight')
    print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename2)}")
    plt.close()


def visualize_causal_graph_from_file(pickle_file, position_file, output_dir, tau_max=3, pc_alpha=0.0001):
    """
    从文件加载数据并可视化因果图

    参数:
        pickle_file: PCMCI结果文件路径
        position_file: 节点位置CSV文件路径
        output_dir: 输出目录
        tau_max: 最大时间滞后
        pc_alpha: 显著性阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 参数格式化，用于文件名
    tau_str = f'tau{tau_max}'
    alpha_str = f'alpha{pc_alpha:.4f}'.replace('.', 'p')  # 0.001 → alpha0p001

    # 加载数据
    graph, val_matrix = load_pcmci_results(pickle_file)
    positions = load_node_positions(position_file)
    N = positions.shape[0]

    # 构建图和统计数据
    G, edge_counts, node_self_counts = build_multi_di_graph(graph, val_matrix, N, tau_max, positions)

    print(f"图中节点数: {G.number_of_nodes()}")
    print(f"图中边数: {G.number_of_edges()}")

    # 可视化因果图
    plot_causal_graph(G, dict(enumerate(positions)), edge_counts, node_self_counts,
                      output_dir, tau_str, alpha_str,
                      title=f"Causal Graph with Time Lag Information (N={N})")

    # 可视化因果矩阵热力图
    plot_causal_matrix(edge_counts, N, output_dir, tau_str, alpha_str)


def main():
    """
    主函数 - 用于独立运行脚本
    """
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 设置默认参数
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

    # 检查是否文件存在
    if not os.path.exists(pickle_file):
        print(f"错误: 找不到PCMCI结果文件 {pickle_file}")
        return

    # 加载数据
    position_file = os.path.join(dataset_dir, 'node_positions.csv')
    if not os.path.exists(position_file):
        print(f"错误: 找不到节点位置文件 {position_file}")
        return

    # 调用可视化函数
    visualize_causal_graph_from_file(pickle_file, position_file, output_dir, tau_max, pc_alpha)


if __name__ == "__main__":
    main()
