"""
load_causal_data()
加载数据
create_tigramite_dataframe()
创建tigramite数据结构
run_pcmci_analysis()
执行PCMCI分析
save_pcmci_results()
保存结果
load_pcmci_results()
加载结果
analyze_causal_links()
分析因果连接
visualize_causal_graph()
可视化因果图
visualize_causal_matrix()
可视化因果矩阵
run_pcmci_causal_analysis()
对外接口

概述
该文件实现了PCMCI（PC-Momentary Conditional Independence）因果发现算法，用于分析时间序列数据中的因果关系。
通过结合PC算法和瞬时条件独立性测试，检测变量间的直接因果连接。

函数说明

1. load_causal_data(dataset_dir)
功能: 加载因果分析所需的数据
参数: dataset_dir - 数据集目录路径
返回: (positions_df, data, var_names)

2. create_tigramite_dataframe(data, var_names)
功能: 创建tigramite DataFrame数据结构
参数: data - 时间序列数据, var_names - 变量名列表
返回: pp.DataFrame对象

3. run_pcmci_analysis(dataframe, tau_max=3, pc_alpha=0.0001)
功能: 运行PCMCI分析
参数: dataframe - tigramite DataFrame, tau_max - 最大时间滞后, pc_alpha - 显著性阈值
返回: PCMCI分析结果字典

4. save_pcmci_results(results, output_dir, tau_max, pc_alpha)
功能: 保存PCMCI分析结果到文件
参数: results - PCMCI结果, output_dir - 输出目录, tau_max - 最大时间滞后, pc_alpha - 显著性阈值
返回: 保存的文件路径

5. load_pcmci_results(pickle_file)
功能: 从文件加载PCMCI分析结果
参数: pickle_file - 结果文件路径
返回: PCMCI分析结果字典

6. analyze_causal_links(loaded_results, N, tau_max=3)
功能: 分析因果连接关系
参数: loaded_results - PCMCI结果, N - 节点数量, tau_max - 最大时间滞后
返回: (graph, val_matrix, node_self_counts, edge_counts)

7. visualize_causal_graph(graph, val_matrix, positions, node_self_counts, edge_counts, N, tau_max, output_dir, pc_alpha)
功能: 可视化因果图（带时间滞后信息）
参数: 多种图形和位置参数

8. visualize_causal_matrix(edge_counts, positions, output_dir, tau_max, pc_alpha)
功能: 可视化因果矩阵热力图
参数: edge_counts - 边连接计数, positions - 节点位置等

9. main(dataset_dir=None, output_dir=None, tau_max=3, pc_alpha=0.0001)
功能: 主函数，执行完整的PCMCI因果分析流程

10. run_pcmci_causal_analysis(dataset_dir, output_dir=None, tau_max=3, pc_alpha=0.0001)
功能: 对外提供的PCMCI因果分析接口
返回: (graph, val_matrix, node_self_counts, edge_counts, pickle_file_path)

PCMCI算法特点：
- 结合PC算法和瞬时条件独立性测试
- 考虑时间滞后效应
- 检测直接因果关系而非相关性
- 支持变量间的线性关系分析

主要应用：
- 时间序列因果关系发现
- 动态系统中的因果结构学习
- 多变量时间序列分析
- 传感器网络中的因果推断
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
from utils.euclidean_distance import compute_euclidean_matrix
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pickle
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

def haversine_km_matrix(lat_deg, lon_deg):
    """返回 N×N 的球面距离矩阵，单位 km"""
    R = 6371.0088
    lat = np.radians(lat_deg.astype(np.float64))
    lon = np.radians(lon_deg.astype(np.float64))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat/2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def build_knn_neighbors(dist_km, k):
    """返回 neighbor_list: list[list[int]]，每个节点的 k 个最近邻（不含自己）"""
    N = dist_km.shape[0]
    neighbors = []
    for i in range(N):
        nn = np.argsort(dist_km[i])[1:k+1]
        neighbors.append(nn.tolist())
    return neighbors

def build_radius_neighbors(dist_km, delta_km, cap_k=None):
    """距离阈值邻居；可选 cap_k 限制最多邻居数（强烈建议给 cap_k）"""
    N = dist_km.shape[0]
    neighbors = []
    for i in range(N):
        cand = np.where((dist_km[i] <= delta_km) & (dist_km[i] > 0))[0]
        if cap_k is not None and cand.size > cap_k:
            cand = cand[np.argsort(dist_km[i, cand])[:cap_k]]
        neighbors.append(cand.tolist())
    return neighbors

def make_selected_links_from_neighbors(neighbor_list, tau_max, include_self=True, self_lags=True):
    """
    Tigramite PCMCI selected_links 格式：
    selected_links[target] = [(source, -tau), ...]
    """
    N = len(neighbor_list)
    selected_links = {j: [] for j in range(N)}
    for j in range(N):  # target
        # 自回归边通常必须保留（否则很多系统会拟合得很怪）
        if include_self and self_lags:
            for tau in range(1, tau_max + 1):
                selected_links[j].append((j, -tau))

        # 空间邻居作为候选 source
        for i in neighbor_list[j]:
            for tau in range(1, tau_max + 1):
                selected_links[j].append((i, -tau))
    return selected_links

def make_link_assumptions_from_neighbors(neighbor_list, tau_max,
                                         include_self=True,
                                         self_link_type='-?>',
                                         other_link_type='-?>'):
    """
    返回 Tigramite 的 link_assumptions 格式：
    link_assumptions[j][(i, -tau)] = link_type

    注意：
    - 用 '-?>' 表示“允许但不强制”的有向候选边
    - 如果提供 link_assumptions，则未出现的边默认“不存在”
    """
    N = len(neighbor_list)
    link_assumptions = {j: {} for j in range(N)}

    for j in range(N):  # target
        if include_self:
            for tau in range(1, tau_max + 1):
                link_assumptions[j][(j, -tau)] = self_link_type  # 自回归：建议也先不强制

        for i in neighbor_list[j]:  # candidate sources
            for tau in range(1, tau_max + 1):
                link_assumptions[j][(i, -tau)] = other_link_type

    return link_assumptions



def load_causal_data(dataset_dir):
    """
    加载因果分析所需的数据

    Parameters:
    dataset_dir (str): 数据集目录路径

    Returns:
    tuple: (positions_df, data, var_names)
    """
    # 加载节点位置数据
    position_file = os.path.join(dataset_dir, 'id_node_positions.csv')
    positions_df = pd.read_csv(position_file)

    # 加载时间序列数据
    data_path = os.path.join(dataset_dir, 'id_standardize_Env_A_timeseries.csv')
    data = pd.read_csv(data_path).values

    # 定义变量名
    var_names = [f'Node_{i}' for i in range(data.shape[1])]

    return positions_df, data, var_names


def create_tigramite_dataframe(data, var_names):
    # 不传 datatime 最稳（默认 0..T-1），也更少出错
    return pp.DataFrame(data.astype(np.float64), var_names=var_names)


def run_pcmci_analysis(dataframe, tau_max=3, pc_alpha=0.0001, link_assumptions=None):
    import time
    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    start_time = time.perf_counter()
    results = pcmci.run_pcmciplus(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_conds_dim=4,
        link_assumptions=link_assumptions,   # ✅ 新接口
        tau_min=1                            # ✅ 你只做滞后边的话建议明确写
    )
    pcmci_time = time.perf_counter() - start_time

    print(f"PCMCI 单次核心分析耗时: {pcmci_time:.2f} 秒")
    return results


def save_pcmci_results(results, output_dir, tau_max, pc_alpha):
    """
    保存PCMCI分析结果

    Parameters:
    results (dict): PCMCI分析结果
    output_dir (str): 输出目录
    tau_max (int): 最大时间滞后
    pc_alpha (float): 显著性阈值

    Returns:
    str: 保存的文件路径
    """
    # 参数格式化，用于文件名
    tau_str = f'tau{tau_max}'
    alpha_str = f'alpha{pc_alpha:.4f}'.replace('.', 'p')

    # 保存 PCMCI 的结果到 .pkl 文件
    filename = f'pcmci_results_{tau_str}_{alpha_str}.pkl'
    pickle_file = os.path.join(output_dir, filename)

    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"完整 PCMCI 结果已保存为 pickle 文件: {pickle_file}")
    return pickle_file


def load_pcmci_results(pickle_file):
    """
    加载PCMCI分析结果

    Parameters:
    pickle_file (str): 结果文件路径

    Returns:
    dict: PCMCI分析结果
    """
    with open(pickle_file, 'rb') as f:
        loaded_results = pickle.load(f)
    return loaded_results


def analyze_causal_links(loaded_results, N, tau_max=3, verbose=False):
    graph = loaded_results['graph']
    val_matrix = loaded_results['val_matrix']

    node_self_counts = np.zeros(N)
    for i in range(N):
        for tau in range(1, tau_max + 1):
            node_self_counts[i] += (graph[i, i, tau] == '-->')

    edge_counts = {}
    for target in range(N):
        for source in range(N):
            # if target == source:
            #     continue
            count = 0
            for tau in range(1, tau_max + 1):
                if graph[source,target, tau] == '-->':
                    count += 1
            if count > 0:
                edge_counts[(source, target)] = count

    if verbose:
        print("edges:", len(edge_counts))
    return graph, val_matrix, node_self_counts, edge_counts


def visualize_causal_graph(graph, val_matrix, positions, node_self_counts, edge_counts,
                           N, tau_max, output_dir, pc_alpha):
    """
    可视化因果图

    Parameters:
    graph: 因果关系图
    val_matrix: 因果效应矩阵
    positions: 节点位置
    node_self_counts: 节点自关联计数
    edge_counts: 边连接计数
    N: 节点数量
    tau_max: 最大时间滞后
    output_dir: 输出目录
    pc_alpha: 显著性阈值
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
    node_sizes = 100 + degrees * 10
    cmap_nodes = plt.cm.YlGn

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, cmap=cmap_nodes,
                           edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=12, font_family= 'Times New Roman',font_color='black', ax=ax)

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
    # for u, v, key, data in self_edges:
    #     lag = data['lag']
    #     causal_effect = data['causal_effect']
    #
    #     # 根据权重大小设置颜色
    #     color = cmap(norm(causal_effect))
    #
    #     # 根据权重绝对值设置边宽度
    #     width = 1 + abs(causal_effect) * 8
    #     x, y = pos[u]
    #     dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    #     dy = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    #     plt.annotate(
    #         '',
    #         xy=(x-dx, y-dx),
    #         xytext=(x-dy, y-dy),
    #         arrowprops=dict(
    #             arrowstyle='simple',
    #             color=color,
    #             lw=width*0.5,
    #             alpha=0.7,
    #             mutation_scale=width*1.1+2
    #         )
    #     )
    #     # 绘制自循环边


    # # 添加边标签（包含时间滞后和效应符号）
    # edge_labels = {}
    # for u, v, key, data in G.edges(keys=True, data=True):
    #     lag = data['lag']
    #     causal_effect = data['causal_effect']
    #     sign = '+' if causal_effect >= 0 else '-'
    #     edge_labels[(u, v, key)] = f"{sign}τ{lag}"
    #
    # # 绘制边滞后标签
    # for (u, v, key), label in edge_labels.items():
    #     # 计算边的中点位置
    #     if u == v:  # 自循环
    #         # 自循环标签位置
    #         x, y = pos[u]
    #         label_pos = (x + 0.005, y + 0.005)
    #     else:  # 普通边
    #         # 普通边中点位置
    #         x1, y1 = pos[u]
    #         x2, y2 = pos[v]
    #         label_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
    #
    #         # 如果有多条边连接同一对节点，稍微偏移标签位置
    #         edge_count = len([k for uu, vv, k, d in G.edges(keys=True, data=True) if (uu, vv) == (u, v)])
    #         if edge_count > 1:
    #             label_pos = (label_pos[0] + 0.005, label_pos[1] + 0.005)
    #
    #     # 根据因果效应符号设置标签颜色
    #     causal_effect = None
    #     for uu, vv, kk, dd in G.edges(keys=True, data=True):
    #         if (uu, vv, kk) == (u, v, key):
    #             causal_effect = dd['causal_effect']
    #             break
    #
    #     label_color = 'red' if causal_effect is not None and causal_effect >= 0 else 'blue'
    #
    #     plt.text(label_pos[0], label_pos[1], label,
    #              fontsize=4, ha='center', va='center', color=label_color,
    #              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 添加颜色图例（因果效应）
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label('Causal Effect', fontsize=14, rotation=270, labelpad=10)

    # 添加节点出边数量颜色图例
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.030, pad=0.05, location='left')
    cbar_nodes.set_label('Number of Outgoing Edges',fontsize=14,rotation=90, labelpad=7)

    # 设置标题和坐标轴
    # plt.title("Causal Graph with Time Lag Information")
    plt.xlabel("Longitude",fontsize=14)
    plt.ylabel("Latitude",fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 保存图像
    tau_str = f'tau{tau_max}'
    alpha_str = f'alpha{pc_alpha:.4f}'.replace('.', 'p')
    filename1 = f'network_causal_graph_{tau_str}_{alpha_str}'
    # 1. 保存高清 SVG (矢量图，无损缩放)
    svg_path = os.path.join(output_dir, filename1 + '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    # 2. 保存 PNG (预览用)
    png_path = os.path.join(output_dir, filename1 + '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')  # 高DPI

    print(f"因果图已保存至：{os.path.join(output_dir, filename1)}")
    plt.close()


def visualize_causal_matrix(edge_counts, positions, output_dir, tau_max, pc_alpha):
    """
    可视化因果矩阵热力图

    Parameters:
    edge_counts: 边连接计数字典
    positions: 节点位置
    output_dir: 输出目录
    tau_max: 最大时间滞后
    pc_alpha: 显著性阈值
    """
    N = positions.shape[0]
    # 创建 N×N 矩阵
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

    # 保存图像
    tau_str = f'tau{tau_max}'
    alpha_str = f'alpha{pc_alpha:.4f}'.replace('.', 'p')
    filename = f'causal_matrix_{tau_str}_{alpha_str}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=1800, bbox_inches='tight')
    print(f"因果矩阵热力图已保存至：{os.path.join(output_dir, filename)}")


def main(dataset_dir=None, output_dir=None, tau_max=3, pc_alpha=0.05):
    """
    主函数：执行完整的PCMCI因果分析流程

    Parameters:
    dataset_dir (str): 数据集目录路径
    output_dir (str): 输出目录路径
    tau_max (int): 最大时间滞后
    pc_alpha (float): 显著性阈值
    """
    # 设置默认路径
    if dataset_dir is None:
        dataset_dir = os.path.join(current_dir, '..', 'dataset/pems_spatial_kmeans_topK')

    if output_dir is None:
        output_dir = os.path.join(current_dir, 'PCMCI_PEMS/N100_1h_standardize_id_7')

    # 确保目标路径存在
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("开始PCMCI因果分析...")

    # 1. 加载数据
    print("1. 加载数据...")
    positions_df, data, var_names = load_causal_data(dataset_dir)
    positions = positions_df[['x', 'y']].values
    N = positions.shape[0]

    # === 计算真实距离(km)并生成空间邻接 ===
    lon = positions_df['x'].to_numpy()
    lat = positions_df['y'].to_numpy()
    dist_km = haversine_km_matrix(lat, lon)

    KNN_K = 10  # 建议 8~15（和你 SCMI 的 k_neighbors 对齐最好）
    neighbor_list = build_knn_neighbors(dist_km, k=KNN_K)
    avg_deg = np.mean([len(nbs) for nbs in neighbor_list])
    max_deg = np.max([len(nbs) for nbs in neighbor_list])
    min_deg = np.min([len(nbs) for nbs in neighbor_list])
    print(f"[PCMCI] neighbor degree: avg={avg_deg:.2f}, min={min_deg}, max={max_deg}")

    link_assumptions = make_link_assumptions_from_neighbors(
        neighbor_list, tau_max=tau_max, include_self=True
    )

    print(f"[PCMCI] Using KNN link_assumptions: k={KNN_K}, tau_max={tau_max}")

    # 2. 创建tigramite DataFrame
    print("2. 创建tigramite DataFrame...")
    dataframe = create_tigramite_dataframe(data, var_names)

    # 3. 运行PCMCI分析
    print("3. 运行PCMCI分析...")
    results = run_pcmci_analysis(dataframe, tau_max, pc_alpha, link_assumptions)

    # 4. 保存结果
    print("4. 保存结果...")
    pickle_file = save_pcmci_results(results, output_dir, tau_max, pc_alpha)

    # 5. 加载结果
    print("5. 加载结果...")
    loaded_results = load_pcmci_results(pickle_file)

    # 6. 分析因果连接
    print("6. 分析因果连接...")
    graph, val_matrix, node_self_counts, edge_counts = analyze_causal_links(
        loaded_results, N, tau_max)

    # 7. 可视化因果图
    print("7. 可视化因果图...")
    visualize_causal_graph(graph, val_matrix, positions, node_self_counts,
                           edge_counts, N, tau_max, output_dir, pc_alpha)

    # 8. 可视化因果矩阵
    print("8. 可视化因果矩阵...")
    visualize_causal_matrix(edge_counts, positions, output_dir, tau_max, pc_alpha)

    print("PCMCI因果分析完成！")


# 便于外部调用的接口函数
def run_pcmci_causal_analysis(dataset_dir, output_dir=None, tau_max=3, pc_alpha=0.0001):
    """
    对外提供的PCMCI因果分析接口

    Parameters:
    dataset_dir (str): 数据集目录路径
    output_dir (str): 输出目录路径
    tau_max (int): 最大时间滞后
    pc_alpha (float): 显著性阈值

    Returns:
    tuple: (graph, val_matrix, node_self_counts, edge_counts, pickle_file_path)
    """
    if output_dir is None:
        output_dir = os.path.join(current_dir, 'PCMCI_2')

    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    positions_df, data, var_names = load_causal_data(dataset_dir)
    positions = positions_df[['x', 'y']].values
    N = positions.shape[0]

    # === 计算真实距离(km)并生成空间邻接 ===
    # lon = positions_df['x'].to_numpy()
    # lat = positions_df['y'].to_numpy()
    from scipy.spatial.distance import pdist, squareform
    # dist_km =  haversine_km_matrix(lat, lon)
    coords = positions_df[['x', 'y']].values
    dist_km = squareform(pdist(coords))
    KNN_K = 10  # 建议 8~15（和你 SCMI 的 k_neighbors 对齐最好）
    neighbor_list = build_knn_neighbors(dist_km, k=KNN_K)
    avg_deg = np.mean([len(nbs) for nbs in neighbor_list])
    max_deg = np.max([len(nbs) for nbs in neighbor_list])
    min_deg = np.min([len(nbs) for nbs in neighbor_list])
    print(f"[PCMCI] neighbor degree: avg={avg_deg:.2f}, min={min_deg}, max={max_deg}")

    link_assumptions = make_link_assumptions_from_neighbors(
        neighbor_list, tau_max=tau_max, include_self=True
    )

    print(f"[PCMCI] Using KNN link_assumptions: k={KNN_K}, tau_max={tau_max}")


    # 创建tigramite DataFrame
    dataframe = create_tigramite_dataframe(data, var_names)

    # 运行PCMCI分析
    results = run_pcmci_analysis(dataframe, tau_max, pc_alpha,link_assumptions)

    # 保存结果
    pickle_file = save_pcmci_results(results, output_dir, tau_max, pc_alpha)

    # 加载结果
    loaded_results = load_pcmci_results(pickle_file)

    # 分析因果连接
    graph, val_matrix, node_self_counts, edge_counts = analyze_causal_links(
        loaded_results, N, tau_max)

    return graph, val_matrix, node_self_counts, edge_counts, pickle_file


if __name__ == "__main__":
    # 当直接运行此脚本时执行完整流程
    main()
