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
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

def load_causal_data(dataset_dir, max_time_steps=None):
    """
    加载因果分析所需的数据

    Parameters:
    dataset_dir (str): 数据集目录路径

    Returns:
    tuple: (positions_df, data, var_names)
    """
    # 加载节点位置数据
    position_file = os.path.join(dataset_dir, 'node_positions.csv')
    positions_df = pd.read_csv(position_file)

    # 加载时间序列数据
    data_path = os.path.join(dataset_dir, 'Env_A_timeseries.csv')
    data = pd.read_csv(data_path).values
    # 如果指定了max_time_steps，则截取前max_time_steps行数据
    if max_time_steps is not None and max_time_steps < data.shape[0]:
        data = data[:max_time_steps, :]
    # 定义变量名
    var_names = [f'Node_{i}' for i in range(data.shape[1])]

    return positions_df, data, var_names


def create_tigramite_dataframe(data, var_names):
    """
    创建tigramite DataFrame

    Parameters:
    data (np.array): 时间序列数据
    var_names (list): 变量名列表

    Returns:
    pp.DataFrame: tigramite DataFrame对象
    """
    return pp.DataFrame(data, var_names=var_names, datatime=np.arange(data.shape[1]))


def run_pcmci_analysis(dataframe, tau_max=3, pc_alpha=0.0001):
    import time

    """
    运行PCMCI分析

    Parameters:
    dataframe (pp.DataFrame): tigramite DataFrame对象
    tau_max (int): 最大时间滞后
    pc_alpha (float): 显著性阈值

    Returns:
    dict: PCMCI分析结果
    """
    # 初始化条件独立性检验
    cond_ind_test = ParCorr(significance='analytic')
    # 初始化 PCMCI 对象
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    # 运行PCMCI分析
    start_time = time.time()
    results = pcmci.run_pcmciplus(
        tau_max=tau_max,
        pc_alpha=pc_alpha
    )
    elapsed_time = time.time() - start_time
    print(f"PCMCI分析耗时: {elapsed_time:.2f} 秒")

    return results, elapsed_time


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


def analyze_causal_links(loaded_results, N, tau_max=3):
    """
    分析因果连接关系

    Parameters:
    loaded_results (dict): 加载的PCMCI结果
    N (int): 节点数量
    tau_max (int): 最大时间滞后

    Returns:
    tuple: (graph, val_matrix, node_self_counts, edge_counts)
    """
    graph = loaded_results['graph']
    val_matrix = loaded_results['val_matrix']

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
            count = 0
            for tau in range(1, tau_max + 1):
                # graph[target, source, tau] == '-->' 表示 source 在 t-τ 时刻对 target 在 t 时刻有因果影响
                if graph[target, source, tau] == '-->':
                    count += 1
                    print(graph[target, source, tau])
                    #print(f"节点{source}(源)对节点{target}(目标)因果效应{val_matrix[target, source, tau]}")
            if count > 0:
                # edge_counts[(source, target)] 表示从source到target的连接
                edge_counts[(source, target)] = count
                print(f"节点{source}链接到节点{target}共{count}条")

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
                    #print(f"添加边: 节点{i} -> 节点{j}, 滞后{tau}, 因果效应{causal_effect:.4f}")

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
    node_sizes = 50 + degrees * 10
    cmap_nodes = plt.cm.YlGn

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, cmap=cmap_nodes,
                           edgecolors='white', linewidths=0.5, alpha=0.9, ax=ax)

    #nx.draw_networkx_labels(G, pos, font_size=12,font_family='Times New Roman',  font_color='black', ax=ax)

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
        width = 0.5 + abs(causal_effect) * 5  # 基础宽度1，根据因果效应绝对值调整

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
        width = 0.5 + abs(causal_effect) * 5

        # 绘制自循环边
        # nx.draw_networkx_edges(G, pos, edgelist=[(u, v, key)],
        #                        width=width,
        #                        edge_color=[color],
        #                        alpha=0.7, arrows=True, arrowsize=20,
        #                        connectionstyle=f'arc3,rad=0.3', ax=ax)  # 弯曲的自循环

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
    #         label_pos = (x + 1.5, y + 1.5)
    #     else:  # 普通边
    #         # 普通边中点位置
    #         x1, y1 = pos[u]
    #         x2, y2 = pos[v]
    #         label_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
    #
    #         # 如果有多条边连接同一对节点，稍微偏移标签位置
    #         edge_count = len([k for uu, vv, k, d in G.edges(keys=True, data=True) if (uu, vv) == (u, v)])
    #         if edge_count > 1:
    #             label_pos = (label_pos[0] + 0.5, label_pos[1] + 0.5)
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
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label('Causal Effect', fontsize=16,rotation=270, labelpad=10)

    # 添加节点出边数量颜色图例
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.YlGn, norm=plt.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1))
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.040, pad=0.05, location='left')
    cbar_nodes.set_label('Number of Outgoing Edges',fontsize=16, rotation=90, labelpad=7)

    # 设置标题和坐标轴
    # plt.title("Causal Graph with Time Lag Information")
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
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


def main(dataset_dir=None, output_dir=None, tau_max=3, pc_alpha=0.05, max_time_steps=None):
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
        dataset_dir = os.path.join(current_dir, '..', 'dataset/simulation_hubs_core/N1000_T10000_Hubs80')

    if output_dir is None:
        output_dir = os.path.join(current_dir, 'PCMCI_ex/N1000_T10000')

    # 确保目标路径存在
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    max_time_steps=10000

    print("开始PCMCI因果分析...")

    # 1. 加载数据
    print("1. 加载数据...")
    positions_df, data, var_names = load_causal_data(dataset_dir, max_time_steps)
    positions = positions_df[['x', 'y']].values
    N = positions.shape[0]

    # 2. 创建tigramite DataFrame
    print("2. 创建tigramite DataFrame...")
    dataframe = create_tigramite_dataframe(data, var_names)

    # 3. 运行PCMCI分析
    print("3. 运行PCMCI分析...")
    results, pcmci_time = run_pcmci_analysis(dataframe, tau_max, pc_alpha)

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
    print(f"PCMCI分析耗时: {pcmci_time:.2f} 秒")
    print("PCMCI因果分析完成！")


# 便于外部调用的接口函数
def run_pcmci_causal_analysis(dataset_dir, output_dir=None, tau_max=3, pc_alpha=0.0001, max_time_steps=None):
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
    positions_df, data, var_names = load_causal_data(dataset_dir, max_time_steps)
    positions = positions_df[['x', 'y']].values
    N = positions.shape[0]

    # 创建tigramite DataFrame
    dataframe = create_tigramite_dataframe(data, var_names)

    # 运行PCMCI分析
    results, pcmci_time = run_pcmci_analysis(dataframe, tau_max, pc_alpha)

    # 保存结果
    pickle_file = save_pcmci_results(results, output_dir, tau_max, pc_alpha)

    # 加载结果
    loaded_results = load_pcmci_results(pickle_file)

    # 分析因果连接
    graph, val_matrix, node_self_counts, edge_counts = analyze_causal_links(
        loaded_results, N, tau_max)

    return graph, val_matrix, node_self_counts, edge_counts, pickle_file, pcmci_time



if __name__ == "__main__":
    # 当直接运行此脚本时执行完整流程
    main()
