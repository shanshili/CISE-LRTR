"""
Causal_simulation_perform.py 代码功能说明

概述
该文件用于分析因果连接网络，计算节点的出边和入边权重，进行节点重要性排序和可视化。
通过多种可视化方式（散点图、子网络图、颜色映射图）展示因果网络中重要节点的空间分布。

函数说明

1. analyze_causal_links(links_coeffs, node_positions, N)
功能: 分析因果连接，计算节点出边和入边的权重总和、连接数量，并进行排序
参数:
  - links_coeffs: 因果连接字典，格式为{目标节点: [(源节点, tau), 系数, 函数]}
  - node_positions: 节点坐标数组，形状为(N, 2)
  - N: 节点总数
返回: 包含分析结果的字典，包括出/入边权重、连接数、排序索引等

2. visualize_top_nodes(links_coeffs, node_positions, analysis_result, output_dir, select_node=18)
功能: 可视化权重最高的指定数量节点及其连接关系
参数:
  - links_coeffs: 因果连接字典
  - node_positions: 节点坐标数组
  - analysis_result: analyze_causal_links返回的分析结果
  - output_dir: 输出目录
  - select_node: 选择的节点数量，默认为18
返回: 无，但会保存可视化图像到指定目录

3. visualize_subnetwork(links_coeffs, node_positions, selected_nodes, weights, title, filename)
功能: 可视化选中节点的子网络
参数:
  - links_coeffs: 因果连接字典
  - node_positions: 节点坐标数组
  - selected_nodes: 选中的节点ID数组
  - weights: 节点权重数组
  - title: 图标题
  - filename: 保存的文件名
返回: 无，但会保存可视化图像

4. plot_causal_importance_scatter(analysis_result, node_positions, output_dir, select_node=18)
功能: 绘制因果重要性排序结果，样式与PageRank.py中的一致
参数:
  - analysis_result: analyze_causal_links返回的分析结果
  - node_positions: 节点坐标数组
  - output_dir: 输出目录
  - select_node: 选择的节点数，默认为18
返回: 无，但会保存散点图到指定目录

5. plot_causal_scatter_weights(analysis_result, node_positions, output_dir, select_node=18)
功能: 绘制因果重要性的颜色映射散点图，使用简单的颜色映射
参数:
  - analysis_result: analyze_causal_links返回的分析结果
  - node_positions: 节点坐标数组
  - output_dir: 输出目录
  - select_node: 选择的节点数，默认为18
返回: 无，但会保存颜色映射散点图到指定目录

6. print_top_nodes(analysis_result, N=10)
功能: 打印最重要的N个节点
参数:
  - analysis_result: analyze_causal_links返回的分析结果
  - N: 打印前N个节点
返回: 无，但会在控制台输出排序结果

7. print_analysis_results(analysis_result, N)
功能: 打印分析结果
参数:
  - analysis_result: analyze_causal_links返回的分析结果
  - N: 节点总数
返回: 无，但会在控制台输出详细分析结果

使用示例
# 假设 links_coeffs, node_positions, N, output_dir 已经定义

# 执行分析
analysis_result = analyze_causal_links(links_coeffs, node_positions, N)

# 打印结果
print_analysis_results(analysis_result, N)

# 可视化结果
visualize_top_nodes(links_coeffs, node_positions, analysis_result, output_dir)

主要功能特点
- 因果权重分析: 计算每个节点的出边和入边权重总和
- 重要性排序: 基于权重对节点进行降序排序
- 多维度可视化: 提供子网络图、散点图、颜色映射图等多种可视化方式
- 节点筛选: 可灵活选择需要可视化的重要节点数量
- 结果导出: 自动保存可视化结果到指定目录
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os


# 假设 links_coeffs 和 node_positions 已定义（从你的代码中获取）
# 这里我们直接使用你代码中的 links_coeffs 和 node_positions

def analyze_causal_links(links_coeffs, node_positions, N):
    """
    分析因果连接，计算节点出边和入边的权重总和，并进行排序和可视化

    Parameters:
    links_coeffs: 因果连接字典
    node_positions: 节点坐标
    N: 节点总数
    """

    # 初始化出边和入边权重总和
    out_weights = np.zeros(N)
    in_weights = np.zeros(N)

    # 初始化出边和入边计数
    out_counts = np.zeros(N)
    in_counts = np.zeros(N)

    # 遍历所有连接，计算每个节点的出边和入边权重
    for target_node in links_coeffs:
        for link in links_coeffs[target_node]:
            (source_node, tau), coeff, func = link
            # 累计入边权重（指向该节点的连接）
            in_weights[target_node] += abs(coeff)
            in_counts[target_node] += 1

            # 累计出边权重（从该节点出发的连接）
            out_weights[source_node] += abs(coeff)
            out_counts[source_node] += 1

    # 创建排序索引
    out_weights_sorted_idx = np.argsort(out_weights)[::-1]  # 降序排列
    in_weights_sorted_idx = np.argsort(in_weights)[::-1]  # 降序排列

    # 选择前18个节点进行可视化
    top_out_nodes = out_weights_sorted_idx[:18]
    top_in_nodes = in_weights_sorted_idx[:18]

    return {
        'out_weights': out_weights,
        'in_weights': in_weights,
        'out_counts': out_counts,
        'in_counts': in_counts,
        'out_weights_sorted_idx': out_weights_sorted_idx,
        'in_weights_sorted_idx': in_weights_sorted_idx,
        'top_out_nodes': top_out_nodes,
        'top_in_nodes': top_in_nodes
    }


def visualize_top_nodes(links_coeffs, node_positions, analysis_result, output_dir, select_node=18):
    """
    可视化权重最高的指定数量节点及其连接关系

    Parameters:
    links_coeffs: 因果连接字典
    node_positions: 节点坐标数组
    analysis_result: 分析结果字典
    output_dir: 输出目录
    select_node: 选择的节点数量，默认为18
    """
    # 根据指定数量选择节点
    top_out_nodes = analysis_result['out_weights_sorted_idx'][:select_node]
    top_in_nodes = analysis_result['in_weights_sorted_idx'][:select_node]
    out_weights = analysis_result['out_weights']
    in_weights = analysis_result['in_weights']

    # 可视化出边权重最高的指定数量节点
    visualize_subnetwork(links_coeffs, node_positions, top_out_nodes,
                         out_weights, f"Top {select_node} Nodes by Outgoing Edge Weights",
                         os.path.join(output_dir, f"top_{select_node}_out_weights_nodes.png"))

    # 可视化入边权重最高的指定数量节点
    visualize_subnetwork(links_coeffs, node_positions, top_in_nodes,
                         in_weights, f"Top {select_node} Nodes by Incoming Edge Weights",
                         os.path.join(output_dir, f"top_{select_node}_in_weights_nodes.png"))


def visualize_subnetwork(links_coeffs, node_positions, selected_nodes, weights, title, filename):
    """
    可视化选中节点的子网络
    """
    # 创建子图
    G = nx.DiGraph()

    # 添加选中的节点
    for node in selected_nodes:
        G.add_node(node)

    # 添加这些节点之间的连接
    for target_node in selected_nodes:
        if target_node in links_coeffs:
            for link in links_coeffs[target_node]:
                (source_node, tau), coeff, func = link
                # 只添加涉及选中节点的连接
                if source_node in selected_nodes:
                    G.add_edge(source_node, target_node, weight=abs(coeff))

    # 设置节点位置
    pos = {i: (node_positions[i, 0], node_positions[i, 1]) for i in selected_nodes}

    # 节点大小基于权重
    node_sizes = [weights[node] * 1000 for node in selected_nodes]
    # 确保节点大小不为0
    node_sizes = [max(size, 100) for size in node_sizes]

    # 边的宽度基于权重
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    edge_widths = [w * 5 for w in edge_weights]

    # 绘制图形
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=8)

    if edge_widths:  # 只有当存在边时才绘制
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.5, arrows=True, arrowsize=20)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_causal_importance_scatter(analysis_result, node_positions, output_dir, select_node=18):
    """
    绘制因果重要性排序结果，样式与PageRank.py中的一致

    Parameters:
    analysis_result: analyze_causal_links返回的分析结果
    node_positions: 节点坐标数组
    output_dir: 输出目录
    select_node: 选择的节点数，默认为18
    """
    # 获取出边权重排序结果
    out_weights_sorted_idx = analysis_result['out_weights_sorted_idx']
    in_weights_sorted_idx = analysis_result['in_weights_sorted_idx']
    out_weights = analysis_result['out_weights']
    in_weights = analysis_result['in_weights']

    # 提取坐标
    lon = node_positions[:, 0]
    lat = node_positions[:, 1]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 基于出边权重的重要节点可视化
    selected_node_out = out_weights_sorted_idx[:select_node]

    # 主要节点选择图
    plt.figure(figsize=(10, 8))
    plt.scatter(lon, lat, s=15, c='gainsboro', label='All nodes')
    plt.scatter(lon[selected_node_out], lat[selected_node_out], s=30, c='#7e2f8e', label=f'Optimized nodes (Top {select_node} Out-weight)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Optimized nodes subset (Top {select_node} Out-weights)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'causal_out_weights_top_{select_node}_nodes.svg'), format='svg', bbox_inches='tight')
    plt.close()

    # 带备选节点的图
    plt.figure(figsize=(10, 8))
    plt.scatter(lon, lat, s=15, c='gainsboro', label='All nodes')
    plt.scatter(lon[selected_node_out], lat[selected_node_out], s=30, c='#7e2f8e', label=f'Optimized nodes (Top {select_node} Out-weight)')
    # 添加备选节点
    if select_node < len(out_weights_sorted_idx):
        backup_node = out_weights_sorted_idx[select_node]
        plt.scatter(lon[backup_node], lat[backup_node], s=30, c='#d400fe',
                    label='Contribution weights backup node')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Optimized nodes subset with backup (Top {select_node} Out-weights)')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'causal_out_weights_top_{select_node}_nodes_backup.svg'), format='svg',
                bbox_inches='tight')
    plt.close()

    # 2. 基于入边权重的重要节点可视化
    selected_node_in = in_weights_sorted_idx[:select_node]

    # 主要节点选择图
    plt.figure(figsize=(10, 8))
    plt.scatter(lon, lat, s=15, c='gainsboro', label='All nodes')
    plt.scatter(lon[selected_node_in], lat[selected_node_in], s=30, c='#7e2f8e', label=f'Optimized nodes (Top {select_node} In-weight)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Optimized nodes subset (Top {select_node} In-weights)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'causal_in_weights_top_{select_node}_nodes.svg'), format='svg', bbox_inches='tight')
    plt.close()


    # 带备选节点的图
    plt.figure(figsize=(10, 8))
    plt.scatter(lon, lat, s=15, c='gainsboro', label='All nodes')
    plt.scatter(lon[selected_node_in], lat[selected_node_in], s=30, c='#7e2f8e', label=f'Optimized nodes (Top {select_node} In-weight)')
    # 添加备选节点
    if select_node < len(in_weights_sorted_idx):
        backup_node = in_weights_sorted_idx[select_node]
        plt.scatter(lon[backup_node], lat[backup_node], s=30, c='#d400fe',
                    label='Contribution weights backup node')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Optimized nodes subset with backup (Top {select_node} In-weights)')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(output_dir, f'causal_in_weights_top_{select_node}_nodes_backup.svg'), format='svg',
                bbox_inches='tight')
    plt.close()

    # # 3. 绘制权重排序曲线图
    # # 出边权重排序曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(out_weights) + 1), np.sort(out_weights)[::-1], '-o', markersize=3)
    # plt.xlabel('Node Rank')
    # plt.ylabel('Out-weight Value')
    # plt.title('Out-weight Distribution of Nodes')
    # plt.grid(True, alpha=0.3)
    # plt.savefig(os.path.join(output_dir, 'causal_out_weights_distribution.svg'), format='svg', bbox_inches='tight')
    # plt.close()
    #
    # # 入边权重排序曲线
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(in_weights) + 1), np.sort(in_weights)[::-1], '-o', markersize=3)
    # plt.xlabel('Node Rank')
    # plt.ylabel('In-weight Value')
    # plt.title('In-weight Distribution of Nodes')
    # plt.grid(True, alpha=0.3)
    # plt.savefig(os.path.join(output_dir, 'causal_in_weights_distribution.svg'), format='svg', bbox_inches='tight')
    # plt.close()

    print(f"因果重要性可视化结果已保存到: {output_dir}")


def plot_causal_scatter_weights(analysis_result, node_positions, output_dir, select_node=18):
    """
    绘制因果重要性的颜色映射散点图，使用简单的颜色映射

    Parameters:
    analysis_result: analyze_causal_links返回的分析结果
    node_positions: 节点坐标数组
    output_dir: 输出目录
    select_node: 选择的节点数，默认为18
    """
    # 获取出边和入边权重
    out_weights_sorted_idx = analysis_result['out_weights_sorted_idx']
    in_weights_sorted_idx = analysis_result['in_weights_sorted_idx']
    out_weights = analysis_result['out_weights']
    in_weights = analysis_result['in_weights']

    # 提取坐标
    lon = node_positions[:, 0]
    lat = node_positions[:, 1]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 基于出边权重的颜色映射散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(lon, lat, s=50, c=out_weights, cmap='plasma_r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Out-weight Distribution of Nodes')

    # 标注极值点
    max_idx_out = np.argmax(out_weights)
    min_idx_out = np.argmin(out_weights)
    plt.annotate(f'Max({max_idx_out}): {out_weights[max_idx_out]:.6f}',
                 (lon[max_idx_out], lat[max_idx_out]),
                 xytext=(lon[max_idx_out] + 0.1, lat[max_idx_out] + 0.1),
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.annotate(f'Min({min_idx_out}): {out_weights[min_idx_out]:.6f}',
                 (lon[min_idx_out], lat[min_idx_out]),
                 xytext=(lon[min_idx_out] + 0.1, lat[min_idx_out] - 0.1),
                 arrowprops=dict(arrowstyle='->', color='blue'))

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Out-weight Value', rotation=270, labelpad=20)

    # 标注选定的重要节点
    selected_node_out = out_weights_sorted_idx[:select_node]
    plt.scatter(lon[selected_node_out], lat[selected_node_out],
                s=80, facecolors='none', edgecolors='red', linewidths=2,
                label=f'Top {select_node} Out-weight Nodes')
    plt.legend()

    plt.savefig(os.path.join(output_dir, f'causal_out_weights_scatter_top_{select_node}.svg'), format='svg',
                bbox_inches='tight')
    plt.close()

    # 2. 基于入边权重的颜色映射散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(lon, lat, s=50, c=in_weights, cmap='plasma_r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('In-weight Distribution of Nodes')

    # 标注极值点
    max_idx_in = np.argmax(in_weights)
    min_idx_in = np.argmin(in_weights)
    plt.annotate(f'Max({max_idx_in}): {in_weights[max_idx_in]:.6f}',
                 (lon[max_idx_in], lat[max_idx_in]),
                 xytext=(lon[max_idx_in] + 0.1, lat[max_idx_in] + 0.1),
                 arrowprops=dict(arrowstyle='->', color='red'))
    plt.annotate(f'Min({min_idx_in}): {in_weights[min_idx_in]:.6f}',
                 (lon[min_idx_in], lat[min_idx_in]),
                 xytext=(lon[min_idx_in] + 0.1, lat[min_idx_in] - 0.1),
                 arrowprops=dict(arrowstyle='->', color='blue'))

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('In-weight Value', rotation=270, labelpad=20)

    # 标注选定的重要节点
    selected_node_in = in_weights_sorted_idx[:select_node]
    plt.scatter(lon[selected_node_in], lat[selected_node_in],
                s=80, facecolors='none', edgecolors='red', linewidths=2,
                label=f'Top {select_node} In-weight Nodes')
    plt.legend()

    plt.savefig(os.path.join(output_dir, f'causal_in_weights_scatter_top_{select_node}.svg'), format='svg',
                bbox_inches='tight')
    plt.close()

    print(f"因果重要性颜色映射散点图已保存到: {output_dir}")


def print_top_nodes(analysis_result, N=10):
    """
    打印最重要的N个节点

    Parameters:
    analysis_result: analyze_causal_links返回的分析结果
    N: 打印前N个节点
    """
    out_weights_sorted_idx = analysis_result['out_weights_sorted_idx']
    in_weights_sorted_idx = analysis_result['in_weights_sorted_idx']
    out_weights = analysis_result['out_weights']
    in_weights = analysis_result['in_weights']

    print("\n=== 基于出边权重的最重要节点 ===")
    print(f"{'排名':<4} {'节点ID':<8} {'出边权重':<12}")
    print("-" * 25)
    for i in range(min(N, len(out_weights_sorted_idx))):
        node_id = out_weights_sorted_idx[i]
        print(f"{i + 1:<4} {node_id:<8} {out_weights[node_id]:<12.6f}")

    print("\n=== 基于入边权重的最重要节点 ===")
    print(f"{'排名':<4} {'节点ID':<8} {'入边权重':<12}")
    print("-" * 25)
    for i in range(min(N, len(in_weights_sorted_idx))):
        node_id = in_weights_sorted_idx[i]
        print(f"{i + 1:<4} {node_id:<8} {in_weights[node_id]:<12.6f}")



def print_analysis_results(analysis_result, N):
    """
    打印分析结果
    """
    out_weights = analysis_result['out_weights']
    in_weights = analysis_result['in_weights']
    out_counts = analysis_result['out_counts']
    in_counts = analysis_result['in_counts']
    out_weights_sorted_idx = analysis_result['out_weights_sorted_idx']
    in_weights_sorted_idx = analysis_result['in_weights_sorted_idx']

    print("=== 节点出边权重排序 (前20个) ===")
    print(f"{'排名':<4} {'节点ID':<8} {'出边权重总和':<15} {'出边数量':<10}")
    print("-" * 40)
    for i in range(min(20, N)):
        node_id = out_weights_sorted_idx[i]
        print(f"{i + 1:<4} {node_id:<8} {out_weights[node_id]:<15.4f} {out_counts[node_id]:<10.0f}")

    print("\n=== 节点入边权重排序 (前20个) ===")
    print(f"{'排名':<4} {'节点ID':<8} {'入边权重总和':<15} {'入边数量':<10}")
    print("-" * 40)
    for i in range(min(20, N)):
        node_id = in_weights_sorted_idx[i]
        print(f"{i + 1:<4} {node_id:<8} {in_weights[node_id]:<15.4f} {in_counts[node_id]:<10.0f}")

# 使用示例（你需要在你的代码中调用这些函数）:
# 假设 links_coeffs, node_positions, N, output_dir 已经定义

# 执行分析
# analysis_result = analyze_causal_links(links_coeffs, node_positions, N)

# 打印结果
# print_analysis_results(analysis_result, N)

# 可视化结果
# visualize_top_nodes(links_coeffs, node_positions, analysis_result, output_dir)
