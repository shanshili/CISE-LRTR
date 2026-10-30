"""
概述
该文件用于构建基于节点位置的K近邻图（相关图），提供从节点位置数据到图结构的完整处理流程。
包括数据加载、图构建、可视化和结果保存功能。

函数说明

1. load_node_positions(csv_file)
功能: 从CSV文件加载节点位置数据
参数:
  - csv_file: CSV文件路径，包含node_id, x, y列
返回: positions（节点坐标数组），node_ids（节点ID列表）

2. create_knn_graph_from_positions(positions, k)
功能: 使用节点位置数据创建K近邻图
参数:
  - positions: 节点坐标数组，形状为(n_nodes, 2)
  - k: 近邻数量
返回: 构建的networkx.Graph对象

3. draw_and_save_knn_graph(G, positions, node_ids, filename, figsize=None)
功能: 根据节点位置绘制K近邻图并保存为SVG格式，保持真实坐标比例
参数:
  - G: networkx.Graph图对象
  - positions: 节点坐标数组
  - node_ids: 节点ID列表
  - filename: 保存的文件名
  - figsize: 图形大小（可选）
返回: 无，但会保存SVG图像文件

4. process_knn_graph_from_positions(positions_file, output_folder, k=5)
功能: 处理节点位置数据，使用K近邻算法生成图并保存相关文件
参数:
  - positions_file: 节点位置CSV文件路径
  - output_folder: 输出文件夹路径
  - k: 近邻数量
返回: G（图对象），adjacency_matrix（邻接矩阵）

5. main()
功能: 主处理函数，设置路径并调用process_knn_graph_from_positions

主要输出文件
1. knn_graph.svg: K近邻图的可视化结果
2. corr_adjacency_matrix.npy: 邻接矩阵文件
3. edges_list.npy: 边列表文件
4. node_positions.npy: 节点位置信息文件

K近邻算法特点
- 基于欧氏距离寻找每个节点的k个最近邻居
- 构建无向图，边权重为距离的倒数
- 保持空间局部性，适合地理空间数据
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})
def load_node_positions(csv_file):
    """
    从CSV文件加载节点位置数据

    Parameters:
    csv_file: str, CSV文件路径，包含node_id, x, y列

    Returns:
    positions: numpy.ndarray, 节点坐标数组，形状为(n_nodes, 2)
    node_ids: list, 节点ID列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 提取坐标数据
    positions = df[['x', 'y']].values
    node_ids = df['node_id'].values

    print(f"加载了 {len(positions)} 个节点的位置数据")
    return positions, node_ids


def create_knn_graph_from_positions(positions, k):
    """
    使用节点位置数据创建K近邻图

    Parameters:
    positions: array-like, 形状为 (n_nodes, 2) 的节点坐标数据
    k: int, 近邻数量

    Returns:
    G: networkx.Graph, K近邻图
    """
    G = nx.Graph()
    n_nodes = positions.shape[0]

    # 添加节点
    for i in range(n_nodes):
        G.add_node(i)

    # 使用sklearn的NearestNeighbors找到K近邻
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nbrs.fit(positions)

    # 获取每个节点的K近邻
    distances, indices = nbrs.kneighbors(positions)

    # 添加边
    for i in range(n_nodes):
        # indices[i, 0] 是节点本身，所以从 indices[i, 1:] 开始
        for j in range(1, k + 1):
            neighbor_idx = indices[i, j]
            # 使用距离的倒数作为边权重（可选）
            weight = 1.0 / (distances[i, j] + 1e-8)  # 避免除零
            G.add_edge(i, neighbor_idx, weight=weight, distance=distances[i, j])

    return G


def draw_and_save_knn_graph(G, positions, node_ids, filename, figsize=None):
    """
    根据节点位置绘制K近邻图并保存为SVG格式，保持真实坐标比例

    Parameters:
    G: networkx.Graph, 图对象
    positions: numpy.ndarray, 节点坐标
    node_ids: list, 节点ID列表
    filename: str, 保存的文件名
    figsize: tuple, 图形大小 (可选)
    """
    # 计算坐标范围
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    # 计算坐标范围的宽高比
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 根据实际坐标比例设置图形大小，保持真实比例
    if figsize is None:
        base_size = 8
        if x_range > y_range:
            figsize = (base_size, base_size * y_range / x_range)
        else:
            figsize = (base_size * x_range / y_range, base_size)

    plt.figure(figsize=figsize)

    # 使用实际节点位置作为布局
    pos = {i: positions[i] for i in range(len(positions))}

    # 获取边的权重用于可视化
    edges = G.edges()
    weights = [G[u][v]['weight'] * 10 for u, v in edges]  # 调整权重放大倍数

    # 绘制图
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=50, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.6, edge_color='gray')
    # nx.draw_networkx_labels(G, pos, labels={i: str(node_ids[i]) for i in range(len(node_ids))},
    #                         font_size=12, font_family='Times New Roman')

    #plt.title("K-Nearest Neighbors Graph based on Node Positions", size=15)

    # 设置坐标轴比例为equal，确保真实比例
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # 添加坐标轴标签
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 保存为SVG格式
    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"K近邻图已保存为: {filename}")


def process_knn_graph_from_positions(positions_file, output_folder, k=5):
    """
    处理节点位置数据，使用K近邻算法生成图并保存为SVG

    Parameters:
    positions_file: str, 节点位置CSV文件路径
    output_folder: str, 输出文件夹路径
    k: int, 近邻数量
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 加载节点位置数据
    print(f"加载节点位置数据: {positions_file}")
    positions, node_ids = load_node_positions(positions_file)
    n_nodes = positions.shape[0]

    print(f"节点数量: {n_nodes}")
    print(f"位置数据范围: x=[{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}], "
          f"y=[{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")

    # 创建K近邻图
    print(f"创建K近邻图 (K={k})...")
    G = create_knn_graph_from_positions(positions, k)

    print(f"图信息: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")

    # 绘制并保存图
    svg_filename = os.path.join(output_folder, "knn_graph.svg")
    draw_and_save_knn_graph(G, positions, node_ids, svg_filename)

    # 保存图的邻接矩阵
    adjacency_matrix = nx.to_numpy_array(G)
    adj_matrix_filename = os.path.join(output_folder, "corr_adjacency_matrix.npy")
    np.save(adj_matrix_filename, adjacency_matrix)
    print(f"邻接矩阵已保存为: {adj_matrix_filename}")

    # 保存图结构为边列表
    edges_list = list(G.edges(data=True))
    edges_filename = os.path.join(output_folder, "edges_list.npy")
    np.save(edges_filename, edges_list)
    print(f"边列表已保存为: {edges_filename}")

    # 保存节点位置信息
    positions_info = np.column_stack([node_ids, positions])
    positions_filename = os.path.join(output_folder, "node_positions.npy")
    np.save(positions_filename, positions_info)
    print(f"节点位置信息已保存为: {positions_filename}")

    return G, adjacency_matrix


# 主处理函数
def main():
    # 设置路径
    project_root = ".."  # 工程文件夹
    data_folder = os.path.join(project_root, "dataset", "simulation_hubs_core/N1000_T10000_Hubs80")
    output_folder = os.path.join(project_root, "Correlation_Graph/simulation_hubs_core/N1000")

    # 节点位置文件路径
    positions_file = os.path.join(data_folder, "node_positions.csv")

    # 参数设置
    k = 10  # K近邻数量

    # 处理数据
    if os.path.exists(positions_file):
        G, adj_matrix = process_knn_graph_from_positions(
            positions_file,
            output_folder,
            k=k
        )
        print("基于节点位置的K近邻图构建完成!")
    else:
        for file in ["positions.csv", "node_positions.csv", "coordinates.csv"]:
            test_path = os.path.join(data_folder, file)
            print(f"  - {test_path}: {'存在' if os.path.exists(test_path) else '不存在'}")


if __name__ == "__main__":
    main()
