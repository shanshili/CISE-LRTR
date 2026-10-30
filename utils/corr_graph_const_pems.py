"""
概述
该文件用于构建基于节点位置的空间图（相关图/约束图），提供从节点位置数据到图结构的完整流程。
支持两种建图方式：
1) KNN 图：每个节点连接 k 个最近邻（图稳定、推荐默认）
2) Radius 图：连接距离阈值 r_km 内的邻居，同时可限制每节点最多邻居数 max_neighbors（避免密集区爆边）
   - 可启用 auto_radius：根据第 k_target 近邻距离分布的分位数自动建议 r_km

与原版不同点（关键改进）
- 位置数据为经纬度（x=Longitude, y=Latitude），距离计算使用 Haversine 球面距离（单位 km），避免经纬度欧氏距离失真
- Radius 图支持 max_neighbors_in_radius 限制每点最多连接多少条边
- 输出边属性包含 distance_km 与 weight(=1/(distance_km+eps))

函数说明

1. load_node_positions(csv_file)
功能: 从CSV文件加载节点位置数据
输入列要求:
  - node_id: 节点ID
  - x: Longitude
  - y: Latitude
返回: positions（经纬度数组 [lon, lat]），node_ids（节点ID列表），df（原始DataFrame）

2. suggest_radius_km(positions, k_target=6, quantile=0.9)
功能: 自动建议 radius_km：使用第 k_target 近邻距离的 quantile 分位数（km）
返回: radius_km, dk_km（每个点第k近邻距离km）

3. create_knn_graph_from_positions(positions, k)
功能: 使用 Haversine 距离构建 KNN 图
返回: networkx.Graph

4. create_radius_graph_from_positions(positions, r_km, max_neighbors=None)
功能: 使用 Haversine 距离构建 radius 图（半径阈值 + 可选最大邻居数）
返回: networkx.Graph

5. draw_and_save_graph(G, positions, node_ids, filename, figsize=None, draw_labels=False)
功能: 根据节点位置绘制图并保存 SVG，保持真实坐标比例（经纬度坐标）
参数:
  - draw_labels: 是否绘制节点ID（N=100可开，但会乱）

6. process_graph_from_positions(...)
功能: 主处理：加载数据 → 建图 → 可视化 → 保存邻接矩阵/边列表/节点位置/meta信息

主要输出文件
- graph.svg: 图可视化结果（knn_graph.svg / radius_graph.svg）
- corr_adjacency_matrix.npy: 邻接矩阵
- edges_list.npy: 边列表（含weight与distance_km）
- node_positions.npy: 节点位置信息（node_id, lon, lat）
- graph_meta.txt: 记录参数与图统计信息

"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# =========================
# 1) 数据加载
# =========================
def load_node_positions(csv_file):
    """
    从CSV文件加载节点位置数据

    Parameters:
    csv_file: str, CSV文件路径，包含 node_id, x(经度), y(纬度)

    Returns:
    positions: numpy.ndarray, 形状为(n_nodes, 2), [lon, lat]
    node_ids: numpy.ndarray, 节点ID
    df: DataFrame, 原始数据（便于携带额外字段）
    """
    df = pd.read_csv(csv_file)

    # 基本检查
    for col in ["node_id", "x", "y"]:
        if col not in df.columns:
            raise ValueError(f"位置文件缺少列: {col} (需要 node_id, x, y)")

    # 转数值，过滤异常
    df["x"] = pd.to_numeric(df["x"], errors="coerce")  # lon
    df["y"] = pd.to_numeric(df["y"], errors="coerce")  # lat
    df = df.dropna(subset=["x", "y", "node_id"]).copy()

    positions = df[["x", "y"]].values  # [lon, lat]
    node_ids = df["node_id"].values

    print(f"加载了 {len(positions)} 个节点的位置数据")
    return positions, node_ids, df


# =========================
# 2) 半径自动建议
# =========================
def suggest_radius_km(positions, k_target=6, quantile=0.9):
    """
    用第 k_target 近邻距离的分位数作为建议 radius_km

    直觉：取到该 radius_km 时，大多数点（quantile）至少能在该半径内找到 k_target 个邻居（大致）。
    """
    # BallTree(haversine) 需要 [lat, lon] 弧度
    lon = positions[:, 0]
    lat = positions[:, 1]
    X = np.deg2rad(np.c_[lat, lon])

    tree = BallTree(X, metric="haversine")
    dist_rad, _ = tree.query(X, k=k_target + 1)  # 含自身
    dk_km = dist_rad[:, k_target] * EARTH_KM
    r_km = float(np.quantile(dk_km, quantile))
    return r_km, dk_km


# =========================
# 3) KNN 图（Haversine）
# =========================
def create_knn_graph_from_positions(positions, k):
    """
    positions: (n,2), [lon, lat]
    构建 KNN 无向图，边属性包含:
      - distance_km
      - weight = 1/(distance_km+eps)
    """
    G = nx.Graph()
    n_nodes = positions.shape[0]
    for i in range(n_nodes):
        G.add_node(i)

    lon = positions[:, 0]
    lat = positions[:, 1]
    X = np.deg2rad(np.c_[lat, lon])

    tree = BallTree(X, metric="haversine")
    dist_rad, ind = tree.query(X, k=k + 1)  # 含自身
    dist_km = dist_rad * EARTH_KM

    for i in range(n_nodes):
        for j in range(1, k + 1):
            nb = int(ind[i, j])
            d = float(dist_km[i, j])
            w = 1.0 / (d + 1e-8)
            if G.has_edge(i, nb):
                # 保留更短距离
                if d < G[i][nb]["distance_km"]:
                    G[i][nb]["distance_km"] = d
                    G[i][nb]["weight"] = w
            else:
                G.add_edge(i, nb, weight=w, distance_km=d)

    return G


# =========================
# 4) Radius 图（距离阈值 + 可选最大邻居数）
# =========================
def create_radius_graph_from_positions(positions, r_km, max_neighbors=None):
    """
    positions: (n,2) [lon, lat]
    在距离阈值 r_km 内连边。
    max_neighbors: 每个节点最多连接多少个最近邻（防止密集区爆边）
    """
    G = nx.Graph()
    n_nodes = positions.shape[0]
    for i in range(n_nodes):
        G.add_node(i)

    lon = positions[:, 0]
    lat = positions[:, 1]
    X = np.deg2rad(np.c_[lat, lon])

    tree = BallTree(X, metric="haversine")
    r_rad = r_km / EARTH_KM

    ind_list, dist_list = tree.query_radius(X, r=r_rad, return_distance=True, sort_results=True)

    for i in range(n_nodes):
        inds = ind_list[i]
        dists = dist_list[i] * EARTH_KM  # km

        # 去掉自己
        mask = inds != i
        inds = inds[mask]
        dists = dists[mask]

        # 限制每点最多邻居数（只取最近的 max_neighbors 个）
        if max_neighbors is not None and len(inds) > max_neighbors:
            inds = inds[:max_neighbors]
            dists = dists[:max_neighbors]

        for nb, d in zip(inds, dists):
            nb = int(nb)
            d = float(d)
            w = 1.0 / (d + 1e-8)
            if G.has_edge(i, nb):
                if d < G[i][nb]["distance_km"]:
                    G[i][nb]["distance_km"] = d
                    G[i][nb]["weight"] = w
            else:
                G.add_edge(i, nb, weight=w, distance_km=d)

    return G


# =========================
# 5) 绘图
# =========================
def draw_and_save_graph(G, positions, node_ids, filename, figsize=None, draw_labels=False):
    """
    positions: [lon, lat]
    """
    x = positions[:, 0]
    y = positions[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if figsize is None:
        base_size = 8
        if x_range > y_range:
            figsize = (base_size, base_size * y_range / max(x_range, 1e-9))
        else:
            figsize = (base_size * x_range / max(y_range, 1e-9), base_size)

    plt.figure(figsize=figsize)
    pos = {i: (x[i], y[i]) for i in range(len(positions))}

    edges = list(G.edges())
    dists = [G[u][v].get("distance_km", 1.0) for u, v in edges]
    # 距离越近线越粗（仅用于视觉）
    widths = [max(0.15, 2.5 / (d + 1e-6)) for d in dists]

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=50, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6,edge_color="gray")

    if draw_labels:
        nx.draw_networkx_labels(
            G, pos,
            labels={i: str(node_ids[i]) for i in range(len(node_ids))},
            font_size=12, font_family='Times New Roman',
        )

    # plt.title("Spatial Graph (Haversine distance on Lat/Lon)", size=13)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Longitude (x)")
    plt.ylabel("Latitude (y)")

    plt.savefig(filename, format="svg", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Graph saved: {filename}")


# =========================
# 6) 主流程：加载→建图→保存
# =========================
def process_graph_from_positions(
    positions_file,
    output_folder,
    mode="knn",
    k=10,
    radius_km=None,
    auto_radius=False,
    k_target_for_radius=6,
    radius_quantile=0.9,
    max_neighbors_in_radius=15,
    draw_labels=False
):
    """
    mode:
      - "knn": 使用 KNN（k）
      - "radius": 使用距离阈值（radius_km） + 最大邻居数 max_neighbors_in_radius
                 可以 auto_radius=True 自动建议 radius_km

    返回:
      G, adjacency_matrix
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"加载节点位置数据: {positions_file}")
    positions, node_ids, df = load_node_positions(positions_file)
    n_nodes = positions.shape[0]

    print(f"节点数量: {n_nodes}")
    print(
        f"范围: lon=[{positions[:,0].min():.4f}, {positions[:,0].max():.4f}], "
        f"lat=[{positions[:,1].min():.4f}, {positions[:,1].max():.4f}]"
    )

    # 建图
    used_radius_km = None
    if mode == "knn":
        print(f"创建 KNN 图: k={k}（Haversine km）...")
        G = create_knn_graph_from_positions(positions, k=k)

    elif mode == "radius":
        if auto_radius:
            used_radius_km, dk_km = suggest_radius_km(
                positions,
                k_target=k_target_for_radius,
                quantile=radius_quantile
            )
            print(f"[auto_radius] k_target={k_target_for_radius}, q={radius_quantile} -> radius_km={used_radius_km:.3f}")
            print(f"[auto_radius] dk_km stats: min={dk_km.min():.3f}, p50={np.median(dk_km):.3f}, p90={np.quantile(dk_km,0.9):.3f}, max={dk_km.max():.3f}")

        if radius_km is None and used_radius_km is None:
            raise ValueError("mode='radius' 需要 radius_km 或 auto_radius=True")
        if radius_km is None:
            radius_km = used_radius_km

        print(f"创建 Radius 图: r={radius_km:.3f} km, max_neighbors={max_neighbors_in_radius} ...")
        G = create_radius_graph_from_positions(
            positions,
            r_km=radius_km,
            max_neighbors=max_neighbors_in_radius
        )
        used_radius_km = radius_km

    else:
        raise ValueError("mode 必须是 'knn' 或 'radius'")

    print(f"图信息: 节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")

    # 保存 SVG
    svg_filename = os.path.join(output_folder, f"{mode}_graph.svg")
    draw_and_save_graph(G, positions, node_ids, svg_filename, draw_labels=draw_labels)

    # 邻接矩阵（无权）/（加权）你可选，这里默认保存加权矩阵（nx 会用 weight 属性）
    adjacency_matrix = nx.to_numpy_array(G, weight="weight")
    adj_matrix_filename = os.path.join(output_folder, "corr_adjacency_matrix.npy")
    np.save(adj_matrix_filename, adjacency_matrix)
    print(f"邻接矩阵已保存为: {adj_matrix_filename}")

    # 边列表
    edges_list = list(G.edges(data=True))
    edges_filename = os.path.join(output_folder, "edges_list.npy")
    np.save(edges_filename, edges_list, allow_pickle=True)
    print(f"边列表已保存为: {edges_filename}")

    # 节点位置（node_id, lon, lat）
    positions_info = np.column_stack([node_ids, positions])
    positions_filename = os.path.join(output_folder, "node_positions.npy")
    np.save(positions_filename, positions_info)
    print(f"节点位置信息已保存为: {positions_filename}")

    # meta 信息
    meta_file = os.path.join(output_folder, "graph_meta.txt")
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write(f"mode={mode}\n")
        f.write(f"k={k}\n")
        f.write(f"radius_km={used_radius_km}\n")
        f.write(f"auto_radius={auto_radius}\n")
        f.write(f"k_target_for_radius={k_target_for_radius}\n")
        f.write(f"radius_quantile={radius_quantile}\n")
        f.write(f"max_neighbors_in_radius={max_neighbors_in_radius}\n")
        f.write(f"n_nodes={G.number_of_nodes()}\n")
        f.write(f"n_edges={G.number_of_edges()}\n")
    print(f"meta 已保存: {meta_file}")

    return G, adjacency_matrix


# =========================
# 7) main：按需配置
# =========================
def main():
    # 路径设置（按你的工程结构改）
    project_root = ".."
    data_folder = os.path.join(project_root, "dataset", "pems_spatial_kmeans_topK")  # 你的 node_positions.csv 所在处
    output_folder = os.path.join(project_root, "Correlation_Graph/PEMS_N100_1h_3")

    positions_file = os.path.join(data_folder, "node_positions.csv")

    # 选择模式：'knn' 或 'radius'
    mode = "radius"  # 推荐先用 radius(auto) 看看边数，再和 knn 比较

    # KNN 参数
    k = 10

    # Radius 参数
    auto_radius = False           # 自动建议半径
    radius_km = 10             # 如果 auto_radius=False，可手动填比如 5.0
    k_target_for_radius = 10      # 希望大多数点在半径内能找到的邻居数（大致）
    radius_quantile = 0.9        # 用第k近邻距离的 90% 分位作为半径（更连通）
    max_neighbors_in_radius = 15 # 半径内邻居过多时，只保留最近的 15 个（防密区爆边）

    draw_labels = False          # N=100 开 True 也行，但容易乱

    if os.path.exists(positions_file):
        G, adj = process_graph_from_positions(
            positions_file=positions_file,
            output_folder=output_folder,
            mode=mode,
            k=k,
            radius_km=radius_km,
            auto_radius=auto_radius,
            k_target_for_radius=k_target_for_radius,
            radius_quantile=radius_quantile,
            max_neighbors_in_radius=max_neighbors_in_radius,
            draw_labels=draw_labels
        )
        print("基于节点位置的空间图构建完成!")
    else:
        print("找不到 positions_file：", positions_file)
        for file in ["positions.csv", "node_positions.csv", "coordinates.csv"]:
            test_path = os.path.join(data_folder, file)
            print(f"  - {test_path}: {'存在' if os.path.exists(test_path) else '不存在'}")


if __name__ == "__main__":
    main()
