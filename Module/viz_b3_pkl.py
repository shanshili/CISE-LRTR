# -*- coding: utf-8 -*-
"""独立可视化 ``B3_CORR_INV.pkl`` 无向子图。

兼容两种 pkl：
1. ``subgraph.save_pkl`` 导出的 ``graph / val_matrix / meta`` 格式；
2. 旧脚本使用的顶层 ``V / E`` 格式。
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd

import viz as V


def lonlat_to_local_km(pos):
    """与 run2_pems.py 保持一致：把经纬度转换为以均值为原点的局部公里坐标。"""
    p = np.asarray(pos, dtype=float)
    lon, lat = p[:, 0], p[:, 1]
    lat0 = np.deg2rad(lat.mean())
    x = (lon - lon.mean()) * 111.320 * np.cos(lat0)
    y = (lat - lat.mean()) * 110.574
    return np.stack([x, y], axis=1)


def corr_from_ts(X):
    c = np.corrcoef(np.asarray(X, dtype=float), rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c.astype(float)


class SubgraphWrapper:
    """解析两种 pkl，并统一成 viz.py 需要的接口。"""

    def __init__(self, data):
        meta = data.get("meta", {}) or {}
        raw_vertices = data.get("V", data.get("top_nodes", meta.get("top_nodes", [])))
        pairs = set()

        if "graph" in data:
            graph = np.asarray(data["graph"])
            val = np.asarray(data["val_matrix"])
            if graph.ndim not in (2, 3) or graph.shape[0] != graph.shape[1]:
                raise ValueError(f"graph 应为方形 [N,N] 或 [N,N,L]，实际为 {graph.shape}")
            if val.shape != graph.shape:
                raise ValueError(
                    f"val_matrix 形状 {val.shape} 与 graph 形状 {graph.shape} 不一致"
                )
            for index in np.argwhere(graph == "-->"):
                i, j = int(index[0]), int(index[1])
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
        else:
            for edge in data.get("E", []):
                if len(edge) < 2:
                    raise ValueError(f"无法解析边记录: {edge!r}")
                i, j = int(edge[0]), int(edge[1])
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

        self.E = [(i, j, 0, 1.0) for i, j in sorted(pairs)]
        self.V = [int(v) for v in raw_vertices]
        if not self.V:
            self.V = sorted({v for pair in pairs for v in pair})
        self.name = data.get("name", meta.get("name", "B3_CORR_INV"))
        self.directed = False
        self.weighted = False


def load_b3_subgraph(pkl_path):
    """按 subgraph.py 的真实保存协议加载 B3；同时保留旧格式兼容性。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"pkl 顶层应为 dict，实际为 {type(data).__name__}")

    if ("graph" in data and "val_matrix" in data) or "V" in data or "E" in data:
        return SubgraphWrapper(data)

    raise KeyError(
        "无法识别 pkl：需要 graph/val_matrix/meta（subgraph.save_pkl 格式），"
        "或顶层 V/E（旧格式）"
    )


def load_positions(csv_path, coordinates_are_km=False):
    df = pd.read_csv(csv_path)
    # lower_to_original = {str(c).lower(): c for c in df.columns}
    # if "x" in lower_to_original and "y" in lower_to_original:
    #     columns = [lower_to_original["x"], lower_to_original["y"]]
    # elif "lon" in lower_to_original and "lat" in lower_to_original:
    #     columns = [lower_to_original["lon"], lower_to_original["lat"]]
    # elif "longitude" in lower_to_original and "latitude" in lower_to_original:
    #     columns = [lower_to_original["longitude"], lower_to_original["latitude"]]
    # else:
    #     raise ValueError(
    #         f"坐标文件缺少 x/y、lon/lat 或 longitude/latitude 列；现有列: {list(df.columns)}"
    #     )
    if "node_id" in df.columns:               # 坐标按 node_id 升序，防止行序错位
        pos_df = df.sort_values("node_id")
    pos = pos_df[["x", "y"]].values

    # pos = df[columns].to_numpy(dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] == 0:
        raise ValueError(f"坐标数组形状不合法: {pos.shape}")
    if not np.isfinite(pos).all():
        raise ValueError("坐标文件含 NaN 或无穷值")
    return pos if coordinates_are_km else lonlat_to_local_km(pos)


def load_timeseries(feature_path, num_nodes):
    """读取成 [T, N]；兼容首列为数值时间/索引列以及输入为 [N, T]。"""
    df = pd.read_csv(feature_path)
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if X.size == 0:
        raise ValueError(f"特征文件没有数值列: {feature_path}")

    if X.shape[1] == num_nodes:
        pass
    elif X.shape[1] == num_nodes + 1:
        warnings.warn("特征文件比节点数多一个数值列，按时间/索引列丢弃第一列")
        X = X[:, 1:]
    elif X.shape[0] == num_nodes:
        X = X.T
    elif X.shape[0] == num_nodes + 1:
        warnings.warn("特征文件比节点数多一行，按索引/表头行丢弃第一行后转置")
        X = X[1:, :].T
    else:
        raise ValueError(
            f"无法把特征数组 {X.shape} 对齐到坐标中的 {num_nodes} 个节点；"
            "期望 [T,N] 或 [N,T]"
        )

    if X.shape[0] < 2:
        raise ValueError("计算相关矩阵至少需要两个时间点")
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def validate_subgraph(sub, num_nodes):
    core = [int(v) for v in sub.V]
    pairs = {(min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))) for e in sub.E}
    all_ids = set(core) | {v for pair in pairs for v in pair}

    bad = sorted(v for v in all_ids if v < 0 or v >= num_nodes)
    if bad:
        raise IndexError(
            f"子图节点编号超出坐标范围 [0, {num_nodes - 1}]: {bad[:10]}"
        )
    if not core:
        raise ValueError(
            "pkl 中没有核心节点。若该文件来自 subgraph.save_pkl，请检查 meta.top_nodes 是否存在"
        )
    if not pairs:
        warnings.warn("B3 子图没有边，将只绘制核心节点")
    return len(core), len(pairs)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize B3_CORR_INV pkl")
    parser.add_argument("--pkl-path", required=True, help="B3_CORR_INV.pkl 路径")
    parser.add_argument(
        "--dataset-dir", required=True, help="包含特征 CSV 和坐标 CSV 的目录"
    )
    parser.add_argument(
        "--feature-csv", default="Env_A_timeseries.csv"   # id_standardize_
    )
    parser.add_argument("--pos-csv", default="node_positions.csv")
    parser.add_argument(
        "--corr-adj-npy",
        default=None,
        help="可选的 corr_adjacency_matrix.npy；不提供时由特征 CSV 计算",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="可选校验值；实际节点数从坐标文件推断",
    )
    parser.add_argument(
        "--coordinates-are-km",
        action="store_true",
        help="坐标已经是局部公里坐标，不再执行经纬度转换",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出文件基名；默认在 pkl 同目录生成 *_subgraph",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-size", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    pkl_path = os.path.abspath(os.path.expanduser(args.pkl_path.strip()))
    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir.strip()))
    pos_path = os.path.join(dataset_dir, args.pos_csv)
    feature_path = os.path.join(dataset_dir, args.feature_csv)

    sub = load_b3_subgraph(pkl_path)
    pos = load_positions(pos_path, coordinates_are_km=args.coordinates_are_km)
    num_nodes = pos.shape[0]
    if args.num_nodes is not None and args.num_nodes != num_nodes:
        raise ValueError(
            f"--num-nodes={args.num_nodes}，但坐标文件有 {num_nodes} 行；请修正输入"
        )
    n_core, n_edges = validate_subgraph(sub, num_nodes)

    if args.corr_adj_npy:
        corr_path = os.path.abspath(os.path.expanduser(args.corr_adj_npy.strip()))
        if not os.path.isfile(corr_path):
            raise FileNotFoundError(f"相关矩阵不存在: {corr_path}")
        corr = np.asarray(np.load(corr_path, allow_pickle=False), dtype=float)
    else:
        corr = corr_from_ts(load_timeseries(feature_path, num_nodes))
    if corr.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"相关矩阵形状应为 ({num_nodes}, {num_nodes})，实际为 {corr.shape}"
        )

    save_path = args.output or (os.path.splitext(pkl_path)[0] + "_subgraph")
    formats = tuple("." + fmt for fmt in args.formats)
    print(
        f"[LOAD] {sub.name}: N={num_nodes}, |V|={n_core}, |E|={n_edges}, "
        f"corr={corr.shape}"
    )
    V.plot_undirected_subgraph(
        sub,
        pos,
        save_path,
        corr_adj=corr,
        font_size=args.font_size,
        dpi=args.dpi,
        formats=formats,
    )


if __name__ == "__main__":
    main()
