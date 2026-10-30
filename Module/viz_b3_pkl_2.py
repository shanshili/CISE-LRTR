# -*- coding: utf-8 -*-
"""独立可视化 ``B3_CORR_INV.pkl`` 无向子图。

兼容两种 pkl：
1. ``subgraph.save_pkl`` 导出的 ``graph / val_matrix / meta`` 格式；
2. 旧脚本使用的顶层 ``V / E`` 格式。

坐标支持：
- 经/纬度列：lon/lat、longitude/latitude，或用 --x-col/--y-col 指定；
- 普通平面坐标：x/y，原值用于绘图；
- --coord-type auto/lonlat/xy 可自动判断或强制指定。
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


def _column_map(df):
    """大小写不敏感地映射列名，同时忽略列名两侧空格。"""
    return {str(c).strip().lower(): c for c in df.columns}


def _find_column(df, requested):
    if requested is None:
        return None
    mapping = _column_map(df)
    key = str(requested).strip().lower()
    if key not in mapping:
        raise ValueError(f"坐标文件中找不到列 {requested!r}；现有列: {list(df.columns)}")
    return mapping[key]


def _pick_coordinate_columns(df, coord_type, x_col=None, y_col=None):
    """返回 (横坐标列, 纵坐标列, 列名提示的类型)。"""
    if (x_col is None) != (y_col is None):
        raise ValueError("--x-col 和 --y-col 必须同时提供")
    if x_col is not None:
        cx, cy = _find_column(df, x_col), _find_column(df, y_col)
        x_key, y_key = str(cx).strip().lower(), str(cy).strip().lower()
        hint = "lonlat" if (
            x_key in {"lon", "longitude", "lng"} and y_key in {"lat", "latitude"}
        ) else ("xy" if (x_key, y_key) == ("x", "y") else None)
        return cx, cy, hint

    columns = _column_map(df)
    lonlat_pairs = (("lon", "lat"), ("longitude", "latitude"), ("lng", "lat"))
    xy_pair = ("x", "y")

    if coord_type in ("auto", "lonlat"):
        for lon_name, lat_name in lonlat_pairs:
            if lon_name in columns and lat_name in columns:
                return columns[lon_name], columns[lat_name], "lonlat"
    if coord_type in ("auto", "xy") and all(name in columns for name in xy_pair):
        return columns["x"], columns["y"], "xy"
    # 很多旧数据虽然把经纬度命名为 x/y，但显式 --coord-type lonlat 时仍应支持。
    if coord_type == "lonlat" and all(name in columns for name in xy_pair):
        return columns["x"], columns["y"], None

    expected = "lon/lat、longitude/latitude 或 x/y"
    raise ValueError(
        f"无法按 coord-type={coord_type!r} 选择坐标列；需要 {expected}；"
        f"现有列: {list(df.columns)}。也可显式传 --x-col 和 --y-col"
    )


def _infer_coord_type(pos, name_hint):
    """auto 模式的保守推断；模糊的 x/y 默认按平面坐标处理。"""
    if name_hint == "lonlat":
        return "lonlat", "列名明确为经纬度"
    if name_hint == "xy":
        x, y = pos[:, 0], pos[:, 1]
        within_geo_bounds = (
            np.all((-180.0 <= x) & (x <= 180.0))
            and np.all((-90.0 <= y) & (y <= 90.0))
        )
        # x/y 无法仅凭小数值可靠区分；|x|>90 且 y 合法是较明确的经纬度信号。
        if within_geo_bounds and np.max(np.abs(x)) > 90.0:
            return "lonlat", "x/y 数值范围很像经纬度"
        return "xy", "x/y 列按平面坐标处理"
    return "xy", "自定义列在 auto 模式下默认按平面坐标处理"


def load_positions(
    csv_path,
    coord_type="auto",
    x_col=None,
    y_col=None,
    node_id_col=None,
    lonlat_output="km",
    xy_unit=None,
):
    """读取坐标并返回 ``(positions, coordinate_info)``。"""
    df = pd.read_csv(csv_path)
    pos_df = df

    # 默认识别 node_id；也允许显式指定其他 ID 列。按 ID 排序以对齐图节点下标。
    resolved_id_col = _find_column(df, node_id_col) if node_id_col else None
    if resolved_id_col is None and "node_id" in _column_map(df):
        resolved_id_col = _column_map(df)["node_id"]
    if resolved_id_col is not None:
        if df[resolved_id_col].duplicated().any():
            raise ValueError(f"节点 ID 列 {resolved_id_col!r} 中存在重复值")
        pos_df = df.sort_values(resolved_id_col, kind="stable")

    cx, cy, name_hint = _pick_coordinate_columns(pos_df, coord_type, x_col, y_col)
    pos = pos_df[[cx, cy]].to_numpy(dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] == 0:
        raise ValueError(f"坐标数组形状不合法: {pos.shape}")
    if not np.isfinite(pos).all():
        raise ValueError("坐标文件含 NaN 或无穷值")

    resolved_type = coord_type
    reason = "命令行显式指定"
    if coord_type == "auto":
        resolved_type, reason = _infer_coord_type(pos, name_hint)

    if resolved_type == "lonlat":
        lon, lat = pos[:, 0], pos[:, 1]
        if not (np.all((-180.0 <= lon) & (lon <= 180.0)) and
                np.all((-90.0 <= lat) & (lat <= 90.0))):
            raise ValueError("经纬度越界：经度应在 [-180,180]，纬度应在 [-90,90]")
        if lonlat_output == "km":
            pos = lonlat_to_local_km(pos)
            x_label, y_label = "Local X (km)", "Local Y (km)"
            transform = "lon/lat -> local km"
        else:
            x_label, y_label = "Longitude (°)", "Latitude (°)"
            transform = "raw lon/lat degrees"
    else:
        unit = f" ({xy_unit})" if xy_unit else ""
        x_label, y_label = f"{cx}{unit}", f"{cy}{unit}"
        transform = "raw Cartesian x/y"

    info = {
        "type": resolved_type,
        "reason": reason,
        "columns": (str(cx), str(cy)),
        "node_id_column": None if resolved_id_col is None else str(resolved_id_col),
        "transform": transform,
        "x_label": x_label,
        "y_label": y_label,
    }
    return pos, info


def set_viz_axis_labels(x_label, y_label):
    """让现有 viz.py 的固定轴标题适配本次坐标类型。"""
    if not hasattr(V, "_frame"):
        return None
    original_frame = V._frame

    def coordinate_frame(ax):
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.margins(0.05)

    V._frame = coordinate_frame
    return original_frame


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
        "--coord-type",
        choices=("auto", "lonlat", "xy"),
        default="auto",
        help="坐标类型：自动判断、经纬度或普通平面 x/y（默认 auto）",
    )
    parser.add_argument("--x-col", default=None, help="横坐标列名；需与 --y-col 同时使用")
    parser.add_argument("--y-col", default=None, help="纵坐标列名；需与 --x-col 同时使用")
    parser.add_argument(
        "--node-id-col",
        default=None,
        help="可选节点 ID 列名；默认发现 node_id 时自动按其排序",
    )
    parser.add_argument(
        "--lonlat-output",
        choices=("km", "degrees"),
        default="km",
        help="经纬度按局部公里坐标或原始角度绘制（默认 km）",
    )
    parser.add_argument(
        "--xy-unit", default=None, help="平面坐标单位，仅用于轴标题，例如 m 或 km"
    )
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
        help="兼容旧参数；等价于 --coord-type xy --xy-unit km",
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
    coord_type = args.coord_type
    xy_unit = args.xy_unit
    if args.coordinates_are_km:
        if coord_type == "lonlat":
            raise ValueError("--coordinates-are-km 与 --coord-type lonlat 冲突")
        coord_type = "xy"
        xy_unit = xy_unit or "km"
        warnings.warn("--coordinates-are-km 已兼容处理；新命令建议使用 --coord-type xy")
    pos, coord_info = load_positions(
        pos_path,
        coord_type=coord_type,
        x_col=args.x_col,
        y_col=args.y_col,
        node_id_col=args.node_id_col,
        lonlat_output=args.lonlat_output,
        xy_unit=xy_unit,
    )
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
    print(
        f"[COORD] columns={coord_info['columns']}, type={coord_info['type']} "
        f"({coord_info['reason']}), transform={coord_info['transform']}, "
        f"node_id={coord_info['node_id_column']}"
    )
    original_frame = set_viz_axis_labels(coord_info["x_label"], coord_info["y_label"])
    try:
        V.plot_undirected_subgraph(
            sub,
            pos,
            save_path,
            corr_adj=corr,
            font_size=args.font_size,
            dpi=args.dpi,
            formats=formats,
        )
    finally:
        if original_frame is not None:
            V._frame = original_frame


if __name__ == "__main__":
    main()
