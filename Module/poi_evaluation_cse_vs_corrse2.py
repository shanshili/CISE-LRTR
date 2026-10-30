import os
import json
import time
import pickle
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
# Paper-friendly naming
# =========================
METHOD_CSE = "CSE"       # Causal Subgraph Extraction
METHOD_CORR = "CorrSE"   # Correlation-based Subgraph Extraction


# =========================
# Hard-coded inputs (edit here)
# =========================
PKL_CSE = r"..\mask_subgraph_visualization\results_pems_20260116_172104_3\threshold_0.5\subgraph_threshold_0.5.pkl"
PKL_CORR = r".\outputs_importance_PEMS\subgraphs_corr\corr_knn_top30_20260115_235352\top30_knn10_union_lag0_tau5\subgraph_top30_knn10_union_lag0_tau5.pkl"

POS_CSV = r"..\dataset\pems_spatial_kmeans_topK\id_node_positions.csv"

OUT_DIR = r".\poi_eval_CSE_vs_CorrSE"

# Main radius + sensitivity analysis radius (optional)
RADII_METERS = [500, 300]

# Overpass
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
OVERPASS_SLEEP_S = 1.0
OVERPASS_TIMEOUT_S = 120


# =========================
# IO: positions
# =========================
import re

def load_positions_csv(pos_csv: str) -> Dict[int, Tuple[float, float]]:
    """
    Load node positions from CSV.

    Supported node_id formats:
      - int: 0, 1, 2, ...
      - str: 'node_0', 'node_1', ...

    Supported coordinate columns:
      - (node_id, x, y)          # PEMS common: x=lon, y=lat
      - (node_id, lon, lat)
      - (node_id, longitude, latitude)
    """
    df = pd.read_csv(pos_csv)

    candidates = [
        ("node_id", "x", "y"),          # <-- your case
        ("id", "x", "y"),
        ("node_id", "lon", "lat"),
        ("id", "lon", "lat"),
        ("node_id", "longitude", "latitude"),
        ("id", "longitude", "latitude"),
    ]

    for a, b, c in candidates:
        if a in df.columns and b in df.columns and c in df.columns:
            id_col, lon_col, lat_col = a, b, c
            print(f"[INFO] Using columns: {id_col}, {lon_col}, {lat_col}")
            break
    else:
        raise ValueError(
            f"Unrecognized columns in {pos_csv}. "
            f"Got: {df.columns.tolist()}"
        )

    pos = {}

    for _, row in df.iterrows():
        raw_id = row[id_col]

        # --- 核心修复逻辑 ---
        if isinstance(raw_id, str):
            # 支持 'node_12' / 'sensor_7' / '12'
            m = re.search(r"\d+", raw_id)
            if m is None:
                raise ValueError(f"Cannot parse node id: {raw_id}")
            nid = int(m.group())
        else:
            nid = int(raw_id)

        lon = float(row[lon_col])
        lat = float(row[lat_col])
        pos[nid] = (lon, lat)

    return pos

# =========================
# IO: subgraph pkls (two formats)
# =========================
def load_pkl_dict(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"PKL is not a dict: {pkl_path}")
    return data


def extract_selected_nodes_from_graph(graph: np.ndarray) -> Set[int]:
    """
    Selected nodes = endpoints of any '-->' edge across all lags.
    graph shape: (N,N,L) or (N,N)
    """
    g = np.array(graph)
    if g.ndim == 3:
        mask = (g == "-->").any(axis=2)
    elif g.ndim == 2:
        mask = (g == "-->")
    else:
        raise ValueError(f"Unexpected graph ndim: {g.ndim}")

    src, dst = np.where(mask)
    return set(int(i) for i in src.tolist()) | set(int(j) for j in dst.tolist())


def extract_selected_nodes_cse(data: Dict[str, Any]) -> Set[int]:
    """
    CSE pkl format (new): prefer meta["top_nodes"] if exists.
    Fallback: endpoints of any '-->' edge in data["graph"].
    """
    meta = data.get("meta", None)
    if isinstance(meta, dict) and "top_nodes" in meta and meta["top_nodes"] is not None:
        return set(int(x) for x in meta["top_nodes"])

    # fallback: old format
    if "graph" not in data:
        raise ValueError("CSE pkl missing both meta.top_nodes and key 'graph'")
    return extract_selected_nodes_from_graph(np.array(data["graph"]))


def extract_selected_nodes_corrse(data: Dict[str, Any]) -> Set[int]:
    # CorrSE pkl format: prefer meta["top_nodes"]
    meta = data.get("meta", None)
    if isinstance(meta, dict) and "top_nodes" in meta:
        return set(int(x) for x in meta["top_nodes"])

    # fallback: if there is a graph
    if "graph" in data:
        return extract_selected_nodes_from_graph(np.array(data["graph"]))

    raise ValueError("CorrSE pkl has neither meta.top_nodes nor graph.")


# =========================
# Overpass POI query + cache
# =========================
def overpass_query(lat: float, lon: float, radius_m: int) -> Dict[str, Any]:
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT_S}];
    (
      node(around:{radius_m},{lat},{lon})["amenity"~"school|university|hospital|clinic"];
      way(around:{radius_m},{lat},{lon})["amenity"~"school|university|hospital|clinic"];
      relation(around:{radius_m},{lat},{lon})["amenity"~"school|university|hospital|clinic"];

      node(around:{radius_m},{lat},{lon})["tourism"~"attraction|museum|viewpoint"];
      way(around:{radius_m},{lat},{lon})["tourism"~"attraction|museum|viewpoint"];
      relation(around:{radius_m},{lat},{lon})["tourism"~"attraction|museum|viewpoint"];

      node(around:{radius_m},{lat},{lon})["leisure"="park"];
      way(around:{radius_m},{lat},{lon})["leisure"="park"];
      relation(around:{radius_m},{lat},{lon})["leisure"="park"];

      node(around:{radius_m},{lat},{lon})["highway"="bus_stop"];
      node(around:{radius_m},{lat},{lon})["highway"="traffic_signals"];
    );
    out center tags;
    """

    last_err = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data=query.encode("utf-8"), timeout=OVERPASS_TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"All Overpass endpoints failed. Last error: {repr(last_err)}")


def parse_pois(overpass_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    pois = []
    for el in overpass_json.get("elements", []):
        tags = el.get("tags", {})

        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue

        cat = None
        if "amenity" in tags:
            cat = f"amenity:{tags['amenity']}"
        elif "tourism" in tags:
            cat = f"tourism:{tags['tourism']}"
        elif "leisure" in tags:
            cat = f"leisure:{tags['leisure']}"
        elif tags.get("highway") in ("bus_stop", "traffic_signals"):
            cat = f"highway:{tags['highway']}"

        pois.append({
            "name": tags.get("name", None),
            "category": cat,
            "lat": float(lat),
            "lon": float(lon),
        })
    return pois


# =========================
# Distance: haversine
# =========================
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def min_dist_to_category(node_lat, node_lon, pois: List[Dict[str, Any]], categories: Set[str]) -> float:
    ds = []
    for p in pois:
        if p.get("category") in categories:
            ds.append(haversine_m(node_lat, node_lon, p["lat"], p["lon"]))
    return float(np.min(ds)) if ds else np.nan


# =========================
# Node-level metrics
# =========================
def count_categories(pois: List[Dict[str, Any]]) -> Dict[str, int]:
    def count_if(catset: Set[str]) -> int:
        return sum(1 for p in pois if p.get("category") in catset)

    school_cats = {"amenity:school", "amenity:university"}
    hosp_cats = {"amenity:hospital", "amenity:clinic"}
    park_cats = {"leisure:park"}
    attract_cats = {"tourism:attraction", "tourism:museum", "tourism:viewpoint"}
    bus_cats = {"highway:bus_stop"}
    sig_cats = {"highway:traffic_signals"}

    out = {
        "poi_school": count_if(school_cats),
        "poi_hospital": count_if(hosp_cats),
        "poi_park": count_if(park_cats),
        "poi_attraction": count_if(attract_cats),
        "poi_bus_stop": count_if(bus_cats),
        "poi_traffic_signals": count_if(sig_cats),
    }
    out["poi_total"] = int(sum(out.values()))
    return out


def clip_dist(d: float, radius_m: int) -> float:
    """
    距离截断：把 NaN / >radius 的距离视作 radius（即“最差可达性”）。
    这样可避免 NaN 让统计/画图变形。
    """
    if pd.isna(d) or d < 0:
        return float(radius_m)
    return float(min(d, radius_m))


def shannon_entropy_from_counts(counts: Dict[str, int], keys: List[str]) -> float:
    """
    功能多样性：Shannon entropy（只对给定 keys 计算）。
    poi_total 很接近时，entropy/组成往往更能拉开差异。
    """
    vals = np.array([max(0, int(counts.get(k, 0))) for k in keys], dtype=float)
    s = float(vals.sum())
    if s <= 0:
        return 0.0
    p = vals / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def distance_weighted_poi_score(
    node_lat: float,
    node_lon: float,
    pois: List[Dict[str, Any]],
    radius_m: int,
    category_weights: Dict[str, float],
    alpha: float = 1.0
) -> float:
    """
    距离加权 POI 强度（核心推荐指标）：
      score = sum_{poi in radius} w_cat * exp(-alpha * d / radius)
    - 同样的 poi_total，如果 POI 更靠近节点，score 会更大
    - alpha 越大越强调“近邻”
    """
    score = 0.0
    for p in pois:
        cat = p.get("category")
        if cat not in category_weights:
            continue
        d = haversine_m(node_lat, node_lon, p["lat"], p["lon"])
        if d > radius_m:
            continue
        w = float(category_weights[cat])
        score += w * float(np.exp(-alpha * d / float(radius_m)))
    return float(score)


def compute_derived_metrics(
    counts: Dict[str, int],
    dist_hosp: float,
    dist_school: float,
    dist_bus: float,
    radius_m: int
) -> Dict[str, float]:
    """
    基于 count + distance 的衍生指标（更适合你现在的现象）：
    1) density：面积归一化的“密度”（可写论文更严谨）
    2) critical_ratio：关键设施占比（组成差异）
    3) poi_entropy：功能多样性
    4) proximity_score：医院/学校可达性的综合分（越大越好）
    """
    # 1) 面积归一化密度（#/km^2）
    area_km2 = np.pi * (radius_m / 1000.0) ** 2
    poi_density_km2 = float(counts.get("poi_total", 0)) / float(area_km2)

    # 2) 关键设施占比（避免 poi_total “背景密度主导”）
    eps = 1e-6
    critical = float(counts.get("poi_hospital", 0) + counts.get("poi_school", 0))
    poi_total = float(counts.get("poi_total", 0))
    critical_ratio = float(critical / (poi_total + eps))

    # 3) 多样性（只在你已有的 6 类上算）
    entropy_keys = [
        "poi_hospital", "poi_school", "poi_park",
        "poi_attraction", "poi_bus_stop", "poi_traffic_signals"
    ]
    poi_entropy = shannon_entropy_from_counts(counts, entropy_keys)

    # 4) 可达性综合分：把距离转成“分数”（越近越高）
    #    用 (1 - d/r) 并对 NaN 做截断，避免缺失导致无法比较
    dh = clip_dist(dist_hosp, radius_m)
    ds = clip_dist(dist_school, radius_m)
    db = clip_dist(dist_bus, radius_m)

    hosp_prox = 1.0 - dh / float(radius_m)
    school_prox = 1.0 - ds / float(radius_m)
    bus_prox = 1.0 - db / float(radius_m)

    # 医院/学校更关键，公交权重小
    proximity_score = float(0.45 * hosp_prox + 0.45 * school_prox + 0.10 * bus_prox)

    return {
        "poi_density_km2": poi_density_km2,
        "critical_ratio": critical_ratio,
        "poi_entropy": poi_entropy,
        "proximity_score": proximity_score,
        "dist_hosp_clip_m": dh,
        "dist_school_clip_m": ds,
        "dist_bus_clip_m": db,
    }



def evaluate_nodes(
    node_ids: List[int],
    positions: Dict[int, Tuple[float, float]],
    radius_m: int,
    cache_path: str
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = json.load(open(cache_path, "r", encoding="utf-8")) if os.path.exists(cache_path) else {}

    # 距离加权 POI：你可以按论文叙事调整权重（医院/学校更重要）
    CAT_W = {
        "amenity:hospital": 3.0,
        "amenity:clinic": 2.5,
        "amenity:school": 2.0,
        "amenity:university": 2.0,
        "highway:bus_stop": 1.2,
        "highway:traffic_signals": 0.6,
        "leisure:park": 0.8,
        "tourism:attraction": 0.5,
        "tourism:museum": 0.5,
        "tourism:viewpoint": 0.5,
    }
    ALPHA = 1.2  # 越大越强调近邻（建议 0.8~1.5 之间）

    rows = []
    for nid in node_ids:
        lon, lat = positions[nid]
        key = f"{nid}_r{radius_m}"

        if key not in cache:
            try:
                data = overpass_query(lat, lon, radius_m)
                cache[key] = parse_pois(data)
                time.sleep(OVERPASS_SLEEP_S)
            except Exception as e:
                # 如果你现在已经跑通并且不常失败，这里可以保留；
                # 若担心“失败缓存为0”影响未来重试，可改成不写 cache
                cache[key] = []
                print(f"[WARN] Overpass failed for node {nid} (r={radius_m}): {repr(e)}")

        pois = cache[key]
        counts = count_categories(pois)

        # nearest distances (meters)
        dist_hosp = min_dist_to_category(lat, lon, pois, {"amenity:hospital", "amenity:clinic"})
        dist_school = min_dist_to_category(lat, lon, pois, {"amenity:school", "amenity:university"})
        dist_bus = min_dist_to_category(lat, lon, pois, {"highway:bus_stop"})

        # ✅ 新增：衍生指标（密度/占比/熵/可达性综合分）
        derived = compute_derived_metrics(counts, dist_hosp, dist_school, dist_bus, radius_m)

        # ✅ 新增：距离加权 POI 强度（比 poi_total 更敏感）
        dw_poi_score = distance_weighted_poi_score(
            node_lat=lat,
            node_lon=lon,
            pois=pois,
            radius_m=radius_m,
            category_weights=CAT_W,
            alpha=ALPHA
        )

        rows.append({
            "node_id": nid,
            "lon": lon,
            "lat": lat,
            "radius_m": radius_m,
            **counts,

            "dist_nearest_hospital_m": dist_hosp,
            "dist_nearest_school_m": dist_school,
            "dist_nearest_bus_stop_m": dist_bus,

            # 新指标
            **derived,
            "dw_poi_score": dw_poi_score,
        })

    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return pd.DataFrame(rows)


# =========================
# Plotting: boxplots
# =========================
def _filter_outliers_iqr(vals: np.ndarray, k: float = 3.0, mode: str = "drop") -> np.ndarray:
    """
    IQR outlier handling.
    - k=1.5: 标准箱线图离群规则（更激进）
    - k=3.0: 更宽松（推荐：只去掉“特别离谱”的点）
    mode:
      - "drop": 直接剔除离群点
      - "clip": 把离群点裁剪到阈值内（winsorize，保留样本量）
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 4:
        return vals

    q1 = np.quantile(vals, 0.25)
    q3 = np.quantile(vals, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return vals

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    if mode == "clip":
        return np.clip(vals, lo, hi)
    # default: drop
    return vals[(vals >= lo) & (vals <= hi)]

def make_boxplot(
    df: pd.DataFrame,
    metric: str,
    out_png: str,
    title: str,
    ylabel: str,
    outlier_k: float = 3.0,          # 推荐 3.0：只去掉极端离谱点
    outlier_mode: str = "drop",      # "drop" 或 "clip"
    show_fliers: bool = False,       # 图上是否显示离群点（建议 False）
):
    order = [
        f"{METHOD_CSE}-selected",
        f"{METHOD_CSE}-nonselected",
        f"{METHOD_CORR}-selected",
        f"{METHOD_CORR}-nonselected",
    ]

    plt.figure(figsize=(5, 5), dpi=220)

    data = []
    dropped_info = {}

    for g in order:
        raw = df.loc[df["group"] == g, metric].values
        raw = raw[~pd.isna(raw)]
        cleaned = _filter_outliers_iqr(raw, k=outlier_k, mode=outlier_mode)

        dropped_info[g] = int(len(raw) - len(cleaned)) if outlier_mode == "drop" else 0
        data.append(cleaned)

    if all(len(d) == 0 for d in data):
        plt.text(0.5, 0.5, f"No valid data for {metric}\n(Overpass blocked or all queries failed)",
                 ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        return

    # 箱线图（不显示离群点，让图更干净）
    plt.boxplot(data, tick_labels=order, showfliers=show_fliers)

    # 叠加每个样本点（jitter）
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        x = np.random.normal(loc=i, scale=0.06, size=len(vals))
        plt.scatter(x, vals, s=8, alpha=0.6)

    # 可选：在标题里标记剔除了多少点（避免审稿人说你“动了数据”）
    if outlier_mode == "drop":
        dropped_total = sum(dropped_info.values())
        if dropped_total > 0:
            plt.title(f"{title}\n(IQR filter k={outlier_k}: dropped {dropped_total} pts)")
        else:
            plt.title(title)
    else:
        plt.title(f"{title}\n(IQR clip k={outlier_k})")

    plt.ylabel(ylabel)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# def make_boxplot(df: pd.DataFrame, metric: str, out_png: str, title: str, ylabel: str):
#     order = [
#         f"{METHOD_CSE}-selected",
#         f"{METHOD_CSE}-nonselected",
#         f"{METHOD_CORR}-selected",
#         f"{METHOD_CORR}-nonselected",
#     ]
#
#     plt.figure(figsize=(5, 5), dpi=220)
#
#     # 准备数据
#     data = []
#     for g in order:
#         vals = df.loc[df["group"] == g, metric].values
#         vals = vals[~pd.isna(vals)]  # 去 NaN
#         data.append(vals)
#
#     # 如果全空，直接画一张提示图（避免“空白图”误解）
#     if all(len(d) == 0 for d in data):
#         plt.text(0.5, 0.5, f"No valid data for {metric}\n(Overpass blocked or all queries failed)",
#                  ha="center", va="center")
#         plt.axis("off")
#         plt.title(title)
#         plt.savefig(out_png, bbox_inches="tight")
#         plt.close()
#         return
#
#     # 箱线图
#     plt.boxplot(data, tick_labels=order, showfliers=True)
#
#     # 叠加每个样本点（jitter）
#     for i, vals in enumerate(data, start=1):
#         if len(vals) == 0:
#             continue
#         x = np.random.normal(loc=i, scale=0.06, size=len(vals))
#         plt.scatter(x, vals, s=8, alpha=0.6)
#
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.xticks(rotation=12, ha="right")
#     plt.tight_layout()
#     plt.savefig(out_png, bbox_inches="tight")
#     plt.close()


def save_combined_plots(df: pd.DataFrame, radius_m: int, output_dir: str):
    """
    生成并保存一个包含三个子图的合成图：
    1. 距离最近的公交站
    2. 距离最近的学校
    3. 500米范围内的POI数量
    """
    os.makedirs(output_dir, exist_ok=True)

    # 设置字体为 Times New Roman
    rcParams['font.family'] = 'Times New Roman'

    # 创建包含3个子图的图形和坐标轴（横向排布）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)

    # 设置子图标题
    axes[0].set_title(f"Distance to Nearest Bus Stop within {radius_m}m")
    axes[1].set_title(f"Distance to Nearest School within {radius_m}m")
    axes[2].set_title(f"POI Count within {radius_m}m")

    # 绘制距离最近的公交站
    axes[0].boxplot(df['dist_nearest_bus_stop_m'], vert=False)
    axes[0].set_xlabel('Distance (m)')

    # 绘制距离最近的学校
    axes[1].boxplot(df['dist_nearest_school_m'], vert=False)
    axes[1].set_xlabel('Distance (m)')

    # 绘制500米范围内的POI数量
    axes[2].boxplot(df['poi_total'], vert=False)
    axes[2].set_xlabel('# POIs')

    # 添加统一的图例
    axes[0].legend(['CISE-LRTR selected', 'CISE-LRTR nonselected', 'MGC-RM selected', 'MGC-RM nonselected'],
                   loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, fancybox=True, ncol=4)

    # 调整布局，避免重叠，并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_plots_r{radius_m}.png"), dpi=600)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    positions = load_positions_csv(POS_CSV)
    all_nodes = sorted(list(positions.keys()))

    data_cse = load_pkl_dict(PKL_CSE)
    data_corr = load_pkl_dict(PKL_CORR)

    sel_cse = extract_selected_nodes_cse(data_cse)
    sel_corr = extract_selected_nodes_corrse(data_corr)

    non_cse = set(all_nodes) - sel_cse
    non_corr = set(all_nodes) - sel_corr

    print(f"{METHOD_CSE} selected nodes: {len(sel_cse)}")
    print(f"{METHOD_CORR} selected nodes: {len(sel_corr)}")

    for radius_m in RADII_METERS:
        run_dir = os.path.join(OUT_DIR, f"r{radius_m}")
        os.makedirs(run_dir, exist_ok=True)

        cache_path = os.path.join(run_dir, f"poi_cache_r{radius_m}.json")

        df_cse_sel = evaluate_nodes(sorted(sel_cse), positions, radius_m, cache_path)
        df_cse_sel["group"] = f"{METHOD_CSE}-selected"

        df_cse_non = evaluate_nodes(sorted(non_cse), positions, radius_m, cache_path)
        df_cse_non["group"] = f"{METHOD_CSE}-nonselected"

        df_corr_sel = evaluate_nodes(sorted(sel_corr), positions, radius_m, cache_path)
        df_corr_sel["group"] = f"{METHOD_CORR}-selected"

        df_corr_non = evaluate_nodes(sorted(non_corr), positions, radius_m, cache_path)
        df_corr_non["group"] = f"{METHOD_CORR}-nonselected"

        df = pd.concat([df_cse_sel, df_cse_non, df_corr_sel, df_corr_non], ignore_index=True)

        # Save node-level metrics
        node_csv = os.path.join(run_dir, f"poi_node_level_r{radius_m}.csv")
        df.to_csv(node_csv, index=False, encoding="utf-8-sig")

        # Generate and save the combined plot for the three metrics
        save_combined_plots(df, radius_m, run_dir)

        print(f"\n[Done radius={radius_m}m] saved in: {run_dir}")
        print(" -", node_csv)

    print("\nAll done. Check:", OUT_DIR)


if __name__ == "__main__":
    main()
